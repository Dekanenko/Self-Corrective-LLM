from typing import Callable, Sequence
from loguru import logger
import asyncio

from langchain.prompts import StringPromptTemplate
from langchain_core.output_parsers import StrOutputParser


async def _generate_responses_for_chunk_deployed(
    model,
    prompt_class: StringPromptTemplate,
    dataset: list[dict], # This represents a single chunk of data
    response_dict_format: dict,
    tokenizer: object | None = None,
    data_processing_function: Callable | None = None,
    prompt_repetitions: int = 1,
    max_length: int = 512,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> list[dict]:
    """
    Generates responses for a single chunk of data, handling both local and remote models.
    """
    responses = []
    
    if not data_processing_function:
        raise ValueError("'data_processing_function' must be provided.")
        
    model_input, additional_info = data_processing_function(dataset)

    for i in range(len(model_input)):
        prompt = prompt_class(input_variables=list(model_input[i].keys()))
        prompt_string = prompt.format(**model_input[i])
        prompts = [prompt_string] * prompt_repetitions

        try:
            if tokenizer:
                # --- Local LLM Generation Path ---
                inputs = tokenizer(
                    text=prompts, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                outputs = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                sample_responses = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            else:
                # --- Remote LLM (SageMaker) Generation Path ---
                parameters = {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "max_tokenization_length": max_length,
                }
                prediction_output = model.predict({
                    "inputs": prompts,
                    "parameters": parameters,
                })
                sample_responses = prediction_output["responses"]

        except Exception as e:
            logger.error(f"Error generating response for prompt '{prompt_string[:100]}...': {e}")
            continue

        response_dict = {
            **response_dict_format,
            "input": prompt_string,
            "response": [response.strip() for response in sample_responses],
            "additional_info": response_dict_format.get("additional_info", {}).copy(),
        }

        if additional_info and i < len(additional_info) and additional_info[i]:
            response_dict["additional_info"].update(additional_info[i])

        responses.append(response_dict)
    
    return responses


async def _generate_responses_for_chunk_api(
    llm,
    prompt_class: StringPromptTemplate,
    dataset: list[dict], # This represents a single chunk of data
    response_dict_format: dict,
    data_processing_function: Callable | None = None,
    prompt_repetitions: int = 1,
    semaphore: asyncio.Semaphore | None = None,
) -> list[dict]:
    """
    Generates responses for a single chunk of data, handling both local and remote models.
    """
    if not data_processing_function:
        raise ValueError("'data_processing_function' must be provided.")
    
    if not semaphore:
        raise ValueError("'semaphore' must be provided.")
        
    model_input, additional_info = data_processing_function(dataset)
    chain = llm | StrOutputParser()

    async def process_item(i: int, item_input: dict) -> dict | None:
        """Safely process one item by making concurrent requests for each repetition."""
        prompt = prompt_class(input_variables=list(item_input.keys()))
        prompt_string = prompt.format(**item_input)

        async def invoke_chain(p_string: str):
            async with semaphore:
                return await chain.ainvoke(p_string, config={"return_exceptions": True})

        tasks = [invoke_chain(prompt_string) for _ in range(prompt_repetitions)]
        
        try:
            results = await asyncio.gather(*tasks)
            
            successful_responses = []
            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"Error in one of the repetitions for item {i}: {res}")
                else:
                    successful_responses.append(res.strip())

            if not successful_responses:
                return None

            response_dict = {
                **response_dict_format,
                "input": prompt_string,
                "response": successful_responses,
                "additional_info": response_dict_format.get("additional_info", {}).copy(),
            }

            if additional_info and i < len(additional_info) and additional_info[i]:
                response_dict["additional_info"].update(additional_info[i])

            return response_dict
        except Exception as e:
            logger.error(f"Error processing item {i} in chunk: {e}")
            return None

    responses = []
    for i in range(len(model_input)):
        # Process one item at a time, sequentially.
        # The concurrency for prompt_repetitions is handled inside process_item.
        result = await process_item(i, model_input[i])
        if result:
            responses.append(result)

    return responses


async def _process_data_chunk(
    agent,
    data_chunk: list[dict],
    extract_args: Callable,
    max_concurrency: int = 10,
) -> list[dict]:
    """
    Processes a single chunk of data items with controlled concurrency using a semaphore.

    This function is designed to run within a single process. It takes a data
    chunk and processes its items by making concurrent API calls, but limits the
    number of parallel calls to `max_concurrency` to avoid overwhelming the API
    and to manage local resources.

    Args:
        agent: The agent instance with a `.model.ainvoke` method.
        data_chunk: A list of dictionaries, where each represents an item to process.
        agent_args: A list of keys to extract from each item for the model's input.
        max_concurrency: The maximum number of API calls to have in-flight at any time.

    Returns:
        A list of results from the model. Failed calls are logged and excluded.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_item(item):
        """Safely process one item using the semaphore."""
        async with semaphore:
            return await agent.model.ainvoke(extract_args(item))

    tasks = [process_item(item) for item in data_chunk]
    
    # Gather results, allowing individual tasks to fail without stopping others.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and log them, keeping only successful results.
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error processing item {i} in chunk: {result}")
        else:
            successful_results.append(result)

    return successful_results


async def generate_responses_concurrently_deployed(
    model,
    prompt_class: StringPromptTemplate,
    data_chunks: list[list[dict]],
    response_dict_format: dict,
    tokenizer: object | None = None,
    data_processing_function: Callable | None = None,
    prompt_repetitions: int = 1,
    max_length: int = 512,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> list[dict]:
    """
    A unified function to generate data concurrently for both local and remote models.
    It determines the model type based on the presence of a tokenizer.
    """
    all_results = []
    
    tasks = [
        _generate_responses_for_chunk_deployed(
            model=model,
            prompt_class=prompt_class,
            dataset=data_chunk,
            response_dict_format=response_dict_format,
            tokenizer=tokenizer,
            data_processing_function=data_processing_function,
            prompt_repetitions=prompt_repetitions,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ) for data_chunk in data_chunks
    ]
    
    result_chunks = await asyncio.gather(*tasks, return_exceptions=False)
    for chunk in result_chunks:
        all_results.extend(chunk)

    return all_results

async def generate_responses_concurrently_api(
    llm,
    prompt_class: StringPromptTemplate,
    data_chunks: list[list[dict]],
    response_dict_format: dict,
    data_processing_function: Callable | None = None,
    prompt_repetitions: int = 1,
) -> list[dict]:
    """
    A unified function to generate data concurrently for both local and remote models.
    It determines the model type based on the presence of a tokenizer.
    """
    all_results = []
    semaphore = asyncio.Semaphore(prompt_repetitions)
    
    tasks = [
        _generate_responses_for_chunk_api(
            llm=llm,
            prompt_class=prompt_class,
            dataset=data_chunk,
            response_dict_format=response_dict_format,
            data_processing_function=data_processing_function,
            prompt_repetitions=prompt_repetitions,
            semaphore=semaphore,
        ) for data_chunk in data_chunks
    ]
    
    result_chunks = await asyncio.gather(*tasks, return_exceptions=False)
    for chunk in result_chunks:
        all_results.extend(chunk)

    return all_results


async def concurrent_data_postprocessing(
    agent,
    data_chunks: list[list[dict]],
    extract_args: Callable,
    max_concurrency: int = 10,
) -> list[dict]:
    """
    Processes multiple data chunks using a multi-level concurrency model.

    This function acts as a high-level orchestrator, assuming that you have already
    split a very large dataset into manageable `data_chunks`. It processes these
    chunks concurrently. Each chunk is handled by `_process_data_chunk`, which
    in turn processes items within that chunk using controlled concurrency.

    Args:
        agent: The agent instance to use for processing.
        data_chunks: A list of data chunks (a list of lists of dictionaries).
        agent_args: A list of keys to pass to the agent's model from each data item.

    Returns:
        A flattened list containing all successful results from all chunks.
    """
    all_results = []
    
    tasks = [
        _process_data_chunk(agent, data_chunk, extract_args, max_concurrency)
        for data_chunk in data_chunks
    ]
    
    # Gather the results from all chunk-processing tasks.
    result_chunks = await asyncio.gather(*tasks, return_exceptions=True)
    
    for chunk_result in result_chunks:
        if isinstance(chunk_result, Exception):
            logger.error(f"A data chunk failed to process entirely: {chunk_result}")
        else:
            all_results.extend(chunk_result)

    return all_results


def split_dataset(dataset: Sequence[dict], num_chunks: int) -> list[Sequence[dict]]:
    """Splits a list of dictionaries into a specified number of chunks."""
    if num_chunks <= 0:
        raise ValueError("Number of chunks must be a positive integer.")
    samples_per_chunk = len(dataset) // num_chunks
    if samples_per_chunk == 0:
        logger.warning(f"Dataset size ({len(dataset)}) is smaller than number of chunks ({num_chunks}). Returning fewer chunks.")
        samples_per_chunk = 1
        num_chunks = len(dataset)

    dataset_chunks = []
    for i in range(num_chunks - 1):
        dataset_chunks.append(dataset[i*samples_per_chunk:(i+1)*samples_per_chunk])

    dataset_chunks.append(dataset[(num_chunks - 1)*samples_per_chunk:])
    return dataset_chunks


def split_columnar_dataset(dataset: dict[str, list], num_chunks: int) -> list[dict[str, list]]:
    """
    Splits a column-oriented dataset (a dictionary of lists) into a specified
    number of chunks. Each chunk is also a column-oriented dictionary.
    """
    if not dataset or not isinstance(dataset, dict):
        raise TypeError("Input must be a non-empty dictionary of lists.")

    first_key = next(iter(dataset))
    num_samples = len(dataset[first_key])

    if num_chunks <= 0:
        raise ValueError("Number of chunks must be a positive integer.")

    chunk_size = num_samples // num_chunks
    if chunk_size == 0:
        logger.warning(f"Dataset size ({num_samples}) is smaller than number of chunks ({num_chunks}). Returning {num_samples} chunks.")
        num_chunks = num_samples
        chunk_size = 1

    chunks = []
    for i in range(num_chunks):
        start_index = i * chunk_size
        # For the last chunk, take all remaining samples
        end_index = (i + 1) * chunk_size if i < num_chunks - 1 else num_samples
        
        chunk_dict = {key: value[start_index:end_index] for key, value in dataset.items()}
        chunks.append(chunk_dict)
        
    return chunks


def nested_split_dataset(dataset: Sequence, num_major_chunks: int, num_minor_chunks: int) -> list[list[dict[str, list]]]:
    """
    Performs a two-level split on a Hugging Face Dataset. It first splits the
    dataset into major chunks, then subdivides those into minor chunks.
    The final output chunks remain in a column-oriented format (dictionary of lists).

    Args:
        dataset: The Hugging Face Dataset to split.
        num_major_chunks: The number of larger, top-level chunks.
        num_minor_chunks: The number of smaller, process-level sub-chunks.

    Returns:
        A nested list of dataset chunks, where the innermost element is a
        column-oriented dictionary of lists.
        Format: list[major_chunk[minor_chunk_dict_of_lists]]
        Example: nested_list[0][0] -> {'source': ['SVAMP', 'MAT'], 'answer': [..., ...]}
    """
    major_chunks_columnar = split_dataset(dataset, num_major_chunks)
    
    nested_final_chunks = []
    for columnar_major_chunk in major_chunks_columnar:
        if columnar_major_chunk and len(next(iter(columnar_major_chunk.values()), [])) > 0:
            minor_chunks_columnar = split_columnar_dataset(columnar_major_chunk, num_minor_chunks)
            nested_final_chunks.append(minor_chunks_columnar)
            
    return nested_final_chunks