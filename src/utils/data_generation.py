from typing import Callable
from loguru import logger
import asyncio

from langchain.prompts import StringPromptTemplate


def split_dataset(dataset: list[dict], num_chunks: int) -> list[list[dict]]:
    """Splits a list of dictionaries into a specified number of chunks."""
    if num_chunks <= 0:
        raise ValueError("Number of chunks must be a positive integer.")
    samples_per_chunk = len(dataset) // num_chunks
    if samples_per_chunk == 0:
        logger.warning(f"Dataset size ({len(dataset)}) is smaller than number of chunks ({num_chunks}). Returning fewer chunks.")
        samples_per_chunk = 1
        
    return [dataset[i:i+samples_per_chunk] for i in range(0, len(dataset), samples_per_chunk)]


async def _generate_data_for_chunk(
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


async def concurrent_data_generation(
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
        _generate_data_for_chunk(
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

async def concurrent_data_postprocessing(
    agent,
    data_chunks: list[list[dict]],
    extract_args: Callable,
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
        _process_data_chunk(agent, data_chunk, extract_args)
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