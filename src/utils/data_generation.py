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


async def generate_concurrently(
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