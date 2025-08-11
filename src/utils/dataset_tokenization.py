
from transformers import AutoTokenizer

def add_deletion_instruction(input_text: str, special_instruction: str, insertion_marker: str) -> str:
    try:
        insertion_index = input_text.find(insertion_marker)
        if insertion_index != -1:
            modified_prompt = (
                input_text[:insertion_index] +
                f"{special_instruction}" +
                input_text[insertion_index:]
            )
            return modified_prompt
        else:
            print(f"Error: The marker '{insertion_marker}' was not found in the text.")

    except Exception as e:
        print(f"An error occurred: {e}")
    return input_text


def process_data(
    item: dict, 
    tokenizer: AutoTokenizer, 
    special_instruction: str, 
    insertion_marker: str, 
    deletion_token_ids: set[int]
) -> dict:
    """
    Process a single data item.
    - Tokenizes the full text at once.
    - Uses a pre-computed set for fast deletion token ID lookups.
    - Creates labels for the completion part of the text.
    - Creates labels for hallucination detection.
    """
    # Prepare prompt and completion strings
    prompt = add_deletion_instruction(item["input"], special_instruction, insertion_marker)
    completion = f'{item["correct_response"]}<|eot_id|>'
    
    # Tokenize the full text in a single, efficient call
    # We are building the full input string first and then tokenizing.
    full_text = prompt + completion
    model_inputs = tokenizer(full_text, add_special_tokens=False)

    # To create the labels, we need to know where the prompt ends and the completion begins.
    # We tokenize the prompt separately *only* to get its length.
    prompt_token_len = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])

    # 1. Create standard Causal LM labels
    # We mask the prompt tokens by setting them to -100, so loss is only calculated on the completion.
    labels = list(model_inputs['input_ids']) # Make a mutable copy
    labels[:prompt_token_len] = [-100] * prompt_token_len
    model_inputs["labels"] = labels

    # 2. Create custom labels for hallucination detection
    # Get only the token IDs corresponding to the completion part of the text.
    completion_tokens = model_inputs['input_ids'][prompt_token_len:]
    
    # Create binary labels (1 for a deletion token, 0 otherwise) for the completion part.
    completion_hallucination_labels = [1 if token_id in deletion_token_ids else 0 for token_id in completion_tokens]
    
    # Prepend mask for the prompt part, so hallucination loss is also only on the completion.
    hallucination_labels = [-100] * prompt_token_len + completion_hallucination_labels
    model_inputs["hallucination_labels"] = hallucination_labels
    
    return model_inputs