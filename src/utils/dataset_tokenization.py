import re
import codecs
from transformers import AutoTokenizer

def extract_deleted_text(correction_string: str) -> str | None:
    """
    Robustly extracts the text to be deleted from a correction string.

    This version handles:
    - Case-insensitive "DELETE" marker.
    - An optional colon ":" after "DELETE".
    - Optional leading/trailing whitespace around the core text.
    - Optional single (') or double (") quotes enclosing the text, which
      are removed from the final output.

    Args:
        correction_string: The string containing correction instructions.

    Returns:
        The cleaned text to be deleted. Returns None if no "DELETE"
        marker is found.
    """
    pattern = r"DELETE:?\s*(?:['\"](.*?)['\"]|(.*?))(?=\s*\bADD:|$)"
    match = re.search(pattern, correction_string, re.IGNORECASE | re.DOTALL)
    
    if match:
        deleted_text = match.group(1) if match.group(1) is not None else match.group(2)
        return deleted_text.strip()
    
    return None


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


# def process_data(
#     item: dict, 
#     tokenizer: AutoTokenizer, 
#     special_instruction: str, 
#     insertion_marker: str, 
#     deletion_token_ids: set[int]
# ) -> dict:
#     """
#     Process a single data item.
#     - Tokenizes the full text at once.
#     - Uses a pre-computed set for fast deletion token ID lookups.
#     - Creates labels for the completion part of the text.
#     - Creates labels for hallucination detection.
#     """
#     # Prepare prompt and completion strings
#     prompt = add_deletion_instruction(item["input"], special_instruction, insertion_marker)
#     completion = f'{item["correct_response"]}<|eot_id|>'
    
#     # Tokenize the full text in a single, efficient call
#     # We are building the full input string first and then tokenizing.
#     full_text = prompt + completion
#     model_inputs = tokenizer(full_text, add_special_tokens=False)

#     # To create the labels, we need to know where the prompt ends and the completion begins.
#     # We tokenize the prompt separately *only* to get its length.
#     prompt_token_len = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])

#     # 1. Create standard Causal LM labels
#     # We mask the prompt tokens by setting them to -100, so loss is only calculated on the completion.
#     labels = list(model_inputs['input_ids']) # Make a mutable copy
#     labels[:prompt_token_len] = [-100] * prompt_token_len
#     model_inputs["labels"] = labels

#     # 2. Create custom labels for hallucination detection
#     # Get only the token IDs corresponding to the completion part of the text.
#     completion_tokens = model_inputs['input_ids'][prompt_token_len:]
    
#     # Create binary labels (1 for a deletion token, 0 otherwise) for the completion part.
#     completion_hallucination_labels = [1 if token_id in deletion_token_ids else 0 for token_id in completion_tokens]
    
#     # Prepend mask for the prompt part, so hallucination loss is also only on the completion.
#     hallucination_labels = [-100] * prompt_token_len + completion_hallucination_labels
#     model_inputs["hallucination_labels"] = hallucination_labels
    
#     return model_inputs


def assign_hallucination_labels(
    item: dict, 
    prompt_token_len: int,
    del_w_token_id: int,
    del_s_token_id: int,
    del_a_token_id: int,
):
    hallucination_labels = item["hallucination_labels"]
    i = prompt_token_len
    while i < len(hallucination_labels):
        if hallucination_labels[i] == 1:
            j = i
            while j < len(hallucination_labels) and hallucination_labels[j] == 1:
                j += 1
            
            if j-1 != i:
                if item["input_ids"][j-1] == del_s_token_id:
                    hallucination_labels[i:j] = [2] * (j-i)
                elif item["input_ids"][j-1] == del_a_token_id:
                    hallucination_labels[i:j] = [3] * (j-i)
                elif j-i > 2 and item["input_ids"][j-1] != del_w_token_id:
                    hallucination_labels[i:j] = [2] * (j-i)
            i = j
        else:
            i += 1
        
    for i in range(prompt_token_len, len(hallucination_labels)):
        if hallucination_labels[i] != 0:
            if item["input_ids"][i] == del_w_token_id:
                hallucination_labels[i] = 1
            elif item["input_ids"][i] == del_s_token_id:
                hallucination_labels[i] = 2
            elif item["input_ids"][i] == del_a_token_id:
                for j in range(i, prompt_token_len-1, -1):
                    if hallucination_labels[j] != 0 and (item["input_ids"][j] != del_w_token_id or item["input_ids"][j] != del_s_token_id):
                        hallucination_labels[j] = 3
            
    item["hallucination_labels"] = hallucination_labels
    return item


def process_data(
    item: dict, 
    tokenizer: AutoTokenizer, 
    special_instruction: str, 
    insertion_marker: str,
    del_w_token_id: int,
    del_s_token_id: int,
    del_a_token_id: int,
) -> dict:
    """
    Processes a single data sample to prepare it for self-correction model training.

    This function performs several key operations:
    1.  Constructs the full model input by combining a prompt with the target
        `correct_response`.
    2.  Tokenizes the full text, retaining character-to-token offset mappings
        which are crucial for precise label generation.
    3.  Creates standard causal language model `labels`, where prompt tokens
        are masked with -100 to exclude them from the loss calculation.
    4.  Generates a second set of `hallucination_labels` using a sophisticated,
        multi-pass strategy to provide a rich training signal for the model's
        hallucination detector.

    The hallucination labeling is performed in three passes:
    - Pass 1 (Validated Substring Matching): Searches for each substring from
      `item['hallucinated_text']`. A match is only considered valid if it is
      not part of a larger word or number (e.g., finding "4" is ignored if it's
      part of "48"). This handles large, contiguous errors precisely. It also
      correctly processes escaped characters (e.g., `\"`) from the input JSON.
    
    - Pass 2 (Deletion Token Labeling): Iterates through the completion tokens.
      If a token is a deletion token (`<DEL_S>`, `<DEL_A>`), it is labeled `1`.
      If the token is `<DEL_W>`, both the token itself and the entire preceding
      word (which may span multiple tokens) are labeled `1`. This uses a robust,
      token-based backtracking to correctly identify word boundaries.

    - Pass 3 (Heuristic Correction): Performs a final backwards scan to correct
      cases where Pass 1 might have failed. If a `<DEL_S>` or `<DEL_A>` token
      is found without a preceding labeled span, this pass heuristically labels
      the appropriate range (the preceding sentence for `<DEL_S>`, or the entire
      completion for `<DEL_A>`). It includes special logic to handle trailing
      punctuation and cases where an un-labeled `<DEL_W>` is found, providing
      a robust fallback.

    Args:
        item (dict): A dictionary representing a single data sample. Must contain
                     keys like 'input', 'correct_response', and optionally
                     'hallucinated_text'.
        tokenizer (AutoTokenizer): The tokenizer instance to use.
        special_instruction (str): A special instruction string to be inserted
                                   into the prompt.
        insertion_marker (str): The marker in the prompt where the special
                                instruction should be inserted.
        del_w_token_id (int): The token ID for the <DEL_W> token.
        del_s_token_id (int): The token ID for the <DEL_S> token.
        del_a_token_id (int): The token ID for the <DEL_A> token.

    Returns:
        dict: A dictionary ready to be used by the Trainer, containing:
              - 'input_ids': The token IDs for the full text.
              - 'attention_mask': The attention mask.
              - 'labels': The causal LM labels with the prompt masked.
              - 'hallucination_labels': The binary hallucination labels.
    """
    # 1. Prepare prompt and completion strings
    prompt = add_deletion_instruction(item["input"], special_instruction, insertion_marker)
    completion = f'{item["correct_response"]}<|eot_id|>'
    full_text = prompt + completion
    
    # 2. Tokenize the full text with offset mappings
    model_inputs = tokenizer(
        full_text, 
        add_special_tokens=False, 
        return_offsets_mapping=True
    )
    input_ids = model_inputs['input_ids']
    offset_mapping = model_inputs['offset_mapping']

    # 3. Create standard Causal LM labels
    prompt_token_len = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
    labels = list(input_ids)
    labels[:prompt_token_len] = [-100] * prompt_token_len
    model_inputs["labels"] = labels

    # 4. Create precise custom labels for hallucination detection
    completion_token_len = len(input_ids) - prompt_token_len
    hallucination_labels = [-100] * prompt_token_len + [0] * completion_token_len
    prompt_char_len = len(prompt)

    # --- Pass 1: Labeling with Validated Substring Matching ---
    for text_to_find in item.get("hallucinated_text", []):
        if not text_to_find:
            continue
        
        try:
            cleaned_text = codecs.decode(text_to_find, 'unicode_escape')
        except UnicodeDecodeError as e:
            # This error occurs with a trailing backslash (common in LaTeX).
            # Fall back to using the text as-is.
            cleaned_text = text_to_find
        
        # Escape the cleaned text to safely use it in a regex pattern
        pattern = re.escape(cleaned_text)
        
        # pattern = re.escape(text_to_find)
        for match in re.finditer(pattern, full_text[prompt_char_len:]):
            start_index = match.start() + prompt_char_len
            end_index = match.end() + prompt_char_len

            # A match is valid UNLESS it's "glued" to alphanumeric characters on both sides.
            # Check char before the match
            start_char_is_alnum = start_index > 0 and full_text[start_index - 1].isalnum()
            
            # Check first char of the match itself
            text_starts_with_alnum = text_to_find[0].isalnum()

            # Check char after the match
            end_char_is_alnum = end_index < len(full_text) and full_text[end_index].isalnum()
            
            # Check last char of the match itself
            text_ends_with_alnum = text_to_find[-1].isalnum()

            # The match is invalid ONLY if it forms a longer word/number with its surroundings.
            # E.g., `pre[MATCH]post` where all are alphanumeric.
            if start_char_is_alnum and text_starts_with_alnum:
                continue # Invalid match, e.g. finding 'port' in 'airport'
            if end_char_is_alnum and text_ends_with_alnum:
                continue # Invalid match, e.g. finding 'cat' in 'catch'
            
            # If we pass the checks, it's a valid match
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if max(token_start, start_index) < min(token_end, end_index):
                    if i >= prompt_token_len:
                        hallucination_labels[i] = 1
            # Since we found the valid match, we can break from the finditer loop
            break

    # --- Pass 2: Special handling for deletion tokens ---
    for i in range(prompt_token_len, len(input_ids)):
        token_id = input_ids[i]

        if token_id == del_w_token_id:
            # Label the <DEL_W> token itself.
            hallucination_labels[i] = 1
            
            # Ensure there is a token before it to delete.
            if i > prompt_token_len:
                
                # Start backtracking from the token immediately before <DEL_W>.
                for j in range(i - 1, prompt_token_len - 1, -1):
                    # Label the current token as part of the deleted word.
                    hallucination_labels[j] = 1
                    
                    # Get the actual text of the token using its offset.
                    tok_start_char, tok_end_char = offset_mapping[j]
                    token_text = full_text[tok_start_char:tok_end_char]
                    
                    # If the token starts with a space or is a single punctuation mark
                    # that acts as a boundary, we've found the start of the word.
                    if token_text.startswith((' ', '\n', '\t')):
                        break
                
        elif token_id == del_s_token_id or token_id == del_a_token_id:
            # Label other deletion tokens (<DEL_S>, <DEL_A>) as 1.
            hallucination_labels[i] = 1
        
    # --- Pass 3: Heuristic correction for missed <DEL_S> and <DEL_A> spans ---
    sentence_boundary_chars = {'.', '!', '?', '\n'}
    # Iterate backwards through the completion tokens
    for i in range(len(input_ids) - 1, prompt_token_len, -1):
        token_id = input_ids[i]
        is_del_s_or_a = token_id == del_s_token_id or token_id == del_a_token_id
        
        # Trigger condition: A <DEL_S/A> token is labeled, but the token before it is not.
        if is_del_s_or_a and hallucination_labels[i] == 1 and hallucination_labels[i - 1] == 0:
            
            span_start_idx = prompt_token_len
            found_other_hallucination = False
            capture_whole_sentence = False
            
            # Scan backwards from the token before the deletion token.
            for j in range(i - 1, prompt_token_len - 1, -1):
                if not capture_whole_sentence and token_id == del_s_token_id and input_ids[j] == del_w_token_id:
                    capture_whole_sentence = True
            
                if not capture_whole_sentence and hallucination_labels[j] == 1:
                    found_other_hallucination = True
                    break
                
                # For <DEL_S>, stop if we find a sentence boundary.
                if capture_whole_sentence or (token_id == del_s_token_id and j < i - 1):
                    tok_start, tok_end = offset_mapping[j]
                    tok_text = full_text[tok_start:tok_end]
                    if any(char in sentence_boundary_chars for char in tok_text):
                        span_start_idx = j + 1
                        break
            
            # If we scanned the whole span without finding existing labels...
            if not found_other_hallucination:
                # ...then apply the correction.
                for k in range(span_start_idx, i):
                    hallucination_labels[k] = 1

    model_inputs["hallucination_labels"] = hallucination_labels
    del model_inputs["offset_mapping"]

    return assign_hallucination_labels(
        model_inputs, prompt_token_len, del_w_token_id, del_s_token_id, del_a_token_id
    )