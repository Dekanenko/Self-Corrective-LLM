from src.models import Error
import re


def ensure_space_after_del_tokens(text: str) -> str:
    """
    Ensures that a space is present after any <DEL_*> token.
    For example, "Hello<DEL_W>world" becomes "Hello<DEL_W> world".
    """
    # Add space after if missing
    text = re.sub(r'(<DEL_[A-Z]>)(?!\s)', r'\1 ', text)

    return text


def apply_del_tokens(text: str) -> str:
    text = ensure_space_after_del_tokens(text)
    text = text.replace('\n\n', '\n').replace('\n', ' \n ')
    words = text.split(' ')
    # A small, maintainable set of common titles that end with a period.
    known_titles = {"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Inc."}

    i = 0
    while i < len(words):
        if "<DEL_A>" in words[i]:
            words = words[i+1:]
            # Since we've reset the list, we should reset the index
            i = 0 
            continue
        elif "<DEL_S>" in words[i]:
            # Find the start of the sentence to delete
            start_of_sentence = -1
            for j in range(i - 1, -1, -1):
                word = words[j]
                
                # A sentence definitely ends with '?' or '!' or a newline
                if word.endswith(('?', '!')) or word == '\n':
                    start_of_sentence = j + 1
                    break

                # If it ends with '.', we must check if it's an abbreviation.
                if word.endswith('.'):
                    # Heuristic 1: Check against a list of known titles.
                    if word in known_titles:
                        continue # It's a title, not the end of a sentence.

                    # Heuristic 2: Check for internal periods. This handles "U.S." and "i.e.".
                    # If the part of the word before the final period still contains a period,
                    # it's part of an acronym.
                    if '.' in word[:-1]:
                        continue # It's an acronym, not the end of a sentence.
                    
                    # If neither heuristic matched, we assume it's the end of the sentence.
                    start_of_sentence = j + 1
                    break
            
            if start_of_sentence == -1: # The sentence starts at the beginning
                start_of_sentence = 0

            # Remove the token and the sentence
            del words[start_of_sentence : i+1]

            # Adjust index to re-check from the point of deletion
            i = start_of_sentence
            continue
        elif "<DEL_W>" in words[i]:
            # Simple case: DEL_W is attached to the word to be replaced
            if "<DEL_W>" in words[i] and words[i] != "<DEL_W>":
                del words[i]
                i -= 1
            elif i > 0 and words[i] == "<DEL_W>":
                del words[i-1:i+1] # delete the word before and the token
                i -=1 # adjust index
                continue

        i += 1
    
    if not words:
        return ""

    # Post-processing step to remove adjacent duplicate words (case-insensitive)
    cleaned_words = [words[0]]
    for i in range(1, len(words)):
        # Compare current word with the last added word, case-insensitively
        if words[i].lower() != cleaned_words[-1].lower():
            cleaned_words.append(words[i])

    # Re-join the words, converting newline tokens back to actual newlines.
    final_text = ' '.join(cleaned_words)
    return final_text.replace(' \n ', '\n')


def format_errors(errors: list[dict]) -> str:
    errors_str = ""
    for i, error in enumerate(errors):
        errors_str += f"Error {i+1}:\n"
        errors_str += f"Error Description: {error['description']}\n"
        errors_str += f"Error Location: {error['location']}\n"
        errors_str += f"How to fix the error: {error['correction']}\n\n"

    return errors_str