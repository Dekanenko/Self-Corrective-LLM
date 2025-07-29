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
    text = text.replace('\n', '. ')
    words = text.split()
    i = 0
    # A small, maintainable set of common titles that end with a period.
    known_titles = {"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Inc."}

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
                
                # A sentence definitely ends with '?' or '!'
                if word.endswith(('?', '!')):
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

    return ' '.join(cleaned_words)


def format_errors(errors: list[Error]) -> str:
    return "\n\n".join([f"Error: {error.error_description}\nError Location: {error.error_location}\nHow to fix the error: {error.error_correction}\n" for error in errors])