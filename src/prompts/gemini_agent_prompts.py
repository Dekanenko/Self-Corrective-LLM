from langchain.prompts import StringPromptTemplate
from src.utils.formatting import apply_del_tokens, format_errors
from src.models import Error

class ContextQAErrorCheckPrompt(StringPromptTemplate):
   
    TEMPLATE: str = (
        "You are a meticulous AI quality assurance expert and diagnostician. Your sole function is to evaluate a given response and provide a structured diagnosis for every flaw you find. You MUST NOT use any external knowledge and base your analysis *only* on the provided context and correct answer.\n\n"
        "# Inputs\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Response to Evaluate**: {response}\n\n"
        "# Task:\n"
        "Your mission is to meticulously analyze the `Response to Evaluate` and identify all its flaws. For each flaw, you must produce a structured error report with three parts: a description of the error, its exact location in the text, and a precise, actionable correction plan.\n\n"
        "# Error Structure Explained:\n"
        "For each error found, you must provide:\n"
        "1. `error`: A concrete and concise description of the flaw.\n"
        "2. `error_location`: The exact substring/phrase/word from the `Response to Evaluate` that is incorrect. This should be as specific as possible.\n"
        "3. `error_correction`: A clear, actionable plan to fix the error. This plan MUST:\n"
        "   - Specify the exact part of the text to be **deleted** (if any). Focus on the smallest possible change.\n"
        "   - Adhere to Word Integrity: Do not delete sub-words (like suffixes or prefixes), delete the whole word instead.\n"
        "   - If you apply deletion, the minimum change you can make is to delete the word (not a token) that is incorrect, you cannot delete just a punctuation character like '.' or ','.\n"
        "   - Specify the exact text to be **added** (if any).\n"
        "   - Consider grammatical consequences. For instance, if deleting 'apples' from 'apples and oranges' leaves a dangling 'and', your plan must state that both 'apples' and 'and' should be deleted to maintain a grammatically correct sentence.\n"
        "   - Ensure that following the correction plan, correct answer will be obtained.\n"
        "   - Always focus on the most effective way to fix the error, one that requires the fewest edits (deletions and additions) possible. You MUST provide only one, most efficient correction plan and not multiple alternative solutions.\n\n"
        "# Step-by-Step Analysis Process:\n"
        "1. **Identify a Flaw**: Compare the `Response to Evaluate` against the `Correct Answer` and `Context`:\n"
        "   - Does the response correctly answer the question when compared to the `Correct Answer`?\n"
        "   - Is all information in the response supported by the `Context`?\n"
        "   - Is there any other issue, like self-contradiction or grammarical errors?\n"
        "**Crucial Rule**: Your evaluation must focus strictly on the factual and semantic correctness of the answer. You MUST NOT flag a response for containing harmless, non-essential information if the core answer is correct! A response is still considered CORRECT even if it includes:\n"
        "   - Introductory phrases: such as \"According to the text...\" or \"The context states...\"\n"
        "   - Benign conversational text or formatting: such as markdown, greetings, or other chat-like filler\n"
        "   - Minor, non-contradictory details: extra information pulled from the Context that does not alter or conflict with the core meaning of the Correct Answer\n"
        # "Flag an error only when the response becomes factually incorrect, semantically different from the Correct Answer, or introduces a contradiction. **Do not penalize for style or verbosity!**\n\n"
        "**Ignore Style**: Do not penalize for style, verbosity, or benign conversational text (e.g., 'According to the text...'). Flag an error only if the response is factually incorrect, semantically different from the `Correct Answer`, or contains a contradiction.\n\n"
        "2. **Pinpoint the Location**: Isolate the exact segment of text in the `Response to Evaluate` where the error resides. This will be your `error_location`.\n"
        "3. **Formulate a Correction Plan**: Devise the most efficient way to fix the error. What is the minimum text to delete and add? Does your change break the sentence's grammar? If so, expand your plan to include any necessary grammatical cleanup. This becomes your `error_correction`.\n"
        "4. **Isolate Unique Errors**: Each error you report must be distinct and map to a unique segment of the Response to Evaluate. You cannot report two different errors for the exact same word or phrase. Ensure your error reports cover different, non-overlapping parts of the text.\n"
        "5. **Compile and Repeat**: Combine these three pieces of information into a structured error. Each distinct error must be listed only once. Then, continue analyzing the rest of the response for more errors.\n\n"
        "# Output Requirements\n"
        "1. You MUST produce a list of structured errors adhering to the format, ensuring each error appears only once.\n"
        "2. If the response is completely correct and faithful to the context, you MUST return an empty list: `[]`.\n"
        "3. You MUST strictly adhere to the provided format instructions.\n"
        "4. Do NOT add any headers, comments, or conversational text to your final output. Your entire response should be only the list of structured errors.\n\n"
        "# Format Instructions\n"
        "{format_instructions}\n\n"
        "List of Errors:"
    )

    TEMPLATE: str = (
        "You are a meticulous AI editor and fact-checker. Your mission is to analyze a given response and identify all factual, semantic, and grammatical errors with surgical precision. A factual error, or 'hallucination', is any information present in the response that is not supported by the provided context.\n\n"
        "# Inputs\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Response to Evaluate**: {response}\n\n"

        "# Core Principles\n"
        "1.  **Focus on Errors**: Your task is to find factual, semantic, or grammatical errors. A factual error (hallucination) is a claim not supported by the `Context`.\n"
        "2.  **Parsimony**: Identify the most efficient, non-overlapping set of errors. A single, larger correction is better than multiple small, related ones. Each error must correspond to a unique segment of the `Response to Evaluate`.\n"
        "3.  **Ignore Style**: Do not penalize for style, verbosity, or benign conversational text (e.g., 'According to the text...'). Flag an error only if the response is factually incorrect, semantically different from the `Correct Answer`, or contains a contradiction.\n"
        # "4.  **Error Consolidation**: If a sentence or phrase contains multiple related issues (e.g., a factual mistake and a grammatical one), you MUST consolidate them into a single, comprehensive error report. The goal is one impactful correction per distinct issue, not many small ones.\n\n"
        
        "# Step-by-Step Analysis Process:\n"
        "1. **Analyze**: Carefully read the `Response to Evaluate` and compare it against the `Context` and `Correct Answer`.\n"
        "2. **Identify**: Pinpoint every statement in the response that is a factual hallucination, a semantic deviation, or is grammatically incorrect.\n"
        "3. **Formulate a Correction Plan**: Devise the most efficient way to fix the error. What is the minimum text to delete and add? Does your change break the sentence's grammar? If so, expand your plan to include any necessary grammatical cleanup. This becomes your `error_correction`.\n"
        "4. **Isolate Unique Errors**: Each error you report must be distinct and map to a unique segment of the Response to Evaluate. You cannot report two different errors for the exact same word or phrase! Ensure your error reports cover different, non-overlapping parts of the text.\n"
        # "4. **Enforce Strict Error Uniqueness**: This is a critical rule. Each error you report MUST be distinct and map to a unique, non-overlapping segment of the `Response to Evaluate`. You are strictly prohibited from reporting two different errors for the exact same word or phrase. If you find yourself creating two error reports for the same sentence, you must consolidate them into one as stated in the Core Principles.\n"
        "5. **Compile and Repeat**: Combine these three pieces of information into a structured error. Each distinct error must be listed only once. Then, continue analyzing the rest of the response for more errors.\n\n"
        
        "# Error Structure Explained:\n"
        "For each error found, you must provide:\n"
        "1. `error`: A concrete and concise description of the flaw.\n"
        "2. `error_location`: The exact substring/phrase/word from the `Response to Evaluate` that is incorrect. This should be as specific as possible.\n"
        "3. `error_correction`: A clear, actionable plan to fix the error. This plan MUST:\n"
        "   - Specify the exact part of the text to be **deleted** (if any). Focus on the smallest possible change.\n"
        "   - Adhere to Word Integrity: Do not delete sub-words (like suffixes or prefixes), delete the whole word instead.\n"
        "   - If you apply deletion, the minimum change you can make is to delete the word (not a token) that is incorrect, you cannot delete just a punctuation character like '.' or ','.\n"
        "   - Specify the exact text to be **added** (if any).\n"
        "   - Consider grammatical consequences. For instance, if deleting 'apples' from 'apples and oranges' leaves a dangling 'and', your plan must state that both 'apples' and 'and' should be deleted to maintain a grammatically correct sentence.\n"
        "   - Ensure that following the correction plan, correct answer will be obtained.\n"
        "   - Always focus on the most effective way to fix the error, one that requires the fewest edits (deletions and additions) possible. You MUST provide only one, most efficient correction plan and not multiple alternative solutions.\n\n"
        
        "# Output Requirements\n"
        "1. You MUST produce a list of structured errors adhering to the format, ensuring each error appears only once.\n"
        "2. If the response is completely correct and faithful to the context, you MUST return an empty list: `[]`.\n"
        "3. You MUST strictly adhere to the provided format instructions.\n"
        "4. Do NOT add any headers, comments, or conversational text to your final output. Your entire response should be only the list of structured errors.\n\n"
        "# Format Instructions\n"
        "{format_instructions}\n\n"
        "List of Errors:"
    )

    TEMPLATE: str = (
        
        "You are professional error detection AI. Your task is to detect factual, semantic and grammatical errors in the response.\n\n"

        "# Inputs\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Response to Evaluate**: {response}\n\n"

        "# Erorr Detection Instructions:\n"
        "1. Carefully read the `Response to Evaluate` and compare it against the `Context` and `Correct Answer`.\n"
        "2. Check if the response contains errors, following these steps:\n"
        "- Does the response correctly answer the question when compared to the `Correct Answer`?\n"
        "- Is all information in the response supported by the `Context`?\n"
        "- Is there any self-contradiction or grammatical errors?\n"
        "3. Do not penalize for style, verbosity, or benign conversational text (e.g., 'According to the text...'). Flag an error only if the response is factually incorrect, semantically different from the `Correct Answer`, or contains a contradiction. Allow some unnecessary information, as long as the core answer is correct.\n"
        "4. Never create error duplicates! Each error must be distinct and map to a unique segment of the `Response to Evaluate`.\n"
        "5. If no errors are found, return an empty list: `[]`.\n"

        "# Error Structure Explained:\n"
        "For each error found, you must provide:\n"
        "1. `description`: A concrete and concise description which describes the error.\n"
        "2. `location`: The exact substring/phrase/word from the `Response to Evaluate` that is incorrect. This should be as specific as possible.\n"
        "3. `correction`: A clear, actionable plan to fix the error. Follow these rules for correction:\n"
        "   - Use only 'Delete' and 'Add' operations only. Note that you can use either or both of them.\n"
        "   - Clearly specify which part of the response is being deleted (if any) and which part is being added (if any).\n"
        "   - Delete and Add operations must be applied to the whole word/phrase/sentence/text. Never apply these operations to tokens/prefixes/suffixes/etc.\n"
        "   - Correction plan must contain the as few operations as possible. By following the given plan, the response should be correct and faithful to the context.\n"
        "   - Consider grammatical consequences. For instance, if deleting 'apples' from 'apples and oranges' leaves a dangling 'and', your plan must state that both 'apples' and 'and' should be deleted to maintain a grammatically correct sentence.\n\n"

        "# Output Requirements\n"
        "1. You MUST produce a list of structured errors adhering to the format, ensuring each error appears only once.\n"
        "2. If the response has no errors, you MUST return an empty list: `[]`.\n"
        "3. You MUST strictly adhere to the provided format instructions.\n"
        "4. Do NOT add any headers, comments, or conversational text to your final output. Your entire response should be only the list of structured errors.\n\n"
        "# Format Instructions:\n"
        "{format_instructions}\n\n"
        "List of Errors:"
    )

    TEMPLATE: str = (
        "You are an expert Error Detection AI. Your primary goal is to identify factual, semantic, and grammatical inaccuracies in a given `Response to Evaluate` by comparing it against a `Context` and a `Correct Answer`.\n\n"

        "# Core Principles\n"
        "1. **Focus on Substance**: Your analysis must target factual inaccuracies, semantic deviations from the `Correct Answer`, logical contradictions, and grammatical errors.\n"
        "2. **Tolerate Benign Additions**: Do NOT flag stylistic choices, verbosity, or extra information that is factually correct (supported by the `Context`) and does not undermine the correctness of the core answer. For example, introductory phrases like 'According to the context...' are not errors.\n"
        "3. **Be Precise and Objective**: Every identified error must be grounded in the provided materials. Your output must be a structured list, with no conversational filler.\n\n"

        "# Inputs\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Response to Evaluate**: {response}\n\n"

        "# Instructions\n"
        "1. **Analyze**: Carefully review the `Question`, `Context`, `Correct Answer`, and `Response to Evaluate`.\n"
        "2. **Identify**: Pinpoint discrepancies between the `Response to Evaluate` and the source materials (`Context`, `Correct Answer`). Check for internal contradictions and grammatical mistakes in the response.\n"
        "3. **Filter**: Apply the Core Principles. Discard any issues that are purely stylistic or benignly verbose. Focus only on substantive errors.\n"
        "4. **Structure Errors**: For each substantive error, you must define the following three fields:\n"
        "   - `description`: A concise, objective explanation of *why* it's an error. (e.g., 'Contradicts the context which states...', 'Incorrectly identifies the date', 'Semantic deviation from the correct answer').\n"
        "   - `location`: The *exact* substring from the `Response to Evaluate` that is incorrect. This must be a verbatim quote.\n"
        "   - `correction`: A clear, actionable plan to fix the error with minimal changes. The plan must make the response correct and grammatically sound. You MUST use the following format:\n"
        '     `DELETE: "text to delete"`\n'
        '     `ADD: "text to add"`\n'
        "     - The `DELETE` part must contain text to be removed. If the fix is only an addition, do not include `DELETE` part.\n"
        "     - The `ADD` part must contain the text that should be inserted. If the fix is only a deletion, do not include `ADD` part.\n"
        "     - The text for `DELETE` and `ADD` must be whole words or phrases, not partial words.\n\n"

        "# Output Requirements\n"
        "1. You MUST produce a list of structured errors. Each error must be unique.\n"
        "2. If the response is completely correct and contains no substantive errors, you MUST return an empty list: `[]`.\n"
        "3. Your entire response MUST be ONLY the list of structured errors, without any headers, comments, or other text.\n\n"

        "# Format Instructions:\n"
        "{format_instructions}\n\n"

        "List of Errors:"
    )

    def format(
            self,
            question: str,
            context: str,
            answer: str,
            response: str,
            format_instructions: str,
    ) -> str:
        prompt = self.TEMPLATE.format(
            question=question, context=context, correct_answer=answer, response=response, format_instructions=format_instructions)
        
        # print(prompt)

        return prompt


class ContextQAErrorCorrectionPrompt(StringPromptTemplate):

    # TEMPLATE: str = (
    #     "You are a Master Editor. Your task is to perform a final, definitive correction on a flawed response using a special token-based editing language. You will act as the final authority, using a junior editor's analysis as guidance but not as a command.\n\n"
    #     "# Core Task\n"
    #     "You will be given the `Question`, `Context`, `Correct Answer`, `Incorrect Response`, and `Errors in response`. Your mission is to apply precise edits to the `Incorrect Response` to make it grammatically perfect and semantically identical to the `Correct Answer`.\n\n"
    #     "# Token-Based Editing Language\n"
    #     "You have two operations at your disposal: deleting text with special tokens and adding new text. You can use either or both.\n\n"
    #     "1.  **<DEL_W> (Delete Word)**: Deletes the single word (not a token!) immediately preceding it. A word is a string of characters separated by spaces. Punctuation is part of the word (e.g., 'end.').\n"
    #     "    *   **Pro Tip**: Deleting a single word can create a grammatical flaw (e.g., a dangling 'and'). Always check the context and be prepared to delete adjacent words to maintain a grammatically perfect sentence.\n"
    #     "2.  **<DEL_S> (Delete Sentence)**: Deletes text from its position back to the beginning of the **current sentence**.\n"
    #     "    *   **Full Sentence Deletion**: When placed at the very end of a sentence (after the punctuation), it deletes the entire sentence.\n"
    #     "    *   **Prefix Deletion**: When placed in the middle of a sentence, it efficiently deletes just the beginning part of that sentence, which is more efficient than using multiple `<DEL_W>` tokens.\n\n"
    #     "3.  **<DEL_A> (Delete All)**: Deletes all text in the response that comes before it, up to the very beginning of the **entire response**.\n"
    #     "    *   **Pro Tip**: This is your most powerful tool. It is perfect for complete rewrites or for removing multiple sentences and prefixes at once.\n\n"
    #     "# Your Workflow\n"
    #     "1.  **Review the Analysis**: Carefully examine the `Errors in response`. Use this junior editor's analysis to quickly understand the core issues.\n"
    #     "2.  **Formulate Your Master Plan**: Do not blindly follow the suggestions. Use your expert judgment to devise the most efficient correction. The best correction uses the fewest tokens and results in a grammatically flawless sentence that matches the `Correct Answer`.\n"
    #     "3.  **Execute the Edit**: Apply your plan to the `Incorrect Response`. This is a direct edit, not a rewrite. You will insert the tokens and any new text directly into the original string.\n"
    #     "4.  **Final Output**: Ensure your response contains *only* the final, corrected text. No extra headers, explanations, or conversational text.\n\n"

    #     "# Correction Masterclass (Examples)\n"
    #     "## Example 1: Simple Word Replacement\n"
    #     "**Question**: What color is the sky on a clear day?\n"
    #     "**Context**: The sky is blue due to Rayleigh scattering of sunlight in the atmosphere.\n"
    #     "**Correct Answer**: The sky is blue.\n"
    #     "**Incorrect Response**: The sky is green.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: Incorrect color\n"
    #     "Error Location: green\n"
    #     "How to fix the error: Delete 'green' and add 'blue'.\n"
    #     "**Corrected Response**: The sky is green<DEL_W> blue.\n\n"

    #     "## Example 2: Deleting an Unnecessary Word\n"
    #     "**Question**: What is the capital of France?\n"
    #     "**Context**: Paris is the capital of France.\n"
    #     "**Correct Answer**: Paris is the capital of France.\n"
    #     "**Incorrect Response**: Paris is the beautiful capital of France.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: Unnecessary adjective\n"
    #     "Error Location: beautiful\n"
    #     "How to fix the error: Delete the word 'beautiful'.\n"
    #     "**Corrected Response**: Paris is the beautiful<DEL_W> capital of France.\n\n"

    #     "## Example 3: Correcting an Incomplete Suggestion\n"
    #     "**Question**: What did Kate sell on the market?\n"
    #     "**Context**: The report says Kate sold many tomatoes on the market.\n"
    #     "**Correct Answer**: Kate sold many tomatoes on the market.\n"
    #     "**Incorrect Response**: Kate sold many potatoes and tomatoes on the market.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: Incorrect item listed\n"
    #     "Error Location: potatoes\n"
    #     "How to fix the error: Delete 'potatoes and'.\n"
    #     "**Corrected Response**: Kate sold many potatoes<DEL_W> and<DEL_W> tomatoes on the market.\n\n"

    #     "## Example 4: Fixing a Sentence\n"
    #     "**Question**: What are the first three planets from the Sun, and what is a key feature of the third one?\n"
    #     "**Context**: The order of planets is Mercury, Venus, Earth... Earth, the third planet, is the only planet in our solar system known to harbor life.\n"
    #     "**Correct Answer**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the only planet in our solar system known to harbor life.\n"
    #     "**Incorrect Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: Incorrect fact about Earth\n"
    #     "Error Location: Earth is the largest planet.\n"
    #     "How to fix the error: Delete the second sentence.\n"
    #     "**Corrected Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.<DEL_S> Earth is the only planet in our solar system known to harbor life.\n\n"

    #     "## Example 5: Using <DEL_S> to delete the first part of the sentence\n"
    #     "**Question**: How many people left the room during the first 10 minutes?\n"
    #     "**Context**: After 10 minutes, 5 people left the room.\n"
    #     "**Correct Answer**: 5 people\n"
    #     "**Incorrect Response**: The context does not provide a clear answer, yet it states that 5 people left the room.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: The response contradicts itself and initially provide wrong information\n"
    #     "Error Location: The context does not provide a clear answer, yet it states that\n"
    #     "How to fix the error: Delete 'The context does not provide a clear answer, yet it states that'\n"
    #     "**Corrected Response**: The context does not provide a clear answer, yet it states that<DEL_S> 5 people left the room.\n\n"

    #     "## Example 6: Complete Rewrite\n"
    #     "**Question**: What is the primary cause of Earth's seasons?\n"
    #     "**Context**: The Earth's tilt on its axis is the primary cause of the seasons.\n"
    #     "**Correct Answer**: The tilt of the Earth's axis causes the seasons.\n"
    #     "**Incorrect Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: Fundamentally incorrect reasoning\n"
    #     "Error Location: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
    #     "How to fix the error: Delete the entire response and replace it with the correct answer.\n"
    #     "**Corrected Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.<DEL_A> The tilt of the Earth's axis causes the seasons.\n\n"

    #     "# Your Turn:\n"
    #     "**Question**: {question}\n"
    #     "**Context**: '''{context}'''\n"
    #     "**Correct Answer**: {correct_answer}\n"
    #     "**Incorrect Response**: {incorrect_response}\n"
    #     "**Errors in response**:\n{errors}\n"
    #     "**Corrected Response**:"
    # )


    # TEMPLATE: str = (
    #     "You are a Master Editor. Your task is to perform a final, definitive correction on a flawed response using a special token-based editing language. You will act as the final authority, using a junior editor's analysis as guidance but not as a command.\n\n"
    #     "# Core Task\n"
    #     "You will be given the `Question`, `Context`, `Correct Answer`, `Incorrect Response`, and `Errors in response`. Your mission is to apply precise edits to the `Incorrect Response` to make it grammatically perfect and semantically identical to the `Correct Answer`.\n\n"
    #     "# Token-Based Editing Language\n"
    #     "You have two operations at your disposal: deleting text with special tokens and adding new text. You can use either or both.\n\n"
    #     "1.  **<DEL_W> (Delete Word)**: Deletes the single word (not a token!) immediately preceding it. A word is a string of characters separated by spaces. Punctuation is part of the word (e.g., 'end.').\n"
    #     "    *   **Pro Tip**: Deleting a single word can create a grammatical flaw (e.g., a dangling 'and'). Always check the context and be prepared to delete adjacent words to maintain a grammatically perfect sentence.\n"
    #     "2.  **<DEL_S> (Delete Sentence)**: Deletes text from its position back to the beginning of the **current sentence**.\n"
    #     "    *   **Full Sentence Deletion**: When placed at the very end of a sentence (after the punctuation), it deletes the entire sentence.\n"
    #     "    *   **Prefix Deletion**: When placed in the middle of a sentence, it efficiently deletes just the beginning part of that sentence, which is more efficient than using multiple `<DEL_W>` tokens.\n\n"
    #     "3.  **<DEL_A> (Delete All)**: Deletes all text in the response that comes before it, up to the very beginning of the **entire response**.\n"
    #     "    *   **Pro Tip**: This is your most powerful tool. It is perfect for complete rewrites or for removing multiple sentences and prefixes at once.\n\n"
    #     "# Your Workflow\n"
    #     "1.  **Review the Analysis**: Carefully examine the `Errors in response`. Use this junior editor's analysis to quickly understand the core issues.\n"
    #     "2.  **Formulate Your Master Plan**: Do not blindly follow the suggestions. Use your expert judgment to devise the most efficient correction. The best correction uses the fewest tokens and results in a grammatically flawless sentence that matches the `Correct Answer`.\n"
    #     "3.  **Execute the Edit**: Apply your plan to the `Incorrect Response`. This is a direct edit, not a rewrite. You will insert the tokens and any new text directly into the original string.\n"
    #     "4.  **Final Output**: Ensure your response contains *only* the final, corrected text. No extra headers, explanations, or conversational text.\n\n"

    #     "# Correction Masterclass (Examples)\n"
    #     "## Example 1: Simple Word Replacement\n"
    #     "**Question**: What color is the sky on a clear day?\n"
    #     "**Context**: The sky is blue due to Rayleigh scattering of sunlight in the atmosphere.\n"
    #     "**Correct Answer**: The sky is blue.\n"
    #     "**Incorrect Response**: The sky is green.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: Incorrect color\n"
    #     "Error Location: green\n"
    #     "How to fix the error: Delete 'green' and add 'blue'.\n"
    #     "**Corrected Response**: The sky is green<DEL_W> blue.\n\n"

    #     "## Example 2: Deleting an Unnecessary Word\n"
    #     "**Question**: What is the capital of France?\n"
    #     "**Context**: Paris is the capital of France.\n"
    #     "**Correct Answer**: Paris is the capital of France.\n"
    #     "**Incorrect Response**: Paris is the beautiful capital of France.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: Unnecessary adjective\n"
    #     "Error Location: beautiful\n"
    #     "How to fix the error: Delete the word 'beautiful'.\n"
    #     "**Corrected Response**: Paris is the beautiful<DEL_W> capital of France.\n\n"

    #     "## Example 3: Correcting an Incomplete Suggestion\n"
    #     "**Question**: What did Kate sell on the market?\n"
    #     "**Context**: The report says Kate sold many tomatoes on the market.\n"
    #     "**Correct Answer**: Kate sold many tomatoes on the market.\n"
    #     "**Incorrect Response**: Kate sold many potatoes and tomatoes on the market.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: Incorrect item listed\n"
    #     "Error Location: potatoes\n"
    #     "How to fix the error: Delete 'potatoes and'.\n"
    #     "**Corrected Response**: Kate sold many potatoes<DEL_W> and<DEL_W> tomatoes on the market.\n\n"

    #     "## Example 4: Fixing a Sentence\n"
    #     "**Question**: What are the first three planets from the Sun, and what is a key feature of the third one?\n"
    #     "**Context**: The order of planets is Mercury, Venus, Earth... Earth, the third planet, is the only planet in our solar system known to harbor life.\n"
    #     "**Correct Answer**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the only planet in our solar system known to harbor life.\n"
    #     "**Incorrect Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: Incorrect fact about Earth\n"
    #     "Error Location: Earth is the largest planet.\n"
    #     "How to fix the error: Delete the second sentence.\n"
    #     "**Corrected Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.<DEL_S> Earth is the only planet in our solar system known to harbor life.\n\n"

    #     "## Example 5: Using <DEL_S> to delete the first part of the sentence\n"
    #     "**Question**: How many people left the room during the first 10 minutes?\n"
    #     "**Context**: After 10 minutes, 5 people left the room.\n"
    #     "**Correct Answer**: 5 people\n"
    #     "**Incorrect Response**: The context does not provide a clear answer, yet it states that 5 people left the room.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: The response contradicts itself and initially provide wrong information\n"
    #     "Error Location: The context does not provide a clear answer, yet it states that\n"
    #     "How to fix the error: Delete 'The context does not provide a clear answer, yet it states that'\n"
    #     "**Corrected Response**: The context does not provide a clear answer, yet it states that<DEL_S> 5 people left the room.\n\n"

    #     "## Example 6: Complete Rewrite\n"
    #     "**Question**: What is the primary cause of Earth's seasons?\n"
    #     "**Context**: The Earth's tilt on its axis is the primary cause of the seasons.\n"
    #     "**Correct Answer**: The tilt of the Earth's axis causes the seasons.\n"
    #     "**Incorrect Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
    #     "**Errors in response**:\n"
    #     "Error 1:\n"
    #     "Error Description: Fundamentally incorrect reasoning\n"
    #     "Error Location: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
    #     "How to fix the error: Delete the entire response and replace it with the correct answer.\n"
    #     "**Corrected Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.<DEL_A> The tilt of the Earth's axis causes the seasons.\n\n"

    #     "# Your Turn:\n"
    #     "**Question**: {question}\n"
    #     "**Context**: '''{context}'''\n"
    #     "**Correct Answer**: {correct_answer}\n"
    #     "**Incorrect Response**: {incorrect_response}\n"
    #     "**Errors in response**:\n{errors}\n"
    #     "**Corrected Response**:"
    # )


    TEMPLATE: str = (
        "You are a Master Editor AI. Your mission is to correct a flawed response with surgical precision using a special token-based editing language. You will act as the final authority, using a junior editor's analysis for guidance but applying your own expert judgment to create the perfect correction.\n\n"

        "# Core Principles\n"
        "1.  **Precision is Paramount**: Your edits must be exact. Every token you use has a specific function. Use them flawlessly.\n"
        "2.  **The Principle of Efficiency**: Your goal is to be efficient. Use `<DEL_W>` for small edits and `<DEL_S>` or `<DEL_A>` for larger ones. **Rule**: You may use up to **three** `<DEL_W>` tokens within a single sentence. If a sentence requires deleting more than three words, you MUST use `<DEL_S>` or `<DEL_A>` instead.\n"
        "3.  **Grammatical Perfection**: The final, corrected response must be grammatically flawless and semantically identical to the `Correct Answer`.\n\n"

        "# Token-Based Editing Language\n"
        "Your edits are performed by inserting tokens and new text into the `Incorrect Response`. You have three deletion tokens. If a correction only requires adding text, no token is needed.\n\n"
        "---\n"
        "### `<DEL_W>`: The Word Deleter\n"
        "   - **Function**: Deletes the single word immediately preceding it.\n"
        "   - **Definition of a Word**: A string of characters separated by spaces. Punctuation is considered part of the word (e.g., 'end.', 'day?').\n"
        "   - **Usage Schema**: `... [word-to-delete]<DEL_W> [replacement-text]`\n"
        "   - **Example**:\n"
        "     - **Incorrect**: The sky is green.\n"
        "     - **Edit**: The sky is green<DEL_W> blue.\n"
        "     - **Result**: The sky is blue.\n"
        "---\n"
        "### `<DEL_S>`: The Sentence Deleter\n"
        "   - **Function**: Deletes text from its position back to the beginning of the current sentence.\n"
        "   - **Usage 1 (Full Sentence Deletion)**: Placing it at the end of a sentence deletes the entire sentence.\n"
        "     - **Schema**: `[Sentence to delete].<DEL_S> [New sentence].`\n"
        "     - **Example**:\n"
        "       - **Incorrect**: Earth is the largest planet. It is covered in water.\n"
        "       - **Edit**: Earth is the largest planet.<DEL_S> Earth is known to harbor life.\n"
        "       - **Result**: Earth is known to harbor life.\n"
        "   - **Usage 2 (Prefix Deletion)**: Placing it mid-sentence deletes the beginning of that sentence.\n"
        "     - **Schema**: `[Prefix to delete]<DEL_S> [rest of sentence]`\n"
        "     - **Example**:\n"
        "       - **Incorrect**: The text says that 5 people left.\n"
        "       - **Edit**: The text says that<DEL_S> 5 people left.\n"
        "       - **Result**: 5 people left.\n"
        "---\n"
        "### `<DEL_A>`: The 'Delete All' Operator\n"
        "   - **Function**: Deletes all text in the response that comes before it. This is your most powerful tool for complete rewrites.\n"
        "   - **Usage Schema**: `[entire-incorrect-response]<DEL_A> [entire-new-response]`\n"
        "   - **Example**:\n"
        "     - **Incorrect**: The sun orbits the Earth.\n"
        "     - **Edit**: The sun orbits the Earth.<DEL_A> The Earth orbits the sun.\n"
        "     - **Result**: The Earth orbits the sun.\n"
        "---"
        "\n\n# Correction Masterclass (Examples)\n"
        "## Example 1: Simple Word Replacement\n"
        "**Incorrect Response**: The sky is green.\n"
        "**Corrected Response**: The sky is green<DEL_W> blue.\n\n"

        "## Example 2: Addition-Only Correction (No Token Needed)\n"
        "**Incorrect Response**: Paris is capital of France.\n"
        "**Corrected Response**: Paris is the capital of France.\n\n"

        "## Example 3: Multi-Word Deletion with `<DEL_W>`\n"
        "**Incorrect Response**: Kate sold many potatoes and tomatoes on the market.\n"
        "**Analysis**: The words 'potatoes' and 'and' are incorrect. Since this requires only two deletions, using `<DEL_W>` twice is the correct and efficient method.\n"
        "**Corrected Response**: Kate sold many potatoes<DEL_W> and<DEL_W> tomatoes on the market.\n\n"

        "## Example 4: Complete Rewrite with `<DEL_A>`\n"
        "**Incorrect Response**: The distance from the sun causes seasons. It gets warmer when we are closer.\n"
        "**Corrected Response**: The distance from the sun causes seasons. It gets warmer when we are closer.<DEL_A> The tilt of the Earth's axis causes the seasons.\n\n"

        "# Your Turn:\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Incorrect Response**: {incorrect_response}\n"
        "**Errors in response**:\n{errors}\n"
        "**Corrected Response**:"
    )

    TEMPLATE: str = (
        "You are a Master Editor AI. Your mission is to correct a flawed response with surgical precision using a special tag-based editing language.\n\n"

        "# Core Principles:\n"
        "1. **Precision is Paramount**: Your edits must be exact. Every tag you use has a specific function. Use them flawlessly.\n"
        "2. **The Principle of Efficiency**: Your goal is to be efficient. Use `<DEL_W>` for small edits and `<DEL_S>` or `<DEL_A>` for larger ones. **Rule**: You may use up to **three** `<DEL_W>` tags within a single sentence. If a sentence requires deleting more than three words, you MUST use `<DEL_S>` or `<DEL_A>` instead.\n"
        "3. **Grammatical Perfection**: The final, corrected response must be grammatically flawless and semantically identical to the `Correct Answer`.\n\n"

        "# Tag-Based Editing Language:\n"
        "Your edits are performed by inserting tags and new text into the `Incorrect Response`. You have three deletion tags. If a correction only requires adding text, no tag is needed.\n\n"
        
        "# How to use <DEL_W>: The Word Deleter\n"
        "- **Function**: Deletes the single word immediately preceding it.\n"
        "- **Definition of a Word**: A string of characters separated by spaces. Punctuation is considered part of the word (e.g., 'end.', 'day?').\n"
        "- **Usage Schema**: `... [word-to-delete]<DEL_W> [replacement-text]`\n"
        "- **Example**:\n"
        "   - **Incorrect**: The sky is green.\n"
        "   - **Edit**: The sky is green<DEL_W> blue.\n"
        "   - **Result**: The sky is blue.\n\n"
        
        "# How to use <DEL_S>: The Sentence Deleter\n"
        "- **Function**: Deletes text from its position back to the beginning of the current sentence.\n"
        "- **Usage 1 (Full Sentence Deletion)**: Placing it at the end of a sentence deletes the entire sentence.\n"
        "- **Schema**: `[Sentence to delete].<DEL_S> [New sentence].`\n"
        "- **Example**:\n"
        "   - **Incorrect**: Earth is the largest planet. It is covered in water.\n"
        "   - **Edit**: Earth is the largest planet.<DEL_S> Earth is known to harbor life.\n"
        "   - **Result**: Earth is known to harbor life.\n"
        "- **Usage 2 (Prefix Deletion)**: Placing it mid-sentence deletes the beginning of that sentence.\n"
        "- **Schema**: `[Prefix to delete]<DEL_S> [rest of sentence]`\n"
        "- **Example**:\n"
        "   - **Incorrect**: The text says that 5 people left.\n"
        "   - **Edit**: The text says that<DEL_S> 5 people left.\n"
        "   - **Result**: 5 people left.\n\n"
        
        "# How to use <DEL_A>: The 'Delete All' Operator\n"
        "- **Function**: Deletes all text in the response that comes before it. This is your most powerful tool for complete rewrites.\n"
        "- **Usage Schema**: `[entire-incorrect-response]<DEL_A> [entire-new-response]`\n"
        "- **Example**:\n"
        "   - **Incorrect**: The sun orbits the Earth.\n"
        "   - **Edit**: The sun orbits the Earth.<DEL_A> The Earth orbits the sun.\n"
        "   - **Result**: The Earth orbits the sun.\n\n"

        "# Correction Workflow: Your Master Editor's Guide:\n"
        "1. **Review the Junior Editor's Analysis**: Treat the `Errors in response` as a report from a junior assistant. Use the `location` field to quickly pinpoint potential issues, but view the `how to fix` field as a *suggestion*, not a command. Your expert judgment takes precedence.\n"
        "2. **Formulate Your Master Plan**: Perform your own independent analysis. Your primary goal is to make the `Incorrect Response` semantically identical to the `Correct Answer`. Compare them directly, using the `Context` as the source of truth, and determine the most precise and minimal changes required.\n"
        "3. **Select the Optimal Tools**: Based on *your* master plan, choose the most efficient tags for the job. Do not simply copy the junior editor's suggestion. Apply the 'Principle of Efficiency' to select the best tag (`<DEL_W>`, `<DEL_S>`, or `<DEL_A>`, if needed) for the correction you have devised.\n"
        "4. **Execute with Precision**: Construct the `Corrected Response` by inserting your chosen tags and any new text directly into the `Incorrect Response`. You are performing a surgical edit, not a full rewrite (unless your plan requires `<DEL_A>`).\n"
        "5. **Final Quality Assurance**: Read your final `Corrected Response` one last time. Does it perfectly match the meaning of the `Correct Answer`? Is it grammatically flawless? Your reputation as a Master Editor is on the line.\n\n"

        "# Correction Masterclass (Examples)\n"
        "## Example 1: Simple Word Replacement\n"
        "**Question**: What color is the sky on a clear day?\n"
        "**Context**: The sky is blue due to Rayleigh scattering of sunlight in the atmosphere.\n"
        "**Correct Answer**: The sky is blue.\n"
        "**Incorrect Response**: The sky is green.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Incorrect color\n"
        "Error Location: green\n"
        "How to fix the error: Delete 'green' and add 'blue'.\n"
        "**Corrected Response**: The sky is green<DEL_W> blue.\n\n"

        "## Example 2: Deleting an Unnecessary Word\n"
        "**Question**: What is the capital of France?\n"
        "**Context**: Paris is the capital of France.\n"
        "**Correct Answer**: Paris is the capital of France.\n"
        "**Incorrect Response**: Paris is the beautiful capital of France.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Unnecessary adjective\n"
        "Error Location: beautiful\n"
        "How to fix the error: Delete the word 'beautiful'.\n"
        "**Corrected Response**: Paris is the beautiful<DEL_W> capital of France.\n\n"

        "## Example 3: Correcting an Incomplete Suggestion\n"
        "**Question**: What did Kate sell on the market?\n"
        "**Context**: The report says Kate sold many tomatoes on the market.\n"
        "**Correct Answer**: Kate sold many tomatoes on the market.\n"
        "**Incorrect Response**: Kate sold many potatoes and tomatoes on the market.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Incorrect item listed\n"
        "Error Location: potatoes\n"
        "How to fix the error: Delete 'potatoes and'.\n"
        "**Corrected Response**: Kate sold many potatoes<DEL_W> and<DEL_W> tomatoes on the market.\n\n"

        "## Example 4: Fixing a Sentence\n"
        "**Question**: What are the first three planets from the Sun, and what is a key feature of the third one?\n"
        "**Context**: The order of planets is Mercury, Venus, Earth... Earth, the third planet, is the only planet in our solar system known to harbor life.\n"
        "**Correct Answer**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the only planet in our solar system known to harbor life.\n"
        "**Incorrect Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Incorrect fact about Earth\n"
        "Error Location: Earth is the largest planet.\n"
        "How to fix the error: Delete the second sentence.\n"
        "**Corrected Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.<DEL_S> Earth is the only planet in our solar system known to harbor life.\n\n"

        "## Example 5: Using <DEL_S> to delete the first part of the sentence\n"
        "**Question**: How many people left the room during the first 10 minutes?\n"
        "**Context**: After 10 minutes, 5 people left the room.\n"
        "**Correct Answer**: 5 people\n"
        "**Incorrect Response**: The context does not provide a clear answer, yet it states that 5 people left the room.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: The response contradicts itself and initially provide wrong information\n"
        "Error Location: The context does not provide a clear answer, yet it states that\n"
        "How to fix the error: Delete 'The context does not provide a clear answer, yet it states that'\n"
        "**Corrected Response**: The context does not provide a clear answer, yet it states that<DEL_S> 5 people left the room.\n\n"

        "## Example 6: Complete Rewrite\n"
        "**Question**: What is the primary cause of Earth's seasons?\n"
        "**Context**: The Earth's tilt on its axis is the primary cause of the seasons.\n"
        "**Correct Answer**: The tilt of the Earth's axis causes the seasons.\n"
        "**Incorrect Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Fundamentally incorrect reasoning\n"
        "Error Location: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
        "How to fix the error: Delete the entire response and replace it with the correct answer.\n"
        "**Corrected Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.<DEL_A> The tilt of the Earth's axis causes the seasons.\n\n"

        "# Your Turn:\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Incorrect Response**: {incorrect_response}\n"
        "**Errors in response**:\n{errors}\n"
        "**Corrected Response**:"
    )


    TEMPLATE: str = (
        "You are a Master Editor AI. Your mission is to correct a flawed response with surgical precision using a special tag-based editing language.\n\n"

        "# Core Principles:\n"
        "1. **Precision is Paramount**: Your edits must be exact. Every tag you use has a specific function. Use them flawlessly.\n"
        "2. **The Principle of Efficiency**: Your goal is to be efficient. Use `<DEL_W>` for small edits and `<DEL_S>` or `<DEL_A>` for larger ones. **Rule**: You may use up to **three** `<DEL_W>` tags within a single sentence. If a sentence requires deleting more than three words, you MUST use `<DEL_S>` or `<DEL_A>` instead.\n"
        "3. **Grammatical Perfection**: The final, corrected response must be grammatically flawless and semantically identical to the `Correct Answer`.\n\n"

        "# Tag-Based Editing Language:\n"
        "Your edits are performed by inserting tags and new text into the `Incorrect Response`. You have three deletion tags. If a correction only requires adding text, no tag is needed.\n\n"
        
        "# How to use <DEL_W>: The Word Deleter\n"
        "- **Function**: Deletes the single word immediately preceding it.\n"
        "- **Definition of a Word**: A string of characters separated by spaces. Punctuation is considered part of the word (e.g., 'end.', 'day?').\n"
        "- **Usage Schema**: `... [word-to-delete]<DEL_W> [replacement-text]`\n"
        "- **Example 1 (Simple Word Replacement)**:\n"
        "   - **Incorrect**: The sky is green.\n"
        "   - **Edit**: The sky is green<DEL_W> blue.\n"
        "   - **Result**: The sky is blue.\n\n"
        
        "# How to use <DEL_S>: The Sentence Deleter\n"
        "- **Function**: Deletes text from its position back to the beginning of the current sentence.\n"
        "- **Usage 1 (Full Sentence Deletion)**: Placing it at the end of a sentence deletes the entire sentence.\n"
        "- **Schema**: `[Sentence to delete].<DEL_S> [New sentence].`\n"
        "- **Example**:\n"
        "   - **Incorrect**: Earth is the largest planet. It is covered in water.\n"
        "   - **Edit**: Earth is the largest planet.<DEL_S> Earth is known to harbor life.\n"
        "   - **Result**: Earth is known to harbor life.\n"
        "- **Usage 2 (Prefix Deletion)**: Placing it mid-sentence deletes the beginning of that sentence.\n"
        "- **Schema**: `[Prefix to delete]<DEL_S> [rest of sentence]`\n"
        "- **Example**:\n"
        "   - **Incorrect**: The text says that 5 people left.\n"
        "   - **Edit**: The text says that<DEL_S> 5 people left.\n"
        "   - **Result**: 5 people left.\n\n"
        "- **Usage 3 (Mid-Sentence Phrase Deletion)**: This is the mandatory method for deleting a phrase or any group of three or more consecutive words from any part of a sentence. The only correct procedure is to append `<DEL_S>` to the end of the original sentence and then write the corrected sentence.\n"
        "- **Schema**: `[The entire original sentence with the part to delete].<DEL_S> [The entire corrected sentence].`\n"
        "- **Example**:\n"
        "   - **Incorrect**: The researcher diligently and with extreme focus recorded the results in the logbook.\n"
        "   - **Analysis**: The phrase 'diligently and with extreme focus' must be removed. It contains 6 words, which is more than the limit for `<DEL_W>`. Therefore, the entire sentence must be rewritten using `<DEL_S>`.\n"
        "   - **Edit**: The researcher diligently and with extreme focus recorded the results in the logbook.<DEL_S> The researcher recorded the results in the logbook.\n"
        "   - **Result**: The researcher recorded the results in the logbook.\n\n"
        
        "# How to use <DEL_A>: The 'Delete All' Operator\n"
        "- **Function**: Deletes all text in the response that comes before it. This is your most powerful tool for complete rewrites.\n"
        "- **Usage Schema**: `[entire-incorrect-response]<DEL_A> [entire-new-response]`\n"
        "- **Example**:\n"
        "   - **Incorrect**: The sun orbits the Earth.\n"
        "   - **Edit**: The sun orbits the Earth.<DEL_A> The Earth orbits the sun.\n"
        "   - **Result**: The Earth orbits the sun.\n\n"

        "# Correction Workflow: Your Master Editor's Guide:\n"
        "1. **Review the Junior Editor's Analysis**: Treat the `Errors in response` as a report from a junior assistant. Use the `location` field to quickly pinpoint potential issues, but view the `how to fix` field as a *suggestion*, not a command. Your expert judgment takes precedence.\n"
        "2. **Formulate Your Master Plan**: Perform your own independent analysis. Your primary goal is to make the `Incorrect Response` semantically identical to the `Correct Answer`. Compare them directly, using the `Context` as the source of truth, and determine the most precise and minimal changes required.\n"
        "3. **Select the Optimal Tools**: Based on *your* master plan, choose the most efficient tags for the job. Do not simply copy the junior editor's suggestion. Apply the 'Principle of Efficiency' to select the best tag (`<DEL_W>`, `<DEL_S>`, or `<DEL_A>`, if needed) for the correction you have devised.\n"
        "4. **Execute with Precision**: Construct the `Corrected Response` by inserting your chosen tags and any new text directly into the `Incorrect Response`. You are performing a surgical edit, not a full rewrite (unless your plan requires `<DEL_A>`).\n"
        "5. **Final Quality Assurance**: Read your final `Corrected Response` one last time. Does it perfectly match the meaning of the `Correct Answer`? Is it grammatically flawless? Your reputation as a Master Editor is on the line.\n\n"

        "# Correction Masterclass (Examples)\n"
        "## Example 1: Simple Word Replacement\n"
        "**Question**: What color is the sky on a clear day?\n"
        "**Context**: The sky is blue due to Rayleigh scattering of sunlight in the atmosphere.\n"
        "**Correct Answer**: The sky is blue.\n"
        "**Incorrect Response**: The sky is green.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Incorrect color\n"
        "Error Location: green\n"
        "How to fix the error: Delete 'green' and add 'blue'.\n"
        "**Corrected Response**: The sky is green<DEL_W> blue.\n\n"

        "## Example 2: Deleting an Unnecessary Word\n"
        "**Question**: What is the capital of France?\n"
        "**Context**: Paris is the capital of France.\n"
        "**Correct Answer**: Paris is the capital of France.\n"
        "**Incorrect Response**: Paris is the beautiful capital of France.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Unnecessary adjective\n"
        "Error Location: beautiful\n"
        "How to fix the error: Delete the word 'beautiful'.\n"
        "**Corrected Response**: Paris is the beautiful<DEL_W> capital of France.\n\n"

        "## Example 3: Correcting an Incomplete Suggestion\n"
        "**Question**: What did Kate sell on the market?\n"
        "**Context**: The report says Kate sold many tomatoes on the market.\n"
        "**Correct Answer**: Kate sold many tomatoes on the market.\n"
        "**Incorrect Response**: Kate sold many potatoes and tomatoes on the market.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Incorrect item listed\n"
        "Error Location: potatoes\n"
        "How to fix the error: Delete 'potatoes and'.\n"
        "**Corrected Response**: Kate sold many potatoes<DEL_W> and<DEL_W> tomatoes on the market.\n\n"

        "## Example 4: Fixing a Sentence\n"
        "**Question**: What are the first three planets from the Sun, and what is a key feature of the third one?\n"
        "**Context**: The order of planets is Mercury, Venus, Earth... Earth, the third planet, is the only planet in our solar system known to harbor life.\n"
        "**Correct Answer**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the only planet in our solar system known to harbor life.\n"
        "**Incorrect Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Incorrect fact about Earth\n"
        "Error Location: Earth is the largest planet.\n"
        "How to fix the error: Delete the second sentence.\n"
        "**Corrected Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.<DEL_S> Earth is the only planet in our solar system known to harbor life.\n\n"

        "## Example 5: Using <DEL_S> to delete the first part of the sentence\n"
        "**Question**: How many people left the room during the first 10 minutes?\n"
        "**Context**: After 10 minutes, 5 people left the room.\n"
        "**Correct Answer**: 5 people\n"
        "**Incorrect Response**: The context does not provide a clear answer, yet it states that 5 people left the room.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: The response contradicts itself and initially provide wrong information\n"
        "Error Location: The context does not provide a clear answer, yet it states that\n"
        "How to fix the error: Delete 'The context does not provide a clear answer, yet it states that'\n"
        "**Corrected Response**: The context does not provide a clear answer, yet it states that<DEL_S> 5 people left the room.\n\n"

        "## Example 6: Deleting a Multi-Word Phrase from a Sentence\n"
        "**Question**: How many individual colleges are part of Notre Dame?\n"
        "**Context**: The undergraduate component of the university is organized into four colleges (Arts and Letters, Science, Engineering, Business) and the Architecture School.\n"
        "**Correct Answer**: There are 4 colleges part of Notre Dame.\n"
        "**Incorrect Response**: There are 4 colleges and 1 Architecture School part of Notre Dame.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: The question asks about the number of colleges, but the response includes the Architecture School, which is not a college.\n"
        "Error Location: colleges and 1 Architecture School\n"
        "How to fix the error: DELETE: 'and 1 Architecture School'\n"
        "**Corrected Response**: There are 4 colleges and 1 Architecture School part of Notre Dame.<DEL_S> There are 4 colleges part of Notre Dame.\n\n"

        "## Example 7: Complete Rewrite\n"
        "**Question**: What is the primary cause of Earth's seasons?\n"
        "**Context**: The Earth's tilt on its axis is the primary cause of the seasons.\n"
        "**Correct Answer**: The tilt of the Earth's axis causes the seasons.\n"
        "**Incorrect Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Fundamentally incorrect reasoning\n"
        "Error Location: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
        "How to fix the error: Delete the entire response and replace it with the correct answer.\n"
        "**Corrected Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.<DEL_A> The tilt of the Earth's axis causes the seasons.\n\n"

        "# Your Turn:\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Incorrect Response**: {incorrect_response}\n"
        "**Errors in response**:\n{errors}\n"
        "**Corrected Response**:"
    )


    def format(
            self,
            question: str,
            context: str,
            answer: str,
            response: str,
            errors: list[Error],
    ) -> str:
        prompt = self.TEMPLATE.format(
            question=question, 
            context=context, 
            correct_answer=answer, 
            incorrect_response=response, 
            errors=format_errors(errors),
        )
        
        # print(prompt)
        
        return prompt


class ContextQAResponseVerificationPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are a meticulous AI Verification Engine. Your sole purpose is to determine if a `Generated Answer` is correct by strictly comparing it against a `Correct Answer` and its supporting `Context`.\n\n"
        "# Task Inputs\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Generated Answer**: {corrected_response}\n\n"
        "# Evaluation Criteria:\n"
        "A `Generated Answer` is considered **CORRECT** if and only if it meets **ALL four** of the following conditions:\n"
        "1. **Semantic Equivalence**: Its core meaning is identical to the `Correct Answer`.\n"
        "2. **Contextual Grounding**: All information within it is fully supported by the provided `Context`.\n"
        "3. **Factual Accuracy**: It contains no factual errors or contradictions.\n"
        "4. **Grammatical Integrity**: It is free of grammatical errors.\n\n"
        "If the `Generated Answer` fails even one of these criteria, it is considered **INCORRECT**.\n\n"
        "# Output Requirement\n"
        "Your entire response MUST be a single word:\n"
        "- `True` if the `Generated Answer` is CORRECT (meeting all four criteria).\n"
        "- `False` if the `Generated Answer` is INCORRECT (failing one or more criteria).\n\n"
        "Do not include any other text, explanations, markdown, or formatting.\n\n"
        "Response (True/False): "
    )

    def format(
            self,
            question: str,
            context: str,
            answer: str,
            corrected_response: str,
    ) -> str:
        corrected_response = apply_del_tokens(corrected_response)
        prompt = self.TEMPLATE.format(
            question=question, 
            context=context,
            correct_answer=answer,
            corrected_response=corrected_response if corrected_response.strip() else "No response provided.",
        )
        
        # print(prompt)
        
        return prompt



class MathQAErrorCheckPrompt(StringPromptTemplate):
   
    TEMPLATE: str = (
        "You are a meticulous AI quality assurance expert. Your sole function is to evaluate a given response based *only* on the provided context and correct answer. You MUST NOT use any external knowledge.\n\n"
        "# Inputs\n"
        "**Question**: {question}\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Response to Evaluate**: {response}\n\n"
        "# Task\n"
        "Your mission is to meticulously analyze the `Response to Evaluate` and identify all its flaws. Compare it against the `Context` and `Correct Answer` to find any discrepancy or poor response quality. Your goal is to create a list of all errors in the response.\n\n"
        "Think step-by-step:\n"
        "1.  Does the response correctly answer the question when compared to the `Correct Answer`?\n"
        "2.  Is all information in the response supported by the `Context`?\n"
        "3.  Does the response introduce any unhelpful or evasive language? For example, phrases like 'The text does not specify...' or 'I cannot answer this...' are considered errors. The goal is to provide a direct answer, not to comment on the context's limitations.\n"
        "4.  Is there any other issue, like self-contradiction, irrelevant information, or conversational filler?\n\n"
        "Based on your analysis, compile a list of all errors you find.\n\n"
        "# Output Requirements\n"
        "1.  You MUST produce a list of errors.\n"
        "2.  For each error found, include a clear and concise description of the issue.\n"
        "3.  If the response is completely correct and faithful to the context, you MUST return an empty list: `[]`.\n"
        "4.  You MUST strictly adhere to the provided format instructions.\n"
        "5.  Do NOT add any headers, comments, or conversational text to your final output. Your entire response should be only the list.\n\n"
        "# Format Instructions\n"
        "{format_instructions}\n\n"
        "List of Errors:"
    )

    TEMPLATE: str = (
        "You are a meticulous AI Math Grader and Logic Verifier. Your sole function is to evaluate a student's `Response to Evaluate` for a given math `Question` and `Correct Answer`. Your analysis must be based purely on mathematical principles.\n\n"
        "# Inputs\n"
        "**Question**: {question}\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Response to Evaluate**: {response}\n\n"
        "# Core Mission\n"
        "Your mission is to meticulously analyze the `Response to Evaluate` and produce a structured list of all its flaws in calculation, logic, and explanation.\n\n"
        "# The Unanswerable Question Rule\n"
        "**Top Priority**: If the `Correct Answer` indicates that no answer exists (e.g., it is empty, or states 'no solution' or 'undefined'), but the `Response to Evaluate` provides a numerical answer, this is a fundamental failure. In this case, you MUST report **only one error** stating this fact and then stop your analysis. As for location and deletion instructions, enter 'The entire response' to make thigns shorter.\n\n"
        "# Principles of Mathematical Evaluation\n"
        "1.  **Numerical Equivalence**: The final numerical result in the `Response to Evaluate` must be identical to the `Correct Answer`. An incorrect final answer is always a high-priority error.\n"
        "2.  **Logical Validity**: The steps and reasoning used to reach the conclusion must be mathematically sound. A correct answer achieved through flawed logic is still an error.\n"
        "3.  **Grammatical Clarity**: The explanation must be written in clear, grammatically correct language.\n"
        "4.  **Tolerance for Phrasing**: Different but logically equivalent phrasing is perfectly acceptable. Do not penalize responses for being more or less verbose if the math is correct.\n\n"
        "# Structured Error Reporting:\n"
        "For each error found, you must provide:\n"
        "1. `error`: A concrete and concise description of the mathematical or grammatical flaw.\n"
        "2. `error_location`: The exact substring from the `Response to Evaluate` that contains the error.\n"
        "3. `error_correction`: A clear, actionable plan using `DELETE:` and `ADD:` commands to fix the error.\n"
        "   - **Word Integrity**: All `DELETE` operations must apply to whole words or numbers. Do not delete parts of words.\n"
        "   - **Ensure that by following the `error_correction` you will get the `Correct Answer` with right reasoning.**\n\n"
        "# Output Requirements\n"
        "1. You MUST produce a list of structured errors.\n"
        "2. If the response is completely correct according to all rules, you MUST return an empty list: `[]`.\n"
        "3. Your entire output MUST be only the list. Do not add headers or conversational text.\n\n"
        "# Format Instructions:\n"
        "{format_instructions}\n\n"
        "List of Errors:"
    )


    TEMPLATE: str = (
        "You are an expert AI Math Grader and Logic Verifier. Your sole function is to evaluate a `Response to Evaluate` for a given math `Question` and `Correct Answer`, based on pure mathematical and logical principles.\n\n"

        "# Core Principles\n"
        "1. **Mathematical and Logical Rigor**: Your primary focus is on correctness. This includes both the final numerical answer and the logical validity of every step in the reasoning. A correct final answer derived from flawed logic is a critical error.\n"
        "2. **Tolerate Methodological Differences**: Do not penalize responses for using a different but mathematically sound method to arrive at the correct answer. Verbosity or stylistic phrasing is not an error if the underlying math and logic are correct.\n"
        "3. **Precision and Objectivity**: Every identified error must be a concrete, verifiable flaw in calculation, logic, or explanation. Your output must be a structured list with no conversational filler.\n\n"

        "# Inputs\n"
        "**Question**: {question}\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Response to Evaluate**: {response}\n\n"

        "# Instructions\n"
        "1. **Top Priority - The Unanswerable Question Rule**: First, check if the `Correct Answer` indicates no solution exists (e.g., it is empty, or states 'no solution'). If so, and the `Response to Evaluate` provides a definite answer, this is a fundamental failure. You MUST report **only one error** for this case and then stop your analysis. The error should be:\n"
        '   - `description`: "Provided a numerical answer when no solution exists."\n'
        '   - `location`: The entire text of the `Response to Evaluate`.\n'
        '   - `correction`: `DELETE: "[The entire text of the response]"`\n`ADD: "No solution exists."`\n'

        "2. **Analyze**: Carefully review the `Question`, `Correct Answer`, and the entire `Response to Evaluate`, including the reasoning and final answer.\n"
        "3. **Identify Flaws**: Pinpoint all discrepancies between the `Response to Evaluate` and the `Correct Answer`. Systematically check for:\n"
        "   - **Calculation Errors**: Incorrect arithmetic operations (e.g., 2+2=5).\n"
        "   - **Logical Fallacies**: Flaws in the reasoning or incorrect application of mathematical theorems/properties.\n"
        "   - **Final Answer Mismatch**: The final result does not match the `Correct Answer`.\n"
        "   - **Grammatical Errors**: Mistakes in the explanation that obscure the mathematical meaning.\n"
        "4. **Structure Errors**: For each substantive flaw found, you must define the following three fields with surgical precision:\n"
        "   - `description`: A concise, objective explanation of *why* it's an error.\n"
        "   - `location`: The *exact* substring from the `Response to Evaluate` that contains the error. This must be a verbatim quote.\n"
        "   - `correction`: A clear, actionable plan to fix the error. By following the plan, the response must become mathematically correct. You MUST use the following format:\n"
        '       `DELETE: "text to delete"`\n'
        '       `ADD: "text to add"`\n'
        "       - The `DELETE` part must contain text to be removed. If the fix is only an addition, do not include `DELETE` part.\n"
        "       - The `ADD` part must contain the text that should be inserted. If the fix is only a deletion, do not include `ADD` part.\n"
        "       - The text for `DELETE` and `ADD` must be whole words, numbers, or phrases, not partial words\n"
        "       **CRITICAL**: The text in `ADD` cannot be identical to the text in `DELETE`. A correction must create a meaningful change. If you find no change is needed, it is not an error.\n\n"

        "# Output Requirements\n"
        "1. You MUST produce a list of structured errors. Each error must be unique.\n"
        "2. If the response is completely correct according to all rules, you MUST return an empty list: `[]`.\n"
        "3. Do not create more than 5 errors!\n"
        "4. Do not include any headers, comments, or other text, except for the JSON output.\n"
        "5. Strictly follow the format instructions:\n"
        "{format_instructions}\n\n"
        "List of Errors:"
    )

    def format(
            self,
            question: str,
            answer: str,
            is_answerable: bool,
            response: str,
            format_instructions: str,
    ) -> str:
        correct_answer = "This question cannot be answered." if not is_answerable else answer
        prompt = self.TEMPLATE.format(
            question=question, correct_answer=correct_answer, response=response, format_instructions=format_instructions)
    
        # print(prompt)
        
        return prompt


class MathQAErrorCorrectionPrompt(StringPromptTemplate):

    TEMPLATE: str =(
        "You are an expert editor, a master of precision and conciseness. Your task is to correct an incorrect response using special deletion tokens to make the absolute minimum changes required.\n\n"
        "# Task:\n"
        "Your goal is to fix the `Incorrect Response` so that it matches the `Correct Answer`. You must use the special deletion tokens to perform the correction with the fewest edits possible.\n"

        "# Special Deletion Tokens:\n"
        "1.  <DEL_W>: Deletes the single word immediately preceding the token. A word is a string of characters separated by spaces. Punctuation is part of the word (e.g., 'end.').\n"
        "2.  <DEL_S>: Deletes the entire sentence immediately preceding the token. A sentence is a group of words ending in a period (.), question mark (?), or exclamation mark (!). The token must be placed right after the punctuation.\n"
        "3.  <DEL_A>: Deletes the entire response that comes before the token. Use this for a complete rewrite when the response is fundamentally wrong.\n\n"

        "# Instructions:\n"
        "1.  **Analyze the Error**: First, carefully compare the `Incorrect Response` to the `Correct Answer` and the `Errors in response` description to pinpoint the exact inaccuracies.\n"
        "2.  **Be Precise and Minimal**: Your objective is to use the fewest tokens and changes to achieve the correction. Do not add any new information not present in the `Correct Answer`.\n"
        "3.  **Choose the Right Token**:\n"
        "   * Use `<DEL_W>` for single-word fixes (deletions or replacements).\n"
        "   * Use `<DEL_S>` when an entire sentence is incorrect.\n"
        "   * Use `<DEL_A>` when the entire response is wrong and needs a fresh start.\n"
        "4.  **Construct the Correction**: Place the deletion token immediately after the text you need to remove. If you are replacing text, add the correct text right after the token.\n"
        "5.  **Final Output**: Provide only the final, corrected response. Do not include any extra text, headers, or explanations.\n\n"

        "# Correction Masterclass (Examples):\n"
        "## Example 1: Simple Word Replacement\n"
        "**Question**: What color is the sky on a clear day?\n"
        "**Context**: The sky is blue due to Rayleigh scattering of sunlight in the atmosphere.\n"
        "**Correct Answer**: The sky is blue.\n"
        "**Incorrect Response**: The sky is green.\n"
        "**Errors in response**: The color is wrong.\n"
        "**Corrected Response**: The sky is green<DEL_W> blue.\n\n"

        "## Example 2: Deleting an Unnecessary Word\n"
        "**Question**: What is the capital of France?\n"
        "**Context**: Paris is the capital of France.\n"
        "**Correct Answer**: Paris is the capital of France.\n"
        "**Incorrect Response**: Paris is the beautiful capital of France.\n"
        "**Errors in response**: The word 'beautiful' is unnecessary and not part of the correct answer.\n"
        "**Corrected Response**: Paris is the beautiful<DEL_W> capital of France.\n\n"

        "## Example 3: Fixing a Sentence\n"
        "**Question**: What are the first three planets from the Sun, and what is a key feature of the third one?\n"
        "**Context**: The order of planets is Mercury, Venus, Earth... Earth, the third planet, is the only planet in our solar system known to harbor life.\n"
        "**Correct Answer**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the only planet in our solar system known to harbor life.\n"
        "**Incorrect Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.\n"
        "**Errors in response**: The second sentence contains an incorrect fact about Earth.\n"
        "**Corrected Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.<DEL_S> Earth is the only planet in our solar system known to harbor life.\n\n"

        "## Example 4: Complete Rewrite\n"
        "**Question**: What is the primary cause of Earth's seasons?\n"
        "**Context**: The Earth's tilt on its axis is the primary cause of the seasons.\n"
        "**Correct Answer**: The tilt of the Earth's axis causes the seasons.\n"
        "**Incorrect Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
        "**Errors in response**: The entire reasoning is incorrect. The cause is the Earth's tilt, not its distance from the sun.\n"
        "**Corrected Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.<DEL_A> The tilt of the Earth's axis causes the seasons.\n\n"

        "# Your Turn:\n"
        "**Question**: {question}\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Incorrect Response**: {incorrect_response}\n"
        "**Errors in response**: {errors}\n"
        "**Corrected Response**:"
    )

    TEMPLATE: str = (
        "You are an AI Math Tutor and Editor. Your task is to perform a final, definitive correction on a flawed mathematical response using a special token-based editing language. You will act as the final authority, using a junior editor's analysis as guidance but not as a command.\n\n"
        "# Core Task\n"
        "You will be given the `Question`, `Correct Answer`, `Incorrect Response`, and `Errors in response`. Your mission is to apply precise edits to the `Incorrect Response` to make it mathematically sound and produce the `Correct Answer`.\n\n"
        "# Token-Based Editing Language:\n"
        "1. **<DEL_W> (Delete Word)**: Deletes the single word or number immediately preceding it. Punctuation is part of the word (e.g., '25.').\n"
        "   - **Pro Tip**: Use this for surgical fixes, like correcting a single incorrect word, number, operator, or variable in an otherwise correct line of reasoning.\n\n"
        "2. **<DEL_S> (Delete Sentence)**: Deletes text from its position back to the beginning of the **current sentence or line**.\n"
        # "   - **Pro Tip**: This is your primary tool for correcting flawed logic. If an entire line of reasoning or a step in the calculation is wrong, use `<DEL_S>` to remove it efficiently instead of using many `<DEL_W>` tokens.\n\n"
        "   - **Pro Tip**: This is your primary tool for fixing flawed equations. Attempting to patch a complex calculation with multiple <DEL_W> tokens is difficult and error-prone. It is far more effective to delete the entire incorrect line or equation using <DEL_S> and then write the correct version.\n\n"
        "3. **<DEL_A> (Delete All)**: Deletes the entire response that comes before it.\n"
        "   - **Pro Tip**: Use this for a complete rewrite when the student's entire approach to the problem is fundamentally wrong and needs a fresh start.\n\n"
        "# Your Workflow\n"
        "1. **Review the Analysis**: Carefully examine the `Errors in response` to understand the core mathematical or logical flaws.\n"
        "2. **Formulate Your Master Plan**: Do not blindly follow the suggestions. Prioritize correcting the logic. If a line of reasoning is incorrect, prefer using `<DEL_S>`. If only a word, number or operator is wrong, use `<DEL_W>`. Devise the most efficient correction that results in a valid mathematical derivation of the `Correct Answer`.\n"
        "3. **Execute the Edit**: Apply your plan to the `Incorrect Response`. This is a direct edit, not a rewrite. You will insert the tokens and any new text directly into the original string.\n"
        "4. **Final Output**: Ensure your response contains *only* the final, corrected text. No extra headers, explanations, or conversational text.\n\n"
        "# Math Correction Examples\n"
        "## Example 1: Fixing a Calculation Error\n"
        "**Question**: What is 5 * 5?\n"
        "**Correct Answer**: 25\n"
        "**Incorrect Response**: 5 * 5 is 24.\n"
        "**Errors in response**:\n"
        "Error: Incorrect result.\n"
        "Error Location: 24\n"
        "Error Correction: Delete '24' and add '25'.\n"
        "**Corrected Response**: 5 * 5 is 24<DEL_W> 25.\n\n"

        "## Example 2: Correcting a Logical Step with <DEL_W>\n"
        "**Question**: What is x if 2x + 5 = 15?\n"
        "**Correct Answer**: x = 5\n"
        "**Incorrect Response**: 2x = 15 + 5, so 2x = 20, so x = 10.\n"
        "**Errors in response**:\n"
        "Error: Incorrect operation in the first step.\n"
        "Error Location: +\n"
        "Error Correction: Delete '+' and add '-'.\n"
        "**Corrected Response**: 2x = 15 +<DEL_W> - 5, so 2x = 10, so x = 5.\n\n"

        "## Example 3: Removing Flawed Reasoning with <DEL_S>\n"
        "**Question**: What is the area of a circle with a radius of 3?\n"
        "**Correct Answer**: The area is 9(pi).\n"
        "**Incorrect Response**: The formula for circumference is 2(pi)r. So the area is 6(pi).\n"
        "**Errors in response**:\n"
        "Error: The student used the wrong formula (circumference instead of area).\n"
        "Error Location: The formula for circumference is 2(pi)r.\n"
        "Error Correction: Delete the first incorrect sentence.\n"
        "**Corrected Response**: The formula for circumference is 2(pi)r.<DEL_S> The formula for area is (pi)r^2, so the area is 9(pi).\n\n"

        "## Example 4: Complete Rewrite with <DEL_A>\n"
        "**Question**: What is the value of 4^3?\n"
        "**Correct Answer**: 64\n"
        "**Incorrect Response**: 4^3 means 4 * 3, which is 12.\n"
        "**Errors in response**:\n"
        "Error: The student misunderstood the concept of exponentiation.\n"
        "Error Location: 4^3 means 4 * 3, which is 12.\n"
        "Error Correction: Delete the entire response and replace it with the correct calculation.\n"
        "**Corrected Response**: 4^3 means 4 * 3, which is 12.<DEL_A> 4^3 means 4 * 4 * 4, which is 64.\n\n"

        "# Your Turn:\n"
        "**Question**: {question}\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Incorrect Response**: {incorrect_response}\n"
        "**Errors in response**:\n{errors}\n"
        "**Corrected Response**:"
    )

    TEMPLATE: str = (
        "You are an AI Math Tutor and Editor. Your task is to perform a final, definitive correction on a flawed mathematical response using a special token-based editing language. You will act as the final authority, using a junior editor's analysis as guidance but not as a command.\n\n"
        "# Core Task\n"
        "You will be given the `Question`, `Correct Answer`, `Incorrect Response`, and `Errors in response`. Your mission is to apply precise edits to the `Incorrect Response` to make it mathematically sound and produce the `Correct Answer`.\n\n"
        "# Token-Based Editing Language:\n"
        "1. **<DEL_W> (Delete Word)**: Deletes the single word or number immediately preceding it. Punctuation is part of the word (e.g., '25.').\n"
        "   - **Pro Tip**: Use this for surgical fixes, like correcting a single incorrect word, number, operator, or variable in an otherwise correct line of reasoning.\n\n"
        "2. **<DEL_S> (Delete Sentence)**: Deletes text from its position back to the beginning of the **current sentence or line**.\n"
        "   - **Pro Tip**: This is your primary tool for fixing flawed equations. Attempting to patch a complex calculation with multiple <DEL_W> tokens is difficult and error-prone. It is far more effective to delete the entire incorrect line or equation using <DEL_S> and then write the correct version.\n\n"
        "3. **<DEL_A> (Delete All)**: Deletes the entire response that comes before it.\n"
        "   - **Pro Tip**: Use this for a complete rewrite when the student's entire approach to the problem is fundamentally wrong and needs a fresh start.\n\n"
        "# Your Workflow\n"
        "1. **Review the Analysis**: Carefully examine the `Errors in response` to understand the core mathematical or logical flaws.\n"
        "2. **Formulate Your Master Plan**: Do not blindly follow the suggestions. Prioritize correcting the logic. If a line of reasoning is incorrect, prefer using `<DEL_S>`. If only a word, number or operator is wrong, use `<DEL_W>`. Devise the most efficient correction that results in a valid mathematical derivation of the `Correct Answer`.\n"
        "3. **Execute the Edit**: Apply your plan to the `Incorrect Response`. This is a direct edit, not a rewrite. You will insert the tokens and any new text directly into the original string.\n"
        "4. **Final Output**: Ensure your response contains *only* the final, corrected text. No extra headers, explanations, or conversational text.\n\n"
        "# Math Correction Examples\n"
        "## Example 1: Fixing a Calculation Error\n"
        "**Question**: What is 5 * 5?\n"
        "**Correct Answer**: 25\n"
        "**Incorrect Response**: 5 * 5 is 24.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Incorrect result.\n"
        "Error Location: 24\n"
        "How to fix the error: Delete '24' and add '25'.\n"
        "**Corrected Response**: 5 * 5 is 24<DEL_W> 25.\n\n"

        "## Example 2: Correcting a Logical Step with <DEL_W>\n"
        "**Question**: What is x if 2x + 5 = 15?\n"
        "**Correct Answer**: x = 5\n"
        "**Incorrect Response**: 2x = 15 + 5, so 2x = 20, so x = 10.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: Incorrect operation in the first step.\n"
        "Error Location: +\n"
        "How to fix the error: Delete '+' and add '-'.\n"
        "**Corrected Response**: 2x = 15 +<DEL_W> - 5, so 2x = 10, so x = 5.\n\n"

        "## Example 3: Removing Flawed Reasoning with <DEL_S>\n"
        "**Question**: What is the area of a circle with a radius of 3?\n"
        "**Correct Answer**: The area is 9(pi).\n"
        "**Incorrect Response**: The formula for circumference is 2(pi)r. So the area is 6(pi).\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: The student used the wrong formula (circumference instead of area).\n"
        "Error Location: The formula for circumference is 2(pi)r.\n"
        "How to fix the error: Delete the first incorrect sentence.\n"
        "**Corrected Response**: The formula for circumference is 2(pi)r.<DEL_S> The formula for area is (pi)r^2, so the area is 9(pi).\n\n"

        "## Example 4: Complete Rewrite with <DEL_A>\n"
        "**Question**: What is the value of 4^3?\n"
        "**Correct Answer**: 64\n"
        "**Incorrect Response**: 4^3 means 4 * 3, which is 12.\n"
        "**Errors in response**:\n"
        "Error 1:\n"
        "Error Description: The student misunderstood the concept of exponentiation.\n"
        "Error Location: 4^3 means 4 * 3, which is 12.\n"
        "How to fix the error: Delete the entire response and replace it with the correct calculation.\n"
        "**Corrected Response**: 4^3 means 4 * 3, which is 12.<DEL_A> 4^3 means 4 * 4 * 4, which is 64.\n\n"

        "# Your Turn:\n"
        "**Question**: {question}\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Incorrect Response**: {incorrect_response}\n"
        "**Errors in response**:\n{errors}\n"
        "**Corrected Response**:"
    )

    def format(
            self,
            question: str,
            answer: str,
            is_answerable: bool,
            response: str,
            errors: str,
    ) -> str:
        correct_answer = "This question cannot be answered." if not is_answerable else answer
        prompt = self.TEMPLATE.format(
            question=question, 
            correct_answer=correct_answer, 
            incorrect_response=response, 
            # errors=format_errors(errors),
            errors=errors,
        )
    
        # print(prompt)
        
        return prompt


class MathQAResponseVerificationPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are a meticulous AI Math Verification Engine. Your sole purpose is to determine if a `Generated Answer` is correct by strictly comparing it against the `Correct Answer` for a given `Question`.\n\n"
        "# Task Inputs\n"
        "**Question**: {question}\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Generated Answer**: {corrected_response}\n\n"
        "# Evaluation Criteria\n"
        "A `Generated Answer` is considered **CORRECT** if and only if it meets ALL of the following conditions:\n"
        "1.  **Numerical and Logical Equivalence**: The final numerical result and the logical reasoning (if present) are identical to the `Correct Answer`. Phrasing can differ, but the mathematical conclusion must be the same.\n"
        "2.  **Factual Accuracy**: It contains no mathematical errors or contradictions within its steps.\n"
        "3.  **Grammatical Integrity**: It is written in clear, grammatically correct language.\n\n"
        "If the `Generated Answer` fails even one of these criteria, it is considered **INCORRECT**.\n\n"
        "# Output Requirement\n"
        "Your entire response MUST be a single word:\n"
        "-   `True` if the `Generated Answer` is CORRECT (meeting all criteria).\n"
        "-   `False` if the `Generated Answer` is INCORRECT (failing one or more criteria).\n\n"
        "Do not include any other text, explanations, markdown, or formatting.\n\n"
        "Response (True/False): "
    )

    def format(
            self,
            question: str,
            is_answerable: bool,
            answer: str,
            corrected_response: str,
    ) -> str:
        corrected_response = apply_del_tokens(corrected_response)
        correct_answer = "This question cannot be answered." if not is_answerable else answer
        prompt = self.TEMPLATE.format(
            question=question, 
            correct_answer=correct_answer,
            corrected_response=corrected_response if corrected_response.strip() else "No response provided.",
        )
        
        # print(prompt)
        
        return prompt