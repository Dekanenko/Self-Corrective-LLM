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
        "**Crucial Rule**: Your evaluation must focus strictly on the factual and semantic correctness of the answer. You MUST NOT flag a response for containing harmless, non-essential information if the core answer is correct. A response is still considered CORRECT even if it includes:\n"
        "   - Introductory phrases: such as \"According to the text...\" or \"The context states...\"\n"
        "   - Benign conversational text or formatting: such as markdown, greetings, or other chat-like filler\n"
        "   - Minor, non-contradictory details: extra information pulled from the Context that does not alter or conflict with the core meaning of the Correct Answer\n"
        "Flag an error only when the response becomes factually incorrect, semantically different from the Correct Answer, or introduces a contradiction. Do not penalize for style or verbosity.\n\n"
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
        
        print(prompt)

        return prompt


class ContextQAErrorCorrectionPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are a Master Editor. Your task is to perform a final, definitive correction on a flawed response using a special token-based editing language. You will act as the final authority, using a junior editor's analysis as guidance but not as a command.\n\n"
        "# Core Task\n"
        "You will be given the `Question`, `Context`, `Correct Answer`, `Incorrect Response`, and `Errors in response`. Your mission is to apply precise edits to the `Incorrect Response` to make it grammatically perfect and semantically identical to the `Correct Answer`.\n\n"
        "# Token-Based Editing Language\n"
        "You have two operations at your disposal: deleting text with special tokens and adding new text. You may use one or both.\n\n"
        "1.  **<DEL_W> (Delete Word)**: Deletes the single word (not a token!) immediately preceding it. A word is a string of characters separated by spaces. Punctuation is part of the word (e.g., 'end.').\n"
        "    *   **Pro Tip**: Deleting a single word can create a grammatical flaw (e.g., a dangling 'and'). Always check the context and be prepared to delete adjacent words to maintain a grammatically perfect sentence.\n\n"
        "2.  **<DEL_S> (Delete Sentence)**: Deletes text from its position back to the beginning of the **current sentence**.\n"
        "    *   **Full Sentence Deletion**: When placed at the very end of a sentence (after the punctuation), it deletes the entire sentence.\n"
        "    *   **Prefix Deletion**: When placed in the middle of a sentence, it efficiently deletes just the beginning part of that sentence, which is more efficient than using multiple `<DEL_W>` tokens.\n\n"
        "3.  **<DEL_A> (Delete All)**: Deletes all text in the response that comes before it, up to the very beginning of the **entire response**.\n"
        "    *   **Pro Tip**: This is your most powerful tool. It is perfect for complete rewrites or for removing multiple sentences and prefixes at once.\n\n"
        "# Your Workflow\n"
        "1.  **Review the Analysis**: Carefully examine the `Errors in response`. Use this junior editor's analysis to quickly understand the core issues.\n"
        "2.  **Formulate Your Master Plan**: Do not blindly follow the suggestions. Use your expert judgment to devise the most efficient correction. The best correction uses the fewest tokens and results in a grammatically flawless sentence that matches the `Correct Answer`.\n"
        "3.  **Execute the Edit**: Apply your plan to the `Incorrect Response`. This is a direct edit, not a rewrite. You will insert the tokens and any new text directly into the original string.\n"
        "4.  **Final Output**: Ensure your response contains *only* the final, corrected text. No extra headers, explanations, or conversational text.\n\n"

        "# Correction Masterclass (Examples)\n"
        "## Example 1: Simple Word Replacement\n"
        "**Question**: What color is the sky on a clear day?\n"
        "**Context**: The sky is blue due to Rayleigh scattering of sunlight in the atmosphere.\n"
        "**Correct Answer**: The sky is blue.\n"
        "**Incorrect Response**: The sky is green.\n"
        "**Errors in response**:\n"
        "Error: Incorrect color\n"
        "Error Location: green\n"
        "Error Correction: Delete 'green' and add 'blue'.\n"
        "**Corrected Response**: The sky is green<DEL_W> blue.\n\n"

        "## Example 2: Deleting an Unnecessary Word\n"
        "**Question**: What is the capital of France?\n"
        "**Context**: Paris is the capital of France.\n"
        "**Correct Answer**: Paris is the capital of France.\n"
        "**Incorrect Response**: Paris is the beautiful capital of France.\n"
        "**Errors in response**:\n"
        "Error: Unnecessary adjective\n"
        "Error Location: beautiful\n"
        "Error Correction: Delete the word 'beautiful'.\n"
        "**Corrected Response**: Paris is the beautiful<DEL_W> capital of France.\n\n"

        "## Example 3: Correcting an Incomplete Suggestion\n"
        "**Question**: What did Kate sell on the market?\n"
        "**Context**: The report says Kate sold many tomatoes on the market.\n"
        "**Correct Answer**: Kate sold many tomatoes on the market.\n"
        "**Incorrect Response**: Kate sold many potatoes and tomatoes on the market.\n"
        "**Errors in response**:\n"
        "Error: Incorrect item listed\n"
        "Error Location: potatoes\n"
        "Error Correction: Delete 'potatoes and'.\n"
        "**Corrected Response**: Kate sold many potatoes<DEL_W> and<DEL_W> tomatoes on the market.\n\n"

        "## Example 4: Fixing a Sentence\n"
        "**Question**: What are the first three planets from the Sun, and what is a key feature of the third one?\n"
        "**Context**: The order of planets is Mercury, Venus, Earth... Earth, the third planet, is the only planet in our solar system known to harbor life.\n"
        "**Correct Answer**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the only planet in our solar system known to harbor life.\n"
        "**Incorrect Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.\n"
        "**Errors in response**:\n"
        "Error: Incorrect fact about Earth\n"
        "Error Location: Earth is the largest planet.\n"
        "Error Correction: Delete the second sentence.\n"
        "**Corrected Response**: The first three planets from the Sun are Mercury, Venus, and Earth. Earth is the largest planet in our solar system.<DEL_S> Earth is the only planet in our solar system known to harbor life.\n\n"

        "## Example 5: Using <DEL_S> to delete the first part of the sentence\n"
        "**Question**: How many people left the room during the first 10 minutes?\n"
        "**Context**: After 10 minutes, 5 people left the room.\n"
        "**Correct Answer**: 5 people\n"
        "**Incorrect Response**: The context does not provide a clear answer, yet it states that 5 people left the room.\n"
        "**Errors in response**:\n"
        "Error: The response contradicts itself and initially provide wrong information\n"
        "Error Location: The context does not provide a clear answer, yet it states that\n"
        "Error Correction: Delete 'The context does not provide a clear answer, yet it states that'\n"
        "**Corrected Response**: The context does not provide a clear answer, yet it states that<DEL_S> 5 people left the room.\n\n"

        "## Example 6: Complete Rewrite\n"
        "**Question**: What is the primary cause of Earth's seasons?\n"
        "**Context**: The Earth's tilt on its axis is the primary cause of the seasons.\n"
        "**Correct Answer**: The tilt of the Earth's axis causes the seasons.\n"
        "**Incorrect Response**: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
        "**Errors in response**:\n"
        "Error: Fundamentally incorrect reasoning\n"
        "Error Location: The distance from the sun causes the seasons. It gets warmer when we are closer to the sun.\n"
        "Error Correction: Delete the entire response and replace it with the correct answer.\n"
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
        
        print(prompt)
        
        return prompt


class ContextQAResponseVerificationPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are a meticulous AI Verification Engine. Your sole purpose is to determine if a `Generated Answer` is correct by strictly comparing it against a `Correct Answer` and its supporting `Context`.\n\n"
        "# Task Inputs\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Generated Answer**: {corrected_response}\n\n"
        "# Evaluation Criteria\n"
        "A `Generated Answer` is considered **CORRECT** if and only if it meets ALL four of the following conditions:\n"
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
        
        print(prompt)
        
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
    
        print(prompt)
        
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
            question=question, correct_answer=correct_answer, incorrect_response=response, errors=errors)
    
        print(prompt)
        
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
        
        print(prompt)
        
        return prompt