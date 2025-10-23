from langchain.prompts import StringPromptTemplate
from src.utils.formatting import apply_del_tokens, format_errors
from src.models import ShortError


class ContextQAErrorCheckPrompt(StringPromptTemplate):

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
            question=question,
            context=context,
            correct_answer=answer,
            response=response,
            format_instructions=format_instructions,
        )

        return prompt


class ContextQAEvalErrorCheckPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are a meticulous and objective AI judge. Your sole purpose is to evaluate the factual and semantic correctness of a given `Response to Evaluate` against a `Context` and a list of `Correct Answers`.\n\n"
        "You are to ignore stylistic differences, tone, or any superfluous yet correct information. Your judgment must be based purely on whether the core answer is semantically equivalent to one of the correct answers and factually supported by the context.\n\n"
        "# Inputs\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n"
        "**Correct Answers (any will do)**: {correct_answer}\n"
        "**Response to Evaluate**: {response}\n\n"
        "# Core Evaluation Principles\n"
        "1. **Semantic Equivalence**: The `Response to Evaluate` is considered correct if its essential meaning aligns with at least one of the `Correct Answers`. The exact phrasing does not matter.\n"
        "2. **Factual Consistency**: All information presented in the `Response to Evaluate` must be consistent with the provided `Context`. Any contradiction is an error.\n"
        "3. **Sufficiency**: The response must correctly address the question posed. Extraneous, but accurate, information is permissible and should not be flagged as an error.\n\n"
        "# Step-by-Step Analysis Guide (Your Internal Thought Process)\n"
        "1.  **Deconstruct**: First, understand the core requirement of the `Question`.\n"
        "2.  **Compare**: Check if the `Response to Evaluate` is semantically equivalent to any of the provided `Correct Answers`.\n"
        "3.  **Verify**: Scrutinize the `Response to Evaluate` for any claims that contradict the `Context`.\n"
        "4.  **Conclude**: If the response is semantically equivalent to a correct answer AND has no factual inconsistencies with the context, it is correct. Otherwise, it is incorrect.\n\n"
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
            question=question,
            context=context,
            correct_answer=answer,
            response=response,
            format_instructions=format_instructions,
        )

        return prompt


class ContextQAErrorCorrectionPrompt(StringPromptTemplate):

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
        errors: list[ShortError],
    ) -> str:
        prompt = self.TEMPLATE.format(
            question=question,
            context=context,
            correct_answer=answer,
            incorrect_response=response,
            errors=format_errors(errors),
        )

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
            corrected_response=(
                corrected_response
                if corrected_response.strip()
                else "No response provided."
            ),
        )

        return prompt


class MathQAErrorCheckPrompt(StringPromptTemplate):

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
        "   - `location`: The entire text of the `Response to Evaluate`.\n"
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
        correct_answer = (
            "This question cannot be answered." if not is_answerable else answer
        )
        prompt = self.TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
            format_instructions=format_instructions,
        )

        return prompt


class MathQAErrorCorrectionPrompt(StringPromptTemplate):

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
        correct_answer = (
            "This question cannot be answered." if not is_answerable else answer
        )
        prompt = self.TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            incorrect_response=response,
            errors=errors,
        )

        return prompt


class MathQAEvalErrorCheckPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are a highly precise and logical evaluator of AI-generated responses. Your single objective is to determine if the final conclusion of a response is correct, without considering the steps taken to reach it."
        "# Task\n"
        "Evaluate the 'Response to Evaluate' against the 'Correct Answer'. Your evaluation must be based exclusively on the final conclusion.\n\n"
        "# Inputs\n"
        "**Question**: {question}\n"
        "**Correct Answer**: {correct_answer}\n"
        "**Response to Evaluate**: {response}\n\n"
        "# Core Evaluation Principles\n"
        "1. **Final Conclusion is Everything**: Your evaluation must be based SOLELY on the final conclusion of the 'Response to Evaluate'. It is correct if it is semantically equivalent to the 'Correct Answer'.\n"
        "2. **Ignore Intermediate Reasoning**: The path to an answer, including any exploratory text, partial calculations, or assumptions, is completely irrelevant. Disregard it entirely.\n"
        "3. **Semantic Equivalence**: Do not perform literal string matching. 'The result is 50' is the same as '50'. 'I cannot answer this' is the same as concluding a question is unanswerable.\n\n"
        "# Handling Unanswerable Questions\n"
        "1. The 'Response to Evaluate' is correct if it meets one of the following criteria:\n"
        "- The response is empty '', 'None', or '0'.\n"
        "- The final conclusion states that there is no answer, that it is impossible to answer the question, or makes a similar statement about the question being unanswerable.\n"
        "Note: Any attempt to solve the question in the reasoning must be ignored. Base your evaluation solely on the final conclusion.\n"
        "2. The 'Response to Evaluate' is incorrect if it provides a specific numerical answer other than '0' or otherwise fails to conclude that the question is unanswerable.\n\n"
        "# Output Requirements\n"
        "1. **JSON Output Only**: Produce a list of structured error objects.\n"
        "2. **Empty List for Correct Responses**: If the final conclusion is correct, you MUST return an empty list: `[]`.\n"
        "3. **No Extra Text**: Do NOT include headers, comments, or any other text outside of the JSON output.\n"
        "4. **Strict Formatting**: Adhere strictly to the format provided:\n"
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
        correct_answer = (
            "This question cannot be answered." if not is_answerable else answer
        )
        prompt = self.TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
            format_instructions=format_instructions,
        )

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
        correct_answer = (
            "This question cannot be answered." if not is_answerable else answer
        )
        prompt = self.TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            corrected_response=(
                corrected_response
                if corrected_response.strip()
                else "No response provided."
            ),
        )

        return prompt
