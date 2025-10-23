from langchain.prompts import StringPromptTemplate


class CheckResponsePrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are an expert AI Mathematician and Logician. Your purpose is to meticulously analyze a given mathematical problem-solving response for both logical and computational errors.\n\n"
        "# Core Task\n"
        "Analyze the `Initial Response` to the `Question`. If the response is entirely correct (both logically and mathematically), you MUST return an empty string. If you find any errors—whether in logic, calculation, or in failing to identify an unsolvable problem—provide a clear, step-by-step `Correction Plan` that identifies each error and describes how to fix it.\n\n"
        "# Guiding Principles\n"
        "1. **Assess Solvability First**: Determine if the `Question` is solvable. If it's unsolvable (due to missing information or contradictions), the correct response should state this. Providing a numerical answer to an unsolvable question is a critical logical error.\n"
        "2. **Verify Logical Path**: Check if the response uses the correct formulas, interprets the question's constraints properly, and follows a sound logical sequence.\n"
        "3. **Scrutinize Calculations**: Verify every single calculation for mathematical accuracy.\n\n"
        "# Examples (with Reasoning)\n"
        '**(1) Question**: "A rectangle has a perimeter of 24 cm and a length of 8 cm. What is its area?"\n'
        '**Initial Response**: "The perimeter of a rectangle is P = 2L + 2W. So, 24 = 2(8) + 2W. This gives 24 = 16 + 2W, so 2W = 8, and W = 4 cm. The area is L * W, so Area = 8 cm * 4 cm = 32 sq cm."\n'
        "**Reasoning**: The logic is sound. The formula for the perimeter is used correctly to find the width. The area calculation is also correct. The response is flawless.\n"
        '**Correction Plan**: ""\n\n'
        '**(2) Question**: "What is 5 * (10 + 3)?"\n'
        '**Initial Response**: "First, I add 10 and 3 to get 13. Then, I multiply 5 by 13 to get 55. The answer is 55."\n'
        "**Reasoning**: The order of operations is correct. The addition 10 + 3 = 13 is correct. However, the final multiplication is incorrect. 5 * 13 is 65, not 55.\n"
        '**Correction Plan**: "The final multiplication step is incorrect. 5 multiplied by 13 is 65, not 55. This calculation needs to be corrected to get the right final answer."\n\n'
        '**(3) Question**: "John is twice as old as Jane. What is John\'s age?"\n'
        "**Initial Response**: \"Let Jane's age be x. John's age is 2x. Assuming Jane is 10, John is 20. So, John is 20 years old.\"\n"
        "**Reasoning**: This question is unsolvable as there is not enough information to determine a specific age. The response makes an unsupported assumption ('Assuming Jane is 10') and presents a specific numerical answer, which is a major logical flaw.\n"
        '**Correction Plan**: "The response incorrectly provides a numerical answer to an unsolvable problem. It is a logical error to make an arbitrary assumption for a variable that is not given. The correct approach is to state that the problem cannot be solved with the information provided."\n\n'
        "# Input\n"
        "**Question**: {question}\n"
        "**Initial Response**: {initial_response}\n\n"
        "# Output Requirements\n"
        "1. If the response is completely correct, your entire output MUST be an empty string, do not include any other text like 'The response is correct' or else.\n"
        "2. If the response contains any errors, your output MUST be a clear and direct correction plan. Do not include any headers or introductory text like 'Correction Plan:'.\n\n"
    )

    def format(
        self,
        question: str,
        initial_response: str,
    ) -> str:
        prompt = self.TEMPLATE.format(
            question=question,
            initial_response=initial_response,
        )

        return prompt


class CorrectResponsePrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert AI Mathematician specializing in self-correction. Your task is to generate a new, correct solution to a mathematical problem by strictly following a provided `Correction` plan.\n\n"
        "# Core Task\n"
        "You are given a `Question`, your `Initial Response` which contained errors, and a `Correction` plan that explains those errors. Your job is to provide a brand new, fully correct `Final Response` that solves the `Question` by applying the insights from the `Correction`.\n\n"
        "# Guiding Principles\n"
        "1. **Internalize the Correction**: First, carefully read and fully understand the error described in the `Correction` plan.\n"
        "2. **Re-Solve, Don't Patch**: Do not simply edit the incorrect part of the `Initial Response`. Discard your previous attempt and formulate a completely new, correct solution path from the beginning, guided by the `Correction`.\n"
        "3. **Show Your Work**: The final response should clearly and logically show the steps taken to arrive at the correct answer.\n"
        "4. **Address the Core Flaw**: Your new response must directly fix the specific logical or computational flaw mentioned in the `Correction`.\n\n"
        "# Examples (with Reasoning)\n"
        '**(1) Question**: "What is 5 * (10 + 3)?"\n'
        '**Initial Response**: "First, I add 10 and 3 to get 13. Then, I multiply 5 by 13 to get 55. The answer is 55."\n'
        '**Correction**: "The final multiplication step is incorrect. 5 multiplied by 13 is 65, not 55. This calculation needs to be corrected to get the right final answer."\n'
        "**Reasoning**: The correction identifies a specific calculation error in the final step. I will re-solve the problem from the beginning and ensure the multiplication is performed correctly.\n"
        '**Final Response**: "To solve 5 * (10 + 3), the first step is to perform the operation inside the parentheses, which is 10 + 3 = 13. The second step is to multiply this result by 5. 5 * 13 = 65. The final answer is 65."\n\n'
        '**(2) Question**: "John is twice as old as Jane. What is John\'s age?"\n'
        "**Initial Response**: \"Let Jane's age be x. John's age is 2x. Assuming Jane is 10, John is 20. So, John is 20 years old.\"\n"
        '**Correction**: "The response incorrectly provides a numerical answer to an unsolvable problem. It is a logical error to make an arbitrary assumption for a variable that is not given. The correct approach is to state that the problem cannot be solved with the information provided."\n'
        "**Reasoning**: The correction points out a fundamental logical error: I cannot assume a value for Jane's age. I must completely discard my previous numerical answer and instead explain why the problem is unsolvable.\n"
        "**Final Response**: \"The problem cannot be solved because it does not provide enough information. While we can establish a relationship between John's and Jane's ages (John's age = 2 * Jane's age), we cannot determine a specific numerical age for John without knowing Jane's age.\"\n\n"
        "# Output Requirements\n"
        "1. Your entire response MUST be the final, correct, and complete solution to the question.\n"
        "2. Do not include headers, comments, or any meta-commentary. Just provide the answer as if you were solving it for the first time correctly.\n\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "# Input\n"
        "**Question**: {question}\n"
        "**Initial Response**: '''{initial_response}'''\n"
        "**Correction**: '''{correction}'''\n\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        "Final Response:"
    )

    def format(
        self,
        question: str,
        initial_response: str,
        correction: str,
    ) -> str:

        prompt = self.TEMPLATE.format(
            question=question,
            initial_response=initial_response,
            correction=correction,
        )

        return prompt
