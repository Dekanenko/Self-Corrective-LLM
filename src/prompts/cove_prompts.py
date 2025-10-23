from langchain.prompts import StringPromptTemplate


class ContextualQAPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a meticulous and factual AI assistant designed for fact-based question answering. Your primary directive is to answer the user's question with unwavering accuracy, using *only* the information available in the provided `Context`.\n\n"
        "You must not, under any circumstances, use external knowledge or make assumptions beyond what is explicitly stated in the text.\n"
        "Your answer should be fully comprehensive but concise, directly addressing the question by synthesizing the relevant information from the context."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "**Context**:\n"
        "'''\n"
        "{context}\n"
        "'''\n\n"
        "**Question**: {query}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )


    def format(self, query: str, context: str) -> str:
        return self.TEMPLATE.format(query=query, context=context)


class GenerateQuestionsPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are an expert Verification Question Generator. Your primary goal is to analyze a response to a question and generate a set of specific, fact-checking questions to verify the claims made in the response.\n\n"
        
        "# Core Task\n"
        "Given an `Initial Question` and a `Baseline Response`, generate a list of precise verification questions. Each question should target a single, verifiable factual claim within the response. These questions are for a fact-checker to use to determine if the response is accurate.\n\n"

        "# Principles of Verification Question Generation\n"
        "1. **Isolate Factual Claims**: Identify all distinct, verifiable facts in the response (e.g., dates, names, locations, numbers, events, definitions).\n"
        "2. **Formulate Direct Questions**: Convert each factual claim into a clear, direct question. The questions should be answerable with a specific piece of information.\n"
        "3. **Comprehensiveness**: Aim to generate questions that cover all the major factual assertions in the response.\n"
        "4. **Neutrality**: Phrase questions in an objective, non-leading manner.\n\n"

        "# Examples\n"
        "**(1) Initial Question**: \"Tell me about the Mexican-American War.\"\n"
        "**Baseline Response**: \"The Mexican–American War was an armed conflict between the United States and Mexico from 1846 to 1848.\"\n"
        "**Verification Questions**:\n"
        "When did the Mexican-American War start and end?\n\n"

        "**(2) Initial Question**: \"Who wrote the Little House books?\"\n"
        "**Baseline Response**: \"The Little House books were written by Laura Ingalls Wilder. The series was published by HarperCollins and the first book came out in 1932.\"\n"
        "**Verification Questions**:\n"
        "Who wrote the Little House book series?\n"
        "Which company published the Little House book series?\n"
        "In what year was the first Little House book published?\n\n"

        "**(3) Initial Question**: \"What is the capital of Australia?\"\n"
        "**Baseline Response**: \"The capital of Australia is Sydney, which is also its largest city. It was established as the capital in 1908 following the federation of the colonies.\"\n"
        "**Verification Questions**:\n"
        "What is the capital city of Australia?\n"
        "When was the capital of Australia established?\n\n"

        "# Input\n"
        "**Initial Question**: {question}\n"
        "**Baseline Response**: {initial_response}\n\n"

        "# Output Requirements\n"
        "1. You MUST produce one or more unique verification questions.\n"
        "2. Each question MUST be on a new line.\n"
        "3. Your entire response MUST consist ONLY of the questions. Do not include headers, comments, explanations, or any list formatting (like bullet points or numbers).\n\n"
        
        "Verification Questions:"
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
        
        # print(prompt)

        return prompt


class QuestionAnswerPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are an expert Answering AI. Your purpose is to provide the most accurate, comprehensive, and well-written answer to a given question.\n\n"
        
        "# Core Task\n"
        "You will be given a `Question` and a `Context`. Your primary goal is to answer the `Question` based on the information within the `Context`. If the `Context` does not contain the necessary information, you should use your own extensive knowledge to provide the best possible answer.\n\n"

        "# Guiding Principles\n"
        "1. **Prioritize Context**: Always attempt to formulate the answer using the provided `Context` first. Your answer should reflect the information given in the text.\n"
        "2. **Synthesize, Don't Just Extract**: Do not simply copy-paste from the context. Read, understand, and then generate a complete, well-structured answer in your own words.\n"
        "3. **Use General Knowledge When Necessary**: If the context is insufficient or does not contain the answer, you are authorized to use your general knowledge. Do not mention that the context was insufficient; just provide the answer directly.\n"
        "4. **Be Comprehensive and Direct**: Ensure your answer is complete and directly addresses the user's question without unnecessary filler.\n\n"

        "# Examples\n"
        "**(1) Question**: \"When did the Mexican-American War start and end?\"\n"
        "**Context**: \"The conflict between the United States and Mexico, known as the Mexican-American War, was fought from 1846 until 1848.\"\n"
        "**Answer**: \"The Mexican-American War started in 1846 and ended in 1848.\"\n\n"

        "**(2) Question**: \"Who wrote the Little House book series?\"\n"
        "**Context**: \"The series was published by HarperCollins and the first book came out in 1932. These stories captured the hearts of many young readers.\"\n"
        "**Answer**: \"The Little House book series was written by Laura Ingalls Wilder.\"\n\n"

        "# Output Requirements\n"
        "1. Your entire response MUST be the answer to the question.\n"
        "2. Do not include headers, comments, or explanations about where you found the information (e.g., 'Based on the context...' or 'From my knowledge...').\n\n"

        "# Input\n"
        "**Question**: {question}\n"
        "**Context**: '''{context}'''\n\n"
    
        "Answer:"
    )

    def format(
            self,
            question: str,
            context: str,
    ) -> str:
        prompt = self.TEMPLATE.format(
            question=question, 
            context=context,
        )
        
        # print(prompt)

        return prompt
    

class QuestionAnswerVerificationPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are a meticulous AI logician. Your sole task is to determine if an `Initial Response` and a new `Answer` are in agreement with respect to a given `Question`.\n\n"
        
        "# Core Task\n"
        "Compare the answer to the `Question` as implied by the `Initial Response` against the information provided in the new `Answer`. You must determine if they are semantically equivalent and factually consistent. Respond with `True` if they agree and `False` if they provide different or contradictory information.\n\n"

        "# Guiding Principles\n"
        "1. **Question-Focused Comparison**: Your judgment must be based *only* on the information that directly answers the `Question`. Ignore any extraneous details in either the `Initial Response` or the `Answer`.\n"
        "2. **Semantic Agreement**: The answers agree if they mean the same thing, even if they are worded differently.\n"
        "3. **Factual Disagreement**: The answers disagree if they contain a factual contradiction.\n\n"

        "# Examples (with Reasoning)\n"
        "**(1) Initial Response**: \"Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes.\"\n"
        "**Question**: \"How often do your nostrils switch?\"\n"
        "**Answer**: \"...the congestion pattern switches about every 2 hours, according to a small 2016 study...\"\n"
        "**Reasoning**: The initial response claims the switch time is 'about every 45 minutes'. The new answer states the switch time is 'about every 2 hours'. 45 minutes and 2 hours are factually different durations. Therefore, they disagree.\n"
        "**Agreement**: False\n\n"

        "**(2) Initial Response**: \"The Little House books were written by Laura Ingalls Wilder. The books were published by HarperCollins.\"\n"
        "**Question**: \"Who published the Little House books?\"\n"
        "**Answer**: \"Written by Laura Ingalls Wilder and published by HarperCollins, these beloved books remain a favorite to this day.\"\n"
        "**Reasoning**: The initial response identifies the publisher as 'HarperCollins'. The new answer also explicitly says 'published by HarperCollins'. The facts are identical. Therefore, they agree.\n"
        "**Agreement**: True\n\n"

        "**(3) Initial Response**: \"Social work is a profession that is based in the philosophical tradition of humanism. It is an intellectual discipline that has its roots in the 1800s.\"\n"
        "**Question**: \"When did social work have its roots?\"\n"
        "**Answer**: \"Social work’s roots were planted in the 1880s, when charity organization societies (COS) were created...\"\n"
        "**Reasoning**: The initial response claims the roots are in the '1800s'. The new answer specifies the '1880s'. This is a more specific claim that contradicts the broader one. For fact-checking, this is a disagreement.\n"
        "**Agreement**: False\n\n"

        "# Input\n"
        "**Initial Response**: {initial_response}\n"
        "**Question**: {question}\n"
        "**Answer**: '''{answer}'''\n\n"

        "# Output Requirements\n"
        "1. Your entire response MUST be a single word: `True` or `False`.\n"
        "2. Do not include headers, comments, reasoning, or any other text in your final output.\n\n"

        "Agreement:"
    )

    def format(
            self,
            initial_response: str,
            question: str,
            answer: str,
    ) -> str:
        prompt = self.TEMPLATE.format(
            initial_response=initial_response, 
            question=question, 
            answer=answer,
        )
        
        # print(prompt)

        return prompt


class RefineResponsePrompt(StringPromptTemplate):


    TEMPLATE: str = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert AI Response Synthesizer. Your task is to rewrite an `Initial Response` based on a list of `Verification Results`. The goal is to produce a final, corrected, and coherent answer that resolves all disagreements while preserving all verified agreements.\n\n"
        
        "# Core Task\n"
        "Carefully analyze the `Initial Response` and the accompanying `Verification Results`. Each result contains a question, a new answer, and an agreement status (`True` or `False`). Rewrite the response to create a `Final Response` that incorporates the correct information from the new answers where `Agreement` is `False`, and remains consistent with the information where `Agreement` is `True`.\n\n"

        "# Guiding Principles\n"
        "1. **Analyze Verification Results**: Systematically review each verification item. \n"
        "2. **Correct Disagreements**: For every item where `Agreement` is `False`, the `Answer` provided is the new source of truth. The corresponding part of the `Initial Response` must be corrected to reflect this new information.\n"
        "3. **Preserve Agreements**: For every item where `Agreement` is `True`, the corresponding part of the `Initial Response` is factually correct and should be preserved in the final output.\n"
        "4. **Synthesize Holistically**: Weave all the corrected and preserved facts into a single, fluent, and comprehensive `Final Response`. The output should be a well-written answer to the `Initial Question`, not a list of changes.\n\n"

        "# Example (with Reasoning)\n"
        "**(1) Initial Question**: \"Tell me about the capital of Australia.\"\n"
        "**Initial Response**: \"The capital of Australia is Sydney, which is also its largest city. It was established as the capital in 1908 following the federation of the colonies.\"\n"
        "**Verification Results**:\n"
        "Verification 1:\n"
        "Question: \"What is the capital city of Australia?\"\n"
        "Answer: \"The capital of Australia is Canberra.\"\n"
        "Agreement: False\n\n"
        "Verification 2:\n"
        "Question: \"What is the largest city in Australia?\"\n"
        "Answer: \"Sydney is the largest city in Australia.\"\n"
        "Agreement: True\n\n"
        "Verification 3:\n"
        "Question: \"When was the capital of Australia established?\"\n"
        "Answer: \"The site for Canberra was chosen in 1908, but construction of the city did not begin until 1913.\"\n"
        "Agreement: False\n\n"
        "**Reasoning**: I have analyzed the verification results. Verification 1 shows the capital is 'Canberra', not 'Sydney', so I must correct this. Verification 2 confirms that the initial response was correct about Sydney being the largest city, so I will keep that fact. Verification 3 shows the 1908 date was incomplete; I need to replace it with the more detailed information from the answer. My final response will combine these changes into a coherent paragraph.\n"
        "**Final Response**: \"The capital of Australia is Canberra. While Sydney is the country's largest city, the site for Canberra was chosen as the capital in 1908, and construction began in 1913.\"\n\n"

        "# Output Requirements\n"
        "1. Your entire response MUST be ONLY the final, refined response string.\n"
        "2. Do not include headers, comments, reasoning, or any other text in your final output.\n\n"
        
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"

        "# Input\n"
        "**Initial Question**: {initial_question}\n"
        "**Initial Response**: {initial_response}\n"
        "**Verification Results**: '''\n{verification_results}'''\n\n"

        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        "Final Response:"
    )

    def format(
            self,
            initial_question: str,
            initial_response: str,
            questions: list[str],
            answers: list[str],
            verified_answer_mask: list[bool],
    ) -> str:

        verification_results = ""
        for i in range(len(questions)):
            verification_results += f"\nVerification {i + 1}:\n"
            verification_results += f"Question: {questions[i]}\n"
            verification_results += f"Answer: {answers[i]}\n"
            verification_results += f"Agreement: {'True' if verified_answer_mask[i] else 'False'}\n"

        prompt = self.TEMPLATE.format(
            initial_question=initial_question, 
            initial_response=initial_response, 
            verification_results=verification_results,
        )
        
        # print(prompt)

        return prompt