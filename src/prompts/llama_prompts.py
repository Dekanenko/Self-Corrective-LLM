from langchain.prompts import StringPromptTemplate

class MathQAPrompt(StringPromptTemplate):
    TEMPLATE: str = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a meticulous AI mathematician. Your task is to solve the following math problem.\n"
        "Think step by step. Show all your reasoning and calculations. After you have solved the problem, clearly state the final numerical answer at the end of your reasoning.\n"
        "If the question is unanswerable (e.g., it is illogical or missing information), you must clearly state that it cannot be answered and briefly explain why. Do not attempt to solve it."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "{query}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

    def format(self, query: str) -> str:
        return self.TEMPLATE.format(query=query)



class ContextualQAPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a specialized question-answering AI. Your task is to give a concise answer to the question using *only* the provided context. "
        "Make sure to always give an answer."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Context:\n"
        "'''\n"
        "{context}\n"
        "'''\n\n"
        "Question: {query}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )


    def format(self, query: str, context: str) -> str:
        return self.TEMPLATE.format(query=query, context=context)
