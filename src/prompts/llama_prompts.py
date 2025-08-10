from langchain.prompts import StringPromptTemplate

class MathQAPrompt(StringPromptTemplate):
    TEMPLATE: str = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a meticulous AI mathematician. Your task is to solve the following math problem.\n\n"
        "Follow these steps carefully:\n"
        "1. **Analyze the problem:** First, understand the given information and what is being asked.\n"
        "2. **Assess solvability:** Determine if the problem is solvable. A problem might be unsolvable if it's illogical, contains contradictions, or lacks necessary information.\n"
        "3. **Solve or Explain:**\n"
        "   - **If solvable:** Provide a step-by-step solution, showing all your reasoning and calculations, and then clearly state the final numerical answer.\n"
        "   - **If unsolvable:** State that the problem cannot be answered and provide a concise explanation.\n\n"
        "Your entire response should only contain the solution and final answer (or the explanation for unsolvable problems). Do not add any conversational headers or extraneous text."
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
