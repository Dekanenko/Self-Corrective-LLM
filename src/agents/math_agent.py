from loguru import logger

from transformers import PreTrainedTokenizer

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import DeepInfra

from src.models import MathAgentState
from src.prompts.math_agent_prompts import (
    CheckResponsePrompt,
    CorrectResponsePrompt,
)


class MathAgent:
    def __init__(
        self,
        response_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        eval_model_name: str = "gemini-2.0-flash",
        temperature: float = 0.5,
        tokenizer: PreTrainedTokenizer = None,
    ):
        self.tokenizer = tokenizer
        self.response_llm = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.response_llm.model_kwargs = {
            "temperature": temperature,
            "max_new_tokens": 512,
        }
        self.eval_llm = ChatGoogleGenerativeAI(
            model=eval_model_name,
            temperature=temperature,
        )
        self.model = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(MathAgentState)

        graph.add_node("check_response", self.check_response)
        graph.add_node("correct_response", self.correct_response)

        graph.set_entry_point("check_response")
        graph.add_conditional_edges(
            "check_response",
            self.should_correct,
            {True: "correct_response", False: END},
        )
        graph.add_edge("correct_response", END)
        return graph

    def check_response(self, state: MathAgentState) -> MathAgentState:
        question = state["question"]
        initial_response = state["initial_response"]

        prompt = CheckResponsePrompt(
            input_variables=[
                "question",
                "initial_response",
            ]
        )

        chain = self.eval_llm

        prompt = prompt.format(
            question=question,
            initial_response=initial_response,
        )
        response = chain.invoke(prompt)

        usage_metadata = response.usage_metadata
        state["input_tokens"] = usage_metadata["input_tokens"] if usage_metadata else 0
        state["output_tokens"] = (
            usage_metadata["output_tokens"] if usage_metadata else 0
        )

        response = response.content.strip()
        state["correction"] = response

        logger.info(f"Generated correction: {state['correction']}")

        return state

    def should_correct(self, state: MathAgentState) -> bool:
        correction = state["correction"]
        return len(correction.strip()) > 15

    def correct_response(self, state: MathAgentState) -> MathAgentState:
        question = state["question"]
        initial_response = state["initial_response"]
        correction = state["correction"]

        prompt = CorrectResponsePrompt(
            input_variables=[
                "question",
                "initial_response",
                "correction",
            ]
        )

        chain = self.response_llm
        prompt = prompt.format(
            question=question,
            initial_response=initial_response,
            correction=correction,
        )
        response = chain.invoke(prompt)

        state["input_tokens"] += len(self.tokenizer.encode(prompt))
        state["output_tokens"] += len(self.tokenizer.encode(response.strip()))

        state["corrected_response"] = response.strip()
        logger.info(f"Corrected response: {state['corrected_response']}")
        return state
