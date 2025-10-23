from loguru import logger

from transformers import PreTrainedTokenizer

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import DeepInfra

from src.models import RAPRState
from src.prompts.rapr_prompts import (
    ContextualQAPrompt,
    GenerateQueriesPrompt,
    RetriveEvidencePrompt,
    AgreementPrompt,
    RefineResponsePrompt,
)


class RAPRAgent:
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
        graph = StateGraph(RAPRState)

        graph.add_node("generate_model_response", self.generate_model_response)
        graph.add_node("generate_queries", self.generate_queries)
        graph.add_node("retrieve_evidence", self.retrieve_evidence)
        graph.add_node("revise", self.revise)

        graph.set_entry_point("generate_model_response")
        graph.add_edge("generate_model_response", "generate_queries")
        graph.add_edge("generate_queries", "retrieve_evidence")
        graph.add_conditional_edges(
            "retrieve_evidence", self.has_evidence, {True: "revise", False: END}
        )
        graph.add_edge("revise", END)
        return graph

    def generate_model_response(self, state: RAPRState) -> RAPRState:
        question = state["question"]
        context = state["context"]

        prompt = ContextualQAPrompt(
            input_variables=[
                "query",
                "context",
            ]
        )

        chain = self.response_llm

        prompt = prompt.format(
            query=question,
            context=context,
        )
        response = chain.invoke(prompt)

        state["input_tokens"] = len(self.tokenizer.encode(prompt))
        state["output_tokens"] = len(self.tokenizer.encode(response.strip()))

        state["model_response"] = response.strip()
        logger.info(f"Generated model response: {state['model_response']}")

        return state

    def generate_queries(self, state: RAPRState) -> RAPRState:
        question = state["question"]
        model_response = state["model_response"]

        prompt = GenerateQueriesPrompt(
            input_variables=[
                "question",
                "statement",
            ]
        )

        chain = self.eval_llm

        prompt = prompt.format(
            question=question,
            statement=model_response,
        )
        response = chain.invoke(prompt)
        usage_metadata = response.usage_metadata
        state["input_tokens"] += usage_metadata["input_tokens"] if usage_metadata else 0
        state["output_tokens"] += (
            usage_metadata["output_tokens"] if usage_metadata else 0
        )

        response = response.content.strip()
        state["queries"] = response.split("\n")
        state["queries"].append(question)

        logger.info(f"Generated queries: {state['queries']}")

        return state

    def retrieve_evidence(self, state: RAPRState) -> RAPRState:
        queries = state["queries"]
        context = state["context"]

        prompt = RetriveEvidencePrompt(
            input_variables=[
                "query",
                "context",
            ]
        )

        chain = self.eval_llm

        retrieved_evidence = []
        for i, query in enumerate(queries):
            prompt = prompt.format(
                query=query,
                context=context,
            )
            response = chain.invoke(prompt)

            usage_metadata = response.usage_metadata
            state["input_tokens"] += (
                usage_metadata["input_tokens"] if usage_metadata else 0
            )
            state["output_tokens"] += (
                usage_metadata["output_tokens"] if usage_metadata else 0
            )

            if response.content:
                retrieved_evidence.append(response.content)
            else:
                retrieved_evidence.append("")

        state["evidence"] = retrieved_evidence
        logger.info(f"Retrieved evidence: {state['evidence']}")
        return state

    def has_evidence(self, state: RAPRState) -> bool:
        evidence = state["evidence"]
        return any(evidence)

    def revise(self, state: RAPRState) -> RAPRState:
        corrected_response = state["model_response"]
        evidence = state["evidence"]
        queries = state["queries"]

        agreement_prompt = AgreementPrompt(
            input_variables=[
                "statement",
                "evidence",
                "query",
            ]
        )
        agreement_chain = self.eval_llm

        for i, (query, evidence_item) in enumerate(zip(queries, evidence)):
            logger.info(
                f"Checking agreement for query: {query} and evidence: {evidence_item}"
            )
            if not evidence_item or len(evidence_item.strip()) < 4:
                logger.info(f"No evidence or evidence is too short, skipping")
                continue

            agreement_prompt_string = agreement_prompt.format(
                statement=corrected_response,
                evidence=evidence_item,
                query=query,
            )
            response = agreement_chain.invoke(agreement_prompt_string)

            usage_metadata = response.usage_metadata
            state["input_tokens"] += (
                usage_metadata["input_tokens"] if usage_metadata else 0
            )
            state["output_tokens"] += (
                usage_metadata["output_tokens"] if usage_metadata else 0
            )

            should_edit = "false" in response.content.lower()

            logger.info(f"Should edit: {should_edit}")

            if should_edit:
                edit_prompt = RefineResponsePrompt(
                    input_variables=[
                        "statement",
                        "evidence",
                        "query",
                    ]
                )
                edit_chain = self.eval_llm

                edit_prompt_string = edit_prompt.format(
                    statement=corrected_response,
                    evidence=evidence_item,
                    query=query,
                )
                response = edit_chain.invoke(edit_prompt_string)

                usage_metadata = response.usage_metadata
                state["input_tokens"] += (
                    usage_metadata["input_tokens"] if usage_metadata else 0
                )
                state["output_tokens"] += (
                    usage_metadata["output_tokens"] if usage_metadata else 0
                )

                corrected_response = response.content

                logger.info(f"Edited response: {corrected_response}")

        state["corrected_response"] = corrected_response
        return state
