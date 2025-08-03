from loguru import logger

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from src.models import ErrorList, ContextQACorrectionState
from src.prompts.gemini_agent_prompts import (
    ContextQAErrorCheckPrompt, 
    ContextQAErrorCorrectionPrompt, 
    ContextQAResponseVerificationPrompt
)
from src.utils.formatting import ensure_space_after_del_tokens

class ContextQACorrectionAgent:
    def __init__(
            self, 
            error_check_model_name: str = "gemini-2.0-flash", 
            correction_model_name: str = "gemini-2.0-flash", 
            error_check_temperature: float = 0.2, 
            correction_temperature: float = 0.3, 
            error_detection_only: bool = False,
            max_responses_to_correct: int = 3,
    ):
        self.error_check_llm = ChatGoogleGenerativeAI(model=error_check_model_name, temperature=error_check_temperature)
        self.correction_llm = ChatGoogleGenerativeAI(model=correction_model_name, temperature=correction_temperature)
        self.error_detection_only = error_detection_only
        self.max_responses_to_correct = max_responses_to_correct

        self.model = self._build_graph().compile()
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ContextQACorrectionState)

        graph.add_node("check_for_errors", self.check_for_errors)
        graph.add_node("get_errors_to_correct", self.get_errors_to_correct)
        graph.add_node("correct_response", self.correct_response)
        graph.add_node("verify_corrected_response", self.verify_corrected_response)
        graph.add_node("filter_verified_responses", self.filter_verified_responses)

        graph.set_entry_point("check_for_errors")
        graph.add_edge("get_errors_to_correct", "correct_response")
        graph.add_edge("correct_response", "verify_corrected_response")
        graph.add_edge("verify_corrected_response", "filter_verified_responses")
        graph.add_edge("filter_verified_responses", END)

        graph.add_conditional_edges("check_for_errors", self.should_correct, {True: "get_errors_to_correct", False: END})
        return graph

    
    def check_for_errors(self, state: ContextQACorrectionState) -> ContextQACorrectionState:
        question = state["question"]
        context = state["context"]
        answer = state["answer"]
        response_batch = state["responses"]

        parser = PydanticOutputParser(pydantic_object=ErrorList)
        prompt = ContextQAErrorCheckPrompt(input_variables=[
            "question", "context", "answer", 
            "response", "format_instructions",
        ])

        # batch inputs
        prompts = [
            prompt.format(
                question=question, 
                context=context, 
                answer=answer, 
                response=res, 
                format_instructions=parser.get_format_instructions(),
            )
            for res in response_batch
        ]
  
        chain = self.error_check_llm | parser
        result = chain.batch(prompts, return_exceptions=True)
        
        errors = []
        wrong_response_number = 0
        responses = []
        for i in range(len(result)):
            if not isinstance(result[i], Exception):
                error_list = [err.dict() for err in result[i].errors]
                errors.append(error_list)
                responses.append(response_batch[i])
                wrong_response_number += 1 if result[i].errors else 0

        logger.info(f"Errors: {errors}")
        logger.info(f"Wrong responses: {wrong_response_number}")

        state["errors"] = errors
        state["wrong_response_number"] = wrong_response_number
        state["responses"] = responses
        return state
    
    def should_correct(self, state: ContextQACorrectionState) -> bool:
        if self.error_detection_only:
            return False
    
        # return bool(self.error_number) and self.error_number < len(errors)
        return bool(state["wrong_response_number"])
    
    def get_errors_to_correct(self, state: ContextQACorrectionState) -> ContextQACorrectionState:
        errors = state["errors"]
        wrong_response_number = state["wrong_response_number"]
        responses = state["responses"]
        num_responses_to_correct = min(wrong_response_number, self.max_responses_to_correct)

        zipped_lists = zip(errors, responses)
        sorted_pairs = sorted(zipped_lists, key=lambda pair: len(pair[0]), reverse=True)
        truncated_pairs = sorted_pairs[:num_responses_to_correct]
        errors, responses = zip(*truncated_pairs)

        state["errors_to_correct"] = errors
        state["responses_to_correct"] = responses

        logger.info(f"Errors to correct: {errors}")
        logger.info(f"Responses to correct: {responses}")

        return state
    
    def correct_response(self, state: ContextQACorrectionState) -> ContextQACorrectionState:
        question = state["question"]
        context = state["context"]
        answer = state["answer"]
        response_batch = state["responses_to_correct"]
        errors = state["errors_to_correct"]

        prompt = ContextQAErrorCorrectionPrompt(input_variables=[
            "question", "context", "answer", 
            "response", "errors",
        ])

        # batch inputs
        prompts = [
            prompt.format(
                question=question, 
                context=context, 
                answer=answer, 
                response=res, 
                errors=err,
            )
            for res, err in zip(response_batch, errors)
        ]
  
        chain = self.correction_llm | StrOutputParser()

        try:
            corrected_responses = chain.batch(prompts)
        except Exception as e:
            logger.error(f"Error in ContextQACorrectionAgent: {e}")
        
        logger.info(f"Corrected responses: {corrected_responses}")
        # corrected_responses = [res.replace("[DELETE_WORD]", "<DEL_W>").replace("[DELETE_SENTENCE]", "<DEL_S>").replace("[DELETE_ALL]", "<DEL_A>") for res in corrected_responses]
        state["corrected_responses"] = [ensure_space_after_del_tokens(res) for res in corrected_responses]
        return state

    def verify_corrected_response(self, state: ContextQACorrectionState) -> ContextQACorrectionState:
        question = state["question"]
        context = state["context"]
        answer = state["answer"]
        corrected_responses = state["corrected_responses"]

        prompt = ContextQAResponseVerificationPrompt(input_variables=[
            "question", "context", "answer", "corrected_response"
        ])

        # batch inputs
        prompts = [
            prompt.format(
                question=question, 
                context=context, 
                answer=answer, 
                corrected_response=res
            )
            for res in corrected_responses
        ]
  
        chain = self.correction_llm | StrOutputParser()

        try:
            verified_response_mask = chain.batch(prompts)
        except Exception as e:
            logger.error(f"Error in ContextQACorrectionAgent: {e}")

        logger.info(f"Verified responses: {verified_response_mask}")

        state["verified_response_mask"] = ['true' in response.lower() for response in verified_response_mask]
        return state

    def filter_verified_responses(self, state: ContextQACorrectionState) -> ContextQACorrectionState:
        """Filters responses and errors, keeping only those marked as verified."""
        responses = state["corrected_responses"]
        mask = state["verified_response_mask"]
        
        state["corrected_responses"] = [
            response for response, verified in zip(responses, mask) if verified
        ]
        return state