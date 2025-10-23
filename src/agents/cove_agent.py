from loguru import logger

from transformers import PreTrainedTokenizer

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import DeepInfra

from src.models import CoVEState
from src.prompts.cove_prompts import (
    ContextualQAPrompt,
    GenerateQuestionsPrompt,
    QuestionAnswerPrompt,
    QuestionAnswerVerificationPrompt,
    RefineResponsePrompt,
)


class CoVEAgent:
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
        graph = StateGraph(CoVEState)

        graph.add_node("generate_initial_response", self.generate_initial_response)
        graph.add_node("generate_questions", self.generate_questions)
        graph.add_node("answer_questions", self.answer_questions)
        graph.add_node("verify_answers", self.verify_answers)
        graph.add_node("generate_new_response", self.generate_new_response)

        graph.set_entry_point("generate_initial_response")
        graph.add_edge("generate_initial_response", "generate_questions")
        graph.add_edge("generate_questions", "answer_questions")
        graph.add_conditional_edges(
            "answer_questions", self.has_answers, {True: "verify_answers", False: END}
        )
        graph.add_conditional_edges(
            "verify_answers",
            self.should_generate_new_response,
            {True: "generate_new_response", False: END},
        )
        graph.add_edge("generate_new_response", END)
        return graph

    def generate_initial_response(self, state: CoVEState) -> CoVEState:
        question = state["initial_question"]
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

        state["initial_response"] = response.strip()
        logger.info(f"Generated initial response: {state['initial_response']}")

        return state

    def generate_questions(self, state: CoVEState) -> CoVEState:
        question = state["initial_question"]
        initial_response = state["initial_response"]

        prompt = GenerateQuestionsPrompt(
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
        state["input_tokens"] += usage_metadata["input_tokens"] if usage_metadata else 0
        state["output_tokens"] += (
            usage_metadata["output_tokens"] if usage_metadata else 0
        )

        response = response.content.strip()
        questions = response.split("\n")
        state["questions"] = questions

        logger.info(f"Generated questions: {state['questions']}")

        return state

    def answer_questions(self, state: CoVEState) -> CoVEState:
        questions = state["questions"]
        context = state["context"]

        prompt = QuestionAnswerPrompt(
            input_variables=[
                "question",
                "context",
            ]
        )

        chain = self.eval_llm

        retrieved_answers = []
        for i, question in enumerate(questions):
            prompt = prompt.format(
                question=question,
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
                retrieved_answers.append(response.content)
            else:
                retrieved_answers.append("")

        state["answers"] = retrieved_answers
        logger.info(f"Retrieved answers: {state['answers']}")
        return state

    def has_answers(self, state: CoVEState) -> bool:
        answers = state["answers"]
        return any(answers)

    def verify_answers(self, state: CoVEState) -> CoVEState:
        initial_response = state["initial_response"]
        questions = state["questions"]
        answers = state["answers"]

        verify_answers_prompt = QuestionAnswerVerificationPrompt(
            input_variables=[
                "initial_response",
                "question",
                "answer",
            ]
        )
        verify_answers_chain = self.eval_llm

        answer_verification_mask = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            logger.info(f"Checking if answer for question: {question} is correct")
            if not answer or len(answer.strip()) < 4:
                answer_verification_mask.append(False)
                logger.info(f"No answer or answer is too short, skipping")
                continue

            verify_answers_prompt_string = verify_answers_prompt.format(
                initial_response=initial_response,
                question=question,
                answer=answer,
            )
            response = verify_answers_chain.invoke(verify_answers_prompt_string)

            usage_metadata = response.usage_metadata
            state["input_tokens"] += (
                usage_metadata["input_tokens"] if usage_metadata else 0
            )
            state["output_tokens"] += (
                usage_metadata["output_tokens"] if usage_metadata else 0
            )

            is_correct = "true" in response.content.lower()
            answer_verification_mask.append(is_correct)

            logger.info(f"Is answer correct: {is_correct}")

        state["answer_verification_mask"] = answer_verification_mask
        logger.info(f"Answer verification mask: {state['answer_verification_mask']}")
        return state

    def should_generate_new_response(self, state: CoVEState) -> bool:
        answer_verification_mask = state["answer_verification_mask"]
        return not all(answer_verification_mask)

    def generate_new_response(self, state: CoVEState) -> CoVEState:
        initial_question = state["initial_question"]
        initial_response = state["initial_response"]
        questions = state["questions"]
        answers = state["answers"]
        verified_answer_mask = state["answer_verification_mask"]

        prompt = RefineResponsePrompt(
            input_variables=[
                "initial_question",
                "initial_response",
                "questions",
                "answers",
                "verified_answer_mask",
            ]
        )

        chain = self.response_llm

        prompt = prompt.format(
            initial_question=initial_question,
            initial_response=initial_response,
            questions=questions,
            answers=answers,
            verified_answer_mask=verified_answer_mask,
        )
        response = chain.invoke(prompt)

        state["input_tokens"] += len(self.tokenizer.encode(prompt))
        state["output_tokens"] += len(self.tokenizer.encode(response.strip()))

        response = response.strip()
        state["corrected_response"] = response

        logger.info(f"Generated corrected response: {state['corrected_response']}")

        return state
