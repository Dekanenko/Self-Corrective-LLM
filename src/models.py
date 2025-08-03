from typing import TypedDict
from pydantic import BaseModel, Field


class Error(BaseModel):
    """
    A model to represent an error found in a response.
    """
    description: str = Field(description="The error found in the response.")
    location: str = Field(description="The location of the error in the response.")
    correction: str = Field(description="Explanation on how to correct the error.")


class ErrorList(BaseModel):
    """
    A model to hold a list of errors found in a response.
    """
    errors: list[Error] = Field(default=[], description="A list of errors found in the response.")

class ContextQACorrectionState(TypedDict):
    input: str = Field(default="")
    question: str = Field(default="")
    context: str = Field(default="")
    answer: str = Field(default="")
    responses: list[str] = Field(default=[])
    errors: list[list[str]] = Field(default=[])
    wrong_response_number: int = Field(default=0)
    responses_to_correct: list[str] = Field(default=[])
    errors_to_correct: list[list[str]] = Field(default=[])
    corrected_responses: list[str] = Field(default=[])
    verified_response_mask: list[bool] = Field(default=[])


class MathQACorrectionState(TypedDict):
    input: str = Field(default="")
    question: str = Field(default="")
    answer: str = Field(default="")
    is_answerable: bool = Field(default=False)
    responses: list[str] = Field(default=[])
    errors: list[list[str]] = Field(default=[])
    wrong_response_number: int = Field(default=0)
    responses_to_correct: list[str] = Field(default=[])
    errors_to_correct: list[list[str]] = Field(default=[])
    corrected_responses: list[str] = Field(default=[])
    verified_response_mask: list[bool] = Field(default=[])
