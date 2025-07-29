from typing import TypedDict
from pydantic import BaseModel, Field


class Error(BaseModel):
    """
    A model to represent an error found in a response.
    """
    error_description: str = Field(description="The error found in the response.")
    error_location: str = Field(description="The location of the error in the response.")
    error_correction: str = Field(description="Explanation on how to correct the error.")


class ErrorList(BaseModel):
    """
    A model to hold a list of errors found in a response.
    """
    errors: list[Error] = Field(default=[], description="A list of errors found in the response.")

class ContextQACorrectionState(TypedDict):
    question: str = Field(default="")
    context: str = Field(default="")
    answer: str = Field(default="")
    responses: list[str] = Field(default=[])
    errors: list[list[str]] = Field(default=[])
    corrected_responses: list[str] = Field(default=[])
    verified_response_mask: list[bool] = Field(default=[])


class MathQACorrectionState(TypedDict):
    question: str = Field(default="")
    answer: str = Field(default="")
    is_answerable: bool = Field(default=False)
    responses: list[str] = Field(default=[])
    errors: list[list[str]] = Field(default=[])
    corrected_responses: list[str] = Field(default=[])
    verified_response_mask: list[bool] = Field(default=[])
