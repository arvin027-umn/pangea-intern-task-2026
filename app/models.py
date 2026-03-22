from pydantic import BaseModel, Field
from typing import Literal

"""Pydantic models for request and response validation
- Uses Literal types to constrain error_type and difficulty to allowed
values in the JSON schema.
- Catches invalid values at Python level before they reach the caller
"""

# Allowed error type strings
ErrorType = Literal[
    "grammar",
    "spelling",
    "word_choice",
    "punctuation",
    "word_order",
    "missing_word",
    "extra_word",
    "conjugation",
    "gender_agreement",
    "number_agreement",
    "tone_register",
    "other",
]

# CEFR difficulty levels
DifficultyLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]

class ErrorDetail(BaseModel):
    """Error found in learner's sentence"""

    original: str = Field(
        description="The exact word or phrase from the original sentence that is wrong"
    )
    correction: str = Field(
        description="The corrected word or phrase that should replace the original"
    )
    error_type: ErrorType = Field(
        description="Category of the error, must be one of the allowed types"
    )
    explanation: str = Field(
        description="A short, friendly explanation written in the learner's native language"
    )

class FeedbackRequest(BaseModel):
    """Incoming request body for /feedback endpoint"""

    sentence: str = Field(
        min_length=1,
        description="The learner's sentence in the language they want to learn",
    )
    target_language: str = Field(
        min_length=2,
        description="The language the learner is studying",
    )
    native_language: str = Field(
        min_length=2,
        description="The learner's language they are using for explanations",
    )

class FeedbackResponse(BaseModel):
    """Response body returned by /feedback endpoint."""

    corrected_sentence: str = Field(
        description="The corrected version of the input sentence"
    )
    is_correct: bool = Field(
        description="True if the original sentence had no errors"
    )
    errors: list[ErrorDetail] = Field(
        default_factory=list,
        description="List of errors found; empty if sentence is correct",
    )
    difficulty: DifficultyLevel = Field(
        description="CEFR difficulty level of the sentence: A1 through C2"
    )