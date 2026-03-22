import json
from pathlib import Path

import jsonschema
import pytest
from app.models import FeedbackRequest, FeedbackResponse, ErrorDetail

SCHEMA_DIR = Path(__file__).parent.parent / "schema"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

"""Schema validation tests -- verify models match JSON schemas

Verify that:
- Valid requests pass schema validation
- Invalid requests are rejected
- All example inputs/outputs from sample_inputs.json are comply with the schema
- Response schema enforces allowed error types and difficulty levels
"""

def load_schema(name: str) -> dict:
    """Load a JSON schema file from the schema directory"""
    return json.loads((SCHEMA_DIR / name).read_text())


def load_examples() -> list[dict]:
    """Load the sample inputs from the examples directory"""
    return json.loads((EXAMPLES_DIR / "sample_inputs.json").read_text())

class TestRequestSchema:
    """Tests for the request JSON schema"""

    def test_valid_request(self):
        schema = load_schema("request.schema.json")
        valid = {
            "sentence": "Hola mundo",
            "target_language": "Spanish",
            "native_language": "English",
        }
        jsonschema.validate(valid, schema)

    def test_missing_sentence_fails(self):
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {"target_language": "Spanish", "native_language": "English"},
                schema,
            )

    def test_empty_sentence_fails(self):
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "sentence": "",
                    "target_language": "Spanish",
                    "native_language": "English",
                },
                schema,
            )

    def test_extra_fields_rejected(self):
        """The request schema has additionalProperties=false so extra fields should cause validation to fail"""
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "sentence": "Hola",
                    "target_language": "Spanish",
                    "native_language": "English",
                    "extra_field": "not allowed",
                },
                schema,
            )

class TestResponseSchema:
    """Tests for the response JSON schema"""

    def test_correct_response(self):
        schema = load_schema("response.schema.json")
        valid = {
            "corrected_sentence": "Hola mundo",
            "is_correct": True,
            "errors": [],
            "difficulty": "A1",
        }
        jsonschema.validate(valid, schema)

    def test_response_with_errors(self):
        schema = load_schema("response.schema.json")
        valid = {
            "corrected_sentence": "Le chat noir",
            "is_correct": False,
            "errors": [
                {
                    "original": "La chat",
                    "correction": "Le chat",
                    "error_type": "gender_agreement",
                    "explanation": "Chat is masculine",
                }
            ],
            "difficulty": "A1",
        }
        jsonschema.validate(valid, schema)

    def test_invalid_difficulty_fails(self):
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "test",
                    "is_correct": True,
                    "errors": [],
                    "difficulty": "Z9",
                },
                schema,
            )

    def test_invalid_error_type_fails(self):
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "test",
                    "is_correct": False,
                    "errors": [
                        {
                            "original": "x",
                            "correction": "y",
                            "error_type": "not_a_real_type",
                            "explanation": "test",
                        }
                    ],
                    "difficulty": "A1",
                },
                schema,
            )

    def test_missing_error_fields_fails(self):
        """Each error object must have all four required fields """
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "test",
                    "is_correct": False,
                    "errors": [
                        {
                            "original": "x",
                            # missing correction, error_type, explanation
                        }
                    ],
                    "difficulty": "A1",
                },
                schema,
            )

    def test_all_valid_error_types_accepted(self):
        """Every allowed error type should pass validation"""
        schema = load_schema("response.schema.json")
        from app.config import validation_error_types

        for error_type in validation_error_types:
            valid = {
                "corrected_sentence": "test",
                "is_correct": False,
                "errors": [
                    {
                        "original": "x",
                        "correction": "y",
                        "error_type": error_type,
                        "explanation": "test",
                    }
                ],
                "difficulty": "A1",
            }
            jsonschema.validate(valid, schema)

    def test_all_valid_difficulty_levels_accepted(self):
        """Every CEFR level should pass validation"""
        schema = load_schema("response.schema.json")
        from app.config import cefr_difficulty_levels

        for level in cefr_difficulty_levels:
            valid = {
                "corrected_sentence": "test",
                "is_correct": True,
                "errors": [],
                "difficulty": level,
            }
            jsonschema.validate(valid, schema)

class TestExamplesMatchSchemas:
    """Verify that all example inputs/outputs conform to the schemas"""

    def test_all_example_requests_valid(self):
        schema = load_schema("request.schema.json")
        for example in load_examples():
            jsonschema.validate(example["request"], schema)

    def test_all_example_responses_valid(self):
        schema = load_schema("response.schema.json")
        for example in load_examples():
            jsonschema.validate(example["expected_response"], schema)

class TestPydanticModels:
    """Verify Pydantic models enforce the same constraints as the schemas"""

    def test_invalid_error_type_rejected_by_pydantic(self):
        """Pydantic should reject error types not in the union of literals defined in the model"""
        with pytest.raises(Exception):
            ErrorDetail(
                original="x",
                correction="y",
                error_type="made_up_type",
                explanation="test",
            )

    def test_invalid_difficulty_rejected_by_pydantic(self):
        """Pydantic should reject difficulty levels not in the union of literals defined in the model"""
        with pytest.raises(Exception):
            FeedbackResponse(
                corrected_sentence="test",
                is_correct=True,
                errors=[],
                difficulty="Z9",
            )

    def test_valid_response_accepted_by_pydantic(self):
        """A well-formed response should be accepted by Pydantic without errors"""
        response = FeedbackResponse(
            corrected_sentence="Hola mundo.",
            is_correct=True,
            errors=[],
            difficulty="A1",
        )
        assert response.difficulty == "A1"
        assert response.is_correct is True

    def test_empty_sentence_rejected_by_pydantic(self):
        """Pydantic should reject empty sentence strings"""
        with pytest.raises(Exception):
            FeedbackRequest(
                sentence="",
                target_language="Spanish",
                native_language="English",
            )