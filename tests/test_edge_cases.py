import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.feedback import get_feedback, _post_process
from app.models import FeedbackRequest, FeedbackResponse, ErrorDetail
from app.config import validation_error_types, cefr_difficulty_levels

"""Edge case and post-processing tests
- Verify behavior with unusual inputs and check that
post-processing logic correctly fixes logical inconsistencies in
LLM output, mocked API responses
"""

def _mock_completion(response_data: dict) -> MagicMock:
    """Build a mock OpenAI response object"""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = 100
    completion.usage.completion_tokens = 50
    completion.usage.prompt_tokens_details = None
    return completion

def _patch_openai(mock_response_data: dict):
    """Creates a mock OpenAI client with the mocked response data"""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion(mock_response_data)
    )
    return patch("app.feedback._get_client", return_value=mock_client)

@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear response cache before each test"""
    from app.cache import clear_cache
    clear_cache()
    yield
    clear_cache()

class TestPostProcessing:
    """Tests _post_process function that fixes LLM inconsistencies"""

    def test_fixes_is_correct_when_errors_present(self):
        """If LLM says is_correct=True but has errors, fix to False"""
        data = {
            "corrected_sentence": "Fixed.",
            "is_correct": True,  # Wrong, so there are errors
            "errors": [
                {
                    "original": "x",
                    "correction": "y",
                    "error_type": "grammar",
                    "explanation": "test",
                }
            ],
            "difficulty": "A1",
        }
        result = _post_process(data, "Original.")
        assert result["is_correct"] is False

    def test_fixes_is_correct_when_no_errors(self):
        """If the LLM says is_correct=False but has no errors, fix to True"""
        data = {
            "corrected_sentence": "Hola mundo.",
            "is_correct": False,  # Wrong, so there are no errors
            "errors": [],
            "difficulty": "A1",
        }
        result = _post_process(data, "Hola mundo.")
        assert result["is_correct"] is True

    def test_restores_original_when_no_errors(self):
        """If no errors, corrected_sentence should match original exactly"""
        original = "Ich bin hier."
        data = {
            "corrected_sentence": "Ich bin hier!",  # LLM changed punctuation
            "is_correct": True,
            "errors": [],
            "difficulty": "A1",
        }
        result = _post_process(data, original)
        assert result["corrected_sentence"] == original

    def test_handles_empty_corrected_sentence(self):
        """If corrected_sentence is empty, fall back to original"""
        original = "Hola."
        data = {
            "corrected_sentence": "",
            "is_correct": True,
            "errors": [],
            "difficulty": "A1",
        }
        result = _post_process(data, original)
        assert result["corrected_sentence"] == original

    def test_passes_through_valid_data(self):
        """Valid data should not be modified"""
        data = {
            "corrected_sentence": "Le chat noir.",
            "is_correct": False,
            "errors": [
                {
                    "original": "La chat",
                    "correction": "Le chat",
                    "error_type": "gender_agreement",
                    "explanation": "test",
                }
            ],
            "difficulty": "A1",
        }
        result = _post_process(data, "La chat noir.")
        assert result["is_correct"] is False
        assert len(result["errors"]) == 1

    def test_fixes_invalid_difficulty(self):
        """If difficulty is not a valid CEFR level, default to A2"""
        data = {
            "corrected_sentence": "Hola.",
            "is_correct": True,
            "errors": [],
            "difficulty": "X9",
        }
        result = _post_process(data, "Hola.")
        assert result["difficulty"] == "A2"

class TestErrorTypeNormalization:
    """Tests _normalize_error_type function"""

    def test_valid_types_pass_through(self):
        """All valid error types should be returned unchanged"""
        from app.feedback import _normalize_error_type
        for t in ["grammar", "spelling", "conjugation", "gender_agreement"]:
            assert _normalize_error_type(t) == t

    def test_aliases_are_mapped(self):
        """Common near labels should map to allowed type."""
        from app.feedback import _normalize_error_type
        assert _normalize_error_type("verb_conjugation") == "conjugation"
        assert _normalize_error_type("tense") == "conjugation"
        assert _normalize_error_type("typo") == "spelling"
        assert _normalize_error_type("vocabulary") == "word_choice"
        assert _normalize_error_type("formality") == "tone_register"
        assert _normalize_error_type("particle") == "grammar"

    def test_unknown_types_become_other(self):
        """Completely unrecognized types should fall back to other"""
        from app.feedback import _normalize_error_type
        assert _normalize_error_type("made_up_thing") == "other"
        assert _normalize_error_type("xyz") == "other"

    def test_case_insensitive(self):
        """Type matching should be case-insensitive"""
        from app.feedback import _normalize_error_type
        assert _normalize_error_type("Grammar") == "grammar"
        assert _normalize_error_type("SPELLING") == "spelling"

    def test_whitespace_stripped(self):
        """Leading and trailing whitespace should be stripped"""
        from app.feedback import _normalize_error_type
        assert _normalize_error_type("  grammar  ") == "grammar"

    def test_post_process_normalizes_error_types(self):
        """_post_process function should normalize error types"""
        data = {
            "corrected_sentence": "Fixed.",
            "is_correct": False,
            "errors": [
                {
                    "original": "x",
                    "correction": "y",
                    "error_type": "verb_conjugation",
                    "explanation": "test",
                }
            ],
            "difficulty": "A1",
        }
        result = _post_process(data, "Original.")
        assert result["errors"][0]["error_type"] == "conjugation"

@pytest.mark.asyncio
async def test_correct_sentence_has_empty_errors():
    """When is_correct is True, the errors list must be empty"""
    mock_data = {
        "corrected_sentence": "Bonjour le monde.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }
    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Bonjour le monde.",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)
    assert result.is_correct is True
    assert result.errors == []

@pytest.mark.asyncio
async def test_incorrect_sentence_has_non_empty_errors():
    """When is_correct is False, the errors list must not be empty"""
    mock_data = {
        "corrected_sentence": "Je suis content.",
        "is_correct": False,
        "errors": [
            {
                "original": "contente",
                "correction": "content",
                "error_type": "gender_agreement",
                "explanation": "Use masculine form with masculine subject.",
            }
        ],
        "difficulty": "A1",
    }
    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Je suis contente.",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)
    assert result.is_correct is False
    assert len(result.errors) > 0

@pytest.mark.asyncio
async def test_post_process_fixes_inconsistent_llm_output():
    """If LLM returns is_correct=True but with errors, post-processing
    should fix it to is_correct=False"""
    mock_data = {
        "corrected_sentence": "Fixed sentence.",
        "is_correct": True,  # Inconsistent, so there are errors
        "errors": [
            {
                "original": "bad",
                "correction": "good",
                "error_type": "word_choice",
                "explanation": "test",
            }
        ],
        "difficulty": "A1",
    }
    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Bad sentence.",
            target_language="English",
            native_language="Spanish",
        )
        result = await get_feedback(request)
    # Post-processing should have fixed this.
    assert result.is_correct is False

@pytest.mark.asyncio
async def test_punctuation_only_error():
    """A sentence where the only error is punctuation"""
    mock_data = {
        "corrected_sentence": "Hola, mundo.",
        "is_correct": False,
        "errors": [
            {
                "original": "Hola mundo",
                "correction": "Hola, mundo",
                "error_type": "punctuation",
                "explanation": "A comma is needed after a greeting.",
            }
        ],
        "difficulty": "A1",
    }
    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Hola mundo.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)
    assert result.errors[0].error_type == "punctuation"

@pytest.mark.asyncio
async def test_word_order_error():
    """A sentence with words in the wrong order"""
    mock_data = {
        "corrected_sentence": "Ich mag rote Blumen.",
        "is_correct": False,
        "errors": [
            {
                "original": "rote ich",
                "correction": "Ich mag rote",
                "error_type": "word_order",
                "explanation": "In German, the subject comes before the verb.",
            }
        ],
        "difficulty": "A2",
    }
    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Rote ich mag Blumen.",
            target_language="German",
            native_language="English",
        )
        result = await get_feedback(request)
    assert result.errors[0].error_type == "word_order"

@pytest.mark.asyncio
async def test_extra_word_error():
    """A sentence with an unnecessary extra word"""
    mock_data = {
        "corrected_sentence": "Je vais a Paris.",
        "is_correct": False,
        "errors": [
            {
                "original": "vais aller",
                "correction": "vais",
                "error_type": "extra_word",
                "explanation": "You do not need both 'vais' and 'aller' here.",
            }
        ],
        "difficulty": "A2",
    }
    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Je vais aller a Paris.",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)
    assert result.errors[0].error_type == "extra_word"

@pytest.mark.asyncio
async def test_missing_word_error():
    """A sentence with a missing word"""
    mock_data = {
        "corrected_sentence": "Yo no tengo hambre.",
        "is_correct": False,
        "errors": [
            {
                "original": "Yo tengo",
                "correction": "Yo no tengo",
                "error_type": "missing_word",
                "explanation": "The word 'no' is needed before the verb for negation.",
            }
        ],
        "difficulty": "A1",
    }
    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Yo tengo hambre.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)
    assert result.errors[0].error_type == "missing_word"

@pytest.mark.asyncio
async def test_very_short_sentence():
    """A single-word sentence should still produce a valid response"""
    mock_data = {
        "corrected_sentence": "Hola.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }
    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Hola.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)
    assert result.difficulty == "A1"
    assert result.is_correct is True

def test_response_model_rejects_bad_difficulty():
    """Pydantic model should reject invalid CEFR levels"""
    with pytest.raises(Exception):
        FeedbackResponse(
            corrected_sentence="test",
            is_correct=True,
            errors=[],
            difficulty="X1",
        )

def test_response_model_rejects_bad_error_type():
    """Pydantic model should reject invalid error types"""
    with pytest.raises(Exception):
        ErrorDetail(
            original="x",
            correction="y",
            error_type="invalid_type",
            explanation="test",
        )

def test_request_model_rejects_empty_sentence():
    """Request model should reject an empty sentence"""
    with pytest.raises(Exception):
        FeedbackRequest(
            sentence="",
            target_language="Spanish",
            native_language="English",
        )

def test_request_model_rejects_short_language():
    """Request model should reject a single char language string"""
    with pytest.raises(Exception):
        FeedbackRequest(
            sentence="Hola",
            target_language="S",
            native_language="English",
        )

@pytest.mark.asyncio
async def test_retry_on_transient_error():
    """The feedback function should retry on errors worth retrying for"""
    from openai import APIConnectionError

    mock_data = {
        "corrected_sentence": "Hola.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    mock_client = MagicMock()
    # First call raises an error, second call succeeds
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            APIConnectionError(request=MagicMock()),
            _mock_completion(mock_data),
        ]
    )

    with patch("app.feedback._get_client", return_value=mock_client):
        with patch("asyncio.sleep", new_callable=AsyncMock):  # Skip actual sleep
            request = FeedbackRequest(
                sentence="Hola.",
                target_language="Spanish",
                native_language="English",
            )
            result = await get_feedback(request)

    assert result.is_correct is True
    # Should be called twice, once failed and once succeeded
    assert mock_client.chat.completions.create.call_count == 2