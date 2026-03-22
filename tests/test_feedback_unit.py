import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.feedback import get_feedback
from app.models import FeedbackRequest

"""Unit tests -- run without an API key using mocked LLM responses

Verify that:
- The feedback function correctly parses LLM output into Pydantic models
- Different languages and error types are handled properly
- The cache is used when available
- Edge cases like correct sentences and multiple errors work as intended
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
    """Clear response cache before each test so tests are independent"""
    from app.cache import clear_cache
    clear_cache()
    yield
    clear_cache()

@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    """A Spanish sentence with a verb conjugation mistake"""
    mock_data = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.errors[0].original == "soy fue"
    assert result.errors[0].correction == "fui"
    assert result.difficulty == "A2"

@pytest.mark.asyncio
async def test_correct_german_sentence():
    """A correct sentence should have is_correct=True and no errors"""
    mock_data = {
        "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == request.sentence
    assert result.difficulty == "B1"

@pytest.mark.asyncio
async def test_french_multiple_gender_errors():
    """A French sentence with two gender agreement mistakes"""
    mock_data = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine in French.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine in French.",
            },
        ],
        "difficulty": "A1",
    }

    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)

@pytest.mark.asyncio
async def test_japanese_particle_error():
    """A Japanese sentence with a wrong particle (non-Latin script)"""
    mock_data = {
        "corrected_sentence": "私は東京に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": "The verb 住む uses に for location, not を.",
            }
        ],
        "difficulty": "A2",
    }

    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) == 1
    assert result.errors[0].correction == "に"

@pytest.mark.asyncio
async def test_portuguese_spelling_and_grammar():
    """A Portuguese sentence with both a spelling error and a grammar error"""
    mock_data = {
        "corrected_sentence": "Eu quero comprar um presente para minha irma, mas nao sei do que ela gosta.",
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": "'Present' in Portuguese is spelled 'presente' with an 's'.",
            },
            {
                "original": "o que ela gosta",
                "correction": "do que ela gosta",
                "error_type": "grammar",
                "explanation": "The verb 'gostar' requires the preposition 'de'.",
            },
        ],
        "difficulty": "B1",
    }

    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="Eu quero comprar um prezente para minha irma, mas nao sei o que ela gosta.",
            target_language="Portuguese",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) == 2
    error_types = {e.error_type for e in result.errors}
    assert "spelling" in error_types
    assert "grammar" in error_types

@pytest.mark.asyncio
async def test_korean_particle_error():
    """A Korean sentence with a particle mistake"""
    mock_data = {
        "corrected_sentence": "나는 어제 학교에 갔어요.",
        "is_correct": False,
        "errors": [
            {
                "original": "학교를",
                "correction": "학교에",
                "error_type": "grammar",
                "explanation": "Use the particle 에 with movement verbs like 가다.",
            }
        ],
        "difficulty": "A2",
    }

    with _patch_openai(mock_data):
        request = FeedbackRequest(
            sentence="나는 어제 학교를 갔어요.",
            target_language="Korean",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.errors[0].error_type == "grammar"

@pytest.mark.asyncio
async def test_cache_prevents_duplicate_api_calls():
    """The second call with the same input should use the cache, not the API"""
    mock_data = {
        "corrected_sentence": "Hola mundo.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion(mock_data)
    )

    with patch("app.feedback._get_client", return_value=mock_client):
        request = FeedbackRequest(
            sentence="Hola mundo.",
            target_language="Spanish",
            native_language="English",
        )
        # First call hits the API
        result1 = await get_feedback(request)
        # Second call should use cache and not hit the API
        result2 = await get_feedback(request)

    # The API should only have been called once
    assert mock_client.chat.completions.create.call_count == 1
    assert result1.corrected_sentence == result2.corrected_sentence

@pytest.mark.asyncio
async def test_all_error_types_are_accepted():
    """Verify that every allowed error type can be parsed without errors"""
    from app.config import validation_error_types

    for error_type in validation_error_types:
        mock_data = {
            "corrected_sentence": "Test.",
            "is_correct": False,
            "errors": [
                {
                    "original": "x",
                    "correction": "y",
                    "error_type": error_type,
                    "explanation": "Test explanation.",
                }
            ],
            "difficulty": "A1",
        }

        with _patch_openai(mock_data):
            request = FeedbackRequest(
                sentence="Test input.",
                target_language="Spanish",
                native_language="English",
            )
            # Clear cache so each iteration calls the mock
            from app.cache import clear_cache
            clear_cache()
            result = await get_feedback(request)
            assert result.errors[0].error_type == error_type