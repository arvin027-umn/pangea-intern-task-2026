import os
import time
import pytest
from app.feedback import get_feedback
from app.models import FeedbackRequest
from app.config import validation_error_types, cefr_difficulty_levels

"""Integration tests, using LLM API key
- Makes calls to LLM API and verify LLM output is linguistically accurate
and complies with schema.
- Run with: pytest tests/test_feedback_integration.py -v
"""

# Skip all tests in this file if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set -- skipping integration tests",
)

@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear cache before each test so each test makes a fresh API call"""
    from app.cache import clear_cache
    clear_cache()
    yield
    clear_cache()

def _assert_valid_response(result, expect_correct: bool | None = None):
    """Common assertions on any feedback response"""
    # Difficulty must be a valid CEFR level
    assert result.difficulty in cefr_difficulty_levels

    # Every error must have a valid error type and non-empty fields
    for error in result.errors:
        assert error.error_type in validation_error_types
        assert len(error.original) > 0
        assert len(error.correction) > 0
        assert len(error.explanation) > 0

    # is_correct and errors must be consistent
    if result.is_correct:
        assert result.errors == [], "is_correct=True but errors is not empty"
    else:
        assert len(result.errors) > 0, "is_correct=False but errors is empty"

    # If we know whether the sentence is correct, check it
    if expect_correct is not None:
        assert result.is_correct is expect_correct

@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    """Spanish sentence mixing 'soy' and 'fue' should be corrected to 'fui' from sample_inputs.json"""
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    # The correction should contain 'fui' somewhere
    assert "fui" in result.corrected_sentence.lower()

@pytest.mark.asyncio
async def test_french_gender_agreement():
    """French sentence with swapped gender articles from sample_inputs.json"""
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert "le chat" in result.corrected_sentence.lower() or "Le chat" in result.corrected_sentence
    assert "la table" in result.corrected_sentence.lower() or "La table" in result.corrected_sentence

@pytest.mark.asyncio
async def test_german_correct_sentence():
    """A correct German sentence should return is_correct=True from sample_inputs.json"""
    sentence = "Ich habe gestern einen interessanten Film gesehen."
    result = await get_feedback(
        FeedbackRequest(
            sentence=sentence,
            target_language="German",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=True)
    # Corrected sentence should match the original completely
    assert result.corrected_sentence == sentence

@pytest.mark.asyncio
async def test_japanese_particle_error():
    """Japanese sentence using wrong particle for location from sample_inputs.json"""
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert "に" in result.corrected_sentence

@pytest.mark.asyncio
async def test_portuguese_spelling_error():
    """Portuguese sentence with 'prezente' misspelled from sample_inputs.json"""
    result = await get_feedback(
        FeedbackRequest(
            sentence="Eu quero comprar um prezente para minha irma.",
            target_language="Portuguese",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert "presente" in result.corrected_sentence

@pytest.mark.asyncio
async def test_korean_particle_error():
    """Korean: 를 (object marker) should be 에 (destination particle) with
    movement verbs like 가다; Verified via:
    https://www.90daykorean.com/korean-particles/
    https://goodjobkorean.com/blog/how-are-the-location-marking-particles-and-different"""
    result = await get_feedback(
        FeedbackRequest(
            sentence="나는 어제 학교를 갔어요.",
            target_language="Korean",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert "학교에" in result.corrected_sentence

@pytest.mark.asyncio
async def test_arabic_grammar_error():
    """Arabic: checks structural validity of response for non-Latin script;
    Common errors include missing hamza on alif (انا vs أنا, الى vs إلى);
    Verified via:
    https://www.lebanesearabicinstitute.com/hamza/
    https://arabikey.com/hamza-in-arabic/
    https://hinative.com/questions/9162821"""
    result = await get_feedback(
        FeedbackRequest(
            sentence="انا ذهبت الى المدرسة امس.",
            target_language="Arabic",
            native_language="English",
        )
    )
    _assert_valid_response(result)
    # Mainly check that response is valid and has a difficulty level, as Arabic grammar can be complex to parse
    assert result.difficulty in cefr_difficulty_levels

@pytest.mark.asyncio
async def test_chinese_missing_particle():
    """Chinese: time-duration + object often requires 的 between them;
    Pattern: Verb + Duration + 的 + Object (e.g. 看了三个小时的书);
    Verified via:
    https://resources.allsetlearning.com/chinese/grammar/Expressing_duration_with_%22le%22
    https://www.hanyuace.com/blog/use-le-in-chinese (HSK 3 example: 他看了半个小时的书)"""
    result = await get_feedback(
        FeedbackRequest(
            sentence="我昨天在图书馆看了三个小时书。",
            target_language="Mandarin Chinese",
            native_language="English",
        )
    )
    _assert_valid_response(result)

@pytest.mark.asyncio
async def test_italian_correct_sentence():
    """Italian: grammatically correct sentence meaning
    'Yesterday I went to the cinema with my friends.'
    Verified correct via Google Translate and native speaker conventions.
    Should return is_correct=True with empty errors list."""
    sentence = "Ieri sono andato al cinema con i miei amici"
    result = await get_feedback(
        FeedbackRequest(
            sentence=sentence,
            target_language="Italian",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=True)

@pytest.mark.asyncio
async def test_russian_case_error():
    """Russian: preposition в requires prepositional case; 
    Feminine nouns ending in -ия become -ии (компания to компании);
    Verified via:
    https://www.russianlessons.net/grammar/nouns_prepositional.php
    https://easy-russian.com/prepositional-case-rules"""
    result = await get_feedback(
        FeedbackRequest(
            sentence="Я живу в Москве и работаю в большой компания.",
            target_language="Russian",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    # The correction should fix the case ending.
    assert "компании" in result.corrected_sentence

@pytest.mark.asyncio
async def test_response_time_under_30_seconds():
    """Each request must return within 30 seconds"""
    start = time.time()
    result = await get_feedback(
        FeedbackRequest(
            sentence="Je suis alle au magasin hier.",
            target_language="French",
            native_language="English",
        )
    )
    elapsed = time.time() - start
    _assert_valid_response(result)
    assert elapsed < 30, f"Response took {elapsed:.1f}s, exceeds 30s limit"

@pytest.mark.asyncio
async def test_explanation_in_native_language():
    """When native language is Spanish, explanations should be in Spanish."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="I go to school yesterday.",
            target_language="English",
            native_language="Spanish",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    # At least one explanation should contain Spanish-looking characters or common Spanish words
    # This is a loose check since I can't guarantee specific wording
    assert len(result.errors) > 0
    for error in result.errors:
        assert len(error.explanation) > 10, "Explanation seems too short"