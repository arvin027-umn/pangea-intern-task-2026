## Language Feedback API Task - Pangea Chat

Provided a sentence in a target language that the learner wants to learn and the language the learner is using to communicate with the API in (native language), the API returns a corrected sentence if the sentence was wrong, with the list of specific errors in simple language alongside a CEFR difficulty rating.

## Setup

```bash
git clone https://github.com/arvin027-umn/intern-task-2026.git
cd intern-task-2026

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY for local integration testing

# Start the server
uvicorn app.main:app --reload
```

In a second terminal, run:

```bash
curl http://localhost:8000/health
# {"status": "ok"}

curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Yo soy fue al mercado ayer.", "target_language": "Spanish", "native_language": "English"}'
```

### Run with Docker and tests

```bash
# Create the docker container
docker compose up --build

# All tests
docker compose exec feedback-api pytest tests/ -v

# Unit tests only (no API key needed)
docker compose exec feedback-api pytest tests/test_api.py tests/test_feedback_unit.py tests/test_schema.py tests/test_edge_cases.py -v
```

### Design Decisions & Justifications

#### General Design Pointers

Every design decision I've made leads back a single question which is: what exactly is it that a language learner really needs when they submit their sentence?

They need to shown what in their answer might be wrong, and they need to be able to understand it in simple language in the language they are using to communicate with the API. The reason I emphasize the simplicity of the feedback is that if it starts to sound like a textbook's explanations, it will bore the user. And, another thing to keep in mind here is the speed of the response, so the feedback needs to be sent to them at a pace that makes it feel like a friendly conversation.

This influenced three specific choices I made in my system prompt:

1. **Feedback/explanations need to be in the learner's native language**
   1. A student that speaks french is attempting to learn English won't be able to effectively grasp the language they are trying to learn if they are given an explanation of their sentence in the target language.
   2. The system prompt therefore reinforces this in different places, as in the step-by-step instructions, the explanation rules, the JSON response's description field, and within the user message.
   3. Some of my earlier experiments showed the model ignoring this instruction, so I had it in multiple places to reinforce it for the model till the responses were reliably in the learner's native language
2. **Minimalistic corrections**
   1. In the system prompt there are specific instructions to not rephrase, restructure, or improve the sentence aside from fixing the errors in it. Because if a learner writes an awkwardly phrased sentence that is technically grammatically correct, then penalizing them for it will negatively impact their confidence.
   2. The post-processing layer for the feedback enforces this by ensuring that if the model returns no errors, the corrected sentence is transformed back to the original input sentence as it was written by the learner.
3. **CEFR Difficulty Level is about complexity, not mistakes made**
   1. CEFR Difficulty Level is a reflection of complexity of sentence being attempted and not about how many mistakes the student made

#### API Choice & Model Choice

I chose OpenAI's API for its structured outputs feature that allows for strict JSON schema enforcement, and this feature has been in place for longer than Anthropic's structured outputs feature, and has a lot more developer support making it simpler to debug. Setting strict to true in the schema forces the model to only produce tokens that lead to valid JSON, therefore, invalid error types, missing fields and incorrect data types are impossible at the token level. OpenAI has certified that the schema compliance is 100% using this mode, and squashes many potential bugs that could come up.

I chose gpt-4o-mini as the model for this task as it is the cheapest model supporting structured outputs, and can scale very well even though it may not be a reasoning giant like gpt-5-mini or the any of the other GPT-5 variants that have come out this year. Since the task description mentioned cost efficiency, this model felt like an easy choice to make. If reliability was more important and this were a production setting, I would swap the model out to use gpt-5-mini as it is a reasoning model and can add in an additional layer of intelligence over just autoregressively predicting the next token, which is essentially what gpt-4o-mini does. The cost of the model for 1M input and output tokens is only 0.15 USD and 0.60 USD respectively. With OpenAI's caching in place and the caching of the response I've implemented on my side for same sentence and same language pair, the cost remains relatively low per request and shouldn't go about 1-2 USD for every 10-15k requests.

I've used OpenAI's Responses API over the Chat Completions API as OpenAI's docs specifically advise to use Responses for new projects and also due the prompt caching feature being offered. But I have also included fallbacks to the Chat Completions API with strict json_schema in case Responses API is unavailable, and Chat Completions API with json_object if for some reason the provider doesn't support json_schema like using Github Models.

#### System Prompt

The system prompt cotains 10 few-shot examples covering Spanish, French, Japanese, German, Korean, Mandarin Chinese, Portuguese, Arabic, English (with Spanish native speaker), and Russian to match the details given about the task's test suite in the FAQ.

The examples in these different languages exist to demonstrate output format across different writing systems and error types, not to hand-hold the model through grammar it already has in its weights. Examples 1-4 and 7 come directly from `sample_inputs.json`. Examples 5, 6, and 8-10 were generated by the model and verified for linguistic accuracy via published grammar references (sources are documented in the integration test docstrings in `test_feedback_integration.py`).

The prompt also contains a step-by-step reasoning section, beginning with reading the sentence carefully word by word, identifying every error, and for each error, determining the minimal span of incorrect text, and then providing a friendly and short response to the user about the errors in the sentence. This procedure is given to the model because structured reasoning improves accuracy on multi-error sentences where the model might otherwise miss the second or third error. All examples sit in the system prompt, so they are part of the static prefix that OpenAI caches, meaning the examples improve accuracy without increasing per-request cost after the first call.

#### What happens if the LLM gets it wrong

Despite the quality of the prompt, LLMs can still sometimes produce inconsistent output. I've handled this at multiple layers:

**Structured Outputs:** The JSON schema is enforced at the token level. The schema includes enum constraints for all 12 error types and all 6 CEFR levels, plus `additionalProperties: false` on every object that restricts the model from randomly adding in fields to the JSON object. Invalid values cannot be generated.

**Post-processing:** A `_post_process` function enforces logical consistency that the schema alone cannot express:

- If there are errors but is_correct is true, it flips is_correct to false
- If there are no errors but is_correct is false, it flips is_correct to true
- If there are no errors, corrected_sentence is set to the original input exactly
- If corrected_sentence is empty, it falls back to the original
- If difficulty is not a valid CEFR level, it defaults to A2

**Error type normalization:** LLMs sometimes return near-miss labels like verb_conjugation instead of conjugation, or typo instead of spelling. A normalization layer maps more than 20 common aliases to the correct allowed values. Completely unrecognized types become of type other instead of erroring out. This is tested with dedicated unit tests in `test_edge_cases.py`.

**Retry with exponential backoff:** Transient API errors (timeouts, connection drops) and malformed JSON are retried up to 2 times with exponential backoff (1s, then 2s). But billing errors like insufficient_quota are never retried because they won't succeed. The function `_is_retryable_rate_limit` checks the error code before deciding.

**Singleton HTTP client:** The OpenAI client is created once and reused across all requests, avoiding repeated TLS handshakes and connection setup. The client timeout is set to 25 seconds, to allow for some buffer time to be under the 30-second response timing requirement.

#### Caching Strategy

There are two layers of caching that have been added in, each attending to a different problem:

**OpenAI prompt caching (automatic):**. The system prompt is about 2700 tokens and is identical across all requests. OpenAI caches prompt prefixes longer than 1024 tokens after the first call, therefore, subsequent requests skip re-processing the prompt. This should cut input token cost by a lot.

**Application-level LRU cache** stores complete responses keyed by (sentence, target_language, native_language). If the same request is submitted again, the cached response is returned instantly with zero API cost and near-zero latency. In a real production setting this is of importance, if ten students are making the same mistake on the same exercise, it shouldn't trigger ten API calls. The cache holds up to 1024 entries with bounded memory. Language names are normalized (lowercased, stripped) for better cache hits. 

#### Verifying Accuracy of outputs for languages I don't speak

I don't speak Japanese, Korean, Russian, Arabic, or Chinese and I only know a little of French and Spanish. So, here is how I verified the API produces correct results for these languages:

**Task-provided examples:** The sample_inputs.json file has known-correct expected outputs for Spanish, French, Japanese, German, and Portuguese, so I've used them as test cases in `test_feedback_integration.py`.

**Published grammar references:** For languages beyond sample_inputs.json, I've cited specific grammar sources for Korean, Russian, Arabic, and Chinese in `test_feedback_integration.py`.

**Structural consistency checks:** Even without knowing a language, the tests verify the following:

1. If is_correct=false, does it have a non-empty errors list?
2. Does the corrected sentence differ from the input?
3. Is the difficulty a valid CEFR level? These will catch model failures regardless of language.

#### Test Suite & What each test layer catches

**test_api.py (12 tests)** tests the actual HTTP endpoints using FastAPI's TestClient: health check returns 200, POST-only health returns 405, missing/empty/invalid fields return 422, valid requests return 200 with correct JSON shape, Content-Type is application/json, unknown paths return 404. These catch issues that unit tests miss, like serialization bugs or FastAPI validation behavior.

**test_feedback_unit.py (8 tests)** tests the feedback pipeline with mocked LLM responses. Covers Spanish conjugation, correct German sentence, French multiple errors, Japanese particles, Portuguese spelling with grammar, and Korean particles. Also tests that the cache prevents duplicate API calls (mock is called exactly once for two identical requests) and that all 12 error types are accepted by the parser.

**test_edge_cases.py (25 tests)** tests the defensive layers that protect against bad LLM output. The TestPostProcessing class (6 tests) verifies that _post_process fixes is_correct/errors inconsistencies, restores the original sentence when there are no errors, handles empty corrected_sentence, and defaults invalid difficulty to A2. The TestErrorTypeNormalization class (6 tests) verifies alias mapping, case insensitivity, whitespace handling, and unknown-type fallback. Additional tests cover every error type (punctuation, word_order, extra_word, missing_word), Pydantic model validation, and retry behavior on transient API errors.

**test_schema.py (17 tests)** validates that Pydantic models and JSON schema files agree on what is valid. Tests both request and response schemas: valid data passes, missing fields fail, invalid enums fail, extra fields are rejected. Validates every error type and CEFR level individually. Confirms all sample_inputs.json examples pass both schemas. Tests FeedbackRequest rejects empty sentences.

**test_feedback_integration.py (12 tests)** makes real API calls across 10 languages: Spanish, French, German, Japanese, Portuguese, Korean, Arabic, Mandarin Chinese, Italian, and Russian, testing specific linguistic corrections, response time staying under 30 seconds, and that explanations respect the user's native language (Spanish explanations for an English error).

### API Endpoints

### POST /feedback

**Request body** (see [schema/request.schema.json](schema/request.schema.json)):

```json
{
  "sentence": "Yo soy fue al mercado ayer.",
  "target_language": "Spanish",
  "native_language": "English"
}
```

**Response body** (see [schema/response.schema.json](schema/response.schema.json)):

```json
{
  "corrected_sentence": "Yo fui al mercado ayer.",
  "is_correct": false,
  "errors": [
    {
      "original": "soy fue",
      "correction": "fui",
      "error_type": "conjugation",
      "explanation": "You mixed two verb forms. 'Soy' is present tense of 'ser' (to be), and 'fue' is past tense of 'ir' (to go). You only need 'fui' (I went)."
    }
  ],
  "difficulty": "A2"
}
```

### GET /health

Returns `{"status": "ok"}` with a 200 status code.
