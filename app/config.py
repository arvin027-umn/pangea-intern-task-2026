validation_error_types = frozenset([
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
])

cefr_difficulty_levels = frozenset(["A1", "A2", "B1", "B2", "C1", "C2"])

# gpt-4o-mini is the cheapest model supporting structured outputs
# so it guarantees valid JSON schema
openai_model = "gpt-4o-mini"

# temperature 0 for most deterministic output
openai_temperature = 0

# max num of cached responses to maintain in memory for avodiing repeated API calls on redundant inputs
cache_size_max = 1024