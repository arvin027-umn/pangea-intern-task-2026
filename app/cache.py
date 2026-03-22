from app.config import cache_size_max

"""In-memory LRU (Least Recently Used)cache for feedback responses
- Avoids redundant API calls when the same sentence, language the user is trying to learn,
and language that the user is communicating in combination is requested more than once
- Saves money and reduces latency in prod
- Cache is stored in process memory, so it resets when the server restarts
"""

def make_cache_key(sentence: str, target_language: str, native_language: str) -> str:
    """Creates deterministic cache key from request fields
    - Normalization by stripping whitespace and lower casing language names
    so they hit the same cache entry
    - Sentence itself is kept as is due to capitalization and spacing being important for
    language correction
    """
    normalized_target = target_language.strip().lower()
    normalized_native = native_language.strip().lower()
    return f"{normalized_target}|{normalized_native}|{sentence}"


# Dictionary as a simple bounded cache
_cache: dict[str, dict] = {}
_cache_order: list[str] = []

def get_cached(key: str) -> dict | None:
    """Returns cached response data if it exists, else None"""
    return _cache.get(key)

def set_cached(key: str, value: dict) -> None:
    """Stores a response in cache, evicting oldest entry if it's full"""
    if key in _cache:
        return
    if len(_cache) >= cache_size_max:
        oldest = _cache_order.pop(0)
        _cache.pop(oldest, None)
    _cache[key] = value
    _cache_order.append(key)


def clear_cache() -> None:
    """Clears all cached entries if needed"""
    _cache.clear()
    _cache_order.clear()