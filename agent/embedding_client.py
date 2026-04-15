"""Embedding client with Ollama local backend (primary) and OpenRouter fallback.

Provides embedding generation via local Ollama server using nomic-embed-text,
with optional fallback to OpenRouter. Designed for graceful degradation —
all public functions return None on failure rather than raising exceptions.

Usage:
    from agent.embedding_client import get_embedding, get_embeddings, content_hash

    vec = get_embedding("hello world")           # np.ndarray (768,) or None
    vecs = get_embeddings(["hello", "world"])     # list of np.ndarray or None
    h = content_hash("hello world")              # "a948904f2f0f479e"
"""

import hashlib
import logging
import os
from typing import List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ollama local backend (primary)
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
OLLAMA_EMBEDDING_DIM: int = 768

# OpenRouter fallback
OPENROUTER_EMBEDDING_URL: str = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
OPENROUTER_EMBEDDING_DIM: int = 1536

# Active config — set by _detect_backend() on first call
EMBEDDING_MODEL: str = OLLAMA_EMBEDDING_MODEL
EMBEDDING_DIM: int = OLLAMA_EMBEDDING_DIM

_MAX_TEXT_LENGTH: int = 30_000
_MAX_BATCH_SIZE: int = 100
_TIMEOUT_SECONDS: float = 30.0

_backend: Optional[str] = None  # "ollama" or "openrouter", detected lazily


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str) -> str:
    """Truncate text to _MAX_TEXT_LENGTH characters."""
    if len(text) > _MAX_TEXT_LENGTH:
        logger.debug(
            "Truncating text from %d to %d chars", len(text), _MAX_TEXT_LENGTH
        )
        return text[:_MAX_TEXT_LENGTH]
    return text


def _detect_backend() -> str:
    """Detect available embedding backend. Prefers Ollama, falls back to OpenRouter."""
    global _backend, EMBEDDING_MODEL, EMBEDDING_DIM

    if _backend is not None:
        return _backend

    # Try Ollama first
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{OLLAMA_BASE_URL}/api/version")
            if resp.status_code == 200:
                _backend = "ollama"
                EMBEDDING_MODEL = OLLAMA_EMBEDDING_MODEL
                EMBEDDING_DIM = OLLAMA_EMBEDDING_DIM
                logger.info("Embedding backend: Ollama (%s) at %s", OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL)
                return _backend
    except Exception:
        pass

    # Fall back to OpenRouter
    if os.environ.get("OPENROUTER_API_KEY"):
        _backend = "openrouter"
        EMBEDDING_MODEL = OPENROUTER_EMBEDDING_MODEL
        EMBEDDING_DIM = OPENROUTER_EMBEDDING_DIM
        logger.info("Embedding backend: OpenRouter (%s)", OPENROUTER_EMBEDDING_MODEL)
        return _backend

    logger.warning("No embedding backend available (Ollama not running, OPENROUTER_API_KEY not set)")
    _backend = "none"
    return _backend


def content_hash(text: str) -> str:
    """Return the first 16 hex characters of the SHA-256 hash of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ollama embedding
# ---------------------------------------------------------------------------


def _get_embeddings_ollama(texts: List[str]) -> Optional[List[np.ndarray]]:
    """Fetch embeddings from local Ollama server."""
    truncated = [_truncate(t) for t in texts]

    # nomic-embed-text expects "search_document: " or "search_query: " prefix
    # For storage, use "search_document: " prefix
    prefixed = [f"search_document: {t}" for t in truncated]

    payload = {
        "model": OLLAMA_EMBEDDING_MODEL,
        "input": prefixed,
    }

    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        embeddings_list = data.get("embeddings", [])

        if len(embeddings_list) != len(texts):
            logger.error(
                "Expected %d embeddings, got %d",
                len(texts),
                len(embeddings_list),
            )
            return None

        results: List[np.ndarray] = []
        for vec_data in embeddings_list:
            vec = np.array(vec_data, dtype=np.float32)
            results.append(vec)

        return results

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Ollama embedding HTTP error %d: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        return None
    except httpx.TimeoutException:
        logger.error("Ollama embedding request timed out after %.0fs", _TIMEOUT_SECONDS)
        return None
    except (httpx.RequestError, KeyError, ValueError, TypeError) as exc:
        logger.error("Ollama embedding request failed: %s", exc)
        return None
    except Exception:
        logger.exception("Unexpected error fetching Ollama embeddings")
        return None


# ---------------------------------------------------------------------------
# OpenRouter embedding (fallback)
# ---------------------------------------------------------------------------


def _get_embeddings_openrouter(texts: List[str]) -> Optional[List[np.ndarray]]:
    """Fetch embeddings from OpenRouter API."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set — embeddings unavailable")
        return None

    truncated = [_truncate(t) for t in texts]

    payload = {
        "model": OPENROUTER_EMBEDDING_MODEL,
        "input": truncated,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.post(
                OPENROUTER_EMBEDDING_URL,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

        data = response.json()

        embedding_objects = data.get("data", [])
        if len(embedding_objects) != len(texts):
            logger.error(
                "Expected %d embeddings, got %d",
                len(texts),
                len(embedding_objects),
            )
            return None

        embedding_objects.sort(key=lambda obj: obj.get("index", 0))

        results: List[np.ndarray] = []
        for obj in embedding_objects:
            vec = np.array(obj["embedding"], dtype=np.float32)
            results.append(vec)

        return results

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Embedding API HTTP error %d: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        return None
    except httpx.TimeoutException:
        logger.error("Embedding API request timed out after %.0fs", _TIMEOUT_SECONDS)
        return None
    except (httpx.RequestError, KeyError, ValueError, TypeError) as exc:
        logger.error("Embedding request failed: %s", exc)
        return None
    except Exception:
        logger.exception("Unexpected error fetching embeddings")
        return None


# ---------------------------------------------------------------------------
# Core embedding functions (public API)
# ---------------------------------------------------------------------------


def get_embeddings(texts: List[str]) -> Optional[List[np.ndarray]]:
    """Fetch embeddings for a batch of texts (max 100).

    Parameters
    ----------
    texts:
        List of input strings.  Each is truncated to 30 000 characters.

    Returns
    -------
    List of numpy arrays in the same order as *texts*, or ``None`` if the
    request fails for any reason. Dimension depends on the active backend
    (768 for Ollama/nomic-embed-text, 1536 for OpenRouter/text-embedding-3-small).
    """
    if not texts:
        return []

    if len(texts) > _MAX_BATCH_SIZE:
        logger.error(
            "Batch size %d exceeds maximum of %d", len(texts), _MAX_BATCH_SIZE
        )
        return None

    backend = _detect_backend()

    if backend == "ollama":
        return _get_embeddings_ollama(texts)
    elif backend == "openrouter":
        return _get_embeddings_openrouter(texts)
    else:
        logger.error("No embedding backend available")
        return None


def get_embedding(text: str) -> Optional[np.ndarray]:
    """Fetch the embedding for a single text string.

    Returns a numpy array (float32), or ``None`` on failure.
    Dimension depends on the active backend.
    """
    result = get_embeddings([text])
    if result is None or len(result) == 0:
        return None
    return result[0]


def get_query_embedding(text: str) -> Optional[np.ndarray]:
    """Fetch embedding for a search query (uses 'search_query:' prefix for nomic).

    For Ollama/nomic-embed-text, this uses the proper 'search_query:' prefix
    instead of 'search_document:' to get better retrieval results.
    """
    backend = _detect_backend()

    if backend == "ollama":
        truncated = _truncate(text)
        prefixed = f"search_query: {truncated}"

        payload = {
            "model": OLLAMA_EMBEDDING_MODEL,
            "input": [prefixed],
        }

        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json=payload,
                )
                response.raise_for_status()

            data = response.json()
            embeddings_list = data.get("embeddings", [])
            if not embeddings_list:
                return None
            return np.array(embeddings_list[0], dtype=np.float32)

        except Exception as exc:
            logger.error("Ollama query embedding failed: %s", exc)
            return None
    else:
        # For OpenRouter, no special prefix needed
        return get_embedding(text)
