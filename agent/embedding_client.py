"""Embedding client for OpenRouter API.

Provides embedding generation via OpenRouter's embedding endpoint using
the text-embedding-3-small model. Designed for graceful degradation —
all public functions return None on failure rather than raising exceptions.

Usage:
    from agent.embedding_client import get_embedding, get_embeddings, content_hash

    vec = get_embedding("hello world")           # np.ndarray (1536,) or None
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

EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
EMBEDDING_DIM: int = 1536
OPENROUTER_EMBEDDING_URL: str = "https://openrouter.ai/api/v1/embeddings"

_MAX_TEXT_LENGTH: int = 30_000
_MAX_BATCH_SIZE: int = 100
_TIMEOUT_SECONDS: float = 30.0


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


def _get_api_key() -> Optional[str]:
    """Retrieve OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        logger.warning("OPENROUTER_API_KEY not set — embeddings unavailable")
        return None
    return key


def content_hash(text: str) -> str:
    """Return the first 16 hex characters of the SHA-256 hash of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core embedding functions
# ---------------------------------------------------------------------------


def get_embeddings(texts: List[str]) -> Optional[List[np.ndarray]]:
    """Fetch embeddings for a batch of texts (max 100).

    Parameters
    ----------
    texts:
        List of input strings.  Each is truncated to 30 000 characters.

    Returns
    -------
    List of numpy arrays (shape ``(1536,)``, dtype float32) in the same
    order as *texts*, or ``None`` if the request fails for any reason.
    """
    if not texts:
        return []

    if len(texts) > _MAX_BATCH_SIZE:
        logger.error(
            "Batch size %d exceeds maximum of %d", len(texts), _MAX_BATCH_SIZE
        )
        return None

    api_key = _get_api_key()
    if api_key is None:
        return None

    truncated = [_truncate(t) for t in texts]

    payload = {
        "model": EMBEDDING_MODEL,
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

        # The API returns embeddings in data[].embedding, sorted by index.
        embedding_objects = data.get("data", [])
        if len(embedding_objects) != len(texts):
            logger.error(
                "Expected %d embeddings, got %d",
                len(texts),
                len(embedding_objects),
            )
            return None

        # Sort by index to guarantee order matches input.
        embedding_objects.sort(key=lambda obj: obj.get("index", 0))

        results: List[np.ndarray] = []
        for obj in embedding_objects:
            vec = np.array(obj["embedding"], dtype=np.float32)
            if vec.shape != (EMBEDDING_DIM,):
                logger.warning(
                    "Unexpected embedding dimension: %s (expected %d)",
                    vec.shape,
                    EMBEDDING_DIM,
                )
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


def get_embedding(text: str) -> Optional[np.ndarray]:
    """Fetch the embedding for a single text string.

    Returns a numpy array of shape ``(1536,)`` (float32), or ``None`` on
    failure.
    """
    result = get_embeddings([text])
    if result is None or len(result) == 0:
        return None
    return result[0]
