"""
Embedding generation utilities.

Uses sentence-transformers (free, local, open-source) by default so the
project runs without an OpenAI API key. Swap in a hosted embeddings API
by replacing generate_embedding() if you prefer.
"""
from functools import lru_cache
from src.config import EMBEDDING_DIMENSION

_model = None


@lru_cache(maxsize=1)
def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2 produces 384-dim vectors; we pad/truncate to
        # EMBEDDING_DIMENSION so the schema stays stable if you later swap
        # in a 1536-dim hosted model.
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def generate_embedding(text: str) -> list:
    """
    Generate an embedding vector for text.

    Args:
        text: Text to embed

    Returns:
        List of EMBEDDING_DIMENSION floats
    """
    model = _get_model()
    vector = model.encode(text).tolist()

    if len(vector) < EMBEDDING_DIMENSION:
        vector = vector + [0.0] * (EMBEDDING_DIMENSION - len(vector))
    elif len(vector) > EMBEDDING_DIMENSION:
        vector = vector[:EMBEDDING_DIMENSION]

    return vector


def get_embedding_dimension() -> int:
    return EMBEDDING_DIMENSION
