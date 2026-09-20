import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model = SentenceTransformer(model_name)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts. Returns float32 array of shape (N, 384)."""
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def encode_query(self, text: str) -> list[float]:
        """Encode a single query string. Returns a plain list for Qdrant search()."""
        return self._model.encode(text, convert_to_numpy=True).tolist()
