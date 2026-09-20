import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import COLLECTION_NAME, EMBEDDING_DIM
from models.schemas import SearchResult, TextChunk


class VectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self._client = QdrantClient(":memory:")
        self._collection = collection_name
        self._ensure_collection()

    def reset_collection(self) -> None:
        """Drop and recreate the collection. Called on each new PDF upload."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._ensure_collection()

    def upsert_chunks(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        """Store chunks with their embeddings. Payload holds all metadata needed for citations."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "text": chunk.text,
                    "doc_name": chunk.doc_name,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        """Cosine similarity search. Returns results sorted by score descending."""
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            SearchResult(
                chunk=TextChunk(
                    text=hit.payload["text"],
                    doc_name=hit.payload["doc_name"],
                    start_page=hit.payload["start_page"],
                    end_page=hit.payload["end_page"],
                    chunk_index=hit.payload["chunk_index"],
                ),
                score=hit.score,
            )
            for hit in hits
        ]

    def _ensure_collection(self) -> None:
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
