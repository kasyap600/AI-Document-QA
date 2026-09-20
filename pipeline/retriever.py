from config import TOP_K
from models.schemas import SearchResult
from pipeline.embedder import Embedder
from pipeline.vectorstore import VectorStore


def retrieve(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    top_k: int = TOP_K,
) -> list[SearchResult]:
    query_vector = embedder.encode_query(question)
    return store.search(query_vector, top_k=top_k)


def format_source_citation(result: SearchResult) -> str:
    if result.chunk.start_page == result.chunk.end_page:
        return f"Page {result.chunk.start_page}"
    return f"Pages {result.chunk.start_page}–{result.chunk.end_page}"


def format_context_for_prompt(results: list[SearchResult]) -> str:
    """Builds the context block injected into the LLM prompt."""
    sections = []
    for i, r in enumerate(results, start=1):
        citation = format_source_citation(r)
        sections.append(f"[Chunk {i} – {citation}]\n{r.chunk.text}")
    return "\n\n".join(sections)
