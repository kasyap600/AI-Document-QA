from config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM


def compute_confidence(top_score: float) -> str:
    """
    Maps the cosine similarity of the best-matching chunk to a label.
    Derived from retrieval score, not LLM self-assessment (more reliable).

      High   >= 0.75  strong semantic match
      Medium >= 0.55  topically related but less specific
      Low     < 0.55  weak match; answer may be speculative
    """
    if top_score >= CONFIDENCE_HIGH:
        return "High"
    if top_score >= CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"
