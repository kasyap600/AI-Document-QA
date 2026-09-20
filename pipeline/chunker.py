import bisect

from config import CHUNK_SIZE, CHUNK_OVERLAP
from models.schemas import PageText, TextChunk


def chunk_pages(
    pages: list[PageText],
    doc_name: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Splits page text into overlapping chunks while tracking which page(s)
    each chunk originated from.

    Strategy:
      1. Concatenate all pages into one string, recording the character
         range each page occupies (page_spans).
      2. Slide a window of `chunk_size` characters, stepping by
         (chunk_size - overlap).
      3. For each window, binary-search page_spans to find the page that
         owns the first character → start_page, and the page that owns
         the last character → end_page.
    """
    if not pages:
        return []

    full_text, page_spans = _build_full_text(pages)
    step = max(1, chunk_size - overlap)
    chunks = []

    for idx, pos in enumerate(range(0, len(full_text), step)):
        text = full_text[pos : pos + chunk_size].strip()
        if not text:
            continue

        start_page = _page_for_offset(page_spans, pos)
        end_page = _page_for_offset(page_spans, min(pos + chunk_size - 1, len(full_text) - 1))

        chunks.append(
            TextChunk(
                text=text,
                doc_name=doc_name,
                start_page=start_page,
                end_page=end_page,
                chunk_index=idx,
            )
        )

    return chunks


def _build_full_text(pages: list[PageText]) -> tuple[str, list[tuple[int, int, int]]]:
    """
    Returns (full_text, page_spans) where each page_span is
    (start_char, end_char, page_number) — end_char is inclusive.
    Pages are joined with a newline separator.
    """
    parts = []
    page_spans: list[tuple[int, int, int]] = []
    cursor = 0

    for page in pages:
        start = cursor
        end = cursor + len(page.text) - 1
        page_spans.append((start, end, page.page_number))
        parts.append(page.text)
        cursor = end + 1 + 1  # +1 for the "\n" separator

    return "\n".join(parts), page_spans


def _page_for_offset(page_spans: list[tuple[int, int, int]], offset: int) -> int:
    """
    Binary search: find which page owns the character at `offset`.
    page_spans is sorted by start_char. Returns the page_number of the
    last span whose start_char <= offset (i.e. the span that contains it).
    """
    starts = [s[0] for s in page_spans]
    idx = bisect.bisect_right(starts, offset) - 1
    idx = max(0, min(idx, len(page_spans) - 1))
    return page_spans[idx][2]
