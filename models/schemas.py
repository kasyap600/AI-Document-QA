from dataclasses import dataclass


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class TextChunk:
    text: str
    doc_name: str
    start_page: int
    end_page: int
    chunk_index: int


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float
