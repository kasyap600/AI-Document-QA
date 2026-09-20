import fitz  # PyMuPDF

from models.schemas import PageText


def extract_pages(pdf_bytes: bytes, filename: str) -> list[PageText]:
    """
    Opens a PDF from raw bytes and returns one PageText per non-empty page.
    Page numbers are 1-indexed.
    Raises ValueError if no text is extracted (likely a scanned/image-only PDF).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(PageText(page_number=page_num, text=text))

    doc.close()

    if not pages:
        raise ValueError(
            f"No text could be extracted from '{filename}'. "
            "The file may be a scanned or image-only PDF."
        )

    return pages
