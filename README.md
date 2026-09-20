# AI Document Q&A

Upload a PDF and ask questions in plain English. Get grounded answers with exact page citations and a confidence indicator.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-orange) ![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-purple)

---

## How it works

```
PDF upload
   ↓
Text extraction (PyMuPDF)          — preserves page numbers
   ↓
Chunking (sliding window)          — 400 chars, 80 char overlap
   ↓
Embeddings (all-MiniLM-L6-v2)     — local, no API key needed
   ↓
Qdrant (in-memory vector store)
   ↓
Retrieval (cosine similarity)
   ↓
Gemini 2.5 Flash                   — answer grounded in retrieved chunks
   ↓
Answer  +  Page citation  +  Confidence (High / Medium / Low)
```

Confidence is derived from the embedding similarity score of the best-matching chunk — not from the LLM's self-assessment, which is unreliable.

---

## Example

**Question:** What is the refund policy?

**Answer:** Customers can request a refund within 30 days of purchase.

**Source:** Page 14 &nbsp; 🟢 High confidence

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <repo-url>
cd AI-Document-QA
python -m venv .venv && source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

> The `all-MiniLM-L6-v2` embedding model (~80 MB) downloads automatically on first run and is cached locally.

**3. Add your Google AI Studio API key**

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your real key
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com).

**4. Run**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project structure

```
AI-Document-QA/
├── app.py                  # Streamlit UI
├── config.py               # Tunable constants (chunk size, model, thresholds)
├── requirements.txt
├── .env.example
│
├── pipeline/
│   ├── extractor.py        # PDF → list[PageText]
│   ├── chunker.py          # Sliding window with page-boundary tracking
│   ├── embedder.py         # SentenceTransformer wrapper
│   ├── vectorstore.py      # Qdrant in-memory CRUD + search
│   └── retriever.py        # Query → ranked results + prompt formatter
│
├── llm/
│   └── gemini_client.py    # Gemini API wrapper with grounded system prompt
│
├── models/
│   └── schemas.py          # Shared dataclasses: PageText, TextChunk, SearchResult
│
└── utils/
    └── confidence.py       # Cosine score → High / Medium / Low
```

---

## Configuration

All knobs are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 400 | Characters per chunk |
| `CHUNK_OVERLAP` | 80 | Overlap between adjacent chunks |
| `TOP_K` | 5 | Chunks retrieved per query |
| `GEMINI_MODEL` | `gemini-2.5-flash` | LLM model |
| `CONFIDENCE_HIGH` | 0.45 | Min score for "High" label |
| `CONFIDENCE_MEDIUM` | 0.30 | Min score for "Medium" label |

---

## Limitations

- **Image-only PDFs** (scanned documents) are not supported — text extraction requires a text-layer PDF.
- The vector store is **in-memory**: data resets when the app restarts. Change `QdrantClient(":memory:")` to `QdrantClient("./qdrant_data")` in `pipeline/vectorstore.py` for persistence.
- One document at a time — uploading a new PDF replaces the previous index.
