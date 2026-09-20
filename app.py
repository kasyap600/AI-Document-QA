import streamlit as st

from llm.gemini_client import GeminiClient
from pipeline.chunker import chunk_pages
from pipeline.embedder import Embedder
from pipeline.extractor import extract_pages
from pipeline.retriever import format_source_citation, retrieve
from pipeline.vectorstore import VectorStore
from utils.confidence import compute_confidence

st.set_page_config(page_title="AI Document Q&A", page_icon="📄", layout="wide")

st.markdown("""
<style>
/* ── page background ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* ── typography ── */
h1 { font-size: 1.8rem !important; font-weight: 700 !important; color: #e6edf3 !important; }
label, p, li, div { color: #c9d1d9 !important; }

/* ── answer card ── */
.answer-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #238636;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin: 0.8rem 0 1.2rem;
    font-size: 1.05rem;
    line-height: 1.6;
    color: #e6edf3 !important;
}

/* ── meta row (source + confidence badges) ── */
.meta-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid transparent;
}
.badge-source  { background: #1c2d3e; border-color: #388bfd; color: #79c0ff !important; }
.badge-high    { background: #122d20; border-color: #238636; color: #56d364 !important; }
.badge-medium  { background: #2d2000; border-color: #9e6a03; color: #d29922 !important; }
.badge-low     { background: #2d0c0c; border-color: #da3633; color: #ff7b72 !important; }

/* ── history entry ── */
.history-q {
    font-size: 0.88rem;
    color: #8b949e !important;
    margin-bottom: 0.2rem;
}
.history-a {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #238636;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.9rem;
    color: #c9d1d9 !important;
    margin-bottom: 0.4rem;
}
.history-meta { font-size: 0.78rem; color: #6e7681 !important; margin-bottom: 1.2rem; }

/* ── chunk expander text ── */
.chunk-text {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-family: monospace;
    font-size: 0.82rem;
    color: #8b949e !important;
    white-space: pre-wrap;
    margin-bottom: 0.6rem;
}

/* ── file uploader & buttons ── */
[data-testid="stFileUploader"] { border-radius: 8px; }
[data-testid="baseButton-primary"] {
    background: #238636 !important;
    border-color: #2ea043 !important;
    color: #fff !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
    font-size: 1rem !important;
    padding: 0.6rem 0.8rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 2px rgba(56,139,253,.25) !important;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


@st.cache_resource
def get_gemini() -> GeminiClient:
    return GeminiClient()


def ingest(pdf_bytes: bytes, filename: str, embedder: Embedder, store: VectorStore) -> tuple[int, int]:
    pages = extract_pages(pdf_bytes, filename)
    chunks = chunk_pages(pages, filename)
    embeddings = embedder.encode_batch([c.text for c in chunks])
    store.reset_collection()
    store.upsert_chunks(chunks, embeddings)
    return len(pages), len(chunks)


def confidence_badge(label: str) -> str:
    icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[label]
    css = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[label]
    return f'<span class="badge {css}">{icon} {label} confidence</span>'


# ── sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded:
        st.markdown(f"**{uploaded.name}**  \n`{uploaded.size / 1024:.1f} KB`")
        if st.button("Process PDF", use_container_width=True, type="primary"):
            with st.spinner("Indexing…"):
                try:
                    n_pages, n_chunks = ingest(
                        uploaded.read(), uploaded.name, get_embedder(), get_store()
                    )
                    st.session_state.update({
                        "doc_ready": True,
                        "doc_name": uploaded.name,
                        "doc_pages": n_pages,
                        "doc_chunks": n_chunks,
                        "history": [],
                    })
                except ValueError as e:
                    st.error(str(e))
                    st.session_state["doc_ready"] = False

    if st.session_state.get("doc_ready"):
        st.success("Ready")
        st.markdown(
            f"**File:** {st.session_state['doc_name']}  \n"
            f"**Pages:** {st.session_state['doc_pages']}  \n"
            f"**Chunks:** {st.session_state['doc_chunks']}"
        )
        if st.button("Clear document", use_container_width=True):
            for k in ("doc_ready", "doc_name", "doc_pages", "doc_chunks", "history"):
                st.session_state.pop(k, None)
            st.rerun()



# ── main panel ──────────────────────────────────────────────────────────────────

st.markdown("# 🔍 AI Document Q&A")

doc_ready = st.session_state.get("doc_ready", False)

if not doc_ready:
    st.markdown(
        """
        <div style='text-align:center;padding:3rem 1rem;color:#8b949e'>
            <div style='font-size:3rem'>📄</div>
            <div style='font-size:1.1rem;margin-top:0.5rem'>Upload a PDF in the sidebar to get started</div>
            <div style='font-size:0.88rem;margin-top:0.4rem'>
                Ask any question — get grounded answers with page citations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    question = st.text_input(
        "question",
        placeholder="Ask a question about the document…",
        label_visibility="collapsed",
    )

    if question:
        with st.spinner("Thinking…"):
            results = retrieve(question, get_embedder(), get_store())
            answer = get_gemini().answer(question, results)
            confidence = compute_confidence(results[0].score)
            source = format_source_citation(results[0])

        # answer card
        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

        # source + confidence badges
        src_badge = f'<span class="badge badge-source">📄 {source}</span>'
        conf_badge = confidence_badge(confidence)
        st.markdown(f'<div class="meta-row">{src_badge}{conf_badge}</div>', unsafe_allow_html=True)

        # retrieved chunks expander
        with st.expander(f"Retrieved chunks ({len(results)})"):
            for i, r in enumerate(results, start=1):
                citation = format_source_citation(r)
                st.markdown(
                    f"**Chunk {i}** — {citation} &nbsp;"
                    f'<span style="color:#6e7681;font-size:0.8rem">score={r.score:.3f}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="chunk-text">{r.chunk.text}</div>', unsafe_allow_html=True)

        # save to history
        history = st.session_state.setdefault("history", [])
        history.insert(0, {"q": question, "a": answer, "source": source, "confidence": confidence})

    # history
    history = st.session_state.get("history", [])
    if len(history) > 1:
        st.markdown("---")
        st.markdown("#### Previous questions")
        for entry in history[1:]:
            conf_css = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[entry["confidence"]]
            st.markdown(f'<div class="history-q">Q: {entry["q"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="history-a">{entry["a"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="history-meta">'
                f'📄 {entry["source"]} &nbsp;·&nbsp; '
                f'<span class="{conf_css}">{entry["confidence"]} confidence</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
