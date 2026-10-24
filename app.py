"""Streamlit web interface for HybridIR dual-index personal search engine."""

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.indexer import DualIndex
from src.retriever import Retriever

DATA_DIR = os.environ.get("DATA_DIR", str(ROOT / "data"))
INDEX_DIR = os.environ.get("INDEX_DIR", str(ROOT / ".index"))

st.set_page_config(page_title="HybridIR Search", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: #f5f7fb; padding: 1rem; border-radius: 0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 HybridIR")
st.caption("Your private notes, searchable with lexical and vector retrieval.")


@st.cache_resource
def get_index():
    idx = DualIndex(data_dir=DATA_DIR, index_dir=INDEX_DIR, use_embeddings=False)
    idx.ensure_loaded()
    return idx


idx = get_index()
retriever = Retriever(idx)

with st.sidebar:
    st.header("Search settings")
    mode = st.selectbox("Search mode", ["hybrid", "bm25", "vector"], index=0)
    top_k = st.slider("Top K", 3, 30, 10)
    st.markdown("---")
    st.metric("Indexed documents", idx.num_docs)
    st.caption("Hybrid combines BM25 and TF-IDF using Reciprocal Rank Fusion.")
    with st.expander("Supported files"):
        st.write("TXT, Markdown, reStructuredText, logs, CSV, JSON, and HTML")
        st.caption("PDF and DOCX are available with their optional reader packages.")
    if st.button("Rebuild index"):
        with st.spinner("Indexing..."):
            idx.build()
            st.cache_resource.clear()
        st.success("Index rebuilt.")
        st.rerun()

with st.form("search_form"):
    query = st.text_input(
        "Search your notes",
        placeholder="Try a topic, phrase, or document title",
        help="Searches document titles and content.",
    )
    submitted = st.form_submit_button("Search", type="primary")

if submitted and query:
    results = retriever.search(query, mode=mode, top_k=top_k)
    st.subheader(f"{len(results)} result{'s' if len(results) != 1 else ''}")
    if not results:
        st.info("No matching notes. Try different keywords or rebuild the index.")
    for position, result in enumerate(results, start=1):
        document = result["doc"]
        with st.container(border=True):
            heading, score = st.columns([5, 1])
            with heading:
                st.write(f"**{position}. {document['title']}**")
            with score:
                st.metric("Score", f"{result['score']:.3f}")
            text = " ".join(document["text"].split())
            snippet = text[:400]
            st.write(snippet + ("..." if len(text) > 400 else ""))
            st.caption(f"{result['source'].upper()} · {document['path']}")
elif submitted:
    st.info("Enter a search query to find matching notes.")
else:
    st.info("Search across your local notes. Your documents stay on this machine.")
