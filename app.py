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

st.title("🔍 HybridIR — Dual-Index Personal Search Engine")
st.caption("BM25 lexical index + vector index fused with Reciprocal Rank Fusion (RRF).")


@st.cache_resource
def get_index():
    idx = DualIndex(data_dir=DATA_DIR, index_dir=INDEX_DIR, use_embeddings=False)
    idx.ensure_loaded()
    return idx


idx = get_index()
retriever = Retriever(idx)

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Search mode", ["hybrid", "bm25", "vector"], index=0)
    top_k = st.slider("Top K", 3, 30, 10)
    st.markdown("---")
    st.write(f"Indexed documents: **{idx.num_docs}**")
    if st.button("Rebuild index"):
        with st.spinner("Indexing..."):
            idx.build()
            st.cache_resource.clear()
        st.success("Index rebuilt.")
        st.rerun()

query = st.text_input("Search your notes", placeholder="e.g. machine learning notes")

if query:
    results = retriever.search(query, mode=mode, top_k=top_k)
    st.subheader(f"Results ({len(results)})")
    if not results:
        st.info("No results found.")
    for r in results:
        d = r["doc"]
        with st.container(border=True):
            st.markdown(f"**{d['title']}**  \n*score: {r['score']:.4f} · source: {r['source']}*")
            snippet = d["text"][:400].replace("\n", " ")
            st.write(snippet + ("..." if len(d["text"]) > 400 else ""))
            st.caption(d["path"])
else:
    st.info("Enter a query to search.")
