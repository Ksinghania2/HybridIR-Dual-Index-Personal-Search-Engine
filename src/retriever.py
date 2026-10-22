"""Retriever: search execution and Reciprocal Rank Fusion (RRF)."""

import os
import re
from typing import List, Dict, Any

import numpy as np

from src.indexer import DualIndex, tokenize

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def _bm25_search(idx: DualIndex, query: str, top_k: int) -> List[Dict[str, Any]]:
    if idx.bm25 is None:
        return []
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scores = idx.bm25.get_scores(q_tokens)
    order = np.argsort(scores)[::-1][:top_k]
    query_terms = set(q_tokens)
    out = []
    for i in order:
        if scores[i] <= 0 or not query_terms.intersection(idx.bm25_tokens[i]):
            continue
        d = idx.documents[i]
        out.append({"doc": d, "score": float(scores[i]), "rank": len(out) + 1})
    return out


def _vector_search(idx: DualIndex, query: str, top_k: int) -> List[Dict[str, Any]]:
    if idx.documents is None or not idx.documents:
        return []
    if idx.use_embeddings and idx.embeddings is not None and idx.st_model is not None:
        q_emb = idx.st_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = (idx.embeddings @ q_emb.T).ravel()
    elif idx.vectorizer is not None and idx.vector_matrix is not None:
        q_vec = idx.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, idx.vector_matrix).ravel()
    else:
        return []
    order = np.argsort(sims)[::-1][:top_k]
    out = []
    for i in order:
        if sims[i] <= 0:
            continue
        d = idx.documents[i]
        out.append({"doc": d, "score": float(sims[i]), "rank": len(out) + 1})
    return out


def reciprocal_rank_fusion(rankings: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists using RRF."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}
    for ranking in rankings:
        for item in ranking:
            did = item["doc"]["id"]
            doc_map[did] = item["doc"]
            r = item.get("rank", 0)
            if r == 0:
                # derive rank from position
                r = ranking.index(item) + 1
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + r)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for did, s in ordered:
        out.append({"doc": doc_map[did], "score": float(s)})
    return out


class Retriever:
    def __init__(self, idx: DualIndex):
        self.idx = idx

    def search(self, query: str, mode: str = "hybrid", top_k: int = 10) -> List[Dict[str, Any]]:
        self.idx.ensure_loaded()
        mode = (mode or "hybrid").lower()
        if mode == "bm25":
            res = _bm25_search(self.idx, query, top_k)
            return [{"doc": r["doc"], "score": r["score"], "source": "bm25"} for r in res]
        if mode == "vector":
            res = _vector_search(self.idx, query, top_k)
            return [{"doc": r["doc"], "score": r["score"], "source": "vector"} for r in res]
        # hybrid
        bm = _bm25_search(self.idx, query, top_k * 2)
        vec = _vector_search(self.idx, query, top_k * 2)
        fused = reciprocal_rank_fusion([bm, vec])
        out = []
        for r in fused[:top_k]:
            out.append({"doc": r["doc"], "score": r["score"], "source": "hybrid"})
        return out
