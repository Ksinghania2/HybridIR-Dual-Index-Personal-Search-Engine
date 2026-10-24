# HybridIR Project Report

**Report date:** 2026-10-24
**Project type:** Local-first Python/Streamlit personal search application

## Purpose

HybridIR searches a local folder of notes using two complementary ranking methods. BM25 handles direct term matching, TF-IDF cosine similarity supplies a second vector-space ranking, and Reciprocal Rank Fusion combines the ranked lists in hybrid mode.

## Architecture

1. `app.py` configures Streamlit, reads `DATA_DIR` and `INDEX_DIR`, loads the index, and renders the search form, mode, result-count, and rebuild controls.
2. `src/indexer.py` recursively reads supported documents, includes titles in the searchable text, builds BM25Plus and TF-IDF indexes, and persists the generated cache.
3. `src/retriever.py` executes BM25, vector, or hybrid search. Exact title and phrase matches receive a relevance boost; hybrid mode fuses rankings with RRF using the default constant `k=60`.
4. `data/` contains two small demonstration notes. Generated files are stored in `.index/`, which is excluded from version control.

## Supported content and behavior

- Standard-library formats: `.txt`, `.md`, `.rst`, `.log`, `.csv`, `.json`, `.html`, and `.htm`.
- Optional formats: `.pdf` via `pdfminer.six`, `.docx` via `python-docx`. DOCX paragraph and table text is indexed.
- The Streamlit app currently uses TF-IDF, not transformer embeddings. The indexer has an optional sentence-transformer path, but the UI initializes it disabled.
- BM25 uses BM25Plus so rare terms remain useful in small collections; a lexical result is returned only when it shares a token with the query.
- Search is local and has no external API dependency.

## Validation record

- Unit tests: **8 passed** with `python -m unittest discover -s tests -v`, covering all retrieval modes, lexical no-match behavior, title-aware ranking, supported file parsing, input validation, saved-index reload, cache rebuilding, and clearing an index after rebuilding an empty collection.
- Python compile check: **passed** with `python -m compileall -q app.py src tests`.
- Streamlit startup smoke check: **passed**; the local health endpoint and app page responded successfully.

## Deployment assessment

The app is suitable for a trusted, private deployment after the final test and startup checks pass. It does not provide authentication, authorization, tenant isolation, or an upload workflow. Do not expose it publicly without adding access controls. Use a writable persistent location for `INDEX_DIR` if indexes should survive restarts. The index cache uses Python pickle; treat it as trusted local data and never load a cache supplied by an untrusted party.

## Repository and release status

The project is on the `main` branch and configured with the GitHub remote `https://github.com/Ksinghania2/HybridIR-Dual-Index-Personal-Search-Engine.git`. The updated unit suite passes (8 tests). Keep `.venv/`, `.index/`, `.env`, and Python cache files out of version control. Review any replacement personal notes before publishing them.