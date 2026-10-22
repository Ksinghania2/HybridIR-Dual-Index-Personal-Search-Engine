# HybridIR Project Report

**Report date:** 2026-10-22  
**Project type:** Local-first Python/Streamlit personal search application

## Purpose

HybridIR searches a local folder of notes using two complementary ranking methods. BM25 handles direct term matching, TF-IDF cosine similarity supplies a second vector-space ranking, and Reciprocal Rank Fusion combines the ranked lists in hybrid mode.

## Architecture

1. `app.py` configures Streamlit, reads `DATA_DIR` and `INDEX_DIR`, loads the index, and renders query, mode, result-count, and rebuild controls.
2. `src/indexer.py` recursively reads supported documents, tokenizes text, builds the BM25Plus and TF-IDF indexes, and persists the generated cache.
3. `src/retriever.py` executes BM25, vector, or hybrid search. Hybrid mode fuses rankings with RRF using the default constant `k=60`.
4. `data/` contains two small demonstration notes. Generated files are stored in `.index/`, which is excluded from version control.

## Supported content and behavior

- Required document formats: `.txt` and `.md`.
- Optional formats: `.pdf` via `pdfminer.six`, `.docx` via `python-docx`.
- The Streamlit app currently uses TF-IDF, not transformer embeddings. The indexer has an optional sentence-transformer path, but the UI initializes it disabled.
- BM25 uses BM25Plus so rare terms remain useful in small collections; a lexical result is returned only when it shares a token with the query.
- Search is local and has no external API dependency.

## Validation record

- Editor diagnostics: no errors reported in the Python source.
- Runtime smoke check: **passed** for document loading and BM25, TF-IDF, and hybrid queries against the included notes.
- BM25 negative-query check: **passed**; an unrelated query returned no results.
- Unit tests: **4 passed** with `python -m unittest discover -s tests -v`, covering all retrieval modes, lexical no-match behavior, saved-index reload, and automatic rebuild of an outdated cache.
- Streamlit startup/browser check: **passed**; the app rendered at `http://localhost:8501`, displayed two indexed documents, and returned the expected result for a hybrid query.

## Deployment assessment

The app is suitable for a trusted, private deployment after the final test and startup checks pass. It does not provide authentication, authorization, tenant isolation, or an upload workflow. Do not expose it publicly without adding access controls. Use a writable persistent location for `INDEX_DIR` if indexes should survive restarts. The index cache uses Python pickle; treat it as trusted local data and never load a cache supplied by an untrusted party.

## Repository and release status

The source folder currently has no Git repository metadata or configured GitHub remote. A push cannot be completed until the project is connected to a repository. Keep `.venv/`, `.index/`, `.env`, and Python cache files out of version control. The included sample notes are generic demonstration content; review any replacement personal notes before publishing them.