# HybridIR: Dual-Index Personal Search Engine

HybridIR is a local-first search app for a folder of personal notes. It combines BM25 lexical retrieval with TF-IDF cosine similarity and fuses their rankings with Reciprocal Rank Fusion (RRF). The Streamlit interface lets you choose a retrieval mode, set the result count, and rebuild the index. Search covers both document titles and content, with title and phrase matches receiving a small relevance boost.

## Requirements

- Python 3.9 or newer
- Packages listed in `requirements.txt`

## Install and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

By default, the app indexes files under `data/` and stores its generated cache in `.index/`. Both locations can be changed with environment variables:

```powershell
$env:DATA_DIR = "C:\path\to\notes"
$env:INDEX_DIR = "C:\path\to\index-cache"
python -m streamlit run app.py
```

The app recursively reads `.txt`, `.md`, `.rst`, `.log`, `.csv`, `.json`, `.html`, and `.htm` files using the Python standard library. PDF and DOCX readers are optional; install the packages for the formats you need:

```powershell
python -m pip install pdfminer.six python-docx
```

The PDF and DOCX formats require `pdfminer.six` and `python-docx`, respectively. Sentence-transformer embeddings are also optional and are not enabled by the current Streamlit app, which uses TF-IDF for predictable local startup.

## Search modes

- `bm25`: lexical ranking using BM25Plus, including useful scores for very small collections. Results must share at least one token with the query.
- `vector`: TF-IDF vectors ranked by cosine similarity. This is a lexical vector-space baseline, not a neural semantic model.
- `hybrid`: combines the BM25 and TF-IDF rankings with RRF (`k=60`).
- Queries can match either a note's title or body. Exact title and phrase matches are given a modest ranking boost.

## Project layout

```text
app.py                 Streamlit UI and runtime configuration
src/indexer.py         Document loading and index construction/persistence
src/retriever.py       Search modes and reciprocal-rank fusion
data/                  Small example note collection
tests/test_search.py   Unit tests for search and index reload behavior
PROJECT_REPORT.md      Architecture, verification, and deployment notes
requirements.txt       Runtime dependencies
```

## Verify

Run the focused unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

The tests use temporary documents and a temporary index directory; they do not modify the sample notes or leave generated index files in the repository. They cover supported text formats, HTML text extraction, title-aware ranking, input validation, index reload, and cache rebuilding.

## Deployment notes

This is a single-process Streamlit application intended for trusted, private use. Run `python -m streamlit run app.py` from the project root for a local demo; it uses `data/` and `.index/` by default. Set `DATA_DIR` and `INDEX_DIR` to use a different notes folder or persistent cache. The app has no authentication or per-user data isolation. Do not expose it publicly without adding access controls. The `.index/` cache is generated locally and is intentionally excluded from version control; do not load pickle index files from untrusted sources.
