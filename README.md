
- **Ingestion** (`ingestion/`): extracts text page-by-page from PDFs, cleans print artifacts, splits into semantic (paragraph-aware) chunks with overlap, embeds via OpenAI's `text-embedding-3-small` (through OpenRouter), indexes with FAISS.
- **Retrieval** (`rag/retriever.py`): hybrid search combining dense embedding similarity and BM25 keyword search, fused via **Reciprocal Rank Fusion (RRF)** rather than a naive weighted score blend (see *Known Limitations* for why).
- **Generation** (`rag/generator.py`): prompts an LLM (`gpt-4o-mini` via OpenRouter) to answer *only* from retrieved context, with explicit "I don't know" fallback instructions and numbered citations.
- **API** (`api/main.py`): FastAPI backend exposing `/ask` and `/health`.
- **UI** (`app/streamlit_chat.py`): Streamlit chat interface calling the API over HTTP.
- **Evaluation** (`eval/`): a hand-built 8-question test set with retrieval precision@K, keyword-match, and LLM-as-judge faithfulness scoring.
- **Streaming responses**: answers stream token-by-token from the LLM through a custom FastAPI streaming endpoint to the Streamlit UI, reducing perceived latency (citations are sent as a delimited final block once the answer text completes).

## Tech stack

Python, FastAPI, Streamlit, FAISS, `rank-bm25`, OpenAI-compatible embeddings/chat via OpenRouter, `pdfplumber`, `tiktoken`.

## Setup

1. Clone the repo and create a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
```
2. Create a `.env` file in the project root:

3. Add 10-K PDFs to `data/raw_pdfs/`.
4. Build the index:
```bash
   python ingestion/embed_and_index.py
```
5. Run the API (terminal 1):
```bash
   uvicorn api.main:app --reload
```
6. Run the UI (terminal 2):
```bash
   streamlit run app/streamlit_chat.py
```

> Docker configuration (`Dockerfile`, `docker-compose.yml`) is included but untested in this environment due to local disk space constraints. The steps above run the full system locally without Docker.

## Evaluation results

Run via `python eval/retrieval_eval.py` against an 8-question hand-labeled test set:

| Metric | Score |
|---|---|
| Retrieval Precision@5 | 87.5% |
| Answer Keyword Match | 87.5% |
| Avg. LLM-Judge Faithfulness | 4.5 / 5 |

**Note on methodology:** initial ground-truth page numbers in the test set were partially incorrect (based on assumptions about document structure rather than verified content). Investigating two apparent failures revealed the *retrieved answers were actually correct* — the test set's expected pages were wrong. Corrected after inspecting actual retrieved citations, which is a real part of building an eval set: your ground truth needs validation too, not just your system.

## Known limitations (and why)

- **BM25 keyword false-positives**: an early hybrid design (linear-weighted blend of normalized dense + BM25 scores) let a single literal keyword match dominate ranking — e.g. a query for "CEO" matched an unrelated exhibit-list page containing the phrase "CEO Performance Award Agreement," outranking the actual signature page. Fixed by switching to **Reciprocal Rank Fusion**, which combines rank position rather than raw score magnitude, making it far more robust to one-off score outliers.
- **Query phrasing sensitivity**: retrieval quality can still vary between semantically identical but differently-phrased queries (e.g. "Who is Apple's CEO?" vs. "Who is the CEO of Apple?") pulling in different top-K chunks. A more complete fix would be **query expansion** — generating 2-3 paraphrased versions of the query via the LLM and retrieving across all of them — not implemented here due to time, but the change is confined to `retriever.py`.
- **Typo sensitivity in pure dense search**: misspelled queries (e.g. "fascal" instead of "fiscal") degrade embedding similarity and can trigger the system's "I don't know" fallback even when relevant content exists. Hybrid search with BM25 mitigates this significantly, since correctly-spelled keywords elsewhere in the query still surface the right chunk.
- **Single-document corpus**: currently indexed against one Apple 10-K; the pipeline is designed to scale to multiple documents/companies without code changes (just add more PDFs and re-run ingestion), but hasn't been load-tested at scale.

## Interview-relevant design decisions

- Chunking is paragraph-aware with token-based overlap, not naive fixed-character splitting, to avoid cutting sentences mid-thought.
- Generation uses `temperature=0.1` deliberately — factual document Q&A should be consistent and grounded, not creative.
- The LLM is explicitly instructed to decline to answer rather than guess, which is the primary anti-hallucination lever in this system.
- Evaluation combines a cheap deterministic check (keyword match) with a more expensive LLM-as-judge faithfulness score, since keyword matching alone can't catch semantically wrong-but-keyword-matching answers.
## Live Demo
- **UI**: https://rag-doc-q.streamlit.app
- **API docs**: https://rag-doc-qa-h9vr.onrender.com/docs

> **Note:** the backend runs on Render's free tier, which spins down after ~15 minutes of inactivity. If the UI shows "API not reachable," the backend is likely asleep — visiting the API docs link above (or just retrying a question) will wake it within 30-60 seconds. This is a free-hosting-tier limitation, not an application bug.