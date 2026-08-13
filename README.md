# local-document-qa

A local (as offline as possible) question-answering system for PDF/TXT documents.

## Purpose

Split the documents a user uploads into chunks, vectorize them with a local
embedding model, store those vectors in Qdrant, find the chunks most
relevant to the user's question (context), send them to a local LLM, and
present the generated answer together with its source file/page/chunk
information. Nothing is invented that isn't in the document content
(grounded / "I don't know" if there's no answer).

## Architecture (high level)

- `app/api` — FastAPI endpoints
- `app/core` — configuration, shared infrastructure
- `app/domain` — domain models / business rules
- `app/models` — Pydantic data schemas
- `app/repositories` — access to storage layers such as Qdrant
- `app/services` — chunking, embedding, retrieval, generation services
- `app/llm` — local LLM clients
- `ui` — Streamlit interface
- `tests/unit`, `tests/integration`, `tests/evaluation` — test layers
- `benchmark` — real measurement/comparison scripts (not a claim, a measurement)
- `finetuning` — only to be used once the baseline RAG + eval set are complete

## Development Status

**Phase 7 — FastAPI + Streamlit interface.** Document upload/list/delete,
chunking, real embedding (BAAI/bge-m3), Qdrant indexing, dense
retrieval (+ optional reranker), RAG answers from a local LLM (LM Studio,
plus an optional Gemini API path) with source verification all work
end-to-end; usable via the FastAPI endpoints and the Streamlit interface.

## Requirements

- Python 3.11
- Docker + Docker Compose (for Qdrant, and optionally for the API/UI containers)
- A local LLM server — tested in this project with [LM Studio](https://lmstudio.ai/)
  (OpenAI-compatible API, `http://localhost:1234`). Ollama or another
  OpenAI-compatible server can also be used via `LLM_BASE_URL`/`LLM_MODEL_ID`.
- (Optional, for real embedding/reranker models) `pip install -e ".[embeddings]"`
  — installs `sentence-transformers`/`torch`; on first run this also
  downloads the BAAI/bge-m3 model (~2.2GB) from HuggingFace.

## Setup — Manual (development)

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -e ".[dev,embeddings,ui]"
cp .env.example .env
docker compose up -d qdrant   # run only Qdrant in Docker
```

Start LM Studio (or another local LLM server) separately and load a
model; the `LLM_BASE_URL`/`LLM_MODEL_ID` in `.env` must match that server.

Run the API:

```bash
uvicorn app.main:app --reload
```

Run the Streamlit interface (in a separate terminal):

```bash
streamlit run ui/streamlit_app.py
```

By default the API is served at `http://localhost:8000` and the UI at
`http://localhost:8501`. The UI reads the API's address from the
`API_BASE_URL` environment variable (default `http://localhost:8000`).

## Setup — Docker Compose

```bash
docker compose up -d --build
```

This starts the Qdrant + API + Streamlit UI containers together. The API
container includes the real embedding model (`sentence-transformers`/`torch`);
on first run the BAAI/bge-m3 model is downloaded inside the container and
stored in the `huggingface_cache` volume (not re-downloaded on restarts).

**LLM server location:** the LLM server used in this project (LM Studio)
is not containerized — it runs on the host machine. `docker-compose.yml`
injects `LLM_BASE_URL=http://host.docker.internal:1234` into the API
container (to reach LM Studio on the host). If you want to run the LLM in
its own container, just change the `LLM_BASE_URL` environment variable to
that service's address — the host/container distinction is entirely
config-driven, no code changes are needed.

## Testing

```bash
pytest                    # unit tests (does not download real models)
pytest -m integration -s  # with real models (requires download + a running LM Studio)
ruff check .
mypy
```

## Direct Access to Qdrant (optional)

```bash
docker compose up -d qdrant
```

## Security Notes

- File size limits and MIME/extension validation are handled by
  `FileValidator` (`MAX_UPLOAD_SIZE_BYTES`).
- Uploaded files are never written to a filesystem path derived from user
  input (everything is processed in memory); the filename is also
  sanitized against path-traversal attempts via `sanitize_filename`.
- All Qdrant queries are scoped by a `document_id` filter; retrieval/QA
  for one document can never access another document's data.
- API errors always come back as a clean `{error_code, message}` JSON;
  a Python stack trace is never shown to the user.

## Note

This repository does **not** contain model weights, PDF documents,
Qdrant data, checkpoints, or real API keys. The `.env` file is not
committed; only `.env.example` is included as a reference.
