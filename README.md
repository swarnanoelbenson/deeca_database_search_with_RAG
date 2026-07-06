# DEECA DataShare NLP

Natural language interface to Victoria's DataShare platform, built for DEECA
(Department of Energy, Environment and Climate Action). Ask for data in plain
English and get back the relevant datasets from DEECA's catalog with access
details and download links.

## Architecture

```
User Query (Gradio chat)
    -> Intent Parser (Claude API)
    -> Semantic Search (PostgreSQL + pgvector)
    -> RAG Synthesis (Claude API)
    -> Gradio Display
```

This is a RAG (Retrieval-Augmented Generation) pattern: real datasets are
retrieved via vector similarity search, then Claude synthesizes a natural
language explanation grounded in that retrieved data.

## Setup

1. Install PostgreSQL locally and enable the `pgvector` extension
   (`CREATE EXTENSION vector;` is run automatically by `init_db()`, but the
   extension binary must be installed on the Postgres server first —
   see https://github.com/pgvector/pgvector#installation).
2. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` and
   `DATABASE_URL`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Populate the dataset catalog from Victoria's CKAN API:
   ```bash
   python -m src.data_fetcher
   ```
5. Launch the app:
   ```bash
   python -m src.gradio_app
   ```
6. Open http://localhost:7860

## Project Structure

- `src/`: Core application code (config, database, embeddings, data fetcher,
  intent parser, semantic search, synthesizer, Gradio app)
- `data/`: Sample queries for manual testing
- `tests/`: Test suite (`pytest tests/`)

## How It Works

1. User asks a question in natural language.
2. `IntentParser` (Claude) extracts intent, data type, region, and
   constraints as structured JSON.
3. `SemanticSearch` embeds the query with `sentence-transformers` and ranks
   datasets by cosine similarity in pgvector, then re-ranks by
   category/region/format matches.
4. `DatasetSynthesizer` (Claude) turns the retrieved datasets into a helpful,
   grounded natural-language answer with links.

## Notes on implementation choices

- Embeddings use the free, local `sentence-transformers` model
  (`all-MiniLM-L6-v2`) rather than a hosted embeddings API, so semantic
  search actually reflects meaning without requiring an OpenAI key. Swap
  `generate_embedding()` in `src/embeddings.py` for a hosted API if you need
  higher-quality embeddings later.
- Ranking combines pgvector cosine similarity (semantic) with a keyword/tag
  scoring pass (category, region, access level, format) for interpretable,
  hybrid relevance.

## Deployment

- **Hugging Face Spaces** (free): push to GitHub, create a Space linked to
  the repo, add env vars in Space settings.
- **Railway** ($5-10/month): connect the repo; Railway auto-detects Python
  and can provision PostgreSQL.
