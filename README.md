---
title: DEECA Data Discovery Assistant
emoji: 🌏
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
app_file: src/gradio_app.py
pinned: false
---

# DEECA Data Discovery Assistant

A natural-language search tool for Victoria's DEECA (Department of Energy,
Environment and Climate Action) open data catalog. Instead of browsing
~1,220 datasets by title or digging through CKAN's tag filters, a user types
a plain-English request — *"biodiversity data for the Dandenongs"*,
*"fire history in Victoria after 2010"* — and gets back the datasets that
actually match, with real access details and direct download links.

**Live app:** https://noelbensonswarna-deeca-data-discovery.hf.space

---

## The idea

DEECA's public catalog is real, large, and genuinely useful — but it's
organized the way a data publisher organizes things, not the way a person
asking a question thinks. Tags are inconsistent ("Inland waters" vs
"water"), titles are technical, and there's no way to ask "what fire data do
you have?" and get a ranked answer.

This project sits on top of that catalog as a **retrieval-augmented search
layer**: an LLM turns a loose question into structured search parameters,
a vector search finds the datasets that are actually relevant, and the
result is rendered as something a human can scan in seconds — not a wall of
raw CKAN JSON.

It is explicitly *not* a chatbot that makes things up about the data. Every
fact shown (title, description, access level, formats, URL) comes straight
from the catalog record. The LLM's job is understanding the question and
writing a one-line heading — never inventing dataset details.

---

## How a query actually flows through the system

```
 User types a question
        │
        ▼
 1. Intent parsing (Claude)
    "fire data in Victoria after 2010"
       → { data_type: ["fire"], region: ["Victoria"],
           constraints: { date_from: "2010" } }
        │
        ▼
 2. Semantic retrieval (pgvector, cosine distance)
    Query text is embedded locally (sentence-transformers),
    compared against every dataset's stored embedding —
    exact nearest-neighbor search, not approximate, at this scale.
        │
        ▼
 3. Hybrid re-ranking (Python)
    Vector similarity is blended with keyword/category/region
    scoring so results are both semantically relevant AND
    grounded in the catalog's actual tags.
        │
        ▼
 4. Deterministic rendering (no LLM)
    Top 5 results are formatted into cards — heading, metadata
    table, real description, direct link — straight from the
    database row. Nothing here is generated text.
        │
        ▼
 Gradio chat window
```

Two design decisions worth calling out:

- **Conversation memory is real, not paraphrased.** A follow-up like "only
  bedrock" needs to know the previous turn asked about water. Rather than
  writing a text summary of prior turns and hoping Claude reads it
  correctly, the app replays Claude's own actual prior JSON reply as a real
  conversation turn (bounded to the last 3 turns, so it can't grow
  unbounded and blow the token budget).

---

## How the data is handled

**Source.** Every dataset comes from Victoria's public CKAN API
(`discover.data.vic.gov.au`), filtered to the DEECA organization. No
scraping, no manual curation — `src/data_fetcher.py` calls
`package_search`, paginating past CKAN's 1,000-row-per-request cap to pull
the full ~1,220-dataset catalog.

**Normalization.** Each CKAN record is mapped into a flat schema: title,
description, owner, category tags, coverage, access level, available
formats, and URL. Two fields are honest placeholders rather than real data,
worth knowing about if you extend this:
- `access_level` is hardcoded to `PUBLIC` — CKAN doesn't expose DEECA's real
  OFFICIAL/PROTECTED/SECRET classification through this API.
- `coverage` is hardcoded to `["Victoria"]` for every row — there's no
  reliable sub-region field (Gippsland, Murray, etc.) in the source data, so
  regional relevance currently comes entirely from the semantic/keyword
  matching step, not a structured filter.

**Embedding.** Each dataset's `title + description` is embedded once at
load time using `sentence-transformers` (`all-MiniLM-L6-v2`) — a small,
free, local model, chosen specifically so semantic search works without
needing a second paid API. The 384-dim output is zero-padded to 1536 dims
to keep the schema compatible with a hosted embedding model later, if
higher-quality embeddings are ever needed.

**Storage.** Datasets and their embeddings live in Postgres with the
`pgvector` extension (`vector(1536)` column, cosine distance operator).
The live deployment uses a hosted Supabase Postgres instance; anything
Postgres + pgvector works locally too.

**Retrieval.** At query time, the retrieval is an *exact* nearest-neighbor
scan (`ORDER BY embedding <=> query_embedding`) — not an approximate index.
At ~1,220 rows that's sub-10ms and simpler to reason about than tuning an
`ivfflat`/`hnsw` index; it's the first thing to add if the catalog grows
into the tens of thousands of rows.

---

## Project structure

```
src/
  config.py            Environment/config loading
  database.py           SQLAlchemy models + pgvector setup
  embeddings.py          Local embedding generation (sentence-transformers)
  data_fetcher.py         CKAN fetch, pagination, normalization, load
  intent_parser.py         Claude call #1: query → structured intent
  semantic_search.py        Vector search + hybrid re-ranking
  synthesizer.py            Deterministic dataset-card rendering
  gradio_app.py              Chat UI, session state, wiring
  diagnose_continuity.py      Standalone script to verify multi-turn context
tests/                  pytest suite (intent parsing, search, rendering)
data/sample_queries.txt  Example queries for manual testing
```

---

## Running it locally

1. **Database.** Either run Postgres locally with `pgvector` installed, or
   point at a hosted instance (e.g. Supabase, which has a one-click
   `vector` extension toggle). If using Supabase, use the **pooled**
   connection string (port 5432/6543 via `pooler.supabase.com`) — the
   direct connection is IPv6-only and won't work from IPv4-only networks.
2. **Configure.** Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`
   and `DATABASE_URL`.
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Load the catalog:**
   ```bash
   python -c "from src.data_fetcher import populate_database; populate_database(rows=2000)"
   ```
   (`rows=2000` fetches the full catalog, paginating automatically; a
   smaller number is useful for a quick local test.)
5. **Run the app:**
   ```bash
   python -m src.gradio_app
   ```
   Open http://localhost:7860.
6. **Run the tests:**
   ```bash
   pytest tests/ -v
   ```
7. **Sanity-check multi-turn context** without touching the UI:
   ```bash
   python -m src.diagnose_continuity
   ```

---

## Known limitations

Worth being upfront about, since they shape what results look like:

- **No real sub-region filtering.** "Gippsland" or "the Dandenongs" rely
  entirely on the embedding/keyword match against descriptions — there's no
  structured geographic field to filter on, since the source data doesn't
  provide one.
- **No real access-level distinction.** Everything reads as `PUBLIC`
  because that's all the CKAN API exposes here, even though DEECA's actual
  classification scheme has more tiers.
- **No real "dataset year."** CKAN's `metadata_modified` is a catalog
  *refresh* timestamp, not when the underlying data was collected — a
  dataset literally titled "2010 Index of Stream Condition" can show a 2026
  "updated" date. The UI deliberately doesn't label this as a "year" to
  avoid misrepresenting decade-old data as current.
- **Date/recency constraints are parsed but not enforced.** The intent
  parser can extract "after 2010" into a structured constraint, but
  `semantic_search.py` doesn't currently filter or score by it — there's no
  reliable per-dataset year field to filter against yet.

## Deployment

The live instance runs on **Hugging Face Spaces** (Gradio SDK, requires a
paid plan as of 2026) with **Supabase** as the Postgres/pgvector host.
`ANTHROPIC_API_KEY` and `DATABASE_URL` are set as Space secrets, not
committed to the repo. The app is hardcoded to port 7860, which is the
Gradio SDK's expected default on HF Spaces — deploying to a host that
assigns its own port (e.g. Render) requires reading `$PORT` instead.
