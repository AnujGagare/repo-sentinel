# Repo Sentinel

A RAG assistant that answers questions about a codebase and **stays in sync
as the codebase changes** — instead of indexing a static snapshot once, it
incrementally re-indexes only the files that changed after every commit, so
answers always reflect current code.

Demo corpus: [FastAPI](https://github.com/fastapi/fastapi) source + docs
(swap in any Python repo — see Configuration below).

## Why this project (not just another "chat with your docs" demo)

- **Incremental indexing, not full rebuilds.** A git-diff watcher detects
  exactly which files changed since the last index and only re-chunks /
  re-embeds those — verified in this repo (both by an automated test and a
  live run) to correctly isolate a single changed file out of thousands of
  tracked files after one commit.
- **AST-based code chunking**, not fixed-size text windows — chunks are
  whole functions/classes/methods, so retrieval never returns a function
  cut in half. Classes with many near-duplicate methods (e.g. a router's
  `get`/`post`/`put`/`patch`/... handlers, which often share one docstring
  template) are automatically detected via docstring-similarity clustering
  and grouped, instead of flooding retrieval with near-identical chunks.
- **A code-specific embedding model**
  (`jinaai/jina-embeddings-v2-base-code`), not a general sentence-similarity
  model. This was a deliberate, evidence-based choice — see "Embedding
  model" below for the diagnosis that led to it.
- **Hybrid retrieval**: BM25 (exact identifier matches) + embeddings
  (semantic matches), merged with reciprocal rank fusion, then a
  cross-encoder reranker narrows the top candidates before they reach the
  LLM.
- **Auto-generated summaries for undocumented functions.** Functions with
  no docstring get a short, cached, LLM-generated one-line summary added to
  their indexed text — directly targeting a real retrieval failure mode
  found via this project's own eval harness (see "Eval-driven debugging"
  below).
- **A real eval harness**, not vibes — a curated question set with known
  correct source chunks, scored automatically for retrieval recall, plus a
  `/debug_retrieve` endpoint that shows results at each pipeline stage
  separately (BM25-only / vector-only / fused / reranked) so a retrieval
  failure can be root-caused instead of guessed at.
- **Grounded, cited answers** — every answer must cite `file.py:line-line`
  for its claims; the LLM is instructed to say "I don't know" rather than
  guess if the retrieved context is insufficient.
- **Production-hardened, not just a working demo**: validated config,
  structured logging, retry logic with backoff, API-key auth + rate
  limiting, a real automated test suite, CI, and a Dockerfile. See
  "Production considerations" below for specifics and honest limitations.

## Architecture

```
Git repo
   │
   ▼
GitWatcher (git diff since last indexed commit)
   │  → only changed files
   ▼
AST code chunker (one chunk per function/class/method;
near-duplicate methods grouped by docstring similarity)
   │
   ▼
Embedder (jina-embeddings-v2-base-code, local, CPU) ──► ChromaDB (vector search)
   │
   └── undocumented functions get an auto-generated,
       cached one-line summary added to their indexed text
                                                          │
   ┌──────────────────────────────────────────────────────┘
   ▼
BM25 index (keyword search)
                                                          │
Query ──► [BM25 search] + [vector search] ──► Reciprocal Rank Fusion
                                                          │
                                                  Cross-encoder reranker
                                                          │
                                                  Top 5 chunks + citations
                                                          │
                                          LLM (Ollama locally / Groq deployed,
                                          with retry-with-backoff on transient failures)
                                                          │
                                                  Grounded, cited answer
```

## Embedding model: why jina-embeddings-v2-base-code, not a general sentence model

This project started with `all-MiniLM-L6-v2` (small, fast, general-purpose).
Its own eval harness caught a real problem with that choice: a genuine,
correct function (`solve_dependencies`, FastAPI's core dependency-resolution
logic) was **unretrievable** for the plainly-worded question "how does
FastAPI resolve dependencies" — consistently missing from the top 40
candidates across multiple retrieval-depth and chunking fixes.

Root-caused via the `/debug_retrieve` diagnostic endpoint down to two
compounding causes:

1. **Token truncation.** `all-MiniLM-L6-v2` has a 256-token limit. The
   target function is ~400 words; its parameter list and type hints alone
   consumed the entire token budget before the embedding model ever saw the
   function's actual logic.
2. **Wrong training objective.** General sentence-similarity models are
   trained on natural-language sentence pairs, not on aligning code with
   natural-language descriptions of what that code does. No amount of
   context window would fix a model that was never trained to make that
   connection.

`jina-embeddings-v2-base-code` addresses both: an 8192-token context window
(no realistic function gets truncated), and training specifically on
code-to-natural-language pairs. This is also standard practice, not a novel
idea — production code search tools use code-trained embeddings, not
general sentence models, for exactly this reason.

**Practical note:** if you're upgrading an existing index from the earlier
MiniLM-based version, you must fully rebuild it — delete `data/index` and
let `full_index()` run again. Mixing 384-dim (MiniLM) and 768-dim
(jina-code) vectors in one collection produces silently meaningless
nearest-neighbor results, not an error.

**A real deployment lesson, not just a local-dev one:** the code-specific
model above is the right choice for local development, but it doesn't fit
Render's free-tier memory limit. Confirmed by an actual failed deployment
— the process was OOM-killed (`exit 137`, Linux's SIGKILL) partway
through startup, because the model's weights plus PyTorch's runtime
overhead exceeded Render's hard 512MB RAM cap on its own, before the rest
of the app even loaded. The fix is `REPO_SENTINEL_EMBEDDING_MODEL`
(`src/config.py`, `src/indexing/embedder.py`) — configurable per
environment, the same pattern already used for `REPO_SENTINEL_LLM_BACKEND`
(Ollama locally, Groq deployed). `render.yaml` sets it to
`all-MiniLM-L6-v2` for the deployed version. This is a genuine, stated
tradeoff — the deployed demo has weaker embedding relevance than local
dev, in exchange for actually fitting in free-tier memory — not something
worth hiding.

## Eval-driven debugging: what actually happened

Worth documenting honestly rather than presenting a clean number with no
story — the real sequence, in order:

1. **0% recall** — turned out to be a path-prefix bug in the eval set
   (`background.py` vs. the indexer's actual `fastapi/background.py`), not
   a retrieval failure. Fixed the eval set, not the retrieval code.
2. **0% again** — a second, different path bug after fixing an unrelated
   `GitWatcher` crash (repo root vs. indexed subdirectory).
3. **37.5% (3/8)** — first honest baseline. Diagnosed a real crowding
   issue: a class with 8 near-identical HTTP-verb methods (`get`, `post`,
   `put`...) was flooding retrieval. Built docstring-similarity clustering
   to fix it (after first trying a simpler line-count heuristic, testing it
   against real data, and finding it didn't match the actual pattern —
   these methods were long, not short, because of large docstrings).
4. **75% (6/8)** after: (a) fixing two overly-strict eval questions that
   were penalizing legitimately-close answers, and (b) confirming via
   `/debug_retrieve` that the 2 remaining misses weren't a retrieval-depth
   problem (widening top-k from 20 to 40 didn't change the outcome).
5. Root-caused those 2 remaining misses to embedding-model limitations
   (256-token truncation losing a long function's actual logic; a general
   sentence-similarity model never trained to align code with natural
   -language queries) and fixed them by switching to a code-specific
   embedding model (`jina-embeddings-v2-base-code`) rather than continuing
   to patch around a model that wasn't suited to the task.
6. **75% again, but with a DIFFERENT 2 misses** after the model swap —
   `solve_dependencies` (the original hardest case) was now correctly
   retrieved, confirming the diagnosis and fix were right. But a different,
   previously-passing question regressed. Honest finding: swapping
   embedding models is not a strict upgrade, it's a different set of
   ranking tradeoffs — exactly why an eval harness matters, so the
   tradeoff is measured rather than assumed away.
7. Investigated the 2 new misses via `/debug_retrieve` at top-30 depth and
   found something more interesting than a retrieval bug: the system was
   surfacing chunks that were **more precise answers than my original eval
   set's ground truth** — e.g. `get_path_param_names` (which literally
   extracts path parameter names via regex) is a more direct answer to
   "how are path parameters extracted" than the `get_request_handler`
   symbol I'd originally picked. Verified against the real source, then
   corrected the eval set rather than the retrieval code.
8. **Final: 87.5% (7/8).** The one remaining "miss" is fully understood,
   not a mystery: `_validate_value_with_model_field` is found and correctly
   judged relevant by the reranker (score -5.041, not the lowest in the
   candidate set) but lands at rank 13 — just outside the eval harness's
   `top_k_rerank=10` cutoff. Confirmed via `/debug_retrieve`'s per-stage
   breakdown. This is a real, honest limitation of the current reranking
   cutoff, documented and left as-is rather than tuned away, since chasing
   a single-digit percentage point further has real diminishing returns
   against the value of an honestly-reported result.

The `/debug_retrieve` endpoint and the `tests/` suite exist specifically so
this kind of diagnosis is repeatable, not a one-off manual investigation.

## Production considerations

**Configuration.** All settings are centralized and validated in
`src/config.py` (pydantic-settings) rather than scattered
`os.environ.get()` calls — invalid configuration (e.g. `llm_backend=groq`
with no `GROQ_API_KEY`) fails immediately at startup with a clear error,
not confusingly on the first request. See `.env.example` for every
available setting.

**Security.** `/query` and `/reindex` support optional API-key auth
(`REPO_SENTINEL_API_KEY` → required `X-API-Key` header) and are always
rate-limited (`REPO_SENTINEL_RATE_LIMIT_PER_MINUTE`, default 20/min per
client IP). This matters once deployed publicly: `/query` calls a
paid-tier-capable LLM API on every request, and an unauthenticated,
unlimited endpoint is a real cost/abuse exposure for a URL that's easy to
find and hit. Auth is disabled by default for local dev (no key set = open
access).

**Resilience.** LLM calls (`generation/llm_client.py`) retry transient
failures (connection errors, timeouts) with backoff, but deliberately don't
retry 4xx client errors, which won't self-resolve. A failed generation
returns a proper 502 to the caller instead of an unhandled exception.
Summary generation failures during indexing (Ollama down, a timeout) are
caught and logged, falling back to no summary for that chunk rather than
failing the entire index.

**Observability.** Structured logging (`src/logging_config.py`) throughout,
replacing ad-hoc `print()` calls. `GET /healthz` reports real readiness
(fails if startup itself failed, not just "the process is running") for use
by deployment platforms' health checks.

**Testing.** A real `pytest` suite (`tests/`, currently 75 tests) covers
chunking, BM25, git-diff detection, the vector store, hybrid retrieval
fusion, config validation, and API security — including regression tests
for bugs found and fixed during development (the `None`-metadata crash, the
`"None::None"` BM25-only-chunk bug, the near-duplicate-method crowding
issue). Heavy external dependencies (the embedding model download, a
running Ollama/Groq backend) are deliberately mocked so the suite runs in
under 2 seconds with no network access — appropriate for CI, and distinct
from (not a replacement for) the manual end-to-end verification described
in "Local development setup" below.

Run it:
```bash
pip install pytest pytest-mock
pytest
```

CI (`.github/workflows/ci.yml`) runs this suite plus a dependency-conflict
check on every push, across Python 3.11 and 3.12.

**Containerization.** A `Dockerfile` is provided for reproducible builds —
runs as a non-root user, includes a `HEALTHCHECK` against `/healthz`. Not
wired into the Render deployment below (Render's Python buildpack is
simpler for this project's actual free-tier deployment target), but this is
what you'd use for any container-based host.

**Known limitations, stated plainly rather than glossed over:**
- ChromaDB runs embedded/single-process here — this app does not
  horizontally scale across multiple instances without moving to a
  client-server vector DB deployment.
- The rate limiter is in-memory and per-process — correct for this
  project's actual single-instance free-tier deployment, but would need to
  move to shared storage (Redis, etc.) if ever run multi-instance.
- Render's free tier has no persistent disk, so the index rebuilds on every
  cold start — a real, documented cost tradeoff for free hosting, not a
  bug (see `render.yaml`'s comment).
- BM25's classic IDF formula produces near-zero or negative scores at very
  small corpus sizes (a handful of chunks) — inherent to the algorithm, not
  a bug in this code, and irrelevant at the corpus sizes this project
  actually indexes (hundreds of chunks). Documented and tested explicitly
  in `tests/test_bm25_index.py`.

## Local development setup

**1. Install Ollama and pull a model:**

```bash
# macOS
brew install ollama
ollama serve &
ollama pull llama3.1:8b
```

If you're on a GPU with limited VRAM (roughly under 6GB), a smaller model
like `llama3.2:3b` will run entirely on-GPU and respond much faster than a
model that has to partially fall back to CPU.

**2. Copy the example environment file and adjust as needed:**

```bash
cp .env.example .env
```

**3. Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**4. Clone the demo repo to index:**

```bash
git clone --depth 1 https://github.com/fastapi/fastapi.git ./data/fastapi
```

**5. Run the API:**

```bash
uvicorn src.api.main:app --reload --port 8000
```

First startup downloads the embedding model (~640MB, one-time) and runs a
full index — a few minutes for FastAPI's `fastapi/` package. Auto-summary
generation for undocumented functions (`REPO_SENTINEL_AUTO_SUMMARIZE`) is
**off by default for local dev** — it adds one LLM call per undocumented
function, which is slow on a local model (verified during this project's
own development: 15-30+ minutes for FastAPI's package on a VRAM
-constrained GPU). It's on by default for the deployed version
(`render.yaml`), where Groq's speed makes the cost negligible. Turn it on
locally with `REPO_SENTINEL_AUTO_SUMMARIZE=true` if you want to compare —
see the "Eval-driven debugging" section above for what it's for.

**6. Run the frontend (React + Vite):**

```bash
cd frontend
npm install
cp .env.example .env.local   # defaults to http://localhost:8000, adjust if needed
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

**7. Try the live-update demo:**

```bash
cd data/fastapi
echo "# demo change" >> fastapi/routing.py
git commit -am "demo: trigger incremental reindex"
curl -X POST http://localhost:8000/reindex
```

Ask a question touching `routing.py` before and after — the retrieved
content reflects the new commit.

## Diagnosing a retrieval miss

```bash
curl "http://localhost:8000/debug_retrieve?question=how+does+dependency+injection+work&top_k=20" | python3 -m json.tool
```

Shows BM25-only results, vector-only results, post-fusion ranking, and
post-rerank ranking, separately — tells you whether a missing chunk never
entered the candidate pool (retrieval-depth or embedding-relevance issue)
or was found but scored low (reranker issue).

## Indexing a different repo

The demo defaults to FastAPI, but any public GitHub/GitLab/Bitbucket repo
can be indexed and made the active, chat-answerable repo — via the
frontend's "Index a different repo" button, or directly:

```bash
curl -X POST http://localhost:8000/index_repo \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/owner/repo"}'
# -> {"job_id": "...", "status": "queued", ...}

curl http://localhost:8000/index_repo/<job_id>
# poll until status is "ready" or "failed"
```

**Why this is async (job + polling), not a single blocking request:**
cloning, chunking, embedding, and auto-summarizing undocumented functions
for a new repo can take anywhere from under a minute to 20+ minutes
(observed directly during this project's own development, dominated by
per-function LLM summary calls on a slower local model) — far too long
for a single HTTP request to hold open without hitting client/proxy
timeouts.

**Security note:** `repo_url` is validated before anything is cloned
(`src/api/repo_validation.py`) — restricted to http(s), an allowlist of
known git hosting providers, and a DNS-resolution check that rejects
private/internal/link-local addresses (including the cloud metadata
address `169.254.169.254` specifically). Without this, accepting an
arbitrary URL and having the server fetch it is a classic SSRF vector.
Covered by 18 tests in `tests/test_repo_validation.py`, including several
specifically adversarial cases (subdomain tricks, path traversal attempts,
the metadata address).

**Scope, stated honestly:** this supports one *active* repo at a time,
swapped on request — not concurrent multi-tenant indexing (separate
storage/session isolation per user). That's a materially bigger feature
(storage isolation, cleanup policies, per-tenant cost accounting) that's
deliberately out of scope here. Indexed repos are also capped at
`MAX_INDEXABLE_FILES` (800 by default, in `src/api/repo_jobs.py`) to bound
worst-case indexing time and cost on a public deployment.

## Running the eval suite

```bash
curl http://localhost:8000/eval | python3 -m json.tool
```

Returns retrieval recall@k against the curated question set in
`src/eval/eval_set.json`. Questions can specify either a single
`expected_file_path`/`expected_symbol_name`, or an `acceptable_answers`
list when more than one chunk legitimately answers the question.

## Deployment (so it's demo-able without your laptop)

Free-tier cloud hosts can't run a local LLM continuously, so the deployed
version swaps generation from Ollama to **Groq's free API** (still serves
open models like Llama — just hosted). Retrieval (chunking, BM25,
embeddings, vector search) is unchanged either way.

1. Get a free Groq API key: https://console.groq.com
2. Generate a random API key for your own deployment (e.g.
   `openssl rand -hex 32`) — this is what protects `/query` and `/reindex`
   once your URL is public
3. Push this repo to GitHub
4. On [Render](https://render.com): New → Blueprint → point at your repo
   (uses `render.yaml` automatically)
5. In the Render dashboard, set `GROQ_API_KEY` and `REPO_SENTINEL_API_KEY`
   (both marked `sync: false` in `render.yaml` so they're never committed)
6. Deploy. First request after a cold start takes several minutes (code
   -specific embedding model download + full index rebuild — see "Known
   limitations" above)
7. Build the frontend against your deployed backend and host it statically:
   ```bash
   cd frontend
   VITE_API_BASE=https://your-app.onrender.com VITE_API_KEY=your-key npm run build
   ```
   Deploy the resulting `dist/` folder to any static host (Netlify, Vercel,
   GitHub Pages — all free).

Once deployed, you can pull it up on **any device with a browser** — no
laptop or local setup needed to demo it.

## Configuration reference

See `.env.example` for the full list. Highlights:

```bash
REPO_SENTINEL_REPO_PATH=/path/to/any/python/repo   # git repo root
REPO_SENTINEL_INDEX_SUBDIR=fastapi                  # subfolder to actually index
REPO_SENTINEL_LLM_BACKEND=ollama                    # or "groq"
REPO_SENTINEL_API_KEY=                              # unset = open access (local dev)
```

Note: `src/eval/eval_set.json` is hand-curated against FastAPI's actual
symbols — write your own eval set for a different repo.

## Project structure

```
src/
  config.py                      centralized, validated settings
  logging_config.py              structured logging setup
  chunking/code_chunker.py       AST-based chunker + near-duplicate-method grouping
  indexing/
    git_watcher.py               git diff -> changed files
    embedder.py                  jina-embeddings-v2-base-code
    vector_store.py              ChromaDB wrapper
    bm25_index.py                BM25 keyword index
    pipeline.py                  orchestrates the above (full + incremental)
  retrieval/
    hybrid_retriever.py          reciprocal rank fusion
    reranker.py                  cross-encoder reranking
  generation/
    llm_client.py                Ollama / Groq adapter, retry-with-backoff, auto-summary generation
  eval/
    run_eval.py                  retrieval recall scoring (single or multi-answer)
    eval_set.json                curated Q&A ground truth
  api/
    main.py                      FastAPI app: /query /status /reindex /debug_retrieve /eval /healthz
    security.py                  API-key auth + rate limiting
tests/                           pytest suite (75 tests, mocked external deps, <2s)
scripts/
  post-commit-hook.sh            git hook for instant reindex
  watch_and_reindex.py           polling alternative to the hook
frontend/                        React + Vite chat UI (chat + live sources rail)
  src/App.jsx                    main component
  src/api.js                     backend API client
  src/styles.css                 design tokens + layout
.github/workflows/ci.yml         CI: tests + dependency check
Dockerfile / .dockerignore       containerized build
render.yaml                      deployment config
.env.example                     all configuration options, documented
