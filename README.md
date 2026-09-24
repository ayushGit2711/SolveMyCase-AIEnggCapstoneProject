# solvemycase: Grounded Indian Legal Procedural Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Corpus: open-india-law](https://img.shields.io/badge/Corpus-open--india--law-green.svg)](https://huggingface.co/datasets/vaquill/open-india-law)

**solvemycase** is an agentic RAG system for Indian citizens dealing with everyday legal incidents. It currently covers three domains:

- 🚗 **Motor vehicle accidents** (MACT claims, FIRs, BNS/IPC offences)
- 🏠 **Property conflicts** (eviction, dispossession, encroachment, injunctions)
- 🛒 **Consumer rights** (defective goods, deficient service, e-commerce refunds)

Generic AI assistants often make up statutory sections or cite irrelevant precedents. **solvemycase** returns a **dual output** instead:

1. **A chronological procedural action plan** in six phases: immediate action → police/administrative → evidence → legal notice → forum filing → limitation and appeal. Each step names the forum, the statutory basis, and the deadline.
2. **Verified statutory provisions and court precedents**. Each citation is checked against the retrieved corpus and links to its official source (India Code, court records). Citations that can't be checked are removed and listed in the response.

This is an AI engineering capstone project. It compares two pipelines side by side:

| | Approach A: Baseline | Approach B: Proposed |
|---|---|---|
| Module | `core/baseline/vanilla_rag.py` | `core/proposed/graph.py` |
| Query handling | Raw user query | Guardrail + decontextualized statute/precedent sub-queries |
| Retrieval | Single-pass dense search | Hybrid dense + BM25 (RRF), criminal-code sub-layer, cross-encoder rerank |
| Generation | One monolithic prompt | Procedural planner agent (LangGraph) |
| Verification | None | Citation grounding node with re-grounding and stripping |

---

## Architecture (Approach B)

```
User Query (factual legal scenario)
         │
         ▼
[1. Intent & Scope Guardrail] ───────────► Out-of-scope / adversarial refusal
         │ (valid; classified into domain)
         ▼
[2. Query Decontextualizer & Expander]
   ├── Statutory section sub-queries (BNS, BNSS, MV Act, CPA 2019, TPA, SRA, CPC)
   └── Precedent sub-queries (Supreme Court & High Court case law)
         │
         ▼
[3. Hybrid Retrieval & Reranker]
   ├── Qdrant dense vector search
   ├── BM25 sparse keyword search (exact sections & Acts), fused via RRF
   ├── Criminal-code deep RAG sub-layer (BNS/BNSS/IPC/CrPC) when criminal indicators exist
   └── Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
         │
         ▼
[4. Procedural Planner Agent]
   └── Drafts the 6-phase chronological roadmap + candidate citations from context only
         │
         ▼
[5. Verification Node (hallucination checker)]
   ├── Matches every cited section / judgment against retrieved context
   ├── Tries exact-section re-grounding in the store before discarding
   └── Strips ungrounded citations and clears the matching action-step references
         │
         ▼
[6. Dual-Output Synthesis]  →  DualOutputResponse
   ├── Chronological action plan
   └── Verified provisions & precedents with official source URLs
```

The graph is defined in [`core/proposed/graph.py`](core/proposed/graph.py). The shared state is defined in [`core/proposed/state.py`](core/proposed/state.py).

### Offline / no-API-key mode

Every LLM-backed component has a deterministic fallback, so the whole system runs, and its tests pass, **without an OpenAI key**:

| Component | With `OPENAI_API_KEY` | Without key (fallback) |
|---|---|---|
| Scope guardrail | `gpt-4o-mini` JSON classification | Keyword heuristics |
| Decontextualizer | `gpt-4o-mini` sub-query generation | Domain-template queries |
| Embeddings | `text-embedding-3-small` | Hashed bag-of-words vectors (1536-d) |
| Reranker | Cross-encoder (sentence-transformers) | Lexical overlap scoring |
| Planner | `gpt-4o` JSON roadmap | Curated per-domain roadmaps |
| LLM judge | `gpt-4o` scoring | Programmatic rubric |

The fallbacks are for development and CI only. Retrieval quality and answer quality are much lower than in the LLM-backed mode.

---

## Repository Layout

> [!IMPORTANT]
> The repository root **is** the `solvemycase` Python package. All imports look like `from solvemycase.core...`. See [Setup](#setup) for how to make it importable.

```
solvemycase/                      ← repo root (package)
├── api/main.py                   FastAPI service
├── config/settings.py            Pydantic Settings (all env-driven configuration)
├── core/
│   ├── baseline/vanilla_rag.py   Approach A
│   ├── guardrails/scope_checker.py
│   ├── retrieval/                decontextualizer.py, reranker.py
│   └── proposed/                 graph.py, state.py, procedural_planner.py, verification_node.py
├── data/
│   ├── ingestion/                downloader.py, normalizer.py, schema.py
│   └── vectorstore/              qdrant_store.py (hybrid search), indexer.py (embeddings + indexing)
├── evaluation/
│   ├── build_dataset.py          Generates benchmark_dataset.json (52 scenarios)
│   ├── comparative_runner.py     Runs A vs B → benchmark_results.json
│   ├── metrics.py                Grounding, hallucination rate, phase completeness, forum accuracy
│   └── llm_judge.py              LLM-as-a-judge (+ programmatic fallback)
├── ui/app.py                     Streamlit side-by-side comparison UI
├── tests/                        pytest suite
└── AGENTS.md / GEMINI.md         Coding guidelines for AI agents & contributors
```

### Legal corpus

- **Statutes:** if `data/raw/in_central_legislation.parquet` is present (downloaded from the [open-india-law](https://huggingface.co/datasets/vaquill/open-india-law) mirror), sections from the target Acts are extracted with DuckDB. If it isn't, a curated set of 20 core provisions is used (MV Act, BNS, BNSS, IPC, CrPC, TPA, SRA, CPC, CPA 2019).
- **Precedents:** a curated set of 6 landmark Supreme Court judgments ([`data/ingestion/normalizer.py`](data/ingestion/normalizer.py)).

---

## Setup

**Requirements:** Python 3.9+ (tested on 3.13). An OpenAI API key is optional; see [offline mode](#offline--no-api-key-mode).

### 1. Clone into a folder named `solvemycase`

Because the repo root is the package, the clone directory must be named `solvemycase`, and you run commands from its **parent** directory:

```bash
mkdir smc && cd smc
git clone git@github.com:ayushGit2711/SolveMyCase-AIEnggCapstoneProject.git solvemycase
# All commands below are run from ./smc (the parent of solvemycase/)
```

> If you already cloned it under a different name, symlink it instead:
> `ln -s /path/to/SolveMyCase-AIEnggCapstoneProject /some/dir/solvemycase` and run from `/some/dir` (or add it to `PYTHONPATH`).

### 2. Create an environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r solvemycase/requirements.txt
```

> [!TIP]
> `sentence-transformers` pulls in PyTorch. If you don't have a GPU, or the CUDA wheels can't be downloaded, install the CPU build first:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 3. Configure (optional)

```bash
cp solvemycase/.env.example .env      # .env is read from the current working directory
# Set OPENAI_API_KEY to enable LLM-backed mode
```

Key settings (all overridable via env vars; see [`config/settings.py`](config/settings.py)):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | *(unset)* | Enables LLM mode; unset means offline fallbacks |
| `OPENAI_MODEL_PRIMARY` / `OPENAI_MODEL_FAST` | `gpt-4o` / `gpt-4o-mini` | Planner & judge / guardrail & decontextualizer |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Dense embeddings |
| `QDRANT_PATH` | `data/qdrant_storage` | Embedded Qdrant (no Docker); relative to package root |
| `QDRANT_URL` / `QDRANT_API_KEY` | *(unset)* | Use a remote Qdrant instead |
| `MAX_RETRIEVED_CHUNKS` / `RERANK_TOP_K` | `8` / `4` | Retrieval depth / contexts passed to the planner |
| `VERIFICATION_STRICT_MODE` | `true` | Strict citation grounding |

### 4. Build the index

```bash
python -m solvemycase.data.ingestion.downloader     # optional: fetch open-india-law parquet (~26 MB)
python -m solvemycase.data.vectorstore.indexer      # embed + index into Qdrant (resets the collection)
```

> [!NOTE]
> If you switch between offline and OpenAI embeddings, re-run the indexer. Hashed and OpenAI vectors are not comparable.

### 5. Run the API or the UI

```bash
uvicorn solvemycase.api.main:app --reload --port 8000        # API docs at http://localhost:8000/docs
PYTHONPATH=. streamlit run solvemycase/ui/app.py             # UI at http://localhost:8501
```

> [!NOTE]
> `streamlit run` only adds the script's own folder (`ui/`) to the import path, so `PYTHONPATH=.` is needed for `import solvemycase` to work.
> The UI calls both pipelines directly and doesn't need the API. In embedded mode (`QDRANT_PATH`), only one process can open the Qdrant store, so run **either** the API **or** the UI. To run both, point them at a remote Qdrant with `QDRANT_URL`.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/domains` | Supported dispute domains |
| `POST` | `/query/baseline` | Run Approach A |
| `POST` | `/query/proposed` | Run Approach B |
| `POST` | `/compare` | Run both on the same scenario |
| `GET` | `/benchmark/summary` | Latest benchmark summary |

Query endpoints take `{"scenario": "<at least 10 characters>"}` and return the `DualOutputResponse` plus latency, grounding metrics, and procedural-completeness metrics.

```bash
curl -X POST localhost:8000/query/proposed -H 'Content-Type: application/json' \
  -d '{"scenario": "A speeding truck hit my father on his scooter in Pune and fled. He has fractures. What should we do?"}'
```

---

## Evaluation Benchmark

- **Dataset:** [`evaluation/benchmark_dataset.json`](evaluation/benchmark_dataset.json) has **52 annotated scenarios**: 15 motor vehicle, 15 property, 15 consumer, and 7 out-of-scope/adversarial guardrail probes. Each has gold-standard domain, forum, and statute metadata. It is regenerated by `evaluation/build_dataset.py`.
- **Metrics:** statutory hallucination rate, citation grounding accuracy, 6-phase procedural completeness, designated-forum accuracy, LLM-judge score (0–5), and latency.

```bash
python -m solvemycase.evaluation.comparative_runner            # all scenarios
python -m solvemycase.evaluation.comparative_runner --limit 5  # quick run
```

**Targets:** hallucination rate 0%, retrieval recall/precision ≥ 85%, relevant precedents in top 3 (MRR), procedural correctness ≥ 4.0/5.

> [!WARNING]
> The committed `benchmark_results.json` is a **3-scenario smoke run in offline mode**. It is not a representative result. Run the full benchmark with an OpenAI key before quoting any numbers.

---

## Testing

```bash
python -m pytest solvemycase/tests -q
```

The suite (20 tests) covers config, normalizer, vector store, both pipelines, metrics/judge, and the API. It runs fully offline.

---

## Project Status

| Phase | Scope | Status |
|---|---|---|
| 1–4 | Scaffold, config, corpus ingestion, hybrid Qdrant store, guardrail, decontextualizer, reranker, planner, verification, LangGraph | ✅ Done |
| — | BNS/BNSS/IPC criminal-code RAG sub-layer + citation re-grounding | ✅ Done |
| 5 | 52-scenario benchmark, metrics, LLM judge, comparative runner | ✅ Done (full LLM run pending) |
| 6 | FastAPI service + Streamlit side-by-side UI | ✅ Done |

### Known limitations / next steps

- The precedent corpus is small (6 curated judgments). Judgment ingestion from open-india-law is not wired up yet.
- The verification retry loop is not implemented yet. `MAX_HALLUCINATION_RETRIES` is defined, but the graph is linear (planner → verification → synthesis), so stripped citations are not re-planned.
- In offline mode, hashed embeddings with domain filtering can return few or no verified citations for a query.
- Packaging: the repo root is the package, so `pip install -e .` does not expose `solvemycase` yet (see [Setup](#setup)).
- The API is open (`CORS *`, no auth) and meant for local or demo use only.

---

## Contributing

Please read [`AGENTS.md`](AGENTS.md) (coding guidelines: zero assumptions, reuse, config over hardcoding, PEP 8, testability, no secrets in code) before opening a PR.

> **Disclaimer:** solvemycase gives procedural information, not legal advice. Consult a qualified advocate for your specific situation.
