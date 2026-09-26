# SOLVE MY CASE (`solvemycase`): Grounded Indian Legal Procedural Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Corpus: open-india-law](https://img.shields.io/badge/Corpus-open--india--law-green.svg)](https://huggingface.co/datasets/vaquill/open-india-law)

**SOLVE MY CASE** (`solvemycase`) is an agentic RAG system for Indian citizens dealing with everyday legal incidents. Its curated statutory and Supreme Court corpus covers:

- 🚗 **Motor vehicle accidents** (MACT claims, insurer liability, FIRs, BNS/IPC offences)
- 🏠 **Property conflicts** (eviction, dispossession, encroachment, specific performance, injunctions)
- 🛒 **Consumer rights** (defective goods, deficient service, e-commerce refunds, product liability)
- 🐾 **Selected general criminal provisions** (BNS / BNSS / PCA 1960 offences including animal cruelty, cheating, criminal breach of trust, unlawful assembly, and FIR registration)

Generic AI assistants often make up statutory sections, cite irrelevant statutes that merely share words with the query, or fabricate court precedents. **SOLVE MY CASE** returns a **dual output** instead:

1. **A chronological procedural action plan** in six phases: immediate action → police/administrative → evidence → legal notice → forum filing → limitation and appeal. Each step names the forum, the statutory basis, and the deadline.
2. **Verified statutory provisions and court precedents**, screened for factual applicability and checked against the retrieved corpus with direct links to official sources (India Code DSpace handles and Indian Kanoon judgment documents). When a legal question falls outside the indexed corpus, the engine states what the dataset covers and gives general procedural steps **without** citing unrelated statutes.

This is an AI engineering capstone project. It compares two pipelines side by side:

| | Approach A: Baseline | Approach B: Proposed |
|---|---|---|
| Module | `core/baseline/vanilla_rag.py` | `core/proposed/graph.py` |
| Query handling | Raw user query | Guardrail + decontextualized statute/precedent sub-queries |
| Retrieval | Single-pass dense search | Hybrid dense + BM25 (RRF), criminal-code sub-layer, multi-query cross-encoder rerank |
| Applicability screening | None | `ApplicabilityJudge` (`gpt-4o-mini`) + honest coverage-gap router |
| Generation | One monolithic prompt | Procedural planner agent (LangGraph) |
| Verification | None | Strict citation grounding node against retrieved context (with repealed-law warnings) |

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
   ├── Statutory section sub-queries (BNS, BNSS, MV Act, CPA 2019, TPA, SRA, CPC, PCA 1960)
   └── Precedent sub-queries (Supreme Court & High Court case law)
         │
         ▼
[3. Hybrid Retrieval & Multi-Query Reranker]
   ├── Qdrant dense vector search (batched embeddings + embedder-space compatibility check)
   ├── BM25 sparse keyword search (exact sections & Acts), fused via RRF
   ├── Criminal-code deep RAG sub-layer (BNS/BNSS/IPC/CrPC) when criminal indicators exist
   └── Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) scored across scenario + sub-queries
         │
         ▼
[3b. Applicability Check & Coverage-Gap Router]
   ├── Screens top reranked candidates for factual & legal applicability (gpt-4o-mini)
   ├── If NO candidate applies (or general_dispute is unscreened) ──► Honest Coverage-Gap Synthesis
   └── Keeps applicable statutes & precedents in rerank order
         │
         ▼
[4. Procedural Planner Agent]
   └── Drafts the 6-phase chronological roadmap + candidate citations from applicable context only
         │
         ▼
[5. Verification Node (hallucination checker)]
   ├── Matches every cited section / judgment strictly against applicable retrieved context
   ├── Flags repealed IPC/CrPC citations when BNS/BNSS equivalents apply
   └── Strips ungrounded citations and clears matching action-step references
         │
         ▼
[6. Dual-Output Synthesis]  →  DualOutputResponse
   ├── Chronological action plan + optional coverage-gap banner
   └── Verified provisions & precedents with direct official source URLs
```

The graph is defined in [`core/proposed/graph.py`](core/proposed/graph.py). The shared state is defined in [`core/proposed/state.py`](core/proposed/state.py).

### Offline / no-API-key mode

Every LLM-backed component has a deterministic fallback, so the whole system runs, and its tests pass, **without an OpenAI key**:

| Component | With `OPENAI_API_KEY` | Without key (fallback) |
|---|---|---|
| Scope guardrail | `gpt-4o-mini` JSON classification | Word-boundary keyword heuristics |
| Decontextualizer | `gpt-4o-mini` sub-query generation | Domain-template queries |
| Embeddings | `text-embedding-3-small` | Hashed bag-of-words vectors (1536-d) |
| Reranker | Cross-encoder (`sentence-transformers`) | Lexical overlap scoring |
| Applicability check | `gpt-4o-mini` candidate screening | Fail-open for covered domains; withholds unscreened `general_dispute` sources |
| Planner | `gpt-4o` JSON roadmap | Curated per-domain roadmaps |
| LLM judge | `gpt-4o` scoring + deterministic uncovered-scenario rubric | Programmatic rubric |

The fallbacks are for development and CI only. Retrieval quality and answer quality are much lower than in the LLM-backed mode.

---

## Repository Layout

> [!IMPORTANT]
> The repository root **is** the `solvemycase` Python package. All imports look like `from solvemycase.core...`. See [Setup](#setup) for how to make it importable.

```
solvemycase/                      ← repo root (package)
├── api/main.py                   FastAPI service
├── config/
│   ├── settings.py               Pydantic Settings (all env-driven configuration)
│   └── openai_client.py          Shared OpenAI client builder
├── core/
│   ├── baseline/vanilla_rag.py   Approach A
│   ├── guardrails/scope_checker.py
│   ├── retrieval/                decontextualizer.py, reranker.py (multi-query cross-encoder)
│   ├── proposed/                 graph.py, state.py, applicability_judge.py, coverage.py,
│   │                             procedural_planner.py, verification_node.py
│   └── telemetry.py              JSONL runtime inference & strip-rate logger
├── data/
│   ├── ingestion/                downloader.py, normalizer.py, schema.py, link_checker.py
│   └── vectorstore/              qdrant_store.py (hybrid search), indexer.py (embeddings + indexing)
├── evaluation/
│   ├── build_dataset.py          Generates benchmark_dataset.json (61 scenarios)
│   ├── comparative_runner.py     Runs A vs B with full ExecutionTrace → benchmark_results.json
│   ├── metrics.py                Post-output grounding, irrelevant-citation rate, IR, Agent & TNR/FPR
│   ├── llm_judge.py              Decomposed 5-criterion rubric-anchored LLM-as-a-judge (+ uncovered rubric)
│   ├── noise_sensitivity.py      Cross-domain distractor chunk stress-test harness
│   └── sme_review.py             Legal expert (SME) annotation store & Judge-vs-Human agreement (MAE/Pearson r/κ)
├── ui/
│   ├── app.py                    Streamlit entrypoint ("SOLVE MY CASE" header + navigation + sidebar)
│   ├── state.py                  Cached engines, live corpus stats & per-domain coverage cards, benchmark loaders
│   ├── views/                    Pages: Get help, Connect to a lawyer, Compare A vs B, Benchmark, How it works
│   └── components/               Reusable widgets: input, progress, action plan, citations, lawyers, export, formatting
├── tests/                        pytest suite (128 tests)
└── AGENTS.md / GEMINI.md         Coding guidelines for AI agents & contributors
```

### Legal corpus

- **Statutes:** if `data/raw/in_central_legislation.parquet` is present (downloaded from the [open-india-law](https://huggingface.co/datasets/vaquill/open-india-law) mirror), sections from the target Acts are extracted with DuckDB. In addition, a curated set of **47 statutory provisions** (`MV Act, BNS, BNSS, IPC, CrPC, TPA, SRA, CPC, CPA 2019, PCA 1960`) with verified `https://www.indiacode.nic.in/handle/123456789/...` links is always merged in so 100% of covered benchmark sections are present.
- **Precedents:** a curated set of **9 landmark Supreme Court judgments** with verified Indian Kanoon (`https://indiankanoon.org/doc/<id>/`) links ([`data/ingestion/normalizer.py`](data/ingestion/normalizer.py)).

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
| `OPENAI_MODEL_PRIMARY` / `OPENAI_MODEL_FAST` | `gpt-4o` / `gpt-4o-mini` | Planner & judge / guardrail, decontextualizer & applicability check |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Dense embeddings |
| `QDRANT_PATH` | `data/qdrant_storage` | Embedded Qdrant (no Docker); relative to package root |
| `QDRANT_URL` / `QDRANT_API_KEY` | *(unset)* | Use a remote Qdrant instead |
| `MAX_RETRIEVED_CHUNKS` / `RERANK_TOP_K` | `10` / `6` | Retrieval depth per query / applicable contexts passed to the planner |
| `APPLICABILITY_CHECK_ENABLED` | `true` | Screen reranked candidates for factual applicability (`gpt-4o-mini`) |
| `APPLICABILITY_CANDIDATE_K` | `10` | Number of top reranked candidates inspected by `ApplicabilityJudge` |
| `RETRIEVAL_MIN_CONFIDENCE` | `0.25` | `RetrievalQualityGate` threshold for unfiltered fallback search |
| `VERIFICATION_STRICT_MODE` | `true` | Strict citation grounding against retrieved context |

### 4. Build the index

```bash
python -m solvemycase.data.ingestion.downloader     # optional: fetch open-india-law parquet (~26 MB)
python -m solvemycase.data.vectorstore.indexer      # embed + index into Qdrant (resets the collection)
python -m solvemycase.data.ingestion.link_checker   # optional: audit corpus & UI links
```

> [!NOTE]
> Each indexed point records its `embedder_id` and `corpus_fingerprint`. If the corpus changes or you switch between offline and OpenAI embeddings, `ensure_index_current` (or re-running the indexer) rebuilds the collection automatically.

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

## Evaluation Benchmark (61 Scenarios — Full OpenAI Results)

- **Dataset:** [`evaluation/benchmark_dataset.json`](evaluation/benchmark_dataset.json) has **61 annotated scenarios**:
  - **46 covered legal scenarios:** 15 motor vehicle, 15 property, 15 consumer, and 1 covered general criminal scenario (`gen_001`: pet animal killed by a watchman, covered by BNS s.325, PCA 1960 s.11, and *AWBI v. A. Nagaraja*).
  - **8 uncovered legal scenarios (`uncov_001`–`uncov_008`):** cyber UPI phishing, unpaid salary/EPF, mutual divorce, 498A/DV Act dowry harassment, online defamation, RTI appeal, NI Act s.138 cheque bounce, and neighbour's pet dog bite. These test whether the pipeline abstains from citing unrelated statutes when the governing law is outside the corpus.
  - **7 out-of-scope / adversarial guardrail probes (`guardrail_001`–`guardrail_007`).**
- **Multi-Layer Evaluation:**
  - **Post-Output Citation Grounding & Irrelevant Citation Rate:** Symmetric grounding verification across Approach A and Approach B, plus Act-aware irrelevant-citation detection, uncovered-topic honesty rate, and false coverage-gap rate.
  - **Retrieval IR Metrics:** Section Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and Context Relevance.
  - **Agentic Control-Flow & Guardrail Metrics:** Node Trajectory Validity (including the `retrieve_and_rerank -> applicability_check -> synthesis` coverage-gap route), Criminal Routing Accuracy, Guardrail Out-of-Scope TNR / In-Scope FPR, and End-to-End Task Success Rate.
  - **Decomposed 5-Criterion LLM-as-a-Judge + SME Calibration:** Rubric-anchored evaluation (`statutory_accuracy`, `procedural_actionability`, `forum_appropriateness`, `hallucination_freedom`, `coherence_and_specificity`), deterministic grading for uncovered legal scenarios, and SME calibration (`evaluation/sme_review.py` & `evaluation/sme_annotations.json`).

```bash
python -m solvemycase.evaluation.comparative_runner                       # all 61 scenarios
python -m solvemycase.evaluation.comparative_runner --limit 5             # quick run
python -m solvemycase.evaluation.comparative_runner --judge-samples 3     # multi-sample judge variance
```

| Evaluation Metric (61 Scenarios) | Approach A (Baseline RAG) | Approach B (Proposed LangGraph) |
|---|---:|---:|
| **Post-Output Hallucination Rate** | 1.3% | **0.0%** |
| **Citation Grounding Accuracy** | 87.4% | **98.1%** |
| **Pre-Verification Strip Rate** | 0.0% | **0.0%** |
| **Irrelevant Citation Rate** | 15.2% | **1.9%** |
| **Uncovered-Topic Honesty Rate** | 25.0% | **100.0%** |
| **False Coverage-Gap Rate** | 0.0% | **2.2%** |
| **Retrieval Section Recall@K** | 43.3% | **80.2%** |
| **Retrieval Mean Reciprocal Rank (MRR)** | 0.84 | **0.92** |
| **Expected Section Recall (Final)** | 44.5% | **81.3%** |
| **Procedural Phase Completeness (0–1)** | 0.67 | **0.84** |
| **Designated Forum Accuracy** | 62.3% | **92.5%** |
| **Agent Trajectory Validity** | 88.5% | **98.4%** |
| **Guardrail Out-of-Scope TNR** | 0.0% | **100.0%** |
| **End-to-End Task Success Rate** | 57.4% | **93.4%** |
| **LLM Judge Overall Rating (0–5)** | 3.27 | **4.58** |
| **Mean End-to-End Latency** | 5.04s | 14.38s |

---

## Testing

```bash
python -m pytest solvemycase/tests -q
```

The suite (**128 tests**) covers config, normalizer & link integrity (56-item corpus), Qdrant hybrid & criminal-code store (including embedder-space compatibility and auto-reindex), `ApplicabilityJudge` and coverage-gap synthesis, both pipelines (including `run_with_trace` and streaming), Act-aware citation parsing & irrelevant-citation metrics, decomposed LLM judge, the FastAPI service, and every Streamlit UI page (`AppTest`, including the `"SOLVE MY CASE"` header, live per-domain coverage cards, `"Connect to a lawyer"` directory, and coverage-gap banner).

---

## Project Status

| Phase | Scope | Status |
|---|---|---|
| 1–4 | Scaffold, config, 56-item corpus ingestion (47 statutes + 9 SC judgments), hybrid Qdrant store, guardrail, decontextualizer, multi-query cross-encoder reranker, `ApplicabilityJudge` & coverage-gap router, planner, strict verification, LangGraph | ✅ Done |
| — | BNS/BNSS/IPC/PCA criminal-code RAG sub-layer + repealed-law warnings + runtime JSONL telemetry | ✅ Done |
| 5 | 61-scenario OpenAI benchmark (46 covered + 8 uncovered + 7 guardrail), irrelevant-citation & coverage-honesty metrics, decomposed 5-dimension LLM judge, SME calibration & noise-sensitivity harness | ✅ Done |
| 6 | FastAPI service + `"SOLVE MY CASE"` Streamlit UI (live per-domain coverage cards, `"Connect to a lawyer"` directory & inline matching, coverage-gap banners & 3-tab Benchmark/SME dashboard) | ✅ Done |

### Known limitations / next steps

- **Focused corpus scope:** The indexed corpus currently covers 47 statutory provisions and 9 Supreme Court judgments across motor vehicle accidents, property disputes, consumer protection, and selected general criminal provisions (including animal cruelty). Questions in uncovered domains (e.g. family/matrimonial law, labour/employment, cyber crime under the IT Act, defamation, or NI Act s.138 cheque bounce) receive an explicit coverage-gap notice and general procedural steps without statutory citations.
- **Linear verification pass:** `MAX_HALLUCINATION_RETRIES` is defined in settings, while the graph runs a single deterministic verification pass (`planner → verification → synthesis`).
- **Offline mode recall:** Without `OPENAI_API_KEY`, hashed bag-of-words embeddings have lower recall, and `general_dispute` queries withhold unscreened citations by design.
- **Packaging:** The repo root is the package, so commands should be run from the parent folder (or with `PYTHONPATH` pointing to the parent folder; see [Setup](#setup)).
- **Demo API posture:** The FastAPI server uses permissive CORS (`*`) and no authentication for local/demo evaluation.

---

## Contributing

Please read [`AGENTS.md`](AGENTS.md) (coding guidelines: zero assumptions, reuse, config over hardcoding, PEP 8, testability, no secrets in code) before opening a PR.

> **Disclaimer:** SOLVE MY CASE (`solvemycase`) gives procedural information, not legal advice. Consult a qualified advocate for your specific situation.

---

Made with ❤️ by **the R.A.M.S**

