# solvemycase: Grounded Indian Legal Procedural Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Corpus: open-india-law](https://img.shields.io/badge/Corpus-open--india--law-green.svg)](https://huggingface.co/datasets/vaquill/open-india-law)

**solvemycase** is an agentic AI legal reasoning system designed for Indian citizens dealing with legal incidents (motor vehicle accidents, property conflicts, and consumer rights violations).

Unlike generic AI assistants that hallucinate statutory sections or cite overturned/irrelevant precedents, **solvemycase** produces a dual-output response:
1. **A structured, chronological procedural action plan** (immediate actions, police/FIR steps under BNS/BNSS, evidentiary steps, legal notices, filing forum, and limitation periods).
2. **Verified statutory provisions and court precedents** retrieved with exact section numbers and direct provenance links to official government sources (India Code and High Court/Supreme Court records).

---

## Architecture Overview

```
User Query (Factual Legal Scenario)
         │
         ▼
[1. Intent & Scope Guardrail] ────────► Out-of-scope / Adversarial Refusal
         │ (Valid)
         ▼
[2. Query Decontextualizer & Expander]
   ├── Statutory Section Lookups (BNS, Motor Vehicles Act, CPA 2019, TPA)
   └── Precedent Lookups (Supreme Court & High Court Case Law)
         │
         ▼
[3. Hybrid Retrieval & Reranker]
   ├── Qdrant Vector Search (Dense Embeddings)
   ├── Sparse BM25 Keyword Search (Exact Sections & Acts)
   └── Cross-Encoder Reranker
         │
         ▼
[4. Procedural Planner (LangGraph ReAct Agent)]
   └── Drafts chronological, phased action roadmap
         │
         ▼
[5. Multi-Agent Verification Node (Hallucination Checker)]
   ├── Scans all cited sections, Acts, and judgments
   ├── Validates against retrieved source context
   └── Strips or flags any ungrounded citation (0% hallucination tolerance)
         │
         ▼
[6. Final Dual Output]
   ├── Chronological Action Plan
   └── Grounded Legal Provisions & Precedents with Official Source URLs
```

---

## Evaluation Benchmark

The system benchmarks two distinct implementations across a **50+ scenario gold-standard dataset**:
- **Approach A (Baseline - Vanilla RAG)**: Direct prompt embedding and single-pass generation.
- **Approach B (Proposed - Advanced RAG + Agent Verification)**: Query decontextualization, hybrid retrieval, cross-encoder reranking, and LangGraph verification.

Key Benchmark Targets:
- **Retrieval Context Recall & Precision**: $\ge 85\%$
- **Mean Reciprocal Rank (MRR)**: Top-3 for relevant precedents
- **Procedural Correctness**: $\ge 4.0 / 5.0$
- **Hallucination Rate**: $0\%$ tolerance

---

## Quickstart

1. **Clone & Set up Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   ```bash
   cp .env.example .env
   # Set your OPENAI_API_KEY in .env
   ```

3. **Ingest Legal Corpus**:
   ```bash
   python -m solvemycase.data.ingestion.downloader
   python -m solvemycase.data.vectorstore.indexer
   ```

4. **Launch API & Interactive Web UI**:
   ```bash
   # Run FastAPI server
   uvicorn solvemycase.api.main:app --reload --port 8000

   # Run Streamlit Frontend
   streamlit run solvemycase/ui/app.py
   ```
