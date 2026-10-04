# PHASE LOG

## Phase 1: Dataset Audit & Project Foundation

**Inspected:**
- The primary dataset `jansahay_final_dataset.csv`.
- Analyzed dimensions (115 rows, 11 columns) and verified column data types.
- Evaluated missing values, exact row duplicates, and field-level duplicates.
- Analyzed language distribution, intent distribution, query structures, and difficulty.
- Audited the project environment to check for required Python libraries.

**Created:**
- `analyze.py`: Python script to parse and compute dataset statistics.
- `reports/dataset_profile.md`: Detailed report.
- `docs/PROJECT_CONTEXT.md`: Living documentation.
- Project Architecture directories.
- `tests/test_basic.py`: Basic unit tests.

---

## Phase 2: Multilingual Preprocessing & Searchable Document Pipeline

**Implemented:**
- Reusable Data Loader, Language Detector, Text Cleaner (TF-IDF + Semantic branches), Document Builder, Query Processor, Pipeline Script.
- Outputs: `data/processed/schemes_clean.csv`, `data/processed/queries.csv`, `data/processed/dataset_stats.json`.
- 17/17 tests passed.

---

## Phase 3: Classical TF-IDF Retrieval Baseline

**Implemented:**
- `TfidfRetriever` class with `fit/search/rank_all/explain/save/load`.
- TF-IDF fitted on 115 scheme documents only (strict data leakage prevention).
- `scripts/train_tfidf.py`, `scripts/cli_search.py`.
- Artifact: `data/artifacts/tfidf/` (vectorizer, 115×6129 matrix, SHA256 hash).
- 25/25 tests passed.

---

## Phase 4: Multilingual Semantic Embedding Retrieval

**Model Selected:** `intfloat/multilingual-e5-base`
**Fallback:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
**Embedding Dimension:** 768
**Index:** (115, 768) NumPy float32 array, L2-normalised.
**E5 Prefixes:** `"query: <text>"` for queries, `"passage: <text>"` for documents.

**Implemented:**
- `jansahay/retrieval/semantic.py`: `SemanticRetriever` class with `build_index/search/rank_all/save/load`.
- `scripts/build_semantic_index.py`: One-shot index builder.
- `scripts/cli_semantic_search.py`: CLI search with low-relevance warnings.
- `tests/test_semantic.py`: 18 structural/behavioral tests.
- Artifact: `data/artifacts/semantic/` (embeddings.npy, config.json, schemes_metadata.csv).

---

## Phase 5: Formal Evaluation

**Evaluation Protocol:**
- Unit: 82 unique user queries (repeated-query rows collapsed; their scheme IDs unioned).
- Top-K = 5 for all rank-based metrics.
- Model configurations fixed from Phase 3 & 4 — no post-hoc tuning.
- Two relevance definitions: **Strict** (exact annotated scheme) and **Topical** (shared Intent).

**Metrics Implemented:**
Hit@1, Hit@5, Recall@1, Recall@5, Precision@5, MRR — all multi-label safe.

**Key Findings (from actual computed results — see `reports/evaluation/evaluation_summary.md`):**
- See evaluation_summary.md for the exact computed numbers.
- TF-IDF produces zero-overlap for Hindi queries (Devanagari vs English scheme text).
- Semantic (E5) performs cross-lingual retrieval for Hindi queries.
- Both models perform comparably on English keyword queries with high vocabulary overlap.

**Files Created:**
- `jansahay/evaluation/metrics.py`: Metric functions (36 unit tests).
- `tests/test_metrics.py`: 36/36 metric unit tests passed.
- `scripts/run_evaluation.py`: Fully reproducible evaluation pipeline.
- `reports/evaluation/metrics.csv` — per-query, per-model (164 rows).
- `reports/evaluation/overall_metrics.csv`
- `reports/evaluation/language_metrics.csv`
- `reports/evaluation/query_type_metrics.csv`
- `reports/evaluation/difficulty_metrics.csv`
- `reports/evaluation/qualitative_examples.csv`
- `reports/evaluation/latency.json`
- `reports/evaluation/evaluation_summary.md`
- `reports/evaluation/plots/` — 6 plots.

---

## Phase 6: FastAPI Backend

**Implemented:**
- Built a clean FastAPI backend to serve the NLP retrieval engine.
- Architecture:
  - `backend/app/main.py`: Entrypoint, CORS, lifespan startup event.
  - `backend/app/core/config.py`: Environment-based configuration, artifact paths.
  - `backend/app/schemas.py`: Pydantic models for strict request/response validation.
  - `backend/app/services/search_service.py`: Singleton artifact loader; core logic delegator to TF-IDF & Semantic classes.
  - `backend/app/routes/search.py`: `POST /api/search`
  - `backend/app/routes/schemes.py`: `GET /api/schemes` & `GET /api/schemes/{scheme_id}`
  - `backend/app/routes/health.py`: `GET /api/health`
- **Model Loading Behavior:**
  - Artifacts (TF-IDF matrices, Semantic embeddings, and `intfloat/multilingual-e5-base` model) are loaded EXACTLY ONCE during FastAPI application startup via `lifespan`.
  - Searches are extremely fast; the model is not reloaded per-request.
- **Testing:**
  - Created `tests/test_api.py` utilizing `TestClient` to perform End-to-End backend tests.
  - Tests validated health checks, scheme lookups, and multilingual search.
  - Empty/whitespace queries and invalid top_k bounds successfully return HTTP 422 as expected.

**What Phase 7 (Frontend) Implemented:**
- Clean, minimal, warm Next.js frontend using Tailwind CSS v4.
- `SearchForm`: Multi-model dropdown toggles (Semantic/TF-IDF/Both), Level intent filters.
- `CompareView`: Side-by-side display of Lexical vs Semantic retrieval.
- `SchemeModal`: Detail popup rendering Benefits, Description, and Retrieval Explanations.
- Score formatting clearly indicates cosine similarities rather than probability.
- Graceful API error handling and zero-match "empty states".
- Responsive scaling across mobile, tablet, and desktop viewports.

**What Phase 8 Needs to Implement:**
1. Final Integration & Architecture Review.
2. Production Deployment Readiness (packaging, dockerfiles if required).
3. Final README updates and academic presentation materials.
