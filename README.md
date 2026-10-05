# JanSahay — A Hindi Semantic Search Engine for Government Welfare Schemes

JanSahay is a multilingual semantic search system for Indian Government welfare schemes, built to address a specific, measured problem: **keyword-based search fails on Hindi and code-mixed queries** when scheme information is written in formal English/transliterated administrative language. JanSahay compares a classical TF-IDF retriever against a multilingual semantic retriever to test whether meaning-based matching can close that gap.

This is a B.Tech Computer Engineering NLP micro-project (Semester V Self-Learning Activity), developed by **Vrinda Garg** and **Divya Choudhary**, Shah and Anchor Kutchhi Engineering College, Mumbai.

---

## Why JanSahay

A Hindi-speaking citizen searching गरीबों के लिए घर (*"house for the poor"*) gets nothing from keyword search against a scheme officially named **Pradhan Mantri Awas Yojana** — there's no shared vocabulary between the two, even though they mean the same thing. JanSahay's evaluation confirms this isn't a hypothetical problem: on the project's 20 held-out Hindi evaluation queries, the TF-IDF baseline scores **exactly 0.0000** on every strict-relevance metric (Hit@1, Hit@5, MRR), while the semantic retriever recovers non-zero performance on the same queries. See [Results](#results) below.

---

## How it works

```
User Query
   → Preprocessing (language-aware: TF-IDF branch / semantic branch)
   → TF-IDF or Semantic (E5) representation
   → Cosine similarity against the indexed scheme corpus
   → Ranked Top-K relevant Government schemes
   → Returned to the user (API → frontend)
```

Two retrievers are implemented side by side, not one replacing the other:

| | TF-IDF (baseline) | Semantic (proposed) |
|---|---|---|
| Representation | `scikit-learn TfidfVectorizer`, 1–2 gram, sublinear TF, L2-normalized | [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) via `sentence-transformers`, 768-dim, L2-normalized |
| Index size | 115 × 6,129 | 115 × 768 |
| Training | None — vocabulary fitting only (no gradient-based training) | None — pretrained encoder used directly, no fine-tuning |
| Fallback | — | `paraphrase-multilingual-MiniLM-L12-v2` if the primary model fails to load |
| Avg. latency | 8.54 ms | 49.76 ms |

Both are served through the same FastAPI backend and exposed side by side in the frontend's comparison view, since the evaluation shows neither one is strictly better across every query type (TF-IDF is faster and competitive on literal English keyword queries; semantic retrieval wins decisively on Hindi and code-mixed queries).

---

## Project structure

```
jansahay/
├── jansahay/                  # Core NLP library
│   ├── preprocessing/         # Dual-branch cleaning, tokenization, document builder
│   ├── retrieval/             # TfidfRetriever, SemanticRetriever
│   └── evaluation/            # Hit@K, Recall@K, Precision@K, MRR
├── backend/                   # FastAPI application
│   └── app/
│       ├── main.py            # Entrypoint, CORS, startup artifact loading
│       ├── core/config.py     # Environment-based configuration
│       ├── routes/            # /api/search, /api/schemes, /api/health
│       └── services/          # Retriever singleton loaded once at startup
├── frontend/                  # Next.js (App Router) + TypeScript + Tailwind v4
│   └── src/
│       ├── app/                # page.tsx
│       └── components/         # SearchForm, CompareView, SchemeCard, SchemeModal
├── scripts/                   # process_data.py, train_tfidf.py, build_semantic_index.py,
│                               # run_evaluation.py, cli_search.py, e2e_test.py
├── data/
│   ├── raw/                   # jansahay_final_dataset.csv (115 schemes)
│   ├── processed/             # schemes_clean.csv, queries.csv (82 unique eval queries)
│   └── artifacts/             # Saved TF-IDF and semantic indexes
├── reports/
│   ├── dataset_profile.md
│   ├── preprocessing_report.md
│   └── evaluation/            # Metrics (CSV/JSON/Markdown) + plots
├── tests/                     # 95 automated tests (pytest)
└── docs/
    ├── PROJECT_CONTEXT.md
    └── PHASE_LOG.md
```

---

## Dataset

115 manually curated and annotated Indian Government welfare schemes (`jansahay_final_dataset.csv`), spanning Central and State level, across Financial Assistance, Education, Employment, Agriculture, Women Welfare, Social Welfare, Pension, Healthcare, and Food Security. Each scheme is paired with a realistic user query in Hindi, English, or code-mixed Hindi-English, annotated with Intent, Query Type, Language, and Difficulty.

- 115 rows, 11 columns, 0 missing values
- Language split: English 45.2%, Code-Mixed 29.6%, Hindi 25.2%
- 33 duplicate `User_Query` values collapse to **82 unique evaluation queries** (ground-truth scheme IDs unioned per unique query)

See `reports/dataset_profile.md` for the full profiling output.

---

## Results

Evaluated on the 82 unique queries, Top-K = 5, under both **Strict** relevance (exact ground-truth scheme) and **Topical** relevance (same Intent category):

| Model | Hit@1 | Hit@5 | MRR | Hit@5 (Topical) | Avg. Latency |
|---|---|---|---|---|---|
| TF-IDF | 0.0488 | 0.1585 | 0.0846 | 0.7195 | 8.54 ms |
| **Semantic (E5)** | **0.0610** | **0.2195** | **0.1161** | **0.9512** | 49.76 ms |

**By language (the headline result):**

| Model | Hindi Hit@1 | Hindi Hit@5 | Hindi MRR |
|---|---|---|---|
| TF-IDF | 0.0000 | 0.0000 | 0.0000 |
| **Semantic (E5)** | **0.1000** | **0.3000** | **0.1667** |

On English queries, TF-IDF is marginally *ahead* of the semantic model (Hit@5: 0.2895 vs. 0.2368) — reported as-is, not adjusted to favour either model. Full breakdowns (by language, difficulty, query type) are in `reports/evaluation/evaluation_summary.md`.

**Known limitations:** small corpus (115 schemes); no lemmatization/stemming (a deliberate choice — see `reports/preprocessing_report.md`); no fine-tuning of the semantic model; the 0.30 semantic similarity threshold is a documented heuristic, not empirically validated; no hybrid/re-ranking layer yet.

---

## Getting started

### Prerequisites
- Python 3.12+
- Node.js (for the frontend)

### Backend

```bash
cd backend
pip install -r requirements.txt

# Build the retrieval indexes (first run only)
python ../scripts/process_data.py
python ../scripts/train_tfidf.py
python ../scripts/build_semantic_index.py

# Run the API
python -m app.main
# or: uvicorn app.main:app --reload
```
The API serves at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Endpoints:**
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/search` | Run a search query against TF-IDF and/or Semantic |
| `GET` | `/api/schemes` | List/filter schemes |
| `GET` | `/api/schemes/{scheme_id}` | Get one scheme's full detail |
| `GET` | `/api/health` | Health check |

### Frontend

```bash
cd frontend
npm install
npm run dev
```
Runs at `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL` if the backend isn't at the default `http://localhost:8000`.

### Running the evaluation

```bash
python scripts/run_evaluation.py
```
Regenerates everything under `reports/evaluation/` (metrics CSVs, `evaluation_summary.md`, and plots) from the current indexes.

### Running tests

```bash
pytest
```
95 tests across preprocessing, TF-IDF, semantic retrieval, evaluation metrics, and the API layer — including a dedicated regression test asserting the Hindi TF-IDF zero-overlap behaviour, and a data-leakage check confirming query-only fields never leak into the indexed scheme documents.

---

## Tech stack

**Backend:** FastAPI, Pydantic, scikit-learn, sentence-transformers, PyTorch, pandas, NumPy
**Frontend:** Next.js (App Router), TypeScript, Tailwind CSS v4
**Testing:** pytest, httpx

---

## Project status

This is an academic research-stage project, not a deployed public service. No containerization, CI/CD, or cloud deployment is currently configured. Scheme data is a static, manually compiled snapshot and is not automatically kept current with actual government portals — it should not be treated as an authoritative source for real eligibility decisions.

## License

Academic project — no license specified yet.
