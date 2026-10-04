# JANSAHAY Project Context

## Project Objective
JanSahay is a Multilingual Semantic Search Engine for Indian Government Welfare Schemes. 
Users can enter queries in English, Hindi, or Hindi-English code-mixed (Hinglish) formats. 
The system ranks and retrieves relevant welfare schemes using classical (TF-IDF) and modern (Multilingual Sentence Embedding) NLP techniques.

## Architectural Phases

1. **Phase 1: Foundation & Audit**
   - Analysis of `jansahay_final_dataset.csv`.
   - Setup of basic structure and test environment.
2. **Phase 2: Preprocessing**
   - Multilingual text cleaning, language detection.
   - Separate NLP processing pipelines for TF-IDF (strict stemming/stopwords) and Semantic (light normalization).
3. **Phase 3: TF-IDF Baseline**
   - Classical lexical retrieval using scikit-learn.
   - Creates a 6,129-feature vocabulary on the 115 schemes.
4. **Phase 4: Semantic Embedding**
   - Multilingual neural retrieval using `intfloat/multilingual-e5-base` via `sentence-transformers`.
   - Projects queries and schemes into a 768-dimensional shared space.
5. **Phase 5: Evaluation**
   - Unbiased evaluation over 82 unique user queries.
   - Benchmarks Strict (exact match) and Topical (intent match) relevance using Hit@K, Recall@K, Precision@K, MRR.
6. **Phase 6: Backend API**
   - FastAPI application serving the NLP models.
   - Pydantic validation, one-time artifact loading at startup (lifespan), and End-to-End API test coverage.
7. **Phase 7: Frontend**
   - Next.js web application utilizing Tailwind CSS v4 and the App Router.
   - Polished, accessible UI allowing users to dynamically compare Semantic and Lexical models.

## Project Structure
- `backend/` - FastAPI backend application and API logic.
- `frontend/` - Next.js React user interface.
  - `src/app/` - App router, global CSS, layout.
  - `src/components/` - React UI components (SearchForm, SchemeCard, CompareView).
    - `routes/` - FastAPI endpoint handlers (search, schemes, health).
    - `services/` - `search_service.py` connects API to NLP retrievers.
- `data/`
  - `artifacts/` - Model states (TF-IDF matrices, E5 Embeddings). Loadable at runtime.
  - `processed/` - Cleaned datasets.
- `docs/` - Living project documentation.
- `jansahay/` - Core NLP Library.
  - `evaluation/` - Metrics and ranking analysis.
  - `preprocessing/` - Text cleaning and document formatting.
  - `retrieval/` - `TfidfRetriever` and `SemanticRetriever` logic.
- `reports/` - Evaluation summaries and dataset analysis.
- `scripts/` - Executable scripts for building artifacts and running evals.
- `tests/` - Comprehensive test suite (NLP models + API endpoints).

## Key Constraints
- **No vector database**: We use NumPy/scikit-learn arrays locally.
- **Data Leakage**: Strict separation between Scheme Text (for index building) and Query Metadata (for evaluation).
- **Model Integrity**: Cosine similarity is correctly used for ranking; it is never presented as a probability.
