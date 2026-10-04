"""
backend/app/services/search_service.py

The ONLY place NLP retrieval is called from.
API routes import this service — they never import retrieval classes directly.

Both retrievers are loaded at application startup (via lifespan) and stored as
module-level singletons.  No re-loading or re-fitting happens per-request.
"""
from __future__ import annotations

import logging
import sys
import os

# Ensure the project root is importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))           # backend/app/services
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
# Also try two levels up from backend/
_ALT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for p in (_PROJECT_ROOT, _ALT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

from jansahay.retrieval.tfidf    import TfidfRetriever
from jansahay.retrieval.semantic import SemanticRetriever
from app.core.config import (
    TFIDF_ARTIFACT_DIR, SEMANTIC_ARTIFACT_DIR, SCHEMES_CSV
)

logger = logging.getLogger("jansahay.search_service")

# ── Module-level singletons ───────────────────────────────────────────────────
_tfidf:    TfidfRetriever    | None = None
_semantic: SemanticRetriever | None = None
_schemes_df: pd.DataFrame    | None = None


def load_all_artifacts() -> None:
    """
    Called ONCE at application startup via the FastAPI lifespan handler.
    Loads schemes CSV, TF-IDF artifact, and semantic embedding artifact.
    Raises RuntimeError with a clear message if anything is missing or stale.
    """
    global _tfidf, _semantic, _schemes_df

    # 1. Load scheme metadata
    if not os.path.exists(SCHEMES_CSV):
        raise RuntimeError(
            f"Schemes CSV not found: {SCHEMES_CSV}\n"
            "Run 'python scripts/process_data.py' to generate it."
        )
    _schemes_df = pd.read_csv(SCHEMES_CSV)
    logger.info(f"Loaded {len(_schemes_df)} schemes from {SCHEMES_CSV}")

    # 2. Load TF-IDF retriever
    if not os.path.exists(TFIDF_ARTIFACT_DIR):
        raise RuntimeError(
            f"TF-IDF artifact directory not found: {TFIDF_ARTIFACT_DIR}\n"
            "Run 'python scripts/train_tfidf.py' to build it."
        )
    try:
        _tfidf = TfidfRetriever.load(TFIDF_ARTIFACT_DIR, current_df=_schemes_df)
        logger.info(f"TF-IDF retriever loaded. Features: {_tfidf.tfidf_matrix.shape[1]}")
    except ValueError as exc:
        raise RuntimeError(f"TF-IDF artifact stale or corrupted: {exc}") from exc

    # 3. Load Semantic retriever (model loaded lazily on first encode call)
    if not os.path.exists(SEMANTIC_ARTIFACT_DIR):
        raise RuntimeError(
            f"Semantic artifact directory not found: {SEMANTIC_ARTIFACT_DIR}\n"
            "Run 'python scripts/build_semantic_index.py' to build it."
        )
    try:
        _semantic = SemanticRetriever.load(SEMANTIC_ARTIFACT_DIR, current_df=_schemes_df)
        # Eagerly load the embedding model so the first real request is fast
        _semantic._load_model()
        logger.info(
            f"Semantic retriever loaded. Model: {_semantic.model_name} "
            f"Dim: {_semantic._embedding_dim}"
        )
    except ValueError as exc:
        raise RuntimeError(f"Semantic artifact stale or corrupted: {exc}") from exc


def get_tfidf() -> TfidfRetriever:
    if _tfidf is None:
        raise RuntimeError("TF-IDF retriever has not been loaded yet.")
    return _tfidf


def get_semantic() -> SemanticRetriever:
    if _semantic is None:
        raise RuntimeError("Semantic retriever has not been loaded yet.")
    return _semantic


def get_schemes_df() -> pd.DataFrame:
    if _schemes_df is None:
        raise RuntimeError("Scheme metadata has not been loaded yet.")
    return _schemes_df


# ── Score types ───────────────────────────────────────────────────────────────
SCORE_TYPE_TFIDF    = "TF-IDF cosine similarity"
SCORE_TYPE_SEMANTIC = "semantic cosine similarity"


def _tfidf_explanation(query: str, scheme_id: int) -> str | None:
    """Return matched TF-IDF terms for a scheme, or None on failure."""
    try:
        exp = get_tfidf().explain(query, scheme_id)
        terms = exp.get("matched_terms", [])
        if terms:
            top = ", ".join(t["term"] for t in terms[:5])
            return f"Matched lexical terms: {top}"
        return "No significant lexical overlap terms found."
    except Exception:
        return None


def _sem_explanation() -> str:
    return (
        "Semantic similarity based on multilingual sentence embeddings "
        f"({get_semantic().model_name}). "
        "The model encodes both the query and scheme documents into a shared "
        "768-dimensional vector space, enabling cross-lingual matching."
    )


def _build_scheme_results(raw_results: list, score_type: str,
                          include_explanation: bool,
                          query: str, model: str) -> list[dict]:
    """
    Converts raw retriever output dicts into the SchemeResult shape expected
    by the API, adding score_type and optional explanation fields.
    """
    out = []
    for r in raw_results:
        result = {
            "scheme_id"      : r["scheme_id"],
            "scheme_name"    : r["scheme_name"],
            "description"    : r.get("description", ""),
            "benefits"       : r.get("benefits", ""),
            "eligibility"    : r.get("eligibility", ""),
            "level"          : r.get("level", ""),
            "intent"         : r.get("intent", ""),
            "similarity_score": r["similarity_score"],
            "score_type"     : score_type,
            "rank"           : r.get("rank", 0),
            "low_relevance"  : r.get("low_relevance", False),
            "explanation"    : None,
        }
        if include_explanation:
            if model == "tfidf":
                result["explanation"] = _tfidf_explanation(query, r["scheme_id"])
            else:
                result["explanation"] = _sem_explanation()
        out.append(result)
    return out


def run_search(
    query: str,
    model: str,
    top_k: int,
    level: str | None,
    intent: str | None,
    include_explanation: bool,
) -> dict:
    """
    Central search function.  Called by API route handlers.
    Returns a dict that maps directly to SearchResponse fields.
    """
    base = {"query": query, "model": model}

    # ── TF-IDF only ────────────────────────────────────────────────────────
    if model == "tfidf":
        raw = get_tfidf().search(
            query, top_k=top_k,
            level_filter=level, intent_filter=intent
        )
        if raw["status"] == "no_lexical_overlap":
            return {**base, "status": "no_results",
                    "message": (
                        "TF-IDF found no lexical overlap between the query and "
                        "scheme documents.  This often happens with Hindi or "
                        "Hinglish queries.  Try model='semantic'."
                    ),
                    "results": []}
        results = _build_scheme_results(
            raw["results"], SCORE_TYPE_TFIDF,
            include_explanation, query, "tfidf"
        )
        return {**base, "status": raw["status"], "message": None,
                "results": results}

    # ── Semantic only ──────────────────────────────────────────────────────
    if model == "semantic":
        raw = get_semantic().search(
            query, top_k=top_k,
            level_filter=level, intent_filter=intent
        )
        results = _build_scheme_results(
            raw["results"], SCORE_TYPE_SEMANTIC,
            include_explanation, query, "semantic"
        )
        status = raw["status"]   # 'success' or 'low_relevance_warning'
        msg = (
            "All top results have similarity score below the heuristic "
            "threshold (0.30). Results may not be meaningful for this query."
            if status == "low_relevance_warning" else None
        )
        return {**base, "status": status, "message": msg, "results": results}

    # ── Both ───────────────────────────────────────────────────────────────
    # model == "both"
    raw_t = get_tfidf().search(
        query, top_k=top_k, level_filter=level, intent_filter=intent
    )
    raw_s = get_semantic().search(
        query, top_k=top_k, level_filter=level, intent_filter=intent
    )

    tfidf_results = [] if raw_t["status"] == "no_lexical_overlap" else \
        _build_scheme_results(raw_t["results"], SCORE_TYPE_TFIDF,
                              include_explanation, query, "tfidf")
    sem_results = _build_scheme_results(
        raw_s["results"], SCORE_TYPE_SEMANTIC,
        include_explanation, query, "semantic"
    )

    # Merged "results" field: de-duplicated union, semantic score preferred
    seen: dict[int, dict] = {}
    for r in sem_results:
        seen[r["scheme_id"]] = r
    for r in tfidf_results:
        if r["scheme_id"] not in seen:
            seen[r["scheme_id"]] = r
    merged = sorted(seen.values(), key=lambda x: x["similarity_score"], reverse=True)
    for i, r in enumerate(merged, 1):
        r["rank"] = i

    t_status = "no_results" if raw_t["status"] == "no_lexical_overlap" else raw_t["status"]
    s_status = raw_s["status"]
    combined_status = "success" if (tfidf_results or sem_results) else "no_results"

    return {
        **base,
        "status"          : combined_status,
        "message"         : None,
        "results"         : merged[:top_k],
        "tfidf_results"   : tfidf_results,
        "semantic_results": sem_results,
    }
