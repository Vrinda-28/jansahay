"""
backend/app/routes/health.py  —  GET /api/health
"""
from fastapi import APIRouter
from app.schemas import HealthResponse
from app.services import search_service

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Backend health check",
    description=(
        "Returns the operational status of the backend and its loaded components. "
        "Does NOT expose system internals, file paths, or model weights."
    ),
)
async def health() -> HealthResponse:
    tfidf_ok    = search_service._tfidf    is not None
    semantic_ok = search_service._semantic is not None
    scheme_count = len(search_service._schemes_df) if search_service._schemes_df is not None else 0
    sem_model    = search_service._semantic.model_name if semantic_ok else "not loaded"
    tfidf_feats  = (search_service._tfidf.tfidf_matrix.shape[1]
                    if tfidf_ok else 0)

    status = "ok" if (tfidf_ok and semantic_ok) else "degraded"

    return HealthResponse(
        status        = status,
        tfidf_loaded  = tfidf_ok,
        semantic_loaded= semantic_ok,
        scheme_count  = scheme_count,
        semantic_model= sem_model,
        tfidf_features= tfidf_feats,
    )
