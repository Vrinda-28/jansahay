"""
backend/app/routes/search.py  —  POST /api/search
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException

from app.schemas import SearchRequest, SearchResponse
from app.services import search_service

logger = logging.getLogger("jansahay.routes.search")
router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search government welfare schemes",
    description=(
        "Submit a natural-language query in English, Hindi, or Hinglish and "
        "retrieve ranked government welfare schemes using TF-IDF or multilingual "
        "sentence-embedding retrieval.  Similarity scores are cosine similarity "
        "values — NOT probabilities or confidence percentages."
    ),
)
async def search_schemes(request: SearchRequest) -> SearchResponse:
    logger.info(
        f"Search request — model={request.model!r} "
        f"top_k={request.top_k} query={request.query[:80]!r}"
    )
    try:
        result = search_service.run_search(
            query               = request.query,
            model               = request.model,
            top_k               = request.top_k,
            level               = request.level,
            intent              = request.intent,
            include_explanation = request.include_explanation,
        )
        return SearchResponse(**result)
    except RuntimeError as exc:
        logger.error(f"Search service error: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception(f"Unexpected error during search: {exc}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during retrieval."
        )
