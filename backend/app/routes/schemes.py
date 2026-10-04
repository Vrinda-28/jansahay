"""
backend/app/routes/schemes.py  —  GET /api/schemes, GET /api/schemes/{scheme_id}
"""
from __future__ import annotations
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.schemas import SchemeDetail, SchemesListResponse
from app.services import search_service

logger = logging.getLogger("jansahay.routes.schemes")
router = APIRouter()


def _row_to_scheme(row) -> SchemeDetail:
    return SchemeDetail(
        scheme_id   = int(row["ID"]),
        scheme_name = str(row["Scheme_Name"]),
        description = str(row["Description"]),
        benefits    = str(row["Benefits"]),
        eligibility = str(row["Eligibility"]),
        level       = str(row["Level"]),
        intent      = str(row["Intent"]),
    )


@router.get(
    "/schemes",
    response_model=SchemesListResponse,
    summary="List all welfare schemes",
    description=(
        "Returns a paginated list of all 115 government welfare schemes. "
        "Supports optional filtering by administrative level and intent domain."
    ),
)
async def list_schemes(
    limit : int          = Query(default=20, ge=1, le=115, description="Max results to return."),
    offset: int          = Query(default=0,  ge=0,         description="Number of results to skip."),
    level : Optional[str]= Query(default=None, description="Filter by Level: 'State' or 'Central'."),
    intent: Optional[str]= Query(default=None, description="Filter by Intent domain."),
) -> SchemesListResponse:
    df = search_service.get_schemes_df().copy()

    if level:
        df = df[df["Level"].str.strip().str.lower() == level.strip().lower()]
    if intent:
        df = df[df["Intent"].str.strip().str.lower() == intent.strip().lower()]

    total    = len(df)
    page     = df.iloc[offset : offset + limit]
    results  = [_row_to_scheme(row) for _, row in page.iterrows()]

    return SchemesListResponse(total=total, limit=limit, offset=offset, results=results)


@router.get(
    "/schemes/{scheme_id}",
    response_model=SchemeDetail,
    summary="Get a single scheme by ID",
    description="Returns complete information for the scheme with the given ID.",
)
async def get_scheme(scheme_id: int) -> SchemeDetail:
    df = search_service.get_schemes_df()
    matches = df[df["ID"] == scheme_id]
    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Scheme with ID {scheme_id} not found."
        )
    return _row_to_scheme(matches.iloc[0])
