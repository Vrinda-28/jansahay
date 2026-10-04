"""
backend/app/schemas.py

Pydantic request/response models for all JanSahay API endpoints.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural-language query in English, Hindi, or Hinglish.",
        examples=["engineering student scholarship scheme",
                  "गरीब छात्रों के लिए पढ़ाई की सहायता",
                  "kisan ko machine kharidne ke liye sahayata"],
    )
    model: Literal["tfidf", "semantic", "both"] = Field(
        default="semantic",
        description=(
            "'tfidf' — TF-IDF cosine similarity (lexical, English-biased); "
            "'semantic' — multilingual sentence-embedding cosine similarity; "
            "'both' — returns results from both models."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of results to return (1–10).",
    )
    level: Optional[str] = Field(
        default=None,
        description="Filter by administrative level, e.g. 'State' or 'Central'.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Filter by scheme intent/domain, e.g. 'Education', 'Agriculture'.",
    )
    include_explanation: bool = Field(
        default=False,
        description=(
            "If true, include a brief explanation of why each result was retrieved. "
            "For TF-IDF: matched lexical terms. "
            "For Semantic: a fixed note about multilingual embeddings."
        ),
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only.")
        return v.strip()


class SchemeResult(BaseModel):
    scheme_id: int
    scheme_name: str
    description: str
    benefits: str
    eligibility: str
    level: str
    intent: str
    similarity_score: float = Field(
        description=(
            "Cosine similarity score in the range [−1, 1].  "
            "This is NOT a probability or confidence percentage."
        )
    )
    score_type: str = Field(
        description=(
            "'TF-IDF cosine similarity' or 'semantic cosine similarity'. "
            "Indicates which retrieval method produced this score."
        )
    )
    rank: int
    low_relevance: bool = Field(
        default=False,
        description=(
            "True when the similarity score is below the heuristic "
            "low-relevance threshold (0.30 for semantic).  "
            "Not applicable for TF-IDF (zero-overlap queries return no results)."
        ),
    )
    explanation: Optional[str] = Field(
        default=None,
        description=(
            "Brief explanation of why this result was retrieved.  "
            "Only present when include_explanation=true in the request."
        ),
    )


class SearchResponse(BaseModel):
    query: str
    model: str
    status: str = Field(
        description=(
            "'success' — at least one result returned; "
            "'no_results' — retrieval produced no matches (e.g. TF-IDF zero-overlap); "
            "'low_relevance_warning' — results returned but all below heuristic threshold."
        )
    )
    message: Optional[str] = None
    results: list[SchemeResult]
    tfidf_results: Optional[list[SchemeResult]] = Field(
        default=None,
        description="Only present when model='both'.",
    )
    semantic_results: Optional[list[SchemeResult]] = Field(
        default=None,
        description="Only present when model='both'.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schemes
# ─────────────────────────────────────────────────────────────────────────────

class SchemeDetail(BaseModel):
    scheme_id: int
    scheme_name: str
    description: str
    benefits: str
    eligibility: str
    level: str
    intent: str


class SchemesListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    results: list[SchemeDetail]


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    tfidf_loaded: bool
    semantic_loaded: bool
    scheme_count: int
    semantic_model: str
    tfidf_features: int
