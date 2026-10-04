"""
backend/app/core/config.py

Application configuration — reads from environment variables with sensible
development defaults.  No secrets are required for local development.
"""
import os
from pathlib import Path

# ── Base paths ────────────────────────────────────────────────────────────────
# The backend app sits at backend/app/; the project root is two levels up.
_HERE = Path(__file__).resolve().parent          # backend/app/core
_APP  = _HERE.parent                             # backend/app
_BACKEND = _APP.parent                           # backend
PROJECT_ROOT = _BACKEND.parent                   # project root (jansahay/)

# ── Artifact directories ──────────────────────────────────────────────────────
TFIDF_ARTIFACT_DIR   = str(
    Path(os.environ.get("TFIDF_ARTIFACT_DIR",
                        str(PROJECT_ROOT / "data" / "artifacts" / "tfidf")))
)
SEMANTIC_ARTIFACT_DIR = str(
    Path(os.environ.get("SEMANTIC_ARTIFACT_DIR",
                        str(PROJECT_ROOT / "data" / "artifacts" / "semantic")))
)
SCHEMES_CSV = str(
    Path(os.environ.get("SCHEMES_CSV",
                        str(PROJECT_ROOT / "data" / "processed" / "schemes_clean.csv")))
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Space-separated list of allowed origins.  Defaults to localhost dev origins.
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000 http://localhost:3001 http://127.0.0.1:3000"
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split() if o.strip()]

# ── API ───────────────────────────────────────────────────────────────────────
API_PREFIX   = "/api"
APP_TITLE    = "JanSahay — Multilingual Scheme Search API"
APP_VERSION  = "1.0.0"
APP_DESCRIPTION = (
    "FastAPI backend for JanSahay, a multilingual semantic search engine "
    "for Indian government welfare schemes.  Exposes TF-IDF and multilingual "
    "embedding (intfloat/multilingual-e5-base) retrieval through a clean REST API."
)
