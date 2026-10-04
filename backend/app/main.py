"""
backend/app/main.py

FastAPI application entrypoint.
Configures CORS, lifespan artifact loading, and registers API routers.
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager

import os
import sys

# Ensure backend directory and project root are in sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))             # .../backend/app
_BACKEND_DIR = os.path.abspath(os.path.join(_HERE, ".."))       # .../backend
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..")) # .../jansahay

for _p in (_BACKEND_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import config
from app.services import search_service
from app.routes import search, schemes, health

# Configure minimal stdout logging for the backend
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:\t  [%(name)s] %(message)s"
)
logger = logging.getLogger("jansahay.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
    Loads scheme metadata, TF-IDF vectors, and Semantic embeddings ONCE.
    If any artifact is missing or stale, this crashes loudly before taking traffic.
    """
    logger.info("Initializing JanSahay backend …")
    try:
        search_service.load_all_artifacts()
        logger.info("All NLP artifacts loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load NLP artifacts: {exc}")
        raise

    yield

    """
    Shutdown:
    Nothing to close explicitly (no DB connections).
    """
    logger.info("Shutting down JanSahay backend.")


app = FastAPI(
    title=config.APP_TITLE,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",           # Swagger UI
    redoc_url=None,             # Disable redoc
    openapi_url="/openapi.json",
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS allowed origins: {config.ALLOWED_ORIGINS}")

# ── API Routes ────────────────────────────────────────────────────────────────
app.include_router(health.router,  prefix=config.API_PREFIX, tags=["Health"])
app.include_router(search.router,  prefix=config.API_PREFIX, tags=["Search"])
app.include_router(schemes.router, prefix=config.API_PREFIX, tags=["Schemes"])

if __name__ == "__main__":
    import uvicorn
    # When run directly (e.g. `python backend/app/main.py`)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
