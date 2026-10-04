"""
tests/test_api.py

Unit tests for FastAPI endpoints (Phase 6).
Uses TestClient to test routing, validation, and expected shapes,
reusing the actual pre-loaded artifacts to ensure End-to-End correctness.
"""
import pytest
from fastapi.testclient import TestClient

# Must import from backend.app.main
try:
    from app.main import app
except ImportError:
    from backend.app.main import app

# Create a test client
# FastApi TestClient natively handles the 'lifespan' events (loading models)
# when we use the context manager.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# ─────────────────────────────────────────────────────────────────────────────
# 1. Health
# ─────────────────────────────────────────────────────────────────────────────
def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["tfidf_loaded"] is True
    assert data["semantic_loaded"] is True
    assert data["scheme_count"] > 0
    assert "e5" in data["semantic_model"].lower()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Schemes
# ─────────────────────────────────────────────────────────────────────────────
def test_list_schemes_default(client):
    res = client.get("/api/schemes")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 0
    assert len(data["results"]) <= 20  # default limit
    assert "scheme_name" in data["results"][0]

def test_list_schemes_pagination(client):
    res = client.get("/api/schemes?limit=5&offset=5")
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 5
    assert data["offset"] == 5

def test_list_schemes_filters(client):
    res = client.get("/api/schemes?level=Central")
    assert res.status_code == 200
    data = res.json()
    for s in data["results"]:
        assert s["level"].lower() == "central"

def test_get_scheme_by_id(client):
    # Scheme ID 1 should exist
    res = client.get("/api/schemes/1")
    assert res.status_code == 200
    assert res.json()["scheme_id"] == 1

def test_get_scheme_not_found(client):
    res = client.get("/api/schemes/999999")
    assert res.status_code == 404

# ─────────────────────────────────────────────────────────────────────────────
# 3. Search
# ─────────────────────────────────────────────────────────────────────────────
def test_search_english_semantic(client):
    req = {
        "query": "scholarship for engineering students",
        "model": "semantic",
        "top_k": 3
    }
    res = client.post("/api/search", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("success", "low_relevance_warning")
    assert data["model"] == "semantic"
    assert len(data["results"]) <= 3
    if data["results"]:
        assert "score_type" in data["results"][0]
        assert "semantic" in data["results"][0]["score_type"]

def test_search_hindi_tfidf_zero_overlap(client):
    req = {
        "query": "क्या सरकार व्यवसाय के लिए ब्याज सब्सिडी देती है",
        "model": "tfidf"
    }
    res = client.post("/api/search", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "no_results"
    assert len(data["results"]) == 0

def test_search_hinglish_both(client):
    req = {
        "query": "kisan machine subsidy",
        "model": "both",
        "top_k": 2
    }
    res = client.post("/api/search", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("success", "no_results")
    assert data["model"] == "both"
    assert "tfidf_results" in data
    assert "semantic_results" in data

def test_search_empty_query_rejected(client):
    req = {
        "query": "   ",
        "model": "semantic"
    }
    res = client.post("/api/search", json=req)
    # Pydantic validator should reject empty/whitespace queries
    assert res.status_code == 422 

def test_search_invalid_model_rejected(client):
    req = {
        "query": "test",
        "model": "magic_model"
    }
    res = client.post("/api/search", json=req)
    assert res.status_code == 422

def test_search_invalid_top_k(client):
    req = {
        "query": "test",
        "model": "semantic",
        "top_k": 50 # max is 10
    }
    res = client.post("/api/search", json=req)
    assert res.status_code == 422

def test_search_with_explanation(client):
    req = {
        "query": "scholarship",
        "model": "tfidf",
        "top_k": 1,
        "include_explanation": True
    }
    res = client.post("/api/search", json=req)
    assert res.status_code == 200
    data = res.json()
    if data["results"]:
        assert "explanation" in data["results"][0]
        assert data["results"][0]["explanation"] is not None
