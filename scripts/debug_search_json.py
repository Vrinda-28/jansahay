import json
from fastapi.testclient import TestClient
try:
    from app.main import app
except ImportError:
    from backend.app.main import app

with TestClient(app) as client:
    res = client.post("/api/search", json={
        "query": "scholarship for engineering students",
        "model": "both",
        "top_k": 1,
        "include_explanation": True
    })
    
    print("Example search request (model=both):")
    print(json.dumps(res.json(), indent=2))
