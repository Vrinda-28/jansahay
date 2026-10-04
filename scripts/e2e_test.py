import httpx
import json

BASE_URL = "http://localhost:8000"

QUERIES = [
    "engineering student scholarship scheme",
    "गरीब छात्रों के लिए छात्रवृत्ति चाहिए",
    "poor students ke liye scholarship chahiye",
    "किसानों के लिए सहायता योजना",
    "kisan ko machine kharidne ke liye sahayata chahiye"
]

def run_tests():
    with httpx.Client(timeout=30.0) as client:
        print("1. Health Check")
        res = client.get(f"{BASE_URL}/api/health")
        print(json.dumps(res.json(), indent=2))
        
        print("\n2. Testing Queries (Both Models)")
        for query in QUERIES:
            print(f"\\nQuery: '{query}'")
            res = client.post(f"{BASE_URL}/api/search", json={
                "query": query,
                "model": "both",
                "top_k": 3,
                "include_explanation": True
            })
            if res.status_code != 200:
                print("Error:", res.text)
                continue
            data = res.json()
            print(f"Status: {data['status']}")
            print(f"TF-IDF matches: {len(data.get('tfidf_results', []))}")
            print(f"Semantic matches: {len(data.get('semantic_results', []))}")
            
            if data['results']:
                top = data['results'][0]
                print(f"Top overall result: {top['scheme_id']} - {top['scheme_name'][:50]}... [{top['score_type']}: {top['similarity_score']}]")

if __name__ == "__main__":
    run_tests()
