"""
CLI semantic search tool for JanSahay Phase 4.

Usage examples:
  python scripts/cli_semantic_search.py --query "engineering student scholarship scheme"
  python scripts/cli_semantic_search.py --query "गरीब छात्रों के लिए पढ़ाई की सहायता योजना"
  python scripts/cli_semantic_search.py --query "kisan ko machine kharidne ke liye sahayata yojana" --top-k 3
  python scripts/cli_semantic_search.py --query "housing scheme for poor families" --level State
"""
import os, sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from jansahay.retrieval.semantic import SemanticRetriever

def main():
    parser = argparse.ArgumentParser(description="JanSahay Semantic Search CLI")
    parser.add_argument("--query",        required=True, help="Search query")
    parser.add_argument("--top-k",        type=int, default=5)
    parser.add_argument("--level",        default=None, help="State | Central")
    parser.add_argument("--intent",       default=None)
    parser.add_argument("--artifact-dir", default="data/artifacts/semantic")
    args = parser.parse_args()

    # Build index if missing
    if not os.path.exists(args.artifact_dir):
        print("Semantic index not found — building now …")
        from scripts.build_semantic_index import build_index
        build_index()

    df = pd.read_csv("data/processed/schemes_clean.csv")
    try:
        retriever = SemanticRetriever.load(args.artifact_dir, current_df=df)
    except ValueError as e:
        print(f"Index error: {e}")
        sys.exit(1)

    # Load model for inference
    retriever._load_model()

    res = retriever.search(
        query        = args.query,
        top_k        = args.top_k,
        level_filter = args.level,
        intent_filter= args.intent,
    )

    print("\n" + "=" * 60)
    print(f"QUERY  : {res['query']}")
    print(f"MODEL  : {res['model']}")
    print(f"STATUS : {res['status']}")
    print("=" * 60)

    if res["status"] == "low_relevance_warning":
        print("\n[⚠] LOW RELEVANCE WARNING")
        print("All top results have similarity score below the heuristic threshold.")
        print("These results may not be meaningful for this query.\n")

    for r in res["results"]:
        flag = " [LOW RELEVANCE]" if r["low_relevance"] else ""
        print(f"\nRank {r['rank']}{flag}")
        print(f"  Scheme ID   : {r['scheme_id']}")
        print(f"  Scheme Name : {r['scheme_name']}")
        print(f"  Score (Semantic Cosine Sim): {r['similarity_score']}")
        print(f"  Level : {r['level']}  |  Intent : {r['intent']}")

if __name__ == "__main__":
    main()
