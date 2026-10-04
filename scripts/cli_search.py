import os
import sys
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from jansahay.retrieval.tfidf import TfidfRetriever

def main():
    parser = argparse.ArgumentParser(description="JANSAHAY Classical TF-IDF Search CLI")
    parser.add_argument('--query', type=str, required=True, help="Search query string")
    parser.add_argument('--top-k', type=int, default=5, help="Number of top results to return")
    parser.add_argument('--level', type=str, default=None, help="Filter by Level (e.g. State, Central)")
    parser.add_argument('--intent', type=str, default=None, help="Filter by Intent (e.g. Education, Agriculture)")
    parser.add_argument('--explain', action='store_true', help="Print matched term explainability")
    parser.add_argument('--artifact-dir', type=str, default='data/artifacts/tfidf', help="Artifact directory path")

    args = parser.parse_args()

    # Ensure model is trained/saved
    if not os.path.exists(args.artifact_dir):
        print(f"Artifacts not found at '{args.artifact_dir}'. Training model first...")
        from scripts.train_tfidf import build_and_save_tfidf_model
        build_and_save_tfidf_model()

    # Load pre-trained model
    processed_schemes_path = 'data/processed/schemes_clean.csv'
    current_df = pd.read_csv(processed_schemes_path) if os.path.exists(processed_schemes_path) else None

    try:
        retriever = TfidfRetriever.load(args.artifact_dir, current_df=current_df)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Perform search
    res = retriever.search(
        query=args.query,
        top_k=args.top_k,
        level_filter=args.level,
        intent_filter=args.intent
    )

    print("\n==================================================")
    print(f"QUERY: '{res['query']}'")
    print(f"Preprocessed TF-IDF Query: '{res['query_tfidf']}'")
    print(f"Search Status: {res['status']}")
    print("==================================================")

    if res['status'] == 'no_lexical_overlap':
        print("\n[!] NO LEXICAL OVERLAP DETECTED.")
        print("Reason: No terms in the query match the indexed scheme document vocabulary.")
        print("No arbitrary schemes returned.")
        return

    results = res.get('results', [])
    print(f"Top {len(results)} Matches:")
    for r in results:
        print(f"\nRank {r['rank']} | Score (TF-IDF Cosine Sim): {r['similarity_score']}")
        print(f"Scheme ID  : {r['scheme_id']}")
        print(f"Scheme Name: {r['scheme_name']}")
        print(f"Level      : {r['level']} | Intent: {r['intent']}")

        if args.explain:
            exp = retriever.explain(args.query, r['scheme_id'])
            matched = exp.get('matched_terms', [])
            if matched:
                terms_str = ", ".join([f"'{m['term']}' (contrib: {m['term_contribution']})" for m in matched])
                print(f"Explainability Overlapping Terms: {terms_str}")
            else:
                print("Explainability: No direct overlapping feature unigrams/bigrams.")

if __name__ == '__main__':
    main()
