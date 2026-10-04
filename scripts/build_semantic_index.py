"""
Build and persist the semantic embedding index for all 115 scheme documents.
Run once; do NOT re-run unnecessarily (model download is large).

Usage:
    python scripts/build_semantic_index.py
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from jansahay.retrieval.semantic import SemanticRetriever

def build_index():
    schemes_path = "data/processed/schemes_clean.csv"
    if not os.path.exists(schemes_path):
        print("Processed schemes not found — running preprocessing pipeline …")
        from scripts.process_data import run_preprocessing_pipeline
        run_preprocessing_pipeline()

    df = pd.read_csv(schemes_path)
    print(f"Loaded {len(df)} schemes.")

    retriever = SemanticRetriever()
    retriever.build_index(df, text_column="doc_text_semantic")
    retriever.save("data/artifacts/semantic")
    print("Done.")

if __name__ == "__main__":
    build_index()
