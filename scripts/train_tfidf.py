import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jansahay.retrieval.tfidf import TfidfRetriever

def build_and_save_tfidf_model():
    print("=== BUILDING & SAVING TF-IDF RETRIEVER MODEL ===")
    processed_schemes_path = 'data/processed/schemes_clean.csv'
    
    if not os.path.exists(processed_schemes_path):
        from scripts.process_data import run_preprocessing_pipeline
        run_preprocessing_pipeline()

    df = pd.read_csv(processed_schemes_path)
    print(f"Loaded {len(df)} schemes from {processed_schemes_path}.")

    # Fit TF-IDF model strictly on scheme documents
    retriever = TfidfRetriever(vectorizer_params={
        'ngram_range': (1, 2),
        'sublinear_tf': True,
        'norm': 'l2',
        'min_df': 1
    })

    retriever.fit(df, text_column='doc_text_tfidf')

    artifact_dir = 'data/artifacts/tfidf'
    retriever.save(artifact_dir)
    print(f"TF-IDF Retriever training complete. Features: {retriever.tfidf_matrix.shape[1]}")

if __name__ == '__main__':
    build_and_save_tfidf_model()
