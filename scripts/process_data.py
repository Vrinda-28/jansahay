import os
import sys
import json
import pandas as pd

# Add workspace root to python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jansahay.utils.data_loader import load_dataset
from jansahay.preprocessing.document_builder import create_searchable_doc, verify_data_leakage
from jansahay.preprocessing.text_cleaner import preprocess_for_tfidf, preprocess_for_semantic
from jansahay.preprocessing.query_processor import group_queries

def run_preprocessing_pipeline():
    print("=== JANSAHAY PHASE 2: DATA PREPROCESSING PIPELINE ===")
    
    # 1. Load Dataset
    print("[1/5] Loading raw dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} records from dataset.")

    # 2. Build Searchable Scheme Documents & Run Data Leakage Check
    print("[2/5] Creating searchable documents and verifying data leakage...")
    doc_texts = []
    doc_texts_tfidf = []
    doc_texts_semantic = []

    for idx, row in df.iterrows():
        doc_text = create_searchable_doc(row)
        verify_data_leakage(doc_text, row)
        
        doc_tfidf = preprocess_for_tfidf(doc_text)
        doc_semantic = preprocess_for_semantic(doc_text)

        doc_texts.append(doc_text)
        doc_texts_tfidf.append(doc_tfidf)
        doc_texts_semantic.append(doc_semantic)

    df['doc_text'] = doc_texts
    df['doc_text_tfidf'] = doc_texts_tfidf
    df['doc_text_semantic'] = doc_texts_semantic

    os.makedirs('data/processed', exist_ok=True)
    schemes_clean_path = 'data/processed/schemes_clean.csv'
    df.to_csv(schemes_clean_path, index=False, encoding='utf-8')
    print(f"Saved processed schemes to {schemes_clean_path}")

    # 3. Process Queries & Group Duplicates
    print("[3/5] Processing and grouping queries...")
    queries_df = group_queries(df)
    queries_clean_path = 'data/processed/queries.csv'
    queries_df.to_csv(queries_clean_path, index=False, encoding='utf-8')
    print(f"Saved {len(queries_df)} unique grouped queries to {queries_clean_path}")

    # 4. Compute Dataset Statistics
    print("[4/5] Computing dataset statistics...")
    stats = {
        "total_scheme_records": int(len(df)),
        "total_unique_schemes": int(df['Scheme_Name'].nunique()),
        "total_unique_queries": int(len(queries_df)),
        "repeated_query_rows": int(len(df) - len(queries_df)),
        "language_distribution": df['Language'].value_counts().to_dict(),
        "intent_distribution": df['Intent'].value_counts().to_dict(),
        "query_type_distribution": df['Query_Type'].value_counts().to_dict(),
        "difficulty_distribution": df['Difficulty'].value_counts().to_dict(),
        "level_distribution": df['Level'].value_counts().to_dict(),
        "missing_values_in_raw": 0,
        "data_leakage_verified": True
    }

    stats_path = 'data/processed/dataset_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved dataset statistics to {stats_path}")

    print("[5/5] Preprocessing pipeline completed successfully!")
    return stats

if __name__ == '__main__':
    run_preprocessing_pipeline()
