import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

def analyze_dataset(file_path):
    df = pd.read_csv(file_path)
    
    print("--- DIMENSIONS ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n--- COLUMN NAMES & TYPES ---")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
        
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())
    
    print("\n--- EXACT DUPLICATE ROWS ---")
    print(df.duplicated().sum())
    
    print("\n--- DUPLICATE ANALYSIS ---")
    print(f"Duplicate Scheme_Name: {df.duplicated(subset=['Scheme_Name']).sum()}")
    print(f"Duplicate Description: {df.duplicated(subset=['Description']).sum()}")
    print(f"Duplicate Eligibility: {df.duplicated(subset=['Eligibility']).sum()}")
    print(f"Duplicate User_Query: {df.duplicated(subset=['User_Query']).sum()}")
    
    print("\n--- IDs ---")
    print(f"Unique IDs: {df['ID'].nunique()}")
    print(f"Total rows: {len(df)}")
    # Print first few and last few IDs to check range
    print(f"ID sample: {df['ID'].head(3).tolist()} ... {df['ID'].tail(3).tolist()}")
    
    print("\n--- UNIQUE VALUES ---")
    for col in ['Level', 'Intent', 'Query_Type', 'Language', 'Difficulty']:
        print(f"{col}: {df[col].unique()}")
        
    print("\n--- DISTRIBUTIONS ---")
    for col in ['Language', 'Intent', 'Query_Type', 'Difficulty', 'Level']:
        print(f"\n{col} Distribution:")
        print(df[col].value_counts())
        
    print("\n--- QUERY REUSE ---")
    unique_queries = df['User_Query'].nunique()
    total_queries = len(df['User_Query'])
    repeated_query_count = total_queries - unique_queries
    print(f"Total unique queries: {unique_queries}")
    print(f"Number of queries that are duplicates of another: {repeated_query_count}")
    
    q_counts = df['User_Query'].value_counts()
    repeated_q = q_counts[q_counts > 1]
    print(f"Queries appearing more than once: {len(repeated_q)}")
    
    print("Top repeated queries:")
    print(repeated_q.head(10))
    
    for q, count in repeated_q.head(5).items():
        schemes = df[df['User_Query'] == q]['Scheme_Name'].unique()
        print(f"\nQuery: '{q}' appears {count} times.")
        print(f"Associated schemes ({len(schemes)}): {schemes}")

    print("\n--- EXACT TEXT DUPLICATES ---")
    for col in ['Description', 'Benefits', 'Eligibility']:
        dup_counts = df[col].value_counts()
        dups = dup_counts[dup_counts > 1]
        print(f"\n{col} duplicates: {len(dups)} unique texts repeated")
        if len(dups) > 0:
            print(f"Top 3 repeated {col}s:")
            print(dups.head(3))
            
    print("\n--- SCHEME TEXT LANGUAGE ---")
    # Quick check for non-ascii characters in description to see if there is Hindi
    desc_non_ascii = df['Description'].apply(lambda x: not str(x).isascii() if pd.notnull(x) else False).sum()
    query_non_ascii = df['User_Query'].apply(lambda x: not str(x).isascii() if pd.notnull(x) else False).sum()
    print(f"Descriptions with non-ASCII chars (possible Hindi): {desc_non_ascii} / {len(df)}")
    print(f"User Queries with non-ASCII chars: {query_non_ascii} / {len(df)}")
    
    print("\nRepresentative Queries per language:")
    for lang in df['Language'].unique():
        sample = df[df['Language'] == lang]['User_Query'].head(2).tolist()
        print(f"[{lang}]: {sample}")

analyze_dataset('jansahay_final_dataset.csv')
