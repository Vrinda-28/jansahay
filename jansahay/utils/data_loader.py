import os
import pandas as pd
import numpy as np

MANDATORY_COLUMNS = [
    'ID', 'Scheme_Name', 'Description', 'Benefits', 
    'Eligibility', 'Level', 'Intent', 'User_Query', 
    'Query_Type', 'Language', 'Difficulty'
]

CANDIDATE_PATHS = [
    'jansahay_final_dataset(2).csv',
    'jansahay_final_dataset.csv',
    'data/raw/jansahay_final_dataset(2).csv',
    'data/raw/jansahay_final_dataset.csv'
]

def find_dataset_path(file_path=None):
    if file_path and os.path.exists(file_path):
        return file_path
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not locate jansahay_final_dataset(2).csv or jansahay_final_dataset.csv")

def load_dataset(file_path=None):
    actual_path = find_dataset_path(file_path)
    df = pd.read_csv(actual_path)
    
    # 1. Validate Columns
    missing_cols = [col for col in MANDATORY_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing mandatory columns: {missing_cols}")
        
    # 2. Check for missing Scheme_Name or ID
    if df['Scheme_Name'].isnull().any():
        invalid_rows = df[df['Scheme_Name'].isnull()].index.tolist()
        raise ValueError(f"Dataset contains missing Scheme_Name at row indices: {invalid_rows}")
        
    if df['ID'].isnull().any():
        invalid_rows = df[df['ID'].isnull()].index.tolist()
        raise ValueError(f"Dataset contains missing ID at row indices: {invalid_rows}")

    # 3. Validate IDs
    if df['ID'].nunique() != len(df):
        raise ValueError("Dataset contains duplicate IDs!")

    # 4. Clean Missing Values in Optional Text Fields safely (no literal 'nan')
    text_fields = ['Description', 'Benefits', 'Eligibility', 'User_Query', 'Level', 'Intent', 'Query_Type', 'Language', 'Difficulty']
    for col in text_fields:
        df[col] = df[col].fillna('').astype(str)
        # Ensure literal 'nan' string is removed if present
        df[col] = df[col].apply(lambda x: '' if x.strip().lower() == 'nan' else x.strip())

    return df
