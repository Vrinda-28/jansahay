import pandas as pd
from jansahay.preprocessing.lang_detector import detect_language_heuristic

def group_queries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups unique User_Query entries, mapping each unique query to:
    - query_id (e.g., Q-001)
    - user_query
    - language
    - query_type
    - difficulty
    - intent
    - relevant_scheme_ids (comma-separated list of scheme IDs)
    """
    grouped_records = []
    
    # Maintain first-seen order of unique queries
    seen_queries = []
    query_map = {}
    
    for idx, row in df.iterrows():
        q_text = str(row['User_Query']).strip()
        scheme_id = row['ID']
        
        if q_text not in query_map:
            seen_queries.append(q_text)
            query_map[q_text] = {
                'user_query': q_text,
                'language': str(row['Language']).strip(),
                'query_type': str(row['Query_Type']).strip(),
                'difficulty': str(row['Difficulty']).strip(),
                'intent': str(row['Intent']).strip(),
                'relevant_scheme_ids': [scheme_id]
            }
        else:
            if scheme_id not in query_map[q_text]['relevant_scheme_ids']:
                query_map[q_text]['relevant_scheme_ids'].append(scheme_id)

    for i, q_text in enumerate(seen_queries, 1):
        q_data = query_map[q_text]
        # Re-detect or verify language heuristic
        detected_lang = detect_language_heuristic(q_text)
        
        grouped_records.append({
            'query_id': f"Q-{i:03d}",
            'user_query': q_data['user_query'],
            'language': q_data['language'],
            'detected_language': detected_lang,
            'query_type': q_data['query_type'],
            'difficulty': q_data['difficulty'],
            'intent': q_data['intent'],
            'relevant_scheme_ids': ",".join(str(sid) for sid in q_data['relevant_scheme_ids'])
        })
        
    return pd.DataFrame(grouped_records)
