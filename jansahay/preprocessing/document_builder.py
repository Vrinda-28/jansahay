import pandas as pd

LEAKAGE_FORBIDDEN_FIELDS = [
    'User_Query', 'Intent', 'Query_Type', 
    'Language', 'Difficulty', 'Level'
]

ALLOWED_DOCUMENT_FIELDS = [
    'Scheme_Name', 'Description', 'Benefits', 'Eligibility'
]

def create_searchable_doc(row) -> str:
    """
    Construct searchable scheme document text using ONLY scheme content fields:
    - Scheme_Name
    - Description
    - Benefits
    - Eligibility
    
    STRICTLY excludes query-side information and annotations:
    - User_Query, Intent, Query_Type, Language, Difficulty, Level.
    """
    def clean_val(val):
        if pd.isnull(val):
            return ""
        s = str(val).strip()
        return "" if s.lower() == 'nan' else s

    name = clean_val(row.get('Scheme_Name', ''))
    desc = clean_val(row.get('Description', ''))
    benefits = clean_val(row.get('Benefits', ''))
    eligibility = clean_val(row.get('Eligibility', ''))

    parts = []
    if name:
        parts.append(f"Scheme Name: {name}")
    if desc:
        parts.append(f"Description: {desc}")
    if benefits:
        parts.append(f"Benefits: {benefits}")
    if eligibility:
        parts.append(f"Eligibility: {eligibility}")

    doc_text = "\n\n".join(parts)
    return doc_text

def verify_data_leakage(doc_text: str, row) -> bool:
    """
    Verifies that no leakage forbidden field headers or metadata values exist in doc_text.
    Raises ValueError if data leakage is detected.
    """
    # 1. Check for field headers
    for forbidden_field in LEAKAGE_FORBIDDEN_FIELDS:
        header = f"{forbidden_field}:"
        if header.lower() in doc_text.lower():
            raise ValueError(f"Data leakage detected! Forbidden field header '{forbidden_field}' found in document text.")

    # 2. Check for User_Query explicit leakage if User_Query is distinct and long enough
    query = str(row.get('User_Query', '')).strip()
    if query and query.lower() != 'nan':
        # If the exact query string appears in doc_text (and isn't just a generic single word), flag it
        if len(query) > 10 and query.lower() in doc_text.lower():
            # Note: in rare cases a query might share words with scheme name, but exact user query string shouldn't be injected
            raise ValueError(f"Data leakage detected! User_Query string '{query}' found inside doc_text.")
            
    return True
