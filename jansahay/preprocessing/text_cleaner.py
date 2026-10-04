import re
import unicodedata

ENGLISH_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 
    'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 
    'below', 'between', 'both', 'but', 'by', 'can', 'can\'t', 'cannot', 'could', 
    'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 
    'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 
    'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 
    'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 
    'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 
    'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 
    'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 
    'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 
    'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 
    'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 
    'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 
    'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 
    'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 
    'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 
    'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 
    'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 
    'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
}

HINDI_STOPWORDS = {
    'के', 'लिए', 'का', 'की', 'को', 'में', 'पर', 'से', 'और', 'है', 'हैं', 'था', 
    'थी', 'थे', 'जाना', 'किया', 'करके', 'कहा', 'उत्तर', 'दिए', 'हुए', 'होने', 
    'होता', 'गया', 'सब', 'इस', 'उस', 'यह', 'वह', 'ने', 'या', 'भी', 'तक', 'एक', 
    'जो', 'तो', 'अपने', 'आप', 'अनुसार', 'अंदर', 'अदालत', 'अमल', 'अस्त', 'आदि'
}

HINGLISH_STOPWORDS = {
    'ke', 'liye', 'ka', 'ki', 'ko', 'se', 'me', 'mein', 'par', 'aur', 
    'hai', 'hain', 'bhi', 'tak', 'ek', 'jo', 'to', 'apne', 'aap', 'ya', 
    'chahiye', 'kya', 'kaise', 'tha', 'thi', 'the'
}

ALL_STOPWORDS = ENGLISH_STOPWORDS.union(HINDI_STOPWORDS).union(HINGLISH_STOPWORDS)

def normalize_unicode(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFKC', text)

def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_devanagari(text: str) -> list:
    """
    Devanagari-safe tokenizer.
    Extracts Devanagari words (including matras, combining characters, nukta: \\u0900-\\u097F)
    and Latin words ([a-zA-Z0-9]+).
    """
    if not text:
        return []
    text = normalize_unicode(text)
    pattern = r'[\u0900-\u097F]+|[a-zA-Z0-9]+'
    tokens = re.findall(pattern, text)
    return tokens

def preprocess_for_tfidf(text: str, remove_stopwords: bool = True) -> str:
    """
    Conservative TF-IDF preprocessing:
    - Unicode & whitespace normalization
    - Devanagari-safe tokenization
    - Lowercasing for Latin text (Devanagari untouched)
    - Punctuation removal
    - Optional Stopword removal (English, Hindi, Hinglish)
    """
    if not text or not isinstance(text, str):
        return ""
        
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    
    tokens = tokenize_devanagari(text)
    
    cleaned_tokens = []
    for token in tokens:
        # Lowercase Latin text
        token_lower = token.lower() if token.isascii() else token
        
        if remove_stopwords and token_lower in ALL_STOPWORDS:
            continue
            
        cleaned_tokens.append(token_lower)
        
    return " ".join(cleaned_tokens)

def preprocess_for_semantic(text: str) -> str:
    """
    Lighter preprocessing for semantic embedding models:
    - Unicode normalization
    - Whitespace normalization
    - Preserves sentence structure, word order, stopwords, and full context intact.
    """
    if not text or not isinstance(text, str):
        return ""
        
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text
