import re

# Comprehensive list of common Hinglish / Romanized Hindi indicator words
HINGLISH_KEYWORDS = {
    'ke', 'liye', 'ko', 'ka', 'ki', 'se', 'me', 'mein', 'par', 'aur', 
    'hai', 'hain', 'bhi', 'tak', 'ek', 'jo', 'to', 'apne', 'aap', 'ya', 
    'karne', 'kya', 'kaise', 'chahiye', 'yojana', 'sahayata', 'kisaan', 
    'garibon', 'padhai', 'ladki', 'khabar', 'kaam', 'milega', 'kaise', 'kare'
}

def detect_language_heuristic(text: str) -> str:
    """
    Heuristic rule-based language detector.
    Categorizes text into: 'Hindi', 'Code-Mixed', or 'English'.
    """
    if not text or not isinstance(text, str):
        return 'English'
        
    devanagari_chars = re.findall(r'[\u0900-\u097F]', text)
    latin_chars = re.findall(r'[a-zA-Z]', text)
    
    num_dev = len(devanagari_chars)
    num_lat = len(latin_chars)
    
    # Pure or predominant Devanagari
    if num_dev > 0 and num_lat == 0:
        return 'Hindi'
        
    # Devanagari mixed with Latin
    if num_dev > 0 and num_lat > 0:
        return 'Code-Mixed'
        
    # Latin text check for Hinglish keywords
    words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    if words.intersection(HINGLISH_KEYWORDS):
        return 'Code-Mixed'
        
    if num_lat > 0:
        return 'English'
        
    return 'English'
