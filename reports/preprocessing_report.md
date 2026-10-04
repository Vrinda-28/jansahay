# Preprocessing Report: JANSAHAY Phase 2

## 1. Preprocessing Overview & Decisions
Phase 2 establishes the core preprocessing architecture for the JANSAHAY semantic search engine. The pipeline handles multilingual textual input across English, Hindi (Devanagari script), and Hinglish (code-mixed).

### Key Decisions:
1. **Dual Preprocessing Strategy**:
   - **TF-IDF Branch (`preprocess_for_tfidf`)**: Unicode normalization (NFKC), whitespace normalization, Devanagari-safe tokenization, lowercasing for ASCII characters, punctuation stripping, and stopword removal (English + Hindi + Hinglish).
   - **Semantic Embedding Branch (`preprocess_for_semantic`)**: Light normalization (Unicode NFKC + whitespace cleanup). Retains sentence structures, punctuation, word order, and stopwords intact to preserve semantic context for sentence embedding models.
2. **Conservative Stemming/Lemmatization**: Avoided aggressive stemming or language-blind lemmatization that corrupts Devanagari matras or misinterprets Hinglish terms.
3. **Strict Data Leakage Prevention**: Searchable scheme document text (`doc_text`) is built exclusively from scheme-content fields (`Scheme_Name`, `Description`, `Benefits`, `Eligibility`) and validated with automated leakage checks.

---

## 2. Before & After Preprocessing Examples

### A. Hindi Query Preprocessing
- **Original Query**: `"गरीबों के लिए छात्रवृत्ति चाहिए"`
- **Devanagari Tokenization**: `["गरीबों", "के", "लिए", "छात्रवृत्ति", "चाहिए"]`
- **TF-IDF Preprocessing Output**: `"गरीबों छात्रवृत्ति"` (Stopwords `'के'`, `'लिए'`, `'चाहिए'` removed; Devanagari characters preserved)
- **Semantic Preprocessing Output**: `"गरीबों के लिए छात्रवृत्ति चाहिए"` (Full natural context retained)

### B. English Preprocessing
- **Original Query**: `"Engineering Student Scholarship Scheme for 2026!"`
- **TF-IDF Preprocessing Output**: `"engineering student scholarship scheme 2026"` (Lowercased, punctuation removed, stopword `'for'` removed)
- **Semantic Preprocessing Output**: `"Engineering Student Scholarship Scheme for 2026!"` (Intact sentence)

### C. Hinglish (Code-Mixed) Preprocessing
- **Original Query**: `"poor students ke liye scholarship chahiye"`
- **Detected Language**: `Code-Mixed`
- **TF-IDF Preprocessing Output**: `"poor students scholarship"` (Hinglish stopwords `'ke'`, `'liye'`, `'chahiye'` removed, Latin lowercased)
- **Semantic Preprocessing Output**: `"poor students ke liye scholarship chahiye"` (Preserved code-mixed sentence structure)

---

## 3. Devanagari-Safe Tokenization & Unicode Normalization
Devanagari text requires careful handling so combining marks (matras, halant, nukta in Unicode range `\u0900-\u097F`) are not detached from consonant bases. 

The custom Devanagari-safe tokenizer utilizes the pattern:
```python
pattern = r'[\u0900-\u097F]+|[a-zA-Z0-9]+'
```
This guarantees that Devanagari words are extracted as coherent units without breaking matras or combining characters.

---

## 4. Rule-Based Heuristic Language Detection
The language detector utility (`detect_language_heuristic`) categorizes queries without external heavy model overhead:
- **Hindi**: Presence of Devanagari script (`\u0900-\u097F`) with no Latin text.
- **Code-Mixed**: Mixture of Devanagari and Latin, or Latin text containing recognized Hinglish marker keywords (`ke`, `liye`, `chahiye`, `sahayata`, `yojana`, `padhai`, `kisaan`, etc.).
- **English**: Purely Latin characters with standard vocabulary.

---

## 5. Query Grouping & Scheme Mapping
The raw dataset contains 115 records with repeated `User_Query` entries. During Phase 2, queries were grouped into `data/processed/queries.csv`:
- **Total Raw Records**: 115
- **Unique Grouped Queries**: 82
- **Repeated Queries**: 33 rows collapsed into relevant scheme mappings.
- **`relevant_scheme_ids` Mapping**:
  - Example Query: `"engineering student scholarship scheme"` (Query ID `Q-003`) -> Maps to Scheme IDs `[2, 89]`.
  - Example Query: `"क्या सरकार व्यवसाय के लिए ब्याज सब्सिडी देती है"` (Query ID `Q-001`) -> Maps to Scheme IDs `[1, 96]`.

---

## 6. Data Leakage Prevention Verification
To prevent target label leakage during search model evaluation:
- `doc_text` is created strictly using `Scheme_Name`, `Description`, `Benefits`, and `Eligibility`.
- Automated test `test_data_leakage_prevention` asserts that `User_Query:`, `Intent:`, `Query_Type:`, `Language:`, `Difficulty:`, and `Level:` headers or raw query strings never leak into `doc_text`.
- **Validation Result**: 100% of the 115 processed scheme records passed data leakage checks.

---

## 7. Generated Datasets Summary
- `data/processed/schemes_clean.csv`: 115 records containing `doc_text`, `doc_text_tfidf`, `doc_text_semantic`.
- `data/processed/queries.csv`: 82 unique query records with `query_id`, `user_query`, `language`, `detected_language`, `query_type`, `difficulty`, `intent`, `relevant_scheme_ids`.
- `data/processed/dataset_stats.json`: Statistical metrics summary.
