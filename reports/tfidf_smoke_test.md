# Classical TF-IDF Retrieval Baseline: Smoke Test Report

## 1. Overview & Goal
This report documents the smoke testing of the classical **TF-IDF + Cosine Similarity Baseline Retriever** (`TfidfRetriever`). The baseline establishes a benchmark for lexical search on the 115 Indian Government Welfare Scheme documents before introducing neural multilingual embeddings in Phase 4.

---

## 2. Vectorizer Configuration & Model Artifacts
- **Vectorizer Class**: `sklearn.feature_extraction.text.TfidfVectorizer`
- **Training Corpus**: Fitted EXCLUSIVELY on 115 preprocessed scheme documents (`doc_text_tfidf`). No `User_Query` or annotation metadata was included.
- **Parameters**:
  - `ngram_range`: `(1, 2)` (Unigrams + Bigrams)
  - `sublinear_tf`: `True` (Replaces TF with $1 + \log(\text{TF})$)
  - `norm`: `'l2'` (Euclidean vector normalization)
  - `min_df`: `1`
- **Total Features Extracted**: 6,129 vocabulary unigrams and bigrams.
- **Artifact Location**: `data/artifacts/tfidf/`
  - `vectorizer.joblib` (Fitted vectorizer)
  - `tfidf_matrix.joblib` (115 x 6129 sparse document matrix)
  - `schemes_metadata.csv` (Scheme records and IDs)
  - `config.json` (Configuration parameters & SHA256 document corpus hash for stale artifact protection)

---

## 3. Sample Searches & Results

### Case A: English Query (High Lexical Overlap)
- **Query**: `"engineering student scholarship scheme"`
- **Preprocessed Query**: `'engineering student scholarship scheme'`
- **Status**: `success`

**Top Matches**:
1. **Scheme ID 59**: AICTE – Saksham Scholarship Scheme For Specially-Abled Student (Degree)
   - **TF-IDF Cosine Similarity**: `0.1674`
   - **Matched Terms (Explainability)**: `'scholarship scheme'` (contrib: 0.0702), `'scholarship'` (contrib: 0.0534), `'student'` (contrib: 0.0393), `'scheme'` (contrib: 0.0046)
2. **Scheme ID 49**: AICTE – Saksham Scholarship Scheme For Specially-Abled Student (Diploma)
   - **TF-IDF Cosine Similarity**: `0.1669`
3. **Scheme ID 72**: AICTE-Swanath Scholarship Scheme For Students
   - **TF-IDF Cosine Similarity**: `0.1532`

---

### Case B: Hinglish / Code-Mixed Query (Partial Lexical Overlap)
- **Query**: `"poor students ke liye scholarship chahiye"`
- **Preprocessed Query**: `'poor students scholarship'` (Hinglish stopwords `'ke'`, `'liye'`, `'chahiye'` removed)
- **Status**: `success`

**Top Matches**:
1. **Scheme ID 72**: AICTE-Swanath Scholarship Scheme For Students
   - **TF-IDF Cosine Similarity**: `0.1087`
   - **Matched Terms (Explainability)**: `'scholarship'` (contrib: 0.0684), `'students'` (contrib: 0.0402)
2. **Scheme ID 3**: Pre-Matric Scholarship for Backward Class Students
   - **TF-IDF Cosine Similarity**: `0.1001`
3. **Scheme ID 19**: Garuda Scheme for Funeral Expense
   - **TF-IDF Cosine Similarity**: `0.0683`
   - **Matched Terms (Explainability)**: `'poor'` (contrib: 0.0683)

---

### Case C: Pure Hindi Devanagari Query (Zero Lexical Overlap)
- **Query**: `"गरीबों के लिए छात्रवृत्ति चाहिए"`
- **Preprocessed Query**: `'गरीबों छात्रवृत्ति'`
- **Status**: `no_lexical_overlap`
- **Max Similarity**: `0.0`
- **Returned Results**: `[]` (0 arbitrary schemes returned)

**Behavior Explanation**: Because the scheme documents are written primarily in English, Devanagari terms like `'गरीबों'` and `'छात्रवृत्ति'` share 0 unigram/bigram overlap with the TF-IDF vocabulary. Rather than returning arbitrary irrelevant schemes, the retriever accurately reports `no_lexical_overlap`. This failure case clearly demonstrates the fundamental limitation of lexical search across language boundaries.

---

### Case D: Unrelated / Out-Of-Domain Query
- **Query**: `"space rocket exploration mars mission"`
- **Preprocessed Query**: `'space rocket exploration mars mission'`
- **Status**: `success`

**Top Matches**:
1. **Scheme ID 80**: "Protected Cultivation" Component of "Establishment of New Gardens (Area Expansion)" Scheme
   - **TF-IDF Cosine Similarity**: `0.0870`
   - **Matched Terms (Explainability)**: `'mission'` (contrib: 0.0870)

**Behavior Explanation**: The word `'mission'` appeared in the scheme document ("Mission for Integrated Development of Horticulture"), leading to a low-confidence lexical match, while unrelated terms (`'space'`, `'rocket'`, `'mars'`) had zero overlap. Zero-similarity schemes were filtered out.

---

## 4. Key Limitations of Classical TF-IDF
1. **Cross-Lingual Blindness**: Fails on Hindi (Devanagari) queries when searching English scheme documents because term overlap is zero.
2. **Synonym Blindness**: Unable to match semantically identical words with different surface forms (e.g., `'home'` vs `'housing'`, `'pension'` vs `'old age assistance'`).
3. **Spelling & Transliteration Sensitivity**: Hinglish spelling variations (e.g., `'sahayata'` vs `'sahaita'`) fail to match unless exact term overlap occurs.

---

## 5. Conclusion & Next Steps
The `TfidfRetriever` baseline works deterministically and reliably for English and shared-vocabulary Hinglish queries, while cleanly handling zero-overlap scenarios for pure Hindi queries. 

> [!NOTE]
> This smoke test verifies functional baseline behavior. Formal quantitative evaluation (MRR, Recall@K, Precision@K) will be performed in Phase 5 to directly compare TF-IDF against Multilingual Embeddings.

### Phase 4 Objectives:
Implement **Multilingual Sentence Embeddings** (e.g., MuRIL / Multilingual Sentence-BERT) to bridge the cross-lingual gap identified in Case C.
