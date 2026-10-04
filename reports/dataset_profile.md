# Dataset Profile: JANSAHAY Phase 1

## 1. Dataset Dimensions
- **Rows**: 115
- **Columns**: 11

## 2. Column Descriptions
- `ID`: Unique identifier (int64)
- `Scheme_Name`: Official name of the scheme (object)
- `Description`: Details about the scheme (object)
- `Benefits`: Financial or other benefits provided (object)
- `Eligibility`: Conditions to avail the scheme (object)
- `Level`: 'State' or 'Central' (object)
- `Intent`: Subject/Sector of the scheme (object)
- `User_Query`: Representative search query (object)
- `Query_Type`: 'Question', 'Natural Language', 'Keyword', 'Conversational' (object)
- `Language`: 'Hindi', 'Code-Mixed', 'English' (object)
- `Difficulty`: 'Medium', 'Easy', 'Hard' (object)

## 3. Missing-Value Analysis
- **0 missing values** across all columns. The dataset is fully complete.

## 4. Duplicate Analysis
- **Exact duplicate rows**: 0
- **Duplicate `Scheme_Name`**: 0 (All 115 schemes are unique)
- **Duplicate `Description`**: 7 occurrences of reused description texts
- **Duplicate `Eligibility`**: 12 occurrences of reused eligibility criteria
- **Duplicate `User_Query`**: 33 duplicate queries (referring to multiple schemes)

## 5. Language Distribution
- **English**: 52 (45.2%)
- **Code-Mixed**: 34 (29.6%)
- **Hindi**: 29 (25.2%)

## 6. Intent Distribution
- Financial Assistance: 30
- Education: 28
- Employment: 21
- Agriculture: 18
- Women Welfare: 9
- Social Welfare: 4
- Pension: 2
- Healthcare: 2
- Food Security: 1

## 7. Query Type Distribution
- Natural Language: 44
- Keyword: 44
- Question: 20
- Conversational: 7

## 8. Difficulty Distribution
- Medium: 60
- Easy: 41
- Hard: 14

## 9. Level Distribution
- State: 86
- Central: 29

## 10. Query Reuse Analysis
- **Total Unique Queries**: 82
- **Repeated Query Count**: 33 queries are re-used for different schemes.
- **Example Reused Queries**:
  - `क्या सरकार व्यवसाय के लिए ब्याज सब्सिडी देती है` (maps to 2 schemes)
  - `poor students ke liye padhai ki sahayata yojana` (maps to 2 schemes)
  - `engineering student scholarship scheme` (maps to 2 schemes)
  - `व्यवसाय शुरू करने के लिए सरकारी सब्सिडी कैसे मिलेगी` (maps to 2 schemes)
- Repeated queries represent the reality that one generic user query can successfully match multiple relevant schemes.

## 11. Description/Eligibility Duplication
- **Description**: 4 unique description texts are repeated across 7 schemes (e.g., umbrella schemes like "Aatmanirbhar Gujarat Scheme").
- **Eligibility**: 8 unique eligibility texts are repeated across 12 schemes (e.g., standard MSME requirements or fisherman community requirements).
- **Benefits**: No exact duplicate texts.

## 12. Important Observations for NLP Modeling
1. **Multilingual Nature**: Queries span three distinct representations. English is dominant, but Code-Mixed and Hindi combined form the majority. This justifies the need for robust multilingual embeddings (like MuRIL).
2. **Text Cleanliness**: The dataset is perfectly complete (no missing values) and scheme names are entirely unique. 
3. **Information Leakage**: The `User_Query`, `Intent`, `Query_Type`, `Language`, and `Difficulty` columns must be strictly excluded from the "scheme document" during vectorization to avoid data leakage.
4. **Target Context**: 14 scheme descriptions contain non-ASCII characters (likely Hindi/regional text), meaning the scheme side is not 100% English. The retrieval model must handle cross-lingual (Hindi query -> English text) and intra-lingual (Hindi -> Hindi) matches.
5. **Overlapping Targets**: Since 33 queries map to multiple schemes, evaluation metrics like Mean Reciprocal Rank (MRR) or Recall@K will be necessary, rather than absolute Accuracy@1.
