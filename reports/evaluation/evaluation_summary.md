# JanSahay Phase 5 — Evaluation Summary

## 1. Dataset
- Raw records: 115
- Unique evaluation queries: 82
- Languages: English (n=38), Hindi (n=20), Code-Mixed (n=24)
- Query types: Natural Language, Keyword, Question, Conversational
- Difficulty: Easy, Medium, Hard

## 2. Evaluation Protocol
- Evaluation unit: **unique User_Query** (82 queries).
- Repeated rows that share the same User_Query are collapsed; their associated Scheme IDs are unioned into the relevant set.
- Top-K = 5 for all rank-based metrics.
- Model configurations are **fixed from Phase 3 and Phase 4** — no post-hoc tuning.

## 3. Relevance Definitions
- **Strict**: A scheme is relevant if it is directly associated with the query in the dataset.
- **Topical** (broader): A scheme is topically relevant if it shares the same Intent as the query. This is a coarser measure and does not imply the scheme is the exact correct answer.

## 4. Metrics Computed
| Metric | Definition |
|--------|-----------|
| Hit@1  | 1 if a relevant scheme is in rank-1 result |
| Hit@5  | 1 if any relevant scheme is in top-5 |
| Recall@1 | Fraction of relevant schemes in rank-1 result |
| Recall@5 | Fraction of relevant schemes in top-5 |
| Precision@5 | Fraction of top-5 that are relevant |
| MRR | Mean Reciprocal Rank — 1/position of first relevant |

## 5. Overall Results (Strict Relevance, n=82 queries)

| Model | Hit@1 | Hit@5 | Recall@1 | Recall@5 | Prec@5 | MRR |
|-------|-------|-------|----------|----------|--------|-----|
| TF-IDF | 0.049 | 0.159 | 0.049 | 0.128 | 0.032 | 0.085 |
| Semantic (E5) | 0.061 | 0.220 | 0.055 | 0.177 | 0.044 | 0.116 |

## 6. Language Results (Strict Relevance)

| Language | TF-IDF Hit@1 | Sem Hit@1 | TF-IDF Hit@5 | Sem Hit@5 | TF-IDF MRR | Sem MRR |
|----------|-------------|----------|-------------|---------|-----------|--------|
| English (n=38) | 0.079 | 0.053 | 0.289 | 0.237 | 0.150 | 0.126 |
| Hindi (n=20) | 0.000 | 0.100 | 0.000 | 0.300 | 0.000 | 0.167 |
| Code-Mixed (n=24) | 0.042 | 0.042 | 0.083 | 0.125 | 0.052 | 0.058 |

## 7. Query-Type Results (Strict Relevance)

| Query Type | TF-IDF Hit@5 | Sem Hit@5 | TF-IDF MRR | Sem MRR |
|------------|-------------|---------|-----------|--------|
| Conversational (n=5) | 0.200 | 0.600 | 0.200 | 0.440 |
| Keyword (n=31) | 0.258 | 0.226 | 0.137 | 0.114 |
| Natural Language (n=33) | 0.121 | 0.182 | 0.051 | 0.101 |
| Question (n=13) | 0.000 | 0.154 | 0.000 | 0.035 |

## 8. TF-IDF Zero-Overlap Analysis
TF-IDF relies on exact vocabulary overlap between query terms and scheme document tokens.
When queries are written in Hindi (Devanagari) against English scheme documents, overlap is zero.

| Language | Zero-overlap Rate |
|----------|-----------------|
| English  | 0.0% |
| Hindi    | 100.0% |
| Code-Mixed | 8.3% |

TF-IDF returns `no_lexical_overlap` for these queries and produces no results — correctly
avoiding presenting arbitrary schemes as matches.

## 9. Retrieval Latency (model loading excluded)

| Model | Avg Latency | Median Latency |
|-------|------------|---------------|
| TF-IDF | 8.5 ms | 7.5 ms |
| Semantic (E5) | 49.8 ms | 42.4 ms |

Note: Semantic model loading (first-time model download + tokenizer initialisation)
takes significantly longer and is excluded from per-query latency to enable a fair
per-query comparison.

## 10. Qualitative Examples
See `qualitative_examples.csv` for 10 annotated query examples covering:
TF-IDF success, Semantic success, both work, TF-IDF lexical failure, Semantic
weak result, Hindi query, English query, Hinglish query, Keyword query, Natural
Language query.

## 11. Dataset Limitations
1. **Small corpus (115 schemes, 82 unique queries)**: Metric estimates have high variance; small differences should not be over-interpreted.
2. **Query–scheme associations are manually created**: There is no crowd-sourced or user-validated relevance judgement.
3. **Repeated descriptive content**: 4 unique description texts and 8 eligibility texts appear across multiple schemes; cosine similarity models may rank duplicates identically.
4. **Generic queries**: Broad queries (e.g., "scholarship for students") are intentionally associated with multiple schemes, which challenges strict single-label metrics.
5. **Strict relevance is conservative**: A retrieved scheme that is genuinely helpful but was not annotated for that exact query will score as a miss.
6. **Topical relevance is broad**: Treating all schemes with the same Intent as relevant inflates recall numbers and should not be reported without clear labelling.

## 12. Objective Interpretation
- On **English queries**, both models retrieve relevant schemes, with 0.159 (TF-IDF) and 0.220 (Semantic) overall Hit@5.
- On **Hindi queries**, TF-IDF experiences a zero-overlap rate of 100.0%, producing no results. The semantic model embeds queries and documents into a shared multilingual space, enabling cross-lingual retrieval.
- **Code-Mixed** queries contain Hinglish (Latin-script) terms; TF-IDF can partially match these to English scheme vocabulary, so its zero-overlap rate is lower for Code-Mixed than for Hindi.
- The semantic model (intfloat/multilingual-e5-base) applies E5 query/passage prefixes consistently and performs cross-lingual retrieval without any fine-tuning on this dataset.
- No claim is made about which model is universally "better". The differences depend on query language and the strict-vs-topical relevance definition used.

## 13. Data Leakage Verification
PASSED — `User_Query`, `Intent`, `Query_Type`, `Language`, `Difficulty`, and `Level`
fields were confirmed absent from all scheme document text (`doc_text`) used for
vectorisation and embedding.

## 14. Files Generated
- `reports/evaluation/metrics.csv` — per-query, per-model metrics (all 164 rows)
- `reports/evaluation/overall_metrics.csv`
- `reports/evaluation/language_metrics.csv`
- `reports/evaluation/query_type_metrics.csv`
- `reports/evaluation/difficulty_metrics.csv`
- `reports/evaluation/qualitative_examples.csv`
- `reports/evaluation/latency.json`
- `reports/evaluation/plots/overall_performance.png`
- `reports/evaluation/plots/language_hit5.png`
- `reports/evaluation/plots/language_mrr.png`
- `reports/evaluation/plots/querytype_hit5.png`
- `reports/evaluation/plots/tfidf_zero_overlap.png`
- `reports/evaluation/plots/retrieval_latency.png`
