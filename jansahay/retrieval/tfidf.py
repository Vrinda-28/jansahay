import os
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from jansahay.preprocessing.text_cleaner import preprocess_for_tfidf

class TfidfRetriever:
    """
    Classical TF-IDF Retrieval Baseline for JANSAHAY.
    
    Fits TF-IDF vectorizer exclusively on preprocessed scheme document text.
    Strictly isolated from query metadata or annotations to prevent data leakage.
    """
    def __init__(self, vectorizer_params=None):
        self.vectorizer_params = vectorizer_params or {
            'ngram_range': (1, 2),
            'sublinear_tf': True,
            'norm': 'l2',
            'min_df': 1
        }
        self.vectorizer = None
        self.tfidf_matrix = None
        self.schemes_df = None
        self.doc_hash = None
        self.is_fitted = False

    def _compute_doc_hash(self, texts: list) -> str:
        """Computes SHA256 hash of preprocessed document corpus for staleness check."""
        corpus_str = "||".join(texts)
        return hashlib.sha256(corpus_str.encode('utf-8')).hexdigest()

    def fit(self, df_or_docs, text_column='doc_text'):
        """
        Fits TF-IDF vectorizer on searchable scheme documents.
        """
        if isinstance(df_or_docs, pd.DataFrame):
            df = df_or_docs.copy()
        else:
            df = pd.DataFrame(df_or_docs)

        if text_column not in df.columns:
            if 'doc_text_tfidf' in df.columns:
                text_column = 'doc_text_tfidf'
            elif 'doc_text' in df.columns:
                df['doc_text_tfidf'] = df['doc_text'].apply(preprocess_for_tfidf)
                text_column = 'doc_text_tfidf'
            else:
                raise ValueError(f"Column '{text_column}' not found in training data.")

        # Ensure doc_text_tfidf exists and is preprocessed
        if 'doc_text_tfidf' not in df.columns or df['doc_text_tfidf'].isnull().any():
            df['doc_text_tfidf'] = df[text_column].apply(preprocess_for_tfidf)

        preprocessed_texts = df['doc_text_tfidf'].tolist()
        self.doc_hash = self._compute_doc_hash(preprocessed_texts)

        # Initialize and fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        self.tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
        self.schemes_df = df.reset_index(drop=True)
        self.is_fitted = True

        return self

    def search(self, query: str, top_k: int = 5, level_filter: str = None, intent_filter: str = None, min_similarity: float = 1e-6) -> dict:
        """
        Searches schemes matching preprocessed query using TF-IDF and Cosine Similarity.
        Does NOT refit the vectorizer.
        """
        if not self.is_fitted:
            raise ValueError("TfidfRetriever is not fitted. Call fit() or load() before searching.")

        # 1. Preprocess query
        query_tfidf = preprocess_for_tfidf(query)

        # 2. Transform query
        query_vec = self.vectorizer.transform([query_tfidf])

        # 3. Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        max_sim = float(np.max(similarities)) if len(similarities) > 0 else 0.0

        # 4. Filter by Level and Intent
        candidate_indices = []
        for idx in range(len(self.schemes_df)):
            row = self.schemes_df.iloc[idx]
            
            # Apply Level filter if specified
            if level_filter:
                if str(row.get('Level', '')).strip().lower() != level_filter.strip().lower():
                    continue

            # Apply Intent filter if specified
            if intent_filter:
                if str(row.get('Intent', '')).strip().lower() != intent_filter.strip().lower():
                    continue

            candidate_indices.append(idx)

        # 5. Extract non-zero similarity matches
        results = []
        for idx in candidate_indices:
            score = float(similarities[idx])
            if score < min_similarity:
                continue

            row = self.schemes_df.iloc[idx]
            results.append({
                'scheme_id': int(row.get('ID', idx + 1)),
                'scheme_name': str(row.get('Scheme_Name', '')),
                'similarity_score': round(float(score), 4),
                'level': str(row.get('Level', '')),
                'intent': str(row.get('Intent', '')),
                'description': str(row.get('Description', '')),
                'benefits': str(row.get('Benefits', '')),
                'eligibility': str(row.get('Eligibility', ''))
            })

        # If no positive similarity scores match
        if not results:
            return {
                'status': 'no_lexical_overlap',
                'query': query,
                'query_tfidf': query_tfidf,
                'top_k': top_k,
                'max_similarity': round(max_sim, 4),
                'results': []
            }

        # Sort by similarity_score descending
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Apply top_k
        top_results = results[:top_k]
        for rank, res in enumerate(top_results, 1):
            res['rank'] = rank

        return {
            'status': 'success',
            'query': query,
            'query_tfidf': query_tfidf,
            'top_k': top_k,
            'total_matches': len(results),
            'results': top_results
        }

    def rank_all(self, query: str) -> list:
        """Returns similarity scores for all scheme documents."""
        if not self.is_fitted:
            raise ValueError("TfidfRetriever is not fitted.")

        query_tfidf = preprocess_for_tfidf(query)
        query_vec = self.vectorizer.transform([query_tfidf])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        ranked = []
        for idx, score in enumerate(similarities):
            scheme_id = int(self.schemes_df.iloc[idx].get('ID', idx + 1))
            ranked.append((scheme_id, float(score)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def explain(self, query: str, scheme_id: int) -> dict:
        """
        Explains TF-IDF matching terms contributing to similarity score for a given scheme.
        """
        if not self.is_fitted:
            raise ValueError("TfidfRetriever is not fitted.")

        query_tfidf = preprocess_for_tfidf(query)
        query_vec = self.vectorizer.transform([query_tfidf]).toarray().flatten()

        # Find row for scheme_id
        matches = self.schemes_df[self.schemes_df['ID'] == scheme_id]
        if matches.empty:
            return {'error': f"Scheme ID {scheme_id} not found."}

        doc_idx = matches.index[0]
        doc_vec = self.tfidf_matrix[doc_idx].toarray().flatten()

        feature_names = self.vectorizer.get_feature_names_out()

        # Find overlapping terms where both query and document have non-zero weight
        overlapping = []
        for idx in range(len(feature_names)):
            q_weight = query_vec[idx]
            d_weight = doc_vec[idx]
            if q_weight > 0 and d_weight > 0:
                contribution = q_weight * d_weight
                overlapping.append({
                    'term': feature_names[idx],
                    'query_tfidf': round(float(q_weight), 4),
                    'doc_tfidf': round(float(d_weight), 4),
                    'term_contribution': round(float(contribution), 4)
                })

        overlapping.sort(key=lambda x: x['term_contribution'], reverse=True)

        return {
            'scheme_id': scheme_id,
            'scheme_name': str(self.schemes_df.iloc[doc_idx].get('Scheme_Name', '')),
            'query': query,
            'query_tfidf': query_tfidf,
            'matched_terms_count': len(overlapping),
            'matched_terms': overlapping
        }

    def save(self, dir_path: str = 'data/artifacts/tfidf'):
        """Saves fitted TF-IDF model artifacts to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted TfidfRetriever.")

        os.makedirs(dir_path, exist_ok=True)

        # 1. Save Vectorizer
        joblib.dump(self.vectorizer, os.path.join(dir_path, 'vectorizer.joblib'))

        # 2. Save TF-IDF Matrix
        joblib.dump(self.tfidf_matrix, os.path.join(dir_path, 'tfidf_matrix.joblib'))

        # 3. Save Schemes Metadata
        self.schemes_df.to_csv(os.path.join(dir_path, 'schemes_metadata.csv'), index=False, encoding='utf-8')

        # 4. Save Config & Hash
        config = {
            'vectorizer_params': self.vectorizer_params,
            'num_schemes': int(len(self.schemes_df)),
            'num_features': int(self.tfidf_matrix.shape[1]),
            'doc_hash': self.doc_hash
        }
        with open(os.path.join(dir_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"TfidfRetriever artifacts successfully saved to {dir_path}")

    @classmethod
    def load(cls, dir_path: str = 'data/artifacts/tfidf', current_df=None):
        """
        Loads saved TF-IDF retriever from disk.
        Includes Stale Model Protection against underlying dataset modifications.
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Artifact directory '{dir_path}' does not exist.")

        config_path = os.path.join(dir_path, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        instance = cls(vectorizer_params=config.get('vectorizer_params'))
        instance.vectorizer = joblib.load(os.path.join(dir_path, 'vectorizer.joblib'))
        instance.tfidf_matrix = joblib.load(os.path.join(dir_path, 'tfidf_matrix.joblib'))
        instance.schemes_df = pd.read_csv(os.path.join(dir_path, 'schemes_metadata.csv'))
        instance.doc_hash = config.get('doc_hash')
        instance.is_fitted = True

        # Stale Model Check
        if current_df is not None:
            if 'doc_text_tfidf' in current_df.columns:
                current_texts = current_df['doc_text_tfidf'].tolist()
            elif 'doc_text' in current_df.columns:
                current_texts = current_df['doc_text'].apply(preprocess_for_tfidf).tolist()
            else:
                current_texts = []

            if current_texts:
                current_hash = instance._compute_doc_hash(current_texts)
                if current_hash != instance.doc_hash:
                    raise ValueError(
                        "Stale model artifact detected! The underlying scheme dataset has changed "
                        "since the model was trained. Please rebuild/retrain the TF-IDF model."
                    )

        return instance
