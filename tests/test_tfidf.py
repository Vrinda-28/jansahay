import unittest
import os
import shutil
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jansahay.retrieval.tfidf import TfidfRetriever
from jansahay.utils.data_loader import load_dataset
from jansahay.preprocessing.document_builder import create_searchable_doc, verify_data_leakage

class TestTfidfRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = 'data/test_artifacts_tfidf'
        df = load_dataset()
        if 'doc_text_tfidf' not in df.columns:
            from jansahay.preprocessing.text_cleaner import preprocess_for_tfidf
            df['doc_text'] = df.apply(create_searchable_doc, axis=1)
            df['doc_text_tfidf'] = df['doc_text'].apply(preprocess_for_tfidf)
        cls.df = df

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_fit(self):
        retriever = TfidfRetriever()
        retriever.fit(self.df)
        
        self.assertTrue(retriever.is_fitted)
        self.assertIsNotNone(retriever.vectorizer)
        self.assertIsNotNone(retriever.tfidf_matrix)
        self.assertEqual(retriever.tfidf_matrix.shape[0], len(self.df))

    def test_save_and_load(self):
        retriever = TfidfRetriever()
        retriever.fit(self.df)
        retriever.save(self.test_dir)
        
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'vectorizer.joblib')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'tfidf_matrix.joblib')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'config.json')))

        loaded_retriever = TfidfRetriever.load(self.test_dir, current_df=self.df)
        self.assertTrue(loaded_retriever.is_fitted)
        self.assertEqual(loaded_retriever.tfidf_matrix.shape, retriever.tfidf_matrix.shape)

    def test_stale_artifact_detection(self):
        retriever = TfidfRetriever()
        retriever.fit(self.df)
        retriever.save(self.test_dir)

        # Create modified df with altered text
        modified_df = self.df.copy()
        modified_df.at[0, 'doc_text_tfidf'] = "completely altered document text for staleness test"

        with self.assertRaises(ValueError) as ctx:
            TfidfRetriever.load(self.test_dir, current_df=modified_df)
            
        self.assertIn("Stale model artifact detected", str(ctx.exception))

    def test_deterministic_ranking_and_top_k(self):
        retriever = TfidfRetriever()
        retriever.fit(self.df)

        query = "engineering student scholarship scheme"
        res1 = retriever.search(query, top_k=5)
        res2 = retriever.search(query, top_k=5)

        self.assertEqual(res1['status'], 'success')
        self.assertEqual(len(res1['results']), 5)
        # Verify rank ordering and deterministic reproducibility
        scores1 = [r['similarity_score'] for r in res1['results']]
        scores2 = [r['similarity_score'] for r in res2['results']]
        self.assertEqual(scores1, scores2)
        self.assertTrue(all(scores1[i] >= scores1[i+1] for i in range(len(scores1)-1)))

    def test_filtering(self):
        retriever = TfidfRetriever()
        retriever.fit(self.df)

        query = "scholarship scheme for students"
        
        # Test Level filter
        res_state = retriever.search(query, level_filter='State')
        for r in res_state.get('results', []):
            self.assertEqual(r['level'].lower(), 'state')

        # Test Intent filter
        res_edu = retriever.search(query, intent_filter='Education')
        for r in res_edu.get('results', []):
            self.assertEqual(r['intent'].lower(), 'education')

    def test_zero_lexical_overlap(self):
        retriever = TfidfRetriever()
        retriever.fit(self.df)

        # Pure Devanagari query with terms non-existent in English scheme corpus
        query = "गरीबों के लिए छात्रवृत्ति चाहिए"
        res = retriever.search(query)

        self.assertEqual(res['status'], 'no_lexical_overlap')
        self.assertEqual(len(res['results']), 0)

    def test_explanation(self):
        retriever = TfidfRetriever()
        retriever.fit(self.df)

        query = "engineering student scholarship scheme"
        search_res = retriever.search(query, top_k=1)
        top_scheme_id = search_res['results'][0]['scheme_id']

        exp = retriever.explain(query, top_scheme_id)
        self.assertEqual(exp['scheme_id'], top_scheme_id)
        self.assertIn('matched_terms', exp)
        self.assertTrue(len(exp['matched_terms']) > 0)

    def test_no_retraining_during_search(self):
        retriever = TfidfRetriever()
        retriever.fit(self.df)

        vocab_size_before = len(retriever.vectorizer.vocabulary_)
        
        # Search queries with unseen Out-Of-Vocabulary words
        retriever.search("quantum computing superconductor supercomputer")
        
        vocab_size_after = len(retriever.vectorizer.vocabulary_)
        self.assertEqual(vocab_size_before, vocab_size_after)

if __name__ == '__main__':
    unittest.main()
