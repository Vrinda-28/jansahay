import unittest
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jansahay.utils.data_loader import load_dataset
from jansahay.preprocessing.text_cleaner import (
    normalize_unicode,
    normalize_whitespace,
    tokenize_devanagari,
    preprocess_for_tfidf,
    preprocess_for_semantic
)
from jansahay.preprocessing.lang_detector import detect_language_heuristic
from jansahay.preprocessing.document_builder import create_searchable_doc, verify_data_leakage
from jansahay.preprocessing.query_processor import group_queries

class TestPreprocessingPipeline(unittest.TestCase):

    def test_dataset_loading(self):
        df = load_dataset()
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 115)
        self.assertEqual(df['ID'].nunique(), 115)

    def test_missing_value_handling(self):
        df = load_dataset()
        for col in ['Description', 'Benefits', 'Eligibility', 'User_Query']:
            # No 'nan' string or NaN object should exist
            self.assertFalse((df[col] == 'nan').any())
            self.assertFalse(df[col].isnull().any())

    def test_unicode_normalization(self):
        text_raw = "ग़रीबों\u0947 के लिए" # Devanagari with combining characters
        normalized = normalize_unicode(text_raw)
        self.assertIsInstance(normalized, str)
        self.assertTrue(len(normalized) > 0)

    def test_hindi_tokenization(self):
        sample1 = "गरीबों के लिए छात्रवृत्ति चाहिए"
        sample2 = "किसानों के लिए सहायता योजना"
        
        tokens1 = tokenize_devanagari(sample1)
        tokens2 = tokenize_devanagari(sample2)
        
        self.assertEqual(tokens1, ["गरीबों", "के", "लिए", "छात्रवृत्ति", "चाहिए"])
        self.assertEqual(tokens2, ["किसानों", "के", "लिए", "सहायता", "योजना"])

    def test_english_preprocessing(self):
        sample = "Engineering Student Scholarship Scheme for 2026!"
        processed_tfidf = preprocess_for_tfidf(sample)
        processed_semantic = preprocess_for_semantic(sample)
        
        # TF-IDF removes stopwords ('for') and converts to lowercase
        self.assertNotIn("for", processed_tfidf.split())
        self.assertIn("engineering", processed_tfidf)
        # Semantic retains natural words
        self.assertIn("Engineering", processed_semantic)

    def test_hinglish_preprocessing(self):
        sample = "poor students ke liye scholarship chahiye"
        processed_tfidf = preprocess_for_tfidf(sample)
        processed_semantic = preprocess_for_semantic(sample)
        
        # TF-IDF removes Hinglish stopwords ('ke', 'liye', 'chahiye')
        self.assertNotIn("ke", processed_tfidf.split())
        self.assertNotIn("liye", processed_tfidf.split())
        self.assertIn("students", processed_tfidf)
        self.assertIn("scholarship", processed_tfidf)
        
        # Semantic preserves Hinglish text intact
        self.assertIn("ke", processed_semantic.split())

    def test_semantic_vs_tfidf_preprocessing(self):
        text = "Scheme for financial assistance to poor farmers in 2026."
        tfidf_out = preprocess_for_tfidf(text)
        semantic_out = preprocess_for_semantic(text)
        
        self.assertNotEqual(tfidf_out, semantic_out)
        self.assertTrue(len(semantic_out) >= len(tfidf_out))
        self.assertIn("for", semantic_out.split())
        self.assertNotIn("for", tfidf_out.split())

    def test_language_detection(self):
        self.assertEqual(detect_language_heuristic("गरीबों के लिए छात्रवृत्ति चाहिए"), "Hindi")
        self.assertEqual(detect_language_heuristic("poor students ke liye scholarship chahiye"), "Code-Mixed")
        self.assertEqual(detect_language_heuristic("scholarship for higher education"), "English")

    def test_query_grouping(self):
        df = load_dataset()
        grouped_df = group_queries(df)
        
        self.assertEqual(len(grouped_df), 82)
        self.assertIn('relevant_scheme_ids', grouped_df.columns)
        self.assertIn('query_id', grouped_df.columns)
        
        # Check repeated query example
        repeated_q = grouped_df[grouped_df['user_query'] == 'engineering student scholarship scheme']
        self.assertEqual(len(repeated_q), 1)
        scheme_ids = repeated_q.iloc[0]['relevant_scheme_ids'].split(',')
        self.assertEqual(len(scheme_ids), 2)

    def test_searchable_document_creation(self):
        sample_row = {
            'Scheme_Name': 'Test Welfare Scheme',
            'Description': 'This is a test description.',
            'Benefits': 'Financial grant of Rs 10000.',
            'Eligibility': 'Low income families.',
            'User_Query': 'how to get financial grant',
            'Intent': 'Financial Assistance',
            'Query_Type': 'Question',
            'Language': 'English',
            'Difficulty': 'Easy',
            'Level': 'Central'
        }
        
        doc_text = create_searchable_doc(sample_row)
        
        self.assertIn("Scheme Name: Test Welfare Scheme", doc_text)
        self.assertIn("Description: This is a test description.", doc_text)
        self.assertIn("Benefits: Financial grant of Rs 10000.", doc_text)
        self.assertIn("Eligibility: Low income families.", doc_text)

    def test_data_leakage_prevention(self):
        sample_row = {
            'Scheme_Name': 'Prime Minister Housing Scheme',
            'Description': 'Housing support for urban poor.',
            'Benefits': 'Subsidized house loan.',
            'Eligibility': 'Annual income below 3 Lakhs.',
            'User_Query': 'ghar ke liye sarkari yojana chahiye',
            'Intent': 'Housing',
            'Query_Type': 'Natural Language',
            'Language': 'Code-Mixed',
            'Difficulty': 'Medium',
            'Level': 'Central'
        }
        
        doc_text = create_searchable_doc(sample_row)
        
        # Verify that no leakage forbidden field header exists in doc_text
        for forbidden in ['User_Query', 'Intent', 'Query_Type', 'Language', 'Difficulty', 'Level']:
            self.assertNotIn(f"{forbidden}:", doc_text)
            
        # Verify function passes leakage check
        self.assertTrue(verify_data_leakage(doc_text, sample_row))
        
        # Test intentional leakage detection
        leaked_doc_text = doc_text + "\nUser_Query: ghar ke liye sarkari yojana"
        with self.assertRaises(ValueError):
            verify_data_leakage(leaked_doc_text, sample_row)

if __name__ == '__main__':
    unittest.main()
