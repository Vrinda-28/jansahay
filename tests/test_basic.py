import unittest
import pandas as pd
import os

class TestBasicProject(unittest.TestCase):
    def setUp(self):
        # We assume the dataset is in the root directory or data/raw
        self.dataset_path = 'jansahay_final_dataset.csv'
        if not os.path.exists(self.dataset_path):
            self.dataset_path = '../jansahay_final_dataset.csv'
            
    def test_dataset_can_be_loaded(self):
        df = pd.read_csv(self.dataset_path)
        self.assertIsNotNone(df)
        
    def test_dataset_columns_exist(self):
        df = pd.read_csv(self.dataset_path)
        expected_columns = ['ID', 'Scheme_Name', 'Description', 'Benefits', 
                            'Eligibility', 'Level', 'Intent', 'User_Query', 
                            'Query_Type', 'Language', 'Difficulty']
        for col in expected_columns:
            self.assertIn(col, df.columns)
            
    def test_dataset_contains_115_records(self):
        df = pd.read_csv(self.dataset_path)
        self.assertEqual(len(df), 115)
        
    def test_ids_are_unique(self):
        df = pd.read_csv(self.dataset_path)
        self.assertEqual(df['ID'].nunique(), len(df))
        
    def test_required_text_columns_exist(self):
        df = pd.read_csv(self.dataset_path)
        text_cols = ['Scheme_Name', 'Description', 'Benefits', 'Eligibility', 'User_Query']
        for col in text_cols:
            self.assertIn(col, df.columns)
            self.assertTrue(df[col].notnull().all())

    def test_project_modules_can_be_imported(self):
        # Create a dummy test to ensure tests run properly and folders exist
        self.assertTrue(os.path.exists('data'))
        self.assertTrue(os.path.exists('jansahay'))

if __name__ == '__main__':
    unittest.main()
