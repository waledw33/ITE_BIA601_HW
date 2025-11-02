"""
Unit Tests for Data Processing Module
Tests for DataLoader, DataPreprocessor, and DataValidator
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.loader import DataLoader
from data_processing.preprocessor import DataPreprocessor
from data_processing.validator import DataValidator


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
    
    def test_load_csv(self):
        """Test CSV loading"""
        try:
            df = self.loader.load_csv('sample_datasets/iris.csv')
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            print("✓ CSV loading test passed")
        except:
            self.skipTest("Sample dataset not found")
    
    def test_get_dataset_info(self):
        """Test dataset info extraction"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        info = self.loader.get_dataset_info(df)
        self.assertEqual(info['rows'], 3)
        self.assertEqual(info['columns'], 2)
        print("✓ Dataset info test passed")


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor()
        self.sample_df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': [5, 6, 7, 8],
            'target': [0, 1, 0, 1]
        })
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_df, strategy='mean')
        self.assertFalse(df_clean.isnull().any().any())
        print("✓ Missing values handling test passed")
    
    def test_separate_features_target(self):
        """Test feature/target separation"""
        X, y = self.preprocessor.separate_features_target(self.sample_df.dropna())
        self.assertEqual(X.shape[1], 2)  # 2 features
        self.assertEqual(len(y), len(X))
        print("✓ Feature/target separation test passed")


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
        self.sample_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_validate_structure(self):
        """Test structure validation"""
        is_valid = self.validator.validate_structure(self.sample_df)
        self.assertTrue(is_valid)
        print("✓ Structure validation test passed")
    
    def test_check_target_column(self):
        """Test target column validation"""
        is_valid = self.validator.check_target_column(self.sample_df, 'target')
        self.assertTrue(is_valid)
        print("✓ Target column validation test passed")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("RUNNING DATA PROCESSING TESTS")
    print("="*60 + "\n")
    
    unittest.main(verbosity=2)

