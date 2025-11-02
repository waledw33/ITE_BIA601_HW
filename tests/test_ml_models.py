"""
Unit Tests for ML Models Module
Tests for MLModels, ModelEvaluator, and PerformanceMetrics
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ml_models import MLModels
from models.evaluator import ModelEvaluator
from models.metrics import PerformanceMetrics


class TestMLModels(unittest.TestCase):
    """Test cases for MLModels class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        
        X, y = load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.ml = MLModels()
    
    def test_get_random_forest(self):
        """Test Random Forest model creation"""
        model = self.ml.get_random_forest()
        self.assertIsNotNone(model)
        print("✓ Random Forest creation test passed")
    
    def test_get_svm(self):
        """Test SVM model creation"""
        model = self.ml.get_svm()
        self.assertIsNotNone(model)
        print("✓ SVM creation test passed")
    
    def test_train_model(self):
        """Test model training"""
        model = self.ml.train(self.X_train, self.y_train, model_name='random_forest', n_estimators=10)
        self.assertIsNotNone(model)
        self.assertIsNotNone(self.ml.trained_model)
        print("✓ Model training test passed")
    
    def test_predict(self):
        """Test prediction"""
        self.ml.train(self.X_train, self.y_train, model_name='random_forest', n_estimators=10)
        predictions = self.ml.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        print("✓ Prediction test passed")


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = ModelEvaluator()
        self.y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        self.y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    
    def test_evaluate(self):
        """Test evaluation"""
        results = self.evaluator.evaluate(self.y_true, self.y_pred)
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1_score', results)
        print("✓ Evaluation test passed")
    
    def test_cross_validate(self):
        """Test cross-validation"""
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = load_iris(return_X_y=True)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        cv_results = self.evaluator.cross_validate(model, X[:50], y[:50], cv=3)
        self.assertIsInstance(cv_results, dict)
        print("✓ Cross-validation test passed")


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for PerformanceMetrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        self.y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    
    def test_calculate_accuracy(self):
        """Test accuracy calculation"""
        accuracy = PerformanceMetrics.calculate_accuracy(self.y_true, self.y_pred)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        print("✓ Accuracy calculation test passed")
    
    def test_calculate_balanced_accuracy(self):
        """Test balanced accuracy calculation"""
        bal_acc = PerformanceMetrics.calculate_balanced_accuracy(self.y_true, self.y_pred)
        self.assertGreaterEqual(bal_acc, 0.0)
        self.assertLessEqual(bal_acc, 1.0)
        print("✓ Balanced accuracy calculation test passed")
    
    def test_get_all_metrics(self):
        """Test getting all metrics"""
        metrics = PerformanceMetrics.get_all_metrics(self.y_true, self.y_pred)
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        print("✓ Get all metrics test passed")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("RUNNING ML MODELS TESTS")
    print("="*60 + "\n")
    
    unittest.main(verbosity=2)

