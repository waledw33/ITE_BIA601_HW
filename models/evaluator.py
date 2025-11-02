"""
Model Evaluator Module
Evaluates model performance with comprehensive metrics
Developed by: Student 3
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, cross_val_predict
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    ModelEvaluator class for comprehensive model evaluation
    Calculates various performance metrics
    """
    
    def __init__(self):
        """Initialize ModelEvaluator"""
        self.results = {}
    
    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """
        Comprehensive evaluation of predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            dict: All evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Multi-class handling
        n_classes = len(np.unique(y_true))
        average_type = 'binary' if n_classes == 2 else 'weighted'
        
        precision = precision_score(y_true, y_pred, average=average_type, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average_type, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_type, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'n_samples': len(y_true),
            'n_classes': n_classes
        }
        
        # Add ROC-AUC for binary classification with probabilities
        if n_classes == 2 and y_pred_proba is not None:
            try:
                if len(y_pred_proba.shape) > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                auc = roc_auc_score(y_true, y_pred_proba)
                results['roc_auc'] = auc
            except:
                pass
        
        self.results = results
        return results
    
    def cross_validate(self, model, X, y, cv=5, scoring_metrics=None):
        """
        Perform cross-validation with multiple metrics
        
        Args:
            model: ML model to evaluate
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            scoring_metrics: List of metrics to calculate
            
        Returns:
            dict: Cross-validation results
        """
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=metric,
                n_jobs=-1
            )
            
            cv_results[metric] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return cv_results
    
    def evaluate_with_cross_validation(self, model, X, y, cv=5):
        """
        Train and evaluate using cross-validation
        
        Args:
            model: ML model
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            
        Returns:
            dict: Comprehensive evaluation results
        """
        # Get cross-validated predictions
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        
        # Evaluate predictions
        results = self.evaluate(y_true=y, y_pred=y_pred)
        
        # Add cross-validation scores
        cv_results = self.cross_validate(model, X, y, cv=cv)
        results['cross_validation'] = cv_results
        
        return results
    
    def print_evaluation_report(self, results=None):
        """
        Print formatted evaluation report
        
        Args:
            results: Evaluation results dict (uses self.results if None)
        """
        if results is None:
            results = self.results
        
        if not results:
            print("No evaluation results available")
            return
        
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        print(f"\nSamples: {results['n_samples']}")
        print(f"Classes: {results['n_classes']}")
        
        print("\n" + "-"*60)
        print("PERFORMANCE METRICS")
        print("-"*60)
        
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        
        if 'roc_auc' in results:
            print(f"ROC-AUC:   {results['roc_auc']:.4f}")
        
        print("\n" + "-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
        print(results['confusion_matrix'])
        
        if 'cross_validation' in results:
            print("\n" + "-"*60)
            print("CROSS-VALIDATION RESULTS")
            print("-"*60)
            
            for metric, scores in results['cross_validation'].items():
                print(f"\n{metric}:")
                print(f"  Mean: {scores['mean']:.4f}")
                print(f"  Std:  {scores['std']:.4f}")
                print(f"  Range: [{scores['min']:.4f}, {scores['max']:.4f}]")
        
        print("\n" + "="*60 + "\n")
    
    def compare_feature_subsets(self, model, X, y, feature_subsets, subset_names=None):
        """
        Compare model performance on different feature subsets
        
        Args:
            model: ML model
            X: Full feature matrix
            y: Target vector
            feature_subsets: List of feature index arrays
            subset_names: Names for each subset
            
        Returns:
            dict: Comparison results
        """
        if subset_names is None:
            subset_names = [f"Subset {i+1}" for i in range(len(feature_subsets))]
        
        print("\n" + "="*60)
        print("FEATURE SUBSET COMPARISON")
        print("="*60)
        
        comparison = {}
        
        for name, indices in zip(subset_names, feature_subsets):
            print(f"\nEvaluating: {name} ({len(indices)} features)")
            
            # Extract features
            X_subset = X[:, indices]
            
            # Get cross-validated predictions
            y_pred = cross_val_predict(model, X_subset, y, cv=5, n_jobs=-1)
            
            # Evaluate
            results = self.evaluate(y, y_pred)
            
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  F1-Score: {results['f1_score']:.4f}")
            
            comparison[name] = {
                'n_features': len(indices),
                'feature_indices': indices,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            }
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        # Sort by accuracy
        sorted_comparison = sorted(
            comparison.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        print(f"\n{'Rank':<6}{'Subset':<20}{'Features':<12}{'Accuracy':<12}{'F1-Score':<12}")
        print("-"*60)
        
        for i, (name, res) in enumerate(sorted_comparison):
            print(f"{i+1:<6}{name:<20}{res['n_features']:<12}{res['accuracy']:<12.4f}{res['f1_score']:<12.4f}")
        
        return comparison
    
    def calculate_improvement(self, baseline_accuracy, new_accuracy):
        """
        Calculate improvement percentage
        
        Args:
            baseline_accuracy: Baseline accuracy
            new_accuracy: New accuracy
            
        Returns:
            float: Improvement percentage
        """
        if baseline_accuracy == 0:
            return 0.0
        
        improvement = ((new_accuracy - baseline_accuracy) / baseline_accuracy) * 100
        return improvement
    
    def get_class_wise_metrics(self, y_true, y_pred):
        """
        Get metrics for each class separately
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            dict: Per-class metrics
        """
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        class_metrics = {}
        
        for class_label, metrics in report.items():
            if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                class_metrics[class_label] = metrics
        
        return class_metrics
    
    def evaluate_features(self, X, y, model_type='random_forest', cv=5):
        """
        Evaluate feature subset performance using cross-validation
        
        Args:
            X: Feature matrix (subset)
            y: Target vector
            model_type: Type of model to use
            cv: Number of cross-validation folds
            
        Returns:
            dict: Evaluation metrics
        """
        import time
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        
        start_time = time.time()
        
        # Select model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'svm':
            model = SVC(kernel='rbf', random_state=42)
        elif model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Get cross-validated predictions
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        
        # Calculate metrics
        n_classes = len(np.unique(y))
        average_type = 'binary' if n_classes == 2 else 'weighted'
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average=average_type, zero_division=0)
        recall = recall_score(y, y_pred, average=average_type, zero_division=0)
        f1 = f1_score(y, y_pred, average=average_type, zero_division=0)
        
        elapsed_time = time.time() - start_time
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'time': elapsed_time,
            'n_features': X.shape[1] if len(X.shape) > 1 else 1
        }


# Example usage
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    print("="*60)
    print("MODEL EVALUATOR EXAMPLE")
    print("="*60)
    
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate
    results = evaluator.evaluate(y_test, y_pred)
    
    # Print report
    evaluator.print_evaluation_report()
    
    # Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION")
    print("="*60)
    
    cv_results = evaluator.evaluate_with_cross_validation(model, X, y, cv=5)
    evaluator.print_evaluation_report(cv_results)
    
    # Compare feature subsets
    print("\n" + "="*60)
    print("FEATURE SUBSET COMPARISON")
    print("="*60)
    
    subsets = [
        np.array([0, 1, 2, 3]),  # All features
        np.array([0, 2]),         # Selected features
        np.array([1, 3])          # Different selection
    ]
    
    names = ['All Features', 'Subset A', 'Subset B']
    
    comparison = evaluator.compare_feature_subsets(
        model, X, y, subsets, names
    )

