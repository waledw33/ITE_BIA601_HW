"""
Performance Metrics Module
Custom metrics and metric calculations
Developed by: Student 3
"""

import numpy as np
from sklearn.metrics import make_scorer


class PerformanceMetrics:
    """
    PerformanceMetrics class for calculating custom performance metrics
    """
    
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """
        Calculate accuracy
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            float: Accuracy score
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def calculate_error_rate(y_true, y_pred):
        """
        Calculate error rate
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            float: Error rate
        """
        return 1.0 - np.mean(y_true == y_pred)
    
    @staticmethod
    def calculate_sensitivity(y_true, y_pred, positive_class=1):
        """
        Calculate sensitivity (True Positive Rate / Recall)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            positive_class: Label of positive class
            
        Returns:
            float: Sensitivity
        """
        tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
        fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
        
        if (tp + fn) == 0:
            return 0.0
        
        return tp / (tp + fn)
    
    @staticmethod
    def calculate_specificity(y_true, y_pred, positive_class=1):
        """
        Calculate specificity (True Negative Rate)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            positive_class: Label of positive class
            
        Returns:
            float: Specificity
        """
        tn = np.sum((y_true != positive_class) & (y_pred != positive_class))
        fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
        
        if (tn + fp) == 0:
            return 0.0
        
        return tn / (tn + fp)
    
    @staticmethod
    def calculate_balanced_accuracy(y_true, y_pred):
        """
        Calculate balanced accuracy (average of sensitivity and specificity)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            float: Balanced accuracy
        """
        classes = np.unique(y_true)
        
        if len(classes) == 2:
            sensitivity = PerformanceMetrics.calculate_sensitivity(y_true, y_pred, classes[1])
            specificity = PerformanceMetrics.calculate_specificity(y_true, y_pred, classes[1])
            return (sensitivity + specificity) / 2.0
        else:
            # For multi-class, calculate per-class accuracy and average
            accuracies = []
            for cls in classes:
                mask = y_true == cls
                if np.sum(mask) > 0:
                    class_acc = np.mean(y_pred[mask] == cls)
                    accuracies.append(class_acc)
            return np.mean(accuracies)
    
    @staticmethod
    def calculate_mcc(y_true, y_pred):
        """
        Calculate Matthews Correlation Coefficient
        For binary classification
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            float: MCC score
        """
        classes = np.unique(y_true)
        if len(classes) != 2:
            return None
        
        pos_class = classes[1]
        neg_class = classes[0]
        
        tp = np.sum((y_true == pos_class) & (y_pred == pos_class))
        tn = np.sum((y_true == neg_class) & (y_pred == neg_class))
        fp = np.sum((y_true == neg_class) & (y_pred == pos_class))
        fn = np.sum((y_true == pos_class) & (y_pred == neg_class))
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def calculate_feature_selection_score(accuracy, n_selected, n_total, alpha=0.9):
        """
        Calculate combined score for feature selection
        Balances accuracy and feature reduction
        
        Args:
            accuracy: Model accuracy
            n_selected: Number of selected features
            n_total: Total number of features
            alpha: Weight for accuracy (0 to 1)
            
        Returns:
            float: Combined score
        """
        feature_reduction = 1.0 - (n_selected / n_total)
        score = (alpha * accuracy) + ((1 - alpha) * feature_reduction)
        return score
    
    @staticmethod
    def calculate_fitness_score(accuracy, n_features_selected, n_features_total, 
                               weight_accuracy=0.9, weight_simplicity=0.1):
        """
        Calculate fitness score for genetic algorithm
        
        Args:
            accuracy: Classification accuracy
            n_features_selected: Number of selected features
            n_features_total: Total number of features available
            weight_accuracy: Weight for accuracy component
            weight_simplicity: Weight for simplicity component
            
        Returns:
            float: Fitness score
        """
        # Accuracy component
        accuracy_component = weight_accuracy * accuracy
        
        # Simplicity component (fewer features is better)
        feature_ratio = n_features_selected / n_features_total
        simplicity_component = weight_simplicity * (1.0 - feature_ratio)
        
        # Combined fitness
        fitness = accuracy_component + simplicity_component
        
        return fitness
    
    @staticmethod
    def get_all_metrics(y_true, y_pred):
        """
        Calculate all available metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            dict: All calculated metrics
        """
        metrics = {
            'accuracy': PerformanceMetrics.calculate_accuracy(y_true, y_pred),
            'error_rate': PerformanceMetrics.calculate_error_rate(y_true, y_pred),
            'balanced_accuracy': PerformanceMetrics.calculate_balanced_accuracy(y_true, y_pred)
        }
        
        # Add binary-specific metrics if applicable
        if len(np.unique(y_true)) == 2:
            metrics['sensitivity'] = PerformanceMetrics.calculate_sensitivity(y_true, y_pred)
            metrics['specificity'] = PerformanceMetrics.calculate_specificity(y_true, y_pred)
            mcc = PerformanceMetrics.calculate_mcc(y_true, y_pred)
            if mcc is not None:
                metrics['mcc'] = mcc
        
        return metrics
    
    @staticmethod
    def create_feature_selection_scorer(alpha=0.9):
        """
        Create a custom scorer for feature selection
        
        Args:
            alpha: Weight for accuracy
            
        Returns:
            sklearn scorer object
        """
        def score_function(estimator, X, y):
            y_pred = estimator.predict(X)
            accuracy = np.mean(y == y_pred)
            n_selected = X.shape[1]
            # Assume this is called during evaluation, so n_total is unknown
            # We'll use a simple weighted accuracy for now
            return accuracy
        
        return make_scorer(score_function)
    
    @staticmethod
    def print_metrics_report(metrics_dict):
        """
        Print formatted metrics report
        
        Args:
            metrics_dict: Dictionary of metric name -> value
        """
        print("\n" + "="*60)
        print("PERFORMANCE METRICS REPORT")
        print("="*60)
        
        for metric_name, value in metrics_dict.items():
            metric_display = metric_name.replace('_', ' ').title()
            print(f"{metric_display:.<40} {value:.4f}")
        
        print("="*60 + "\n")
    
    @staticmethod
    def compare_metrics(metrics1, metrics2, names=None):
        """
        Compare two sets of metrics
        
        Args:
            metrics1: First metrics dictionary
            metrics2: Second metrics dictionary
            names: Names for the two sets (optional)
            
        Returns:
            dict: Comparison results
        """
        if names is None:
            names = ['Set 1', 'Set 2']
        
        print("\n" + "="*70)
        print("METRICS COMPARISON")
        print("="*70)
        
        print(f"\n{'Metric':<25} {names[0]:<20} {names[1]:<20} {'Difference':<15}")
        print("-"*70)
        
        comparison = {}
        
        for metric in metrics1.keys():
            if metric in metrics2:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                diff = val2 - val1
                
                print(f"{metric:<25} {val1:<20.4f} {val2:<20.4f} {diff:+.4f}")
                
                comparison[metric] = {
                    names[0]: val1,
                    names[1]: val2,
                    'difference': diff,
                    'improvement_pct': (diff / val1 * 100) if val1 != 0 else 0
                }
        
        print("="*70 + "\n")
        
        return comparison


# Example usage
if __name__ == '__main__':
    print("="*60)
    print("PERFORMANCE METRICS EXAMPLE")
    print("="*60)
    
    # Sample predictions (binary classification)
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    
    print(f"\nTrue labels:      {y_true}")
    print(f"Predicted labels: {y_pred}")
    
    # Calculate all metrics
    metrics = PerformanceMetrics.get_all_metrics(y_true, y_pred)
    
    # Print metrics
    PerformanceMetrics.print_metrics_report(metrics)
    
    # Calculate feature selection score
    print("="*60)
    print("FEATURE SELECTION SCORING")
    print("="*60)
    
    accuracy = 0.85
    n_selected = 5
    n_total = 20
    
    score = PerformanceMetrics.calculate_feature_selection_score(
        accuracy, n_selected, n_total, alpha=0.9
    )
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Features selected: {n_selected}/{n_total}")
    print(f"Feature selection score: {score:.4f}")
    
    # Compare two scenarios
    print("\n" + "="*60)
    print("COMPARING TWO SCENARIOS")
    print("="*60)
    
    metrics1 = PerformanceMetrics.get_all_metrics(y_true, y_pred)
    
    y_pred2 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    metrics2 = PerformanceMetrics.get_all_metrics(y_true, y_pred2)
    
    comparison = PerformanceMetrics.compare_metrics(
        metrics1, metrics2,
        names=['Scenario A', 'Scenario B']
    )

