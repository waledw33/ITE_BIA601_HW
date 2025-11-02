"""
Traditional Feature Selection Methods
Implements statistical and filter-based feature selection methods
Developed by: Student 6
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso


class TraditionalFeatureSelection:
    """
    TraditionalFeatureSelection class implementing various traditional methods
    Chi-Square, ANOVA F-test, Mutual Information, RFE, etc.
    """
    
    def __init__(self):
        """Initialize TraditionalFeatureSelection"""
        self.selected_features = {}
        self.feature_scores = {}
    
    def chi_square_selection(self, X, y, k='all'):
        """
        Chi-Square test for feature selection
        Works with non-negative features
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select ('all' or int)
            
        Returns:
            Selected feature indices and scores
        """
        # Ensure non-negative values
        X_pos = X - X.min() if X.min() < 0 else X
        
        # Perform Chi-Square test
        chi_scores, p_values = chi2(X_pos, y)
        
        # Select features
        if k == 'all':
            k = X.shape[1]
        
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_pos, y)
        
        selected_indices = selector.get_support(indices=True)
        
        self.feature_scores['chi_square'] = chi_scores
        self.selected_features['chi_square'] = selected_indices
        
        print(f"Chi-Square Selection: {len(selected_indices)} features selected")
        return selected_indices, chi_scores
    
    def anova_f_test(self, X, y, k='all'):
        """
        ANOVA F-test for feature selection
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            
        Returns:
            Selected feature indices and F-scores
        """
        # Perform ANOVA F-test
        f_scores, p_values = f_classif(X, y)
        
        # Select features
        if k == 'all':
            k = X.shape[1]
        
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        
        self.feature_scores['anova_f'] = f_scores
        self.selected_features['anova_f'] = selected_indices
        
        print(f"ANOVA F-test Selection: {len(selected_indices)} features selected")
        return selected_indices, f_scores
    
    def mutual_information(self, X, y, k='all'):
        """
        Mutual Information for feature selection
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            
        Returns:
            Selected feature indices and MI scores
        """
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Select features
        if k == 'all':
            k = X.shape[1]
        
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        
        self.feature_scores['mutual_info'] = mi_scores
        self.selected_features['mutual_info'] = selected_indices
        
        print(f"Mutual Information Selection: {len(selected_indices)} features selected")
        return selected_indices, mi_scores
    
    def random_forest_importance(self, X, y, threshold='mean'):
        """
        Random Forest feature importance
        
        Args:
            X: Feature matrix
            y: Target vector
            threshold: Importance threshold ('mean', 'median', or float)
            
        Returns:
            Selected feature indices and importance scores
        """
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select features based on threshold
        if threshold == 'mean':
            threshold_value = np.mean(importances)
        elif threshold == 'median':
            threshold_value = np.median(importances)
        else:
            threshold_value = float(threshold)
        
        selected_indices = np.where(importances >= threshold_value)[0]
        
        self.feature_scores['rf_importance'] = importances
        self.selected_features['rf_importance'] = selected_indices
        
        print(f"Random Forest Importance: {len(selected_indices)} features selected")
        print(f"  Threshold: {threshold_value:.4f}")
        return selected_indices, importances
    
    def recursive_feature_elimination(self, X, y, n_features_to_select=None):
        """
        Recursive Feature Elimination (RFE)
        
        Args:
            X: Feature matrix
            y: Target vector
            n_features_to_select: Number of features to select (None = half)
            
        Returns:
            Selected feature indices and rankings
        """
        if n_features_to_select is None:
            n_features_to_select = X.shape[1] // 2
        
        # Create estimator
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Perform RFE
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        rankings = selector.ranking_
        
        self.feature_scores['rfe_ranking'] = rankings
        self.selected_features['rfe'] = selected_indices
        
        print(f"RFE Selection: {len(selected_indices)} features selected")
        return selected_indices, rankings
    
    def lasso_selection(self, X, y, alpha=0.01):
        """
        LASSO (L1 regularization) for feature selection
        
        Args:
            X: Feature matrix
            y: Target vector
            alpha: Regularization strength
            
        Returns:
            Selected feature indices and coefficients
        """
        # Train LASSO
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X, y)
        
        # Get coefficients
        coefficients = np.abs(lasso.coef_)
        
        # Select features with non-zero coefficients
        selected_indices = np.where(coefficients > 0)[0]
        
        self.feature_scores['lasso_coef'] = coefficients
        self.selected_features['lasso'] = selected_indices
        
        print(f"LASSO Selection: {len(selected_indices)} features selected")
        return selected_indices, coefficients
    
    def correlation_based_selection(self, X, y, threshold=0.5):
        """
        Correlation-based feature selection
        Select features highly correlated with target
        
        Args:
            X: Feature matrix
            y: Target vector
            threshold: Correlation threshold
            
        Returns:
            Selected feature indices and correlations
        """
        # Convert to DataFrame for correlation calculation
        df = pd.DataFrame(X)
        df['target'] = y
        
        # Calculate correlations with target
        correlations = df.corr()['target'].abs().values[:-1]  # Exclude target itself
        
        # Select features above threshold
        selected_indices = np.where(correlations >= threshold)[0]
        
        self.feature_scores['correlation'] = correlations
        self.selected_features['correlation'] = selected_indices
        
        print(f"Correlation Selection: {len(selected_indices)} features selected")
        print(f"  Threshold: {threshold}")
        return selected_indices, correlations
    
    def variance_threshold_selection(self, X, threshold=0.01):
        """
        Variance threshold feature selection
        Remove features with low variance
        
        Args:
            X: Feature matrix
            threshold: Variance threshold
            
        Returns:
            Selected feature indices and variances
        """
        # Calculate variances
        variances = np.var(X, axis=0)
        
        # Select features above threshold
        selected_indices = np.where(variances >= threshold)[0]
        
        self.feature_scores['variance'] = variances
        self.selected_features['variance'] = selected_indices
        
        print(f"Variance Threshold Selection: {len(selected_indices)} features selected")
        return selected_indices, variances
    
    def compare_all_methods(self, X, y, k=10):
        """
        Compare all traditional methods
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select for comparison
            
        Returns:
            dict: Comparison results
        """
        print("\n" + "="*60)
        print("COMPARING TRADITIONAL FEATURE SELECTION METHODS")
        print("="*60)
        
        results = {}
        
        # Chi-Square
        try:
            indices, scores = self.chi_square_selection(X, y, k=k)
            results['chi_square'] = {
                'indices': indices,
                'scores': scores,
                'n_features': len(indices)
            }
        except Exception as e:
            print(f"Chi-Square Error: {str(e)}")
        
        # ANOVA F-test
        try:
            indices, scores = self.anova_f_test(X, y, k=k)
            results['anova_f'] = {
                'indices': indices,
                'scores': scores,
                'n_features': len(indices)
            }
        except Exception as e:
            print(f"ANOVA F-test Error: {str(e)}")
        
        # Mutual Information
        try:
            indices, scores = self.mutual_information(X, y, k=k)
            results['mutual_info'] = {
                'indices': indices,
                'scores': scores,
                'n_features': len(indices)
            }
        except Exception as e:
            print(f"Mutual Information Error: {str(e)}")
        
        # Random Forest Importance
        try:
            indices, scores = self.random_forest_importance(X, y)
            # Select top k
            top_k_indices = indices[np.argsort(scores[indices])[-k:]]
            results['rf_importance'] = {
                'indices': top_k_indices,
                'scores': scores,
                'n_features': len(top_k_indices)
            }
        except Exception as e:
            print(f"Random Forest Error: {str(e)}")
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETE")
        print("="*60)
        
        return results
    
    def get_feature_rankings(self, method='all'):
        """
        Get feature rankings for specified method
        
        Args:
            method: Method name or 'all'
            
        Returns:
            dict: Feature rankings
        """
        if method == 'all':
            return self.feature_scores
        elif method in self.feature_scores:
            return self.feature_scores[method]
        else:
            raise ValueError(f"Method '{method}' not found")


# Example usage
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    
    print("="*60)
    print("TRADITIONAL FEATURE SELECTION EXAMPLE")
    print("="*60)
    
    # Load dataset
    X, y = load_iris(return_X_y=True)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize selector
    selector = TraditionalFeatureSelection()
    
    # Compare all methods
    results = selector.compare_all_methods(X, y, k=2)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  Selected features: {result['indices']}")
        print(f"  Number of features: {result['n_features']}")

