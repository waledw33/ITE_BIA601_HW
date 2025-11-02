"""
Machine Learning Models Module
Implements various ML classifiers for feature evaluation
Developed by: Student 3
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class MLModels:
    """
    MLModels class for training and using various ML classifiers
    """
    
    def __init__(self, random_state=42):
        """
        Initialize ML Models
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_model = None
        self.model_name = None
    
    def get_random_forest(self, n_estimators=100, max_depth=None):
        """
        Get Random Forest classifier
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            
        Returns:
            RandomForestClassifier instance
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def get_svm(self, kernel='rbf', C=1.0):
        """
        Get Support Vector Machine classifier
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            
        Returns:
            SVC instance
        """
        return SVC(
            kernel=kernel,
            C=C,
            random_state=self.random_state
        )
    
    def get_knn(self, n_neighbors=5):
        """
        Get K-Nearest Neighbors classifier
        
        Args:
            n_neighbors: Number of neighbors
            
        Returns:
            KNeighborsClassifier instance
        """
        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            n_jobs=-1
        )
    
    def get_decision_tree(self, max_depth=None):
        """
        Get Decision Tree classifier
        
        Args:
            max_depth: Maximum depth of tree
            
        Returns:
            DecisionTreeClassifier instance
        """
        return DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=self.random_state
        )
    
    def get_naive_bayes(self):
        """
        Get Gaussian Naive Bayes classifier
        
        Returns:
            GaussianNB instance
        """
        return GaussianNB()
    
    def get_logistic_regression(self, max_iter=1000):
        """
        Get Logistic Regression classifier
        
        Args:
            max_iter: Maximum number of iterations
            
        Returns:
            LogisticRegression instance
        """
        return LogisticRegression(
            max_iter=max_iter,
            random_state=self.random_state
        )
    
    def get_gradient_boosting(self, n_estimators=100):
        """
        Get Gradient Boosting classifier
        
        Args:
            n_estimators: Number of boosting stages
            
        Returns:
            GradientBoostingClassifier instance
        """
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state
        )
    
    def get_model(self, model_name='random_forest', **kwargs):
        """
        Get a model by name with custom parameters
        
        Args:
            model_name: Name of the model
            **kwargs: Model-specific parameters
            
        Returns:
            Classifier instance
        """
        if model_name == 'random_forest':
            return self.get_random_forest(**kwargs)
        elif model_name == 'svm':
            return self.get_svm(**kwargs)
        elif model_name == 'knn':
            return self.get_knn(**kwargs)
        elif model_name == 'decision_tree':
            return self.get_decision_tree(**kwargs)
        elif model_name == 'naive_bayes':
            return self.get_naive_bayes()
        elif model_name == 'logistic_regression':
            return self.get_logistic_regression(**kwargs)
        elif model_name == 'gradient_boosting':
            return self.get_gradient_boosting(**kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def train(self, X_train, y_train, model_name='random_forest', **kwargs):
        """
        Train a model on training data
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of model to train
            **kwargs: Model-specific parameters
            
        Returns:
            Trained model
        """
        # Get model
        model = self.get_model(model_name, **kwargs)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Store trained model
        self.trained_model = model
        self.model_name = model_name
        
        # Calculate training accuracy
        train_accuracy = model.score(X_train, y_train)
        print(f"{model_name} trained successfully")
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        return model
    
    def predict(self, X):
        """
        Make predictions using trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if self.trained_model is None:
            raise ValueError("No model has been trained. Call train() first.")
        
        return self.trained_model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities (if supported by model)
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if self.trained_model is None:
            raise ValueError("No model has been trained. Call train() first.")
        
        if hasattr(self.trained_model, 'predict_proba'):
            return self.trained_model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_name} does not support predict_proba")
    
    def evaluate_model(self, X, y, model_name='random_forest', cv=5, **kwargs):
        """
        Evaluate model using cross-validation
        
        Args:
            X: Feature matrix
            y: Target vector
            model_name: Name of model to evaluate
            cv: Number of cross-validation folds
            **kwargs: Model-specific parameters
            
        Returns:
            dict: Evaluation results
        """
        # Get model
        model = self.get_model(model_name, **kwargs)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        results = {
            'model': model_name,
            'cv_scores': cv_scores,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'min_accuracy': np.min(cv_scores),
            'max_accuracy': np.max(cv_scores)
        }
        
        return results
    
    def compare_models(self, X, y, model_list=None, cv=5):
        """
        Compare multiple models using cross-validation
        
        Args:
            X: Feature matrix
            y: Target vector
            model_list: List of model names to compare
            cv: Number of cross-validation folds
            
        Returns:
            dict: Comparison results for all models
        """
        if model_list is None:
            model_list = ['random_forest', 'svm', 'knn', 'decision_tree', 'naive_bayes']
        
        print(f"\nComparing {len(model_list)} models with {cv}-fold cross-validation...")
        print("="*60)
        
        results = {}
        
        for model_name in model_list:
            print(f"\nEvaluating {model_name}...")
            try:
                model_results = self.evaluate_model(X, y, model_name, cv=cv)
                results[model_name] = model_results
                
                print(f"  Mean Accuracy: {model_results['mean_accuracy']:.4f} "
                      f"(± {model_results['std_accuracy']:.4f})")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        # Sort by mean accuracy
        sorted_results = sorted(
            [(name, res) for name, res in results.items() if 'mean_accuracy' in res],
            key=lambda x: x[1]['mean_accuracy'],
            reverse=True
        )
        
        for i, (name, res) in enumerate(sorted_results):
            print(f"{i+1}. {name:20s} : {res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}")
        
        return results
    
    def get_feature_importance(self, X, y, feature_names=None):
        """
        Get feature importance (for tree-based models)
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            dict: Feature importances
        """
        # Train Random Forest for feature importance
        model = self.get_random_forest()
        model.fit(X, y)
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Create dictionary of feature importances
        feature_importance = {
            name: importance 
            for name, importance in zip(feature_names, importances)
        }
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features)


# Example usage
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    print("="*60)
    print("ML MODELS EXAMPLE")
    print("="*60)
    
    # Load dataset
    X, y = load_iris(return_X_y=True)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize ML Models
    ml = MLModels()
    
    # Train a single model
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    model = ml.train(X_train, y_train, model_name='random_forest')
    
    # Make predictions
    y_pred = ml.predict(X_test)
    test_accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Compare models
    print("\n" + "="*60)
    print("COMPARING MULTIPLE MODELS")
    print("="*60)
    
    results = ml.compare_models(X, y, cv=5)
    
    # Get feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    importance = ml.get_feature_importance(X, y)
    for feature, imp in importance.items():
        print(f"{feature}: {imp:.4f}")

