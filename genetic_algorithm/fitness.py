"""
Fitness Evaluation Module - Performance Assessment
Evaluates chromosome quality based on classification accuracy and feature count
Part of: Advanced Feature Selection Research Project
Research Team: AI & Machine Learning Laboratory
"""

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class FitnessEvaluator:
    """
    Fitness Evaluator for Feature Selection Chromosomes
    
    This class assesses the quality of feature subsets by training
    machine learning models and measuring their performance. Fitness
    is calculated as a weighted combination of:
    - Classification accuracy (model performance)
    - Feature reduction (parsimony principle)
    
    The evaluator supports multiple ML models and offers both fast
    train-test evaluation and more robust cross-validation.
    """
    
    def __init__(self, X, y, model_type='random_forest', alpha=0.9, 
                 cv_folds=3, use_fast_evaluation=True):
        """
        Initialize fitness evaluator
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Complete feature matrix
        y : array-like, shape (n_samples,)
            Target labels
        model_type : str, default='random_forest'
            Classification model to use for evaluation
            Options: 'random_forest', 'svm', 'knn'
        alpha : float, default=0.9
            Weight for accuracy in fitness calculation (0.0-1.0)
            fitness = alpha * accuracy + (1-alpha) * feature_reduction
            Higher values prioritize accuracy over feature reduction
        cv_folds : int, default=3
            Number of cross-validation folds (for slow evaluation)
        use_fast_evaluation : bool, default=True
            If True, uses train-test split (faster)
            If False, uses cross-validation (more robust but slower)
        """
        self.X = X
        self.y = y
        self.model_type = model_type
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.use_fast_evaluation = use_fast_evaluation
        self.n_features = X.shape[1]
        
        # For fast evaluation, create train-test split once
        if use_fast_evaluation:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        
        # Initialize model
        self.model = self._create_model()
        
        # Evaluation counter for statistics
        self.evaluation_count = 0
    
    def _create_model(self):
        """
        Create machine learning model based on specified type
        
        Returns:
        --------
        sklearn classifier
            Configured classification model
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=50, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                random_state=42
            )
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def evaluate_chromosome(self, chromosome):
        """
        Evaluate fitness of a single chromosome
        
        This method:
        1. Extracts selected features from chromosome
        2. Trains ML model on selected features
        3. Measures classification accuracy
        4. Calculates fitness score
        5. Updates chromosome metrics
        
        Parameters:
        -----------
        chromosome : Chromosome
            Chromosome object to evaluate
            
        Returns:
        --------
        float
            Fitness score (0.0-1.0, higher is better)
        """
        # Get selected feature indices
        selected_features = chromosome.get_selected_features()
        n_selected = len(selected_features)
        
        # Handle edge case: no features selected
        if n_selected == 0:
            chromosome.fitness = 0.0
            chromosome.accuracy = 0.0
            return 0.0
        
        # Evaluate model performance with selected features
        try:
            if self.use_fast_evaluation:
                # Fast evaluation: simple train-test split
                X_train_selected = self.X_train[:, selected_features]
                X_test_selected = self.X_test[:, selected_features]
                
                # Create fresh model instance
                model = self._create_model()
                model.fit(X_train_selected, self.y_train)
                accuracy = model.score(X_test_selected, self.y_test)
            else:
                # Robust evaluation: cross-validation
                X_selected = self.X[:, selected_features]
                cv_scores = cross_val_score(
                    self.model, 
                    X_selected, 
                    self.y,
                    cv=self.cv_folds,
                    scoring='accuracy',
                    n_jobs=-1
                )
                accuracy = np.mean(cv_scores)
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            accuracy = 0.0
        
        # Calculate feature reduction ratio
        feature_reduction = 1.0 - (n_selected / self.n_features)
        
        # Calculate weighted fitness score
        fitness = (self.alpha * accuracy) + ((1 - self.alpha) * feature_reduction)
        
        # Update chromosome metrics
        chromosome.fitness = fitness
        chromosome.accuracy = accuracy
        
        # Increment evaluation counter
        self.evaluation_count += 1
        
        return fitness
    
    def evaluate_population(self, population):
        """
        Evaluate fitness for entire population
        
        Parameters:
        -----------
        population : list
            List of Chromosome objects to evaluate
            
        Returns:
        --------
        ndarray
            Array of fitness scores for each chromosome
        """
        fitness_scores = []
        
        for chromosome in population:
            fitness = self.evaluate_chromosome(chromosome)
            fitness_scores.append(fitness)
        
        return np.array(fitness_scores)
    
    def get_best_chromosome(self, population):
        """
        Find chromosome with highest fitness in population
        
        Parameters:
        -----------
        population : list
            List of Chromosome objects
            
        Returns:
        --------
        Chromosome
            Individual with maximum fitness
        """
        return max(population, key=lambda c: c.fitness)
    
    def get_worst_chromosome(self, population):
        """
        Find chromosome with lowest fitness in population
        
        Parameters:
        -----------
        population : list
            List of Chromosome objects
            
        Returns:
        --------
        Chromosome
            Individual with minimum fitness
        """
        return min(population, key=lambda c: c.fitness)
    
    def get_average_fitness(self, population):
        """
        Calculate average fitness of population
        
        Parameters:
        -----------
        population : list
            List of Chromosome objects
            
        Returns:
        --------
        float
            Mean fitness value
        """
        return np.mean([c.fitness for c in population])
    
    def get_population_stats(self, population):
        """
        Compute comprehensive population statistics
        
        Parameters:
        -----------
        population : list
            List of Chromosome objects
            
        Returns:
        --------
        dict
            Dictionary containing:
            - best_fitness: Maximum fitness in population
            - worst_fitness: Minimum fitness in population
            - avg_fitness: Mean fitness
            - std_fitness: Standard deviation of fitness
            - best_accuracy: Maximum accuracy achieved
            - avg_features: Average number of selected features
        """
        fitness_values = [c.fitness for c in population]
        
        return {
            'best_fitness': np.max(fitness_values),
            'worst_fitness': np.min(fitness_values),
            'avg_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'best_accuracy': max([c.accuracy for c in population]),
            'avg_features': np.mean([c.n_selected_features for c in population])
        }
    
    def evaluate_feature_subset(self, feature_indices):
        """
        Directly evaluate a specific feature subset
        
        Helper method for testing specific feature combinations
        without creating a chromosome object.
        
        Parameters:
        -----------
        feature_indices : array-like
            Indices of features to evaluate
            
        Returns:
        --------
        float
            Classification accuracy for feature subset
        """
        if len(feature_indices) == 0:
            return 0.0
        
        X_selected = self.X[:, feature_indices]
        
        try:
            cv_scores = cross_val_score(
                self.model,
                X_selected,
                self.y,
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=-1
            )
            accuracy = np.mean(cv_scores)
        except:
            accuracy = 0.0
        
        return accuracy


# Module Test & Demonstration
if __name__ == '__main__':
    from .chromosome import Chromosome
    from sklearn.datasets import load_iris
    
    print("="*70)
    print("FITNESS EVALUATOR MODULE DEMONSTRATION")
    print("="*70)
    
    # Load sample dataset
    X, y = load_iris(return_X_y=True)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create fitness evaluator
    evaluator = FitnessEvaluator(X, y, model_type='random_forest', alpha=0.9)
    
    # Create and evaluate a chromosome
    chromosome = Chromosome(n_features=X.shape[1])
    print(f"\nChromosome genes: {chromosome.genes}")
    print(f"Selected features: {chromosome.get_selected_features()}")
    
    # Evaluate fitness
    fitness = evaluator.evaluate_chromosome(chromosome)
    print(f"\nEvaluation Results:")
    print(f"  Fitness Score: {fitness:.4f}")
    print(f"  Accuracy: {chromosome.accuracy:.4f}")
    print(f"  Selected Features: {chromosome.n_selected_features}/{X.shape[1]}")
    
    # Create and evaluate population
    print("\n" + "="*70)
    print("POPULATION EVALUATION")
    print("="*70)
    
    population = Chromosome.create_random_population(10, X.shape[1])
    evaluator.evaluate_population(population)
    
    stats = evaluator.get_population_stats(population)
    print(f"\nPopulation Statistics:")
    print(f"  Best Fitness: {stats['best_fitness']:.4f}")
    print(f"  Worst Fitness: {stats['worst_fitness']:.4f}")
    print(f"  Average Fitness: {stats['avg_fitness']:.4f}")
    print(f"  Std Deviation: {stats['std_fitness']:.4f}")
    print(f"  Best Accuracy: {stats['best_accuracy']:.4f}")
    print(f"  Average Features: {stats['avg_features']:.1f}")
    
    # Get best chromosome
    best = evaluator.get_best_chromosome(population)
    print(f"\nBest Chromosome:")
    print(f"  {best}")
    
    print("\n" + "="*70)
