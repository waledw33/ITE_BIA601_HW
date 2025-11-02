"""
Genetic Algorithm Engine - Feature Selection Framework
Main orchestrator for evolutionary optimization of feature subsets
Part of: Advanced Feature Selection Research Project
Research Team: AI & Machine Learning Laboratory
"""

import numpy as np
import time
from .chromosome import Chromosome
from .fitness import FitnessEvaluator
from .operators import GeneticOperators


class GeneticAlgorithm:
    """
    Genetic Algorithm Engine for Feature Selection
    
    This class implements a complete genetic algorithm workflow for
    optimizing feature selection in machine learning pipelines. It uses
    evolutionary principles to search for optimal feature subsets that
    maximize model performance while minimizing feature count.
    
    The algorithm follows standard GA phases:
    1. Initialization - Create random population
    2. Evaluation - Assess fitness of each individual
    3. Selection - Choose parents for reproduction
    4. Crossover - Combine parent genes
    5. Mutation - Introduce random variations
    6. Replacement - Form new generation
    """
    
    def __init__(self, 
                 population_size=50,
                 n_generations=100,
                 crossover_rate=0.8,
                 mutation_rate=0.1,
                 elitism_rate=0.1,
                 tournament_size=3,
                 alpha=0.9,
                 model_type='random_forest',
                 selection_method='tournament',
                 crossover_method='single_point',
                 verbose=True):
        """
        Initialize Genetic Algorithm engine
        
        Parameters:
        -----------
        population_size : int, default=50
            Number of individuals in each generation
        n_generations : int, default=100
            Number of evolutionary cycles to execute
        crossover_rate : float, default=0.8
            Probability of crossover between parent pairs (0.0-1.0)
        mutation_rate : float, default=0.1
            Probability of gene mutation (0.0-1.0)
        elitism_rate : float, default=0.1
            Proportion of best individuals preserved each generation
        tournament_size : int, default=3
            Number of individuals in tournament selection
        alpha : float, default=0.9
            Weight for accuracy vs. feature reduction tradeoff
            (higher values prioritize accuracy)
        model_type : str, default='random_forest'
            Machine learning model for fitness evaluation
            Options: 'random_forest', 'svm', 'knn'
        selection_method : str, default='tournament'
            Parent selection strategy
            Options: 'tournament', 'roulette', 'rank'
        crossover_method : str, default='single_point'
            Crossover operator type
            Options: 'single_point', 'two_point', 'uniform'
        verbose : bool, default=True
            Enable detailed progress output
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.model_type = model_type
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.verbose = verbose
        
        # Calculate number of elite individuals
        self.n_elite = max(1, int(population_size * elitism_rate))
        
        # Initialize components
        self.operators = GeneticOperators(crossover_rate, mutation_rate)
        self.evaluator = None
        
        # Track best solution and evolution history
        self.best_chromosome = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'best_accuracy': [],
            'avg_features': []
        }
    
    def initialize_population(self, n_features):
        """
        Create initial random population
        
        Parameters:
        -----------
        n_features : int
            Number of features in the dataset
            
        Returns:
        --------
        list
            List of randomly initialized chromosomes
        """
        if self.verbose:
            print(f"Initializing population of {self.population_size} individuals...")
        
        population = Chromosome.create_random_population(
            self.population_size,
            n_features
        )
        
        return population
    
    def select_parent(self, population):
        """
        Select a parent for reproduction
        
        Parameters:
        -----------
        population : list
            Current generation of chromosomes
            
        Returns:
        --------
        Chromosome
            Selected parent chromosome
        """
        if self.selection_method == 'tournament':
            return self.operators.tournament_selection(population, self.tournament_size)
        elif self.selection_method == 'roulette':
            return self.operators.roulette_wheel_selection(population)
        elif self.selection_method == 'rank':
            return self.operators.rank_selection(population)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def crossover_parents(self, parent1, parent2):
        """
        Perform crossover between two parents
        
        Parameters:
        -----------
        parent1, parent2 : Chromosome
            Parent chromosomes for breeding
            
        Returns:
        --------
        tuple
            Two offspring chromosomes (child1, child2)
        """
        if self.crossover_method == 'single_point':
            return self.operators.single_point_crossover(parent1, parent2)
        elif self.crossover_method == 'two_point':
            return self.operators.two_point_crossover(parent1, parent2)
        elif self.crossover_method == 'uniform':
            return self.operators.uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")
    
    def mutate_individual(self, chromosome):
        """
        Apply mutation operator to chromosome
        
        Parameters:
        -----------
        chromosome : Chromosome
            Individual to mutate
            
        Returns:
        --------
        Chromosome
            Mutated chromosome
        """
        return self.operators.bit_flip_mutation(chromosome)
    
    def evolve_generation(self, population):
        """
        Evolve one complete generation
        
        Parameters:
        -----------
        population : list
            Current population
            
        Returns:
        --------
        list
            New population after evolution
        """
        # Elitism: preserve best individuals
        elite = self.operators.elitism_selection(population, self.n_elite)
        
        # Start new generation with elite
        new_population = elite.copy()
        
        # Fill rest of population with offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.select_parent(population)
            parent2 = self.select_parent(population)
            
            # Crossover
            offspring1, offspring2 = self.crossover_parents(parent1, parent2)
            
            # Mutation
            offspring1 = self.mutate_individual(offspring1)
            offspring2 = self.mutate_individual(offspring2)
            
            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        # Ensure exact population size
        new_population = new_population[:self.population_size]
        
        return new_population
    
    def update_history(self, population, generation):
        """
        Record generation statistics
        
        Parameters:
        -----------
        population : list
            Current population
        generation : int
            Generation number
        """
        stats = self.evaluator.get_population_stats(population)
        
        self.history['best_fitness'].append(stats['best_fitness'])
        self.history['avg_fitness'].append(stats['avg_fitness'])
        self.history['worst_fitness'].append(stats['worst_fitness'])
        self.history['best_accuracy'].append(stats['best_accuracy'])
        self.history['avg_features'].append(stats['avg_features'])
        
        # Update best solution
        best = self.evaluator.get_best_chromosome(population)
        if self.best_chromosome is None or best.fitness > self.best_chromosome.fitness:
            self.best_chromosome = best.copy()
        
        # Print progress
        if self.verbose:
            print(f"Generation {generation:3d} | "
                  f"Best: {stats['best_fitness']:.4f} | "
                  f"Avg: {stats['avg_fitness']:.4f} | "
                  f"Accuracy: {stats['best_accuracy']:.4f} | "
                  f"Features: {stats['avg_features']:.1f}")
    
    def fit(self, X, y):
        """
        Execute genetic algorithm on dataset
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target vector
            
        Returns:
        --------
        Chromosome
            Best solution found
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*70)
            print("GENETIC ALGORITHM FOR FEATURE SELECTION")
            print("="*70)
            print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Population Size: {self.population_size}")
            print(f"Generations: {self.n_generations}")
            print(f"Crossover Rate: {self.crossover_rate}")
            print(f"Mutation Rate: {self.mutation_rate}")
            print(f"Elite Count: {self.n_elite}")
            print(f"ML Model: {self.model_type}")
            print("="*70)
        
        # Initialize fitness evaluator
        self.evaluator = FitnessEvaluator(
            X, y, 
            model_type=self.model_type,
            alpha=self.alpha
        )
        
        # Initialize population
        population = self.initialize_population(X.shape[1])
        
        # Evaluate initial population
        if self.verbose:
            print("\nEvaluating initial population...")
        self.evaluator.evaluate_population(population)
        self.update_history(population, 0)
        
        # Evolution loop
        if self.verbose:
            print(f"\n{'='*70}")
            print("EVOLUTION PROGRESS")
            print(f"{'='*70}\n")
        
        for generation in range(1, self.n_generations + 1):
            # Evolve new generation
            population = self.evolve_generation(population)
            
            # Evaluate new population
            self.evaluator.evaluate_population(population)
            
            # Update history
            self.update_history(population, generation)
        
        # Final results
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("OPTIMIZATION COMPLETE")
            print(f"{'='*70}")
            print(f"Execution Time: {elapsed_time:.2f} seconds")
            print(f"\nBest Solution Found:")
            print(f"  Fitness Score: {self.best_chromosome.fitness:.4f}")
            print(f"  Accuracy: {self.best_chromosome.accuracy:.4f}")
            print(f"  Features Selected: {self.best_chromosome.n_selected_features}/{X.shape[1]}")
            print(f"  Feature Reduction: {(1 - self.best_chromosome.n_selected_features/X.shape[1])*100:.1f}%")
            print(f"  Selected Indices: {self.best_chromosome.get_selected_features().tolist()}")
            print(f"{'='*70}\n")
        
        return self.best_chromosome
    
    def get_selected_features(self):
        """
        Get indices of features selected by best chromosome
        
        Returns:
        --------
        ndarray
            Array of selected feature indices
        """
        if self.best_chromosome is None:
            raise ValueError("GA has not been run yet. Call fit() first.")
        return self.best_chromosome.get_selected_features()
    
    def get_history(self):
        """
        Get evolution history
        
        Returns:
        --------
        dict
            History of fitness values across generations
        """
        return self.history
    
    def plot_history(self):
        """
        Plot evolution history (requires matplotlib)
        """
        try:
            import matplotlib.pyplot as plt
            
            generations = range(len(self.history['best_fitness']))
            
            plt.figure(figsize=(14, 5))
            
            # Plot fitness evolution
            plt.subplot(1, 3, 1)
            plt.plot(generations, self.history['best_fitness'], 
                    label='Best', linewidth=2, color='#2563eb')
            plt.plot(generations, self.history['avg_fitness'], 
                    label='Average', linewidth=2, color='#10b981')
            plt.xlabel('Generation', fontweight='bold')
            plt.ylabel('Fitness', fontweight='bold')
            plt.title('Fitness Evolution', fontweight='bold', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot accuracy evolution
            plt.subplot(1, 3, 2)
            plt.plot(generations, self.history['best_accuracy'], 
                    linewidth=2, color='#8b5cf6')
            plt.xlabel('Generation', fontweight='bold')
            plt.ylabel('Accuracy', fontweight='bold')
            plt.title('Best Accuracy Evolution', fontweight='bold', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Plot feature count
            plt.subplot(1, 3, 3)
            plt.plot(generations, self.history['avg_features'], 
                    linewidth=2, color='#f59e0b')
            plt.xlabel('Generation', fontweight='bold')
            plt.ylabel('Average Features Selected', fontweight='bold')
            plt.title('Feature Selection Evolution', fontweight='bold', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Cannot plot history.")


# Module Test & Demonstration
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    
    print("="*70)
    print("GENETIC ALGORITHM MODULE DEMONSTRATION")
    print("="*70)
    
    # Load sample dataset
    X, y = load_iris(return_X_y=True)
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features\n")
    
    # Create and configure GA
    ga = GeneticAlgorithm(
        population_size=20,
        n_generations=30,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_rate=0.1,
        alpha=0.9,
        model_type='random_forest',
        verbose=True
    )
    
    # Execute genetic algorithm
    best_solution = ga.fit(X, y)
    
    # Get selected features
    selected_features = ga.get_selected_features()
    print(f"Final selected feature indices: {selected_features}")
    
    # Plot evolution history (if matplotlib available)
    # ga.plot_history()
