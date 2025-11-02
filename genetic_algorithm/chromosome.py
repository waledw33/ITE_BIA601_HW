"""
Chromosome Module - Genetic Feature Selection
Represents a single solution (feature subset) in the genetic algorithm
Part of: Advanced Feature Selection Framework
Research Team: AI & Machine Learning Laboratory
"""

import numpy as np
import random


class Chromosome:
    """
    Chromosome Class - Binary Feature Representation
    
    Each gene (bit) represents a feature in the dataset:
    - 1 = Feature is selected for model training
    - 0 = Feature is excluded from model training
    
    This binary encoding allows efficient genetic operations and
    straightforward interpretation of feature selection patterns.
    """
    
    def __init__(self, n_features, genes=None):
        """
        Initialize a new chromosome instance
        
        Parameters:
        -----------
        n_features : int
            Total number of features in the dataset
        genes : array-like, optional
            Predefined gene sequence. If None, generates random genes
            ensuring at least one feature is selected
        """
        self.n_features = n_features
        
        if genes is not None:
            self.genes = np.array(genes, dtype=int)
        else:
            # Generate random gene sequence
            # Ensure at least one feature is selected to avoid empty solutions
            self.genes = np.random.randint(0, 2, size=n_features)
            while self.genes.sum() == 0:
                self.genes = np.random.randint(0, 2, size=n_features)
        
        # Performance metrics
        self.fitness = 0.0
        self.accuracy = 0.0
        self.n_selected_features = int(self.genes.sum())
    
    def get_selected_features(self):
        """
        Retrieve indices of selected features
        
        Returns:
        --------
        ndarray
            Array of indices where genes equal 1 (selected features)
        """
        return np.where(self.genes == 1)[0]
    
    def get_feature_mask(self):
        """
        Generate boolean mask for feature selection
        
        Returns:
        --------
        ndarray
            Boolean array indicating selected features
        """
        return self.genes.astype(bool)
    
    def count_selected(self):
        """
        Count number of selected features
        
        Returns:
        --------
        int
            Number of features with gene value 1
        """
        return int(self.genes.sum())
    
    def flip_gene(self, position):
        """
        Flip a single gene (mutation operation)
        
        Parameters:
        -----------
        position : int
            Index of gene to flip (0 becomes 1, 1 becomes 0)
        """
        if 0 <= position < self.n_features:
            self.genes[position] = 1 - self.genes[position]
            self.n_selected_features = self.count_selected()
    
    def set_gene(self, position, value):
        """
        Set specific gene value
        
        Parameters:
        -----------
        position : int
            Gene index to modify
        value : int
            New value (must be 0 or 1)
        """
        if 0 <= position < self.n_features and value in [0, 1]:
            self.genes[position] = value
            self.n_selected_features = self.count_selected()
    
    def copy(self):
        """
        Create deep copy of chromosome
        
        Returns:
        --------
        Chromosome
            New chromosome instance with identical genes and metrics
        """
        clone = Chromosome(self.n_features, genes=self.genes.copy())
        clone.fitness = self.fitness
        clone.accuracy = self.accuracy
        return clone
    
    def __str__(self):
        """
        String representation for display
        
        Returns:
        --------
        str
            Human-readable chromosome description
        """
        genes_str = ''.join(map(str, self.genes))
        return f"Chromosome(genes={genes_str}, fitness={self.fitness:.4f}, selected={self.n_selected_features}/{self.n_features})"
    
    def __repr__(self):
        """
        Programming representation
        """
        return self.__str__()
    
    def __lt__(self, other):
        """
        Less than comparison operator (for sorting by fitness)
        """
        return self.fitness < other.fitness
    
    def __eq__(self, other):
        """
        Equality comparison based on gene sequence
        """
        if not isinstance(other, Chromosome):
            return False
        return np.array_equal(self.genes, other.genes)
    
    def hamming_distance(self, other):
        """
        Calculate Hamming distance between two chromosomes
        
        The Hamming distance represents the number of differing genes,
        useful for measuring population diversity.
        
        Parameters:
        -----------
        other : Chromosome
            Another chromosome instance for comparison
            
        Returns:
        --------
        int
            Number of differing gene positions
        """
        if not isinstance(other, Chromosome):
            raise TypeError("Comparison requires another Chromosome instance")
        
        return np.sum(self.genes != other.genes)
    
    @staticmethod
    def create_random_population(population_size, n_features):
        """
        Generate a random population of chromosomes
        
        Parameters:
        -----------
        population_size : int
            Number of chromosomes to create
        n_features : int
            Number of features (genes per chromosome)
            
        Returns:
        --------
        list
            List of randomly initialized Chromosome objects
        """
        population = []
        for _ in range(population_size):
            chromosome = Chromosome(n_features)
            population.append(chromosome)
        return population
    
    @staticmethod
    def create_uniform_population(population_size, n_features, selection_probability=0.5):
        """
        Create population with uniform feature selection probability
        
        This method allows control over the expected number of selected
        features in the initial population.
        
        Parameters:
        -----------
        population_size : int
            Number of chromosomes to generate
        n_features : int
            Number of features per chromosome
        selection_probability : float
            Probability of each feature being selected (default: 0.5)
            
        Returns:
        --------
        list
            List of Chromosome objects with controlled feature density
        """
        population = []
        for _ in range(population_size):
            genes = (np.random.random(n_features) < selection_probability).astype(int)
            # Ensure at least one feature is selected
            if genes.sum() == 0:
                genes[random.randint(0, n_features-1)] = 1
            chromosome = Chromosome(n_features, genes=genes)
            population.append(chromosome)
        return population


# Module Test & Demonstration
if __name__ == '__main__':
    print("="*70)
    print("CHROMOSOME MODULE DEMONSTRATION")
    print("="*70)
    
    # Create a random chromosome with 10 features
    n_features = 10
    chromosome1 = Chromosome(n_features)
    
    print(f"\n1. Random Chromosome Generation:")
    print(f"   {chromosome1}")
    print(f"   Selected features: {chromosome1.get_selected_features()}")
    
    # Create a specific chromosome
    genes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    chromosome2 = Chromosome(n_features, genes=genes)
    print(f"\n2. Specific Chromosome Creation:")
    print(f"   {chromosome2}")
    print(f"   Selected features: {chromosome2.get_selected_features()}")
    
    # Demonstrate mutation
    print(f"\n3. Mutation Operation:")
    print(f"   Before mutation: {chromosome2.genes}")
    chromosome2.flip_gene(1)
    print(f"   After flipping gene at index 1: {chromosome2.genes}")
    
    # Calculate Hamming distance
    distance = chromosome1.hamming_distance(chromosome2)
    print(f"\n4. Hamming Distance Calculation:")
    print(f"   Distance between chromosomes: {distance} genes differ")
    
    # Create random population
    population = Chromosome.create_random_population(5, n_features)
    print(f"\n5. Population Generation (size=5):")
    for i, chrom in enumerate(population):
        print(f"   Individual {i+1}: {chrom}")
    
    print("\n" + "="*70)

