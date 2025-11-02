"""
Genetic Operators Module - Evolutionary Mechanics
Implements core genetic algorithm operations: selection, crossover, and mutation
Part of: Advanced Feature Selection Research Project
Research Team: AI & Machine Learning Laboratory
"""

import numpy as np
import random
from .chromosome import Chromosome


class GeneticOperators:
    """
    Genetic Operators for Chromosome Manipulation
    
    This class provides all the genetic operations required for
    evolutionary algorithm execution, including:
    - Selection strategies (tournament, roulette, rank, elitism)
    - Crossover methods (single-point, two-point, uniform)
    - Mutation operators (bit-flip, random resetting, swap)
    
    These operations collectively enable exploration and exploitation
    of the feature selection search space.
    """
    
    def __init__(self, crossover_rate=0.8, mutation_rate=0.1):
        """
        Initialize genetic operators with specified rates
        
        Parameters:
        -----------
        crossover_rate : float, default=0.8
            Probability of performing crossover between parents (0.0-1.0)
            Higher values increase exploration of gene combinations
        mutation_rate : float, default=0.1
            Probability of mutating each gene (0.0-1.0)
            Higher values increase random variation
        """
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    # ==================== SELECTION OPERATORS ====================
    
    def tournament_selection(self, population, tournament_size=3):
        """
        Tournament selection - competitive selection strategy
        
        Randomly selects a subset of individuals and returns the
        best performer. This method balances selection pressure
        and diversity preservation.
        
        Parameters:
        -----------
        population : list
            List of Chromosome objects
        tournament_size : int, default=3
            Number of competitors in each tournament
            
        Returns:
        --------
        Chromosome
            Winner of the tournament (copy)
        """
        # Randomly select tournament participants
        tournament = random.sample(population, tournament_size)
        
        # Return the fittest individual
        winner = max(tournament, key=lambda c: c.fitness)
        return winner.copy()
    
    def roulette_wheel_selection(self, population):
        """
        Roulette wheel selection - fitness-proportionate selection
        
        Selection probability is proportional to fitness score.
        This method tends to favor high-fitness individuals but
        still gives chances to weaker ones.
        
        Parameters:
        -----------
        population : list
            List of Chromosome objects
            
        Returns:
        --------
        Chromosome
            Selected individual (copy)
        """
        # Extract fitness values
        fitness_values = np.array([c.fitness for c in population])
        
        # Handle negative fitness (shift to positive range)
        if fitness_values.min() < 0:
            fitness_values = fitness_values - fitness_values.min()
        
        # Calculate selection probabilities
        total_fitness = fitness_values.sum()
        if total_fitness == 0:
            # If all fitness values are 0, select randomly
            return random.choice(population).copy()
        
        probabilities = fitness_values / total_fitness
        
        # Select based on probabilities
        selected_idx = np.random.choice(len(population), p=probabilities)
        return population[selected_idx].copy()
    
    def rank_selection(self, population):
        """
        Rank-based selection - position-based selection
        
        Selection probability based on rank rather than raw fitness.
        This prevents premature convergence when fitness differences
        are large.
        
        Parameters:
        -----------
        population : list
            List of Chromosome objects
            
        Returns:
        --------
        Chromosome
            Selected individual (copy)
        """
        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda c: c.fitness)
        
        # Assign ranks (1 to N, where N is population size)
        ranks = np.arange(1, len(population) + 1)
        
        # Calculate selection probabilities based on ranks
        total_rank = ranks.sum()
        probabilities = ranks / total_rank
        
        # Select based on rank probabilities
        selected_idx = np.random.choice(len(population), p=probabilities)
        return sorted_pop[selected_idx].copy()
    
    def elitism_selection(self, population, n_elite):
        """
        Elitism - preserve best individuals
        
        Directly transfers the top-performing individuals to the
        next generation without modification. This ensures that
        the best solutions found are never lost.
        
        Parameters:
        -----------
        population : list
            List of Chromosome objects
        n_elite : int
            Number of elite individuals to preserve
            
        Returns:
        --------
        list
            Top n_elite chromosomes (copies)
        """
        # Sort by fitness (descending order)
        sorted_pop = sorted(population, key=lambda c: c.fitness, reverse=True)
        
        # Return top N individuals
        return [c.copy() for c in sorted_pop[:n_elite]]
    
    # ==================== CROSSOVER OPERATORS ====================
    
    def single_point_crossover(self, parent1, parent2):
        """
        Single-point crossover
        
        Splits parent genes at a random point and exchanges the segments.
        This is the simplest and most commonly used crossover method.
        
        Example:
            Parent1: [1,1,0,0,1,1,0,0]  Point = 4
            Parent2: [0,0,1,1,0,0,1,1]
            Child1:  [1,1,0,0|0,0,1,1]
            Child2:  [0,0,1,1|1,1,0,0]
        
        Parameters:
        -----------
        parent1, parent2 : Chromosome
            Parent chromosomes for breeding
            
        Returns:
        --------
        tuple
            (child1, child2) - Two offspring chromosomes
        """
        # Probabilistically decide whether to perform crossover
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Select random crossover point
        n_genes = parent1.n_features
        crossover_point = random.randint(1, n_genes - 1)
        
        # Create offspring by combining parent segments
        child1_genes = np.concatenate([
            parent1.genes[:crossover_point],
            parent2.genes[crossover_point:]
        ])
        
        child2_genes = np.concatenate([
            parent2.genes[:crossover_point],
            parent1.genes[crossover_point:]
        ])
        
        # Ensure at least one feature is selected in each child
        if child1_genes.sum() == 0:
            child1_genes[random.randint(0, n_genes-1)] = 1
        if child2_genes.sum() == 0:
            child2_genes[random.randint(0, n_genes-1)] = 1
        
        child1 = Chromosome(n_genes, genes=child1_genes)
        child2 = Chromosome(n_genes, genes=child2_genes)
        
        return child1, child2
    
    def two_point_crossover(self, parent1, parent2):
        """
        Two-point crossover
        
        Splits at two random points and exchanges the middle segment.
        This allows for more complex gene combinations than single-point.
        
        Example:
            Parent1: [1,1|0,0,1,1|0,0]  Points = 2,6
            Parent2: [0,0|1,1,0,0|1,1]
            Child1:  [1,1|1,1,0,0|0,0]
            Child2:  [0,0|0,0,1,1|1,1]
        
        Parameters:
        -----------
        parent1, parent2 : Chromosome
            Parent chromosomes for breeding
            
        Returns:
        --------
        tuple
            (child1, child2) - Two offspring chromosomes
        """
        # Probabilistically decide whether to perform crossover
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        n_genes = parent1.n_features
        
        # Select two random crossover points
        point1 = random.randint(0, n_genes - 2)
        point2 = random.randint(point1 + 1, n_genes - 1)
        
        # Create offspring by exchanging middle segment
        child1_genes = np.concatenate([
            parent1.genes[:point1],
            parent2.genes[point1:point2],
            parent1.genes[point2:]
        ])
        
        child2_genes = np.concatenate([
            parent2.genes[:point1],
            parent1.genes[point1:point2],
            parent2.genes[point2:]
        ])
        
        # Ensure at least one feature is selected
        if child1_genes.sum() == 0:
            child1_genes[random.randint(0, n_genes-1)] = 1
        if child2_genes.sum() == 0:
            child2_genes[random.randint(0, n_genes-1)] = 1
        
        child1 = Chromosome(n_genes, genes=child1_genes)
        child2 = Chromosome(n_genes, genes=child2_genes)
        
        return child1, child2
    
    def uniform_crossover(self, parent1, parent2):
        """
        Uniform crossover
        
        Each gene is independently chosen from either parent with
        equal probability. This provides maximum mixing of parent genes.
        
        Example:
            Parent1: [1,1,0,0,1,1,0,0]
            Parent2: [0,0,1,1,0,0,1,1]
            Mask:    [1,0,1,0,1,0,1,0]
            Child1:  [1,0,0,1,1,0,0,1]
            Child2:  [0,1,1,0,0,1,1,0]
        
        Parameters:
        -----------
        parent1, parent2 : Chromosome
            Parent chromosomes for breeding
            
        Returns:
        --------
        tuple
            (child1, child2) - Two offspring chromosomes
        """
        # Probabilistically decide whether to perform crossover
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        n_genes = parent1.n_features
        
        # Create random binary mask
        mask = np.random.randint(0, 2, size=n_genes).astype(bool)
        
        # Create offspring using mask
        child1_genes = np.where(mask, parent1.genes, parent2.genes)
        child2_genes = np.where(mask, parent2.genes, parent1.genes)
        
        # Ensure at least one feature is selected
        if child1_genes.sum() == 0:
            child1_genes[random.randint(0, n_genes-1)] = 1
        if child2_genes.sum() == 0:
            child2_genes[random.randint(0, n_genes-1)] = 1
        
        child1 = Chromosome(n_genes, genes=child1_genes)
        child2 = Chromosome(n_genes, genes=child2_genes)
        
        return child1, child2
    
    # ==================== MUTATION OPERATORS ====================
    
    def bit_flip_mutation(self, chromosome):
        """
        Bit-flip mutation - standard binary mutation
        
        Each gene has a probability (mutation_rate) of being flipped
        (0→1 or 1→0). This introduces small random variations that
        help escape local optima.
        
        Parameters:
        -----------
        chromosome : Chromosome
            Individual to mutate
            
        Returns:
        --------
        Chromosome
            Mutated chromosome
        """
        mutated = chromosome.copy()
        
        # Iterate through each gene
        for i in range(mutated.n_features):
            # Mutate with probability mutation_rate
            if random.random() < self.mutation_rate:
                mutated.flip_gene(i)
        
        # Ensure at least one feature remains selected
        if mutated.genes.sum() == 0:
            random_idx = random.randint(0, mutated.n_features - 1)
            mutated.set_gene(random_idx, 1)
        
        return mutated
    
    def random_resetting_mutation(self, chromosome):
        """
        Random resetting mutation
        
        Similar to bit-flip, but instead of flipping, genes are
        randomly reset to 0 or 1. This can create larger jumps
        in the search space.
        
        Parameters:
        -----------
        chromosome : Chromosome
            Individual to mutate
            
        Returns:
        --------
        Chromosome
            Mutated chromosome
        """
        mutated = chromosome.copy()
        
        # Iterate through each gene
        for i in range(mutated.n_features):
            # Mutate with probability mutation_rate
            if random.random() < self.mutation_rate:
                mutated.set_gene(i, random.randint(0, 1))
        
        # Ensure at least one feature remains selected
        if mutated.genes.sum() == 0:
            random_idx = random.randint(0, mutated.n_features - 1)
            mutated.set_gene(random_idx, 1)
        
        return mutated
    
    def swap_mutation(self, chromosome):
        """
        Swap mutation
        
        Swaps the values of two randomly selected genes. This
        maintains the number of selected features while changing
        which features are selected.
        
        Parameters:
        -----------
        chromosome : Chromosome
            Individual to mutate
            
        Returns:
        --------
        Chromosome
            Mutated chromosome
        """
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        # Select two random gene positions
        idx1, idx2 = random.sample(range(mutated.n_features), 2)
        
        # Swap gene values
        temp = mutated.genes[idx1]
        mutated.genes[idx1] = mutated.genes[idx2]
        mutated.genes[idx2] = temp
        
        mutated.n_selected_features = mutated.count_selected()
        
        return mutated


# Module Test & Demonstration
if __name__ == '__main__':
    print("="*70)
    print("GENETIC OPERATORS MODULE DEMONSTRATION")
    print("="*70)
    
    # Initialize operators
    operators = GeneticOperators(crossover_rate=0.8, mutation_rate=0.1)
    
    # Create parent chromosomes
    n_features = 10
    parent1 = Chromosome(n_features, genes=[1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    parent2 = Chromosome(n_features, genes=[0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    
    # Assign fitness scores
    parent1.fitness = 0.85
    parent2.fitness = 0.75
    
    print("\nParent Chromosomes:")
    print(f"Parent 1: {parent1.genes} (fitness={parent1.fitness})")
    print(f"Parent 2: {parent2.genes} (fitness={parent2.fitness})")
    
    # Demonstrate crossover operators
    print("\n" + "="*70)
    print("CROSSOVER OPERATIONS")
    print("="*70)
    
    print("\n1. Single-point crossover:")
    child1, child2 = operators.single_point_crossover(parent1, parent2)
    print(f"   Child 1: {child1.genes}")
    print(f"   Child 2: {child2.genes}")
    
    print("\n2. Two-point crossover:")
    child1, child2 = operators.two_point_crossover(parent1, parent2)
    print(f"   Child 1: {child1.genes}")
    print(f"   Child 2: {child2.genes}")
    
    print("\n3. Uniform crossover:")
    child1, child2 = operators.uniform_crossover(parent1, parent2)
    print(f"   Child 1: {child1.genes}")
    print(f"   Child 2: {child2.genes}")
    
    # Demonstrate mutation operators
    print("\n" + "="*70)
    print("MUTATION OPERATIONS")
    print("="*70)
    
    test_chromosome = Chromosome(n_features, genes=[1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    print(f"\nOriginal chromosome: {test_chromosome.genes}")
    
    mutated = operators.bit_flip_mutation(test_chromosome)
    print(f"After bit-flip mutation: {mutated.genes}")
    
    mutated = operators.random_resetting_mutation(test_chromosome)
    print(f"After random resetting: {mutated.genes}")
    
    # Demonstrate selection operators
    print("\n" + "="*70)
    print("SELECTION OPERATIONS")
    print("="*70)
    
    # Create a population
    population = Chromosome.create_random_population(5, n_features)
    print("\nPopulation:")
    for i, chrom in enumerate(population):
        chrom.fitness = np.random.random()  # Assign random fitness
        print(f"  {i+1}. {chrom.genes} (fitness={chrom.fitness:.3f})")
    
    print("\nTournament selection (tournament_size=3):")
    selected = operators.tournament_selection(population, tournament_size=3)
    print(f"  Selected: {selected.genes} (fitness={selected.fitness:.3f})")
    
    print("\nRoulette wheel selection:")
    selected = operators.roulette_wheel_selection(population)
    print(f"  Selected: {selected.genes} (fitness={selected.fitness:.3f})")
    
    print("\nRank-based selection:")
    selected = operators.rank_selection(population)
    print(f"  Selected: {selected.genes} (fitness={selected.fitness:.3f})")
    
    print("\nElitism selection (n_elite=2):")
    elite = operators.elitism_selection(population, n_elite=2)
    print(f"  Elite 1: {elite[0].genes} (fitness={elite[0].fitness:.3f})")
    print(f"  Elite 2: {elite[1].genes} (fitness={elite[1].fitness:.3f})")
    
    print("\n" + "="*70)
