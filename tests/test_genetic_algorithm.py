"""
Unit Tests for Genetic Algorithm Module
Tests for Chromosome, Fitness, Operators, and GA Engine
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genetic_algorithm.chromosome import Chromosome
from genetic_algorithm.operators import GeneticOperators
from genetic_algorithm.fitness import FitnessEvaluator
from genetic_algorithm.ga_engine import GeneticAlgorithm


class TestChromosome(unittest.TestCase):
    """Test cases for Chromosome class"""
    
    def test_chromosome_creation(self):
        """Test chromosome creation"""
        chromosome = Chromosome(n_features=10)
        self.assertEqual(len(chromosome.genes), 10)
        self.assertTrue(chromosome.genes.sum() > 0)  # At least one gene is 1
        print("✓ Chromosome creation test passed")
    
    def test_get_selected_features(self):
        """Test getting selected features"""
        genes = [1, 0, 1, 0, 1]
        chromosome = Chromosome(5, genes=genes)
        selected = chromosome.get_selected_features()
        expected = np.array([0, 2, 4])
        np.testing.assert_array_equal(selected, expected)
        print("✓ Get selected features test passed")
    
    def test_flip_gene(self):
        """Test gene flipping"""
        genes = [1, 0, 1, 0, 1]
        chromosome = Chromosome(5, genes=genes)
        original = chromosome.genes[0]
        chromosome.flip_gene(0)
        self.assertEqual(chromosome.genes[0], 1 - original)
        print("✓ Gene flipping test passed")


class TestGeneticOperators(unittest.TestCase):
    """Test cases for Genetic Operators"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.operators = GeneticOperators(crossover_rate=0.8, mutation_rate=0.1)
        self.parent1 = Chromosome(10, genes=[1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
        self.parent2 = Chromosome(10, genes=[0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        self.parent1.fitness = 0.8
        self.parent2.fitness = 0.7
    
    def test_crossover(self):
        """Test crossover operation"""
        child1, child2 = self.operators.single_point_crossover(self.parent1, self.parent2)
        self.assertEqual(len(child1.genes), 10)
        self.assertEqual(len(child2.genes), 10)
        self.assertTrue(child1.genes.sum() > 0)
        self.assertTrue(child2.genes.sum() > 0)
        print("✓ Crossover test passed")
    
    def test_mutation(self):
        """Test mutation operation"""
        original = self.parent1.copy()
        mutated = self.operators.bit_flip_mutation(self.parent1)
        self.assertEqual(len(mutated.genes), len(original.genes))
        print("✓ Mutation test passed")
    
    def test_tournament_selection(self):
        """Test tournament selection"""
        population = [self.parent1, self.parent2]
        selected = self.operators.tournament_selection(population, tournament_size=2)
        self.assertIsInstance(selected, Chromosome)
        print("✓ Tournament selection test passed")


class TestFitnessEvaluator(unittest.TestCase):
    """Test cases for Fitness Evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create simple dataset
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        self.X = X[:50]  # Use subset for speed
        self.y = y[:50]
        self.evaluator = FitnessEvaluator(self.X, self.y, model_type='random_forest')
    
    def test_evaluate_chromosome(self):
        """Test chromosome evaluation"""
        chromosome = Chromosome(self.X.shape[1])
        fitness = self.evaluator.evaluate_chromosome(chromosome)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
        print("✓ Chromosome evaluation test passed")
    
    def test_evaluate_population(self):
        """Test population evaluation"""
        population = Chromosome.create_random_population(5, self.X.shape[1])
        fitness_scores = self.evaluator.evaluate_population(population)
        self.assertEqual(len(fitness_scores), 5)
        print("✓ Population evaluation test passed")


class TestGeneticAlgorithm(unittest.TestCase):
    """Test cases for GA Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        self.X = X[:50]  # Use subset for speed
        self.y = y[:50]
    
    def test_ga_initialization(self):
        """Test GA initialization"""
        ga = GeneticAlgorithm(
            population_size=10,
            n_generations=5,
            verbose=False
        )
        self.assertEqual(ga.population_size, 10)
        self.assertEqual(ga.n_generations, 5)
        print("✓ GA initialization test passed")
    
    def test_ga_fit(self):
        """Test GA fitting"""
        ga = GeneticAlgorithm(
            population_size=10,
            n_generations=5,
            verbose=False
        )
        best = ga.fit(self.X, self.y)
        self.assertIsInstance(best, Chromosome)
        self.assertGreater(best.fitness, 0)
        print("✓ GA fitting test passed")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("RUNNING GENETIC ALGORITHM TESTS")
    print("="*60 + "\n")
    
    unittest.main(verbosity=2)

