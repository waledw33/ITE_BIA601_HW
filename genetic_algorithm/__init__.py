"""
Genetic Algorithm for Feature Selection
Bio-inspired algorithmic implementation for intelligent attribute selection
Part of: Advanced Feature Selection Research Project
Research Team: AI & Machine Learning Laboratory
"""

from .chromosome import Chromosome
from .fitness import FitnessEvaluator
from .operators import GeneticOperators
from .ga_engine import GeneticAlgorithm

__all__ = ['Chromosome', 'FitnessEvaluator', 'GeneticOperators', 'GeneticAlgorithm']

