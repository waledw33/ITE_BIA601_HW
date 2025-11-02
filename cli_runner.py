#!/usr/bin/env python3
"""
Command-Line Interface for Feature Selection Framework
Allows execution of genetic algorithm experiments via terminal
Part of: Advanced Feature Selection Research Project
Research Team: AI & Machine Learning Laboratory
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from genetic_algorithm.ga_engine import GeneticAlgorithm
from data_processing.loader import FileReader
from data_processing.preprocessor import DataPreprocessor
import numpy as np


class FeatureSelectionCLI:
    """
    Command-Line Interface Handler for Genetic Algorithm Feature Selection
    """
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self):
        """
        Create argument parser with all CLI options
        """
        parser = argparse.ArgumentParser(
            description='Advanced Feature Selection using Genetic Algorithms',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  Basic usage:
    python cli_runner.py --dataset data.csv --target class

  Full configuration:
    python cli_runner.py \\
        --dataset data.csv \\
        --target outcome \\
        --population 100 \\
        --generations 200 \\
        --crossover 0.8 \\
        --mutation 0.1 \\
        --model random_forest \\
        --output results.json

  With visualization:
    python cli_runner.py --dataset data.csv --target class --plot
            """
        )
        
        # Required arguments
        parser.add_argument(
            '--dataset',
            type=str,
            required=True,
            help='Path to dataset file (CSV or Excel)'
        )
        
        parser.add_argument(
            '--target',
            type=str,
            required=True,
            help='Name of target column in dataset'
        )
        
        # Genetic Algorithm parameters
        ga_group = parser.add_argument_group('Genetic Algorithm Parameters')
        
        ga_group.add_argument(
            '--population',
            type=int,
            default=50,
            help='Population size (default: 50)'
        )
        
        ga_group.add_argument(
            '--generations',
            type=int,
            default=100,
            help='Number of generations (default: 100)'
        )
        
        ga_group.add_argument(
            '--crossover',
            type=float,
            default=0.8,
            help='Crossover rate (default: 0.8)'
        )
        
        ga_group.add_argument(
            '--mutation',
            type=float,
            default=0.1,
            help='Mutation rate (default: 0.1)'
        )
        
        ga_group.add_argument(
            '--elitism',
            type=float,
            default=0.1,
            help='Elitism rate (default: 0.1)'
        )
        
        ga_group.add_argument(
            '--alpha',
            type=float,
            default=0.9,
            help='Accuracy weight in fitness function (default: 0.9)'
        )
        
        # Model selection
        parser.add_argument(
            '--model',
            type=str,
            choices=['random_forest', 'svm', 'knn'],
            default='random_forest',
            help='Machine learning model for evaluation (default: random_forest)'
        )
        
        # Selection method
        parser.add_argument(
            '--selection',
            type=str,
            choices=['tournament', 'roulette', 'rank'],
            default='tournament',
            help='Parent selection method (default: tournament)'
        )
        
        # Crossover method
        parser.add_argument(
            '--crossover-method',
            type=str,
            choices=['single_point', 'two_point', 'uniform'],
            default='single_point',
            help='Crossover operator type (default: single_point)'
        )
        
        # Output options
        output_group = parser.add_argument_group('Output Options')
        
        output_group.add_argument(
            '--output',
            type=str,
            default='results.json',
            help='Output file for results (default: results.json)'
        )
        
        output_group.add_argument(
            '--plot',
            action='store_true',
            help='Display evolution plot after completion'
        )
        
        output_group.add_argument(
            '--verbose',
            action='store_true',
            help='Enable detailed progress output'
        )
        
        output_group.add_argument(
            '--quiet',
            action='store_true',
            help='Suppress all output except final results'
        )
        
        return parser
    
    def load_and_preprocess_data(self, dataset_path, target_column, verbose=True):
        """
        Load and preprocess dataset
        
        Parameters:
        -----------
        dataset_path : str
            Path to dataset file
        target_column : str
            Name of target column
        verbose : bool
            Enable progress messages
            
        Returns:
        --------
        tuple : (X, y, feature_names)
        """
        if verbose:
            print(f"\n{'='*70}")
            print("DATA LOADING & PREPROCESSING")
            print(f"{'='*70}")
        
        # Load data
        if verbose:
            print(f"\nLoading dataset: {dataset_path}")
        
        loader = FileReader()
        df = loader.auto_load(dataset_path)
        
        if verbose:
            print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        
        # Check target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found. Available: {df.columns.tolist()}")
        
        # Preprocess
        if verbose:
            print("\nPreprocessing data...")
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(df, strategy='mean')
        df_encoded = preprocessor.encode_categorical_features(df_clean)
        X, y = preprocessor.separate_features_target(df_encoded, target_column)
        X_normalized = preprocessor.normalize_features(X.values, method='standard')
        feature_names = X.columns.tolist()
        
        if verbose:
            print(f"Preprocessing complete: {X_normalized.shape[1]} features ready")
        
        return X_normalized, y.values, feature_names
    
    def run_experiment(self, args):
        """
        Execute genetic algorithm experiment
        
        Parameters:
        -----------
        args : Namespace
            Parsed command-line arguments
        """
        verbose = args.verbose and not args.quiet
        
        # Print header
        if not args.quiet:
            print("\n" + "="*70)
            print("GENETIC ALGORITHM FEATURE SELECTION FRAMEWORK")
            print("="*70)
            print("\nResearch Project: Advanced Feature Selection")
            print("AI & Machine Learning Laboratory")
            print("="*70)
        
        # Load and preprocess data
        try:
            X, y, feature_names = self.load_and_preprocess_data(
                args.dataset,
                args.target,
                verbose=verbose
            )
        except Exception as e:
            print(f"\nError loading dataset: {str(e)}", file=sys.stderr)
            sys.exit(1)
        
        # Initialize genetic algorithm
        if verbose:
            print(f"\n{'='*70}")
            print("GENETIC ALGORITHM CONFIGURATION")
            print(f"{'='*70}")
            print(f"Population Size: {args.population}")
            print(f"Generations: {args.generations}")
            print(f"Crossover Rate: {args.crossover}")
            print(f"Mutation Rate: {args.mutation}")
            print(f"Elitism Rate: {args.elitism}")
            print(f"Alpha (Accuracy Weight): {args.alpha}")
            print(f"ML Model: {args.model}")
            print(f"Selection Method: {args.selection}")
            print(f"Crossover Method: {args.crossover_method}")
            print(f"{'='*70}")
        
        ga = GeneticAlgorithm(
            population_size=args.population,
            n_generations=args.generations,
            crossover_rate=args.crossover,
            mutation_rate=args.mutation,
            elitism_rate=args.elitism,
            alpha=args.alpha,
            model_type=args.model,
            selection_method=args.selection,
            crossover_method=args.crossover_method,
            verbose=verbose
        )
        
        # Run genetic algorithm
        try:
            best_solution = ga.fit(X, y)
        except Exception as e:
            print(f"\nError during GA execution: {str(e)}", file=sys.stderr)
            sys.exit(1)
        
        # Get selected features
        selected_indices = best_solution.get_selected_features()
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Prepare results
        results = {
            'dataset': args.dataset,
            'target_column': args.target,
            'total_features': len(feature_names),
            'selected_features_count': len(selected_features),
            'selected_features': selected_features,
            'selected_indices': selected_indices.tolist(),
            'fitness_score': float(best_solution.fitness),
            'accuracy': float(best_solution.accuracy),
            'feature_reduction_percent': float((1 - len(selected_features) / len(feature_names)) * 100),
            'ga_parameters': {
                'population_size': args.population,
                'n_generations': args.generations,
                'crossover_rate': args.crossover,
                'mutation_rate': args.mutation,
                'elitism_rate': args.elitism,
                'alpha': args.alpha,
                'model_type': args.model,
                'selection_method': args.selection,
                'crossover_method': args.crossover_method
            },
            'evolution_history': {
                'best_fitness': [float(x) for x in ga.history['best_fitness']],
                'avg_fitness': [float(x) for x in ga.history['avg_fitness']],
                'best_accuracy': [float(x) for x in ga.history['best_accuracy']],
                'avg_features': [float(x) for x in ga.history['avg_features']]
            }
        }
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not args.quiet:
            print(f"\n{'='*70}")
            print("RESULTS SAVED")
            print(f"{'='*70}")
            print(f"Output file: {output_path.absolute()}")
            print(f"{'='*70}\n")
        
        # Plot evolution if requested
        if args.plot:
            try:
                ga.plot_history()
            except Exception as e:
                print(f"\nWarning: Could not display plot: {str(e)}", file=sys.stderr)
        
        return results
    
    def run(self):
        """
        Main CLI execution method
        """
        args = self.parser.parse_args()
        
        # Validate arguments
        if args.verbose and args.quiet:
            print("Error: Cannot use --verbose and --quiet together", file=sys.stderr)
            sys.exit(1)
        
        if not (0 <= args.crossover <= 1):
            print("Error: Crossover rate must be between 0 and 1", file=sys.stderr)
            sys.exit(1)
        
        if not (0 <= args.mutation <= 1):
            print("Error: Mutation rate must be between 0 and 1", file=sys.stderr)
            sys.exit(1)
        
        if not (0 <= args.alpha <= 1):
            print("Error: Alpha must be between 0 and 1", file=sys.stderr)
            sys.exit(1)
        
        # Run experiment
        try:
            results = self.run_experiment(args)
            sys.exit(0)
        except KeyboardInterrupt:
            print("\n\nExperiment interrupted by user.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    cli = FeatureSelectionCLI()
    cli.run()


if __name__ == '__main__':
    main()

