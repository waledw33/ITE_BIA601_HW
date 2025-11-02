"""
Web Application Routes - Feature Selection Framework
API endpoints and view handlers for the research platform
Part of: Advanced Feature Selection Research Project
Research Team: AI & Machine Learning Laboratory
"""

from flask import Blueprint, render_template, request, jsonify, send_file
import os
import sys
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import json
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from genetic_algorithm.ga_engine import GeneticAlgorithm
from genetic_algorithm.chromosome import Chromosome
from data_processing.loader import FileReader
from data_processing.preprocessor import DataPreprocessor
from comparison.traditional_methods import TraditionalFeatureSelection
from models.evaluator import ModelEvaluator

# Create Blueprint
web_routes = Blueprint('main', __name__)

# Allowed file types
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== VIEW ROUTES ====================

@web_routes.route('/')
def index():
    """
    Home page - Main landing page
    Displays project overview and navigation
    """
    return render_template('index_new.html')


@web_routes.route('/team')
def team():
    """
    Research team page
    Displays team members and their roles
    """
    return render_template('team.html')


@web_routes.route('/upload')
def upload_page():
    """
    Upload page - Dataset upload interface
    Interface for uploading datasets and configuring experiments
    """
    return render_template('upload_new.html')


@web_routes.route('/results')
def results_page():
    """
    Results page
    Displays GA results and visualizations
    """
    return render_template('results_new.html')


@web_routes.route('/comparison')
def comparison_page():
    """
    Comparison page
    Displays comparison with traditional methods
    """
    return render_template('comparison.html')


@web_routes.route('/documentation')
def documentation():
    """
    Documentation page
    Technical documentation and API reference
    """
    return render_template('documentation.html')


# ==================== API ENDPOINTS ====================

@web_routes.route('/api/upload', methods=['POST'])
def upload_file():
    """
    File upload endpoint
    Accepts CSV/Excel files and validates them
    """
    # Check if file is present in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only CSV and Excel files are allowed'}), 400
    
    try:
        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        # Quick file read test
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Prepare file metadata
        metadata = {
            'filename': filename,
            'filepath': filepath,
            'rows': len(df),
            'columns': len(df.columns),
            'features': df.columns.tolist(),
            'success': True
        }
        
        return jsonify(metadata), 200
        
    except Exception as e:
        return jsonify({'error': f'File processing error: {str(e)}'}), 500


@web_routes.route('/api/run-ga', methods=['POST'])
def run_genetic_algorithm():
    """
    Genetic algorithm execution endpoint
    Receives parameters and runs the GA on uploaded dataset
    """
    try:
        print("\n" + "="*70)
        print("GA EXECUTION REQUEST RECEIVED")
        print("="*70)
        
        params = request.get_json()
        print(f"Parameters received: {params}")
        
        # Extract parameters
        filepath = params.get('filepath')
        population_size = params.get('population_size', 50)
        n_generations = params.get('generations', 100)
        crossover_rate = params.get('crossover_rate', 0.8)
        mutation_rate = params.get('mutation_rate', 0.1)
        model_type = params.get('model_type', 'random_forest')
        target_column = params.get('target_column', 'target')
        
        print(f"\nConfiguration:")
        print(f"  Filepath: {filepath}")
        print(f"  Target: {target_column}")
        print(f"  Population: {population_size}")
        print(f"  Generations: {n_generations}")
        
        # Validate filepath
        if not filepath:
            return jsonify({'error': 'Filepath is required'}), 400
        
        # Convert relative paths to absolute paths
        if not os.path.isabs(filepath):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            filepath = os.path.join(base_dir, filepath)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filepath}'}), 404
        
        # Load data
        print("\n[1/6] Loading data...")
        loader = FileReader()
        df = loader.auto_load(filepath)
        print(f"  Data loaded: {df.shape}")
        
        # Check if target column exists
        if target_column not in df.columns:
            return jsonify({
                'error': f'Target column "{target_column}" not found in dataset. Available columns: {df.columns.tolist()}'
            }), 400
        
        # Preprocess data
        print("\n[2/6] Preprocessing data...")
        preprocessor = DataPreprocessor()
        
        # Handle missing values and encode categorical
        print("  Handling missing values...")
        df_clean = preprocessor.handle_missing_values(df, strategy='mean')
        print("  Encoding categorical features...")
        df_encoded = preprocessor.encode_categorical_features(df_clean)
        
        # Separate features and target
        print("  Separating features and target...")
        X, y = preprocessor.separate_features_target(df_encoded, target_column)
        
        # Normalize features
        print("  Normalizing features...")
        X_normalized = preprocessor.normalize_features(X.values, method='standard')
        
        # Store feature names
        feature_names = X.columns.tolist()
        print(f"  Features ready: {len(feature_names)} features")
        
        # Create and run Genetic Algorithm
        print("\n[3/6] Initializing Genetic Algorithm...")
        ga = GeneticAlgorithm(
            population_size=population_size,
            n_generations=n_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_rate=0.1,
            alpha=0.9,
            model_type=model_type,
            verbose=False  # Disable console output for web
        )
        
        # Run evolution
        print("\n[4/6] Running genetic algorithm...")
        best_solution = ga.fit(X_normalized, y.values)
        print("  Evolution complete!")
        
        # Get results
        print("\n[5/6] Extracting results...")
        selected_indices = best_solution.get_selected_features().tolist()
        selected_feature_names = [feature_names[i] for i in selected_indices]
        history = ga.get_history()
        print(f"  Selected {len(selected_indices)} features")
        
        # Prepare result
        print("\n[6/6] Preparing response...")
        result = {
            'success': True,
            'selected_features': selected_feature_names,
            'selected_indices': selected_indices,
            'fitness_score': float(best_solution.fitness),
            'accuracy': float(best_solution.accuracy),
            'n_selected_features': int(best_solution.n_selected_features),
            'n_total_features': len(feature_names),
            'feature_reduction_percent': float((1 - best_solution.n_selected_features / len(feature_names)) * 100),
            'generation_history': {
                'best_fitness': [float(x) for x in history['best_fitness']],
                'avg_fitness': [float(x) for x in history['avg_fitness']],
                'worst_fitness': [float(x) for x in history['worst_fitness']],
                'best_accuracy': [float(x) for x in history['best_accuracy']],
                'avg_features': [float(x) for x in history['avg_features']]
            },
            'message': 'Genetic algorithm completed successfully'
        }
        
        print("\n✅ GA EXECUTION COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
        return jsonify(result), 200
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("\n❌ ERROR IN GA EXECUTION:")
        print(error_details)
        print("="*70 + "\n")
        return jsonify({'error': f'Error running GA: {str(e)}'}), 500


@web_routes.route('/api/comparison', methods=['POST'])
def run_comparison():
    """
    API endpoint to run comparison analysis
    Compares GA results with traditional feature selection methods
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        filepath = data.get('filepath')
        target_column = data.get('target_column', 'target')
        k_features = data.get('k_features', 10)
        
        # Validate filepath
        if not filepath:
            return jsonify({'error': 'Filepath is required'}), 400
        
        # Convert relative paths to absolute paths
        if not os.path.isabs(filepath):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            filepath = os.path.join(base_dir, filepath)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filepath}'}), 404
        
        # Load and preprocess data
        loader = FileReader()
        df = loader.auto_load(filepath)
        
        # Check if target column exists
        if target_column not in df.columns:
            return jsonify({
                'error': f'Target column "{target_column}" not found. Available: {df.columns.tolist()}'
            }), 400
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(df, strategy='mean')
        df_encoded = preprocessor.encode_categorical_features(df_clean)
        X, y = preprocessor.separate_features_target(df_encoded, target_column)
        X_normalized = preprocessor.normalize_features(X.values, method='standard')
        feature_names = X.columns.tolist()
        
        # Initialize traditional feature selection
        selector = TraditionalFeatureSelection()
        evaluator = ModelEvaluator()
        
        # Run traditional methods
        methods_results = {}
        
        # Chi-Square
        try:
            chi_indices, chi_scores = selector.chi_square_selection(X_normalized, y.values, k=k_features)
            chi_features = [feature_names[i] for i in chi_indices]
            chi_metrics = evaluator.evaluate_features(X_normalized[:, chi_indices], y.values)
            methods_results['chi_square'] = {
                'features': chi_features,
                'n_features': len(chi_features),
                'accuracy': float(chi_metrics['accuracy']),
                'precision': float(chi_metrics['precision']),
                'recall': float(chi_metrics['recall']),
                'f1_score': float(chi_metrics['f1_score']),
                'time': float(chi_metrics.get('time', 0))
            }
        except Exception as e:
            print(f"Chi-Square error: {str(e)}")
            methods_results['chi_square'] = {'error': str(e)}
        
        # ANOVA F-test
        try:
            anova_indices, anova_scores = selector.anova_f_test(X_normalized, y.values, k=k_features)
            anova_features = [feature_names[i] for i in anova_indices]
            anova_metrics = evaluator.evaluate_features(X_normalized[:, anova_indices], y.values)
            methods_results['anova_f'] = {
                'features': anova_features,
                'n_features': len(anova_features),
                'accuracy': float(anova_metrics['accuracy']),
                'precision': float(anova_metrics['precision']),
                'recall': float(anova_metrics['recall']),
                'f1_score': float(anova_metrics['f1_score']),
                'time': float(anova_metrics.get('time', 0))
            }
        except Exception as e:
            print(f"ANOVA error: {str(e)}")
            methods_results['anova_f'] = {'error': str(e)}
        
        # Mutual Information
        try:
            mi_indices, mi_scores = selector.mutual_information(X_normalized, y.values, k=k_features)
            mi_features = [feature_names[i] for i in mi_indices]
            mi_metrics = evaluator.evaluate_features(X_normalized[:, mi_indices], y.values)
            methods_results['mutual_info'] = {
                'features': mi_features,
                'n_features': len(mi_features),
                'accuracy': float(mi_metrics['accuracy']),
                'precision': float(mi_metrics['precision']),
                'recall': float(mi_metrics['recall']),
                'f1_score': float(mi_metrics['f1_score']),
                'time': float(mi_metrics.get('time', 0))
            }
        except Exception as e:
            print(f"Mutual Information error: {str(e)}")
            methods_results['mutual_info'] = {'error': str(e)}
        
        # Random Forest Importance
        try:
            rf_indices, rf_scores = selector.random_forest_importance(X_normalized, y.values)
            # Select top k features
            if len(rf_indices) > k_features:
                top_k_idx = np.argsort(rf_scores[rf_indices])[-k_features:]
                rf_indices = rf_indices[top_k_idx]
            rf_features = [feature_names[i] for i in rf_indices]
            rf_metrics = evaluator.evaluate_features(X_normalized[:, rf_indices], y.values)
            methods_results['rf_importance'] = {
                'features': rf_features,
                'n_features': len(rf_features),
                'accuracy': float(rf_metrics['accuracy']),
                'precision': float(rf_metrics['precision']),
                'recall': float(rf_metrics['recall']),
                'f1_score': float(rf_metrics['f1_score']),
                'time': float(rf_metrics.get('time', 0))
            }
        except Exception as e:
            print(f"Random Forest error: {str(e)}")
            methods_results['rf_importance'] = {'error': str(e)}
        
        # Statistical analysis
        statistical_analysis = {
            'total_features': len(feature_names),
            'methods_compared': len([m for m in methods_results.values() if 'error' not in m]),
            'summary': 'Comparison completed successfully'
        }
        
        result = {
            'success': True,
            'traditional_results': methods_results,
            'statistical_analysis': statistical_analysis,
            'message': 'Comparison analysis completed'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in run_comparison: {error_details}")
        return jsonify({'error': f'Error running comparison: {str(e)}'}), 500


@web_routes.route('/api/datasets')
def list_sample_datasets():
    """
    API endpoint to list available sample datasets
    Returns list of pre-loaded benchmark datasets
    """
    try:
        datasets_dir = 'sample_datasets'
        if not os.path.exists(datasets_dir):
            return jsonify({'datasets': []}), 200
        
        datasets = []
        for filename in os.listdir(datasets_dir):
            if filename.endswith(('.csv', '.xlsx', '.xls')):
                filepath = os.path.join(datasets_dir, filename)
                datasets.append({
                    'name': filename,
                    'path': filepath
                })
        
        return jsonify({'datasets': datasets}), 200
        
    except Exception as e:
        return jsonify({'error': f'Error listing datasets: {str(e)}'}), 500


# Error handlers
@web_routes.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@web_routes.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500
