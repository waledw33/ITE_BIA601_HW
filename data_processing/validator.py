"""
Data Validator Module
Validates data quality and structure
Developed by: Student 1
"""

import pandas as pd
import numpy as np


class DataValidator:
    """
    DataValidator class for checking data quality and structure
    Ensures data meets requirements for feature selection
    """
    
    def __init__(self):
        """Initialize the DataValidator"""
        self.validation_results = {}
    
    def validate_structure(self, df):
        """
        Validate basic data structure
        
        Args:
            df: pandas DataFrame
            
        Returns:
            bool: True if valid, False otherwise
        """
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("Dataset is empty")
        
        # Check if DataFrame has at least 2 columns (features + target)
        if len(df.columns) < 2:
            errors.append("Dataset must have at least 2 columns (features + target)")
        
        # Check if DataFrame has sufficient rows
        if len(df) < 10:
            errors.append("Dataset has too few samples (minimum 10 required)")
        
        # Store results
        self.validation_results['structure'] = {
            'valid': len(errors) == 0,
            'errors': errors
        }
        
        if errors:
            print("Structure Validation FAILED:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("Structure Validation PASSED")
            return True
    
    def check_missing_values(self, df, threshold=0.5):
        """
        Check for excessive missing values
        
        Args:
            df: pandas DataFrame
            threshold: Maximum allowed proportion of missing values (default: 0.5)
            
        Returns:
            bool: True if acceptable, False otherwise
        """
        warnings = []
        
        # Calculate missing value percentage for each column
        missing_pct = df.isnull().sum() / len(df)
        
        # Find columns with excessive missing values
        problematic_cols = missing_pct[missing_pct > threshold]
        
        if len(problematic_cols) > 0:
            warnings.append(f"{len(problematic_cols)} columns have >{threshold*100}% missing values")
            for col, pct in problematic_cols.items():
                warnings.append(f"  - {col}: {pct*100:.1f}% missing")
        
        # Calculate overall missing percentage
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        overall_pct = total_missing / total_cells
        
        # Store results
        self.validation_results['missing_values'] = {
            'overall_percentage': overall_pct,
            'problematic_columns': problematic_cols.to_dict(),
            'warnings': warnings
        }
        
        if warnings:
            print("Missing Values Check - WARNINGS:")
            for warning in warnings:
                print(f"  {warning}")
            return False
        else:
            print(f"Missing Values Check PASSED (overall: {overall_pct*100:.2f}%)")
            return True
    
    def check_data_types(self, df):
        """
        Check data types of columns
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: Information about data types
        """
        # Count different data types
        type_counts = df.dtypes.value_counts().to_dict()
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        info = {
            'type_distribution': type_counts,
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols,
            'n_numerical': len(numerical_cols),
            'n_categorical': len(categorical_cols)
        }
        
        self.validation_results['data_types'] = info
        
        print("Data Types:")
        print(f"  Numerical columns: {len(numerical_cols)}")
        print(f"  Categorical columns: {len(categorical_cols)}")
        
        return info
    
    def check_target_column(self, df, target_column='target'):
        """
        Validate target column
        
        Args:
            df: pandas DataFrame
            target_column: Name of target column
            
        Returns:
            bool: True if valid, False otherwise
        """
        errors = []
        
        # Check if target column exists
        if target_column not in df.columns:
            errors.append(f"Target column '{target_column}' not found")
            self.validation_results['target'] = {
                'valid': False,
                'errors': errors
            }
            return False
        
        # Get target values
        y = df[target_column]
        
        # Check number of classes
        n_classes = y.nunique()
        if n_classes < 2:
            errors.append(f"Target has only {n_classes} class (minimum 2 required)")
        
        # Check class distribution
        class_counts = y.value_counts()
        min_samples = class_counts.min()
        if min_samples < 2:
            errors.append(f"Some classes have too few samples (minimum 2 per class)")
        
        # Store results
        self.validation_results['target'] = {
            'valid': len(errors) == 0,
            'n_classes': n_classes,
            'class_distribution': class_counts.to_dict(),
            'errors': errors
        }
        
        if errors:
            print("Target Column Validation FAILED:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print(f"Target Column Validation PASSED ({n_classes} classes)")
            print(f"  Class distribution: {class_counts.to_dict()}")
            return True
    
    def check_feature_variance(self, df, target_column='target', threshold=0.01):
        """
        Check for low-variance features
        
        Args:
            df: pandas DataFrame
            target_column: Name of target column
            threshold: Minimum variance threshold
            
        Returns:
            list: Columns with low variance
        """
        # Get feature columns (exclude target)
        feature_cols = [col for col in df.columns if col != target_column]
        
        # Calculate variance for numerical columns
        low_variance_cols = []
        for col in feature_cols:
            if df[col].dtype in [np.number]:
                variance = df[col].var()
                if variance < threshold:
                    low_variance_cols.append((col, variance))
        
        if low_variance_cols:
            print(f"Warning: {len(low_variance_cols)} features have low variance:")
            for col, var in low_variance_cols:
                print(f"  - {col}: variance = {var:.6f}")
        else:
            print("All features have acceptable variance")
        
        return low_variance_cols
    
    def validate_dataset(self, df, target_column='target'):
        """
        Comprehensive dataset validation
        
        Args:
            df: pandas DataFrame
            target_column: Name of target column
            
        Returns:
            bool: True if all validations pass, False otherwise
        """
        print("\n" + "="*60)
        print("DATASET VALIDATION")
        print("="*60)
        
        all_valid = True
        
        # Run all validation checks
        print("\n[1/5] Validating structure...")
        if not self.validate_structure(df):
            all_valid = False
        
        print("\n[2/5] Checking missing values...")
        self.check_missing_values(df)
        
        print("\n[3/5] Checking data types...")
        self.check_data_types(df)
        
        print("\n[4/5] Validating target column...")
        if not self.check_target_column(df, target_column):
            all_valid = False
        
        print("\n[5/5] Checking feature variance...")
        self.check_feature_variance(df, target_column)
        
        print("\n" + "="*60)
        if all_valid:
            print("✓ DATASET VALIDATION PASSED")
        else:
            print("✗ DATASET VALIDATION FAILED")
        print("="*60)
        
        return all_valid
    
    def get_validation_report(self):
        """
        Get detailed validation report
        
        Returns:
            dict: Validation results
        """
        return self.validation_results


# Example usage
if __name__ == '__main__':
    from loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_file('sample_datasets/iris.csv')
    
    # Validate data
    validator = DataValidator()
    is_valid = validator.validate_dataset(df)
    
    if is_valid:
        print("\nDataset is ready for processing!")
    else:
        print("\nDataset has issues that need to be addressed")
        print("\nValidation Report:")
        print(validator.get_validation_report())

