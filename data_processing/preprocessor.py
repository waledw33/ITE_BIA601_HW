"""
Data Preprocessor Module
Handles data preprocessing including cleaning, encoding, and normalization
Developed by: Student 1
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    DataPreprocessor class for cleaning and transforming data
    Handles missing values, encoding, and normalization
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor"""
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Args:
            df: pandas DataFrame
            strategy: Strategy for handling missing values
                     'mean': Fill with column mean (numerical)
                     'median': Fill with column median (numerical)
                     'mode': Fill with most frequent value
                     'drop': Drop rows with missing values
                     
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            # Drop rows with any missing values
            df_clean = df_clean.dropna()
            print(f"Dropped rows with missing values. New shape: {df_clean.shape}")
        
        else:
            # Fill missing values for each column
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if df_clean[col].dtype in ['float64', 'int64']:
                        # Numerical column
                        if strategy == 'mean':
                            fill_value = df_clean[col].mean()
                        elif strategy == 'median':
                            fill_value = df_clean[col].median()
                        else:  # mode
                            fill_value = df_clean[col].mode()[0]
                        
                        df_clean[col].fillna(fill_value, inplace=True)
                        print(f"Filled {col} missing values with {strategy}: {fill_value:.2f}")
                    
                    else:
                        # Categorical column - use mode
                        fill_value = df_clean[col].mode()[0]
                        df_clean[col].fillna(fill_value, inplace=True)
                        print(f"Filled {col} missing values with mode: {fill_value}")
        
        return df_clean
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features to numerical values
        
        Args:
            df: pandas DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Find categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Create label encoder for this column
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
            print(f"Encoded categorical feature: {col}")
            print(f"  Classes: {le.classes_.tolist()}")
        
        return df_encoded
    
    def normalize_features(self, X, method='standard'):
        """
        Normalize/standardize numerical features
        
        Args:
            X: Feature matrix (numpy array or DataFrame)
            method: Normalization method
                   'standard': StandardScaler (mean=0, std=1)
                   'minmax': MinMaxScaler (range 0-1)
                   
        Returns:
            Normalized feature matrix
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit and transform
        X_normalized = self.scaler.fit_transform(X)
        print(f"Features normalized using {method} scaling")
        
        return X_normalized
    
    def separate_features_target(self, df, target_column='target'):
        """
        Separate features and target variable
        
        Args:
            df: pandas DataFrame
            target_column: Name of target column
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.feature_names = X.columns.tolist()
        
        print(f"Separated features and target:")
        print(f"  Features: {X.shape[1]} columns")
        print(f"  Samples: {X.shape[0]} rows")
        print(f"  Target classes: {y.nunique()}")
        
        return X, y
    
    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set (default: 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"\nData split into train/test:")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, df, target_column='target', 
                           normalize=True, test_size=0.2):
        """
        Complete preprocessing pipeline
        
        Args:
            df: pandas DataFrame
            target_column: Name of target column
            normalize: Whether to normalize features
            test_size: Proportion of test set
            
        Returns:
            X_train, X_test, y_train, y_test (preprocessed and split)
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Handle missing values
        print("\n[1/5] Handling missing values...")
        df_clean = self.handle_missing_values(df, strategy='mean')
        
        # Step 2: Encode categorical features
        print("\n[2/5] Encoding categorical features...")
        df_encoded = self.encode_categorical_features(df_clean)
        
        # Step 3: Separate features and target
        print("\n[3/5] Separating features and target...")
        X, y = self.separate_features_target(df_encoded, target_column)
        
        # Step 4: Normalize features
        if normalize:
            print("\n[4/5] Normalizing features...")
            X = self.normalize_features(X, method='standard')
        else:
            print("\n[4/5] Skipping normalization")
            X = X.values  # Convert to numpy array
        
        # Step 5: Split data
        print("\n[5/5] Splitting into train/test sets...")
        X_train, X_test, y_train, y_test = self.split_train_test(
            X, y, test_size=test_size
        )
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return X_train, X_test, y_train, y_test


# Example usage
if __name__ == '__main__':
    from loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_file('sample_datasets/iris.csv')
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df)
    
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

