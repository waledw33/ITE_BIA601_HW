"""
Data Loading Module - File Reader
Loads data from various file formats (CSV, Excel)
Part of: Advanced Feature Selection Research Project
Research Team: AI & Machine Learning Laboratory
"""

import pandas as pd
import numpy as np
from pathlib import Path


class FileReader:
    """
    FileReader Class - Multi-format Data Loader
    
    Supports CSV and Excel file formats with automatic format detection.
    Provides dataset information and preview capabilities.
    """
    
    def __init__(self):
        """Initialize file reader with supported formats"""
        self.supported_formats = ['.csv', '.xlsx', '.xls']
    
    def load_csv(self, file_path, encoding='utf-8'):
        """
        Load CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file
        encoding : str, default='utf-8'
            File encoding
            
        Returns:
        --------
        DataFrame
            Loaded data
        """
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            return data
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
    
    def load_excel(self, file_path, sheet_name=0):
        """
        Load Excel file
        
        Parameters:
        -----------
        file_path : str
            Path to Excel file
        sheet_name : int or str, default=0
            Sheet number or name to load
            
        Returns:
        --------
        DataFrame
            Loaded data
        """
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            return data
        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")
    
    def auto_load(self, file_path):
        """
        Auto-detect file format and load
        
        Automatically detects file format based on extension
        and loads the data accordingly.
        
        Parameters:
        -----------
        file_path : str
            Path to data file
            
        Returns:
        --------
        DataFrame
            Loaded data
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        extension = path.suffix.lower()
        
        # Check if format is supported
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}. Supported formats: {self.supported_formats}")
        
        # Load based on extension
        if extension == '.csv':
            return self.load_csv(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self.load_excel(file_path)
    
    def get_dataset_info(self, df):
        """
        Get comprehensive dataset information
        
        Parameters:
        -----------
        df : DataFrame
            pandas DataFrame to analyze
            
        Returns:
        --------
        dict
            Dictionary containing dataset statistics
        """
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'features': df.columns.tolist(),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        return info
    
    def preview_data(self, df, n_rows=5):
        """
        Display dataset preview
        
        Parameters:
        -----------
        df : DataFrame
            pandas DataFrame to preview
        n_rows : int, default=5
            Number of rows to display
        """
        print("\n" + "="*70)
        print("DATASET PREVIEW")
        print("="*70)
        print(f"\nFirst {n_rows} rows:")
        print(df.head(n_rows))
        
        print(f"\n\nDataset Information:")
        print(f"Total Rows: {len(df)}")
        print(f"Total Columns: {len(df.columns)}")
        print(f"\nColumn Names: {df.columns.tolist()}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        print("="*70)


# Module Test & Demonstration
if __name__ == '__main__':
    print("="*70)
    print("FILE READER MODULE DEMONSTRATION")
    print("="*70)
    
    # Test the FileReader
    loader = FileReader()
    
    # Load sample dataset
    try:
        print("\nLoading sample dataset...")
        df = loader.auto_load('sample_datasets/iris.csv')
        
        # Preview data
        loader.preview_data(df)
        
        # Get dataset info
        info = loader.get_dataset_info(df)
        print(f"\nDataset has {info['rows']} rows and {info['columns']} columns")
        print(f"Memory usage: {info['memory_usage']} bytes")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
