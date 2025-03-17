import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Callable, Optional, Tuple, Any
from ..database.data_loader import DataLoader  # Updated import

class DataProcessor:
    """Refactored DataProcessor using DataLoader for data access"""
    
    def __init__(self, data_loader: DataLoader, config: Optional[Dict] = None):
        self.data_loader = data_loader  # Changed from Database to DataLoader
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.cleaning_strategies = {
            'ffill_bfill': self._handle_missing_values_ffill_bfill,
            'interpolate': self._handle_missing_values_interpolate,
            'drop': self._handle_missing_values_drop
        }
    
    def _handle_missing_values_ffill_bfill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with forward fill and backward fill."""
        return df.ffill().bfill()
    
    def _handle_missing_values_interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with time-based interpolation."""
        return df.interpolate(method='time')
    
    def _handle_missing_values_drop(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by dropping them."""
        return df.dropna()
    
    def _get_cleaning_strategy(self, strategy_name: str) -> Callable:
        """Get the cleaning strategy function by name."""
        if strategy_name not in self.cleaning_strategies:
            self.logger.warning(f"Unknown cleaning strategy: {strategy_name}, using ffill_bfill")
            return self.cleaning_strategies['ffill_bfill']
        return self.cleaning_strategies[strategy_name]
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = 'ffill_bfill') -> pd.DataFrame:
        """Handle missing values using the specified strategy."""
        df = df.asfreq('D')  # Ensure daily frequency
        strategy_func = self._get_cleaning_strategy(strategy)
        return strategy_func(df)
    
    def _detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Identify and cap outliers using the specified method."""
        if columns is None:
            columns = ['close', 'volume']
        
        if method == 'iqr':
            for col in columns:
                if col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    upper_bound = q3 + 1.5 * iqr
                    lower_bound = q1 - 1.5 * iqr
                    df[col] = np.where(df[col] > upper_bound, upper_bound,
                                    np.where(df[col] < lower_bound, lower_bound, df[col]))
        elif method == 'zscore':
            for col in columns:
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    df[col] = np.where(abs(df[col] - mean) > 3 * std, 
                                    np.sign(df[col] - mean) * 3 * std + mean, 
                                    df[col])
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return issues."""
        issues = {}
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            issues['missing_values'] = missing[missing > 0].to_dict()
        
        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            issues['duplicate_rows'] = duplicates
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            issues['constant_columns'] = constant_cols
        
        # Check for extreme values
        for col in ['close', 'volume']:
            if col in df.columns:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                extreme_count = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
                if extreme_count > 0:
                    issues.setdefault('extreme_values', {})[col] = extreme_count
        
        return issues
    
    def load_and_clean_data(self, symbol: str, cleaning_strategy: str = 'ffill_bfill', 
                           outlier_method: str = 'iqr') -> Tuple[pd.DataFrame, Dict]:
        """Load and preprocess data using DataLoader"""
        # Get data through DataLoader instead of direct DB access
        df = self.data_loader.get_historical_data(symbol)
        issues = {}
        
        if df.empty:
            self.logger.warning(f"No data found for symbol: {symbol}")
            return df, {'error': 'No data found'}
        
        # Validate data before cleaning
        pre_clean_issues = self._validate_data(df)
        if pre_clean_issues:
            issues['pre_cleaning'] = pre_clean_issues
            self.logger.info(f"Data quality issues found before cleaning: {pre_clean_issues}")
        
        # Apply cleaning steps
        df = self._handle_missing_values(df, strategy=cleaning_strategy)
        df = self._detect_outliers(df, method=outlier_method)
        
        # Validate data after cleaning
        post_clean_issues = self._validate_data(df)
        if post_clean_issues:
            issues['post_cleaning'] = post_clean_issues
            self.logger.warning(f"Data quality issues remain after cleaning: {post_clean_issues}")
        
        return df, issues
    
