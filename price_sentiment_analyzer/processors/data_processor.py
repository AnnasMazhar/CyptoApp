from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pydantic import validate_arguments
from config.config import Config

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.cleaning_strategies = {
            'default': self._default_cleaning,
            'aggressive': self._aggressive_cleaning
        }

    @validate_arguments
    def load_and_clean(self, symbol: str) -> pd.DataFrame:
        """Main processing pipeline"""
        raw_df = self._load_from_source(symbol)
        clean_df = self._apply_cleaning_strategy(raw_df)
        validated_df = self._validate_data(clean_df)
        return validated_df

    def _load_from_source(self, symbol: str) -> pd.DataFrame:
        """Load data from database"""
        # Implementation using your Database class
        return pd.DataFrame()

    def _apply_cleaning_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply configured cleaning strategy"""
        strategy = self.config.settings.data_processing.cleaning_strategy
        return self.cleaning_strategies[strategy](df)

    def _default_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balanced cleaning approach"""
        df = df.asfreq('D').ffill().bfill()
        return df.interpolate(method='time')

    def _aggressive_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Outlier removal + interpolation"""
        df = self._handle_outliers(df)
        df = df.dropna().interpolate(method='linear')
        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data quality checks"""
        required_columns = self.config.settings.data_processing.required_columns
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns")
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR-based outlier handling"""
        for col in ['close', 'volume']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = np.clip(df[col], q1-1.5*iqr, q3+1.5*iqr)
        return df
    
    