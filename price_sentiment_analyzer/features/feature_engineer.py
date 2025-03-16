from typing import List, Dict
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from config.config import Config

class FeatureConfig(BaseModel):
    lag_periods: List[int] = Field(default=[1, 3, 5])
    ta_indicators: Dict[str, Dict] = Field(
        default={
            'sma': {'windows': [20, 50]},
            'rsi': {'period': 14}
        }
    )

class FeatureEngineer:
    def __init__(self, config: Config):
        self.feature_registry = {}
        self.cfg = FeatureConfig(**config.settings.features)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline"""
        df = self._calculate_ta_features(df)
        df = self._create_price_features(df)
        df = self._add_lagged_features(df)
        self._register_features(df.columns)
        return df

    def _calculate_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Technical indicator features"""
        for indicator, params in self.cfg.ta_indicators.items():
            if indicator == 'sma':
                for window in params['windows']:
                    df[f'SMA_{window}'] = df['close'].rolling(window).mean()
            elif indicator == 'rsi':
                df['RSI'] = self._calculate_rsi(df['close'], params['period'])
        return df

    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-derived features"""
        df['daily_return'] = df['close'].pct_change()
        df['volatility_7d'] = df['daily_return'].rolling(7).std()
        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features"""
        for lag in self.cfg.lag_periods:
            df[f'close_lag{lag}'] = df['close'].shift(lag)
        return df.dropna()

    def _register_features(self, features: List[str]):
        """Track generated features"""
        self.feature_registry = {
            'technical': [f for f in features if f.startswith('SMA_') or f == 'RSI'],
            'price_derived': [f for f in features if 'return' in f or 'volatility' in f],
            'lagged': [f for f in features if 'lag' in f]
        }
    def save_feature_metadata(self, db: Database):
        """Store feature definitions in database"""
        for feature_type, features in self.feature_registry.items():
            db.store_features(
                feature_type=feature_type,
                features=features,
                importance=self.model.get_feature_importance()
            )

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        return 100 - (100 / (1 + (avg_gain / avg_loss)))
    
