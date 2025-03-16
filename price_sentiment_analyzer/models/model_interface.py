from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

class BaseModel(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train model with time-series validation"""
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions"""
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance"""
    
    @abstractmethod
    def save(self, path: str) -> bool:
        """Persist model to storage"""
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from storage"""

class TSModelMixin:
    """Time-series specific functionality"""
    def create_validation_splits(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple]:
        splits = []
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, test_idx in tscv.split(X):
            splits.append((
                X.iloc[train_idx], X.iloc[test_idx],
                y.iloc[train_idx], y.iloc[test_idx]
            ))
        return splits
    
    