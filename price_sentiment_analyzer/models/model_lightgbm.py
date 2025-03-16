import lightgbm as lgb
import pandas as pd
from typing import Dict, Any
from .model_interface import BaseModel, TSModelMixin

class LightGBMModel(BaseModel, TSModelMixin):
    def __init__(self, params: Dict[str, Any] = None):
        self.model = None
        self.params = params or {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05
        }
        self.feature_importance = pd.DataFrame()

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        metrics = {}
        for X_train, X_test, y_train, y_test in self.create_validation_splits(X, y):
            train_data = lgb.Dataset(X_train, label=y_train)
            self.model = lgb.train(self.params, train_data)
            preds = self.model.predict(X_test).argmax(axis=1)
            metrics = self._update_metrics(metrics, y_test, preds)
        
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importance()
        })
        return self._average_metrics(metrics)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X).argmax(axis=1))

    def get_feature_importance(self) -> pd.DataFrame:
        return self.feature_importance

    # Implement save/load methods

    