import os
import pickle
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

class ModelInterface(ABC):
    """Abstract base class for all predictive models."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return performance metrics."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass
    
    @abstractmethod
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform hyperparameter optimization."""
        pass
    
    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize the model to bytes."""
        pass
    
    @classmethod
    @abstractmethod
    def deserialize(cls, data: bytes) -> 'ModelInterface':
        """Deserialize bytes into a model instance."""
        pass
class ModelEvaluator:
    """Utility class for model evaluation and tracking."""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, float]] = []
    
    def evaluate_model(self, model: ModelInterface, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on a given dataset."""
        predictions = model.predict(X)
        
        # Calculate various metrics
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y, predictions, average='weighted', zero_division=0)
        }
        
        # If the model outputs probabilities, calculate ROC AUC
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(X)
                if probas.shape[1] == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y, probas[:, 1])
                else:  # Multiclass
                    metrics['roc_auc'] = roc_auc_score(y, probas, multi_class='ovr')
            except Exception:
                pass  # Skip ROC AUC if it fails
        
        self.metrics_history.append(metrics)
        return metrics
    
    def cross_validate(self, model: ModelInterface, X: pd.DataFrame, y: pd.Series, 
                      n_splits: int = 5) -> Dict[str, float]:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics_list = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.train(X_train, y_train)
            fold_metrics = self.evaluate_model(model, X_test, y_test)
            metrics_list.append(fold_metrics)
        
        # Average the metrics across folds
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in metrics_list])
            for metric in metrics_list[0]
        }
        
        return avg_metrics

# Implementation of model classes
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class LightGBMModel(ModelInterface):
    """LightGBM model implementation."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model = LGBMClassifier(**self.params)
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return performance metrics."""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, val_idx = list(tscv.split(X))[-1]
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        preds = self.model.predict(X_val)
        metrics = {
            'accuracy': accuracy_score(y_val, preds),
            'precision': precision_score(y_val, preds, average='weighted', zero_division=0)
        }
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data."""
        if self.feature_names:
            X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for new data."""
        if self.feature_names:
            X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.feature_names:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform hyperparameter optimization."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'num_leaves': [10, 20, 31]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        X_scaled = self.scaler.fit_transform(X)
        
        grid_search = GridSearchCV(
            estimator=LGBMClassifier(),
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        self.params = grid_search.best_params_
        self.model = LGBMClassifier(**self.params)
        self.model.fit(X_scaled, y)
        
        return self.params
    
    def serialize(self) -> bytes:
        """Serialize the model to bytes."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': self.params
        }
        return pickle.dumps(model_data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'LightGBMModel':
        """Deserialize bytes into a model instance."""
        model_data = pickle.loads(data)
        
        instance = cls(params=model_data['params'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        
        return instance


class RandomForestModel(ModelInterface):
    """Random Forest model implementation."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model = RandomForestClassifier(**self.params)
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return performance metrics."""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, val_idx = list(tscv.split(X))[-1]
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_val)
        metrics = {
            'accuracy': accuracy_score(y_val, preds),
            'precision': precision_score(y_val, preds, average='weighted', zero_division=0)
        }
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data."""
        if self.feature_names:
            X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for new data."""
        if self.feature_names:
            X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.feature_names:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform hyperparameter optimization."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        X_scaled = self.scaler.fit_transform(X)
        
        grid_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(),
            param_distributions=param_grid,
            n_iter=10,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        self.params = grid_search.best_params_
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_scaled, y)
        
        return self.params
    
    def serialize(self) -> bytes:
        """Serialize the model to bytes."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': self.params
        }
        return pickle.dumps(model_data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'RandomForestModel':
        """Deserialize bytes into a model instance."""
        model_data = pickle.loads(data)
        
        instance = cls(params=model_data['params'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        
        return instance


class XGBoostModel(ModelInterface):
    """XGBoost model implementation."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model = XGBClassifier(**self.params)
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return performance metrics."""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, val_idx = list(tscv.split(X))[-1]
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        preds = self.model.predict(X_val)
        metrics = {
            'accuracy': accuracy_score(y_val, preds),
            'precision': precision_score(y_val, preds, average='weighted', zero_division=0)
        }
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data."""
        if self.feature_names:
            X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for new data."""
        if self.feature_names:
            X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.feature_names:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform hyperparameter optimization."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        X_scaled = self.scaler.fit_transform(X)
        
        grid_search = RandomizedSearchCV(
            estimator=XGBClassifier(),
            param_distributions=param_grid,
            n_iter=10,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        self.params = grid_search.best_params_
        self.model = XGBClassifier(**self.params)
        self.model.fit(X_scaled, y)
        
        return self.params
    
    def serialize(self) -> bytes:
        """Serialize the model to bytes."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': self.params
        }
        return pickle.dumps(model_data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'XGBoostModel':
        """Deserialize bytes into a model instance."""
        model_data = pickle.loads(data)
        
        instance = cls(params=model_data['params'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        
        return instance

