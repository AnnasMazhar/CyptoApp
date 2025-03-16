from typing import Dict, Optional
import pandas as pd
from config.config import Config
from processors.data_processor import DataProcessor
from features.feature_engineer import FeatureEngineer
from models.model_factory import ModelFactory
from models.lightgbm_model import LightGBMModel
from analysis.backtester import Backtester
from utils.logging_utils import log_execution


class PriceSentimentPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model = ModelFactory.create(config.settings.model.default)
        self.backtester = Backtester()

    def run(self, symbol: str) -> Dict:
        """Complete pipeline execution"""
        # Data preparation
        clean_data = self.data_processor.load_and_clean(symbol)
        features = self.feature_engineer.engineer_features(clean_data)
        
        # Model training
        labels = self._create_labels(features)
        self.model = LightGBMModel(self.config.settings.models.lightgbm)
        metrics = self.model.train(features, labels)
        
        # Backtesting
        predictions = self.model.predict(features)
        backtest_results = self.backtester.run_backtest(clean_data, predictions)
        
        return {
            'model_metrics': metrics,
            'backtest_results': backtest_results
        }

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create target labels"""
        horizon = self.config.settings.model.training.horizon
        future_returns = df['close'].pct_change(horizon).shift(-horizon)
        return pd.cut(future_returns, 
                     bins=[-np.inf, -0.05, 0.05, np.inf],
                     labels=[-1, 0, 1])
    
