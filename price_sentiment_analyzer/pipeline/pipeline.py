# prediction_pipeline.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..processors.data_processor import DataProcessor
from ..features.feature_engineer import FeatureEngineer
from ..models.model_factory import ModelFactory
from ..analysis.backtester import AdvancedBacktester
from ..database.data_loader import DataLoader

class PriceSentimentPipeline:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = config_path
        self.db = DataLoader(self.config.get('database.path'))
        self.processor = DataProcessor(self.db, self.config.get('data_processing'))
        self.feature_engineer = FeatureEngineer(
            lag_periods=self.config.get('feature_engineering.lag_periods', [1, 3, 5])
        )
        self.backtester = AdvancedBacktester(
            initial_capital=self.config.get('backtesting.initial_capital', 10000)
        )
        self.model = None
        self.features = None
        self.scaler = None

    def run_pipeline(self, symbol: str) -> Tuple[Dict, pd.Series]:
        """Complete training and forecasting workflow"""
        # 1. Load and clean data
        raw_data, _ = self.processor.load_and_clean_data(
            symbol,
            cleaning_strategy=self.config.get('data_processing.cleaning_strategy'),
            outlier_method=self.config.get('data_processing.outlier_method')
        )
        
        # 2. Feature engineering
        self.features = self.feature_engineer.engineer_features(
            raw_data,
            selected_features=self.config.get('feature_engineering.selected_features')
        )
        
        # 3. Create labels
        labels = self._create_labels(
            self.features,
            threshold=self.config.get('model.training.threshold', 0.05),
            horizon=self.config.get('model.training.horizon', 30)
        )
        
        # 4. Train model
        model_type = self.config.get('model.type', 'lightgbm')
        self.model = ModelFactory.create_model(
            model_type,
            params=self.config.get(f'model.params.{model_type}')
        )
        train_metrics = self._train_model(self.features, labels)
        
        # 5. Backtest strategy
        signals = self.model.predict(self.features)
        backtest_results = self.backtester.run_backtest(self.features, signals)
        
        # 6. Generate forecast
        forecast = self._generate_forecast(
            self.features,
            days=self.config.get('forecasting.days', 60)
        )
        
        return {**train_metrics, **backtest_results}, forecast

    def _create_labels(self, df: pd.DataFrame, threshold: float, horizon: int) -> pd.Series:
        """Create target labels for supervised learning"""
        future_prices = df['close'].shift(-horizon)
        future_returns = (future_prices - df['close']) / df['close']
        return pd.Series(
            np.select([future_returns > threshold, future_returns < -threshold], [1, -1], 0),
            index=df.index,
            name='label'
        ).dropna()

    def _train_model(self, features: pd.DataFrame, labels: pd.Series) -> Dict:
        """Train model with validation"""
        X = features.drop(columns=['close', 'volume']).join(labels, how='inner').dropna()
        y = X.pop('label')
        
        # Train with cross-validation
        train_metrics = self.model.train(X, y)
        
        # Feature importance tracking
        if self.config.get('feature_engineering.track_importance', True):
            self.feature_engineer.update_feature_importance(
                self.model.get_feature_importance()
            )
        
        return train_metrics

    def _generate_forecast(self, historical_data: pd.DataFrame, days: int) -> pd.Series:
        """Generate future predictions"""
        forecast_data = historical_data.copy()
        last_date = forecast_data.index[-1]
        
        # Propagate features for forecasting
        for i in range(days):
            new_date = last_date + pd.DateOffset(days=i+1)
            new_row = forecast_data.iloc[-1].copy()
            
            # Update lagged features
            for lag in self.feature_engineer.lag_periods:
                for feature in self.feature_engineer.registry.get_features_by_category('price'):
                    new_row[f'{feature}_lag{lag}'] = forecast_data[feature].iloc[-lag]
            
            new_row.name = new_date
            forecast_data = pd.concat([forecast_data, pd.DataFrame([new_row])])
        
        # Prepare forecast features
        X_forecast = forecast_data.tail(days)
        X_forecast = X_forecast[self.model.feature_names]
        
        # Generate predictions
        predictions = self.model.predict(X_forecast)
        return pd.Series(
            predictions,
            index=pd.date_range(start=last_date, periods=days+1)[1:],
            name='sentiment'
        )

    def visualize_forecast(self, forecast: pd.Series):
        """Visualize prediction results"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        forecast.plot(kind='bar', title='60-Day Sentiment Forecast')
        plt.ylabel('Sentiment Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    pipeline = PriceSentimentPipeline()
    results, forecast = pipeline.run_pipeline(symbol='BTC')
    
    print("\nTraining & Backtest Results:")
    for metric, value in results.items():
        print(f"{metric:>20}: {value:.2f}")
    
    print("\n60-Day Sentiment Forecast:")
    print(forecast)
    
    pipeline.visualize_forecast(forecast)

