# feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Set, Tuple
from functools import wraps
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureRegistry:
    """A registry to track available features and their metadata."""
    
    def __init__(self):
        self.features: Dict[str, Dict] = {}
        self.categories: Set[str] = set()
        
    def register(self, category: str, description: str = "", dependencies: List[str] = None):
        """
        Decorator to register a feature generation function.
        
        Args:
            category: Category of the feature (e.g., 'momentum', 'volatility')
            description: Description of what the feature represents
            dependencies: List of column names required for this feature
        """
        dependencies = dependencies or []
        
        def decorator(func: Callable):
            feature_name = func.__name__.replace('_calculate_', '')
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            self.features[feature_name] = {
                'function': func,
                'category': category,
                'description': description,
                'dependencies': dependencies
            }
            self.categories.add(category)
            
            return wrapper
        
        return decorator
    
    def get_feature_info(self) -> Dict:
        """Get information about all registered features."""
        return {
            name: {k: v for k, v in info.items() if k != 'function'}
            for name, info in self.features.items()
        }
    
    def get_features_by_category(self, category: str) -> List[str]:
        """Get all feature names belonging to a specific category."""
        return [name for name, info in self.features.items() 
                if info['category'] == category]
    
    def get_all_categories(self) -> List[str]:
        """Get all feature categories."""
        return list(self.categories)


class FeatureEngineer:
    """Enhanced feature engineering module with feature registry."""
    
    registry = FeatureRegistry()
    
    def __init__(self, lag_periods: List[int] = None):
        self.lag_periods = lag_periods or [1, 3, 5]
        self.feature_importance: Dict[str, float] = {}
        
    def engineer_features(self, df: pd.DataFrame, 
                         selected_features: List[str] = None) -> pd.DataFrame:
        """
        Full feature engineering pipeline with optional feature selection.
        
        Args:
            df: DataFrame with OHLCV data
            selected_features: List of specific features to generate, or None for all
        
        Returns:
            DataFrame with engineered features
        """
        # Start with price features
        df = self.create_price_features(df)
        
        # Generate technical indicators
        if selected_features:
            # Filter to only requested features
            for feature in selected_features:
                if feature in self.registry.features:
                    func = self.registry.features[feature]['function']
                    result = func(self, df)
                    if isinstance(result, pd.DataFrame):
                        df = pd.concat([df, result], axis=1)
                    elif isinstance(result, pd.Series):
                        df[feature] = result
        else:
            # Generate all registered features
            df = self.calculate_all_ta_indicators(df)
        
        # Add lagged features
        df = self.add_lagged_features(df)
        
        return df.dropna()
    
    def calculate_all_ta_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all registered technical indicators."""
        for feature_name, feature_info in self.registry.features.items():
            try:
                func = feature_info['function']
                result = func(self, df)
                
                if isinstance(result, pd.DataFrame):
                    df = pd.concat([df, result], axis=1)
                elif isinstance(result, pd.Series):
                    df[feature_name] = result
            except Exception as e:
                logger.warning(f"Failed to calculate {feature_name}: {str(e)}")
        
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based metrics"""
        df['daily_return'] = df['close'].pct_change()
        df['volatility_7d'] = df['daily_return'].rolling(7).std()
        df['volatility_30d'] = df['daily_return'].rolling(30).std()
        return df
    
    def add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged versions of features."""
        features_to_lag = [col for col in df.columns if col not in ['close', 'volume']]
        for lag in self.lag_periods:
            lagged = df[features_to_lag].shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in lagged.columns]
            df = pd.concat([df, lagged], axis=1)
        return df
    
    def update_feature_importance(self, importances: Dict[str, float]):
        """Update feature importance scores from model training."""
        self.feature_importance = importances
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the top N most important features."""
        return sorted(self.feature_importance.items(), 
                     key=lambda x: x[1], reverse=True)[:n]
    
    @registry.register(category='momentum', 
                     description='Relative Strength Index', 
                     dependencies=['close'])
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
        return 100 - (100 / (1 + rs))
    
    @registry.register(category='momentum', 
                     description='Moving Average Convergence Divergence', 
                     dependencies=['close'])
    def _calculate_macd(self, df: pd.DataFrame, 
                       fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> pd.DataFrame:
        """Calculate MACD and signal line."""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return pd.DataFrame({
            'MACD': macd,
            'MACD_SIGNAL': signal_line
        })
    
    @registry.register(category='volatility', 
                     description='Bollinger Bands', 
                     dependencies=['close'])
    def _calculate_bollinger_bands(self, df: pd.DataFrame, 
                                  window: int = 20, 
                                  num_std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        return pd.DataFrame({
            'BB_UPPER': sma + (std * num_std),
            'BB_MIDDLE': sma,
            'BB_LOWER': sma - (std * num_std)
        })
    
    @registry.register(category='volatility', 
                     description='Average True Range', 
                     dependencies=['high', 'low', 'close'])
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr = pd.DataFrame({
            'HL': df['high'] - df['low'],
            'HC': (df['high'] - df['close'].shift()).abs(),
            'LC': (df['low'] - df['close'].shift()).abs()
        }).max(axis=1)
        return tr.rolling(window).mean()
    
    @registry.register(category='volume', 
                     description='On-Balance Volume', 
                     dependencies=['close', 'volume'])
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        return df['volume'].mask(df['close'] < df['close'].shift(), -df['volume']).cumsum()
    
    @registry.register(category='volume', 
                     description='Accumulation/Distribution Line', 
                     dependencies=['high', 'low', 'close', 'volume'])
    def _calculate_adl(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 1e-9)
        mfv = mfm * df['volume']
        return mfv.cumsum()
    
    @registry.register(category='trend', 
                     description='Simple Moving Average - 20 periods', 
                     dependencies=['close'])
    def _calculate_sma_20(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 20-period Simple Moving Average."""
        return df['close'].rolling(20).mean()
    
    @registry.register(category='trend', 
                     description='Simple Moving Average - 50 periods', 
                     dependencies=['close'])
    def _calculate_sma_50(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 50-period Simple Moving Average."""
        return df['close'].rolling(50).mean()
    
    @registry.register(category='trend', 
                     description='Exponential Moving Average - 50 periods', 
                     dependencies=['close'])
    def _calculate_ema_50(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 50-period Exponential Moving Average."""
        return df['close'].ewm(span=50).mean()
    
    @registry.register(category='trend', 
                     description='Weighted Moving Average - 14 periods', 
                     dependencies=['close'])
    def _calculate_wma_14(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 14-period Weighted Moving Average."""
        return df['close'].rolling(14).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum()
        )
    
    @registry.register(category='momentum', 
                     description='Stochastic Oscillator', 
                     dependencies=['high', 'low', 'close'])
    def _calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, 1e-9)
        d = k.rolling(3).mean()
        
        return pd.DataFrame({
            'STOCH_K': k,
            'STOCH_D': d
        })
