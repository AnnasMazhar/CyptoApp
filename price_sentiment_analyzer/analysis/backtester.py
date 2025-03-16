import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class Backtester:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.metrics = {}

    def run_backtest(self, df: pd.DataFrame, signals: pd.Series) -> Dict:
        """Full backtesting pipeline"""
        returns = self._calculate_returns(df, signals)
        self._compute_performance_metrics(returns)
        self._generate_visualizations(returns)
        return self.metrics

    def _calculate_returns(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns"""
        positions = signals.shift(1).fillna(0)
        returns = df['daily_return'] * positions
        return returns - self._calculate_transaction_costs(positions)

    def _compute_performance_metrics(self, returns: pd.Series):
        """Calculate comprehensive metrics"""
        cumulative = (1 + returns).cumprod()
        self.metrics = {
            'total_return': cumulative.iloc[-1] - 1,
            'annualized_return': returns.mean() * 252,
            'max_drawdown': (cumulative / cumulative.cummax() - 1).min(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252)
        }

    def _generate_visualizations(self, returns: pd.Series):
        """Create performance plots"""
        plt.figure(figsize=(12, 6))
        (1 + returns).cumprod().plot(title='Cumulative Returns')
        plt.savefig('cumulative_returns.png')
        plt.close()


        