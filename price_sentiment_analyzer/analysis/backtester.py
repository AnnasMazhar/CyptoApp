# backtester.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import Dict, Tuple, Optional
from enum import Enum
import scipy.stats as stats

class StrategyType(Enum):
    SIMPLE = "simple"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    MOMENTUM = "momentum"
    PAIRS = "pairs"

class RiskManager:
    def __init__(self, max_leverage: float = 1.0, stop_loss: float = 0.1,
                 take_profit: float = 0.15, volatility_window: int = 21):
        self.max_leverage = max_leverage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.volatility_window = volatility_window
        
    def calculate_position_size(self, prices: pd.Series, volatility: pd.Series) -> pd.Series:
        """Volatility-adjusted position sizing"""
        return np.clip(0.1 / volatility, 0, self.max_leverage)

    def apply_risk_management(self, positions: pd.Series, returns: pd.Series) -> pd.Series:
        """Apply stop loss and take profit rules"""
        cumulative_returns = (1 + returns).cumprod()
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        
        # Apply stop loss
        positions[drawdown > self.stop_loss] = 0
        
        # Apply take profit
        positions[cumulative_returns > 1 + self.take_profit] = 0
        
        return positions

class AdvancedBacktester:
    def __init__(self, initial_capital: float = 100000, 
                 transaction_cost: float = 0.0005,
                 strategy_type: StrategyType = StrategyType.SIMPLE):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.strategy_type = strategy_type
        self.risk_manager = RiskManager()
        self.metrics: Dict[str, float] = {}
        self.trade_history = pd.DataFrame()

    def run_backtest(self, prices: pd.DataFrame, signals: pd.Series,
                    benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        """Enhanced backtesting pipeline"""
        returns = self._calculate_returns(prices, signals)
        self._compute_advanced_metrics(returns, prices, benchmark)
        self._generate_detailed_visualizations(returns, prices, benchmark)
        return self.metrics

    def _calculate_returns(self, prices: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns with multiple strategy options"""
        if self.strategy_type == StrategyType.SIMPLE:
            positions = self._simple_strategy(signals)
        elif self.strategy_type == StrategyType.VOLATILITY_ADJUSTED:
            positions = self._volatility_adjusted_strategy(prices, signals)
        elif self.strategy_type == StrategyType.MOMENTUM:
            positions = self._momentum_strategy(prices, signals)
            
        positions = self.risk_manager.apply_risk_management(positions, prices['returns'])
        trades = positions.diff().fillna(0).abs()
        
        strategy_returns = positions.shift(1) * prices['returns']
        strategy_returns -= trades * self.transaction_cost
        return strategy_returns

    def _simple_strategy(self, signals: pd.Series) -> pd.Series:
        return signals.replace({-1: -1, 0: 0})

    def _volatility_adjusted_strategy(self, prices: pd.DataFrame, signals: pd.Series) -> pd.Series:
        volatility = prices['returns'].rolling(self.risk_manager.volatility_window).std()
        position_size = self.risk_manager.calculate_position_size(prices['close'], volatility)
        return signals * position_size

    def _momentum_strategy(self, prices: pd.DataFrame, signals: pd.Series) -> pd.Series:
        momentum = prices['close'].pct_change(21)
        return signals * np.where(momentum > 0, 1, -1)

    def _compute_advanced_metrics(self, returns: pd.Series, 
                                prices: pd.DataFrame,
                                benchmark: pd.Series) -> None:
        """Calculate comprehensive performance metrics"""
        cumulative = (1 + returns).cumprod()
        drawdown = 1 - cumulative / cumulative.cummax()
        
        # Basic metrics
        self.metrics = {
            'total_return': cumulative.iloc[-1] - 1,
            'annualized_return': returns.mean() * 252,
            'max_drawdown': drawdown.min(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': (returns.mean() * 252) / abs(drawdown.min()),
            'win_rate': (returns > 0).mean(),
            'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()),
            'var_95': self._calculate_var(returns),
            'beta': self._calculate_beta(returns, benchmark),
            'alpha': self._calculate_alpha(returns, benchmark),
        }
        
        # Trade analysis
        self._analyze_trades(returns)

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0
        return returns.mean() / downside_returns.std() * np.sqrt(252)

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        return np.percentile(returns, 100 * (1 - confidence))

    def _calculate_beta(self, strategy_returns: pd.Series, 
                       benchmark_returns: pd.Series) -> float:
        if benchmark_returns is None:
            return 0
        cov = np.cov(strategy_returns, benchmark_returns)
        return cov[0, 1] / cov[1, 1]

    def _calculate_alpha(self, strategy_returns: pd.Series,
                        benchmark_returns: pd.Series) -> float:
        if benchmark_returns is None:
            return 0
        beta = self._calculate_beta(strategy_returns, benchmark_returns)
        return (strategy_returns.mean() - beta * benchmark_returns.mean()) * 252

    def _analyze_trades(self, returns: pd.Series) -> None:
        """Analyze individual trades"""
        positions = returns != 0
        trade_starts = positions.diff().fillna(False) > 0
        trade_ends = positions.diff().fillna(False) < 0
        
        self.trade_history = pd.DataFrame({
            'entry_date': returns.index[trade_starts],
            'exit_date': returns.index[trade_ends],
            'return': returns[trade_ends].values
        })

    def _generate_detailed_visualizations(self, returns: pd.Series,
                                         prices: pd.DataFrame,
                                         benchmark: pd.Series) -> None:
        """Create enhanced performance visualizations"""
        plt.figure(figsize=(15, 10))
        
        # Cumulative Returns
        plt.subplot(3, 2, 1)
        (1 + returns).cumprod().plot(label='Strategy')
        if benchmark is not None:
            (1 + benchmark).cumprod().plot(label='Benchmark')
        plt.title('Cumulative Returns')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        
        # Drawdown
        plt.subplot(3, 2, 2)
        drawdown = 1 - (1 + returns).cumprod() / (1 + returns).cumprod().cummax()
        drawdown.plot()
        plt.title('Drawdown')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        
        # Monthly Returns Heatmap
        plt.subplot(3, 2, 3)
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.unstack().T
        plt.imshow(monthly_returns, cmap='RdYlGn', aspect='auto')
        plt.colorbar(label='Return')
        plt.title('Monthly Returns Heatmap')
        
        # Rolling Sharpe Ratio
        plt.subplot(3, 2, 4)
        rolling_sharpe = returns.rolling(63).mean() / returns.rolling(63).std() * np.sqrt(252)
        rolling_sharpe.plot()
        plt.title('6-Month Rolling Sharpe Ratio')
        
        # Trade Histogram
        plt.subplot(3, 2, 5)
        self.trade_history['return'].hist(bins=50)
        plt.title('Trade Returns Distribution')
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
        
        # Correlation with Benchmark
        if benchmark is not None:
            plt.subplot(3, 2, 6)
            plt.scatter(benchmark, returns, alpha=0.3)
            plt.xlabel('Benchmark Returns')
            plt.ylabel('Strategy Returns')
            plt.title('Daily Returns Correlation')
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.close()

    def generate_report(self) -> str:
        """Generate formatted PDF/HTML report"""
        # Implementation would use library like weasyprint or pdfkit
        return "Backtest report generated"
    
    