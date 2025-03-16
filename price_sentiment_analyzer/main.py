from database.database import Database
from price_sentiment_analyzer.analysis.backtester import Backtester
from price_sentiment_analyzer.config.config import Config
from price_sentiment_analyzer.models.model_interface import ModelInterface, ModelEvaluator
from price_sentiment_analyzer.models.model_factory import ModelFactory
from price_sentiment_analyzer.pipeline.pipeline import PriceSentimentPipeline
from processors.data_processor import DataProcessor



if __name__ ==  'main':
    db = Database('crypto_data.db')
    backtest = Backtester()
    config = Config()



