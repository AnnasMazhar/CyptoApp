import unittest
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_loader import DataLoader
from database import Database
import os

class TestDataLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Use in-memory database for testing"""
        cls.test_db = ":memory:"
        cls.dl = DataLoader(cls.test_db)
        
        # Initialize schema
        with Database(cls.test_db) as db:
            db._apply_migrations()

    def setUp(self):
        """Fresh connection for each test"""
        self.conn = sqlite3.connect(self.test_db)
        self.dl = DataLoader(self.test_db)

    def tearDown(self):
        self.conn.close()
        os.remove(self.test_db) if os.path.exists(self.test_db) else None

    def test_historical_data_roundtrip(self):
        """Test saving and retrieving historical data"""
        test_symbol = "BTC"
        test_data = pd.DataFrame({
            'date': [datetime.now().date().isoformat()],
            'open': [50000.0],
            'high': [51000.0],
            'low': [49000.0],
            'close': [50500.0],
            'volume': [1000000.0]
        })
        
        # Test save
        success = self.dl.save_historical_data(test_symbol, test_data)
        self.assertTrue(success)
        
        # Test retrieve
        df = self.dl.get_historical_data(test_symbol)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['close'], 50500.0)

    def test_duplicate_handling(self):
        """Test duplicate data insertion prevention"""
        test_symbol = "ETH"
        test_data = pd.DataFrame({
            'date': ['2023-01-01'],
            'open': [2000.0],
            'high': [2100.0],
            'low': [1900.0],
            'close': [2050.0],
            'volume': [500000.0]
        })
        
        # First insert
        self.dl.save_historical_data(test_symbol, test_data)
        # Duplicate insert
        self.dl.save_historical_data(test_symbol, test_data)
        
        df = self.dl.get_historical_data(test_symbol)
        self.assertEqual(len(df), 1)

    def test_model_versioning(self):
        """Test model version storage and retrieval"""
        test_model = {"dummy": "model"}
        metrics = {'mae': 0.1, 'r2_score': 0.95}
        
        # Save model
        version_id = self.dl.save_model_version(
            test_model, metrics, 
            model_type="regression",
            feature_names=["feature1", "feature2"],
            model_params={"param1": 100}
        )
        self.assertIsNotNone(version_id)
        
        # Retrieve model
        vid, model, features, params = self.dl.get_latest_model()
        self.assertEqual(vid, version_id)
        self.assertEqual(features, ["feature1", "feature2"])

    def test_predictions_workflow(self):
        """Test prediction storage and retrieval"""
        # Save model first
        version_id = self.dl.save_model_version(
            {"model": "test"}, {}, "test", [], {}
        )
        
        # Save prediction
        success = self.dl.save_prediction(
            symbol="BTC",
            prediction_date="2023-01-01",
            forecast_date="2023-01-02",
            predicted_price=50000.0,
            model_version=version_id
        )
        self.assertTrue(success)
        
        # Retrieve predictions
        df = self.dl.get_predictions("BTC", "2023-01-01", "2023-01-02")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['predicted_price'], 50000.0)

    def test_news_sentiment_processing(self):
        """Test news data storage and sentiment calculation"""
        test_articles = [{
            'symbol': 'BTC',
            'date': '2023-01-01T12:00:00',
            'title': 'Bitcoin soars!',
            'content': 'Positive news about cryptocurrency',
            'sentiment_positive': 0.9,
            'sentiment_negative': 0.1,
            'sentiment_neutral': 0.0,
            'source': 'TestNews'
        }]
        
        # Save news
        success = self.dl.save_news_data(test_articles)
        self.assertTrue(success)
        
        # Get sentiments
        df = self.dl.get_latest_sentiments('BTC')
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]['sentiment_score'], 0.9*0.6 - 0.1*0.3)

    def test_schema_migrations(self):
        """Verify all migrations are applied successfully"""
        db = Database(self.test_db)
        with db._get_connection() as conn:
            version = conn.execute('SELECT version FROM __version__').fetchone()[0]
            self.assertEqual(version, 7)  # Update to latest version
            
            # Check existence of critical tables
            tables = [row[0] for row in 
                     conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for t in ['historical_prices', 'model_versions', 'model_predictions']:
                self.assertIn(t, tables)

if __name__ == '__main__':
    unittest.main()