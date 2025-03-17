import sqlite3
import pandas as pd
import joblib
import time
import json
import logging
import yaml
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

class DataLoader:
    """Handles all database query operations with logging and connection management"""
    def __init__(self, config_path: str = 'config.yaml'):
        self.logger = logging.getLogger(f"{__name__}.DataLoader")
        try:
            config = self._load_config(config_path)
            self.db_path = config['database']['path']
            self.logger.info(f"Initialized DataLoader for database: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise
    @contextmanager
    def _get_connection(self):
        """Connection context manager with error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    @contextmanager
    def _transaction(self):
        start_time = time.time()
        with self._get_connection() as conn:
            try:
                yield conn
                conn.commit()
                self.logger.info(f"Transaction committed in {time.time() - start_time:.2f}s")
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Rollback after {time.time() - start_time:.2f}s: {e}")
                raise

    @contextmanager
    def _transaction(self):
        """Transaction context manager with rollback handling"""
        with self._get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Transaction rolled back: {str(e)}")
                raise

    # Historical Price Data Methods
    def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Safer historical data retrieval"""
        self.logger.info(f"Fetching historical data for {symbol}")
        try:
            with self._get_connection() as conn:
                # Verify table exists first
                table_exists = conn.execute('''
                    SELECT count(*) FROM sqlite_master 
                    WHERE type='table' AND name='historical_prices'
                ''').fetchone()[0]
                
                if not table_exists:
                    self.logger.error("historical_prices table does not exist")
                    return pd.DataFrame()
                
                # Parameterized query with explicit column names
                query = '''
                    SELECT 
                        date, 
                        open, 
                        high, 
                        low, 
                        close, 
                        volume 
                    FROM historical_prices 
                    WHERE symbol = :symbol
                    ORDER BY date
                '''
                df = pd.read_sql_query(query, conn, params={'symbol': symbol})
                df['date'] = pd.to_datetime(df['date'])
                return df.set_index('date')
            
        except sqlite3.Error as e:
            self.logger.error(f"SQL Error: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return pd.DataFrame()

    def save_historical_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Save historical price data to database"""
        self.logger.info(f"Saving historical data for {symbol}")
        try:
            prepared_data = self._prepare_historical_data(symbol, data)
            with self._transaction() as conn:
                self._upsert_historical_data(conn, prepared_data)
            self.logger.info(f"Saved {len(prepared_data)} records for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Save historical data failed: {str(e)}")
            return False

    def _prepare_historical_data(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare historical data for insertion"""
        data = data.rename(columns=str.lower)
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        data = data[required_cols].copy()
        data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        data['symbol'] = symbol
        return data.dropna(subset=['date', 'symbol']).drop_duplicates()

    def _upsert_historical_data(self, conn, data: pd.DataFrame):
        """Perform upsert operation for historical data"""
        data.to_sql('temp_prices', conn, if_exists='replace', index=False)
        conn.execute('''
            INSERT OR IGNORE INTO historical_prices
            SELECT date, open, high, low, close, volume, symbol 
            FROM temp_prices
        ''')
        conn.execute('DROP TABLE IF EXISTS temp_prices')

    # Model Management Methods
    def save_model_version(self, model: Any, metrics: Dict[str, float], 
                         model_type: str, feature_names: List[str], 
                         model_params: Dict[str, Any]) -> Optional[int]:
        """Save ML model version with metadata"""
        self.logger.info(f"Saving new {model_type} model version")
        try:
            model_blob = joblib.dumps(model)
            with self._transaction() as conn:
                cursor = conn.execute('''
                    INSERT INTO model_versions 
                    (mae, r2_score, model_type, model_data, feature_names, model_params)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.get('mae'),
                    metrics.get('r2_score'),
                    model_type,
                    model_blob,
                    json.dumps(feature_names),
                    json.dumps(model_params)
                ))
                version_id = cursor.lastrowid
                self.logger.info(f"Saved model version {version_id}")
                return version_id
        except Exception as e:
            self.logger.error(f"Model save failed: {str(e)}")
            return None

    def get_latest_model(self, model_type: Optional[str] = None
                        ) -> Tuple[Optional[int], Optional[Any], 
                                 Optional[List[str]], Optional[Dict[str, Any]]]:
        """Retrieve latest model version with metadata"""
        try:
            with self._get_connection() as conn:
                query = '''
                    SELECT version_id, model_data, feature_names, model_params
                    FROM model_versions
                    WHERE 1=1
                '''
                params = []
                if model_type:
                    query += ' AND model_type = ?'
                    params.append(model_type)
                query += ' ORDER BY version_id DESC LIMIT 1'

                cursor = conn.execute(query, params)
                row = cursor.fetchone()

                if row:
                    model = joblib.loads(row['model_data'])
                    feature_names = json.loads(row['feature_names'])
                    model_params = json.loads(row['model_params'])
                    self.logger.info(f"Loaded model version {row['version_id']}")
                    return (row['version_id'], model, feature_names, model_params)
                return (None, None, None, None)
        except Exception as e:
            self.logger.error(f"Model load failed: {str(e)}")
            return (None, None, None, None)

    # Prediction Management Methods
    def get_predictions(self, symbol: str, start_date: str, 
                       end_date: str) -> pd.DataFrame:
        """Retrieve predictions for analysis"""
        try:
            with self._get_connection() as conn:
                query = '''
                    SELECT p.*, v.model_type
                    FROM model_predictions p
                    JOIN model_versions v ON p.model_version = v.version_id
                    WHERE symbol = ?
                    AND forecast_date BETWEEN ? AND ?
                    ORDER BY forecast_date
                '''
                params = (symbol, start_date, end_date)
                df = pd.read_sql_query(query, conn, params=params, 
                                     parse_dates=['prediction_date', 'forecast_date'])
                self.logger.info(f"Retrieved {len(df)} predictions for {symbol}")
                return df
        except Exception as e:
            self.logger.error(f"Prediction retrieval failed: {str(e)}")
            return pd.DataFrame()

    def save_prediction(self, symbol: str, prediction_date: str,
                       forecast_date: str, predicted_price: float,
                       model_version: int, actual_price: Optional[float] = None) -> bool:
        """Store prediction with optional actual price"""
        try:
            with self._transaction() as conn:
                conn.execute('''
                    INSERT INTO model_predictions
                    (symbol, prediction_date, forecast_date,
                    predicted_price, actual_price, model_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (symbol, prediction_date, forecast_date,
                     predicted_price, actual_price, model_version))
                self.logger.info(f"Saved prediction for {symbol} on {prediction_date}")
                return True
        except Exception as e:
            self.logger.error(f"Prediction save failed: {str(e)}")
            return False

    # News Data Methods
    def save_news_data(self, articles: List[Dict]) -> bool:
        """Batch save news articles"""
        try:
            with self._transaction() as conn:
                self._insert_news_data(conn, articles)
            return True
        except Exception as e:
            self.logger.error(f"News data save failed: {str(e)}")
            return False

    def _insert_news_data(self, conn, articles: List[Dict]):
        """Internal method for news data insertion"""
        insert_query = '''
            INSERT OR IGNORE INTO news_data (
                symbol, published_at, title, content,
                sentiment_positive, sentiment_negative, sentiment_neutral,
                source_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        records = [self._prepare_news_record(article) for article in articles]
        conn.executemany(insert_query, records)

    def _prepare_news_record(self, article: Dict) -> Tuple:
        """Prepare individual news article record"""
        return (
            article['symbol'],
            pd.to_datetime(article['published_at']).isoformat(),
            article['title'][:255],
            article.get('content', '')[:5000],
            article.get('sentiment_positive', 0.0),
            article.get('sentiment_negative', 0.0),
            article.get('sentiment_neutral', 0.0),
            article.get('source', 'Unknown')[:50]
        )

    # Sentiment Analysis Methods
    def get_latest_sentiments(self, symbol: str, 
                             limit: int = 10) -> pd.DataFrame:
        """Retrieve processed sentiment data"""
        try:
            with self._get_connection() as conn:
                query = '''
                    SELECT published_at AS date, title, content AS summary,
                        (COALESCE(sentiment_positive,0)*0.6
                        - COALESCE(sentiment_negative,0)*0.3
                        + COALESCE(sentiment_neutral,0)*0.1) AS sentiment_score
                    FROM news_data
                    WHERE symbol = ?
                    ORDER BY published_at DESC
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, conn, 
                                     params=(symbol, limit), 
                                     parse_dates=['date'])
                self.logger.info(f"Retrieved {len(df)} sentiments for {symbol}")
                return df
        except Exception as e:
            self.logger.error(f"Sentiment retrieval failed: {str(e)}")
            return pd.DataFrame()
        
    def _load_config(self, config_path: str) -> dict:
        """Load and validate configuration file with encoding handling"""
        try:
            # Use UTF-8 encoding and strict error handling
            with open(config_path, 'r', encoding='utf-8', errors='strict') as f:
                config = yaml.safe_load(f)
            
            if not config.get('database') or not config['database'].get('path'):
                raise ValueError("Missing database path in config.yaml")
                
            return config
            
        except UnicodeDecodeError as e:
            self.logger.error(f"Invalid encoding in {config_path}: {str(e)}. Ensure the file is saved in UTF-8.")
            raise
        except Exception as e:
            self.logger.error(f"Config loading failed: {str(e)}")
            raise

