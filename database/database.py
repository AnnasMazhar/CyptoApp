import sqlite3
import pandas as pd
import json
from datetime import datetime
import joblib
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import logging
import pickle

class Database:
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connection_pool = []
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self._initialize_pool()
        self._initialize_database()

    def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            self.connection_pool.append(conn)

    def _initialize_database(self) -> None:
        with self.transaction() as conn:
            self._create_version_table(conn)
            self._apply_migrations(conn)

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with context management."""
        if not self.connection_pool:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
            conn.close()
        else:
            conn = self.connection_pool.pop()
            try:
                yield conn
            finally:
                self.connection_pool.append(conn)

    @contextmanager
    def transaction(self):
        """Context manager for transaction handling."""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Transaction error: {str(e)}")
                raise

    def _create_version_table(self, conn):
        """Create version tracking table if not exists"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS __version__ (
                version INTEGER PRIMARY KEY,
                migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        if not conn.execute('SELECT version FROM __version__').fetchone():
            conn.execute('INSERT INTO __version__ (version) VALUES (0)')

    def _apply_migrations(self, conn):
        """Execute schema migrations in order"""
        migrations = [
            {
                'version': 1,
                'description': 'Initial schema setup',
                'up': self._create_initial_schema
            },
            {
                'version': 2,
                'description': 'Rename news_data.date to published_at',
                'up': self._migrate_v2
            },
            {
                'version': 3,
                'description': 'Rename source to source_name in news_data',
                'up': self._migrate_v3
            },
            {
                'version': 4,
                'description': 'Add model storage tables',
                'up': self._migrate_v4
            },
            {
                'version': 5,
                'description': 'Add prediction history table',
                'up': self._migrate_v5
            },
            {
                'version': 6,
                'description': 'Add price sentiment table',
                'up': self._migrate_v6
            },
            {
                'version': 7,
                'description': 'Add model metadata columns',
                'up': self._migrate_v7
            }
        ]

        current_version = conn.execute('SELECT version FROM __version__').fetchone()[0]
        for migration in sorted(migrations, key=lambda x: x['version']):
            if current_version < migration['version']:
                try:
                    migration['up'](conn)
                    conn.execute('UPDATE __version__ SET version = ?', (migration['version'],))
                    self.logger.info(f"Applied migration v{migration['version']}")
                except Exception as e:
                    self.logger.error(f"Migration v{migration['version']} failed: {str(e)}")
                    raise

    def _create_initial_schema(self, conn):
        """Initial schema (v1) - FIXED"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS historical_prices (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, date)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                published_at TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                sentiment_positive REAL DEFAULT 0,
                sentiment_negative REAL DEFAULT 0,
                sentiment_neutral REAL DEFAULT 0,
                source_name TEXT,
                UNIQUE(symbol, published_at, title)
            )
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_news_data_symbol_published
            ON news_data(symbol, published_at)
        ''')

    def _migrate_v2(self, conn):
        """Migration to rename date -> published_at (v2)"""
        conn.execute('''
            CREATE TABLE news_data_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                published_at TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                sentiment_positive REAL DEFAULT 0,
                sentiment_negative REAL DEFAULT 0,
                sentiment_neutral REAL DEFAULT 0,
                source_name TEXT,
                UNIQUE(symbol, published_at, title)
            )
        ''')
        conn.execute('''
            INSERT INTO news_data_new SELECT
            id, symbol, date, title, content,
            sentiment_positive, sentiment_negative, sentiment_neutral, source
            FROM news_data
        ''')
        conn.execute('DROP TABLE news_data')
        conn.execute('ALTER TABLE news_data_new RENAME TO news_data')
        conn.execute('''
            CREATE INDEX idx_news_data_symbol_published
            ON news_data(symbol, published_at)
        ''')

    def _migrate_v3(self, conn):
        """Migration to rename source -> source_name (v3)"""
        conn.execute('''
            CREATE TABLE news_data_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                published_at TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                sentiment_positive REAL DEFAULT 0,
                sentiment_negative REAL DEFAULT 0,
                sentiment_neutral REAL DEFAULT 0,
                source_name TEXT,
                UNIQUE(symbol, published_at, title)
            )
        ''')
        conn.execute('''
            INSERT INTO news_data_new SELECT
            id, symbol, published_at, title, content,
            sentiment_positive, sentiment_negative, sentiment_neutral, source
            FROM news_data
        ''')
        conn.execute('DROP TABLE news_data')
        conn.execute('ALTER TABLE news_data_new RENAME TO news_data')
        conn.execute('''
            CREATE INDEX idx_news_data_symbol_published
            ON news_data(symbol, published_at)
        ''')

    def _migrate_v4(self, conn):
        """Create model versions table (v4)"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id INTEGER PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mae REAL,
                r2_score REAL,
                model_type TEXT,
                feature_names TEXT,
                model_params TEXT,
                model_data BLOB
            )
        ''')

    def _migrate_v5(self, conn):
        """Create prediction history table (v5)"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_predictions (
                prediction_id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                forecast_date TEXT NOT NULL,
                predicted_price REAL,
                actual_price REAL,
                model_version INTEGER,
                FOREIGN KEY(model_version) REFERENCES model_versions(version_id)
            )
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date
            ON model_predictions(symbol, prediction_date)
        ''')

    def _migrate_v6(self, conn):
        """Create price sentiment table (v6) - FIXED"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS price_sentiment (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                sentiment TEXT CHECK(sentiment IN ('positive', 'negative', 'neutral')),
                metrics TEXT,
                PRIMARY KEY (symbol, date)
            )
        ''')
        conn.execute('''
            CREATE INDEX idx_price_sentiment_symbol_date
            ON price_sentiment(symbol, date)
        ''')

    def _migrate_v7(self, conn):
        """Add model metadata columns (v7)"""
        conn.execute('''
            ALTER TABLE model_versions
            ADD COLUMN feature_names TEXT
        ''')
        conn.execute('''
            ALTER TABLE model_versions
            ADD COLUMN model_params TEXT
        ''')
        conn.execute('''
            CREATE INDEX idx_model_versions_type
            ON model_versions(model_type)
        ''')

    def save_model_version(self, model: Any, metrics: Dict[str, float], model_type: str,
                            feature_names: List[str], model_params: Dict[str, Any]) -> Optional[int]:
        """Save model with metadata"""
        try:
            model_blob = joblib.dumps(model)
            with self.transaction() as conn:
                cursor = conn.execute('''
                    INSERT INTO model_versions
                    (mae, r2_score, model_type, model_data,
                    feature_names, model_params)
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
                self.logger.info(f"Saved model version {version_id} successfully")
                return version_id
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return None

    def save_prediction(self, symbol: str, prediction_date: str,
                        forecast_date: str, predicted_price: float,
                        model_version: int, actual_price: Optional[float] = None) -> bool:
        """Store a prediction with optional actual price"""
        try:
            with self.transaction() as conn:
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
            self.logger.error(f"Error saving prediction: {str(e)}")
            return False

    def get_latest_model(self, model_type: Optional[str] = None) -> Tuple[Optional[int], Optional[Any], Optional[List[str]], Optional[Dict[str, Any]]]:
        """Retrieve model with metadata"""
        try:
            with self.get_connection() as conn:
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
                    self.logger.info(f"Loaded model version {row['version_id']} successfully")
                    return (row['version_id'], model, feature_names, model_params)
                else:
                    self.logger.warning(f"No model found with model_type: {model_type}")
                    return (None, None, None, None)
        except Exception as e:
            self.logger.error(f"Model load failed: {str(e)}")
            return (None, None, None, None)

    def get_predictions(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve predictions for analysis"""
        try:
            with self.get_connection() as conn:
                query = '''
                    SELECT p.*, v.model_type
                    FROM model_predictions p
                    JOIN model_versions v ON p.model_version = v.version_id
                    WHERE symbol = ?
                    AND forecast_date BETWEEN ? AND ?
                    ORDER BY forecast_date
                '''
                params = (symbol, start_date, end_date)
                df = pd.read_sql_query(query, conn, params=params, parse_dates=['prediction_date', 'forecast_date'])
                self.logger.info(f"Retrieved predictions for {symbol} between {start_date} and {end_date}")
                return df
        except Exception as e:
            self.logger.error(f"Error retrieving predictions: {str(e)}")
            return pd.DataFrame()

    def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Get historical price data for a symbol."""
        try:
            with self.get_connection() as conn:
                query = "SELECT date, open, high, low, close, volume FROM historical_prices WHERE symbol = ? ORDER BY date"
                cursor = conn.execute(query, (symbol,))
                data = cursor.fetchall()

                if not data:
                    self.logger.warning(f"No historical data found for symbol: {symbol}")
                    return pd.DataFrame()

                df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                self.logger.info(f"Retrieved historical data for {symbol}")
                return df
        except Exception as e:
            self.logger.error(f"Error retrieving historical data: {str(e)}")
            return pd.DataFrame()

    def apply_migration(self, version: str, script: str) -> bool:
        """Apply a database migration script."""
        try:
            with self.transaction() as conn:
                # Check if migration was already applied
                cursor = conn.execute(
                    "SELECT id FROM migrations WHERE version = ?",
                    (version,)
                )
                existing = cursor.fetchone()
                if existing:
                    self.logger.info(f"Migration {version} already applied")
                    return False

                # Apply the migration
                for statement in script.split(';'):
                    if statement.strip():
                        conn.execute(statement)

                # Record the migration
                conn.execute(
                    "INSERT INTO migrations (version) VALUES (?)",
                    (version,)
                )

                self.logger.info(f"Applied migration {version} successfully")
                return True
        except Exception as e:
            self.logger.error(f"Migration error: {str(e)}")
            return False

    def get_latest_sentiments(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """Retrieve processed sentiment data"""
        try:
            with self.get_connection() as conn:
                query = '''
                    SELECT
                        published_at AS date,
                        title,
                        content AS summary,
                        (COALESCE(sentiment_positive,0)*0.6
                        - COALESCE(sentiment_negative,0)*0.3
                        + COALESCE(sentiment_neutral,0)*0.1) AS sentiment_score
                    FROM news_data
                    WHERE symbol = ?
                    ORDER BY published_at DESC
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, conn, params=(symbol, limit), parse_dates=['date'])
                self.logger.info(f"Retrieved latest sentiments for {symbol}")
                return df
        except Exception as e:
            self.logger.error(f"Error retrieving sentiments: {str(e)}")
            return pd.DataFrame()
