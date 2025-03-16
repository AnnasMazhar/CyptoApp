import sqlite3
import pandas as pd
from datetime import datetime
import joblib
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import logging

class Database:
    def __init__(self, db_name: str = 'crypto_data.db', pool_size: int = 5):
        self.db_name = db_name
        self.pool_size = pool_size
        self._logger = logging.getLogger(__name__)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_connection_pool()
        self._initialize_version_table()
        self._migrate()
    
    def _init_connection_pool(self):
        """Initialize SQLite connection pool"""
        self.conn = sqlite3.connect(
            self.db_name,
            check_same_thread=False,
            timeout=30
        )
        # Pooling simulation for SQLite
        self.conn.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def managed_cursor(self) -> sqlite3.Cursor:
        """Context manager for connection handling"""
        try:
            cursor = self.conn.cursor()
            yield cursor
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self._logger.error(f"Transaction failed: {str(e)}")
            raise
        finally:
            cursor.close()

    def _initialize_version_table(self):
        """Create version tracking table if not exists"""
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS __version__ (
                    version INTEGER PRIMARY KEY,
                    migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            if not self.conn.execute('SELECT version FROM __version__').fetchone():
                self.conn.execute('INSERT INTO __version__ (version) VALUES (0)')

    def _migrate(self):
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

        current_version = self.conn.execute('SELECT version FROM __version__').fetchone()[0]
        
        for migration in sorted(migrations, key=lambda x: x['version']):
            if current_version < migration['version']:
                try:
                    with self.conn:
                        migration['up']()
                        self.conn.execute(
                            'UPDATE __version__ SET version = ?', 
                            (migration['version'],)
                        )
                        print(f"Applied migration v{migration['version']}")
                except Exception as e:
                    print(f"Migration v{migration['version']} failed: {str(e)}")
                    raise

    def _create_initial_schema(self):
        """Initial schema (v1) - FIXED"""
        cursor = self.conn.cursor()
        cursor.execute('''
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
        ''')  # Added closing parenthesis and fixed syntax

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                sentiment_positive REAL DEFAULT 0,
                sentiment_negative REAL DEFAULT 0,
                sentiment_neutral REAL DEFAULT 0,
                source TEXT,
                UNIQUE(symbol, date, title)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_news_data_symbol_date 
            ON news_data(symbol, date)
        ''')

    def _migrate_v2(self):
        """Migration to rename date -> published_at (v2)"""
        self.conn.execute('''
            CREATE TABLE news_data_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                published_at TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                sentiment_positive REAL DEFAULT 0,
                sentiment_negative REAL DEFAULT 0,
                sentiment_neutral REAL DEFAULT 0,
                source TEXT,
                UNIQUE(symbol, published_at, title))
        ''')
        self.conn.execute('''
            INSERT INTO news_data_new SELECT 
                id, symbol, date, title, content, 
                sentiment_positive, sentiment_negative, sentiment_neutral, source
            FROM news_data
        ''')
        self.conn.execute('DROP TABLE news_data')
        self.conn.execute('ALTER TABLE news_data_new RENAME TO news_data')
        self.conn.execute('''
            CREATE INDEX idx_news_data_symbol_published 
            ON news_data(symbol, published_at)
        ''')

    def _migrate_v3(self):
        """Migration to rename source -> source_name (v3)"""
        self.conn.execute('''
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
                UNIQUE(symbol, published_at, title))
        ''')
        self.conn.execute('''
            INSERT INTO news_data_new SELECT 
                id, symbol, published_at, title, content, 
                sentiment_positive, sentiment_negative, sentiment_neutral, source
            FROM news_data
        ''')
        self.conn.execute('DROP TABLE news_data')
        self.conn.execute('ALTER TABLE news_data_new RENAME TO news_data')
        self.conn.execute('''
            CREATE INDEX idx_news_data_symbol_published 
            ON news_data(symbol, published_at)
        ''')
    def _migrate_v4(self):
        """Create model versions table (v4)"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                    version_id INTEGER PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mae REAL,
                    r2_score REAL,
                    model_type TEXT,
                    feature_names TEXT,  -- Store as JSON array
                    model_params TEXT    -- Store as JSON
                    model_data BLOB
                )
            ''')

    def _migrate_v5(self):
        """Create prediction history table (v5)"""
        self.conn.execute('''
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
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date 
            ON model_predictions(symbol, prediction_date)
        ''')
    def _migrate_v6(self):
        """Create price sentiment table (v6) - FIXED"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS price_sentiment (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                sentiment TEXT CHECK(sentiment IN ('positive', 'negative', 'neutral')),
                metrics TEXT,  -- Changed from JSON to TEXT
                PRIMARY KEY (symbol, date)
            )
        ''')
        self.conn.execute('''
            CREATE INDEX idx_price_sentiment_symbol_date 
            ON price_sentiment(symbol, date)
        ''')

    def _migrate_v7(self):
        """Add model metadata columns (v7)"""
        self.conn.execute('''
            ALTER TABLE model_versions
            ADD COLUMN feature_names TEXT
        ''')
        self.conn.execute('''
            ALTER TABLE model_versions
            ADD COLUMN model_params TEXT
        ''')
        self.conn.execute('''
            CREATE INDEX idx_model_versions_type 
            ON model_versions(model_type)
        ''')

    # Add new methods for model management
    def save_model_version(self, model: object, metrics: dict, model_type: str,
                      feature_names: list, model_params: dict) -> int:
        """Save model with metadata"""
        try:
            model_blob = joblib.dumps(model)
            with self.conn:
                cursor = self.conn.execute('''
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
                return cursor.lastrowid
                
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return -1

    def save_prediction(self, symbol: str, prediction_date: str, 
                       forecast_date: str, predicted_price: float,
                       model_version: int, actual_price: float = None) -> bool:
        """Store a prediction with optional actual price"""
        try:
            with self.conn:
                self.conn.execute('''
                    INSERT INTO model_predictions 
                    (symbol, prediction_date, forecast_date, 
                     predicted_price, actual_price, model_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (symbol, prediction_date, forecast_date, 
                      predicted_price, actual_price, model_version))
            return True
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return False

    def get_latest_model(self, model_type: str = None) -> tuple:
        """Retrieve model with metadata"""
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
        
        row = self.conn.execute(query, params).fetchone()
        if row:
            try:
                model = joblib.loads(row[1])
                return (
                    row[0],  # version_id
                    model,
                    json.loads(row[2]),  # feature_names
                    json.loads(row[3])  # model_params
                )
            except Exception as e:
                print(f"Model load failed: {str(e)}")
        return (None, None, None, None)

    def get_predictions(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve predictions for analysis"""
        return pd.read_sql_query('''
            SELECT p.*, v.model_type 
            FROM model_predictions p
            JOIN model_versions v ON p.model_version = v.version_id
            WHERE symbol = ?
            AND forecast_date BETWEEN ? AND ?
            ORDER BY forecast_date
        ''', self.conn, params=(symbol, start_date, end_date), parse_dates=['prediction_date', 'forecast_date'])


    def save_historical_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Save price data with conflict resolution"""
        try:
            # Clean and prepare data
            data = data.rename(columns=str.lower)
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # Make sure the required columns exist
            for col in required_cols:
                if col not in data.columns:
                    print(f"Missing required column: {col}")
                    return False
                    
            # Use only required columns
            data = data[required_cols].copy()
            
            # Convert and validate dates
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data = data.dropna(subset=['date'])
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            
            # Add symbol column - always ensure it's the last column in the DataFrame
            data['symbol'] = symbol
            
            # Remove duplicates - keep first occurrence
            data = data.drop_duplicates(
                subset=['date', 'symbol'], 
                keep='first'
            )
            
            # Use UPSERT to handle conflicts
            with self.conn:
                # Create temporary table with correct column order
                final_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
                data = data[final_columns]
                data.to_sql(
                    'temp_prices',
                    self.conn,
                    if_exists='replace',
                    index=False
                )
                
                # Merge data using UPSERT - explicitly specify column order
                self.conn.execute('''
                    INSERT OR IGNORE INTO historical_prices
                    (date, open, high, low, close, volume, symbol)
                    SELECT date, open, high, low, close, volume, symbol FROM temp_prices
                ''')
                
                # Cleanup temporary table
                self.conn.execute('DROP TABLE IF EXISTS temp_prices')
                
            print(f"✅ Successfully saved {len(data)} records for {symbol}")
            return True
            
        except sqlite3.IntegrityError as e:
            print(f"⛔ Database error: {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return False

    def save_news_data(self, articles: List[Dict]) -> bool:
        #"""Batch insert news articles"""
        try:
            insert_query = '''
                INSERT OR IGNORE INTO news_data (
                    symbol, published_at, title, content,
                    sentiment_positive, sentiment_negative, sentiment_neutral,
                    source_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''
            records = [(
                article['symbol'],
                article['date'].isoformat(),
                article['title'][:255],
                article.get('content', '')[:5000],
                article.get('sentiment_positive', 0),
                article.get('sentiment_negative', 0),
                article.get('sentiment_neutral', 0),
                article.get('source', 'Unknown')[:50]
            ) for article in articles]

            with self.conn:
                self.conn.executemany(insert_query, records)
            return True
        except Exception as e:
            print(f"Error saving news data: {str(e)}")
            return False

    def get_historical_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """Retrieve price data"""
        query = '''
            SELECT date, open, high, low, close, volume
            FROM historical_prices
            WHERE symbol = ?
        '''
        params = [symbol]
        if days:
            query += " AND date >= date('now', ?)"
            params.append(f'-{days} days')
        query += " ORDER BY date ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df

    def get_latest_sentiments(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """Retrieve processed sentiment data"""
        query = '''
            SELECT 
                published_at AS date,
                title,
                content AS summary,
                (COALESCE(sentiment_positive,0)*0.6 
                - COALESCE(sentiment_negative,0)*0.3 
                + COALESCE(sentiment_neutral,0)*0.1 AS sentiment_score,
                source_name AS source
            FROM news_data 
            WHERE symbol = ?
            ORDER BY published_at DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, self.conn, params=[symbol, limit], parse_dates=['date'])
        
        if not df.empty:
            df = df.replace(r'^\s*$', pd.NA, regex=True)
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0)
            text_cols = ['title', 'summary', 'source']
            df[text_cols] = df[text_cols].fillna('Unknown').applymap(
                lambda x: x.strip()[:500] if isinstance(x, str) else x)
            return df.dropna(subset=['date'])
        return pd.DataFrame()

    def get_sentiment_features(self, symbol: str) -> pd.DataFrame:
        """Get aggregated sentiment features"""
        query = '''
            SELECT DATE(published_at) as date,
                AVG(sentiment_positive) as sent_pos,
                AVG(sentiment_negative) as sent_neg
            FROM news_data
            WHERE symbol = ?
            GROUP BY DATE(published_at)
        '''
        return pd.read_sql_query(query, self.conn, params=[symbol], parse_dates=['date'])



    # Utility methods
    def get_table_names(self) -> List[str]:
        """List all tables"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return [row[0] for row in cursor.fetchall()]

    def get_table_sample(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        """Sample table data"""
        try:
            return pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", self.conn)
        except Exception as e:
            print(f"Error sampling {table_name}: {str(e)}")
            return pd.DataFrame()
    def validate_model_compatibility(self, current_features: list, saved_features: list) -> bool:
        """Ensure features match model requirements"""
        missing = set(saved_features) - set(current_features)
        extra = set(current_features) - set(saved_features)
        
        if missing:
            print(f"Missing features: {missing}")
            return False
        if extra:
            print(f"Extra features: {extra}")
        return True

    def get_model_metadata(self, version_id: int) -> dict:
        """Retrieve model metadata"""
        row = self.conn.execute('''
            SELECT feature_names, model_params 
            FROM model_versions
            WHERE version_id = ?
        ''', (version_id,)).fetchone()
        
        return {
            'feature_names': json.loads(row[0]),
            'params': json.loads(row[1])
        } if row else {}

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    
if __name__ == "__main__":
    db = Database('crypto_data.db')
    print(db.get_table_names())
    print(db.get_table_sample('model_versions'))
    db.close()
