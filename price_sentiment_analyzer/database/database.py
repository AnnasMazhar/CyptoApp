import sqlite3
import pandas as pd
import json
import yaml
import os
from datetime import datetime
import joblib
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import logging
import pickle
import sqlite3
import logging
from contextlib import contextmanager

class Database:
    """Handles database schema creation and migrations only"""
    def __init__(self, db_path: str):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.db_path = db_path
        try:
            if not os.path.exists(self.db_path):
                open(self.db_path, 'w').close()

            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            with self._get_connection() as conn:
                    self._apply_migrations(conn)
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise


    @contextmanager
    def get_connection(self):  # Renamed from _get_connection
        """Public connection context manager"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def _get_database_name(self, db_path):
        """Fetches the database name from an SQLite database."""
        try:
            self.cursor.execute("PRAGMA database_list;")
            result = self.cursor.fetchall()

            if result:
                for seq, name, path in result:
                    if path == db_path:
                        return name
            return None

        except sqlite3.Error as e:
            self.logger.error(f"Error fetching database name: {e}")
            return None



    def _apply_migrations(self, conn):
        """Apply all schema migrations"""
        self._create_version_table(conn)
        current_version = self._get_current_version(conn)
        
        migrations = [  # Fixed list syntax
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
        ]  # Added closing square bracket
        
        for migration in sorted(migrations, key=lambda x: x['version']):
            if migration['version'] > current_version:
                try:
                    migration['up'](conn)  # Use existing connection
                    conn.execute('''
                        UPDATE __version__ 
                        SET version = ?
                    ''', (migration['version'],))  # Removed migrated_at (already handled by DEFAULT)
                    self.logger.info(f"Applied migration v{migration['version']}")
                except Exception as e:
                    self.logger.error(f"Migration v{migration['version']} failed: {str(e)}")
                    raise  # Preserve stack trace


    def _load_config(self, config_path: str) -> dict:
        """Shared config loading logic"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
                
            if not config.get('database') or not config['database'].get('path'):
                raise ValueError("Invalid config: Missing database configuration")
                
            return config

    def _create_version_table(self, conn):
        """Initialize version tracking table"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS __version__ (
                version INTEGER PRIMARY KEY,
                migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        if not conn.execute('SELECT version FROM __version__').fetchone():
            conn.execute('INSERT INTO __version__ (version) VALUES (0)')

    def _get_current_version(self, conn):
        """Get current schema version"""
        return conn.execute('SELECT version FROM __version__').fetchone()[0]


    
    def _create_initial_schema(self, conn):
        """Initial schema (v1)"""
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS historical_prices (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, date)
            );

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
            );

            CREATE INDEX IF NOT EXISTS idx_news_data_symbol_published
            ON news_data(symbol, published_at);
        ''')
        self.logger.info("Created initial database schema")

    def _migrate_v2(self, conn):
        """Migration to rename date -> published_at (v2)"""
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS news_data_new (
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
            );

            INSERT INTO news_data_new SELECT
            id, symbol, published_at, title, content,
            sentiment_positive, sentiment_negative, sentiment_neutral, source_name
            FROM news_data;

            DROP TABLE news_data;
            ALTER TABLE news_data_new RENAME TO news_data;

            CREATE INDEX IF NOT EXISTS idx_news_data_symbol_published
            ON news_data(symbol, published_at);
        ''')

    def _migrate_v3(self, conn):
        """Migration to rename source -> source_name (v3)"""
        # This migration is not needed as the column is already named source_name

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
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS model_predictions (
                prediction_id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                forecast_date TEXT NOT NULL,
                predicted_price REAL,
                actual_price REAL,
                model_version INTEGER,
                FOREIGN KEY(model_version) REFERENCES model_versions(version_id)
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date
            ON model_predictions(symbol, prediction_date);
        ''')

    def _migrate_v6(self, conn):
        """Create price sentiment table (v6)"""
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS price_sentiment (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                sentiment TEXT CHECK(sentiment IN ('positive', 'negative', 'neutral')),
                metrics TEXT,
                PRIMARY KEY (symbol, date)
            );

            CREATE INDEX IF NOT EXISTS idx_price_sentiment_symbol_date
            ON price_sentiment(symbol, date);
        ''')

    def _migrate_v7(self, conn):
        conn.executescript('''
            CREATE INDEX IF NOT EXISTS idx_model_versions_type ON model_versions (model_type);
        ''')
            

    #Utility
    def get_table_names(self) -> List[str]:
        with self.get_connection() as conn:  # Fixed method name
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [row[0] for row in cursor.fetchall()]
