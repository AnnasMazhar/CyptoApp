# seed_prices.py

import feedparser
from urllib.parse import urlparse
import pandas as pd
import re
from datetime import datetime
import random
from dateutil.relativedelta import relativedelta
from database import Database
from textblob import TextBlob
import requests
import yfinance as yf
from bs4 import BeautifulSoup
import time
import xml.etree.ElementTree as ET

CRYPTO_MAPPING = {
    'BTC': ['bitcoin', 'btc', 'satosh', 'digital gold', 'halving',
            'store of value', 'sha-256', 'bit coin', '₿', 'lightning network',
            'segwit', 'taproot'],
    'ETH': ['ethereum', 'eth', 'vitalik', 'smart contract', 'dapp',
            'gas fee', 'merge', 'pos', 'shapella', 'danksharding',
            'erc-20', 'defi'],
    'BNB': ['binance', 'bnb', 'cz', 'exchange coin'],
    'XRP': ['xrp', 'ripple', 'brad garlinghouse', 'cross-border'],
    'ADA': ['cardano', 'ada', 'charles hoskinson', 'ouroboros'],
    'DOGE': ['dogecoin', 'doge', 'shiba', 'meme coin'],
    'SOL': ['solana', 'sol', 'anatoly yakovenko', 'high throughput'],
    'DOT': ['polkadot', 'dot', 'gavin wood', 'parachain'],
    'UNI': ['uniswap', 'uni', 'dex', 'automated market maker'],
    'LINK': ['chainlink', 'link', 'oracle', 'sergey nazarov']
}

def print_db_stats(db):
    """Display database statistics with enhanced formatting"""
    print("\n📊 Database Statistics:")
    tables = db.get_table_names()
    for table in tables:
        try:
            sample = db.get_table_sample(table, 1)
            count = db.conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[^2_0]
            print(f"- {table.upper():<20} {count:>6} records")
            if not sample.empty:
                print(f"  Latest entry: {sample.iloc[^2_0]['date'] if 'date' in sample else 'N/A'}")
        except Exception as e:
            print(f"- {table.upper():<20} : Error - {str(e)}")

def seed_historical_prices(db):
    """Fixed price seeding implementation"""
    crypto_config = {
        'BTC': {'sources': ['yfinance', 'coingecko'], 'start': '2010-07-17'},
        'ETH': {'sources': ['yfinance', 'coingecko'], 'start': '2015-08-07'},
        'BNB': {'sources': ['yfinance', 'coingecko'], 'start': '2017-07-25'},
        'XRP': {'sources': ['yfinance', 'coingecko'], 'start': '2014-08-04'},
        'ADA': {'sources': ['yfinance', 'coingecko'], 'start': '2017-10-01'},
        'SOL': {'sources': ['yfinance', 'coingecko'], 'start': '2020-04-10'},
        'DOT': {'sources': ['yfinance', 'coingecko'], 'start': '2020-08-19'},
        'DOGE': {'sources': ['yfinance', 'coingecko'], 'start': '2013-12-15'},
        'UNI': {'sources': ['yfinance', 'coingecko'], 'start': '2020-09-17'},
        'LINK': {'sources': ['yfinance', 'coingecko'], 'start': '2017-09-20'}
    }

    for symbol, config in crypto_config.items():
        print(f"\n🔨 Processing {symbol} historical prices...")
        dfs = []
        for source in config['sources']:
            try:
                if source == 'yfinance':
                    df = fetch_yfinance(symbol)
                elif source == 'coingecko':
                    df = fetch_coingecko(symbol)

                if not df.empty:
                    print(f"✅ {source}: Found {len(df)} raw records")
                    dfs.append(df)
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ {source} error: {str(e)}")
                continue

        if dfs:
            try:
                combined_df = pd.concat(dfs, ignore_index=True) # Important: Reset Index
                prepared_df = prepare_dataframe(combined_df, symbol)

                if prepared_df is not None and not prepared_df.empty:

                    # Get initial count
                    initial_count = db.conn.execute(
                        'SELECT COUNT(*) FROM historical_prices WHERE symbol = ?',
                        (symbol,)
                    ).fetchone()[^2_0]

                    # Save data
                    success = db.save_historical_data(symbol, prepared_df)

                    if success:
                        # Get final count
                        final_count = db.conn.execute(
                            'SELECT COUNT(*) FROM historical_prices WHERE symbol = ?',
                            (symbol,)
                        ).fetchone()[^2_0]
                        inserted = final_count - initial_count
                        print(f"💾 Inserted {inserted} new records for {symbol}")
                        print(f"Database now has {final_count} total records for {symbol}")
                    else:
                        print("❌ Database save failed")
                else:
                    print("⚠️ No valid data after processing")
            except Exception as e:
                print(f"❌ Final processing error: {str(e)}")
        else:
            print(f"⚠️ No data found for {symbol}")

def prepare_dataframe(df, symbol):
    """Ensure DataFrame matches database schema EXACTLY"""
    required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    try:
        # Check for required columns BEFORE any processing
        missing_columns = [col for col in required_columns if col not in df.columns and col != 'symbol']  #Exclude symbol as we add this
        if missing_columns:
            print(f"⚠️ Missing columns in DataFrame: {missing_columns}")
            return None

        # Rename columns to lowercase
        df.columns = df.columns.str.lower()

        # Process date first
        if 'date' in df.columns: #Ensure date exists after potential renaming
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        else:
            print("⚠️ 'date' column missing after renaming.")
            return None

        # Add symbol column
        df['symbol'] = symbol

        # Ensure correct column order
        df = df[required_columns]

        # Validate numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns: #Check if column exists before converting
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"⚠️ Numeric column '{col}' missing.")
                return None

        # Final cleanup
        df = df.drop_duplicates(['symbol', 'date']).dropna()

        print(f"✅ Validated columns: {list(df.columns)}")
        print(df.head())  # Print the first few rows of the DataFrame
        return df
    except Exception as e:
        print(f"❌ Data preparation failed: {str(e)}")
        return None

def fetch_yfinance(symbol):
    """Fetch and normalize Yahoo Finance data"""
    try:
        ticker = yf.Ticker(f"{symbol}-USD")
        df = ticker.history(period="max").reset_index()

        # Rename and order columns - Ensure Date is handled correctly
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
        else:
            print("⚠️ 'Date' column not found in yfinance data.")
            return pd.DataFrame()

        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Select columns before renaming
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        return df
    except Exception as e:
        print(f"❌ YFinance error: {str(e)}")
        return pd.DataFrame()

def fetch_coingecko(symbol):
    """Fetch historical data from CoinGecko with proper error handling"""
    coin_ids = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'BNB': 'binancecoin',
        'XRP': 'ripple',
        'ADA': 'cardano',
        'SOL': 'solana',
        'DOT': 'polkadot',
        'DOGE': 'dogecoin',
        'UNI': 'uniswap',
        'LINK': 'chainlink'
    }

    if symbol not in coin_ids:
        print(f"⚠️ No CoinGecko mapping for {symbol}")
        return pd.DataFrame()

    try:
        coin_id = coin_ids[symbol]
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {'vs_currency': 'usd', 'days': 'max'}
        response = requests.get(url, params=params, timeout=20)
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
        df['volume'] = 0  # CoinGecko doesn't provide historical volume

        return pd.DataFrame({
            'date': df['date'],
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': [^2_0] * len(df)  # CoinGecko doesn't provide volume
        })
    except Exception as e:
        print(f"❌ CoinGecko error: {str(e)}")
        return pd.DataFrame()

def seed_and_verify():
    """Improved seeding with proper validation"""
    db = Database()
    print("🚀 Starting data seeding...")
    seed_historical_prices(db)
    print("\n✅ Seeding complete!")
    print_db_stats(db)

if __name__ == "__main__":
    seed_and_verify()
    
    