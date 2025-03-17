# seed_prices.py

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from data_loader import DataLoader
import requests
import yfinance as yf
import time

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

def print_db_stats(data_loader: DataLoader):
    """Display database statistics with enhanced formatting"""
    print("\n📊 Database Statistics:")
    try:
        with data_loader._get_connection() as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            for table in tables:
                table_name = table[0]
                count = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
                print(f"- {table_name.upper():<20} {count:>6} records")
                if table_name == 'historical_prices':
                    latest = conn.execute(
                        'SELECT MAX(date) FROM historical_prices'
                    ).fetchone()[0]
                    print(f"  Latest historical entry: {latest}")
    except Exception as e:
        print(f"Error getting stats: {str(e)}")

def seed_historical_prices(data_loader: DataLoader, years=20):
    """Fixed price seeding implementation for the last 'years' years."""
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

    end_date = datetime.now()
    start_date = end_date - relativedelta(years=years)

    for symbol, config in crypto_config.items():
        print(f"\n🔨 Processing {symbol} historical prices...")
        dfs = []
        for source in config['sources']:
            try:
                if source == 'yfinance':
                    df = fetch_yfinance(symbol, start_date, end_date)
                elif source == 'coingecko':
                    df = fetch_coingecko(symbol, start_date, end_date)

                if not df.empty:
                    print(f"✅ {source}: Found {len(df)} raw records")
                    dfs.append(df)
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ {source} error: {str(e)}")
                continue

        if dfs:
            try:
                combined_df = pd.concat(dfs, ignore_index=True)
                prepared_df = prepare_dataframe(combined_df, symbol)

                if not prepared_df.empty:
                    success = data_loader.save_historical_data(symbol, prepared_df)
                    
                    if success:
                        df = data_loader.get_historical_data(symbol)
                        print(f"💾 Total records for {symbol}: {len(df)}")
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
        missing_columns = [col for col in required_columns if col not in df.columns and col != 'symbol']
        if missing_columns:
            print(f"⚠️ Missing columns in DataFrame: {missing_columns}")
            return None

        df.columns = df.columns.str.lower()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        else:
            print("⚠️ 'date' column missing after renaming.")
            return None

        df['symbol'] = symbol
        df = df[required_columns]

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"⚠️ Numeric column '{col}' missing.")
                return None

        df = df.drop_duplicates(['symbol', 'date']).dropna()
        print(f"✅ Validated columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Data preparation failed: {str(e)}")
        return None

def fetch_yfinance(symbol, start_date, end_date):
    """Fetch and normalize Yahoo Finance data for a specified date range."""
    try:
        ticker = yf.Ticker(f"{symbol}-USD")
        df = ticker.history(start=start_date, end=end_date).reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"❌ YFinance error: {str(e)}")
        return pd.DataFrame()

def fetch_coingecko(symbol, start_date, end_date):
    """Fetch historical data from CoinGecko for a specified date range."""
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
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {'vs_currency': 'usd', 'from': start_timestamp, 'to': end_timestamp}
        response = requests.get(url, params=params, timeout=20)
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
        return df[['date', 'open', 'high', 'low', 'close']].assign(volume=0)
    except Exception as e:
        print(f"❌ CoinGecko error: {str(e)}")
        return pd.DataFrame()

def seed_and_verify(config_path='config.yaml', years=20):
    """Improved seeding with proper validation"""
    data_loader = DataLoader(config_path)  # Use config path from argument
    print("🚀 Starting data seeding...")
    seed_historical_prices(data_loader, years)
    print("\n✅ Seeding complete!")
    print_db_stats(data_loader)

if __name__ == "__main__":
    seed_and_verify()

