import argparse
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from price_sentiment_analyzer.pipeline.pipeline import PriceSentimentPipeline
from price_sentiment_analyzer.database.database import Database  # Only needed for migrations
from price_sentiment_analyzer.database.data_loader import DataLoader
import yaml
from price_sentiment_analyzer.database.seed_prices import seed_and_verify  # Add this import


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )


with open("config.yaml") as f:
    config = yaml.safe_load(f)
db_path = config["database"]["path"]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with error handling"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {str(e)}")
        raise

def seed_demo_data(config_path: str):
    """Seed database using external data sources"""
    logger = logging.getLogger(__name__)
    try:
        logger.info("Seeding historical price data from external sources...")
        seed_and_verify(config_path)
        logger.info("Data seeding completed successfully")
        return True
    except Exception as e:
        logger.error(f"Data seeding failed: {str(e)}")
        raise

def save_results(results: dict, forecast: pd.Series, output_dir: str):
    """Save results and forecasts to files"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.Series(results).to_csv(f'{output_dir}/metrics.csv')
    forecast.to_csv(f'{output_dir}/forecast.csv')
    
    fig = pd.DataFrame({'Forecast': forecast}).plot(kind='bar', 
                                                  title='60-Day Sentiment Forecast').get_figure()
    fig.savefig(f'{output_dir}/forecast.png', bbox_inches='tight')
    plt.close(fig)

def initialize_database(config_path: str) -> DataLoader:
    """Safer database initialization"""
    logger = logging.getLogger(__name__)
    try:
        config = load_config(config_path)
        db_path = config["database"]["path"]
        
        logger.info("Initializing database schema...")
        db = Database(db_path)  # CORRECT: Use db_path
        data_loader = DataLoader(config_path)
        with data_loader._get_connection() as conn:
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
            
        logger.info("Database initialized successfully")
        return data_loader
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def generate_sample_data(symbol: str = 'BTC') -> pd.DataFrame:
    """Generate synthetic demo data"""
    dates = pd.date_range(start='2020-01-01', periods=500)
    np.random.seed(42)
    return pd.DataFrame({
        'date': dates,
        'open': np.random.normal(30000, 5000, 500).cumsum(),
        'high': np.random.normal(30500, 5000, 500).cumsum(),
        'low': np.random.normal(29500, 5000, 500).cumsum(),
        'close': np.random.normal(30000, 5000, 500).cumsum(),
        'volume': np.random.poisson(100000, 500),
        'symbol': symbol
    })

def seed_demo_data(data_loader: DataLoader):
    """Seed database using DataLoader"""
    logger = logging.getLogger(__name__)
    try:
        logger.info("Seeding demo data...")
        sample_data = generate_sample_data()
        
        if data_loader.save_historical_data('BTC', sample_data):
            logger.info(f"Seeded {len(sample_data)} records")
            
            # Generate visualization
            plt.figure(figsize=(10, 5))
            sample_data.set_index('date')['close'].plot(title='Sample Closing Prices')
            plt.savefig('demo_data_preview.png')
            logger.info("Saved sample data visualization")
            
            return True
        return False
    except Exception as e:
        logger.error(f"Data seeding failed: {str(e)}")
        raise

def view_table_sample(data_loader: DataLoader, table_name: str, sample_size: int = 10):
    """View table data using DataLoader"""
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Fetching sample from {table_name}...")
        df = data_loader.get_historical_data('BTC')  # Example for historical prices
        
        if not df.empty:
            logger.info(f"\n=== {table_name.upper()} SAMPLE ===")
            logger.info(df.sample(min(sample_size, len(df))).to_string())
            
            # Basic stats
            logger.info("\n📈 Statistics:")
            logger.info(df.describe().to_string())
            
            # Visualization
            plt.figure(figsize=(10, 5))
            df['close'].plot(title='Closing Prices Preview')
            plt.savefig(f'{table_name}_preview.png')
            return True
        return False
    except Exception as e:
        logger.error(f"Data retrieval failed: {str(e)}")
        raise

def generate_data_visualizations(data: pd.DataFrame, table_name: str):
    """Generate visualization files"""
    try:
        vis_dir = Path("visualizations")
        vis_dir.mkdir(exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        data['close'].plot(ax=ax, title=f'{table_name} Closing Prices')
        fig.savefig(vis_dir/f'{table_name}_prices.png')
        plt.close(fig)
        return True
    except Exception as e:
        logging.warning(f"Visualization failed: {str(e)}")
        return False

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='Crypto Price Sentiment Analysis')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--symbol', default='BTC', help='Cryptocurrency symbol')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--init-db', action='store_true', help='Initialize database')
    parser.add_argument('--seed-demo', action='store_true', help='Seed demo data')
    parser.add_argument('--view-table', help='Table name to preview')
    parser.add_argument('--run-pipeline', action='store_true',  # Changed to store_true
                        help='Run Price Sentiment Pipeline')


    args = parser.parse_args()

    try:
        # Load config first
        config = load_config(args.config)  # Use the function
        db_path = config["database"]["path"]
        
        # Initialize components with config
        db = Database(db_path)
        data_loader = DataLoader(args.config)
        
        # Example usage
        if args.init_db:
            logger.info("Initializing database schema...")
            initialize_database(args.config)


        # Table preview logic
        if args.view_table:
            data_loader = DataLoader(args.config)
            view_table_sample(data_loader, args.view_table)

        # Pipeline execution logic
        if args.run_pipeline:
            logger.info(f"\n{'='*30} Starting Analysis Pipeline {'='*30}")
            pipeline = PriceSentimentPipeline(args.config)
            results, forecast = pipeline.run_pipeline(args.symbol)
            
            save_results(results, forecast, args.output)
            logger.info("\n=== FINAL RESULTS ===")
            logger.info(pd.Series(results).to_string())
            logger.info(f"Results saved to {args.output} directory")

    except Exception as e:
        logger.error(f"Operation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

# Initialize DB and seed data (optional standalone steps)
# python main.py --init-db
# python main.py --seed-demo

# # Full pipeline execution (initializes DB, seeds data, runs analysis)
# python main.py --run-pipeline --symbol BTC --output results

# # View table data
# python main.py --view-table historical_prices

