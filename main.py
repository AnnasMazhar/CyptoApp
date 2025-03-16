import argparse
import logging
import os
from pathlib import Path
import pandas as pd
from price_sentiment_analyzer.pipeline.pipeline import PriceSentimentPipeline
from database.database import Database  # Add database import

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )

def save_results(results: dict, forecast: pd.Series, output_dir: str):
    """Save results and forecasts to files"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    pd.Series(results).to_csv(f'{output_dir}/metrics.csv')
    
    # Save forecast
    forecast.to_csv(f'{output_dir}/forecast.csv')
    
    # Save plots
    plt = pd.DataFrame({'Forecast': forecast}).plot(kind='bar', title='60-Day Sentiment Forecast').get_figure()
    plt.savefig(f'{output_dir}/forecast.png', bbox_inches='tight')
    plt.close()

def initialize_database(config_path: str):
    """Initialize database schema and tables"""
    logger = logging.getLogger(__name__)
    try:
        logger.info("Initializing database schema...")
        db = Database(config_path)
        db.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        if 'db' in locals():
            db.close()

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Sentiment Analysis Pipeline')
    parser.add_argument('--symbol', default='BTC', help='Cryptocurrency symbol (default: BTC)')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--init-db', action='store_true', help='Initialize database tables')
    parser.add_argument('--seed-demo-data', action='store_true', 
                      help='Seed database with demo data (requires --init-db)')
    args = parser.parse_args()

    try:
        # Handle database initialization first
        if args.init_db:
            initialize_database(args.config)
            if args.seed_demo_data:
                logger.info("Seeding demo data...")
                db = Database(args.config)
                try:
                    # Add demo data seeding logic here
                    db.seed_demo_data()
                    logger.info("Demo data seeded successfully")
                finally:
                    db.close()
            return

        logger.info(f"Initializing pipeline for {args.symbol}")
        pipeline = PriceSentimentPipeline(args.config)
        
        logger.info("Running analysis pipeline...")
        results, forecast = pipeline.run_pipeline(args.symbol)
        
        logger.info("Saving results...")
        save_results(results, forecast, args.output)
        
        logger.info("\n=== Analysis Results ===")
        for metric, value in results.items():
            logger.info(f"{metric.replace('_', ' ').title():<25}: {value:.4f}")
            
        logger.info("\n=== 60-Day Forecast ===")
        logger.info(forecast.to_string())

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()