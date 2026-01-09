"""
Complete Cryptocurrency Data Pipeline
Pipe & Filter Architecture with Scheduler
Meets SRS Requirements: FR1, FR2, FR3, FR3.1, FR6, FR10
"""
import os

import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import time
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import sys
import argparse
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")


class CryptoPipeline:
    """
    Pipe & Filter Architecture for Cryptocurrency Data Collection
    """

    def __init__(self, database_url):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        logger.info("✅ Pipeline initialized")

    # ==================== FILTER 1: GET SYMBOLS ====================

    def filter1_get_symbols(self, limit=1000):
        """
        FR1: Auto Symbol Download
        FR1.1: Symbol Validation (volume-based, USDT pairs only)

        Fetches top cryptocurrencies from Binance API
        Returns: List of validated symbols with metadata
        """
        logger.info("🔍 FILTER 1: Fetching top symbols from Binance...")

        try:
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                timeout=10
            )
            response.raise_for_status()
            tickers = response.json()

            # FR1.1: Validate symbols
            validated_tickers = []
            for ticker in tickers:
                symbol = ticker['symbol']

                # Only USDT pairs
                if not symbol.endswith('USDT'):
                    continue

                # Exclude low liquidity (volume < $100k)
                volume_usd = float(ticker['quoteVolume'])
                if volume_usd < 100000:
                    continue

                ticker['volume_float'] = volume_usd
                validated_tickers.append(ticker)

            # Sort by volume and take top N
            validated_tickers.sort(key=lambda x: x['volume_float'], reverse=True)
            top_symbols = validated_tickers[:limit]

            logger.info(f"✅ FILTER 1 Complete: {len(top_symbols)} validated symbols")
            return top_symbols

        except Exception as e:
            logger.error(f"❌ FILTER 1 Failed: {e}")
            return []

    # ==================== FILTER 2: CHECK DATABASE STATUS ====================

    def filter2_check_last_date(self, symbols):
        """
        FR2: Database Status Check

        For each symbol, checks if data exists in database
        and identifies the last available date

        Returns: Symbols with last_date metadata
        """
        logger.info("📊 FILTER 2: Checking database status...")

        try:
            with self.engine.connect() as conn:
                # Get last date for each symbol
                query = text("""
                             SELECT symbol, MAX(date) as last_date
                             FROM crypto_prices
                             GROUP BY symbol
                             """)
                result = conn.execute(query)
                last_dates = {row[0]: row[1] for row in result}

            # Add last_date to each symbol
            new_symbols = 0
            existing_symbols = 0

            for symbol in symbols:
                symbol_name = symbol['symbol']
                last_date = last_dates.get(symbol_name)

                if last_date:
                    symbol['last_date'] = last_date.strftime('%Y-%m-%d')
                    existing_symbols += 1
                else:
                    symbol['last_date'] = None
                    new_symbols += 1

            logger.info(f"✅ FILTER 2 Complete: {new_symbols} new, {existing_symbols} existing")
            return symbols

        except Exception as e:
            logger.error(f"❌ FILTER 2 Failed: {e}")
            # If table doesn't exist, all symbols are new
            for symbol in symbols:
                symbol['last_date'] = None
            return symbols

    # ==================== FILTER 3: FETCH & STORE DATA ====================

    def filter3_fill_data(self, symbols, days_back=3650):
        """
        FR3: Historical Data Fetch
        FR3.1: Incremental Updates
        NFR5: API Rate Limiting (10 req/sec)

        Downloads OHLCV data and stores directly to PostgreSQL
        """
        logger.info(f"📥 FILTER 3: Fetching and storing data...")

        total_records = 0
        total_inserted = 0
        skipped_symbols = 0

        for i, ticker in enumerate(symbols, 1):
            symbol = ticker['symbol']

            # Determine date range
            if ticker['last_date']:
                # FR3.1: Incremental update
                start = datetime.strptime(ticker['last_date'], '%Y-%m-%d') + timedelta(days=1)
                logger.info(f"   [{i}/{len(symbols)}] {symbol}: Incremental from {start.date()}")
            else:
                # FR3: Full historical download
                start = datetime.now() - timedelta(days=days_back)
                logger.info(f"   [{i}/{len(symbols)}] {symbol}: Full download from {start.date()}")

            end = datetime.now()

            # Skip if already up to date
            if start >= end:
                logger.info(f"   [{i}/{len(symbols)}] {symbol}: Already up to date ✓")
                continue

            try:
                # Fetch data from Binance
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': '1d',
                    'startTime': int(start.timestamp() * 1000),
                    'endTime': int(end.timestamp() * 1000),
                    'limit': 1000
                }

                response = requests.get(url, params=params, timeout=10)

                if response.status_code != 200:
                    logger.warning(f"   [{i}/{len(symbols)}] {symbol}: API error {response.status_code}")
                    skipped_symbols += 1
                    continue

                klines = response.json()

                if not klines or isinstance(klines, dict):
                    logger.warning(f"   [{i}/{len(symbols)}] {symbol}: No data returned")
                    skipped_symbols += 1
                    continue

                # Store directly to database
                inserted = self._store_data_to_db(symbol, klines)
                total_records += len(klines)
                total_inserted += inserted

                logger.info(f"   [{i}/{len(symbols)}] {symbol}: Stored {inserted} records ✓")

                # NFR5: Rate limiting (10 requests/sec = 100ms delay)
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"   [{i}/{len(symbols)}] {symbol}: Error - {e}")
                skipped_symbols += 1
                continue

        logger.info(f"✅ FILTER 3 Complete: {total_inserted}/{total_records} records stored")
        logger.info(f"   Skipped {skipped_symbols} symbols due to errors")

        return total_inserted

    def _store_data_to_db(self, symbol, klines):
        """
        FR5: Data Storage with duplicate prevention
        FR4: Data Transformation (normalize prices, format dates)
        """
        inserted = 0

        try:
            with self.engine.connect() as conn:
                for k in klines:
                    # FR4: Transform data
                    date = datetime.fromtimestamp(k[0] / 1000).strftime('%Y-%m-%d')
                    open_price = round(float(k[1]), 8)
                    high = round(float(k[2]), 8)
                    low = round(float(k[3]), 8)
                    close = round(float(k[4]), 8)
                    volume = round(float(k[5]), 8)

                    # FR5: Insert with duplicate prevention
                    query = text("""
                                 INSERT INTO crypto_prices
                                     (symbol, date, open, high, low, close, volume)
                                 VALUES (:symbol, :date, :open, :high, :low, :close,
                                         :volume) ON CONFLICT (symbol, date) DO NOTHING
                                 """)

                    result = conn.execute(query, {
                        'symbol': symbol,
                        'date': date,
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume
                    })

                    if result.rowcount > 0:
                        inserted += 1

                conn.commit()

        except Exception as e:
            logger.error(f"Database error for {symbol}: {e}")

        return inserted

    # ==================== MAIN PIPELINE ====================

    def run_pipeline(self, limit=1000, days_back=3650):
        """
        Execute complete pipeline: Filter1 → Filter2 → Filter3
        """
        logger.info("=" * 80)
        logger.info("🚀 STARTING CRYPTOCURRENCY DATA PIPELINE")
        logger.info("=" * 80)

        start_time = time.time()

        # Filter 1: Get symbols
        symbols = self.filter1_get_symbols(limit)
        if not symbols:
            logger.error("❌ Pipeline failed: No symbols fetched")
            return False

        # Filter 2: Check database status
        symbols_with_dates = self.filter2_check_last_date(symbols)

        # Filter 3: Fetch and store data
        total_inserted = self.filter3_fill_data(symbols_with_dates, days_back)

        elapsed = time.time() - start_time

        logger.info("=" * 80)
        logger.info(f"✅ PIPELINE COMPLETE")
        logger.info(f"   Time: {elapsed:.2f}s")
        logger.info(f"   Records inserted: {total_inserted:,}")
        logger.info("=" * 80)

        # Show database stats
        self._show_database_stats()

        return True

    def _show_database_stats(self):
        """Display database statistics"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                             SELECT COUNT(DISTINCT symbol) as total_symbols,
                                    COUNT(*)               as total_records,
                                    MIN(date)              as earliest_date,
                                    MAX(date)              as latest_date
                             FROM crypto_prices
                             """)
                result = conn.execute(query)
                row = result.fetchone()

                logger.info("\n📊 DATABASE STATISTICS:")
                logger.info(f"   Total Symbols: {row[0]:,}")
                logger.info(f"   Total Records: {row[1]:,}")
                logger.info(f"   Date Range: {row[2]} to {row[3]}")
        except Exception as e:
            logger.warning(f"Could not fetch stats: {e}")


# ==================== SCHEDULER (FR6) ====================

class PipelineScheduler:
    """
    FR6: Scheduled Execution
    Runs pipeline daily at configured time
    """

    def __init__(self, pipeline, schedule_time="00:00"):
        self.pipeline = pipeline
        self.scheduler = BackgroundScheduler()
        self.schedule_time = schedule_time
        logger.info(f"⏰ Scheduler initialized for daily run at {schedule_time} UTC")

    def start(self):
        """Start the scheduler"""
        # Parse schedule time
        hour, minute = map(int, self.schedule_time.split(':'))

        # Add job
        self.scheduler.add_job(
            func=self.pipeline.run_pipeline,
            trigger=CronTrigger(hour=hour, minute=minute),
            id='daily_crypto_update',
            name='Daily Cryptocurrency Data Update',
            replace_existing=True
        )

        self.scheduler.start()
        logger.info(f"✅ Scheduler started - will run daily at {self.schedule_time} UTC")
        logger.info("   Press Ctrl+C to stop")

    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("🛑 Scheduler stopped")


# ==================== MAIN (FR10: Manual Trigger) ====================

def main():
    """
    FR10: Manual Trigger Support
    Command-line interface for pipeline execution
    """
    parser = argparse.ArgumentParser(
        description='Cryptocurrency Data Pipeline - Pipe & Filter Architecture'
    )

    parser.add_argument(
        '--run-now',
        action='store_true',
        help='Run pipeline immediately (manual trigger)'
    )

    parser.add_argument(
        '--schedule',
        action='store_true',
        help='Start scheduled execution (runs daily)'
    )

    parser.add_argument(
        '--schedule-time',
        default='00:00',
        help='Time to run daily (HH:MM format, UTC). Default: 00:00'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='Number of top cryptocurrencies to track. Default: 1000'
    )

    parser.add_argument(
        '--days-back',
        type=int,
        default=3650,
        help='Days of historical data for new symbols. Default: 3650 (10 years)'
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = CryptoPipeline(DATABASE_URL)

    # FR10: Manual trigger
    if args.run_now:
        logger.info("🔥 Manual trigger activated")
        success = pipeline.run_pipeline(limit=args.limit, days_back=args.days_back)
        sys.exit(0 if success else 1)

    # FR6: Scheduled execution
    elif args.schedule:
        logger.info("⏰ Starting scheduled execution mode")

        # Run once immediately
        logger.info("🔥 Running initial pipeline execution...")
        pipeline.run_pipeline(limit=args.limit, days_back=args.days_back)

        # Start scheduler
        scheduler = PipelineScheduler(pipeline, args.schedule_time)
        scheduler.start()

        try:
            # Keep running
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            scheduler.stop()
            logger.info("👋 Exiting...")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python pipeline.py --run-now                    # Run pipeline once")
        print("  python pipeline.py --schedule                   # Run daily at 00:00 UTC")
        print("  python pipeline.py --schedule --schedule-time 02:30  # Run daily at 02:30 UTC")
        print("  python pipeline.py --run-now --limit 500        # Track top 500 cryptos")


if __name__ == '__main__':
    main()