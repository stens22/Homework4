"""
Repository Pattern - Database Access Layer
Handles all database operations

FIXED VERSION - Shows recent data instead of 2017 data
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta  # ← FIXED: Added imports
import logging

logger = logging.getLogger(__name__)


class CryptoRepository:
    """Repository for cryptocurrency price data"""

    def __init__(self, connection_string: str):
        """Initialize with database connection string"""
        self.connection_string = connection_string
        self.engine = None
        self._connect()

    def _connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.connection_string)
            logger.info("✅ Database connected")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    def get_price_history(self, symbol: str, days_back: int = 100) -> pd.DataFrame:
        try:
            query = """
                    SELECT date, open, high, low, close, volume
                    FROM crypto_prices
                    WHERE symbol = :symbol
                    ORDER BY date DESC
                        LIMIT :limit \
                    """

            params = {'symbol': symbol.upper(), 'limit': days_back}

            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)

            if df.empty:
                logger.warning(f"⚠️  No data found for {symbol}")
                return pd.DataFrame()

            df = df.sort_values('date', ascending=True).reset_index(drop=True)
            df['date'] = pd.to_datetime(df['date'])

            logger.info(f"📊 {symbol}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching price history: {e}")
            return pd.DataFrame()

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent price for a cryptocurrency"""
        try:
            query = """
                SELECT symbol, close as current_price, open, high, low, volume, date
                FROM crypto_prices
                WHERE symbol = :symbol
                ORDER BY date DESC
                LIMIT 1
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'symbol': symbol.upper()})
                row = result.fetchone()

                if row:
                    return {
                        'symbol': row[0],
                        'current_price': float(row[1]),
                        'open': float(row[2]),
                        'high': float(row[3]),
                        'low': float(row[4]),
                        'volume': float(row[5]),
                        'date': str(row[6])
                    }
            return None

        except Exception as e:
            logger.error(f"Error fetching latest price: {e}")
            return None

    def get_latest_prices_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get latest prices for all cryptocurrencies

        FIXED: Now uses MAX(date) to get only the most recent price for each symbol
        """
        try:
            # ✅ FIXED: Use CTE to get only the latest date for each symbol
            query = """
                WITH latest_dates AS (
                    SELECT symbol, MAX(date) as max_date
                    FROM crypto_prices
                    GROUP BY symbol
                )
                SELECT cp.symbol, cp.close as current_price, cp.open, cp.high, cp.low, cp.volume, cp.date
                FROM crypto_prices cp
                INNER JOIN latest_dates ld 
                    ON cp.symbol = ld.symbol AND cp.date = ld.max_date
                ORDER BY cp.volume DESC
                LIMIT :limit
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'limit': limit})

                data = []
                for row in result:
                    data.append({
                        'symbol': row[0],
                        'current_price': float(row[1]),
                        'open': float(row[2]),
                        'high': float(row[3]),
                        'low': float(row[4]),
                        'volume': float(row[5]),
                        'date': str(row[6])
                    })

                logger.info(f"Retrieved {len(data)} cryptocurrencies (latest prices)")
                return data

        except Exception as e:
            logger.error(f"Error fetching latest prices: {e}")
            return []

    def get_top_movers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get cryptocurrencies with biggest price changes"""
        try:
            query = """
                WITH latest_two AS (
                    SELECT symbol, close, date, 
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn
                    FROM crypto_prices
                )
                SELECT t1.symbol,
                       t1.close as current_price,
                       t2.close as previous_price,
                       ((t1.close - t2.close) / t2.close * 100) as change_percent
                FROM latest_two t1
                LEFT JOIN latest_two t2 ON t1.symbol = t2.symbol AND t2.rn = 2
                WHERE t1.rn = 1
                ORDER BY ABS((t1.close - t2.close) / t2.close * 100) DESC 
                LIMIT :limit
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'limit': limit})

                data = []
                for row in result:
                    data.append({
                        'symbol': row[0],
                        'current_price': float(row[1]) if row[1] else 0,
                        'previous_price': float(row[2]) if row[2] else 0,
                        'change_percent': float(row[3]) if row[3] else 0
                    })

                return data

        except Exception as e:
            logger.error(f"Error fetching top movers: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            query = """
                SELECT COUNT(DISTINCT symbol) as total_cryptos,
                       COUNT(*) as total_records,
                       MAX(date) as latest_date,
                       MIN(date) as earliest_date
                FROM crypto_prices
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()

                return {
                    'total_cryptocurrencies': row[0],
                    'total_records': row[1],
                    'latest_date': str(row[2]),
                    'earliest_date': str(row[3])
                }

        except Exception as e:
            logger.error(f"Error fetching stats: {e}")
            return {}