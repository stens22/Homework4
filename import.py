"""
CSV Import Script for CryptoPinkPatrol (IMPROVED)
Automatically detects CSV format and imports data
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import sys

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:569Stanche$@cryptopink_db:5432/crypto"
)
engine = create_engine(DATABASE_URL, echo=True, future=True)

with Session(engine) as session:
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS crypto_prices (
            symbol VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            volume NUMERIC,
            PRIMARY KEY (symbol, date)
        );
    """))
    session.commit()

print("✅ crypto_prices table ensured")

# ⚠️ IMPORTANT: Replace YOUR_PASSWORD with your actual PostgreSQL password!

def import_csv(csv_file_path):
    """Import cryptocurrency data from CSV file to PostgreSQL"""

    print("=" * 60)
    print("📊 CryptoPinkPatrol - CSV Data Import")
    print("=" * 60)
    print(f"\n📁 Reading CSV file: {csv_file_path}")

    try:
        # Read CSV file - pandas will auto-detect separator
        df = pd.read_csv(csv_file_path)
        print(f"✅ CSV loaded successfully")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {', '.join(df.columns)}")

    except FileNotFoundError:
        print(f"❌ Error: File not found: {csv_file_path}")
        print("\n💡 Make sure the file path is correct")
        return
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print(f"\n📊 Data Preview:")
    print(df.head(3).to_string())

    # Check required columns
    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"\n❌ Missing required columns: {', '.join(missing_cols)}")
        print(f"\n💡 Available columns: {', '.join(df.columns)}")
        return

    print(f"\n✅ All required columns present")

    # Connect to database
    print(f"\n🔌 Connecting to database...")
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n💡 Make sure:")
        print("   1. PostgreSQL is running")
        print("   2. Database 'crypto' exists")
        print("   3. Table 'crypto_prices' exists")
        print("   4. Password in DATABASE_URL is correct")
        print(f"\n   Current DATABASE_URL: {DATABASE_URL}")
        return

    # Clean and prepare data
    print(f"\n🧹 Cleaning and preparing data...")

    # Select only needed columns
    df_clean = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Convert date - try different formats
    date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']
    date_converted = False

    for date_format in date_formats:
        try:
            df_clean['date'] = pd.to_datetime(df_clean['date'], format=date_format)
            print(f"   ✅ Date format detected: {date_format}")
            date_converted = True
            break
        except:
            continue

    if not date_converted:
        # Let pandas try to figure it out
        try:
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            print(f"   ✅ Date format auto-detected")
        except:
            print(f"   ❌ Could not parse dates. Please check date format.")
            return

    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Remove rows with missing data
    before_count = len(df_clean)
    df_clean = df_clean.dropna()
    after_count = len(df_clean)

    if before_count > after_count:
        print(f"   ⚠️  Removed {before_count - after_count:,} rows with missing data")

    print(f"✅ Data cleaned: {len(df_clean):,} rows ready to import")

    # Get unique symbols
    symbols = df_clean['symbol'].unique()
    print(f"\n💰 Cryptocurrencies to import: {len(symbols)}")

    # Show top 10 symbols by record count
    symbol_counts = df_clean['symbol'].value_counts()
    print("\n   Top symbols by record count:")
    for symbol in symbol_counts.head(10).index:
        count = symbol_counts[symbol]
        print(f"   {symbol}: {count:,} records")

    if len(symbols) > 10:
        print(f"   ... and {len(symbols) - 10} more")

    # Import data
    print(f"\n📥 Importing data to database...")
    print(f"   This may take a while for {len(df_clean):,} rows...")

    try:
        with engine.connect() as conn:
            # Check for duplicates before importing
            existing_check = conn.execute(text("""
                SELECT COUNT(*) FROM crypto_prices
            """))
            existing_count = existing_check.fetchone()[0]

            if existing_count > 0:
                print(f"   ℹ️  Database already has {existing_count:,} records")
                response = input("   Continue? This will skip duplicates. (y/n): ").lower()
                if response != 'y':
                    print("   Import cancelled")
                    return

            imported = 0
            skipped = 0
            errors = 0

            # Import in batches for better performance
            batch_size = 1000
            total_batches = (len(df_clean) + batch_size - 1) // batch_size

            print(f"   Importing in batches of {batch_size}...")

            for i in range(0, len(df_clean), batch_size):
                batch = df_clean.iloc[i:i + batch_size]
                batch_num = (i // batch_size) + 1

                for idx, row in batch.iterrows():
                    try:
                        query = text("""
                            INSERT INTO crypto_prices (symbol, date, open, high, low, close, volume)
                            VALUES (:symbol, :date, :open, :high, :low, :close, :volume)
                        """)

                        conn.execute(query, {
                            'symbol': row['symbol'],
                            'date': row['date'].date(),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume'])
                        })

                        imported += 1

                    except Exception as e:
                        if "duplicate key value" in str(e).lower():
                            skipped += 1
                        else:
                            errors += 1
                            if errors <= 3:  # Show first 3 errors only
                                print(f"      ⚠️  Error: {str(e)[:100]}")

                # Commit after each batch
                conn.commit()

                # Show progress
                percent = (batch_num / total_batches) * 100
                print(
                    f"   Progress: {batch_num}/{total_batches} batches ({percent:.1f}%) - Imported: {imported:,}, Skipped: {skipped:,}")

            print(f"\n✅ Import complete!")
            print(f"   Imported: {imported:,} rows")
            if skipped > 0:
                print(f"   Skipped (duplicates): {skipped:,} rows")
            if errors > 0:
                print(f"   Errors: {errors} rows")

        # Show final database summary
        print(f"\n📊 Database Summary:")
        print("-" * 60)

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    symbol, 
                    COUNT(*) as records,
                    MIN(date) as first_date,
                    MAX(date) as last_date
                FROM crypto_prices
                GROUP BY symbol
                ORDER BY records DESC
                LIMIT 20
            """))

            print(f"{'Symbol':<12} {'Records':<10} {'From':<12} {'To':<12}")
            print("-" * 60)

            total_records = 0
            ready_symbols = []

            for row in result:
                print(f"{row[0]:<12} {row[1]:<10} {str(row[2]):<12} {str(row[3]):<12}")
                total_records += row[1]
                if row[1] >= 20:
                    ready_symbols.append(row[0])

            # Get total count
            total_result = conn.execute(text("SELECT COUNT(*) FROM crypto_prices"))
            total_all = total_result.fetchone()[0]

            # Get total symbols
            symbols_result = conn.execute(text("SELECT COUNT(DISTINCT symbol) FROM crypto_prices"))
            total_symbols = symbols_result.fetchone()[0]

            print("-" * 60)
            print(f"{'TOTAL':<12} {total_all:<10} ({total_symbols} symbols)")
            print()

            print(f"✅ Symbols ready for technical analysis (20+ records): {len(ready_symbols)}")
            if len(ready_symbols) <= 20:
                print(f"   {', '.join(ready_symbols)}")
            else:
                print(f"   {', '.join(ready_symbols[:20])} ... and {len(ready_symbols) - 20} more")

    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("🎉 Import Successful!")
    print("=" * 60)
    print("\n✅ You can now:")
    print("   1. Start your Flask app: python app.py")
    print("   2. Test the API:")
    if ready_symbols:
        print(f"      curl http://127.0.0.1:5000/api/technical-analysis/{ready_symbols[0]}")
    print("   3. Open the frontend and use Technical Analysis page")


if __name__ == '__main__':
    print("\n🔬 CryptoPinkPatrol - CSV Import Tool\n")

    # Check command line arguments
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Ask for file path
        print("Enter the path to your CSV file:")
        print("(Example: data.csv or C:\\Users\\YourName\\Downloads\\crypto_data.csv)")
        csv_file = input("\nFile path: ").strip().strip('"')

    if not csv_file:
        print("❌ No file specified")
        print("\nUsage: python import_csv_data_v2.py your_data.csv")
        sys.exit(1)

    # Run import
    import_csv(csv_file)