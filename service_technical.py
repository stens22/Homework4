"""
Technical Analysis Microservice
Port: 5001
Provides technical analysis endpoints for cryptocurrency data
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from datetime import datetime
from dotenv import load_dotenv  # ← ADDED: Load .env file

from strategies import TechnicalAnalysisStrategy
from models import CryptoRepository

# ✅ FIXED: Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration - Now reads from .env
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    logger.error("❌ DATABASE_URL not found in environment variables!")
    logger.error("Make sure you have a .env file with DATABASE_URL set")
    exit(1)

# Initialize repository and strategy
repository = CryptoRepository(DATABASE_URL)
technical_strategy = TechnicalAnalysisStrategy(repository=repository)

logger.info("✅ Technical Analysis Microservice initialized")


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'service': 'Technical Analysis',
        'status': 'healthy',
        'version': '1.0.0',
        'port': 5001,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/analyze/<symbol>')
def analyze(symbol):
    """
    Analyze cryptocurrency with technical indicators

    Query params:
    - timeframe: 'short', 'medium', 'long', or 'all' (default: 'all')
    """
    try:
        timeframe = request.args.get('timeframe', 'all')

        logger.info(f"📊 Technical analysis request: {symbol}, timeframe: {timeframe}")

        if timeframe == 'all':
            # Analyze all timeframes
            results = {}
            for tf in ['short', 'medium', 'long']:
                logger.info(f"   Analyzing {tf} timeframe...")
                analysis = technical_strategy.analyze(symbol, timeframe=tf)

                timeframe_key = f'1_{"day" if tf == "short" else "week" if tf == "medium" else "month"}'

                if analysis.get('success'):
                    results[timeframe_key] = analysis['data']
                    logger.info(f"   ✅ {timeframe_key}: {analysis['data']['summary']['recommendation']}")
                else:
                    error_msg = analysis.get('error', 'Analysis failed')
                    results[timeframe_key] = {'error': error_msg}
                    logger.warning(f"   ❌ {timeframe_key}: {error_msg}")

            logger.info(f"📈 Completed analysis for {symbol}")

            return jsonify({
                'success': True,
                'data': {
                    'symbol': symbol.upper(),
                    'timeframes': results,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            })
        else:
            # Single timeframe
            analysis = technical_strategy.analyze(symbol, timeframe=timeframe)
            return jsonify(analysis)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/signal/<symbol>')
def get_signal(symbol):
    """
    Get trading signal based on technical analysis

    Query params:
    - timeframe: 'short', 'medium', or 'long' (default: 'short')
    """
    try:
        timeframe = request.args.get('timeframe', 'short')

        logger.info(f"📊 Signal request: {symbol}, timeframe: {timeframe}")

        signal = technical_strategy.get_signal(symbol, timeframe=timeframe)

        return jsonify(signal)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/indicators/<symbol>')
def get_indicators(symbol):
    """
    Get raw indicator values (for charting)

    Query params:
    - days_back: Number of days of data (default: 100)
    """
    try:
        days_back = request.args.get('days_back', 100, type=int)

        # Get data
        df = repository.get_price_history(symbol, days_back)

        if df.empty:
            return jsonify({
                'success': False,
                'error': f'No data found for {symbol}'
            }), 404

        # Calculate indicators
        from strategies.technical_strategy import TechnicalAnalysisStrategy
        ta = TechnicalAnalysisStrategy(repository)

        # Get raw indicator data
        result = []
        for idx in df.index[-30:]:  # Last 30 data points
            result.append({
                'date': str(df.loc[idx, 'date']),
                'close': float(df.loc[idx, 'close']),
            })

        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'data': result
        })

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("📊 TECHNICAL ANALYSIS MICROSERVICE")
    print("=" * 70)
    print(f"🔧 Service: Technical Analysis")
    print(f"🌐 Port: 5001")
    print(f"📊 Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'Not configured'}")
    print("=" * 70)
    print("\nEndpoints:")
    print("  - GET  /health")
    print("  - GET  /analyze/<symbol>?timeframe=all|short|medium|long")
    print("  - GET  /signal/<symbol>?timeframe=short|medium|long")
    print("  - GET  /indicators/<symbol>?days_back=100")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5001)