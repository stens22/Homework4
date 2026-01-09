"""
LSTM Prediction Microservice
Port: 5003
Provides LSTM-based price prediction endpoints
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from datetime import datetime

from strategies import LSTMPredictionStrategy
from models import CryptoRepository
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_URL = os.environ.get("DATABASE_URL")

# Initialize repository and strategy
repository = CryptoRepository(DATABASE_URL)
lstm_strategy = LSTMPredictionStrategy(repository=repository)

logger.info("✅ LSTM Prediction Microservice initialized")


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'service': 'LSTM Prediction',
        'status': 'healthy',
        'version': '1.0.0',
        'port': 5003,
        'model_type': 'Bidirectional LSTM',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict/<symbol>')
def predict(symbol):
    """
    Predict future prices using LSTM

    Query params:
    - lookback: Number of days to look back (default: 30, min: 10, max: 365)
    - days_ahead: Number of days to predict (default: 7, min: 1, max: 90)
    """
    try:
        lookback = request.args.get('lookback', 30, type=int)
        days_ahead = request.args.get('days_ahead', 7, type=int)

        # Validate parameters
        if not (10 <= lookback <= 365):
            return jsonify({
                'success': False,
                'error': 'Lookback must be between 10 and 365 days'
            }), 400

        if not (1 <= days_ahead <= 90):
            return jsonify({
                'success': False,
                'error': 'Days ahead must be between 1 and 90'
            }), 400

        logger.info(f"🤖 LSTM prediction: {symbol}, lookback: {lookback}, forecast: {days_ahead}")

        analysis = lstm_strategy.analyze(
            symbol,
            lookback_period=lookback,
            days_ahead=days_ahead
        )

        if analysis.get('success'):
            logger.info(f"✅ Prediction completed for {symbol}")
        else:
            logger.warning(f"❌ Error: {analysis.get('error')}")

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/signal/<symbol>')
def get_signal(symbol):
    """
    Get trading signal based on LSTM prediction

    Query params:
    - lookback: Number of days to look back (default: 30)
    - days_ahead: Number of days to predict (default: 7)
    """
    try:
        lookback = request.args.get('lookback', 30, type=int)
        days_ahead = request.args.get('days_ahead', 7, type=int)

        logger.info(f"🤖 Signal request: {symbol}")

        signal = lstm_strategy.get_signal(
            symbol,
            lookback_period=lookback,
            days_ahead=days_ahead
        )

        return jsonify(signal)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict prices for multiple cryptocurrencies

    Body: {
        "symbols": ["BTC", "ETH"],
        "lookback": 30,
        "days_ahead": 7
    }
    """
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        lookback = data.get('lookback', 30)
        days_ahead = data.get('days_ahead', 7)

        if not symbols:
            return jsonify({
                'success': False,
                'error': 'No symbols provided'
            }), 400

        logger.info(f"🤖 Batch prediction: {len(symbols)} symbols")

        results = {}
        for symbol in symbols[:5]:  # Limit to 5 to avoid timeout
            analysis = lstm_strategy.analyze(
                symbol,
                lookback_period=lookback,
                days_ahead=days_ahead
            )
            results[symbol] = analysis

        return jsonify({
            'success': True,
            'count': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🤖 LSTM PREDICTION MICROSERVICE")
    print("=" * 70)
    print(f"🔧 Service: LSTM Prediction")
    print(f"🌐 Port: 5003")
    print(f"📊 Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'Not configured'}")
    print(f"⚡ Model: Bidirectional LSTM")
    print("=" * 70)
    print("\nEndpoints:")
    print("  - GET  /health")
    print("  - GET  /predict/<symbol>?lookback=30&days_ahead=7")
    print("  - GET  /signal/<symbol>?lookback=30&days_ahead=7")
    print("  - POST /batch-predict")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5003)