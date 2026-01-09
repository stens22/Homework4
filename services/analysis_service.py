"""
Analysis Service - Technical Analysis & LSTM Predictions
Port: 5001
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategies and models
from strategies import TechnicalAnalysisStrategy, LSTMPredictionStrategy
from facade import AnalyzerFactory
from models import CryptoRepository, APIResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_URL = os.environ["DATABASE_URL"]

# Initialize repository
repository = CryptoRepository(DATABASE_URL)

# Initialize factory
factory = AnalyzerFactory({'repository': repository})

# Create strategies
technical_strategy = factory.create_strategy('technical')
lstm_strategy = factory.create_strategy('lstm')

logger.info("✅ Analysis Service initialized")


# ==================== TECHNICAL ANALYSIS ROUTES ====================

@app.route('/api/technical-analysis/<symbol>')
def get_technical_analysis(symbol):
    """
    Get comprehensive technical analysis for a cryptocurrency

    Query params:
    - timeframe: 'short', 'medium', 'long', or 'all' (default: 'all')
    """
    try:
        timeframe = request.args.get('timeframe', 'all')

        logger.info(f"Technical analysis request: {symbol}, timeframe: {timeframe}")

        if timeframe == 'all':
            # Analyze all timeframes
            results = {}
            for tf in ['short', 'medium', 'long']:
                analysis = technical_strategy.analyze(symbol, timeframe=tf)
                if analysis['success']:
                    results[f'1_{"day" if tf == "short" else "week" if tf == "medium" else "month"}'] = analysis['data']

            if not results:
                response = APIResponse.error_response(f"No data found for {symbol}")
                return jsonify(response.to_dict()), 404

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

            if not analysis['success']:
                return jsonify(analysis), 404 if 'not found' in analysis.get('error', '').lower() else 500

            return jsonify(analysis)

    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        response = APIResponse.error_response(str(e))
        return jsonify(response.to_dict()), 500


@app.route('/api/technical-signal/<symbol>')
def get_technical_signal(symbol):
    """Get trading signal from technical analysis"""
    try:
        timeframe = request.args.get('timeframe', 'short')

        logger.info(f"Technical signal request: {symbol}, timeframe: {timeframe}")

        signal = technical_strategy.get_signal(symbol, timeframe=timeframe)

        if not signal['success']:
            return jsonify(signal), 500

        return jsonify(signal)

    except Exception as e:
        logger.error(f"Error getting technical signal: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ==================== LSTM PREDICTION ROUTES ====================

@app.route('/api/lstm-prediction/<symbol>')
def get_lstm_prediction(symbol):
    """
    Get LSTM price prediction

    Query params:
    - lookback: Number of days to look back (default: 30, min: 10, max: 365)
    - days_ahead: Number of days to predict (default: 7, min: 1, max: 90)
    """
    try:
        lookback = request.args.get('lookback', 30, type=int)
        days_ahead = request.args.get('days_ahead', 7, type=int)

        logger.info(f"LSTM prediction request: {symbol}, lookback: {lookback}, days_ahead: {days_ahead}")

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

        # Run LSTM prediction
        analysis = lstm_strategy.analyze(
            symbol,
            lookback_period=lookback,
            days_ahead=days_ahead
        )

        if not analysis['success']:
            return jsonify(analysis), 404 if 'not found' in analysis.get('error', '').lower() else 500

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"Error in LSTM prediction: {e}")
        response = APIResponse.error_response(str(e))
        return jsonify(response.to_dict()), 500


@app.route('/api/lstm-signal/<symbol>')
def get_lstm_signal(symbol):
    """Get trading signal from LSTM prediction"""
    try:
        lookback = request.args.get('lookback', 30, type=int)
        days_ahead = request.args.get('days_ahead', 7, type=int)

        logger.info(f"LSTM signal request: {symbol}")

        signal = lstm_strategy.get_signal(
            symbol,
            lookback_period=lookback,
            days_ahead=days_ahead
        )

        if not signal['success']:
            return jsonify(signal), 500

        return jsonify(signal)

    except Exception as e:
        logger.error(f"Error getting LSTM signal: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ==================== COMBINED ANALYSIS ====================

@app.route('/api/combined-analysis/<symbol>')
def get_combined_analysis(symbol):
    """Get combined technical and LSTM analysis"""
    try:
        logger.info(f"Combined analysis request: {symbol}")

        # Get technical signal
        tech_signal = technical_strategy.get_signal(symbol, timeframe='short')

        # Get LSTM signal
        lstm_signal = lstm_strategy.get_signal(symbol, lookback_period=30, days_ahead=7)

        # Combine results
        combined = {
            'success': True,
            'symbol': symbol.upper(),
            'technical': tech_signal if tech_signal.get('success') else {'error': tech_signal.get('error')},
            'lstm': lstm_signal if lstm_signal.get('success') else {'error': lstm_signal.get('error')},
            'timestamp': datetime.now().isoformat()
        }

        # Calculate overall signal if both succeeded
        if tech_signal.get('success') and lstm_signal.get('success'):
            tech_val = 1 if 'BUY' in tech_signal['signal'] else -1 if 'SELL' in tech_signal['signal'] else 0
            lstm_val = 1 if 'BUY' in lstm_signal['signal'] else -1 if 'SELL' in lstm_signal['signal'] else 0

            combined_val = tech_val + lstm_val

            if combined_val > 1:
                overall = 'STRONG BUY'
            elif combined_val > 0:
                overall = 'BUY'
            elif combined_val < -1:
                overall = 'STRONG SELL'
            elif combined_val < 0:
                overall = 'SELL'
            else:
                overall = 'HOLD'

            combined['overall_signal'] = overall
            combined['confidence'] = (tech_signal['confidence'] + lstm_signal['confidence']) / 2

        return jsonify(combined)

    except Exception as e:
        logger.error(f"Error in combined analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ==================== HEALTH CHECK ====================

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'service': 'Analysis Service',
        'status': 'healthy',
        'strategies': ['technical', 'lstm'],
        'database': 'connected' if repository.engine else 'disconnected',
        'timestamp': datetime.now().isoformat()
    })


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("📈 ANALYSIS SERVICE STARTING")
    print("=" * 70)
    print(f"📊 Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'Not configured'}")
    print(f"🔧 Strategies: Technical Analysis, LSTM Prediction")
    print(f"🌐 Server: http://localhost:5001")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5001)