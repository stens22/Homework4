"""
CryptoPinkPatrol - API Gateway
Microservices Architecture - Cloud Ready
Routes requests to appropriate microservices
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import logging
from datetime import datetime
import requests
from dotenv import load_dotenv


from models.repository import CryptoRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
load_dotenv()
# Configuration - Cloud Ready
DATABASE_URL = os.environ.get("DATABASE_URL")

# Initialize Repository (for database routes)
repository = CryptoRepository(DATABASE_URL)

# Microservices Configuration - Read from Environment for Cloud Deployment
MICROSERVICES = {
    'technical': os.getenv('TECHNICAL_SERVICE_URL', 'http://localhost:5001'),
    'sentiment': os.getenv('SENTIMENT_SERVICE_URL', 'http://localhost:5002'),
    'lstm': os.getenv('LSTM_SERVICE_URL', 'http://localhost:5003'),
    'onchain': os.getenv('ONCHAIN_SERVICE_URL', 'http://localhost:5004')
}

logger.info("✅ API Gateway initialized")
logger.info(f"🌐 Technical Service: {MICROSERVICES['technical']}")
logger.info(f"🌐 Sentiment Service: {MICROSERVICES['sentiment']}")
logger.info(f"🌐 LSTM Service: {MICROSERVICES['lstm']}")
logger.info(f"🌐 On-Chain Service: {MICROSERVICES['onchain']}")


# ==================== HELPER FUNCTIONS ====================

def call_microservice(service_name: str, endpoint: str, method: str = 'GET', **kwargs):
    """
    Call a microservice and return the response

    Args:
        service_name: Name of the microservice ('technical', 'sentiment', etc.)
        endpoint: API endpoint to call
        method: HTTP method (GET, POST, etc.)
        **kwargs: Additional arguments for requests (params, json, etc.)
    """
    try:
        base_url = MICROSERVICES.get(service_name)

        if not base_url:
            return {
                'success': False,
                'error': f'Unknown microservice: {service_name}'
            }, 500

        url = f"{base_url}{endpoint}"
        logger.info(f"🔄 Calling {service_name} microservice: {url}")

        response = requests.request(method, url, timeout=60, **kwargs)

        # Return JSON response
        return response.json(), response.status_code

    except requests.exceptions.Timeout:
        logger.error(f"⏱️ Timeout calling {service_name} microservice")
        return {
            'success': False,
            'error': f'{service_name} service timeout'
        }, 504

    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Cannot connect to {service_name} microservice")
        return {
            'success': False,
            'error': f'{service_name} service unavailable. Make sure it\'s running on {base_url}'
        }, 503

    except Exception as e:
        logger.error(f"❌ Error calling {service_name}: {e}")
        return {
            'success': False,
            'error': str(e)
        }, 500


# ==================== FRONTEND ROUTES ====================

@app.route('/')
def home():
    """Serve login page"""
    try:
        return send_from_directory('.', 'login.html')
    except:
        return jsonify({
            'message': 'CryptoPinkPatrol API Gateway',
            'version': '1.0.0',
            'status': 'running',
            'services': list(MICROSERVICES.keys()),
            'endpoints': {
                'health': '/api/health',
                'technical': '/api/technical-analysis/<symbol>',
                'sentiment': '/api/sentiment/analyze/<symbol>',
                'lstm': '/api/lstm-prediction/<symbol>',
                'onchain': '/api/onchain/analyze/<symbol>',
                'combined': '/api/combined-all/<symbol>'
            }
        })


@app.route('/front.html')
def front():
    """Serve main application page"""
    try:
        return send_from_directory('.', 'front.html')
    except:
        return "<h1>Main App</h1><p>Please ensure front.html is in the same directory</p>"


# ==================== DATABASE ROUTES (Direct Access) ====================

@app.route('/api/latest-prices')
def get_latest_prices():
    """Get latest prices for all cryptocurrencies"""
    try:
        data = repository.get_latest_prices_all(limit=100)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/crypto/<symbol>')
def get_crypto_details(symbol):
    """Get detailed data for specific cryptocurrency"""
    try:
        df = repository.get_price_history(symbol, days_back=30)

        if df.empty:
            return jsonify({'success': False, 'error': f'No data for {symbol}'}), 404

        data = df.to_dict('records')
        return jsonify({'success': True, 'symbol': symbol.upper(), 'data': data})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get database statistics"""
    try:
        stats = repository.get_database_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/top-movers')
def get_top_movers():
    """Get top price movers"""
    try:
        data = repository.get_top_movers(limit=10)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== TECHNICAL ANALYSIS (Microservice Proxy) ====================

@app.route('/api/technical-analysis/<symbol>')
def get_technical_analysis(symbol):
    """Route to Technical Analysis Microservice"""
    timeframe = request.args.get('timeframe', 'all')

    result, status = call_microservice(
        'technical',
        f'/analyze/{symbol}',
        params={'timeframe': timeframe}
    )

    return jsonify(result), status


@app.route('/api/technical-signal/<symbol>')
def get_technical_signal(symbol):
    """Route to Technical Analysis Microservice - Signal"""
    timeframe = request.args.get('timeframe', 'short')

    result, status = call_microservice(
        'technical',
        f'/signal/{symbol}',
        params={'timeframe': timeframe}
    )

    return jsonify(result), status


# ==================== SENTIMENT ANALYSIS (Microservice Proxy) ====================

@app.route('/api/sentiment/analyze/<symbol>')
def analyze_sentiment(symbol):
    """Route to Sentiment Analysis Microservice"""
    days_back = request.args.get('days_back', 7, type=int)

    result, status = call_microservice(
        'sentiment',
        f'/analyze/{symbol}',
        params={'days_back': days_back}
    )

    return jsonify(result), status


@app.route('/api/sentiment/signal/<symbol>')
def get_sentiment_signal(symbol):
    """Route to Sentiment Analysis Microservice - Signal"""
    days_back = request.args.get('days_back', 7, type=int)

    result, status = call_microservice(
        'sentiment',
        f'/signal/{symbol}',
        params={'days_back': days_back}
    )

    return jsonify(result), status


@app.route('/api/sentiment/combined/<symbol>')
def get_combined_sentiment(symbol):
    """Combined sentiment + technical analysis"""
    try:
        days_back = request.args.get('days_back', 7, type=int)

        # Get both signals from microservices
        sentiment_result, _ = call_microservice(
            'sentiment',
            f'/signal/{symbol}',
            params={'days_back': days_back}
        )

        technical_result, _ = call_microservice(
            'technical',
            f'/signal/{symbol}',
            params={'timeframe': 'short'}
        )

        # Combine signals
        sent_val = 1 if 'BUY' in sentiment_result.get('signal', '') else -1 if 'SELL' in sentiment_result.get('signal', '') else 0
        tech_val = 1 if 'BUY' in technical_result.get('signal', '') else -1 if 'SELL' in technical_result.get('signal', '') else 0

        combined_val = sent_val + tech_val

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

        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'overall_recommendation': overall,
            'sentiment_analysis': sentiment_result,
            'technical_analysis': technical_result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== LSTM PREDICTION (Microservice Proxy) ====================

@app.route('/api/lstm-prediction/<symbol>')
def get_lstm_prediction(symbol):
    """Route to LSTM Prediction Microservice"""
    lookback = request.args.get('lookback', 30, type=int)
    days_ahead = request.args.get('days_ahead', 7, type=int)

    result, status = call_microservice(
        'lstm',
        f'/predict/{symbol}',
        params={'lookback': lookback, 'days_ahead': days_ahead}
    )

    return jsonify(result), status


@app.route('/api/lstm-signal/<symbol>')
def get_lstm_signal(symbol):
    """Route to LSTM Prediction Microservice - Signal"""
    lookback = request.args.get('lookback', 30, type=int)
    days_ahead = request.args.get('days_ahead', 7, type=int)

    result, status = call_microservice(
        'lstm',
        f'/signal/{symbol}',
        params={'lookback': lookback, 'days_ahead': days_ahead}
    )

    return jsonify(result), status


# ==================== ON-CHAIN METRICS (Microservice Proxy) ====================

@app.route('/api/onchain/analyze/<symbol>')
def analyze_onchain(symbol):
    """Route to On-Chain Metrics Microservice"""
    days_back = request.args.get('days_back', 30, type=int)

    result, status = call_microservice(
        'onchain',
        f'/analyze/{symbol}',
        params={'days_back': days_back}
    )

    return jsonify(result), status


@app.route('/api/onchain/signal/<symbol>')
def get_onchain_signal(symbol):
    """Route to On-Chain Metrics Microservice - Signal"""
    days_back = request.args.get('days_back', 30, type=int)

    result, status = call_microservice(
        'onchain',
        f'/signal/{symbol}',
        params={'days_back': days_back}
    )

    return jsonify(result), status


@app.route('/api/combined-all/<symbol>')
def get_combined_all(symbol):
    """Combined Technical + Sentiment + On-Chain analysis"""
    try:
        logger.info(f"🎯 Combined analysis request for {symbol}")

        # Get all signals from microservices
        technical_result, _ = call_microservice('technical', f'/signal/{symbol}', params={'timeframe': 'short'})
        sentiment_result, _ = call_microservice('sentiment', f'/signal/{symbol}', params={'days_back': 7})
        onchain_result, _ = call_microservice('onchain', f'/signal/{symbol}', params={'days_back': 30})

        # Calculate combined signal
        tech_val = 1 if 'BUY' in technical_result.get('signal', '') else -1 if 'SELL' in technical_result.get('signal', '') else 0
        sent_val = 1 if 'BUY' in sentiment_result.get('signal', '') else -1 if 'SELL' in sentiment_result.get('signal', '') else 0
        onch_val = 1 if 'BUY' in onchain_result.get('signal', '') else -1 if 'SELL' in onchain_result.get('signal', '') else 0

        combined_val = tech_val + sent_val + onch_val

        if combined_val >= 2:
            overall = 'STRONG BUY'
        elif combined_val > 0:
            overall = 'BUY'
        elif combined_val <= -2:
            overall = 'STRONG SELL'
        elif combined_val < 0:
            overall = 'SELL'
        else:
            overall = 'HOLD'

        # Calculate average confidence
        confidences = []
        if technical_result.get('success'):
            confidences.append(technical_result.get('confidence', 50))
        if sentiment_result.get('success'):
            confidences.append(sentiment_result.get('confidence', 50))
        if onchain_result.get('success'):
            confidences.append(onchain_result.get('confidence', 50))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 50

        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'overall_recommendation': overall,
            'confidence': round(avg_confidence, 2),
            'technical_analysis': technical_result,
            'sentiment_analysis': sentiment_result,
            'onchain_analysis': onchain_result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in combined analysis: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== HEALTH CHECK ====================

@app.route('/api/health')
def health_check():
    """Health check for API Gateway and all microservices"""
    services_status = {}

    # Check each microservice
    for service_name, base_url in MICROSERVICES.items():
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            services_status[service_name] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'url': base_url
            }
        except:
            services_status[service_name] = {
                'status': 'unavailable',
                'url': base_url
            }

    all_healthy = all(s['status'] == 'healthy' for s in services_status.values())

    return jsonify({
        'success': True,
        'service': 'API Gateway',
        'status': 'healthy' if all_healthy else 'degraded',
        'database': 'connected' if repository.engine else 'disconnected',
        'microservices': services_status,
        'timestamp': datetime.now().isoformat()
    })


# ==================== MICROSERVICES STATUS ====================

@app.route('/api/services/status')
def services_status():
    """Get detailed status of all microservices"""
    services = {}

    for service_name, base_url in MICROSERVICES.items():
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                services[service_name] = response.json()
            else:
                services[service_name] = {
                    'status': 'unhealthy',
                    'error': f'HTTP {response.status_code}'
                }
        except Exception as e:
            services[service_name] = {
                'status': 'unavailable',
                'error': str(e)
            }

    return jsonify({
        'success': True,
        'services': services,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))

    print("\n" + "=" * 70)
    print("🚀 CryptoPinkPatrol - API GATEWAY")
    print("=" * 70)
    print(f"🏗️  Architecture: Microservices")
    print(f"📊 Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'Not configured'}")
    print(f"🌐 Gateway Port: {port}")
    print("=" * 70)
    print("\nMicroservices Configuration:")
    for service, url in MICROSERVICES.items():
        print(f"  📡 {service.capitalize():12} -> {url}")
    print("=" * 70)
    print("\n⚠️  IMPORTANT: Make sure all microservices are running!")
    print("   Run each in a separate terminal:")
    print("   - python service_technical.py")
    print("   - python service_sentiment.py")
    print("   - python service_lstm.py")
    print("   - python service_onchain.py")
    print("=" * 70 + "\n")

    app.run(debug=False, host='0.0.0.0', port=port)