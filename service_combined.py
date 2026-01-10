"""
Combined Sentiment + On-Chain Analysis Microservice
Port: 5002
Routes based on 'type' query parameter
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from datetime import datetime

from strategies import SentimentAnalysisStrategy, OnChainMetricsStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "e997ed76471d483ab7b54e9a36aef523")
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "7fa89098949bf97ff42275daba737b7f30bb01b6")
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", None)
CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY", None)

# Initialize strategies
sentiment_strategy = SentimentAnalysisStrategy(
    news_api_key=NEWS_API_KEY,
    cryptopanic_key=CRYPTOPANIC_API_KEY
)

onchain_strategy = OnChainMetricsStrategy(
    glassnode_key=GLASSNODE_API_KEY,
    cryptoquant_key=CRYPTOQUANT_API_KEY
)

logger.info("✅ Combined Sentiment + On-Chain Microservice initialized")


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'service': 'Combined Sentiment + On-Chain',
        'status': 'healthy',
        'version': '2.0.0',
        'port': 5002,
        'features': ['sentiment', 'onchain'],
        'apis_configured': {
            'newsapi': bool(NEWS_API_KEY),
            'cryptopanic': bool(CRYPTOPANIC_API_KEY),
            'glassnode': bool(GLASSNODE_API_KEY),
            'cryptoquant': bool(CRYPTOQUANT_API_KEY)
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/analyze/<symbol>')
def analyze(symbol):
    """
    Unified analysis endpoint - routes based on 'type' parameter

    Query params:
    - type: 'sentiment' or 'onchain' (default: 'sentiment')
    - days_back: Number of days to analyze

    Examples:
    - /analyze/BTC?type=sentiment&days_back=7
    - /analyze/BTC?type=onchain&days_back=30
    """
    analysis_type = request.args.get('type', 'sentiment').lower()

    if analysis_type == 'sentiment':
        return analyze_sentiment(symbol)
    elif analysis_type == 'onchain':
        return analyze_onchain(symbol)
    else:
        return jsonify({
            'success': False,
            'error': f"Invalid type '{analysis_type}'. Use 'sentiment' or 'onchain'"
        }), 400


def analyze_sentiment(symbol):
    """Sentiment analysis"""
    try:
        days_back = request.args.get('days_back', 7, type=int)
        days_back = min(days_back, 30)

        logger.info(f"💭 Sentiment analysis: {symbol}, days_back: {days_back}")

        analysis = sentiment_strategy.analyze(symbol, days_back=days_back)

        if analysis.get('success'):
            logger.info(f"✅ Sentiment: {analysis['data'].get('overall_sentiment', 'N/A')}")
        else:
            logger.warning(f"❌ Error: {analysis.get('error')}")

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"❌ Sentiment error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def analyze_onchain(symbol):
    """On-chain analysis"""
    try:
        days_back = request.args.get('days_back', 30, type=int)
        days_back = min(days_back, 90)

        logger.info(f"🔗 On-chain analysis: {symbol}, days_back: {days_back}")

        analysis = onchain_strategy.analyze(symbol, days_back=days_back)

        if analysis.get('success'):
            logger.info(f"✅ Signal: {analysis['data'].get('overall_signal', 'N/A')}")
        else:
            logger.warning(f"❌ Error: {analysis.get('error')}")

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"❌ On-chain error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/signal/<symbol>')
def get_signal(symbol):
    """
    Get trading signal - routes based on 'type' parameter

    Query params:
    - type: 'sentiment' or 'onchain' (default: 'sentiment')
    - days_back: Number of days to analyze
    """
    analysis_type = request.args.get('type', 'sentiment').lower()
    days_back = request.args.get('days_back', 7 if analysis_type == 'sentiment' else 30, type=int)

    try:
        logger.info(f"📊 Signal request: {symbol}, type: {analysis_type}")

        if analysis_type == 'sentiment':
            signal = sentiment_strategy.get_signal(symbol, days_back=days_back)
        elif analysis_type == 'onchain':
            signal = onchain_strategy.get_signal(symbol, days_back=days_back)
        else:
            return jsonify({
                'success': False,
                'error': f"Invalid type '{analysis_type}'"
            }), 400

        return jsonify(signal)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/metrics/<symbol>/<metric_name>')
def get_specific_metric(symbol, metric_name):
    """
    Get a specific on-chain metric

    Available metrics:
    - active_addresses, transaction_count, exchange_flows
    - whale_movements, hash_rate, nvt_ratio, mvrv_ratio
    """
    try:
        days_back = request.args.get('days_back', 30, type=int)

        logger.info(f"🔗 Metric request: {symbol}/{metric_name}")

        analysis = onchain_strategy.analyze(symbol, days_back=days_back)

        if not analysis.get('success'):
            return jsonify(analysis), 500

        metrics = analysis['data']['metrics']

        if metric_name not in metrics:
            return jsonify({
                'success': False,
                'error': f'Metric "{metric_name}" not found',
                'available_metrics': list(metrics.keys())
            }), 404

        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'metric': metric_name,
            'data': metrics[metric_name],
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Batch analysis - routes based on 'type' parameter

    Body: {
        "symbols": ["BTC", "ETH", "SOL"],
        "type": "sentiment",  // or "onchain"
        "days_back": 7
    }
    """
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        analysis_type = data.get('type', 'sentiment').lower()
        days_back = data.get('days_back', 7 if analysis_type == 'sentiment' else 30)

        if not symbols:
            return jsonify({
                'success': False,
                'error': 'No symbols provided'
            }), 400

        logger.info(f"📊 Batch {analysis_type} analysis: {len(symbols)} symbols")

        results = {}
        for symbol in symbols[:10]:
            if analysis_type == 'sentiment':
                analysis = sentiment_strategy.analyze(symbol, days_back=days_back)
            elif analysis_type == 'onchain':
                analysis = onchain_strategy.analyze(symbol, days_back=days_back)
            else:
                return jsonify({
                    'success': False,
                    'error': f"Invalid type '{analysis_type}'"
                }), 400

            results[symbol] = analysis

        return jsonify({
            'success': True,
            'type': analysis_type,
            'count': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🔗💭 COMBINED SENTIMENT + ON-CHAIN MICROSERVICE")
    print("=" * 70)
    print(f"🔧 Service: Combined Analysis")
    print(f"🌐 Port: 5002")
    print(f"📰 NewsAPI: {'✅' if NEWS_API_KEY else '❌'}")
    print(f"🔗 CryptoPanic: {'✅' if CRYPTOPANIC_API_KEY else '❌'}")
    print(f"🔑 Glassnode: {'✅' if GLASSNODE_API_KEY else '❌ (using mock data)'}")
    print(f"🔑 CryptoQuant: {'✅' if CRYPTOQUANT_API_KEY else '❌ (using mock data)'}")
    print("=" * 70)
    print("\nEndpoints:")
    print("  - GET  /health")
    print("  - GET  /analyze/<symbol>?type=sentiment&days_back=7")
    print("  - GET  /analyze/<symbol>?type=onchain&days_back=30")
    print("  - GET  /signal/<symbol>?type=sentiment")
    print("  - GET  /metrics/<symbol>/<metric_name>")
    print("  - POST /batch-analyze")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5002)