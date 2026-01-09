"""
Sentiment Analysis Microservice
Port: 5002
Provides sentiment analysis endpoints for cryptocurrency news and social media
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from datetime import datetime

from strategies import SentimentAnalysisStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "e997ed76471d483ab7b54e9a36aef523")
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "7fa89098949bf97ff42275daba737b7f30bb01b6")

# Initialize strategy
sentiment_strategy = SentimentAnalysisStrategy(
    news_api_key=NEWS_API_KEY,
    cryptopanic_key=CRYPTOPANIC_API_KEY
)

logger.info("✅ Sentiment Analysis Microservice initialized")


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'service': 'Sentiment Analysis',
        'status': 'healthy',
        'version': '1.0.0',
        'port': 5002,
        'apis_configured': {
            'newsapi': bool(NEWS_API_KEY),
            'cryptopanic': bool(CRYPTOPANIC_API_KEY)
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/analyze/<symbol>')
def analyze(symbol):
    """
    Analyze sentiment for a cryptocurrency

    Query params:
    - days_back: Number of days to analyze (default: 7, max: 30)
    """
    try:
        days_back = request.args.get('days_back', 7, type=int)
        days_back = min(days_back, 30)  # Cap at 30 days

        logger.info(f"💭 Sentiment analysis request: {symbol}, days_back: {days_back}")

        analysis = sentiment_strategy.analyze(symbol, days_back=days_back)

        if analysis.get('success'):
            logger.info(f"✅ Sentiment: {analysis['data'].get('overall_sentiment', 'N/A')}")
        else:
            logger.warning(f"❌ Error: {analysis.get('error')}")

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/signal/<symbol>')
def get_signal(symbol):
    """
    Get trading signal based on sentiment

    Query params:
    - days_back: Number of days to analyze (default: 7)
    """
    try:
        days_back = request.args.get('days_back', 7, type=int)

        logger.info(f"💭 Signal request: {symbol}, days_back: {days_back}")

        signal = sentiment_strategy.get_signal(symbol, days_back=days_back)

        return jsonify(signal)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze sentiment for multiple cryptocurrencies

    Body: {
        "symbols": ["BTC", "ETH", "SOL"],
        "days_back": 7
    }
    """
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        days_back = data.get('days_back', 7)

        if not symbols:
            return jsonify({
                'success': False,
                'error': 'No symbols provided'
            }), 400

        logger.info(f"💭 Batch analysis: {len(symbols)} symbols")

        results = {}
        for symbol in symbols[:10]:  # Limit to 10 symbols
            analysis = sentiment_strategy.analyze(symbol, days_back=days_back)
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
    print("💭 SENTIMENT ANALYSIS MICROSERVICE")
    print("=" * 70)
    print(f"🔧 Service: Sentiment Analysis")
    print(f"🌐 Port: 5002")
    print(f"📰 NewsAPI: {'✅ Configured' if NEWS_API_KEY else '❌ Not configured'}")
    print(f"🔗 CryptoPanic: {'✅ Configured' if CRYPTOPANIC_API_KEY else '❌ Not configured'}")
    print("=" * 70)
    print("\nEndpoints:")
    print("  - GET  /health")
    print("  - GET  /analyze/<symbol>?days_back=7")
    print("  - GET  /signal/<symbol>?days_back=7")
    print("  - POST /batch-analyze")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5002)