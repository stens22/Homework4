"""
Sentiment Service - Sentiment Analysis & News
Port: 5002
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategies
from strategies import SentimentAnalysisStrategy, OnChainMetricsStrategy
from facade import AnalyzerFactory
from models import APIResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "e997ed76471d483ab7b54e9a36aef523")
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "7fa89098949bf97ff42275daba737b7f30bb01b6")
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", None)
CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY", None)

# Initialize factory
factory = AnalyzerFactory({
    'news_api_key': NEWS_API_KEY,
    'cryptopanic_api_key': CRYPTOPANIC_API_KEY,
    'glassnode_api_key': GLASSNODE_API_KEY,
    'cryptoquant_api_key': CRYPTOQUANT_API_KEY
})

# Create strategies
sentiment_strategy = factory.create_strategy('sentiment')
onchain_strategy = factory.create_strategy('onchain')

logger.info("✅ Sentiment Service initialized")


# ==================== SENTIMENT ANALYSIS ROUTES ====================

@app.route('/api/sentiment/analyze/<symbol>')
def analyze_sentiment(symbol):
    """
    Get comprehensive sentiment analysis

    Query params:
    - days_back: Number of days to look back (default: 7)
    """
    try:
        days_back = request.args.get('days_back', 7, type=int)

        logger.info(f"Sentiment analysis request: {symbol}, days_back: {days_back}")

        # Validate parameters
        if not (1 <= days_back <= 30):
            return jsonify({
                'success': False,
                'error': 'days_back must be between 1 and 30'
            }), 400

        analysis = sentiment_strategy.analyze(symbol, days_back=days_back)

        if not analysis['success']:
            return jsonify(analysis), 500

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        response = APIResponse.error_response(str(e))
        return jsonify(response.to_dict()), 500


@app.route('/api/sentiment/signal/<symbol>')
def get_sentiment_signal(symbol):
    """Get trading signal based on sentiment"""
    try:
        days_back = request.args.get('days_back', 7, type=int)

        logger.info(f"Sentiment signal request: {symbol}")

        signal = sentiment_strategy.get_signal(symbol, days_back=days_back)

        if not signal['success']:
            return jsonify(signal), 500

        return jsonify(signal)

    except Exception as e:
        logger.error(f"Error getting sentiment signal: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/sentiment/text', methods=['POST'])
def analyze_text():
    """
    Analyze sentiment of custom text

    Body:
    {
        "text": "Bitcoin price is soaring!"
    }
    """
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400

        text = data['text']

        if not text or len(text.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Text cannot be empty'
            }), 400

        result = sentiment_strategy._analyze_text_sentiment(text)

        return jsonify({
            'success': True,
            'text': text,
            'analysis': result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/sentiment/compare')
def compare_sentiment():
    """
    Compare sentiment across multiple cryptocurrencies

    Query params:
    - symbols: Comma-separated list (default: BTC,ETH,BNB)
    - days_back: Number of days (default: 7)
    """
    try:
        symbols_param = request.args.get('symbols', 'BTCUSDT,ETHUSDT,BNBUSDT')
        symbols = [s.strip().upper() for s in symbols_param.split(',')]
        days_back = request.args.get('days_back', 7, type=int)

        if len(symbols) > 10:
            return jsonify({
                'success': False,
                'error': 'Maximum 10 symbols allowed'
            }), 400

        logger.info(f"Sentiment comparison request: {symbols}")

        results = []
        for symbol in symbols:
            try:
                sentiment = sentiment_strategy.analyze(symbol, days_back=days_back)

                if sentiment.get('success'):
                    data = sentiment['data']
                    results.append({
                        'symbol': symbol,
                        'overall_sentiment': data.get('overall_sentiment', 'NEUTRAL'),
                        'average_polarity': data.get('average_polarity', 0),
                        'confidence': data.get('confidence', 0),
                        'total_articles': data.get('total_articles', 0)
                    })
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        # Sort by polarity (most bullish first)
        results.sort(key=lambda x: x['average_polarity'], reverse=True)

        return jsonify({
            'success': True,
            'comparison': results,
            'most_bullish': results[0]['symbol'] if results else None,
            'most_bearish': results[-1]['symbol'] if results else None,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in sentiment comparison: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ==================== ON-CHAIN ANALYSIS ROUTES ====================

@app.route('/api/onchain/analyze/<symbol>')
def analyze_onchain(symbol):
    """
    Get on-chain metrics analysis

    Query params:
    - days_back: Number of days (default: 30)
    """
    try:
        days_back = request.args.get('days_back', 30, type=int)

        logger.info(f"On-chain analysis request: {symbol}, days_back: {days_back}")

        analysis = onchain_strategy.analyze(symbol, days_back=days_back)

        if not analysis['success']:
            return jsonify(analysis), 500

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"Error in on-chain analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/onchain/signal/<symbol>')
def get_onchain_signal(symbol):
    """Get trading signal from on-chain metrics"""
    try:
        days_back = request.args.get('days_back', 30, type=int)

        logger.info(f"On-chain signal request: {symbol}")

        signal = onchain_strategy.get_signal(symbol, days_back=days_back)

        if not signal['success']:
            return jsonify(signal), 500

        return jsonify(signal)

    except Exception as e:
        logger.error(f"Error getting on-chain signal: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ==================== COMBINED ANALYSIS ====================

@app.route('/api/sentiment/combined/<symbol>')
def get_combined_analysis(symbol):
    """
    Get combined sentiment and on-chain analysis

    Query params:
    - days_back: Number of days (default: 7)
    """
    try:
        days_back = request.args.get('days_back', 7, type=int)

        logger.info(f"Combined sentiment analysis request: {symbol}")

        # Get sentiment signal
        sentiment_signal = sentiment_strategy.get_signal(symbol, days_back=days_back)

        # Get on-chain signal
        onchain_signal = onchain_strategy.get_signal(symbol, days_back=days_back)

        # Combine results
        combined = {
            'success': True,
            'symbol': symbol.upper(),
            'sentiment_analysis': sentiment_signal if sentiment_signal.get('success') else {
                'error': sentiment_signal.get('error')},
            'onchain_analysis': onchain_signal if onchain_signal.get('success') else {
                'error': onchain_signal.get('error')},
            'timestamp': datetime.now().isoformat()
        }

        # Calculate overall recommendation
        if sentiment_signal.get('success') and onchain_signal.get('success'):
            sent_val = 1 if 'BUY' in sentiment_signal['signal'] else -1 if 'SELL' in sentiment_signal['signal'] else 0
            chain_val = 1 if 'BUY' in onchain_signal['signal'] else -1 if 'SELL' in onchain_signal['signal'] else 0

            combined_val = sent_val + chain_val

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

            combined['overall_recommendation'] = overall
            combined['confidence'] = f"Based on sentiment and on-chain metrics"
            combined['individual_signals'] = {
                'sentiment': sentiment_signal['signal'],
                'onchain': onchain_signal['signal']
            }

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
        'service': 'Sentiment Service',
        'status': 'healthy',
        'strategies': ['sentiment', 'onchain'],
        'apis_configured': {
            'newsapi': NEWS_API_KEY is not None,
            'cryptopanic': CRYPTOPANIC_API_KEY is not None,
            'glassnode': GLASSNODE_API_KEY is not None,
            'cryptoquant': CRYPTOQUANT_API_KEY is not None
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/health/sentiment')
def sentiment_health():
    """Sentiment service specific health check"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'service': 'sentiment_analysis',
        'apis_configured': {
            'newsapi': sentiment_strategy.news_api_key is not None,
            'cryptopanic': sentiment_strategy.cryptopanic_key is not None
        },
        'cache_size': len(sentiment_strategy.cache),
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
    print("💭 SENTIMENT SERVICE STARTING")
    print("=" * 70)
    print(f"📰 NewsAPI: {'✅ Configured' if NEWS_API_KEY else '❌ Not configured'}")
    print(f"💬 CryptoPanic: {'✅ Configured' if CRYPTOPANIC_API_KEY else '❌ Not configured'}")
    print(f"🔗 Glassnode: {'✅ Configured' if GLASSNODE_API_KEY else '❌ Not configured'}")
    print(f"📊 CryptoQuant: {'✅ Configured' if CRYPTOQUANT_API_KEY else '❌ Not configured'}")
    print(f"🌐 Server: http://localhost:5002")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5002)