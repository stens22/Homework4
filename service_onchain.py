"""
On-Chain Metrics Microservice
Port: 5004
Provides blockchain data analysis endpoints
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from datetime import datetime

from strategies import OnChainMetricsStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", None)
CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY", None)

# Initialize strategy
onchain_strategy = OnChainMetricsStrategy(
    glassnode_key=GLASSNODE_API_KEY,
    cryptoquant_key=CRYPTOQUANT_API_KEY
)

logger.info("✅ On-Chain Metrics Microservice initialized")


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'service': 'On-Chain Metrics',
        'status': 'healthy',
        'version': '1.0.0',
        'port': 5004,
        'apis_configured': {
            'glassnode': bool(GLASSNODE_API_KEY),
            'cryptoquant': bool(CRYPTOQUANT_API_KEY),
            'blockchain_com': True  # Free API
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/analyze/<symbol>')
def analyze(symbol):
    """
    Analyze on-chain metrics for a cryptocurrency

    Query params:
    - days_back: Number of days to analyze (default: 30, max: 90)
    """
    try:
        days_back = request.args.get('days_back', 30, type=int)
        days_back = min(days_back, 90)  # Cap at 90 days

        logger.info(f"🔗 On-chain analysis: {symbol}, days_back: {days_back}")

        analysis = onchain_strategy.analyze(symbol, days_back=days_back)

        if analysis.get('success'):
            logger.info(f"✅ Signal: {analysis['data'].get('overall_signal', 'N/A')}")
        else:
            logger.warning(f"❌ Error: {analysis.get('error')}")

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/signal/<symbol>')
def get_signal(symbol):
    """
    Get trading signal based on on-chain metrics

    Query params:
    - days_back: Number of days to analyze (default: 30)
    """
    try:
        days_back = request.args.get('days_back', 30, type=int)

        logger.info(f"🔗 Signal request: {symbol}, days_back: {days_back}")

        signal = onchain_strategy.get_signal(symbol, days_back=days_back)

        return jsonify(signal)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/metrics/<symbol>/<metric_name>')
def get_specific_metric(symbol, metric_name):
    """
    Get a specific on-chain metric

    Available metrics:
    - active_addresses
    - transaction_count
    - exchange_flows
    - whale_movements
    - hash_rate
    - nvt_ratio
    - mvrv_ratio

    Query params:
    - days_back: Number of days (default: 30)
    """
    try:
        days_back = request.args.get('days_back', 30, type=int)

        logger.info(f"🔗 Metric request: {symbol}/{metric_name}")

        # Get full analysis
        analysis = onchain_strategy.analyze(symbol, days_back=days_back)

        if not analysis.get('success'):
            return jsonify(analysis), 500

        # Extract specific metric
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
    Analyze on-chain metrics for multiple cryptocurrencies

    Body: {
        "symbols": ["BTC", "ETH", "LTC"],
        "days_back": 30
    }
    """
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        days_back = data.get('days_back', 30)

        if not symbols:
            return jsonify({
                'success': False,
                'error': 'No symbols provided'
            }), 400

        logger.info(f"🔗 Batch analysis: {len(symbols)} symbols")

        results = {}
        for symbol in symbols[:10]:  # Limit to 10 symbols
            analysis = onchain_strategy.analyze(symbol, days_back=days_back)
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
    print("🔗 ON-CHAIN METRICS MICROSERVICE")
    print("=" * 70)
    print(f"🔧 Service: On-Chain Metrics")
    print(f"🌐 Port: 5004")
    print(f"🔑 Glassnode: {'✅ Configured' if GLASSNODE_API_KEY else '❌ Not configured (using mock data)'}")
    print(f"🔑 CryptoQuant: {'✅ Configured' if CRYPTOQUANT_API_KEY else '❌ Not configured (using mock data)'}")
    print(f"🌐 Blockchain.com: ✅ Free API")
    print("=" * 70)
    print("\nEndpoints:")
    print("  - GET  /health")
    print("  - GET  /analyze/<symbol>?days_back=30")
    print("  - GET  /signal/<symbol>?days_back=30")
    print("  - GET  /metrics/<symbol>/<metric_name>")
    print("  - POST /batch-analyze")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5004)