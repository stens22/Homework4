"""
On-Chain Metrics Strategy
Analyzes blockchain data for trading signals
"""
from .base_strategy import AnalysisStrategy
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OnChainMetricsStrategy(AnalysisStrategy):
    """
    Strategy for on-chain metrics analysis
    Analyzes blockchain data: active addresses, whale movements, exchange flows, etc.
    """

    def __init__(self, glassnode_key: Optional[str] = None,
                 cryptoquant_key: Optional[str] = None):
        """
        Initialize on-chain metrics strategy

        Args:
            glassnode_key: API key for Glassnode
            cryptoquant_key: API key for CryptoQuant
        """
        super().__init__("OnChainMetrics")
        self.glassnode_key = glassnode_key
        self.cryptoquant_key = cryptoquant_key

    def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive on-chain analysis

        Args:
            symbol: Cryptocurrency symbol
            days_back: Days of history to analyze (default: 30)

        Returns:
            On-chain analysis results
        """
        try:
            if not self.validate_symbol(symbol):
                return self.format_response(
                    False,
                    error=f"Invalid symbol: {symbol}",
                    symbol=symbol
                )

            days_back = kwargs.get('days_back', 30)

            # Gather on-chain metrics
            metrics = {
                'active_addresses': self._get_active_addresses(symbol, days_back),
                'transaction_count': self._get_transaction_count(symbol, days_back),
                'exchange_flows': self._get_exchange_flows(symbol, days_back),
                'whale_movements': self._get_whale_movements(symbol, days_back),
                'hash_rate': self._get_hash_rate(symbol, days_back),
                'nvt_ratio': self._get_nvt_ratio(symbol, days_back),
                'mvrv_ratio': self._get_mvrv_ratio(symbol, days_back)
            }

            # Calculate overall signal
            overall_signal = self._calculate_overall_signal(metrics)

            result = {
                'metrics': metrics,
                'overall_signal': overall_signal,
                'analysis_timestamp': datetime.now().isoformat()
            }

            self.log_analysis(symbol, f"On-chain signal: {overall_signal}")

            return self.format_response(
                True,
                data=result,
                symbol=symbol
            )

        except Exception as e:
            self.logger.error(f"Error in on-chain analysis: {e}")
            return self.format_response(
                False,
                error=str(e),
                symbol=symbol
            )

    def get_signal(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Get trading signal based on on-chain metrics

        Args:
            symbol: Cryptocurrency symbol
            **kwargs: Additional parameters

        Returns:
            Trading signal with confidence
        """
        try:
            analysis = self.analyze(symbol, **kwargs)

            if not analysis['success']:
                return analysis

            overall_signal = analysis['data']['overall_signal']

            # Parse signal
            if 'BULLISH' in overall_signal:
                signal = 'BUY'
                confidence = 75
            elif 'BEARISH' in overall_signal:
                signal = 'SELL'
                confidence = 75
            else:
                signal = 'HOLD'
                confidence = 50

            return {
                'success': True,
                'symbol': symbol.upper(),
                'strategy': self.name,
                'signal': signal,
                'confidence': confidence,
                'reason': overall_signal,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting signal: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy': self.name,
                'timestamp': datetime.now().isoformat()
            }

    # Individual metric methods (mock implementations)
    # In production, these would call actual APIs

    def _get_active_addresses(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Get active addresses metric - REAL DATA for BTC!"""

        # Only BTC supported by Blockchain.com free API
        if symbol not in ['BTC', 'BTCUSDT']:
            logger.info(f"Using mock data for {symbol} (Blockchain.com only supports BTC)")
            return self._mock_active_addresses(symbol, days_back)

        try:
            import requests

            logger.info(f"📡 Fetching REAL active addresses data for BTC from Blockchain.com...")

            # Blockchain.com API endpoint
            url = f"https://api.blockchain.info/charts/n-unique-addresses?timespan={days_back}days&format=json"

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get('values') or len(data['values']) < 2:
                raise ValueError("Insufficient data from API")

            # Get latest and previous values
            latest_value = int(data['values'][-1]['y'])
            previous_value = int(data['values'][-2]['y'])

            # Calculate trend
            trend = 'UP' if latest_value > previous_value else 'DOWN'
            change_pct = ((latest_value - previous_value) / previous_value) * 100

            logger.info(f"✅ Real data fetched: {latest_value:,} active addresses (trend: {trend})")

            return {
                'success': True,
                'latest_value': latest_value,
                'trend': trend,
                'interpretation': f'Network activity is {trend.lower()} ({change_pct:+.1f}% vs previous day)',
                'data_source': '🌐 Real data from Blockchain.com'
            }

        except requests.exceptions.Timeout:
            logger.warning(f"⏱️ Blockchain.com API timeout, using mock data")
            return self._mock_active_addresses(symbol, days_back)

        except requests.exceptions.RequestException as e:
            logger.warning(f"❌ Blockchain.com API error: {e}, using mock data")
            return self._mock_active_addresses(symbol, days_back)

        except Exception as e:
            logger.error(f"❌ Unexpected error fetching real data: {e}")
            return self._mock_active_addresses(symbol, days_back)

    def _mock_active_addresses(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Fallback mock data"""
        base = {'BTC': 800000, 'ETH': 450000, 'LTC': 150000}.get(symbol, 400000)
        latest = base + np.random.randint(-50000, 50000)
        trend = 'UP' if np.random.random() > 0.5 else 'DOWN'

        return {
            'success': True,
            'latest_value': latest,
            'trend': trend,
            'interpretation': f'Network activity is {trend.lower()}',
            'data_source': '📊 Simulated data'
        }

    def _get_transaction_count(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Get transaction count metric"""
        base = {'BTC': 350000, 'ETH': 1200000, 'LTC': 200000}.get(symbol, 500000)
        latest = base + np.random.randint(-100000, 100000)
        avg_7d = latest + np.random.randint(-50000, 50000)

        return {
            'success': True,
            'latest_count': latest,
            'avg_7d': avg_7d,
            'trend': 'UP' if latest > avg_7d else 'DOWN'
        }

    def _get_exchange_flows(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Get exchange flow metric"""
        inflow = np.random.uniform(100, 1000)
        outflow = np.random.uniform(200, 1500)
        net_flow = outflow - inflow
        signal = 'BULLISH' if net_flow > 0 else 'BEARISH'

        return {
            'success': True,
            'latest_inflow': inflow,
            'latest_outflow': outflow,
            'net_flow': net_flow,
            'signal': signal,
            'interpretation': f'{signal} - {"Accumulation" if net_flow > 0 else "Distribution"}'
        }

    def _get_whale_movements(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Get whale movement metric"""
        whale_count = np.random.randint(5, 50)
        avg_amount = np.random.uniform(100, 1000)
        sentiment = 'HIGH' if whale_count > 25 else 'MEDIUM' if whale_count > 15 else 'LOW'

        return {
            'success': True,
            'whale_count_7d': whale_count,
            'avg_whale_amount': avg_amount,
            'whale_sentiment': f'{sentiment} activity'
        }

    def _get_hash_rate(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Get hash rate metric (PoW coins only)"""
        if symbol not in ['BTC', 'LTC', 'BCH']:
            return {'success': False, 'error': 'Hash rate only for PoW coins'}

        base = {'BTC': 400, 'LTC': 800, 'BCH': 300}.get(symbol, 400)
        latest = base + np.random.uniform(-50, 50)
        trend = 'UP' if np.random.random() > 0.5 else 'DOWN'

        return {
            'success': True,
            'latest_hash_rate': latest,
            'trend': trend,
            'network_security': 'Very High' if trend == 'UP' else 'High'
        }

    def _get_nvt_ratio(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Get NVT ratio metric"""
        nvt = np.random.uniform(50, 150)

        if nvt < 50:
            status = 'UNDERVALUED'
        elif nvt < 100:
            status = 'FAIRLY_VALUED'
        else:
            status = 'OVERVALUED'

        return {
            'success': True,
            'latest_nvt': nvt,
            'valuation_status': status
        }

    def _get_mvrv_ratio(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Get MVRV ratio metric"""
        mvrv = np.random.uniform(0.8, 3.5)

        if mvrv < 0.5:
            signal = 'EXTREME_UNDERVALUATION'
        elif mvrv < 1.0:
            signal = 'UNDERVALUED'
        elif mvrv < 3.0:
            signal = 'FAIRLY_VALUED'
        else:
            signal = 'EXTREME_OVERVALUATION'

        return {
            'success': True,
            'latest_mvrv': mvrv,
            'signal': signal
        }

    def _calculate_overall_signal(self, metrics: Dict) -> str:
        """Calculate overall signal from all metrics"""
        bullish = 0
        bearish = 0

        for metric_name, metric_data in metrics.items():
            if not metric_data.get('success'):
                continue

            if 'signal' in metric_data:
                if 'BULLISH' in str(metric_data['signal']):
                    bullish += 1
                elif 'BEARISH' in str(metric_data['signal']):
                    bearish += 1
            elif 'trend' in metric_data:
                if metric_data['trend'] == 'UP':
                    bullish += 1
                elif metric_data['trend'] == 'DOWN':
                    bearish += 1

        if bullish > bearish:
            return 'BULLISH - On-chain metrics suggest upward pressure'
        elif bearish > bullish:
            return 'BEARISH - On-chain metrics suggest downward pressure'
        else:
            return 'NEUTRAL - Mixed on-chain signals'