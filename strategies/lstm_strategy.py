"""
LSTM Price Prediction Strategy
Wraps your existing LSTM model in Strategy Pattern
"""
from strategies.base_strategy import AnalysisStrategy
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class LSTMPredictionStrategy(AnalysisStrategy):
    """Strategy for LSTM-based price prediction"""

    def __init__(self, repository=None):
        super().__init__("LSTMPrediction")
        self.repository = repository

    def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Perform LSTM price prediction"""
        try:
            if not self.validate_symbol(symbol):
                return self.format_response(False, error=f"Invalid symbol: {symbol}", symbol=symbol)

            lookback_period = kwargs.get('lookback_period', 60)
            days_ahead = kwargs.get('days_ahead', 7)

            # Get data from repository
            if not self.repository:
                return self.format_response(False, error="No repository configured", symbol=symbol)

            df = self.repository.get_price_history(symbol, lookback_period + 100)

            if df.empty or len(df) < lookback_period + 30:
                return self.format_response(
                    False,
                    error=f"Insufficient data: need {lookback_period + 30} records, have {len(df)}",
                    symbol=symbol
                )

            # Use your existing LSTM model
            try:
                import os
                import sys

                # Debug: Show where Python is looking
                current_dir = os.path.dirname(os.path.abspath(__file__))
                logger.info(f"🔍 Current file: {__file__}")
                logger.info(f"🔍 Current directory: {current_dir}")
                logger.info(f"🔍 Python path: {sys.path[:3]}")

                # Check if lstm_model.py exists
                lstm_path = os.path.join(current_dir, 'lstm_model.py')
                logger.info(f"🔍 Looking for: {lstm_path}")
                logger.info(f"🔍 File exists: {os.path.exists(lstm_path)}")
                strategies_dir = os.path.dirname(os.path.abspath(__file__))
                if strategies_dir not in sys.path:
                    sys.path.insert(0, strategies_dir)

                from lstm_model import predict_cryptocurrency_price
                from lstm_model import predict_cryptocurrency_price

                logger.info("✅ LSTM model imported successfully!")
                results = predict_cryptocurrency_price(
                    df=df,
                    symbol=symbol,
                    lookback_period=lookback_period,
                    days_ahead=days_ahead
                )

                self.log_analysis(symbol, f"Predicted {days_ahead} days ahead")
                return self.format_response(True, data=results, symbol=symbol)

            except ImportError as e:
                logger.error(f"❌ ImportError: {e}")
                logger.warning("📊 Using mock prediction")
                return self._generate_mock_prediction(df, symbol, days_ahead)

        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return self.format_response(False, error=str(e), symbol=symbol)

    def get_signal(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get trading signal based on LSTM prediction"""
        try:
            analysis = self.analyze(symbol, **kwargs)

            if not analysis['success']:
                return analysis

            predictions = analysis['data']['predictions']['predictions']
            current_price = analysis['data']['predictions']['current_price']
            confidence_score = analysis['data']['predictions']['confidence_score']

            # Calculate average predicted change
            import numpy as np
            avg_change = np.mean([p['change_from_current'] for p in predictions])

            # Generate signal
            if avg_change > 5 and confidence_score > 70:
                signal = 'STRONG BUY'
            elif avg_change > 2:
                signal = 'BUY'
            elif avg_change < -5 and confidence_score > 70:
                signal = 'STRONG SELL'
            elif avg_change < -2:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            return {
                'success': True,
                'symbol': symbol.upper(),
                'strategy': self.name,
                'signal': signal,
                'confidence': confidence_score,
                'reason': f"LSTM predicts {avg_change:+.2f}% change",
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {'success': False, 'error': str(e), 'strategy': self.name}

    def _generate_mock_prediction(self, df, symbol: str, days_ahead: int):
        """Fallback mock prediction"""
        import numpy as np
        current_price = float(df['close'].iloc[-1])

        predictions = []
        for day in range(1, days_ahead + 1):
            change = np.random.uniform(-3, 4)
            predicted_price = current_price * (1 + change / 100)

            predictions.append({
                'day': day,
                'predicted_price': float(predicted_price),
                'change_from_current': float(change),
                'trend': 'BULLISH' if change > 0 else 'BEARISH'
            })

        return self.format_response(True, data={
            'symbol': symbol,
            'predictions': {
                'current_price': current_price,
                'predictions': predictions,
                'confidence_score': 50.0
            }
        }, symbol=symbol)