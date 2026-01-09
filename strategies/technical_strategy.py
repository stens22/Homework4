"""
Technical Analysis Strategy
Implements technical indicators (RSI, MACD, Moving Averages, etc.)
"""
from .base_strategy import AnalysisStrategy
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TechnicalAnalysisStrategy(AnalysisStrategy):
    """
    Strategy for technical analysis using price indicators
    Analyzes RSI, MACD, Moving Averages, Bollinger Bands, etc.
    """

    def __init__(self, repository=None):
        """
        Initialize technical analysis strategy

        Args:
            repository: Database repository for fetching price data
        """
        super().__init__("TechnicalAnalysis")
        self.repository = repository

    def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis

        Args:
            symbol: Cryptocurrency symbol
            timeframe: 'short' (1d), 'medium' (1w), 'long' (1m)
            days_back: Number of days of data to analyze

        Returns:
            Technical analysis results
        """
        try:
            # Validate symbol
            if not self.validate_symbol(symbol):
                return self.format_response(
                    False,
                    error=f"Invalid symbol: {symbol}",
                    symbol=symbol
                )

            # Get parameters
            timeframe = kwargs.get('timeframe', 'short')
            days_back = kwargs.get('days_back', 60)

            # Fetch price data
            if not self.repository:
                return self.format_response(
                    False,
                    error="No repository configured",
                    symbol=symbol
                )

            df = self.repository.get_price_history(symbol, days_back)

            if df.empty:
                return self.format_response(
                    False,
                    error=f"No data found for {symbol}",
                    symbol=symbol
                )

            # Perform analysis
            analysis_result = self._perform_technical_analysis(df, timeframe)

            self.log_analysis(symbol, f"Completed - Signal: {analysis_result['summary']['recommendation']}")

            return self.format_response(
                True,
                data=analysis_result,
                symbol=symbol
            )

        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return self.format_response(
                False,
                error=str(e),
                symbol=symbol
            )

    def get_signal(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Get trading signal based on technical indicators

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

            summary = analysis['data']['summary']

            # Calculate confidence based on signal agreement
            total_signals = summary['total_signals']
            buy_signals = summary['buy_signals']
            sell_signals = summary['sell_signals']

            # Confidence is how unified the signals are
            if summary['recommendation'] in ['STRONG BUY', 'STRONG SELL']:
                confidence = 85 + (10 * (max(buy_signals, sell_signals) / total_signals))
            elif summary['recommendation'] in ['BUY', 'SELL']:
                confidence = 60 + (20 * (max(buy_signals, sell_signals) / total_signals))
            else:
                confidence = 40

            return {
                'success': True,
                'symbol': symbol.upper(),
                'strategy': self.name,
                'signal': summary['recommendation'],
                'confidence': round(min(confidence, 100), 2),
                'reason': f"{buy_signals} buy signals, {sell_signals} sell signals from {total_signals} indicators",
                'details': {
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'hold_signals': summary['hold_signals']
                },
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

    def _perform_technical_analysis(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Internal method to perform technical analysis calculations

        Args:
            df: DataFrame with OHLCV data
            timeframe: Analysis timeframe

        Returns:
            Analysis results
        """
        # Set parameters based on timeframe
        if timeframe == 'short':
            rsi_period = 14
            ma_period = 20
            macd_fast, macd_slow, macd_signal = 12, 26, 9
        elif timeframe == 'medium':
            rsi_period = 21
            ma_period = 30
            macd_fast, macd_slow, macd_signal = 18, 39, 13
        else:  # long
            rsi_period = 28
            ma_period = 50
            macd_fast, macd_slow, macd_signal = 24, 52, 18

        # Calculate indicators
        oscillators = self._calculate_oscillators(df, rsi_period, macd_fast, macd_slow, macd_signal)
        moving_averages = self._calculate_moving_averages(df, ma_period)

        # Generate signals
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0

        for indicator in oscillators.values():
            if indicator.get('signal') == 'BUY':
                buy_signals += 1
            elif indicator.get('signal') == 'SELL':
                sell_signals += 1
            else:
                hold_signals += 1

        for indicator in moving_averages.values():
            if indicator.get('signal') in ['BUY', 'STRONG']:
                buy_signals += 1
            elif indicator.get('signal') in ['SELL', 'WEAK']:
                sell_signals += 1
            else:
                hold_signals += 1

        total_signals = buy_signals + sell_signals + hold_signals

        # Determine overall recommendation
        if buy_signals > sell_signals and buy_signals > hold_signals:
            recommendation = 'STRONG BUY' if buy_signals > total_signals * 0.7 else 'BUY'
        elif sell_signals > buy_signals and sell_signals > hold_signals:
            recommendation = 'STRONG SELL' if sell_signals > total_signals * 0.7 else 'SELL'
        else:
            recommendation = 'HOLD'

        # Get price info
        latest_idx = df.index[-1]
        current_price = float(df.loc[latest_idx, 'close'])
        current_volume = float(df.loc[latest_idx, 'volume'])

        return {
            'oscillators': oscillators,
            'moving_averages': moving_averages,
            'summary': {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'total_signals': total_signals,
                'recommendation': recommendation,
                'timeframe': timeframe
            },
            'price_info': {
                'current_price': current_price,
                'current_volume': current_volume,
                'date': str(df.loc[latest_idx, 'date'])
            }
        }

    def _calculate_oscillators(self, df, rsi_period, macd_fast, macd_slow, macd_signal):
        """Calculate oscillator indicators (RSI, MACD, Stochastic, ADX, CCI)"""
        oscillators = {}

        # RSI
        rsi = self._calculate_rsi(df, rsi_period)
        rsi_value = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
        oscillators['RSI'] = {
            'value': rsi_value,
            'signal': self._get_rsi_signal(rsi_value),
            'description': f'RSI: {rsi_value:.2f}' if rsi_value else 'Insufficient data'
        }

        # MACD
        macd_data = self._calculate_macd(df, macd_fast, macd_slow, macd_signal)
        macd_val = float(macd_data['macd'].iloc[-1]) if not pd.isna(macd_data['macd'].iloc[-1]) else None
        signal_val = float(macd_data['signal'].iloc[-1]) if not pd.isna(macd_data['signal'].iloc[-1]) else None

        oscillators['MACD'] = {
            'macd': macd_val,
            'signal_line': signal_val,
            'histogram': float(macd_data['histogram'].iloc[-1]) if not pd.isna(
                macd_data['histogram'].iloc[-1]) else None,
            'signal': 'BUY' if macd_val and signal_val and macd_val > signal_val else 'SELL',
            'description': f'MACD: {macd_val:.2f}' if macd_val else 'Insufficient data'
        }

        # Stochastic
        stoch_data = self._calculate_stochastic(df)
        k_val = float(stoch_data['k'].iloc[-1]) if not pd.isna(stoch_data['k'].iloc[-1]) else None
        oscillators['Stochastic'] = {
            'k': k_val,
            'd': float(stoch_data['d'].iloc[-1]) if not pd.isna(stoch_data['d'].iloc[-1]) else None,
            'signal': self._get_stochastic_signal(k_val),
            'description': f'%K: {k_val:.2f}' if k_val else 'Insufficient data'
        }

        # ADX
        adx_data = self._calculate_adx(df)
        adx_val = float(adx_data['adx'].iloc[-1]) if not pd.isna(adx_data['adx'].iloc[-1]) else None
        plus_di = float(adx_data['plus_di'].iloc[-1]) if not pd.isna(adx_data['plus_di'].iloc[-1]) else None
        minus_di = float(adx_data['minus_di'].iloc[-1]) if not pd.isna(adx_data['minus_di'].iloc[-1]) else None

        oscillators['ADX'] = {
            'value': adx_val,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'signal': self._get_adx_signal(adx_val, plus_di, minus_di),
            'description': f'ADX: {adx_val:.2f}' if adx_val else 'Insufficient data'
        }

        # CCI
        cci = self._calculate_cci(df)
        cci_val = float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else None
        oscillators['CCI'] = {
            'value': cci_val,
            'signal': self._get_cci_signal(cci_val),
            'description': f'CCI: {cci_val:.2f}' if cci_val else 'Insufficient data'
        }

        return oscillators

    def _calculate_moving_averages(self, df, ma_period):
        """Calculate moving average indicators"""
        moving_averages = {}
        current_price = float(df['close'].iloc[-1])

        # SMA
        sma = df['close'].rolling(window=ma_period).mean()
        sma_val = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None
        moving_averages[f'SMA_{ma_period}'] = {
            'value': sma_val,
            'signal': 'BUY' if sma_val and current_price > sma_val else 'SELL',
            'description': f'Price {"above" if sma_val and current_price > sma_val else "below"} SMA'
        }

        # EMA
        ema = df['close'].ewm(span=ma_period, adjust=False).mean()
        ema_val = float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else None
        moving_averages[f'EMA_{ma_period}'] = {
            'value': ema_val,
            'signal': 'BUY' if ema_val and current_price > ema_val else 'SELL',
            'description': f'Price {"above" if ema_val and current_price > ema_val else "below"} EMA'
        }

        # WMA
        weights = np.arange(1, ma_period + 1)
        wma = df['close'].rolling(window=ma_period).apply(
            lambda x: np.sum(weights * x[-ma_period:]) / weights.sum() if len(x) >= ma_period else np.nan,
            raw=True
        )
        wma_val = float(wma.iloc[-1]) if not pd.isna(wma.iloc[-1]) else None
        moving_averages[f'WMA_{ma_period}'] = {
            'value': wma_val,
            'signal': 'BUY' if wma_val and current_price > wma_val else 'SELL',
            'description': f'Price {"above" if wma_val and current_price > wma_val else "below"} WMA'
        }

        # Bollinger Bands
        sma_bb = df['close'].rolling(window=ma_period).mean()
        std_bb = df['close'].rolling(window=ma_period).std()
        upper = sma_bb + (std_bb * 2)
        lower = sma_bb - (std_bb * 2)

        upper_val = float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None
        lower_val = float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None
        middle_val = float(sma_bb.iloc[-1]) if not pd.isna(sma_bb.iloc[-1]) else None

        bb_signal = 'HOLD'
        if lower_val and current_price <= lower_val:
            bb_signal = 'BUY'
        elif upper_val and current_price >= upper_val:
            bb_signal = 'SELL'

        moving_averages['Bollinger_Bands'] = {
            'upper': upper_val,
            'middle': middle_val,
            'lower': lower_val,
            'signal': bb_signal,
            'description': 'Price within bands' if bb_signal == 'HOLD' else f'Price at {"lower" if bb_signal == "BUY" else "upper"} band'
        }

        # Volume MA
        volume_ma = df['volume'].rolling(window=ma_period).mean()
        vol_ma_val = float(volume_ma.iloc[-1]) if not pd.isna(volume_ma.iloc[-1]) else None
        current_volume = float(df['volume'].iloc[-1])

        moving_averages['Volume_MA'] = {
            'value': vol_ma_val,
            'current_volume': current_volume,
            'signal': 'STRONG' if vol_ma_val and current_volume > vol_ma_val * 1.5 else 'WEAK',
            'description': f'Volume: {current_volume / 1e6:.2f}M'
        }

        return moving_averages

    # Helper calculation methods
    def _calculate_rsi(self, df, period):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, df, fast, slow, signal):
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        }

    def _calculate_stochastic(self, df, period=14):
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        k = k.rolling(window=3).mean()
        d = k.rolling(window=3).mean()
        return {'k': k, 'd': d}

    def _calculate_adx(self, df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}

    def _calculate_cci(self, df, period=20):
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad)

    # Signal generation helpers
    def _get_rsi_signal(self, rsi):
        if pd.isna(rsi):
            return 'HOLD'
        return 'BUY' if rsi < 30 else 'SELL' if rsi > 70 else 'HOLD'

    def _get_stochastic_signal(self, k):
        if pd.isna(k):
            return 'HOLD'
        return 'BUY' if k < 20 else 'SELL' if k > 80 else 'HOLD'

    def _get_adx_signal(self, adx, plus_di, minus_di):
        if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
            return 'HOLD'
        if adx < 20:
            return 'HOLD'
        return 'BUY' if plus_di > minus_di else 'SELL'

    def _get_cci_signal(self, cci):
        if pd.isna(cci):
            return 'HOLD'
        return 'BUY' if cci < -100 else 'SELL' if cci > 100 else 'HOLD'