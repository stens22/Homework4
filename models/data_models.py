"""
Data Models
Define clean data structures for the application
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SentimentType(Enum):
    """Sentiment types"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


@dataclass
class CryptoPrice:
    """Model for cryptocurrency price data"""
    symbol: str
    current_price: float
    open: float
    high: float
    low: float
    volume: float
    date: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'date': self.date
        }

    @property
    def change_percent(self) -> float:
        """Calculate percentage change from open"""
        if self.open == 0:
            return 0.0
        return ((self.current_price - self.open) / self.open) * 100


@dataclass
class TradingSignal:
    """Model for trading signals"""
    symbol: str
    signal: SignalType
    confidence: float
    strategy: str
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'signal': self.signal.value if isinstance(self.signal, SignalType) else self.signal,
            'confidence': self.confidence,
            'strategy': self.strategy,
            'reason': self.reason,
            'timestamp': self.timestamp,
            'details': self.details
        }

    @property
    def is_buy(self) -> bool:
        """Check if signal is a buy signal"""
        return 'BUY' in (self.signal.value if isinstance(self.signal, SignalType) else self.signal)

    @property
    def is_sell(self) -> bool:
        """Check if signal is a sell signal"""
        return 'SELL' in (self.signal.value if isinstance(self.signal, SignalType) else self.signal)


@dataclass
class AnalysisResult:
    """Model for analysis results"""
    symbol: str
    strategy: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'symbol': self.symbol,
            'strategy': self.strategy,
            'success': self.success,
            'timestamp': self.timestamp
        }

        if self.success:
            result['data'] = self.data
        else:
            result['error'] = self.error

        return result


@dataclass
class UserPortfolio:
    """Model for user portfolio/wallet"""
    user_id: str
    assets: List[Dict[str, Any]] = field(default_factory=list)
    total_value: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_asset(self, symbol: str, amount: float, price: float):
        """Add asset to portfolio"""
        self.assets.append({
            'symbol': symbol,
            'amount': amount,
            'price': price,
            'value': amount * price
        })
        self._recalculate_total()

    def remove_asset(self, symbol: str):
        """Remove asset from portfolio"""
        self.assets = [a for a in self.assets if a['symbol'] != symbol]
        self._recalculate_total()

    def _recalculate_total(self):
        """Recalculate total portfolio value"""
        self.total_value = sum(a['value'] for a in self.assets)
        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'assets': self.assets,
            'total_value': self.total_value,
            'last_updated': self.last_updated
        }


@dataclass
class APIResponse:
    """Standard API response model"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        response = {
            'success': self.success,
            'timestamp': self.timestamp
        }

        if self.success:
            if self.data is not None:
                response['data'] = self.data
            if self.message:
                response['message'] = self.message
        else:
            if self.error:
                response['error'] = self.error

        return response

    @staticmethod
    def success_response(data: Any = None, message: str = None) -> 'APIResponse':
        """Create success response"""
        return APIResponse(success=True, data=data, message=message)

    @staticmethod
    def error_response(error: str) -> 'APIResponse':
        """Create error response"""
        return APIResponse(success=False, error=error)


@dataclass
class ServiceConfig:
    """Configuration for services"""
    service_name: str
    host: str = "localhost"
    port: int = 5000
    database_url: Optional[str] = None
    api_keys: Dict[str, str] = field(default_factory=dict)
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'service_name': self.service_name,
            'host': self.host,
            'port': self.port,
            'database_url': self.database_url,
            'api_keys': self.api_keys,
            'debug': self.debug
        }


# Utility functions for model conversion

def dict_to_crypto_price(data: Dict[str, Any]) -> CryptoPrice:
    """Convert dictionary to CryptoPrice model"""
    return CryptoPrice(
        symbol=data['symbol'],
        current_price=data['current_price'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        volume=data['volume'],
        date=data['date']
    )


def dict_to_trading_signal(data: Dict[str, Any]) -> TradingSignal:
    """Convert dictionary to TradingSignal model"""
    signal_value = data['signal']
    if isinstance(signal_value, str):
        try:
            signal = SignalType[signal_value.replace(' ', '_')]
        except KeyError:
            signal = signal_value
    else:
        signal = signal_value

    return TradingSignal(
        symbol=data['symbol'],
        signal=signal,
        confidence=data['confidence'],
        strategy=data['strategy'],
        reason=data['reason'],
        timestamp=data.get('timestamp', datetime.now().isoformat()),
        details=data.get('details', {})
    )