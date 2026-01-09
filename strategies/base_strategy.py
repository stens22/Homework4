"""
Base Strategy Pattern for Analysis
All analysis strategies inherit from this base class
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AnalysisStrategy(ABC):
    """Abstract base class for all analysis strategies"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Perform analysis on cryptocurrency symbol"""
        pass

    @abstractmethod
    def get_signal(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get trading signal (BUY/SELL/HOLD)"""
        pass

    def validate_symbol(self, symbol: str) -> bool:
        """Validate cryptocurrency symbol"""
        if not symbol or not isinstance(symbol, str):
            return False
        if len(symbol) < 3 or len(symbol) > 10:
            return False
        return symbol.isalnum()

    def format_response(self, success: bool, data: Dict = None,
                        error: str = None, symbol: str = None) -> Dict[str, Any]:
        """Format response in standard format"""
        response = {
            'success': success,
            'strategy': self.name,
            'timestamp': datetime.now().isoformat()
        }

        if symbol:
            response['symbol'] = symbol.upper()
        if success and data:
            response['data'] = data
        if not success and error:
            response['error'] = error

        return response

    def log_analysis(self, symbol: str, result: str):
        """Log analysis execution"""
        self.logger.info(f"[{self.name}] {symbol}: {result}")


class AnalysisContext:
    """Context class that uses an AnalysisStrategy"""

    def __init__(self, strategy: AnalysisStrategy = None):
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: AnalysisStrategy):
        self._strategy = strategy
        logger.info(f"Strategy changed to: {strategy.name}")

    def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        if not self._strategy:
            return {'success': False, 'error': 'No strategy set'}
        return self._strategy.analyze(symbol, **kwargs)

    def get_signal(self, symbol: str, **kwargs) -> Dict[str, Any]:
        if not self._strategy:
            return {'success': False, 'error': 'No strategy set'}
        return self._strategy.get_signal(symbol, **kwargs)