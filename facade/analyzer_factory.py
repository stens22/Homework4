"""
Factory Pattern for Analysis Strategies
Creates appropriate strategy instances based on type
"""
from typing import Optional, Dict, Any
from strategies import (
    AnalysisStrategy,
    TechnicalAnalysisStrategy,
    SentimentAnalysisStrategy,
    LSTMPredictionStrategy,
    OnChainMetricsStrategy
)
import logging

logger = logging.getLogger(__name__)


class AnalyzerFactory:
    """
    Factory class for creating analysis strategies

    Implements the Factory Pattern to centralize strategy creation
    and manage dependencies (API keys, repositories, etc.)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize factory with configuration

        Args:
            config: Dictionary containing API keys and other config:
                {
                    'repository': DatabaseRepository instance,
                    'news_api_key': str,
                    'cryptopanic_api_key': str,
                    'glassnode_api_key': str,
                    'cryptoquant_api_key': str
                }
        """
        self.config = config or {}
        self._strategy_types = {
            'technical': TechnicalAnalysisStrategy,
            'sentiment': SentimentAnalysisStrategy,
            'lstm': LSTMPredictionStrategy,
            'onchain': OnChainMetricsStrategy
        }

        logger.info("AnalyzerFactory initialized")

    def create_strategy(self, strategy_type: str) -> Optional[AnalysisStrategy]:
        """
        Create an analysis strategy of the specified type

        Args:
            strategy_type: Type of strategy to create
                          ('technical', 'sentiment', 'lstm', 'onchain')

        Returns:
            Instantiated strategy or None if type not found

        Raises:
            ValueError: If strategy type is unknown
        """
        strategy_type = strategy_type.lower()

        if strategy_type not in self._strategy_types:
            available = ', '.join(self._strategy_types.keys())
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available types: {available}"
            )

        # Create strategy with appropriate dependencies
        if strategy_type == 'technical':
            repository = self.config.get('repository')
            strategy = TechnicalAnalysisStrategy(repository=repository)

        elif strategy_type == 'sentiment':
            news_api_key = self.config.get('news_api_key')
            cryptopanic_key = self.config.get('cryptopanic_api_key')
            strategy = SentimentAnalysisStrategy(
                news_api_key=news_api_key,
                cryptopanic_key=cryptopanic_key
            )

        elif strategy_type == 'lstm':
            repository = self.config.get('repository')
            strategy = LSTMPredictionStrategy(repository=repository)

        elif strategy_type == 'onchain':
            glassnode_key = self.config.get('glassnode_api_key')
            cryptoquant_key = self.config.get('cryptoquant_api_key')
            strategy = OnChainMetricsStrategy(
                glassnode_key=glassnode_key,
                cryptoquant_key=cryptoquant_key
            )

        logger.info(f"Created strategy: {strategy.name}")
        return strategy

    def create_all_strategies(self) -> Dict[str, AnalysisStrategy]:
        """
        Create all available strategies

        Returns:
            Dictionary mapping strategy type to strategy instance
        """
        strategies = {}

        for strategy_type in self._strategy_types.keys():
            try:
                strategies[strategy_type] = self.create_strategy(strategy_type)
            except Exception as e:
                logger.error(f"Failed to create {strategy_type} strategy: {e}")

        logger.info(f"Created {len(strategies)} strategies")
        return strategies

    def get_available_strategies(self) -> list:
        """
        Get list of available strategy types

        Returns:
            List of strategy type names
        """
        return list(self._strategy_types.keys())

    def update_config(self, config: Dict[str, Any]):
        """
        Update factory configuration

        Args:
            config: New configuration dictionary
        """
        self.config.update(config)
        logger.info("Factory configuration updated")


class AnalyzerManager:
    """
    Manager class that coordinates multiple strategies

    This class provides a unified interface for running analysis
    across multiple strategies at once.
    """

    def __init__(self, factory: AnalyzerFactory):
        """
        Initialize manager with a factory

        Args:
            factory: AnalyzerFactory instance
        """
        self.factory = factory
        self.strategies = {}
        logger.info("AnalyzerManager initialized")

    def register_strategy(self, name: str, strategy: AnalysisStrategy):
        """
        Register a strategy with the manager

        Args:
            name: Name to identify the strategy
            strategy: Strategy instance
        """
        self.strategies[name] = strategy
        logger.info(f"Registered strategy: {name}")

    def run_analysis(self, symbol: str, strategy_types: list = None, **kwargs) -> Dict[str, Any]:
        """
        Run analysis using multiple strategies

        Args:
            symbol: Cryptocurrency symbol
            strategy_types: List of strategy types to use (None = all)
            **kwargs: Additional parameters for strategies

        Returns:
            Dictionary with results from each strategy:
            {
                'symbol': str,
                'results': {
                    'technical': {...},
                    'sentiment': {...},
                    ...
                },
                'timestamp': str
            }
        """
        from datetime import datetime

        if strategy_types is None:
            strategy_types = self.factory.get_available_strategies()

        results = {}

        for strategy_type in strategy_types:
            try:
                # Create or get strategy
                if strategy_type not in self.strategies:
                    strategy = self.factory.create_strategy(strategy_type)
                    self.strategies[strategy_type] = strategy
                else:
                    strategy = self.strategies[strategy_type]

                # Run analysis
                result = strategy.analyze(symbol, **kwargs)
                results[strategy_type] = result

            except Exception as e:
                logger.error(f"Error running {strategy_type} strategy: {e}")
                results[strategy_type] = {
                    'success': False,
                    'error': str(e)
                }

        return {
            'symbol': symbol.upper(),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

    def get_combined_signal(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Get combined trading signal from all strategies

        Args:
            symbol: Cryptocurrency symbol
            **kwargs: Additional parameters

        Returns:
            Combined signal with confidence
        """
        from datetime import datetime

        # Get signals from all strategies
        signals = {}
        for strategy_type, strategy in self.strategies.items():
            try:
                signal = strategy.get_signal(symbol, **kwargs)
                if signal.get('success'):
                    signals[strategy_type] = signal
            except Exception as e:
                logger.error(f"Error getting signal from {strategy_type}: {e}")

        if not signals:
            return {
                'success': False,
                'error': 'No signals available',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }

        # Aggregate signals
        buy_count = sum(1 for s in signals.values() if 'BUY' in s.get('signal', ''))
        sell_count = sum(1 for s in signals.values() if 'SELL' in s.get('signal', ''))
        hold_count = len(signals) - buy_count - sell_count

        # Calculate combined signal
        if buy_count > sell_count and buy_count > hold_count:
            combined_signal = 'STRONG BUY' if buy_count >= len(signals) * 0.75 else 'BUY'
        elif sell_count > buy_count and sell_count > hold_count:
            combined_signal = 'STRONG SELL' if sell_count >= len(signals) * 0.75 else 'SELL'
        else:
            combined_signal = 'HOLD'

        # Calculate confidence
        max_agreement = max(buy_count, sell_count, hold_count)
        confidence = (max_agreement / len(signals)) * 100

        return {
            'success': True,
            'symbol': symbol.upper(),
            'combined_signal': combined_signal,
            'confidence': round(confidence, 2),
            'individual_signals': signals,
            'signal_distribution': {
                'buy': buy_count,
                'sell': sell_count,
                'hold': hold_count
            },
            'timestamp': datetime.now().isoformat()
        }


# Convenience function for quick strategy creation
def create_analyzer(strategy_type: str, **config) -> AnalysisStrategy:
    """
    Quick function to create a strategy

    Args:
        strategy_type: Type of strategy to create
        **config: Configuration parameters

    Returns:
        Strategy instance

    Example:
        analyzer = create_analyzer('technical', repository=my_repo)
        result = analyzer.analyze('BTCUSDT')
    """
    factory = AnalyzerFactory(config)
    return factory.create_strategy(strategy_type)