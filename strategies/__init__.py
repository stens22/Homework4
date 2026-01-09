"""
Strategies Package
Contains all analysis strategies using Strategy Pattern
"""
from .base_strategy import AnalysisStrategy, AnalysisContext
from .technical_strategy import TechnicalAnalysisStrategy
from .sentiment_strategy import SentimentAnalysisStrategy
from .lstm_strategy import LSTMPredictionStrategy
from .onchain_strategy import OnChainMetricsStrategy

__all__ = [
    'AnalysisStrategy',
    'AnalysisContext',
    'TechnicalAnalysisStrategy',
    'SentimentAnalysisStrategy',
    'LSTMPredictionStrategy',
    'OnChainMetricsStrategy'
]