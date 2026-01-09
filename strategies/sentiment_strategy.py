"""
Sentiment Analysis Strategy
Analyzes news and social media sentiment for cryptocurrencies
"""
from .base_strategy import AnalysisStrategy
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import requests
import numpy as np
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)


class SentimentAnalysisStrategy(AnalysisStrategy):
    """
    Strategy for sentiment analysis using news and social media
    Uses NewsAPI and CryptoPanic for data sources
    """

    def __init__(self, news_api_key: Optional[str] = None, cryptopanic_key: Optional[str] = None):
        """
        Initialize sentiment analysis strategy

        Args:
            news_api_key: API key for NewsAPI
            cryptopanic_key: API key for CryptoPanic
        """
        super().__init__("SentimentAnalysis")
        self.news_api_key = news_api_key
        self.cryptopanic_key = cryptopanic_key
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)

        self.news_api_url = "https://newsapi.org/v2/everything"

    def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis

        Args:
            symbol: Cryptocurrency symbol
            days_back: Number of days to analyze (default: 7)

        Returns:
            Sentiment analysis results
        """
        try:
            if not self.validate_symbol(symbol):
                return self.format_response(
                    False,
                    error=f"Invalid symbol: {symbol}",
                    symbol=symbol
                )

            days_back = kwargs.get('days_back', 7)

            # Gather data from sources
            sources_data = []

            if self.news_api_key:
                news_data = self._get_newsapi_data(symbol, days_back)
                if news_data.get('success'):
                    sources_data.append(news_data)

            if self.cryptopanic_key:
                panic_data = self._get_cryptopanic_data(symbol, days_back)
                if panic_data.get('success'):
                    sources_data.append(panic_data)

            if not sources_data:
                return self.format_response(
                    False,
                    error="No sentiment data sources available or all sources failed",
                    symbol=symbol
                )

            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(sources_data)

            result = {
                'sources': sources_data,
                **overall_sentiment
            }

            self.log_analysis(
                symbol,
                f"Sentiment: {overall_sentiment['overall_sentiment']} "
                f"(Score: {overall_sentiment['average_polarity']:.3f})"
            )

            return self.format_response(
                True,
                data=result,
                symbol=symbol
            )

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return self.format_response(
                False,
                error=str(e),
                symbol=symbol
            )

    def get_signal(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Get trading signal based on sentiment

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

            sentiment_data = analysis['data']
            polarity = sentiment_data['average_polarity']
            confidence = sentiment_data['confidence']

            # Generate signal
            if polarity > 0.2 and confidence > 60:
                signal = 'STRONG BUY'
            elif polarity > 0.1:
                signal = 'BUY'
            elif polarity < -0.2 and confidence > 60:
                signal = 'STRONG SELL'
            elif polarity < -0.1:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            return {
                'success': True,
                'symbol': symbol.upper(),
                'strategy': self.name,
                'signal': signal,
                'confidence': confidence,
                'reason': f"Sentiment score: {polarity:.3f} from {sentiment_data['total_articles']} articles",
                'details': {
                    'sentiment': sentiment_data['overall_sentiment'],
                    'polarity': polarity,
                    'articles_analyzed': sentiment_data['total_articles']
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

    def _get_newsapi_data(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Fetch and analyze news from NewsAPI"""
        cache_key = f"newsapi_{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"

        if cache_key in self.cache:
            cached_time = self.cache[cache_key].get('timestamp')
            if cached_time and (datetime.now() - cached_time) < self.cache_duration:
                self.logger.info(f"Using cached NewsAPI data for {symbol}")
                return self.cache[cache_key]['data']

        try:
            crypto_name = self._get_crypto_name(symbol)
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            params = {
                'q': f'{crypto_name} OR {symbol}',
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': from_date,
                'pageSize': 50
            }

            self.logger.info(f"Fetching NewsAPI data for {symbol}...")
            response = requests.get(self.news_api_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                self.logger.info(f"NewsAPI returned {len(articles)} articles")

                analyzed_articles = []
                for article in articles:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    content = f"{title} {description}"

                    sentiment = self._analyze_text_sentiment(content)

                    analyzed_articles.append({
                        'title': title,
                        'description': description,
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'sentiment': sentiment['sentiment'],
                        'polarity': sentiment['polarity'],
                        'confidence': sentiment['confidence']
                    })

                result = {
                    'success': True,
                    'source': 'NewsAPI',
                    'symbol': symbol,
                    'articles_count': len(analyzed_articles),
                    'articles': analyzed_articles,
                    'timestamp': datetime.now().isoformat()
                }

                self.cache[cache_key] = {
                    'data': result,
                    'timestamp': datetime.now()
                }

                return result

            return {'success': False, 'error': data.get('message', 'Unknown error')}

        except Exception as e:
            self.logger.error(f"NewsAPI error: {e}")
            return {'success': False, 'error': str(e)}

    def _get_cryptopanic_data(self, symbol: str, days_back: int) -> Dict[str, Any]:
        """Fetch and analyze posts from CryptoPanic"""
        cache_key = f"cryptopanic_{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"

        if cache_key in self.cache:
            cached_time = self.cache[cache_key].get('timestamp')
            if cached_time and (datetime.now() - cached_time) < self.cache_duration:
                self.logger.info(f"Using cached CryptoPanic data for {symbol}")
                return self.cache[cache_key]['data']

        try:
            clean_symbol = symbol.replace('USDT', '').replace('USD', '').replace('BUSD', '')

            url = 'https://cryptopanic.com/api/developer/v2/posts/'
            params = {
                'auth_token': self.cryptopanic_key,
                'currencies': clean_symbol,
                'public': 'true',
                'kind': 'news'
            }

            self.logger.info(f"Fetching CryptoPanic data for {symbol}...")
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                self.logger.info(f"CryptoPanic returned {len(results)} posts")

                analyzed_posts = []
                for post in results:
                    title = post.get('title', '')
                    sentiment_data = self._analyze_text_sentiment(title)
                    votes = post.get('votes', {})

                    analyzed_posts.append({
                        'title': title,
                        'url': post.get('url', ''),
                        'published_at': post.get('published_at', ''),
                        'source': post.get('source', {}).get('title', 'Unknown'),
                        'sentiment': sentiment_data['sentiment'],
                        'polarity': sentiment_data['polarity'],
                        'votes': {
                            'positive': votes.get('positive', 0),
                            'negative': votes.get('negative', 0),
                            'important': votes.get('important', 0),
                            'liked': votes.get('liked', 0),
                            'disliked': votes.get('disliked', 0)
                        }
                    })

                result = {
                    'success': True,
                    'source': 'CryptoPanic',
                    'symbol': symbol,
                    'posts_count': len(analyzed_posts),
                    'posts': analyzed_posts,
                    'timestamp': datetime.now().isoformat()
                }

                self.cache[cache_key] = {
                    'data': result,
                    'timestamp': datetime.now()
                }

                return result

            return {'success': False, 'error': f'HTTP {response.status_code}'}

        except Exception as e:
            self.logger.error(f"CryptoPanic error: {e}")
            return {'success': False, 'error': str(e)}

    def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using TextBlob

        Args:
            text: Text to analyze

        Returns:
            Sentiment analysis result
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity > 0.1:
                sentiment = 'POSITIVE'
            elif polarity < -0.1:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'

            return {
                'sentiment': sentiment,
                'polarity': round(polarity, 4),
                'subjectivity': round(subjectivity, 4),
                'confidence': abs(polarity)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            return {
                'sentiment': 'NEUTRAL',
                'polarity': 0.0,
                'subjectivity': 0.5,
                'confidence': 0.0
            }

    def _calculate_overall_sentiment(self, sources: List[Dict]) -> Dict[str, Any]:
        """Calculate overall sentiment from multiple sources"""
        all_polarities = []
        sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}

        for source in sources:
            if 'articles' in source:
                for article in source['articles']:
                    all_polarities.append(article['polarity'])
                    sentiment_counts[article['sentiment']] += 1
            elif 'posts' in source:
                for post in source['posts']:
                    all_polarities.append(post['polarity'])
                    sentiment_counts[post['sentiment']] += 1

        if not all_polarities:
            return {
                'overall_sentiment': 'NEUTRAL',
                'average_polarity': 0.0,
                'sentiment_distribution': sentiment_counts,
                'confidence': 0.0,
                'total_articles': 0
            }

        avg_polarity = np.mean(all_polarities)
        std_polarity = np.std(all_polarities)

        if avg_polarity > 0.1:
            overall = 'BULLISH'
        elif avg_polarity < -0.1:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'

        confidence = max(0, min(100, (1 - std_polarity) * 100))

        return {
            'overall_sentiment': overall,
            'average_polarity': round(avg_polarity, 4),
            'sentiment_distribution': sentiment_counts,
            'confidence': round(confidence, 2),
            'total_articles': len(all_polarities),
            'polarity_std': round(std_polarity, 4)
        }

    def _get_crypto_name(self, symbol: str) -> str:
        """Convert symbol to common name"""
        crypto_names = {
            'BTC': 'Bitcoin',
            'BTCUSDT': 'Bitcoin',
            'ETH': 'Ethereum',
            'ETHUSDT': 'Ethereum',
            'BNB': 'Binance Coin',
            'BNBUSDT': 'Binance Coin',
            'XRP': 'Ripple',
            'XRPUSDT': 'Ripple',
            'ADA': 'Cardano',
            'ADAUSDT': 'Cardano',
            'SOL': 'Solana',
            'SOLUSDT': 'Solana',
            'DOT': 'Polkadot',
            'DOTUSDT': 'Polkadot',
            'DOGE': 'Dogecoin',
            'DOGEUSDT': 'Dogecoin',
            'MATIC': 'Polygon',
            'MATICUSDT': 'Polygon',
            'LTC': 'Litecoin',
            'LTCUSDT': 'Litecoin',
        }
        return crypto_names.get(symbol.upper(), symbol.replace('USDT', ''))