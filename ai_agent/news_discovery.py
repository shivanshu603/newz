from typing import List, Dict, Any
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from config import Config
from utils import logger
from models import Article, TrendingTopic
from scraper import NewsScraper
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import os
from urllib.parse import urlencode

class NewsDiscoverer:
    def __init__(self):
        self.scraper = NewsScraper()

        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.min_word_length = 3

        self.sources = Config.NEWS_SOURCES  # List of news sources to scrape
        self.trending_window = Config.TRENDING_WINDOW_HOURS

        self.session = None
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.google_news_url = "https://news.google.com/rss/search"

    async def discover_articles(self) -> List[Dict[str, Any]]:
        """Discover new articles from configured sources."""
        try:
            articles = []
            # Fetch from news sites
            news_sites = [
                {
                    'url': 'https://www.theverge.com/tech',
                    'selectors': {
                        'article': 'article',
                        'title': 'h2',
                        'description': '.p-dek'
                    }
                },
                {
                    'url': 'https://techcrunch.com',
                    'selectors': {
                        'article': 'article',
                        'title': 'h2',
                        'description': '.post-block__content'
                    }
                }
            ]
            
            async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'}) as session:
                for site in news_sites:
                    try:
                        async with session.get(site['url']) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')
                                
                                for article in soup.select(site['selectors']['article'])[:5]:
                                    title = article.select_one(site['selectors']['title'])
                                    desc = article.select_one(site['selectors']['description'])
                                    
                                    if title:
                                        article_data = {
                                            'title': title.text.strip(),
                                            'description': desc.text.strip() if desc else '',
                                            'source': site['url'],
                                            'type': 'news',
                                            'keywords': self._extract_keywords(title.text.strip())
                                        }
                                        if self._validate_article_data(article_data):
                                            articles.append(article_data)
                                    
                    except Exception as e:
                        logger.error(f"Error scraping {site['url']}: {e}")
                        continue
                    
                    await asyncio.sleep(1)  # Polite delay between requests

            return articles

        except Exception as e:
            logger.error(f"Error discovering articles: {str(e)}")
            return []

    def _validate_article_data(self, data: Dict[str, Any]) -> bool:
        """Validate article data quality"""
        if not data.get('title') or len(data['title']) < 20:
            return False
            
        # Check for English content
        if not self._is_english_text(data['title']):
            return False
            
        # Check content length if available
        if 'content' in data and len(data['content']) < 100:
            return False
            
        return True

    def _is_english_text(self, text: str) -> bool:
        """Check if text is primarily English"""
        try:
            # Count ASCII vs non-ASCII characters
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            return ascii_chars / len(text) > 0.8
        except:
            return False

    async def get_trending_topics(self) -> List[TrendingTopic]:
        """Discover trending topics from articles"""
        try:
            # Get articles
            articles = await self.discover_articles()
            
            # Extract and analyze topics 
            topics = self._extract_topics(articles)
            trending_topics = self._analyze_trending_topics(topics)
            
            return trending_topics[:Config.MAX_TRENDING_TOPICS]
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {str(e)}")
            return []

    def _extract_topics(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Extract and count topics from articles"""
        topic_counts = {}
        for article in articles:
            try:
                title = article.get('title', '')
                description = article.get('description', '')
                if not title or not description:
                    continue
                
                # Combine title and description for analysis
                text = f"{title} {description}"
                
                # Tokenize and analyze
                tokens = word_tokenize(text.lower())
                
                # Count meaningful words
                for token in tokens:
                    if (len(token) >= self.min_word_length and 
                        token not in self.stop_words and 
                        token.isalnum()):
                        topic_counts[token] = topic_counts.get(token, 0) + 1
            except Exception as e:
                logger.error(f"Error extracting topics: {str(e)}")
                continue
        return topic_counts

    def _analyze_trending_topics(self, topics: Dict[str, int]) -> List[TrendingTopic]:
        """Analyze and rank trending topics"""
        trending = []
        for topic, count in topics.items():
            try:
                # Create trending topic object
                trending_topic = TrendingTopic(
                    name=topic,
                    frequency=count,
                    first_seen=datetime.now(),
                    last_seen=datetime.now()
                )
                trending.append(trending_topic)
            except Exception as e:
                logger.error(f"Error creating trending topic: {str(e)}")
                continue
        
        # Sort by frequency and recency
        return sorted(
            trending,
            key=lambda x: (x.frequency, x.last_seen),
            reverse=True
        )

    def _filter_topics(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out common words and non-English topics"""
        filtered = []
        for topic in topics:
            name = topic.get('name', '').lower()
            
            # Skip if topic name is too short or is a stop word
            if (len(name) < self.min_word_length or 
                name in self.stop_words or 
                not name.isascii()):
                continue
                
            # Skip common words and articles
            if name in {'the', 'and', 'or', 'but', 'to', 'a', 'an', 'in', 'on', 'at', 'for'}:
                continue
                
            # Add only if it's a meaningful topic
            if self._is_meaningful_topic(name):
                filtered.append(topic)
                
        return filtered

    def _is_meaningful_topic(self, topic: str) -> bool:
        """Check if topic is meaningful"""
        # Skip single letters or numbers
        if len(topic) <= 1 or topic.isdigit():
            return False
            
        # Skip if mostly special characters
        alpha_count = sum(c.isalpha() for c in topic)
        if alpha_count / len(topic) < 0.5:
            return False
            
        return True

    def _clean_description(self, description: str) -> str:
        """Clean article description text"""
        # Remove HTML tags
        soup = BeautifulSoup(description, 'html.parser')
        text = soup.get_text()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate to reasonable length
        return text[:500] + ('...' if len(text) > 500 else '')

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using NLP"""
        try:
            # Tokenize and clean text
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and short words
            keywords = [
                word for word in tokens 
                if word not in self.stop_words 
                and len(word) >= self.min_word_length
                and word.isalnum()
            ]
            
            # Count frequencies
            from collections import Counter
            keyword_freq = Counter(keywords)
            
            # Return top keywords
            return [word for word, count in keyword_freq.most_common(10)]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    async def get_news(self, topic: str) -> Dict:
        """Get news articles about a topic"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            results = {
                'facts': [],
                'sources': []
            }

            # Try News API first if available
            if self.news_api_key:
                news_api_results = await self._fetch_from_news_api(topic)
                if news_api_results:
                    results['facts'].extend(news_api_results.get('facts', []))
                    results['sources'].extend(news_api_results.get('sources', []))

            # Get Google News results
            google_news_results = await self._fetch_from_google_news(topic)
            if google_news_results:
                results['facts'].extend(google_news_results.get('facts', []))
                results['sources'].extend(google_news_results.get('sources', []))

            # Deduplicate results
            results['facts'] = list(set(results['facts']))
            results['sources'] = list(set(results['sources']))

            return results

        except Exception as e:
            logger.error(f"Error getting news: {e}")
            return {'facts': [], 'sources': []}

    async def _fetch_from_news_api(self, topic: str) -> Dict:
        """Fetch news from News API"""
        try:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': topic,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.news_api_key
            }

            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        
                        facts = []
                        sources = []
                        
                        for article in articles:
                            if article and isinstance(article, dict):
                                if article.get('description'):
                                    facts.append(article['description'])
                                if article.get('url'):
                                    sources.append(article['url'])
                                
                        return {'facts': facts, 'sources': sources}
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error fetching from News API: {e}")
            except Exception as e:
                logger.error(f"Error processing News API response: {e}")
                    
            return {'facts': [], 'sources': []}

        except Exception as e:
            logger.error(f"Error fetching from News API: {e}")
            return {'facts': [], 'sources': []}

    async def _fetch_from_google_news(self, topic: str) -> Dict:
        """Fetch news from Google News RSS"""
        try:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()

            params = {'q': topic, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'}
            url = f"{self.google_news_url}?{urlencode(params)}"
            
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        facts = []
                        sources = []
                        
                        if feed and hasattr(feed, 'entries'):
                            for entry in feed.entries:
                                if entry:
                                    if hasattr(entry, 'summary'):
                                        facts.append(entry.summary)
                                    if hasattr(entry, 'link'):
                                        sources.append(entry.link)
                                
                        return {'facts': facts, 'sources': sources}
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error fetching from Google News: {e}")
            except Exception as e:
                logger.error(f"Error processing Google News response: {e}")
                    
            return {'facts': [], 'sources': []}

        except Exception as e:
            logger.error(f"Error fetching from Google News: {e}")
            return {'facts': [], 'sources': []}
