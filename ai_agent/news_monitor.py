from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from bs4 import BeautifulSoup, Tag  # Add Tag import
from utils import logger
from config import Config  # Import Config for NEWS_SOURCES


class NewsMonitor:
    def __init__(self):
        self.is_running = False

    async def initialize(self):
        """Initialize the news monitor."""
        self.is_running = True

    async def scrape_google_trends(self) -> List[str]:
        """Scrape trending topics from Google Trends."""
        try:
            url = "https://trends.google.com/trends/trendingsearches/daily"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if not response.ok:
                        return []
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    trends: List[str] = []
                    for trend in soup.find_all('div', class_='trend-item'):
                        if trend and trend.get_text(strip=True):
                            trends.append(trend.get_text(strip=True))
                    return trends
        except Exception as e:
            logger.error(f"Error fetching Google Trends: {e}")
            return []

    async def scrape_google_news(self) -> List[Dict[str, str]]:
        """Scrape news articles from Google News."""
        try:
            url = "https://news.google.com/topstories"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if not response.ok:
                        return []
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    articles: List[Dict[str, str]] = []
                    
                    for article in soup.find_all('article'):
                        title_elem = article.find('h3')
                        link_elem = article.find('a')
                        
                        if title_elem and link_elem:
                            title = title_elem.get_text(strip=True)
                            href = link_elem.get('href', '')
                            
                            if title and href:
                                articles.append({
                                    "title": title,
                                    "url": href,
                                    "publishedAt": "2023-01-01T00:00:00Z"
                                })
                    return articles
        except Exception as e:
            logger.error(f"Error fetching Google News articles: {e}")
            return []

    async def scrape_google_discover(self) -> List[Dict[str, str]]:
        """Scrape articles from Google Discover."""
        try:
            url = "https://www.google.com/discover"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if not response.ok:
                        return []
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    articles: List[Dict[str, str]] = []
                    
                    for article in soup.find_all('article'):
                        title_elem = article.find('h3')
                        link_elem = article.find('a')
                        
                        if title_elem and link_elem:
                            title = title_elem.get_text(strip=True)
                            href = link_elem.get('href', '')
                            
                            if title and href:
                                articles.append({
                                    "title": title,
                                    "url": href,
                                    "publishedAt": "2023-01-01T00:00:00Z"
                                })
                    return articles
        except Exception as e:
            logger.error(f"Error fetching Google Discover articles: {e}")
            return []

    async def scrape_twitter(self) -> List[str]:
        """Scrape trending topics from Twitter."""
        try:
            url = "https://twitter.com/explore/tabs/trending"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if not response.ok:
                        return []
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    trends: List[str] = []
                    for trend in soup.find_all('span', class_='trend-item'):
                        if trend and trend.get_text(strip=True):
                            trends.append(trend.get_text(strip=True))
                    return trends
        except Exception as e:
            logger.error(f"Error fetching Twitter trends: {e}")
            return []

    async def generate_content(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Generate content using a GPT model based on scraped articles."""
        # Placeholder for content generation logic
        return []

    async def publish_to_wordpress(self, content: List[Dict[str, str]]) -> None:
        """Publish generated content to WordPress."""
        # Placeholder for WordPress publishing logic
        pass

    async def get_rss_topics(self) -> List[str]:
        """Get topics from configured RSS feeds."""
        topics = []
        try:
            import feedparser
            logger.info(f"Checking {len(Config.NEWS_SOURCES)} RSS feeds")
            
            for url in Config.NEWS_SOURCES:
                try:
                    logger.debug(f"Fetching RSS feed: {url}")
                    feed = feedparser.parse(url)
                    
                    if hasattr(feed, 'status') and feed.status != 200:
                        logger.warning(f"RSS feed {url} returned status {getattr(feed, 'status', 'unknown')}")
                        continue
                        
                    if not hasattr(feed, 'entries') or not feed.entries:
                        logger.warning(f"No entries found in RSS feed: {url}")
                        continue
                        
                    logger.debug(f"RSS feed {url} has {len(feed.entries)} entries")
                    
                    for i, entry in enumerate(feed.entries[:5]):  # Limit to first 5 entries per feed
                        if hasattr(entry, 'title'):
                            topics.append(entry.title)
                            logger.debug(f"Found topic: {entry.title[:50]}...")
                            
                except Exception as e:
                    logger.error(f"Error processing RSS feed {url}: {str(e)}")
                    continue
                    
            logger.info(f"Found {len(topics)} raw topics before deduplication")
            unique_topics = list(set(topics))
            logger.info(f"Returning {len(unique_topics)} unique topics")
            return unique_topics
            
        except Exception as e:
            logger.error(f"Error fetching RSS feeds: {str(e)}", exc_info=True)
            return []

    async def monitor_sources(self):
        """Continuously monitor news sources for trending topics."""
        while self.is_running:
            try:
                # First try RSS feeds
                trending_topics = await self.get_rss_topics()
                
                # Fall back to scraping if RSS fails
                if not trending_topics:
                    trending_topics = await self.scrape_google_trends()
                
                logger.info(f"Found {len(trending_topics)} trending topics")
                
                # Process topics through RAG
                if trending_topics:
                    from utils.rag_helper import RAGHelper
                    rag = RAGHelper()
                    for topic in trending_topics:
                        try:
                            # Ensure topic is a string
                            topic_str = str(topic)
                            context = await rag.get_context(topic_str)
                            if context:
                                logger.info(f"Processed topic through RAG: {topic_str[:50]}...")
                        except Exception as e:
                            logger.error(f"Error processing topic {topic}: {e}")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            await asyncio.sleep(3600)  # Wait for an hour before the next check

    async def monitor_news(self) -> List[Dict[str, Any]]:
        """Monitor news sources and return aggregated news data."""
        try:
            news_articles = await self.scrape_google_news()
            discover_articles = await self.scrape_google_discover()
            return news_articles + discover_articles
        except Exception as e:
            logger.error(f"Error monitoring news: {e}")
            return []

    async def stop(self):
        """Stop monitoring news sources."""
        self.is_running = False
