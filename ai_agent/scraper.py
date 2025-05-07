import aiohttp
import asyncio
from typing import Optional, List, Dict
from bs4 import BeautifulSoup
from datetime import datetime
import feedparser
from utils import logger
from config import Config

class NewsScraper:
    async def scrape_source(self, source: Dict, additional_sources: Optional[List[str]] = None) -> List[Dict]:
        """Scrape articles from a given source URL."""
        url = source['url']
        articles = await self.scrape_rss(url)

        if additional_sources:
            for source in additional_sources:
                articles.extend(await self.scrape_rss(source))
        return articles

    async def scrape_rss(self, feed_url: str) -> List[Dict]:
        """Scrape articles from an RSS feed."""
        try:
            logger.info(f"Fetching RSS feed from: {feed_url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(feed_url, timeout=Config.SCRAPER_TIMEOUT) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_rss_feed(content)
                    else:
                        logger.warning(f"Failed to fetch RSS feed {feed_url}: Status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error scraping RSS feed {feed_url}: {e}")
            return []

    def _parse_date(self, date_str: str) -> str:
        """Parse date from various formats."""
        try:
            return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z').isoformat()
        except ValueError:
            try:
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z').isoformat()
            except ValueError:
                return date_str

    def _parse_rss_feed(self, feed_content: str) -> List[Dict]:
        """Parse RSS feed content."""
        feed = feedparser.parse(feed_content)
        articles = []

        
        for entry in feed.entries:
            title = entry.get('title', 'Untitled Article')
            # Ensure title meets length requirements
            if len(title) < 10:
                title = f"News Article: {title}"
            elif len(title) > 200:
                title = title[:197] + "..."
                
            article = {
                'title': title,
                'link': entry.get('link', ''),
                'published': self._parse_date(entry.get('published', '')),
                'summary': entry.get('summary', ''),
                'source': feed.feed.get('title', 'Unknown Source')
            }
            articles.append(article)
        return articles

    # Other methods remain unchanged...
