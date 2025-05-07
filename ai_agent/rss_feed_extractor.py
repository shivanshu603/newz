import feedparser
import asyncio
import aiohttp
from typing import List, Dict
from datetime import datetime
from utils import logger
from bs4 import BeautifulSoup

class RSSFeedExtractor:
    def __init__(self, feed_urls: List[str]):
        self.feed_urls = feed_urls
        self.session = None
    
    async def _init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def extract_topics(self) -> List[Dict]:
        """Extract topics from RSS feeds"""
        await self._init_session()
        all_topics = []
        
        try:
            for feed_url in self.feed_urls:
                try:
                    topics = await self._process_feed(feed_url)
                    all_topics.extend(topics)
                    logger.info(f"Extracted {len(topics)} topics from {feed_url}")
                except Exception as e:
                    logger.error(f"Error processing feed {feed_url}: {e}")
                    continue
            
            return self._deduplicate_topics(all_topics)
        
        except Exception as e:
            logger.error(f"Error in extract_topics: {e}")
            return []
        
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    async def _process_feed(self, feed_url: str) -> List[Dict]:
        """Process a single RSS feed"""
        try:
            async with self.session.get(feed_url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch feed {feed_url}: {response.status}")
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                topics = []
                for entry in feed.entries[:10]:  # Process top 10 entries
                    topic = self._extract_topic_from_entry(entry)
                    if topic:
                        topics.append(topic)
                
                return topics
                
        except Exception as e:
            logger.error(f"Error processing feed {feed_url}: {e}")
            return []

    def _extract_topic_from_entry(self, entry) -> Dict:
        """Extract topic information from a feed entry"""
        try:
            # Extract content
            content = ''
            if hasattr(entry, 'content'):
                content = entry.content[0].value
            elif hasattr(entry, 'summary'):
                content = entry.summary
            
            # Clean content
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                content = soup.get_text()
            
            # Create topic dictionary
            topic = {
                'name': entry.title,
                'description': content[:500],  # Limit description length
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat()),
                'source': entry.get('source', {}).get('title', 'RSS Feed'),
                'keywords': self._extract_keywords(entry),
                'type': 'rss'
            }
            
            return topic
            
        except Exception as e:
            logger.error(f"Error extracting topic from entry: {e}")
            return None

    def _extract_keywords(self, entry) -> List[str]:
        """Extract keywords from entry tags or categories"""
        keywords = set()
        
        # Try to get tags
        if hasattr(entry, 'tags'):
            for tag in entry.tags:
                keywords.add(tag.term.lower())
                
        # Try to get categories
        if hasattr(entry, 'categories'):
            for category in entry.categories:
                keywords.add(category.lower())
                
        return list(keywords)

    def _deduplicate_topics(self, topics: List[Dict]) -> List[Dict]:
        """Remove duplicate topics based on name"""
        seen = set()
        unique_topics = []
        
        for topic in topics:
            name = topic['name'].lower()
            if name not in seen:
                seen.add(name)
                unique_topics.append(topic)
        
        return unique_topics

# For testing
async def main():
    from config import Config
    extractor = RSSFeedExtractor(Config.get_news_sources())
    topics = await extractor.extract_topics()
    print(f"Found {len(topics)} topics:")
    for topic in topics:
        print(f"- {topic['name']}")

if __name__ == "__main__":
    asyncio.run(main())
