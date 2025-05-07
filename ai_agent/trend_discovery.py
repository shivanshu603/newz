import asyncio
import aiohttp
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Set
from utils import logger
from config import Config
import re
import random
from fake_useragent import UserAgent
import json
import xml.etree.ElementTree as ET
import pickle
import os
from difflib import SequenceMatcher

class TrendDiscovery:
    def __init__(self):
        self.ua = UserAgent()
        self.headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        self.google_trends_url = "https://trends.google.com/trends/trendingsearches/daily?geo=US"
        self.google_rss_url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US"
        self.twitter_trends_url = "https://www.trendsmap.com/united-states"  # Alternative source
        self.reddit_url = "https://old.reddit.com/r/{}/hot"
        self.subreddits = ['technology', 'science', 'programming', 'worldnews']

        # Add default RSS feeds in case Config.RSS_FEEDS is not available
        self.default_rss_feeds = [
            {
                'url': 'https://news.google.com/news/rss',
                'name': 'Google News'
            },
            {
                'url': 'https://feeds.bbci.co.uk/news/rss.xml',
                'name': 'BBC News'
            },
            {
                'url': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
                'name': 'NY Times'
            }
        ]

        # Add backup URLs
        self.backup_urls = {
            'google_trends': [
                "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US",
                "https://trends.google.com/trends/hottrends/atom/feed?pn=p1"
            ],
            'twitter_trends': [
                "https://trends24.in/united-states/",
                "https://getdaytrends.com/united-states/",
                "https://twitter-trends.iamrohit.in/united-states"
            ]
        }

        # Add topic tracking
        self.history_file = 'topic_history.pkl'
        self.processed_topics: Set[str] = self._load_topic_history()
        self.topic_expiry = timedelta(days=7)  # Don't repeat topics for 7 days
        self.min_topic_similarity = 0.8  # Similarity threshold for duplicate detection
        
        # Expand subreddits for more variety
        self.subreddits = [
            'technology', 'science', 'programming', 'worldnews',
            'business', 'finance', 'tech', 'artificial', 'datascience',
            'crypto', 'space', 'environment', 'cybersecurity'
        ]

    def _load_topic_history(self) -> Set[str]:
        """Load previously processed topics"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    history_data = pickle.load(f)
                    # Clean up expired topics
                    current_time = datetime.now()
                    return {
                        topic for topic, timestamp in history_data.items()
                        if current_time - timestamp < self.topic_expiry
                    }
            return set()
        except Exception as e:
            logger.error(f"Error loading topic history: {e}")
            return set()

    def _save_topic_history(self, topic: str):
        """Save processed topic with timestamp"""
        try:
            history_data = {}
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    history_data = pickle.load(f)
            
            # Add new topic with timestamp
            history_data[topic] = datetime.now()
            
            # Save updated history
            with open(self.history_file, 'wb') as f:
                pickle.dump(history_data, f)
        except Exception as e:
            logger.error(f"Error saving topic history: {e}")

    def _is_duplicate_topic(self, topic: str) -> bool:
        """Check if topic is duplicate or too similar to recent topics"""
        try:
            topic_lower = topic.lower()
            
            # Exact match check
            if topic_lower in self.processed_topics:
                return True
            
            # Similarity check
            for processed_topic in self.processed_topics:
                similarity = SequenceMatcher(None, topic_lower, processed_topic.lower()).ratio()
                if similarity > self.min_topic_similarity:
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking duplicate topic: {e}")
            return False

    async def get_trending_topics(self) -> List[Dict]:
        """Get trending topics with duplicate checking"""
        try:
            all_topics = []
            seen_topics = set()
            
            # Collect topics from all sources
            results = await asyncio.gather(
                self._scrape_google_trends(),
                self._scrape_reddit_trends(),
                self._scrape_twitter_trends(),
                self._get_news_from_rss()
            )
            
            # Process each source's results
            for source_topics in results:
                for topic in source_topics:
                    topic_name = topic['name'].lower()
                    
                    # Skip if duplicate or too similar
                    if topic_name in seen_topics or self._is_duplicate_topic(topic_name):
                        continue
                    
                    seen_topics.add(topic_name)
                    all_topics.append(topic)
            
            # Score and sort topics
            scored_topics = await self._score_and_deduplicate_topics(all_topics)
            
            # Update history with new topics
            for topic in scored_topics[:10]:
                self._save_topic_history(topic['name'])
            
            return scored_topics[:10]
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []

    async def _scrape_google_trends(self) -> List[Dict]:
        """Scrape Google Trends from RSS feed directly"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US"
                headers = {
                    **self.headers,
                    'Accept': 'application/xml',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        trends = []
                        for entry in feed.entries[:10]:  # Get top 10 trends
                            trend = {
                                'name': entry.title,
                                'source': 'google_trends',
                                'type': 'trend',
                                'description': entry.get('description', ''),
                                'published': entry.get('published', datetime.now().isoformat()),
                                'url': entry.get('link', '')
                            }
                            trends.append(trend)
                            logger.info(f"Found Google trend: {entry.title}")
                        
                        return trends
                        
            return []
            
        except Exception as e:
            logger.error(f"Error scraping Google Trends: {e}")
            return []

    async def _scrape_google_trends_api(self) -> List[Dict]:
        """Try Google Trends API endpoint"""
        try:
            headers = {
                **self.headers,
                'Accept': 'application/json',
                'referer': 'https://trends.google.com/trends/explore'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.google_trends_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.text()
                        # Remove ")]}'\n" prefix that Google adds
                        json_data = json.loads(data.replace(")]}'", ""))
                        
                        trends = []
                        for trend in json_data.get('storySummaries', {}).get('trendingStories', []):
                            trends.append({
                                'name': trend.get('title', ''),
                                'source': 'google_trends',
                                'type': 'trend',
                                'description': trend.get('summary', ''),
                                'published': datetime.now().isoformat()
                            })
                        return trends
            return []
            
        except Exception as e:
            logger.error(f"Error in Google Trends API: {e}")
            return []

    async def _scrape_google_trends_rss(self) -> List[Dict]:
        """Fallback method to get trends from RSS feed"""
        try:
            rss_url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US"
            async with aiohttp.ClientSession() as session:
                async with session.get(rss_url, headers=self.headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        return [{
                            'name': entry.title,
                            'source': 'google_trends',
                            'type': 'trend',
                            'description': entry.get('description', ''),
                            'published': datetime.now().isoformat(),
                            'url': entry.get('link', '')
                        } for entry in feed.entries[:10]]
            return []
        except Exception as e:
            logger.error(f"Error scraping Google Trends RSS: {e}")
            return []

    async def _scrape_google_trends_web(self) -> List[Dict]:
        """Scrape Google Trends through web page"""
        try:
            headers = {
                **self.headers,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.google_trends_url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        trends = []
                        # Try multiple selectors for redundancy
                        trend_elements = (
                            soup.select('.trending-searches-list .list-item') or
                            soup.select('.feed-item-header') or
                            soup.select('.details')
                        )
                        
                        for element in trend_elements:
                            title = (
                                element.select_one('.title') or
                                element.select_one('a') or
                                element
                            )
                            
                            if title and title.text.strip():
                                trends.append({
                                    'name': title.text.strip(),
                                    'source': 'google_trends',
                                    'type': 'trend',
                                    'published': datetime.now().isoformat()
                                })
                        
                        if trends:
                            return trends[:10]

            return []
            
        except Exception as e:
            logger.error(f"Error scraping Google Trends web: {e}")
            return []

    async def _scrape_reddit_trends(self) -> List[Dict]:
        """Scrape Reddit trends with improved variety"""
        trends = []
        seen_titles = set()
        
        try:
            headers = {
                **self.headers,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'
            }
            
            # Shuffle subreddits for variety
            random.shuffle(self.subreddits)
            
            for subreddit in self.subreddits[:5]:  # Take 5 random subreddits
                url = self.reddit_url.format(subreddit)
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            posts = soup.select('#siteTable .thing')
                            for post in posts:
                                title = post.select_one('a.title')
                                score = post.select_one('.score')
                                
                                if title:
                                    title_text = title.text.strip()
                                    
                                    # Skip if title already seen or is duplicate
                                    if (title_text in seen_titles or 
                                        self._is_duplicate_topic(title_text)):
                                        continue
                                    
                                    seen_titles.add(title_text)
                                    
                                    # Skip promotional content
                                    if any(promo in title_text.lower() for promo in [
                                        'subscribe', 'click here', 'discount', 'sale',
                                        'offer', 'limited time', 'buy now'
                                    ]):
                                        continue
                                    
                                    trends.append({
                                        'name': title_text,
                                        'source': 'reddit',
                                        'subreddit': subreddit,
                                        'type': 'trend',
                                        'score': int(score['title'].split()[0]) if score and score.get('title') else 0,
                                        'published': datetime.now().isoformat()
                                    })
                                    
                                    # Save to history
                                    self._save_topic_history(title_text)
                
                await asyncio.sleep(1)
            
            # Sort by score and return top unique trends
            trends.sort(key=lambda x: x.get('score', 0), reverse=True)
            return trends[:10]  # Return top 10 unique trends
            
        except Exception as e:
            logger.error(f"Error scraping Reddit trends: {e}")
            return []

    async def _scrape_twitter_trends(self) -> List[Dict]:
        """Scrape Twitter trends using trends24.in"""
        try:
            sources = [
                {
                    'url': 'https://trends24.in/united-states/',
                    'selector': '.trend-card .trend-card__list-item'
                },
                {
                    'url': 'https://twitter-trends.iamrohit.in/united-states',
                    'selector': 'table tr td:first-child'
                }
            ]
            
            for source in sources:
                headers = {
                    **self.headers,
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(source['url'], headers=headers) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            trends = []
                            
                            items = soup.select(source['selector'])
                            for item in items:
                                name = item.get_text().strip()
                                if name and not name.startswith(('#', 'Note:', 'Trending')):
                                    trends.append({
                                        'name': name,
                                        'source': 'twitter',
                                        'type': 'trend',
                                        'published': datetime.now().isoformat()
                                    })
                                    logger.info(f"Found Twitter trend: {name}")
                            
                            if trends:
                                return trends[:10]
            
            return []
            
        except Exception as e:
            logger.error(f"Error scraping Twitter trends: {e}")
            return []

    async def _get_news_from_rss(self) -> List[Dict]:
        """Get news from RSS feeds"""
        news = []
        try:
            # Use Config.RSS_FEEDS if available, otherwise use default feeds
            feeds = getattr(Config, 'RSS_FEEDS', self.default_rss_feeds)
            
            for feed in feeds:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(feed['url'], headers=self.headers) as response:
                            if response.status == 200:
                                content = await response.text()
                                parsed_feed = feedparser.parse(content)
                                
                                for entry in parsed_feed.entries[:5]:  # Top 5 from each feed
                                    news.append({
                                        'name': entry.title,
                                        'description': entry.get('description', ''),
                                        'source': feed['name'],
                                        'type': 'news',
                                        'published': entry.get('published', datetime.now().isoformat()),
                                        'url': entry.get('link', '')
                                    })
                                    logger.info(f"Found RSS article: {entry.title}")
                    
                    await asyncio.sleep(1)  # Be nice to RSS servers
                    
                except Exception as e:
                    logger.error(f"Error fetching RSS feed {feed['url']}: {e}")
                    continue
                    
            return news
            
        except Exception as e:
            logger.error(f"Error getting RSS news: {e}")
            return []

    async def _score_and_deduplicate_topics(self, topics: List[Dict]) -> List[Dict]:
        """Score and deduplicate topics"""
        try:
            scored_topics = {}
            
            for topic in topics:
                # Create a normalized key for deduplication
                key = re.sub(r'[^\w\s]', '', topic['name'].lower())
                
                # Calculate topic score
                score = self._calculate_topic_score(topic)
                
                # Keep the highest scoring version of duplicate topics
                if key not in scored_topics or score > scored_topics[key]['score']:
                    topic['score'] = score
                    scored_topics[key] = topic
            
            # Sort by score and return
            return sorted(scored_topics.values(), key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error scoring topics: {e}")
            return topics

    def _calculate_topic_score(self, topic: Dict) -> float:
        """Calculate topic relevance score"""
        try:
            score = 0.5  # Base score
            
            # Source quality
            source_weights = {
                'google_trends': 0.8,
                'reddit': 0.7,
                'twitter': 0.6
            }
            score += source_weights.get(topic.get('source', '').lower(), 0.5) * 0.3
            
            # Reddit score bonus
            if topic.get('source') == 'reddit' and 'score' in topic:
                normalized_score = min(1.0, int(topic['score']) / 10000)
                score += normalized_score * 0.2
            
            # Recency bonus
            if 'published' in topic:
                try:
                    published = datetime.fromisoformat(topic['published'].replace('Z', '+00:00'))
                    age_hours = (datetime.now() - published).total_seconds() / 3600
                    recency_score = max(0, 1 - (age_hours / 24))  # Newer is better
                    score += recency_score * 0.2
                except:
                    pass
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating topic score: {e}")
            return 0.5

# Modified main function for better testing
async def main():
    try:
        discoverer = TrendDiscovery()
        
        # Test Google Trends
        print("\nFetching Google Trends...")
        google_trends = await discoverer._scrape_google_trends()
        print(f"Found {len(google_trends)} Google trends")
        for trend in google_trends[:3]:
            print(f"- {trend['name']}")

        # Test Reddit Trends
        print("\nFetching Reddit Trends...")
        reddit_trends = await discoverer._scrape_reddit_trends()
        print(f"Found {len(reddit_trends)} Reddit trends")
        for trend in reddit_trends[:3]:
            print(f"- {trend['name']}")

        # Test Twitter Trends
        print("\nFetching Twitter Trends...")
        twitter_trends = await discoverer._scrape_twitter_trends()
        print(f"Found {len(twitter_trends)} Twitter trends")
        for trend in twitter_trends[:3]:
            print(f"- {trend['name']}")

        # Test RSS Feeds
        print("\nFetching RSS News...")
        rss_news = await discoverer._get_news_from_rss()
        print(f"Found {len(rss_news)} RSS articles")
        for news in rss_news[:3]:
            print(f"- {news['name']}")

        # Get combined trending topics
        print("\nGetting combined trending topics...")
        all_topics = await discoverer.get_trending_topics()
        print(f"\nFound {len(all_topics)} total trending topics:")
        for topic in all_topics[:5]:
            print(f"- {topic['name']} (Source: {topic['source']}, Score: {topic.get('score', 'N/A')})")

    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
