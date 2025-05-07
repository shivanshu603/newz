import aiohttp
import asyncio
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from datetime import datetime
import feedparser
from langdetect import detect, DetectorFactory
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
import re
import random
import json
from urllib.parse import quote
import signal
import sys

from config import Config      # Ensure your Config module is set up
from utils import logger       # Ensure logger is configured (e.e., logging.basicConfig(level=logging.INFO))

# Set seed for reproducible language detection results
DetectorFactory.seed = 0

class TrendingTopicDiscoverer:
    def __init__(self):
        self.user_agent = UserAgent()
        self.headers = {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        self.running = True
        self.last_run: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Configure intervals
        self.iteration_interval = getattr(Config, 'TREND_DISCOVERY_INTERVAL', 300)  # 5 minutes default
        self.error_retry_delay = getattr(Config, 'TREND_DISCOVERY_RETRY_DELAY', 60)
        self.max_retries = getattr(Config, 'TREND_DISCOVERY_MAX_RETRIES', 3)
        
        logger.info(f"Initialized TrendingTopicDiscoverer with {self.iteration_interval}s interval")

    async def start(self):
        """Start the discovery process"""
        self._session = aiohttp.ClientSession()
        logger.info("Started trending topic discovery")

    async def stop(self):
        """Stop the discovery process gracefully"""
        self.running = False
        if self._session:
            await self._session.close()
        logger.info("Stopped trending topic discovery")

    async def should_continue(self) -> bool:
        """Check if we should continue running based on various conditions"""
        if not self.running:
            return False
            
        # Add any additional checks here (e.g., resource limits, quotas)
        return True

    async def get_trending_topics(self) -> List[Dict]:
        """Get trending topics from multiple sources"""
        try:
            # Get topics from multiple sources in parallel with more sources
            tasks = [
                asyncio.create_task(self._get_tech_news_trends()),
                asyncio.create_task(self._get_reddit_trends()),
                asyncio.create_task(self.scrape_news_trends()),
                asyncio.create_task(self._get_google_trends()),
                asyncio.create_task(self._get_hackernews_trends()),
                asyncio.create_task(self._get_product_hunt_trends()),
                asyncio.create_task(self._get_github_trends()),
                asyncio.create_task(self._get_medium_trends())
            ]

            # Add timeout to prevent hanging
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_topics = []
            for result in results:
                if isinstance(result, list):
                    all_topics.extend(result)

            # Log the number of topics found from each source
            logger.info(f"Found a total of {len(all_topics)} topics from all sources")
            logger.debug(f"All topics: {all_topics}")

            # Filter and clean topics
            valid_topics = []
            seen_titles = set()
            
            for topic in all_topics:
                title = topic.get('name', '').strip()
                if (title 
                    and len(title) > 10 
                    and title not in seen_titles 
                    and self.is_english(title)
                    and not self._is_excluded_topic(title)):
                    seen_titles.add(title)
                    valid_topics.append(topic)

            logger.info(f"Found {len(valid_topics)} valid topics")
            return valid_topics[:20]  # Return top 20 topics

        except Exception as e:
            logger.error(f"Error in get_trending_topics: {e}")
            return []

    async def _get_google_trends(self) -> List[Dict]:
        """Retrieve trending searches from Google Trends public RSS feeds (no API required)."""
        trends = []
        try:
            # Default regions if Config.GOOGLE_TRENDS_REGIONS is not available
            regions = getattr(Config, 'GOOGLE_TRENDS_REGIONS', ['US', 'GB', 'CA', 'AU', 'IN'])

            # Start with global trends
            trend_urls = ["https://trends.google.com/trends/trendingsearches/daily/rss"]

            # Add region-specific trends
            trend_urls.extend([
                f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={region}"
                for region in regions
            ])

            logger.info(f"Fetching Google Trends from {len(trend_urls)} sources via public RSS feeds")
            
            # Verify we have a valid session
            if not self._session:
                logger.error("No active session for Google Trends fetch")
                return []

            async with aiohttp.ClientSession(headers=self.headers) as session:
                for url in trend_urls:
                    try:
                        async with session.get(url, timeout=15) as response:
                            if response.status == 200:
                                content = await response.text()
                                feed = feedparser.parse(content)

                                for entry in feed.entries[:5]:  # Get top 5 from each feed
                                    # Skip non-English content
                                    if not self.is_english(entry.title):
                                        continue

                                    # Get description from news item snippet or create one
                                    description = entry.get('ht:news_item_snippet', '')
                                    if not description and hasattr(entry, 'summary'):
                                        # Remove HTML tags from summary
                                        from bs4 import BeautifulSoup
                                        soup = BeautifulSoup(entry.summary, 'html.parser')
                                        description = soup.get_text()[:500]

                                    if not description:
                                        description = f"Trending search on Google: {entry.title}"

                                    trends.append({
                                        'name': entry.title,
                                        'description': description,
                                        'url': entry.link,
                                        'source': 'Google Trends',
                                        'type': 'trend',
                                        'timestamp': datetime.now().isoformat()
                                    })
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout fetching Google Trends from {url}")
                        continue
                    except Exception as e:
                        logger.error(f"Error fetching Google Trends from {url}: {e}")
                        continue

                    await asyncio.sleep(1)  # Be nice to Google's servers

            logger.info(f"Found {len(trends)} topics from Google Trends")
            return trends

        except Exception as e:
            logger.error(f"Error retrieving Google Trends: {e}")
            return []

    def is_english(self, text: str) -> bool:
        """Return True if the detected language is English."""
        try:
            return detect(text) == 'en'
        except Exception as e:
            logger.error(f"Language detection failed for text: {text}. Error: {e}")
            return False

    async def scrape_news_trends(self) -> List[Dict]:
        """Analyze news articles to extract trending topics."""
        try:
            from news_discovery import NewsDiscoverer
            discoverer = NewsDiscoverer()
            articles = await discoverer.discover_articles()
            
            # Simple frequency analysis of keywords in article titles
            word_counts = {}
            for article in articles:
                for word in article.title.split():
                    word_lower = word.lower()
                    if word_lower in word_counts:
                        word_counts[word_lower] += 1
                    else:
                        word_counts[word_lower] = 1
            
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            return [{
                'name': word[0],
                'count': word[1],
                'source': 'News Analysis',
                'timestamp': datetime.now().isoformat()
            } for word in sorted_words[:10]]
        except Exception as e:
            logger.error(f"Error analyzing news trends: {e}")
            return []

    async def _get_tech_news_trends(self) -> List[Dict]:
        """Get trending topics from tech news sites"""
        topics = []
        try:
            tech_sites = [
                {
                    'url': 'https://techcrunch.com',
                    'selector': 'article h2 a',
                    'name': 'TechCrunch'
                },
                {
                    'url': 'https://www.theverge.com/tech',
                    'selector': 'h2.c-entry-box--compact__title',
                    'name': 'The Verge'
                },
                {
                    'url': 'https://news.ycombinator.com',
                    'selector': '.storylink',
                    'name': 'Hacker News'
                }
            ]

            async with aiohttp.ClientSession(headers=self.headers) as session:
                for site in tech_sites:
                    try:
                        async with session.get(site['url']) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')
                                
                                for title in soup.select(site['selector'])[:5]:
                                    topics.append({
                                        'name': title.text.strip(),
                                        'description': '',
                                        'source': site['name'],
                                        'type': 'tech_news',
                                        'url': title.get('href', '')
                                    })
                    except Exception as e:
                        logger.error(f"Error scraping {site['name']}: {e}")
                        continue

            return topics

        except Exception as e:
            logger.error(f"Error getting tech news trends: {e}")
            return []

    async def _get_reddit_trends(self) -> List[Dict]:
        """Get trending topics from Reddit using public RSS feeds only (no API required)"""
        logger.info("Getting Reddit trends via public RSS feeds (no API required)")
        return await self._get_reddit_trends_via_rss()

    async def _get_reddit_trends_via_rss(self) -> List[Dict]:
        """Get Reddit trends via public RSS feeds (no API required)"""
        topics = []
        try:
            # Expanded list of subreddits to fetch via RSS
            subreddits = [
                'technology', 'science', 'programming', 'worldnews',
                'futurology', 'space', 'business', 'webdev', 'datascience',
                'artificial', 'machinelearning', 'coding', 'learnprogramming',
                'web_design', 'SEO', 'digitalmarketing', 'contentmarketing',
                'marketing', 'socialmedia', 'blogging'
            ]

            # Randomize and select a subset to avoid too many requests
            import random
            random.shuffle(subreddits)
            selected_subreddits = subreddits[:10]  # Select 10 random subreddits

            logger.info(f"Fetching Reddit trends via RSS from subreddits: {', '.join(selected_subreddits)}")

            async with aiohttp.ClientSession(headers=self.headers) as session:
                for subreddit in selected_subreddits:
                    try:
                        # Use the public RSS feed
                        url = f"https://www.reddit.com/r/{subreddit}/.rss"
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                content = await response.text()
                                feed = feedparser.parse(content)

                                for entry in feed.entries[:5]:  # Get top 5 from each feed
                                    # Extract title from the entry
                                    title = entry.title
                                    if title.startswith('[') and ']' in title:
                                        # Remove subreddit prefix if present
                                        title = title[title.find(']')+1:].strip()

                                    # Skip very short titles or non-English content
                                    if len(title) < 20 or not self.is_english(title):
                                        continue

                                    # Clean up description
                                    description = ""
                                    if hasattr(entry, 'summary'):
                                        # Remove HTML tags from summary
                                        from bs4 import BeautifulSoup
                                        soup = BeautifulSoup(entry.summary, 'html.parser')
                                        description = soup.get_text()[:500]

                                    if not description:
                                        description = f"Trending discussion on Reddit about {title}"

                                    topics.append({
                                        'name': title,
                                        'description': description,
                                        'url': entry.link,
                                        'source': f"Reddit r/{subreddit}",
                                        'type': 'reddit',
                                        'score': 0  # No score available via RSS
                                    })
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout fetching Reddit RSS for r/{subreddit}")
                        continue
                    except Exception as e:
                        logger.error(f"Error fetching Reddit RSS for r/{subreddit}: {e}")
                        continue

                    await asyncio.sleep(1)  # Be nice to Reddit's servers

            logger.info(f"Found {len(topics)} topics from Reddit RSS feeds")
            return topics

        except Exception as e:
            logger.error(f"Error getting Reddit trends via RSS: {e}")
            return []

    async def _get_hackernews_trends(self) -> List[Dict]:
        """Get trending topics from Hacker News using RSS feed (no API required)"""
        topics = []
        try:
            logger.info("Fetching trends from Hacker News via RSS feed")

            async with aiohttp.ClientSession(headers=self.headers) as session:
                # Use the public RSS feed instead of the API
                url = "https://news.ycombinator.com/rss"
                try:
                    async with session.get(url, timeout=15) as response:
                        if response.status == 200:
                            content = await response.text()
                            feed = feedparser.parse(content)

                            for entry in feed.entries[:20]:  # Get top 20 stories
                                # Skip very short titles or non-English content
                                if len(entry.title) < 15 or not self.is_english(entry.title):
                                    continue

                                # Clean up description
                                description = ""
                                if hasattr(entry, 'summary'):
                                    # Remove HTML tags from summary
                                    from bs4 import BeautifulSoup
                                    soup = BeautifulSoup(entry.summary, 'html.parser')
                                    description = soup.get_text()[:500]

                                if not description:
                                    description = f"Trending discussion on Hacker News about {entry.title}"

                                topics.append({
                                    'name': entry.title,
                                    'description': description,
                                    'url': entry.link,
                                    'source': 'Hacker News',
                                    'type': 'tech_news',
                                    'score': 0  # No score available via RSS
                                })
                except asyncio.TimeoutError:
                    logger.warning("Timeout fetching Hacker News RSS feed")
                except Exception as e:
                    logger.error(f"Error fetching Hacker News RSS feed: {e}")

                # If RSS feed fails or returns no results, try scraping the front page
                if not topics:
                    logger.info("Attempting to scrape Hacker News front page as fallback")
                    try:
                        async with session.get("https://news.ycombinator.com/", timeout=15) as response:
                            if response.status == 200:
                                html = await response.text()
                                from bs4 import BeautifulSoup
                                soup = BeautifulSoup(html, 'html.parser')

                                # Find story titles and links
                                story_rows = soup.find_all('tr', class_='athing')
                                for row in story_rows[:20]:  # Get top 20 stories
                                    try:
                                        title_cell = row.find('td', class_='title')
                                        if not title_cell:
                                            continue

                                        title_link = title_cell.find('a')
                                        if not title_link:
                                            continue

                                        title = title_link.text.strip()

                                        # Skip very short titles or non-English content
                                        if len(title) < 15 or not self.is_english(title):
                                            continue

                                        url = title_link.get('href', '')
                                        if url and not url.startswith('http'):
                                            url = f"https://news.ycombinator.com/{url}"

                                        topics.append({
                                            'name': title,
                                            'description': f"Trending discussion on Hacker News about {title}",
                                            'url': url,
                                            'source': 'Hacker News',
                                            'type': 'tech_news',
                                            'score': 0
                                        })
                                    except Exception as e:
                                        logger.error(f"Error parsing Hacker News story: {e}")
                                        continue
                    except Exception as e:
                        logger.error(f"Error scraping Hacker News front page: {e}")

            logger.info(f"Found {len(topics)} topics from Hacker News")
            return topics

        except Exception as e:
            logger.error(f"Error getting Hacker News trends: {e}")
            return []

    async def _get_product_hunt_trends(self) -> List[Dict]:
        """Get trending topics from Product Hunt using web scraping (no API required)"""
        topics = []
        try:
            logger.info("Fetching trends from Product Hunt via web scraping")

            async with aiohttp.ClientSession(headers=self.headers) as session:
                try:
                    async with session.get("https://www.producthunt.com/", timeout=15) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')

                            # Find product cards
                            product_cards = soup.select('div[data-test="product-card"]')

                            # If the selector doesn't work, try alternative selectors
                            if not product_cards:
                                product_cards = soup.select('.styles_item__Dk_nz')

                            if not product_cards:
                                # Try more generic selectors
                                product_cards = soup.select('article') or soup.select('.product-item')

                            if not product_cards:
                                logger.warning("Could not find product cards on Product Hunt, trying alternative approach")
                                # Try to find any heading followed by a paragraph
                                headings = soup.find_all('h3')
                                for heading in headings:
                                    try:
                                        name = heading.text.strip()
                                        # Skip very short names or non-English content
                                        if len(name) < 3 or not self.is_english(name):
                                            continue

                                        # Try to find a description near the heading
                                        paragraph = heading.find_next('p')
                                        tagline = paragraph.text.strip() if paragraph else "New product"

                                        topics.append({
                                            'name': f"{name}: {tagline}",
                                            'description': f"New product on Product Hunt: {name} - {tagline}",
                                            'url': "https://www.producthunt.com/",
                                            'source': 'Product Hunt',
                                            'type': 'product',
                                        })
                                    except Exception as e:
                                        logger.error(f"Error parsing Product Hunt heading: {e}")
                                        continue

                            # Process the product cards if found
                            for card in product_cards:
                                try:
                                    # Extract product name and tagline
                                    name_elem = card.select_one('h3') or card.select_one('h2') or card.select_one('.product-name')
                                    tagline_elem = card.select_one('p') or card.select_one('.product-tagline')

                                    if name_elem:
                                        name = name_elem.text.strip()
                                        # Skip very short names or non-English content
                                        if len(name) < 3 or not self.is_english(name):
                                            continue

                                        tagline = tagline_elem.text.strip() if tagline_elem else "New product"

                                        # Try to find the URL
                                        link = card.find('a')
                                        url = link.get('href') if link else None
                                        if url and not url.startswith('http'):
                                            url = f"https://www.producthunt.com{url}"
                                        else:
                                            url = f"https://www.producthunt.com/posts/{name.lower().replace(' ', '-')}"

                                        topics.append({
                                            'name': f"{name}: {tagline}",
                                            'description': f"New product on Product Hunt: {name} - {tagline}",
                                            'url': url,
                                            'source': 'Product Hunt',
                                            'type': 'product',
                                        })
                                except Exception as e:
                                    logger.error(f"Error parsing Product Hunt card: {e}")
                                    continue
                except asyncio.TimeoutError:
                    logger.warning("Timeout fetching Product Hunt page")
                except Exception as e:
                    logger.error(f"Error fetching Product Hunt page: {e}")

            logger.info(f"Found {len(topics)} topics from Product Hunt")
            return topics

        except Exception as e:
            logger.error(f"Error getting Product Hunt trends: {e}")
            return []

    async def _get_github_trends(self) -> List[Dict]:
        """Get trending topics from GitHub using web scraping (no API required)"""
        topics = []
        try:
            logger.info("Fetching trends from GitHub via web scraping")

            async with aiohttp.ClientSession(headers=self.headers) as session:
                try:
                    async with session.get("https://github.com/trending", timeout=15) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')

                            # Find repository articles
                            repo_articles = soup.select('article.Box-row')

                            # If the selector doesn't work, try alternative selectors
                            if not repo_articles:
                                repo_articles = soup.select('.Box-row')

                            if not repo_articles:
                                # Try more generic selectors
                                repo_articles = soup.select('article') or soup.select('.repo-list-item')

                            if not repo_articles:
                                logger.warning("Could not find repository articles on GitHub, trying alternative approach")
                                # Try to find any repository links
                                repo_links = soup.select('a[href*="/"]')
                                for link in repo_links:
                                    try:
                                        href = link.get('href', '')
                                        # Check if it looks like a repository path
                                        if href and href.count('/') == 1 and href.startswith('/') and len(href) > 2:
                                            repo_path = href[1:]  # Remove leading slash

                                            # Skip non-English content
                                            if not self.is_english(repo_path):
                                                continue

                                            topics.append({
                                                'name': f"Trending GitHub Project: {repo_path}",
                                                'description': f"Trending GitHub repository: {repo_path}",
                                                'url': f"https://github.com{href}",
                                                'source': 'GitHub Trending',
                                                'type': 'development',
                                            })
                                    except Exception as e:
                                        logger.error(f"Error parsing GitHub trending link: {e}")
                                        continue

                            # Process the repository articles if found
                            for article in repo_articles:
                                try:
                                    # Extract repository name and description
                                    name_elem = article.select_one('h2 a') or article.select_one('h3 a') or article.select_one('a[href*="/"]')
                                    desc_elem = article.select_one('p')

                                    if name_elem:
                                        repo_path = name_elem.text.strip().replace('\n', '').replace(' ', '')

                                        # Skip empty or non-English repo names
                                        if not repo_path or not self.is_english(repo_path):
                                            continue

                                        # Clean up repo path
                                        repo_path = repo_path.replace('\n', '').replace(' ', '')

                                        # Get description
                                        description = desc_elem.text.strip() if desc_elem else f"Trending GitHub repository: {repo_path}"

                                        # Get URL
                                        href = name_elem.get('href')
                                        if href:
                                            if not href.startswith('http'):
                                                url = f"https://github.com{href}"
                                            else:
                                                url = href
                                        else:
                                            url = f"https://github.com/{repo_path}"

                                        topics.append({
                                            'name': f"Trending GitHub Project: {repo_path}",
                                            'description': description,
                                            'url': url,
                                            'source': 'GitHub Trending',
                                            'type': 'development',
                                        })
                                except Exception as e:
                                    logger.error(f"Error parsing GitHub trending repo: {e}")
                                    continue
                except asyncio.TimeoutError:
                    logger.warning("Timeout fetching GitHub trending page")
                except Exception as e:
                    logger.error(f"Error fetching GitHub trending page: {e}")

            logger.info(f"Found {len(topics)} topics from GitHub")
            return topics

        except Exception as e:
            logger.error(f"Error getting GitHub trends: {e}")
            return []

    async def _get_medium_trends(self) -> List[Dict]:
        """Get trending topics from Medium"""
        topics = []
        try:
            logger.info("Fetching trends from Medium")

            # Try different Medium topic feeds
            medium_topics = ['technology', 'programming', 'data-science', 'artificial-intelligence', 'startup', 'marketing']

            async with aiohttp.ClientSession(headers=self.headers) as session:
                for topic in medium_topics:
                    try:
                        url = f"https://medium.com/feed/topic/{topic}"
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.text()
                                feed = feedparser.parse(content)

                                for entry in feed.entries[:5]:  # Get top 5 articles per topic
                                    # Get summary safely
                                    summary = entry.get('summary', '')
                                    description = summary[:500] if summary else ''

                                    topics.append({
                                        'name': entry.title,
                                        'description': description,
                                        'url': entry.link,
                                        'source': f"Medium {topic}",
                                        'type': 'article',
                                        'published': entry.get('published', '')
                                    })
                    except Exception as e:
                        logger.error(f"Error fetching Medium topic {topic}: {e}")
                        continue

                    await asyncio.sleep(1)  # Be nice to Medium's servers

            logger.info(f"Found {len(topics)} topics from Medium")
            return topics

        except Exception as e:
            logger.error(f"Error getting Medium trends: {e}")
            return []

    async def get_github_trends(self) -> List[str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://github.com/trending', headers=self.headers) as response:
                    if response.status == 200:
                        soup = BeautifulSoup(await response.text(), 'html.parser')
                        trending_repos = []
                        
                        for repo in soup.select('article.Box-row'):
                            href_element = repo.select_one('h2 a')
                            if href_element and (href := href_element.get('href')):
                                if href and href.count('/') == 1 and href.startswith('/') and len(href) > 2:
                                    repo_path = href[1:]  # Remove leading slash
                                    trending_repos.append(repo_path)
                        
                        return trending_repos
            return []
        except Exception as e:
            logger.error(f"Error fetching GitHub trends: {e}")
            return []

    async def get_tech_news(self) -> List[Dict]:
        try:
            async with aiohttp.ClientSession() as session:
                news_items = []
                
                for feed_url in self.tech_news_feeds:
                    try:
                        async with session.get(feed_url, headers=self.headers) as response:
                            if response.status == 200:
                                feed_content = await response.text()
                                feed = feedparser.parse(feed_content)
                                
                                for entry in feed.entries[:5]:  # Get top 5 entries from each feed
                                    url = getattr(entry, 'link', None)
                                    if url and isinstance(url, str):
                                        if not url.startswith('http'):
                                            continue
                                            
                                        title = getattr(entry, 'title', '')
                                        if title and isinstance(title, str):
                                            try:
                                                lang = detect(title)
                                                if lang == 'en':  # Only include English articles
                                                    news_items.append({
                                                        'title': title,
                                                        'url': url
                                                    })
                                            except Exception as e:
                                                logger.warning(f"Language detection failed: {e}")
                                                continue
                    except Exception as e:
                        logger.error(f"Error processing feed {feed_url}: {e}")
                        continue
                        
                return news_items
        except Exception as e:
            logger.error(f"Error fetching tech news: {e}")
            return []

    def _is_excluded_topic(self, title: str) -> bool:
        """Check if topic should be excluded"""
        # Exclude very short titles
        if len(title) < 10:
            return True
            
        # Exclude non-alphanumeric titles
        if not any(c.isalnum() for c in title):
            return True
            
        # Exclude common spam patterns
        spam_patterns = [
            r'\b(sex|porn|xxx|dating)\b',
            r'\b(buy|sell|discount|offer)\b',
            r'\b(casino|gambling|betting)\b',
            r'\b(hack|crack|keygen)\b'
        ]
        
        return any(re.search(pattern, title.lower()) for pattern in spam_patterns)

    async def process_urls(self, urls: List[str]) -> List[Dict]:
        """Process a list of URLs to extract trending topics"""
        results = []
        async with aiohttp.ClientSession(headers=self.headers) as session:
            for url in urls:
                if not url:  # Skip None or empty strings
                    continue
                    
                try:
                    url_str = str(url).strip()
                    if not url_str:
                        continue
                        
                    if not url_str.startswith(('http://', 'https://')):
                        url_str = f'https://{url_str}'
                    
                    async with session.get(url_str) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            for link in soup.find_all('a'):
                                href = link.get('href')
                                if not href:  # Skip None or empty hrefs
                                    continue
                                    
                                href_str = str(href).strip()
                                if not href_str:
                                    continue
                                    
                                if href_str.count('/') == 1 and href_str.startswith('/') and len(href_str) > 2:
                                    repo_path = href_str[1:]  # Remove leading slash
                                    results.append({
                                        'url': f"{url_str}{href_str}",
                                        'title': repo_path,
                                        'timestamp': datetime.now().isoformat()
                                    })
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    continue
        
        return results

    async def process_raw_urls(self, urls: List[str]) -> List[str]:
        """Process raw URLs and ensure they're properly formatted"""
        processed_urls = []
        for url in urls:
            if not url:  # Skip None or empty strings
                continue
                
            url_str = str(url).strip()
            if not url_str:
                continue
                
            if not url_str.startswith(('http://', 'https://')):
                url_str = f'https://{url_str}'
            processed_urls.append(url_str)
        return processed_urls

    async def extract_topics(self) -> List[Dict]:
        """Main method to extract trending topics"""
        try:
            raw_urls = await self.get_urls()
            if not raw_urls:
                logger.warning("No URLs found to process")
                return []
                
            processed_urls = await self.process_raw_urls(raw_urls)
            if not processed_urls:
                logger.warning("No valid URLs to process")
                return []
                
            results = await self.process_urls(processed_urls)
            return results
            
        except Exception as e:
            logger.error(f"Error in extract_topics: {e}")
            return []

    async def continue_iteration(self) -> bool:
        """Check if we should continue iterating"""
        try:
            current_time = datetime.now()
            if not hasattr(self, 'last_run_time'):
                self.last_run_time = current_time
                return True
                
            time_diff = current_time - self.last_run_time
            should_continue = time_diff.total_seconds() >= self.iteration_interval
            
            if should_continue:
                self.last_run_time = current_time
                
            return should_continue
            
        except Exception as e:
            logger.error(f"Error checking iteration status: {e}")
            return False

# Main async function to continuously fetch trending topics
async def main():
    discoverer = TrendingTopicDiscoverer()
    
    def handle_shutdown(signum, frame):
        logger.info("Shutdown signal received, stopping gracefully...")
        asyncio.create_task(discoverer.stop())
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    await discoverer.start()
    retry_count = 0
    
    try:
        while await discoverer.should_continue():
            try:
                topics = await discoverer.get_trending_topics()
                if topics:
                    logger.info(f"Found {len(topics)} trending topics")
                    retry_count = 0  # Reset retry count on success
                else:
                    logger.warning("No trending topics found in this iteration")
                
                await asyncio.sleep(discoverer.iteration_interval)
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Error in trend discovery loop (attempt {retry_count}/{discoverer.max_retries}): {e}")
                
                if retry_count >= discoverer.max_retries:
                    logger.error("Max retries exceeded, stopping trend discovery")
                    break
                    
                await asyncio.sleep(discoverer.error_retry_delay)
    
    finally:
        await discoverer.stop()
        logger.info("Trend discovery process ended")

if __name__ == "__main__":
    asyncio.run(main())
