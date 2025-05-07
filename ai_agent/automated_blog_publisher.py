import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from utils import logger
from config import Config
from models import Article as ModelArticle
from blog_generator import BlogGenerator
from rss_feed_extractor import extract_topics_from_rss
from google_trends_extractor import extract_trending_topics
from twitter_topic_extractor import extract_topics_from_twitter
from wordpress_integration import WordPressClient
from seo_optimizer import SEOOptimizer
from news_monitor import NewsMonitor
from image_scraper import scrape_images

# Use the Article class from models.py but extend it with additional methods
# This avoids having two different Article classes
Article = ModelArticle

# Add dictionary-style access methods to the Article class
def article_getitem(self, key: str) -> Any:
    """Enable dictionary-style access to attributes"""
    if not hasattr(self, key):
        return None
    return getattr(self, key)

# Add the methods to the Article class
setattr(Article, '__getitem__', article_getitem)

# The get method is already defined in the Article class in models.py

class BlogPublisher:
    """Automated blog publisher with enhanced workflow and monitoring."""

    def __init__(self):
        self.blog_generator = BlogGenerator()
        self.seo_optimizer = SEOOptimizer()
        self.news_monitor = NewsMonitor()
        self.wp_client = WordPressClient()
        self.is_running = False

    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Initialize news monitor
            await self.news_monitor.initialize()
            logger.info("Automated blog publisher initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize blog publisher: {e}")
            return False

    async def publish_from_article(self, article: Optional[Article]) -> Optional[int]:
        """Publish a blog post from a news article"""
        if not article:
            logger.error("No article provided")
            return None

        try:
            blog_post = await self.blog_generator.generate_article(article)
            if not blog_post or not blog_post.get('status') == 'success':
                logger.error("Failed to generate blog content.")
                return None

            # Access article attributes safely
            article_data = {
                'title': getattr(article, 'title', ''),
                'content': getattr(article, 'content', ''),
                'keywords': getattr(article, 'keywords', []),
                'meta': {
                    'description': getattr(article, 'meta_description', ''),
                    'keywords': getattr(article, 'keywords', [])
                }
            }

            # Extract topics from RSS feeds, Google Trends, and Twitter
            rss_topics = await extract_topics_from_rss()
            trending_topics = await extract_trending_topics()
            twitter_topics = await extract_topics_from_twitter()

            # Combine all topics for processing
            all_topics = rss_topics + trending_topics + twitter_topics

            # Generate blog content using the first topic
            if all_topics:
                blog_post = await self.blog_generator.generate_article(all_topics[0])

                if not blog_post or not isinstance(blog_post, dict) or blog_post.get('status') != 'success':
                    logger.error("Failed to generate blog content.")
                    return None

                # Optimize SEO for the generated blog post
                blog_content = blog_post.get('content', '')
                seo_report = self.seo_optimizer.generate_seo_report(
                    blog_content,
                    keywords=[getattr(article, 'title', '')]
                )
                if seo_report.get('status') != 'success':
                    logger.error("SEO optimization failed.")
                    return None
                
                # Scrape images
                article_title = getattr(article, 'title', '')
                images = scrape_images(article_title, limit=5)
                if not images:
                    logger.warning("No images found for the article.")
                
                # Publish to WordPress with images
                post_id = await self.wp_client.publish_post(
                    title=blog_post.get('title', ''),
                    content=blog_post.get('content', ''),
                    tags=blog_post.get('tags', []),
                    categories=blog_post.get('categories', []),
                    images=images
                )
                if not post_id:
                    logger.error("Failed to publish post to WordPress.")
                    return None

                logger.info(f"Successfully published blog post with ID: {post_id} at {datetime.now()}.")
                return post_id

        except Exception as e:
            logger.error(f"Failed to publish blog post at {datetime.now()}: {e}")
            return None

    async def run(self):
        """Run the blog publisher continuously."""
        self.is_running = True
        logger.info("Blog publisher started.")

        while self.is_running:
            try:
                # Fetch new articles
                articles = await self.news_monitor.get_latest_articles()
                if not articles:
                    logger.info("No new articles found.")
                    await asyncio.sleep(Config.NEWS_CHECK_INTERVAL)
                    continue

                # Publish articles
                for article in articles:
                    await self.publish_from_article(article)

                await asyncio.sleep(Config.PUBLISH_INTERVAL)

            except Exception as e:
                logger.error(f"Error in blog publisher: {e}")
                await asyncio.sleep(Config.RETRY_INTERVAL)

def _handle_article(article: Optional[Article]) -> Dict[str, Any]:
    """Safely handle article data with proper null checks"""
    if not article:
        return {}
        
    return {
        'title': article.title,
        'content': article.content,
        'keywords': article.keywords,
        'meta_description': article.meta_description,
        'images': article.images or []
    }
