import schedule
import time
import asyncio
from datetime import datetime
from blog_generator import BlogGenerator
from trending_topic_discoverer import TrendingTopicDiscoverer
from blog_publisher import BlogPublisher
from models import Session
from config import Config
from utils import logger
from typing import Dict



class BlogScheduler:
    def __init__(self):
        self.generator = BlogGenerator()
        self.topic_discoverer = TrendingTopicDiscoverer()
        self.publisher = BlogPublisher(
            wp_url=Config.WORDPRESS_SITE_URL,
            wp_username=Config.WORDPRESS_USERNAME,
            wp_password=Config.WORDPRESS_PASSWORD
        )




    async def generate_and_publish_blog(self):
        """Generate and immediately publish a single blog with relevant images"""
        try:
            # Get trending topic
            topic = await self._get_trending_topic()
            
            # Generate blog content
            article = await self.generator.generate_article(topic)
            if not article:
                logger.error("Failed to generate article")
                return False
            
            # Get relevant images for the content
            from image_scraper import ImageScraper
            scraper = ImageScraper()
            images = await scraper.get_relevant_images(topic, num_images=3)
            if images:
                article.images = images
            
            # Publish immediately with images
            if await self.publisher.publish_article(article):
                logger.info(f"Successfully published article: {article.title}")
                return True
            return False
                
        except Exception as e:
            logger.error(f"Error in article generation/publishing: {str(e)}")
            return False



    async def _get_trending_topic(self) -> Dict:
        """Get a trending topic for blog generation"""
        try:
            topics = await self.topic_discoverer.get_trending_topics()
            if topics:
                return topics[0]  # Use first trending topic
            raise ValueError("No trending topics found")
        except Exception as e:
            logger.error(f"Error getting trending topics: {str(e)}")
            raise  # Re-raise the exception instead of using fallback



    async def start_immediate_publishing(self):
        """Start immediate blog generation and publishing"""
        logger.info("Starting immediate blog publishing")
        while True:
            try:
                # Generate and publish one blog at a time
                if await self.generate_and_publish_blog():
                    # Wait before next article
                    await asyncio.sleep(100)  # 5 minutes between articles
            except Exception as e:
                logger.error(f"Error in publishing loop: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying



if __name__ == "__main__":
    scheduler = BlogScheduler()
    asyncio.run(scheduler.start_immediate_publishing())
