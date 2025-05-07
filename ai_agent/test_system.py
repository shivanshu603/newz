import asyncio
import logging
from blog_generator import BlogGenerator
from wordpress_integration import WordPressClient
from config import Config
from utils import logger

async def test_wordpress_connection():
    """Test WordPress connectivity"""
    wp_client = WordPressClient()
    if await wp_client.verify_connection():
        logger.info("WordPress connection successful!")
        return True
    logger.error("WordPress connection failed!")
    return False

async def test_content_generation():
    """Test content generation"""
    generator = BlogGenerator()
    for topic in Config.TEST_TOPICS:
        logger.info(f"Testing content generation for: {topic['name']}")
        article = await generator.generate_article(topic)
        if article:
            logger.info("Content generation successful!")
            logger.info(f"Title: {article.title}")
            logger.info(f"Content preview: {article.content[:200]}")
            return True
    return False

async def main():
    """Run system tests"""
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Test WordPress connection
        logger.info("Testing WordPress connection...")
        wp_connected = await test_wordpress_connection()
        
        # Test content generation
        logger.info("Testing content generation...")
        content_generated = await test_content_generation()
        
        if wp_connected and content_generated:
            logger.info("All systems operational!")
        else:
            logger.error("System check failed!")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
