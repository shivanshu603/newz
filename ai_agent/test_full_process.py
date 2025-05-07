import asyncio
from blog_generator import BlogGenerator
from blog_publisher import BlogPublisher
from config import Config
from utils import logger
from models import Article
from datetime import datetime

async def test_full_process():
    # 1. Create test topic
    test_topic = {
        'name': 'Test Article: Technology Impact',
        'description': 'A test article about the impact of technology on modern society',
        'type': 'test',
        'categories': ['Technology', 'Society']
    }

    # 2. Initialize components
    generator = BlogGenerator()
    publisher = BlogPublisher(
        wp_url=Config.WORDPRESS_SITE_URL,
        wp_username=Config.WORDPRESS_USERNAME,
        wp_password=Config.WORDPRESS_PASSWORD
    )

    try:
        # 3. Verify WordPress connection
        logger.info("Verifying WordPress connection...")
        if not await publisher._verify_connection():
            logger.error("WordPress connection failed!")
            return

        # 4. Generate article
        logger.info("Generating test article...")
        article = await generator.generate_article(test_topic)
        if not article:
            logger.error("Failed to generate article")
            return

        logger.info(f"Generated article with title: {article.title}")
        logger.info(f"Content preview: {article.content[:200]}...")

        # 5. Publish article
        logger.info("Attempting to publish article...")
        success = await publisher.publish_article(article)
        
        if success:
            logger.info("🎉 Test successful! Article was published")
        else:
            logger.error("Failed to publish article")

    except Exception as e:
        logger.error(f"Error during test: {e}")

if __name__ == "__main__":
    logger.info("Starting full process test...")
    asyncio.run(test_full_process())
