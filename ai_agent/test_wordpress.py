import asyncio
from blog_publisher import BlogPublisher
from config import Config
from utils import logger

async def test_wordpress_connection():
    publisher = BlogPublisher(
        wp_url=Config.WORDPRESS_SITE_URL,
        wp_username=Config.WORDPRESS_USERNAME,
        wp_password=Config.WORDPRESS_PASSWORD
    )
    
    logger.info("Testing WordPress connection...")
    if await publisher._verify_connection():
        logger.info("WordPress connection successful!")
        return True
    else:
        logger.error("WordPress connection failed!")
        return False

if __name__ == "__main__":
    asyncio.run(test_wordpress_connection())
