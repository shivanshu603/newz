import asyncio
import logging
from blog_generator import BlogGenerator
from wordpress_integration import WordPressClient
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def quick_test():
    """Run a quick test of content generation and WordPress integration"""
    try:
        logger.info("Starting quick test...")
        
        # Initialize generator with timeout
        generator = BlogGenerator()
        logger.info("BlogGenerator initialized")
        
        # Simple test topic
        test_topic = {
            "name": "Test Topic",
            "keywords": ["test"],
            "trending": True,
            "published_at": datetime.now().isoformat()
        }
        
        logger.info("Generating test article...")
        
        # Add timeout to article generation
        try:
            async with asyncio.timeout(60):  # 60 second timeout
                article = await generator.generate_article(test_topic)
        except asyncio.TimeoutError:
            logger.error("Article generation timed out")
            return
        
        if article:
            logger.info("Article generated successfully")
            logger.info(f"Title: {article.title}")
            logger.info(f"Content preview: {article.content[:200]}...")
        else:
            logger.error("No article was generated")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.exception(e)

if __name__ == "__main__":
    asyncio.run(quick_test())
