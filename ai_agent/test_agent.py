import asyncio
from continuous_operation import ContinuousOperator
from utils import logger
import traceback

async def main():
    try:
        logger.info("Starting test of AI agent...")
        operator = ContinuousOperator()
        
        # Test with a sample topic
        test_topic = {
            "name": "Top 10 AI Tools That Will Transform Your Content Creation in 2025",
            "description": "Exploring the most innovative AI tools that will revolutionize content creation in 2025",
            "category": "Technology",
            "source": "test"
        }
        
        logger.info(f"Testing article generation with topic: {test_topic['name']}")
        article = await operator.generate_article(test_topic)
        
        if article:
            logger.info(f"Successfully generated article: {article.title}")
            logger.info(f"Content length: {len(article.content)} characters")
            print("\n--- Article Preview ---")
            print(f"Title: {article.title}")
            print(f"Meta: {article.meta_description}")
            print(f"Content (first 500 chars): {article.content[:500]}...")
        else:
            logger.error("Failed to generate article")
            
    except Exception as e:
        logger.error(f"Error in test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 