import asyncio
import argparse
import logging
from blog_generator_new import BlogGenerator
from models import Article

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_content_generation(topic: str = "Artificial Intelligence"):
    """Test article generation with a specific topic"""
    try:
        logger.info(f"Testing article generation for topic: {topic}")
        
        # Initialize the blog generator
        generator = BlogGenerator()
        
        # Create a topic dictionary
        topic_dict = {
            'name': topic,
            'description': f"A comprehensive guide to {topic}",
            'type': 'guide'
        }
        
        # Generate an article
        logger.info("Generating article...")
        article = await generator.generate_article(topic_dict)
        
        if article:
            logger.info(f"Successfully generated article: {article.title}")
            logger.info(f"Content length: {len(article.content)} characters, {len(article.content.split())} words")
            
            # Print article details
            print("\n" + "="*50)
            print(f"TITLE: {article.title}")
            print("="*50)
            print(f"META DESCRIPTION: {article.meta_description}")
            print("-"*50)
            print("CONTENT PREVIEW (first 500 chars):")
            print(article.content[:500] + "...")
            print("-"*50)
            print(f"KEYWORDS: {', '.join(article.keywords[:10])}")
            print("="*50)
            
            # Save the article to a file
            filename = f"{topic.replace(' ', '_').lower()}_article.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {article.title}\n\n")
                f.write(article.content)
            
            logger.info(f"Article saved to {filename}")
            return True
        else:
            logger.error("Failed to generate article")
            return False
            
    except Exception as e:
        logger.error(f"Error in test_content_generation: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test blog article generation')
    parser.add_argument('--topic', type=str, default="Artificial Intelligence",
                      help='Topic for the article generation')
    args = parser.parse_args()
    
    asyncio.run(test_content_generation(args.topic))
