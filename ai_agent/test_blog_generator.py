import unittest
import asyncio
import aiohttp
import platform
import sys
from blog_generator import BlogGenerator
from models import Article
from utils import logger


class TestBlogGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = BlogGenerator()
        
        # Set up event loop for Windows
        if platform.system() == 'Windows':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    def tearDown(self):
        if hasattr(self, 'generator'):
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.generator.close())
            except Exception:
                pass

    async def async_test_generate_article(self):
        """Test article generation with proper error handling"""
        logger.debug("Testing article generation...")
        
        # Create a longer timeout for content generation
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
        
        # Create a test topic with more context
        test_topic = {
            'name': 'Artificial Intelligence in Content Creation',
            'description': 'A comprehensive guide to how AI is transforming content creation and digital marketing, including current applications, best practices, and future trends.',
            'type': 'technology',
            'context': 'Focus on practical applications and real-world examples'
        }

        try:
            # Generate article with explicit timeout
            article = await asyncio.wait_for(
                self.generator.generate_article(test_topic),
                timeout=300
            )
            
            # Detailed error checking
            if article is None:
                logger.error("Article generation failed")
                self.fail("Article generation returned None")

            self.assertIsInstance(article, Article)

            # Now we can safely access article attributes
            self.assertTrue(hasattr(article, 'content'), "Article missing content attribute")
            self.assertIsNotNone(article.content, "Article content is None")
            self.assertTrue(len(article.content.split()) >= 800, f"Content too short: {len(article.content.split())} words")
            self.assertTrue(len(article.title) > 0, "Article missing title attribute")
            self.assertTrue(len(article.meta_description) > 0, "Article missing meta_description attribute")

        except asyncio.TimeoutError:
            self.fail("Article generation timed out")
        except Exception as e:
            logger.error(f"Error in test: {e}")
            self.fail(f"Unexpected error: {str(e)}")
            
        finally:
            # Cleanup
            await self.generator.close()

    def test_generate_article(self):
        """Wrapper for async test"""
        print("\n=== STARTING TEST ===\n", file=sys.stderr)
        print("1. Initializing test...", file=sys.stderr)
        logger.debug("Running test for article generation...")
        
        try:
            print("2. Creating event loop...", file=sys.stderr)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            print("3. Running async test...", file=sys.stderr)
            result = loop.run_until_complete(self.async_test_generate_article())
            
            print("\n4. Test completed successfully", file=sys.stderr)
            print(f"Generated article length: {len(result.content.split())} words", file=sys.stderr)
            print(f"Article title: {result.title}", file=sys.stderr)
            return result
            
        except Exception as e:
            print(f"\nTEST FAILED: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            print("\n=== TEST FINISHED ===\n", file=sys.stderr)
            loop.close()


if __name__ == "__main__":
    unittest.main()
