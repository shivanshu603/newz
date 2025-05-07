import asyncio
import unittest
from models import Article
from blog_publisher import BlogPublisher
from config import Config

class TestBlogPublisher(unittest.TestCase):
    def setUp(self):
        self.publisher = BlogPublisher(
            wp_url=Config.WORDPRESS_SITE_URL,
            wp_username=Config.WORDPRESS_USERNAME,
            wp_password=Config.WORDPRESS_PASSWORD
        )
        self.sample_article = Article(
            title="Test Article",
            content="This is a test article content.",
            meta_description="Test article description",
            categories=[1],
            tags=[1, 2],
            slug="test-article"
        )

    async def test_verify_connection(self):
        result = await self.publisher.verify_connection()
        self.assertTrue(result, "Failed to verify WordPress connection")

    async def test_publish_article(self):
        result = await self.publisher.publish_article(self.sample_article)
        self.assertTrue(result, "Failed to publish the test article")

if __name__ == "__main__":
    asyncio.run(unittest.main())
