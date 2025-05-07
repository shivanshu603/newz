from news_discovery import NewsDiscoverer
import asyncio
import unittest
from models import Article  # Assuming Article is defined in models.py

class TestNewsDiscoverer(unittest.TestCase):
    def setUp(self):
        self.discoverer = NewsDiscoverer()

    def test_discover_articles(self):
        """Test the discover_articles method"""
        articles = asyncio.run(self.discoverer.discover_articles())
        self.assertIsInstance(articles, list)  # Ensure it returns a list
        for article in articles:
            self.assertIsInstance(article, Article)  # Ensure each item is an Article

if __name__ == "__main__":
    unittest.main()
