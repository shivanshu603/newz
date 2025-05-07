import unittest
from scraper import NewsScraper

class TestNewsScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = NewsScraper()

    def test_scrape_source(self):
        # Test with a known RSS feed URL
        url = "https://example.com/rss"  # Replace with a valid RSS feed URL for testing
        articles = self.scraper.scrape_source(url)
        self.assertIsInstance(articles, list)
        self.assertGreater(len(articles), 0)

    def test_scrape_article(self):
        # Test with a known article URL
        url = "https://example.com/article"  # Replace with a valid article URL for testing
        article = self.scraper.scrape_article(url)
        self.assertIsInstance(article, dict)
        self.assertIn('title', article)
        self.assertIn('content', article)

if __name__ == '__main__':
    unittest.main()
