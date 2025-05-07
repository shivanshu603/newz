import unittest
from news_discovery import NewsDiscoverer

class TestNewsDiscoverer(unittest.TestCase):
    def setUp(self):
        self.discoverer = NewsDiscoverer()

    def test_discover_articles(self):
        articles = self.discoverer.discover_articles()
        self.assertIsInstance(articles, list)
        self.assertGreater(len(articles), 0)

    def test_get_trending_topics(self):
        topics = self.discoverer.get_trending_topics()
        self.assertIsInstance(topics, list)
        self.assertGreater(len(topics), 0)

if __name__ == '__main__':
    unittest.main()
