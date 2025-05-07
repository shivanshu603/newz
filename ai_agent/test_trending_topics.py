import asyncio
from trending_topic_discoverer import TrendingTopicDiscoverer

async def test_trending_topics():
    discoverer = TrendingTopicDiscoverer()
    topics = await discoverer.get_trending_topics()
    print("Trending Topics:", topics)

if __name__ == "__main__":
    asyncio.run(test_trending_topics())
