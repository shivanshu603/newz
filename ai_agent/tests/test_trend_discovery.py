import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trend_discovery import TrendDiscovery
from utils import logger

async def test_sources():
    discoverer = TrendDiscovery()
    
    # Test Google Trends
    print("\n=== Testing Google Trends ===")
    google_trends = await discoverer._get_google_trends()
    print(f"Found {len(google_trends)} Google trends")
    for trend in google_trends[:3]:
        print(f"- {trend['name']} (Score: {trend.get('interest_score', 'N/A')})")

    # Test Reddit Trends
    print("\n=== Testing Reddit Trends ===")
    reddit_trends = await discoverer._get_reddit_trends()
    print(f"Found {len(reddit_trends)} Reddit trends")
    for trend in reddit_trends[:3]:
        print(f"- {trend['name']} (Subreddit: {trend.get('subreddit')})")

    # Test Twitter Trends
    print("\n=== Testing Twitter Trends ===")
    twitter_trends = await discoverer._get_twitter_trends()
    print(f"Found {len(twitter_trends)} Twitter trends")
    for trend in twitter_trends[:3]:
        print(f"- {trend['name']}")

    # Test YouTube Trends
    print("\n=== Testing YouTube Trends ===")
    youtube_trends = await discoverer._get_youtube_trends()
    print(f"Found {len(youtube_trends)} YouTube trends")
    for trend in youtube_trends[:3]:
        print(f"- {trend['name']}")

    # Test RSS Feeds
    print("\n=== Testing RSS Feeds ===")
    rss_topics = await discoverer._get_rss_topics(5)
    print(f"Found {len(rss_topics)} RSS topics")
    for topic in rss_topics[:3]:
        print(f"- {topic['name']} (Score: {topic.get('score', 'N/A')})")

    # Test Combined Topics
    print("\n=== Testing Combined Topics ===")
    all_topics = await discoverer.get_next_topics(10)
    print(f"Found {len(all_topics)} total topics")
    print("\nTop 5 topics:")
    for topic in all_topics[:5]:
        print(f"- {topic['name']} (Source: {topic.get('source', 'unknown')})")

    return all_topics

if __name__ == "__main__":
    all_topics = asyncio.run(test_sources())
