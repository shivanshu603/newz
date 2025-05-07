import asyncio
from trend_discovery import TrendDiscovery
import json

async def test_trend_discovery():
    try:
        discoverer = TrendDiscovery()
        
        print("\n=== Testing Trend Discovery ===")
        
        # Test each source individually
        print("\n1. Testing Google Trends...")
        google_trends = await discoverer._get_google_trends()
        print(f"Found {len(google_trends)} Google Trends:")
        for trend in google_trends[:5]:
            print(f"- {trend['name']} (Interest Score: {trend.get('interest_score', 'N/A')})")

        print("\n2. Testing Reddit Trends...")
        reddit_trends = await discoverer._get_reddit_trends()
        print(f"Found {len(reddit_trends)} Reddit Trends:")
        for trend in reddit_trends[:5]:
            print(f"- {trend['name']} (Subreddit: {trend.get('subreddit', 'N/A')})")

        print("\n3. Testing Combined Topics...")
        all_topics = await discoverer.get_next_topics(10)
        print(f"\nFound {len(all_topics)} total topics:")
        for topic in all_topics[:5]:
            print(f"- {topic['name']} (Source: {topic.get('source', 'unknown')}, Score: {topic.get('score', 'N/A')})")

        return all_topics

    except Exception as e:
        print(f"Error during testing: {e}")
        return None

if __name__ == "__main__":
    all_topics = asyncio.run(test_trend_discovery())
    
    # Save results to file for inspection
    if all_topics:
        with open('trend_results.json', 'w') as f:
            json.dump(all_topics, f, indent=2)
        print("\nResults saved to trend_results.json")
