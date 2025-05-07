import asyncio
from wordpress_integration import WordPressClient

async def test_publish_article():
    wp_client = WordPressClient()
    # No need to initialize for basic auth


    # Test data for publishing
    title = "Test Article"
    content = "This is a test article to verify publishing functionality."
    tags = [1]  # Replace with actual tag ID
    categories = [1]  # Replace with actual category ID


    # Attempt to publish the article
    post_id = await wp_client.publish_post(title, content, tags=tags, categories=categories)

    if post_id:
        print(f"Article published successfully with ID: {post_id}")
    else:
        print("Failed to publish article.")

if __name__ == "__main__":
    asyncio.run(test_publish_article())
