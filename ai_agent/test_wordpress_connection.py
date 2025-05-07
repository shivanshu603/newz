import asyncio
from wordpress_integration import WordPressClient

async def test_wordpress_connection():
    client = WordPressClient()
    is_connected = await client.connect()
    if is_connected:
        print("Successfully connected to WordPress API.")
    else:
        print("Failed to connect to WordPress API.")

if __name__ == "__main__":
    asyncio.run(test_wordpress_connection())
