import os
import logging
import aiohttp
import base64
import requests
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class WordPressClient:
    """Client for interacting with WordPress via REST API."""
    
    def __init__(self):
        """Initialize the WordPress client with credentials from environment variables."""
        self.url = os.getenv("WORDPRESS_URL")
        self.username = os.getenv("WORDPRESS_USERNAME")
        self.password = os.getenv("WORDPRESS_PASSWORD")
        
        logger.info("WordPress credentials loaded successfully.")
        if not self.url or not self.username or not self.password:

            logger.error("WordPress credentials are missing. Please check your .env file.")
            raise ValueError("WordPress credentials are not set. Please check your .env file.")
        
        # Initialize REST API credentials
        self.rest_url = self.url + "wp-json/wp/v2/"
        self.auth = (self.username, self.password)
        
    async def connect(self) -> bool:
        """Establish a connection to the WordPress API."""
        return await self.verify_connection()

    async def verify_connection(self) -> bool:
        """Verify WordPress connection with proper error handling."""
        try:
            async with aiohttp.ClientSession() as session:
                # Test authentication first
                auth_url = f"{self.url}/wp-json/wp/v2/users/me"
                headers = {'Authorization': f'Basic {self._get_auth_header()}'}
                
                async with session.get(auth_url, headers=headers) as response:
                    if response.status == 401:
                        logger.error("WordPress authentication failed. Check credentials.")
                        return False
                    elif response.status != 200:
                        logger.error(f"WordPress API error: {response.status}")
                        return False
                    
                    logger.info("Successfully authenticated with WordPress.")
                    return True
        except aiohttp.ClientConnectorError:
            logger.error(f"Could not connect to WordPress site: {self.url}")
            return False
        except Exception as e:
            logger.error(f"WordPress connection error: {str(e)}")
            return False

    async def upload_image(self, image_path: str) -> Optional[int]:
        """Upload an image to WordPress and return the image ID."""
        try:
            async with aiohttp.ClientSession() as session:
                with open(image_path, 'rb') as img:
                    image_data = img.read()
                headers = {
                    'Authorization': f'Basic {self._get_auth_header()}',
                    'Content-Type': 'image/jpeg'  # Adjust based on image type
                }
                response = await session.post(
                    f"{self.url}/wp-json/wp/v2/media",
                    headers=headers,
                    data=image_data
                )
                if response.status in (200, 201):
                    data = await response.json()
                    return data.get('id')
                else:
                    logger.error(f"Failed to upload image: {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            return None

    async def publish_post(self, title: str, content: str, tags: List[str], categories: List[str], images: List[str]) -> Optional[int]:
        """Publish a post to WordPress."""
        try:
            # Upload images and get their IDs
            image_ids = []
            for image in images:
                image_id = await self.upload_image(image)
                if image_id:
                    image_ids.append(image_id)

            post_data = {
                'title': title,
                'content': content,
                'status': 'publish',
                'categories': [await self._get_category_id(cat) for cat in categories],
                'tags': [await self._get_tag_id(tag) for tag in tags],
                'featured_media': image_ids[0] if image_ids else None  # Set featured image if available
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.url}/wp-json/wp/v2/posts",
                    json=post_data,
                    headers={'Authorization': f'Basic {self._get_auth_header()}'}
                ) as response:
                    if response.status in (200, 201):
                        data = await response.json()
                        return data.get('id')
                    else:
                        logger.error(f"Failed to publish post: {await response.text()} - Status Code: {response.status}") 

                        return None
        except Exception as e:
            logger.error(f"Error publishing post: {e}")
            return None

    async def get_categories(self) -> List[Dict]:
        """Get WordPress categories."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.url}/wp-json/wp/v2/categories",
                    headers={'Authorization': f'Basic {self._get_auth_header()}'}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return []
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    async def get_post(self, post_id: int) -> Optional[Dict]:
        """Get post by ID."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.url}/wp-json/wp/v2/posts/{post_id}",
                    headers={'Authorization': f'Basic {self._get_auth_header()}'}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"Error getting post: {e}")
            return None

    def _get_auth_header(self) -> str:
        """Get base64 encoded auth header."""
        auth_string = f"{self.username}:{self.password}"
        return base64.b64encode(auth_string.encode()).decode()
                
    async def _get_category_id(self, category_name: str) -> Optional[int]:
        """Get category ID by name."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.url}/wp-json/wp/v2/categories"
                headers = {'Authorization': f'Basic {self._get_auth_header()}'}
                params = {'search': category_name}
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data:
                            return data[0]['id']
                    return None
        except Exception as e:
            logger.error(f"Error getting category ID: {e}")
            return None
            
    async def _get_tag_id(self, tag_name: str) -> Optional[int]:
        """Get tag ID by name."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.url}/wp-json/wp/v2/tags"
                headers = {'Authorization': f'Basic {self._get_auth_header()}'}
                params = {'search': tag_name}
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data:
                            return data[0]['id']
                    return None
        except Exception as e:
            logger.error(f"Error getting tag ID: {e}")
            return None
