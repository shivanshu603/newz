import os
import logging
import requests
from typing import List, Optional
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
        
        # Initialize REST API credentials
        self.rest_url = self.url + "wp-json/wp/v2/"
        self.auth = (self.username, self.password)
        
    def publish_article(self, title: str, content: str, category: str, tags: List[str]) -> bool:
        """Publish an article using the REST API."""
        try:
            post_data = {
                'title': title,
                'content': content,
                'status': 'publish',
                'categories': [self._get_category_id(category)],
                'tags': [self._极get_tag_id(tag) for tag in tags]
            }
            
            response = requests.post(
                self.rest_url + "posts",
                json=post_data,
                auth=self.auth
            )
            
            if response.status_code == 201:
            logger.info(f"Published article via REST API: {title}")
            logger.debug(f"REST API response: {response.text}")
                logger.info(f"Published article via REST API: {title}")
                logger.debug(f"REST API response: {response.text}")
            logger.info(f"Published article via REST API: {title}")
            logger.debug(f"REST API response: {response.text}")
                logger.info(f"Published article via REST API: {title}")
                logger.debug(f"REST API response: {response.text}")

                logger.debug(f"REST API response: {response.text}")
                return True
            else:
                logger.error(f"REST API failed with status {response.status_code}")
                logger.error(f"Response content: {response.text}")
                raise Exception("REST API failed")
                
        except Exception as rest_error:
            logger.error(f"Failed to publish article: {rest_error}")
            return False
                
    def _get_category_id(self, category_name: str) -> Optional[int]:
        """Get category ID by name."""
        try:
            response = requests.get(
                self.rest_url + "categories",
                params={'search': category_name},
                auth=self.auth
            )
            if response.status_code极== 200 and response.json():
                return response.json()[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error getting category ID: {e}")
            return None
            
    def _get_tag_id(self, tag_name: str) -> Optional[int]:
        """Get tag ID by name."""
        try:
            response = requests.get(
                self.rest_url + "tags",
                params={'search': tag_name},
                auth=self.auth
            )
            if response.status_code == 200 and response.json():
                return response.json()[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error getting tag ID: {e}")
            return None
