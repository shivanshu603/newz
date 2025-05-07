from wordpress_integration import WordPressIntegration
from wordpress_integration import WordPressClient

from config import Config
from utils import logger
from typing import List

class WPManager:
    def __init__(self):
        self.wp_integration = WordPressIntegration()
        self.wp_client = WordPressClient()

        self.category_cache = {}

    def publish_content(self, title: str, content: str, categories: List[str], 
                       tags: List[str], featured_image: str = None, 
                       meta_description: str = None, schema_markup: str = None) -> bool:
        """Publish content to WordPress with enhanced features"""
        try:
            # Create categories if they don't exist
            category_ids = [self._get_or_create_category(cat) for cat in categories]
            
            # Prepare post data
            post_data = {
                'title': title,
                'content': content,
                'categories': category_ids,
                'tags': tags,
                'status': 'publish'
            }
            
            if featured_image:
                post_data['featured_media'] = self._upload_media(featured_image)
                
            if meta_description:
                post_data['meta'] = {
                    'description': meta_description
                }
                
            if schema_markup:
                post_data['meta']['schema_markup'] = schema_markup
                
            # Create post
            return self.wp.create_post(post_data)
            
        except Exception as e:
            logger.error(f"Error publishing content: {str(e)}")
            return False

    def generate_categories(self, topic: str) -> List[str]:
        """Generate relevant categories for a topic"""
        # Basic category generation logic
        main_category = topic.lower().replace(' ', '-')
        return [main_category, 'ai-generated', 'technology']

    def _get_or_create_category(self, category_name: str) -> int:
        """Get or create a WordPress category"""
        if category_name in self.category_cache:
            return self.category_cache[category_name]
            
        # Check if category exists
        category_id = self.wp.get_category_id(category_name)
        if category_id:
            self.category_cache[category_name] = category_id
            return category_id
            
        # Create new category
        new_category = self.wp.create_category(category_name)
        if new_category:
            self.category_cache[category_name] = new_category['id']
            return new_category['id']
            
        return 0

    def _upload_media(self, image_url: str) -> int:
        """Upload media to WordPress"""
        try:
            media = self.wp.upload_media(image_url)
            return media['id']
        except Exception as e:
            logger.error(f"Error uploading media: {str(e)}")
            return 0
