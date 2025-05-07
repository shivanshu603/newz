import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from utils import logger
from models import Article
import aiohttp
from config import Config
import re
import json
from utils.content_humanizer import ContentHumanizer
import time  # Added missing import for time

from category_detector import CategoryDetector
from content_formatter import ContentFormatter

class BlogPublisher:
    def __init__(self, wp_url: str, wp_username: str, wp_password: str):
        self.wp_url = wp_url or Config.WORDPRESS_SITE_URL
        self.wp_username = wp_username or Config.WORDPRESS_USERNAME
        self.wp_password = wp_password or Config.WORDPRESS_PASSWORD
        self.session = None
        self.retry_count = 0
        self.max_retries = Config.MAX_RETRIES
        self.retry_delay = Config.RETRY_DELAY
        
        # Initialize the content humanizer
        try:
            self.humanizer = ContentHumanizer()
            logger.info("ContentHumanizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ContentHumanizer: {e}")
            self.humanizer = None

        # Initialize CategoryDetector and ContentFormatter
        self.category_detector = CategoryDetector()
        self.content_formatter = ContentFormatter()



    async def cleanup(self):
        """Cleanup resources such as aiohttp session"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                logger.info("Closed aiohttp session successfully")
        except Exception as e:
            logger.error(f"Error during BlogPublisher cleanup: {e}")


    async def _init_session(self) -> bool:
        """Initialize aiohttp session with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                if self.session is None:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.get(f"{self.wp_url}/wp-json/wp/v2/posts", auth=aiohttp.BasicAuth(self.wp_username, self.wp_password)) as response:
                    if response.status == 200:
                        logger.info("Successfully initialized WordPress session")
                        return True
                    else:
                        logger.error(f"Failed to initialize session, status: {response.status}")
                
            except Exception as e:
                logger.error(f"Session initialization error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if self.session:
                    await self.session.close()
                    self.session = None
                
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return False

    async def verify_connection(self) -> bool:
        """Verify WordPress connection with enhanced error handling"""
        if not await self._init_session() or not self.session:
            return False

        try:
            test_url = f"{self.wp_url}/wp-json/"
            async with self.session.get(test_url) as response:
                if response.status == 200:
                    logger.info("WordPress connection verified successfully")
                    
                    # Check if the theme supports post-thumbnails (featured images)
                    await self._ensure_theme_supports_featured_images()
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"WordPress connection failed: Status {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error verifying WordPress connection: {str(e)}")
            return False
            
    async def _ensure_theme_supports_featured_images(self) -> bool:
        """Check if the current theme supports featured images and try to enable if not"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot check theme support: session not initialized")
            return False
            
        try:
            # First check if the theme already supports post-thumbnails
            auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
            
            # Get current theme
            async with self.session.get(f"{self.wp_url}/wp-json/wp/v2/themes?status=active", auth=auth) as response:
                if response.status == 200:
                    themes_data = await response.json()
                    if themes_data:
                        active_theme = next(iter(themes_data.values()))
                        theme_supports = active_theme.get('theme_supports', {})
                        
                        if theme_supports.get('post-thumbnails', False):
                            logger.info("Current theme already supports featured images")
                            return True
                        else:
                            logger.warning("Current theme may not support featured images")
                            
                            # Try to enable theme support via REST API if possible
                            # Note: This might not work on all WordPress installations as it requires specific permissions
                            try:
                                # This is a custom endpoint that might not exist on all WordPress installations
                                custom_endpoint = f"{self.wp_url}/wp-json/custom/v1/enable-featured-images"
                                async with self.session.post(custom_endpoint, auth=auth) as enable_response:
                                    if enable_response.status in (200, 201):
                                        logger.info("Successfully enabled featured images support")
                                        return True
                            except Exception as e:
                                logger.warning(f"Could not enable featured images via API: {e}")
                            
                            # Log instructions for manual enabling
                            logger.warning("To enable featured images manually, go to WordPress admin -> Appearance -> Theme Editor -> functions.php and add: add_theme_support('post-thumbnails');")
                            return False
                    else:
                        logger.warning("No active theme data found")
                        return False
                else:
                    logger.error(f"Could not check theme support. Status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error checking theme support: {e}")
            return False

    async def upload_image_from_url(self, image_url: str, article_title: str = None) -> Optional[int]:
        """Download image from URL and upload to WordPress media library, return media ID"""
        try:
            # Skip if image URL is None or empty
            if not image_url or not isinstance(image_url, str):
                logger.warning(f"Empty or invalid image URL provided: {image_url}")
                return None
                
            # Basic URL validation
            if not (image_url.startswith('http://') or image_url.startswith('https://')):
                logger.warning(f"Invalid image URL format: {image_url}")
                return None
                
            # Skip URLs from known problematic sources
            blocked_domains = ['placeholder.com', 'example.com', 'test.com']
            if any(domain in image_url.lower() for domain in blocked_domains):
                logger.warning(f"Skipping image from blocked domain: {image_url}")
                return None
                
            logger.info(f"Attempting to download image from URL: {image_url}")
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(image_url, timeout=10) as response:
                        if response.status != 200:
                            logger.error(f"Failed to download image from {image_url}, status {response.status}")
                            return None
                            
                        # Check content type to ensure it's an image
                        content_type = response.headers.get('Content-Type', '')
                        if not content_type.startswith('image/'):
                            logger.error(f"URL does not point to an image. Content-Type: {content_type}")
                            return None
                            
                        # Check file size (avoid tiny or huge images)
                        content_length = int(response.headers.get('Content-Length', 0))
                        if content_length > 0:
                            if content_length < 10000:  # Less than 10KB
                                logger.warning(f"Image too small ({content_length} bytes), skipping: {image_url}")
                                return None
                            if content_length > 5000000:  # More than 5MB
                                logger.warning(f"Image too large ({content_length} bytes), skipping: {image_url}")
                                return None
                                
                        image_data = await response.read()
                        
                        # Validate image data size as a backup check
                        if len(image_data) < 10000:  # Less than 10KB
                            logger.warning(f"Downloaded image too small ({len(image_data)} bytes), skipping")
                            return None
                            
                        # Validate filename and extension
                        filename = image_url.split("/")[-1].split("?")[0]
                        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
                        
                        if not filename or not any(filename.lower().endswith(ext) for ext in valid_extensions):
                            # Generate a more descriptive filename using article title if available
                            if article_title:
                                # Create a slug from the title for the filename
                                slug = re.sub(r'[^a-z0-9]+', '-', article_title.lower()).strip('-')
                                filename = f"{slug[:40]}_{int(time.time())}.jpg"
                            else:
                                filename = f"image_{int(time.time())}.jpg"
                            content_type = 'image/jpeg'  # Force content type to jpeg if unknown or invalid
                        
                        headers = {
                            'Content-Disposition': f'attachment; filename="{filename}"',
                            'Content-Type': content_type
                        }

                        logger.info(f"Uploading image with filename: {filename} and content type: {content_type}")

                        auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
                        media_url = f"{self.wp_url}/wp-json/wp/v2/media"

                        async with aiohttp.ClientSession() as upload_session:
                            async with upload_session.post(
                                media_url,
                                data=image_data,
                                headers=headers,
                                auth=auth,
                                timeout=30
                            ) as upload_response:
                                if upload_response.status in (200, 201):
                                    media = await upload_response.json()
                                    media_id = media.get('id')
                                    logger.info(f"Uploaded image {filename} with media ID {media_id}")
                                    return media_id
                                else:
                                    error_text = await upload_response.text()
                                    logger.error(f"Failed to upload image {filename}, status {upload_response.status}, error: {error_text}")
                                    return None
                except asyncio.TimeoutError:
                    logger.error(f"Timeout downloading image from {image_url}")
                    return None
                    
        except Exception as e:
            logger.error(f"Exception in upload_image_from_url: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


    async def publish_article(self, article: Article) -> bool:
        """Publish article to WordPress with categories and tags"""
        try:
            if not article:
                logger.error("No article provided")
                return False

            logger.info(f"Starting to publish article: {article.title}")

            # Handle categories using CategoryDetector if missing
            if not article.categories:
                detected_categories = self.category_detector.detect_categories(
                    article.title,
                    article.content,
                    getattr(article, 'keywords', None)
                )
                article.categories = detected_categories
                logger.info(f"Detected categories: {article.categories}")
            
            # Store original category IDs for later use
            original_category_ids = article.categories.copy() if isinstance(article.categories, list) else []
            
            # For logging purposes only, convert category IDs to names
            id_to_name = {v: k for k, v in self.category_detector.category_mappings.items()}
            category_names = [id_to_name.get(cat_id, 'global news') for cat_id in article.categories]
            logger.info(f"Categories correspond to names: {category_names}")
            
            # Keep the original category IDs
            article.category_ids = original_category_ids

            # Handle tags using CategoryDetector if missing
            if not article.tags:
                detected_tags = self.category_detector.extract_tags_from_title(article.title)
                article.tags = detected_tags
                logger.info(f"Detected tags: {article.tags}")

            # Handle images first
            image_ids = []
            max_images = min(Config.MAX_IMAGES_PER_ARTICLE, 2)  # Limit to 2 images max
            
            # Check if article has images
            has_images = (hasattr(article, 'images') and 
                         isinstance(article.images, list) and 
                         len(article.images) > 0)
            
            if not has_images:
                logger.info("Article has no images, skipping image processing")
            else:
                logger.info(f"Article has {len(article.images)} images, processing up to {max_images}")
                
                # Filter out None or empty URLs
                valid_images = []
                for img in article.images:
                    # Check if image is a dictionary
                    if not isinstance(img, dict):
                        continue
                        
                    # Get the URL safely
                    url = img.get('url')
                    
                    # Check if URL is a valid string and starts with http:// or https://
                    if url and isinstance(url, str) and (url.startswith('http://') or url.startswith('https://')):
                        valid_images.append(img)
                
                if not valid_images:
                    logger.warning("No valid image URLs found in article")
                else:
                    # Process only up to max_images
                    for i, image in enumerate(valid_images[:max_images]):
                        try:
                            image_url = image.get('url', '')
                            if not image_url:
                                continue
                                
                            logger.info(f"Processing image {i+1}/{max_images}: {image_url}")
                            
                            # Pass article title for better filename generation
                            media_id = await self.upload_image_from_url(image_url, article.title)
                            
                            if media_id:
                                image_ids.append(media_id)
                                image['id'] = media_id

                                # Set a more descriptive alt text based on article title
                                alt_text = f"{article.title} - {i+1}"
                                await self._update_media_alt_text(media_id, alt_text)
                                logger.info(f"Updated alt text for media ID: {media_id}")
                            else:
                                logger.warning(f"Failed to upload image: {image_url}")
                        except Exception as e:
                            logger.error(f"Failed to process image {image.get('url', 'unknown')}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())

            # Format content using ContentFormatter
            formatted_content = self.content_formatter.format_article(article.content)

            # Convert markdown to WordPress format
            formatted_content = self._convert_markdown_to_gutenberg(formatted_content)
            
            # Store the original image IDs before we modify them
            original_image_ids = image_ids.copy() if image_ids else []
            
            # If we have images, strategically place them in the content
            if image_ids:
                # Split content into paragraphs to insert images at appropriate positions
                paragraphs = formatted_content.split('<!-- /wp:paragraph -->')
                
                # Set the first image as the featured image but DON'T include it in the content
                # to avoid duplication
                if len(image_ids) > 0:
                    media_id = image_ids[0]
                    # Set as featured image
                    article.featured_image_id = media_id
                    logger.info(f"Using first image (ID: {media_id}) as featured image only - not adding to content")
                    
                    # Remove this image from the list so we don't use it again
                    image_ids.pop(0)

                
                # If we have images and enough paragraphs, place first content image in the middle
                # (remember the actual first image is now used as featured image only)
                if len(paragraphs) >= 5 and len(image_ids) > 0:
                    middle_index = len(paragraphs) // 2
                    media_id = image_ids[0]
                    image_url = await self._get_media_url(media_id)
                    
                    if image_url:
                        logger.info(f"Placing first content image in the middle, media ID: {media_id}")
                        image_block = f'\n<!-- wp:image {{"id":{media_id},"sizeSlug":"large","linkDestination":"none"}} -->\n'
                        image_block += f'<figure class="wp-block-image size-large"><img src="{image_url}" alt="{article.title} supporting image" class="wp-image-{media_id}"/></figure>\n'
                        image_block += '<!-- /wp:image -->\n'
                        
                        paragraphs[middle_index] += image_block
                        
                        # Remove this image from the list
                        image_ids.pop(0)
                
                # Rejoin the paragraphs
                formatted_content = '<!-- /wp:paragraph -->'.join(paragraphs)
                
                # If we still have any remaining images, append them at the end
                # but space them out throughout the content
                if image_ids and len(image_ids) > 0:
                    # Re-split the content to distribute remaining images
                    paragraphs = formatted_content.split('<!-- /wp:paragraph -->')
                    
                    # Calculate spacing between images
                    if len(paragraphs) > 3 and len(image_ids) > 0:
                        spacing = max(3, len(paragraphs) // (len(image_ids) + 1))
                        
                        for i, media_id in enumerate(image_ids):
                            image_url = await self._get_media_url(media_id)
                            if image_url:
                                # Calculate position to insert image
                                position = min((i + 1) * spacing, len(paragraphs) - 1)
                                
                                logger.info(f"Placing image {i+1} at position {position}, media ID: {media_id}")
                                image_block = f'\n<!-- wp:image {{"id":{media_id},"sizeSlug":"large","linkDestination":"none"}} -->\n'
                                image_block += f'<figure class="wp-block-image size-large"><img src="{image_url}" alt="{article.title} additional image" class="wp-image-{media_id}"/></figure>\n'
                                image_block += '<!-- /wp:image -->\n'
                                
                                # Add image block after the paragraph at this position
                                paragraphs[position] += image_block
                        
                        # Rejoin the paragraphs with distributed images
                        formatted_content = '<!-- /wp:paragraph -->'.join(paragraphs)
                    else:
                        # If not enough paragraphs, just append images at the end
                        for media_id in image_ids:
                            image_url = await self._get_media_url(media_id)
                            if image_url:
                                logger.info(f"Appending image at end of content: {media_id}")
                                formatted_content += f'\n<!-- wp:image {{"id":{media_id},"sizeSlug":"large","linkDestination":"none"}} -->\n'
                                formatted_content += f'<figure class="wp-block-image size-large"><img src="{image_url}" alt="{article.title} additional image" class="wp-image-{media_id}"/></figure>\n'
                                formatted_content += '<!-- /wp:image -->\n'
            
            # Ensure the featured image ID is set
            if not hasattr(article, 'featured_image_id') or not article.featured_image_id:
                if original_image_ids and len(original_image_ids) > 0:
                    article.featured_image_id = original_image_ids[0]
                    logger.info(f"Setting featured_image_id to first image: {article.featured_image_id}")


            # We'll set the featured image ID when processing images
            # This is now handled in the image processing section to avoid duplication

            # Prepare post data
            # Convert tag names to tag IDs
            tag_ids = []
            if article.tags:
                for tag in article.tags:
                    tag_id = await self._get_or_create_tag(tag)
                    if tag_id:
                        tag_ids.append(tag_id)

            # Use the original category IDs that were detected
            category_ids = getattr(article, 'category_ids', [])
            
            # Ensure all category IDs are integers
            int_category_ids = [int(cat_id) for cat_id in category_ids if str(cat_id).isdigit()]
            
            # If no valid category IDs, default to Uncategorized (ID: 1)
            if not int_category_ids:
                int_category_ids = [1]
                
            logger.info(f"Using category IDs for post: {int_category_ids}")

            # Make sure featured_image_id is an integer
            featured_media_id = 0
            if hasattr(article, 'featured_image_id') and article.featured_image_id:
                try:
                    featured_media_id = int(article.featured_image_id)
                    logger.info(f"Setting featured_media to ID: {featured_media_id}")
                except (ValueError, TypeError):
                    logger.error(f"Invalid featured_image_id: {article.featured_image_id}")
                    featured_media_id = 0

            # Ensure we have a valid featured image ID
            if featured_media_id == 0 and image_ids and len(image_ids) > 0:
                featured_media_id = image_ids[0]
                logger.info(f"Using first image as featured_media: {featured_media_id}")
                
            # Create post data with proper featured image settings
            post_data = {
                'title': article.title,
                'content': formatted_content,
                'status': 'publish',
                'categories': int_category_ids,  # Use validated integer category IDs
                'tags': tag_ids,
                'featured_media': featured_media_id,
                # Add meta data for thumbnail to ensure it works with all themes
                'meta': {
                    '_thumbnail_id': str(featured_media_id)  # Some WordPress installations expect a string
                }
            }



            # Add excerpt/meta description if present
            if hasattr(article, 'meta_description') and article.meta_description:
                post_data['excerpt'] = {'raw': article.meta_description}

            # Create WordPress post
            logger.info("Creating WordPress post with data:")
            logger.info(f"Title: {post_data['title']}")
            logger.info(f"Categories: {post_data['categories']}")
            logger.info(f"Featured Media ID: {post_data['featured_media']}")
            logger.info(f"Meta Thumbnail ID: {post_data['meta'].get('_thumbnail_id')}")
            
            auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.wp_url}/wp-json/wp/v2/posts", json=post_data, auth=auth) as response:
                    if response.status in (200, 201):
                        post = await response.json()
                        post_id = post.get('id')
                        if post_id:
                            # Log the response to verify featured image was set
                            logger.info(f"Post created with ID: {post_id}")
                            logger.info(f"Featured media in response: {post.get('featured_media')}")
                            logger.info(f"Post link: {post.get('link')}")
                            logger.info(f"Post created with ID: {post_id}")

                            # Add categories and tags
                            logger.info(f"Adding categories and tags to post {post_id}")
                            await self._add_categories_and_tags(post_id, int_category_ids, article.tags)

                            # Explicitly update the post to set the featured image to ensure WordPress associates it properly
                            # Use either the featured_image_id from the article or the first image ID if available
                            featured_media_id = 0
                            if hasattr(article, 'featured_image_id') and article.featured_image_id:
                                try:
                                    featured_media_id = int(article.featured_image_id)
                                except (ValueError, TypeError):
                                    logger.error(f"Invalid featured_image_id: {article.featured_image_id}")
                                    featured_media_id = 0
                            
                            # If we don't have a featured image ID yet, use the first image ID if available
                            if featured_media_id == 0 and image_ids and len(image_ids) > 0:
                                featured_media_id = image_ids[0]
                                logger.info(f"Using first image as featured_media: {featured_media_id}")
                            
                            if featured_media_id > 0:
                                logger.info(f"Setting featured image for post {post_id} with media ID {featured_media_id}")
                                
                                # Try multiple approaches to ensure the featured image is set correctly
                                
                                # Approach 1: Update post with featured_media field
                                update_data = {'featured_media': featured_media_id}
                                async with session.post(f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}", json=update_data, auth=auth) as update_response:
                                    if update_response.status in (200, 201):
                                        logger.info(f"Approach 1: Successfully updated featured image for post {post_id}")
                                    else:
                                        error_text = await update_response.text()
                                        logger.error(f"Approach 1: Failed to update featured image. Status: {update_response.status}, Error: {error_text}")
                                
                                # Approach 2: Set _thumbnail_id meta field
                                thumbnail_data = {
                                    'meta': {
                                        '_thumbnail_id': str(featured_media_id)  # Use string format for better compatibility
                                    }
                                }
                                async with session.post(f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}", json=thumbnail_data, auth=auth) as thumbnail_response:
                                    if thumbnail_response.status in (200, 201):
                                        logger.info(f"Approach 2: Successfully set _thumbnail_id meta for post {post_id}")
                                    else:
                                        error_text = await thumbnail_response.text()
                                        logger.error(f"Approach 2: Failed to set _thumbnail_id meta. Status: {thumbnail_response.status}, Error: {error_text}")
                                
                                # Approach 3: Try using a direct WordPress REST API endpoint for post meta
                                try:
                                    meta_endpoint = f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}/meta"
                                    meta_payload = {
                                        'key': '_thumbnail_id',
                                        'value': str(featured_media_id)
                                    }
                                    async with session.post(meta_endpoint, json=meta_payload, auth=auth) as meta_response:
                                        if meta_response.status in (200, 201):
                                            logger.info(f"Approach 3: Successfully set post meta directly")
                                        else:
                                            error_text = await meta_response.text()
                                            logger.error(f"Approach 3: Failed to set post meta directly. Status: {meta_response.status}, Error: {error_text}")
                                except Exception as e:
                                    logger.error(f"Approach 3: Exception setting post meta: {e}")
                                
                                # Verify that the featured image is properly set
                                if await self._verify_featured_image(post_id, featured_media_id):
                                    logger.info(f"Verified featured image is correctly set for post {post_id}")
                                else:
                                    logger.warning(f"Featured image verification failed. Manual check recommended.")
                                    
                                    # Log instructions for manual setting
                                    post_url = post.get('link', f"{self.wp_url}/?p={post_id}")
                                    logger.warning(f"To manually set the featured image, go to WordPress admin: {self.wp_url}/wp-admin/post.php?post={post_id}&action=edit")
                            else:
                                logger.warning(f"No valid featured image ID available for post {post_id}")

                            return True
                        else:
                            logger.error("Post creation response missing ID")
                            return False
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to create post. Status: {response.status}, Error: {error_text}")
                        return False



        except Exception as e:
            logger.error(f"Error publishing article: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


    async def _add_categories_and_tags(self, post_id: int, categories: List[int], tags: List[str]) -> bool:
        """Add categories and tags to a WordPress post"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot add categories and tags: session not initialized")
            return False
        post_endpoint = f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}"
        auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
        payload = {}
        if categories:
            # Log categories type and content for debugging
            logger.info(f"_add_categories_and_tags called with categories: {categories} (types: {[type(c) for c in categories]})")
            # Ensure all categories are integers
            int_categories = [int(cat_id) for cat_id in categories if str(cat_id).isdigit()]
            if int_categories:
                payload['categories'] = int_categories
            else:
                logger.warning(f"No valid integer category IDs found in: {categories}")
                payload['categories'] = [1]  # Default to Uncategorized
        if tags:
            # Convert tag names to tag IDs asynchronously
            tag_ids = []
            for tag in tags:
                tag_id = await self._get_or_create_tag(tag)
                if tag_id:
                    tag_ids.append(tag_id)
            payload['tags'] = tag_ids
        if not payload:
            logger.info("No categories or tags to add")
            return True
        try:
            async with self.session.post(post_endpoint, json=payload, auth=auth) as response:
                if response.status in (200, 201):
                    logger.info(f"Successfully added categories and tags to post {post_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to add categories and tags to post {post_id}, status {response.status}, error: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Exception adding categories and tags to post {post_id}: {e}")
            return False



    async def _get_or_create_tag(self, tag_name: str) -> Optional[int]:
        """Get existing tag ID by name or create a new tag in WordPress and return its ID"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot get or create tag: session not initialized")
            return None
        tags_endpoint = f"{self.wp_url}/wp-json/wp/v2/tags"
        auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
        try:
            # Search for existing tag by name
            params = {'search': tag_name}
            async with self.session.get(tags_endpoint, params=params, auth=auth) as response:
                if response.status == 200:
                    tags = await response.json()
                    for tag in tags:
                        if tag.get('name').lower() == tag_name.lower():
                            return tag.get('id')
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to search tags for '{tag_name}', status {response.status}, error: {error_text}")
                    return None

            # Tag not found, create new tag
            payload = {'name': tag_name}
            async with self.session.post(tags_endpoint, json=payload, auth=auth) as response:
                if response.status in (200, 201):
                    tag = await response.json()
                    tag_id = tag.get('id')
                    logger.info(f"Created new tag '{tag_name}' with ID {tag_id}")
                    return tag_id
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create tag '{tag_name}', status {response.status}, error: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Exception in _get_or_create_tag for '{tag_name}': {e}")
            return None

    def _convert_markdown_to_gutenberg(self, content: str) -> str:
        """Convert markdown headings and formatting to WordPress Gutenberg blocks"""
        try:
            # Split content into lines for processing
            lines = content.split('\n')
            gutenberg_content = []
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Check if this is a markdown heading
                heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
                if heading_match:
                    # This is a heading - convert to WordPress heading block
                    hashes = heading_match.group(1)
                    heading_text = heading_match.group(2).strip()
                    heading_level = len(hashes)
                    
                    # Create WordPress heading block
                    gutenberg_content.append(f'<!-- wp:heading {{"level":{heading_level}}} -->')
                    gutenberg_content.append(f'<h{heading_level}>{heading_text}</h{heading_level}>')
                    gutenberg_content.append(f'<!-- /wp:heading -->')
                    gutenberg_content.append('')  # Add empty line for spacing
                
                # Check if this is a list item
                elif line.startswith('- ') or line.startswith('* '):
                    # This is a list item - collect all list items
                    list_items = []
                    list_type = 'ul'  # unordered list
                    while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                        item_text = lines[i].strip()[2:].strip()
                        list_items.append(f'<li>{item_text}</li>')
                        i += 1
                    
                    # Create WordPress list block
                    gutenberg_content.append(f'<!-- wp:list -->')
                    gutenberg_content.append(f'<{list_type}>')
                    gutenberg_content.extend(list_items)
                    gutenberg_content.append(f'</{list_type}>')
                    gutenberg_content.append(f'<!-- /wp:list -->')
                    gutenberg_content.append('')  # Add empty line for spacing
                    
                    # Continue without incrementing i since we already advanced it in the loop
                    continue
                
                # Check if this is a paragraph
                elif line and not line.startswith('<!--'):
                    # This is a paragraph - convert to WordPress paragraph block
                    gutenberg_content.append(f'<!-- wp:paragraph -->')
                    gutenberg_content.append(f'<p>{line}</p>')
                    gutenberg_content.append(f'<!-- /wp:paragraph -->')
                    gutenberg_content.append('')  # Add empty line for spacing
                
                # If it's already a WordPress block or empty line, keep as is
                else:
                    gutenberg_content.append(line)
                
                i += 1
            
            # Join all lines back together
            return '\n'.join(gutenberg_content)
            
        except Exception as e:
            logger.error(f"Error converting markdown to Gutenberg: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return original content if conversion fails
            return content

    async def _update_media_alt_text(self, media_id: int, alt_text: str) -> bool:
        """Update the alt text of a media item in WordPress"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot update media alt text: session not initialized")
            return False
        media_endpoint = f"{self.wp_url}/wp-json/wp/v2/media/{media_id}"
        payload = {
            "alt_text": alt_text
        }
        try:
            auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
            async with self.session.post(media_endpoint, json=payload, auth=auth) as response:
                if response.status in (200, 201):
                    logger.info(f"Successfully updated alt text for media ID {media_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to update alt text for media ID {media_id}, status {response.status}, error: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Exception updating alt text for media ID {media_id}: {e}")
            return False

    async def _get_media_url(self, media_id: int) -> Optional[str]:
        """Retrieve the source URL of a media item from WordPress"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot get media URL: session not initialized")
            return None
        media_endpoint = f"{self.wp_url}/wp-json/wp/v2/media/{media_id}"
        try:
            auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
            async with self.session.get(media_endpoint, auth=auth) as response:
                if response.status == 200:
                    media = await response.json()
                    source_url = media.get('source_url')
                    if source_url:
                        return source_url
                    else:
                        logger.error(f"Media ID {media_id} has no source_url field")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get media URL for media ID {media_id}, status {response.status}, error: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Exception getting media URL for media ID {media_id}: {e}")
            return None
            
    def _handle_api_error(self, service_name: str, error: Exception) -> None:
        """Handle API errors consistently"""
        error_message = str(error)
        logger.error(f"{service_name} API error: {error_message}")
        
        # Check for common API errors
        if "NoneType" in error_message and "has no attribute" in error_message:
            logger.warning(f"Possible missing API key or configuration for {service_name}")
        elif "rate limit" in error_message.lower() or "429" in error_message:
            logger.warning(f"{service_name} rate limit exceeded, consider implementing backoff")
        elif "unauthorized" in error_message.lower() or "401" in error_message:
            logger.warning(f"{service_name} authentication failed, check API key")
        
        # Log the full traceback for debugging
        import traceback
        logger.debug(f"{service_name} API error traceback: {traceback.format_exc()}")
        
    async def _set_featured_image_direct(self, post_id: int, media_id: int) -> bool:
        """Set featured image using direct WordPress API calls"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot set featured image: session not initialized")
            return False
            
        try:
            auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
            
            # Try multiple approaches to set the featured image
            
            # Approach 1: Standard REST API
            update_data = {'featured_media': media_id}
            async with self.session.post(f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}", json=update_data, auth=auth) as response:
                if response.status in (200, 201):
                    logger.info(f"Direct approach 1: Successfully set featured image for post {post_id}")
                    success1 = True
                else:
                    error_text = await response.text()
                    logger.error(f"Direct approach 1: Failed to set featured image. Status: {response.status}, Error: {error_text}")
                    success1 = False
            
            # Approach 2: Set post meta directly
            meta_data = {'meta': {'_thumbnail_id': str(media_id)}}
            async with self.session.post(f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}", json=meta_data, auth=auth) as response:
                if response.status in (200, 201):
                    logger.info(f"Direct approach 2: Successfully set _thumbnail_id meta for post {post_id}")
                    success2 = True
                else:
                    error_text = await response.text()
                    logger.error(f"Direct approach 2: Failed to set _thumbnail_id meta. Status: {response.status}, Error: {error_text}")
                    success2 = False
            
            # Approach 3: Try using a custom endpoint if available
            try:
                custom_endpoint = f"{self.wp_url}/wp-json/wp/v2/set-featured-image/{post_id}/{media_id}"
                async with self.session.post(custom_endpoint, auth=auth) as response:
                    if response.status in (200, 201):
                        logger.info(f"Direct approach 3: Successfully used custom endpoint")
                        success3 = True
                    else:
                        logger.warning(f"Direct approach 3: Custom endpoint not available or failed. Status: {response.status}")
                        success3 = False
            except Exception:
                success3 = False
                
            return success1 or success2 or success3
            
        except Exception as e:
            logger.error(f"Error in direct featured image setting: {e}")
            return False
    
    async def _verify_featured_image(self, post_id: int, media_id: int) -> bool:
        """Verify that the featured image is properly set for a post"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot verify featured image: session not initialized")
            return False
            
        try:
            auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
            async with self.session.get(f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}", auth=auth) as response:
                if response.status == 200:
                    post_data = await response.json()
                    
                    # Check if featured_media is set correctly
                    if post_data.get('featured_media') == media_id:
                        logger.info(f"Featured image (ID: {media_id}) is correctly set for post {post_id}")
                        return True
                    else:
                        logger.warning(f"Featured image mismatch for post {post_id}. Expected: {media_id}, Got: {post_data.get('featured_media')}")
                        
                        # If verification fails, try setting it directly
                        logger.info(f"Attempting to set featured image directly...")
                        if await self._set_featured_image_direct(post_id, media_id):
                            logger.info(f"Successfully set featured image directly")
                            return True
                        return False
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to verify featured image. Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error verifying featured image: {e}")
            return False
