import aiohttp
import asyncio
import time
import re
from typing import Optional, Dict, List
from utils import logger
from models import Article
import json
import base64
from image_tools import optimize_image

class WordPressPublisher:
    def __init__(self, wp_url: str, wp_username: str, wp_password: str):
        self.wp_url = wp_url.rstrip('/')
        self.wp_api_url = f"{self.wp_url}/wp-json/wp/v2"
        self.headers = {
            "Authorization": f"Basic {base64.b64encode(f'{wp_username}:{wp_password}'.encode()).decode()}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
    async def verify_credentials(self):
        """Verify WordPress credentials work"""
        try:
            # Check if the WordPress URL is valid
            if not self.wp_url or self.wp_url == 'http://your-wordpress-site.com':
                logger.warning("WordPress URL is not configured properly. Using offline mode.")
                return False

            # Check if credentials are provided
            if not self.headers["Authorization"] or ':' not in base64.b64decode(self.headers["Authorization"].split(' ')[1]).decode():
                logger.warning("WordPress credentials are not configured. Using offline mode.")
                return False

            # Try to connect with a timeout
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        f"{self.wp_api_url}/users/me",
                        headers=self.headers
                    ) as response:
                        if response.status == 200:
                            logger.info("WordPress connection successful")
                            return True
                        logger.error(f"WordPress authentication failed: {response.status}")
                        return False
            except aiohttp.ClientConnectorError as e:
                logger.error(f"Cannot connect to WordPress site: {e}")
                logger.info("Continuing in offline mode - content will be generated but not published")
                return False
            except asyncio.TimeoutError:
                logger.error("WordPress connection timed out")
                logger.info("Continuing in offline mode - content will be generated but not published")
                return False
        except Exception as e:
            logger.error(f"Error verifying WordPress credentials: {e}")
            logger.info("Continuing in offline mode - content will be generated but not published")
            return False

    async def _upload_images(self, images: List[Dict]) -> List[Dict]:
        """Upload images to WordPress and get media IDs"""
        try:
            if not images:
                logger.warning("No images provided for upload")
                return []
                
            uploaded_images = []
            async with aiohttp.ClientSession() as session:
                for image in images:
                    try:
                        # Check if image already has data
                        image_data = None
                        if 'data' in image and image['data']:
                            # Use the already downloaded and optimized data
                            image_data = image['data']
                            logger.info(f"Using pre-downloaded image data for {image.get('id', 'unknown')}")
                        else:
                            # Download image
                            logger.info(f"Downloading image from URL: {image['url']}")
                            try:
                                async with session.get(image['url'], timeout=30) as response:
                                    if response.status == 200:
                                        image_data = await response.read()
                                        logger.info(f"Successfully downloaded image: {len(image_data)} bytes")
                                    else:
                                        logger.warning(f"Failed to download image, status: {response.status}")
                                        continue
                            except Exception as download_error:
                                logger.error(f"Error downloading image: {download_error}")
                                continue
                        
                        if not image_data or len(image_data) < 1000:
                            logger.warning(f"Image data too small or empty: {len(image_data) if image_data else 0} bytes")
                            continue
                            
                        # Optimize image
                        optimized_data = await optimize_image(image_data)
                        if not optimized_data:
                            logger.warning("Image optimization failed")
                            # Try using original data as fallback
                            optimized_data = image_data
                        
                        # Upload to WordPress
                        media_id = await self._upload_image(
                            session,
                            optimized_data,
                            image.get('alt_text', 'Article image')
                        )
                        
                        if media_id:
                            logger.info(f"Successfully uploaded image with ID: {media_id}")
                            image['wp_media_id'] = media_id
                            uploaded_images.append(image)
                        else:
                            logger.warning("Failed to get media ID for uploaded image")
                    
                    except Exception as img_error:
                        logger.error(f"Error processing individual image: {img_error}")
                        continue

            logger.info(f"Successfully uploaded {len(uploaded_images)} out of {len(images)} images")
            return uploaded_images

        except Exception as e:
            logger.error(f"Error in image upload process: {e}")
            return []

    async def upload_image_from_url(self, url: str, alt_text: str = "Article image") -> Optional[int]:
        """Download an image from a URL and upload it to WordPress"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        if not image_data or len(image_data) < 1000:
                            logger.warning(f"Image data too small or empty: {len(image_data) if image_data else 0} bytes")
                            return None
                        optimized_data = await optimize_image(image_data)
                        if not optimized_data:
                            logger.warning("Image optimization failed, using original data")
                            optimized_data = image_data
                        media_id = await self._upload_image(session, optimized_data, alt_text)
                        return media_id
                    else:
                        logger.error(f"Failed to download image from URL: {url} with status {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error uploading image from URL: {e}")
            return None


    async def publish_article(self, article: Article) -> bool:
        """Publish article with images to WordPress"""
        try:
            logger.info(f"Starting to publish article: {article.title}")
            
            # Ensure article has images
            if not hasattr(article, 'images') or not article.images:
                logger.warning("Article has no images, setting empty list")
                article.images = []
            
            # Upload images first
            uploaded_images = await self._upload_images(article.images)
            logger.info(f"Uploaded {len(uploaded_images)} images")

            # Update content with WordPress media URLs if images were uploaded
            if uploaded_images:
                # Get the media details for each uploaded image to get the WordPress URL
                media_details = {}
                async with aiohttp.ClientSession() as session:
                    for image in uploaded_images:
                        if image.get('wp_media_id'):
                            try:
                                async with session.get(
                                    f"{self.wp_api_url}/media/{image['wp_media_id']}",
                                    headers=self.headers
                                ) as response:
                                    if response.status == 200:
                                        media_data = await response.json()
                                        # Store the WordPress URL for this media
                                        if 'source_url' in media_data:
                                            media_details[image['wp_media_id']] = {
                                                'url': media_data['source_url'],
                                                'alt': image.get('alt_text', '')
                                            }
                                            logger.info(f"Retrieved WordPress URL for media {image['wp_media_id']}: {media_data['source_url']}")
                            except Exception as e:
                                logger.error(f"Error getting media details: {e}")
                
                # Now update the content with the WordPress media
                for image in uploaded_images:
                    if image.get('wp_media_id') and image['wp_media_id'] in media_details:
                        wp_url = media_details[image['wp_media_id']]['url']
                        alt_text = media_details[image['wp_media_id']]['alt']
                        
                        # Check if the image URL is in the content
                        if image['url'] in article.content:
                            article.content = article.content.replace(
                                image['url'],
                                f'<!-- wp:image {{ "id": {image["wp_media_id"]}, "sizeSlug": "large" }} -->\n'
                                f'<figure class="wp-block-image size-large"><img src="{wp_url}" '
                                f'alt="{alt_text}" class="wp-image-{image["wp_media_id"]}"/>'
                                f'<figcaption>{image.get("caption", "")}</figcaption></figure>\n'
                                '<!-- /wp:image -->'
                            )
                            logger.info(f"Replaced image URL in content with WordPress media: {image['wp_media_id']}")
                        else:
                            # If image URL is not in content, insert the image at appropriate positions
                            if image.get('placement') == 'header' and not article.content.startswith('<!-- wp:image'):
                                # Insert header image at the beginning
                                article.content = (
                                    f'<!-- wp:image {{ "id": {image["wp_media_id"]}, "sizeSlug": "large", "linkDestination": "none", "align": "wide" }} -->\n'
                                    f'<figure class="wp-block-image alignwide size-large"><img src="{wp_url}" '
                                    f'alt="{alt_text}" class="wp-image-{image["wp_media_id"]}"/>'
                                    f'<figcaption>{image.get("caption", "")}</figcaption></figure>\n'
                                    '<!-- /wp:image -->\n\n'
                                ) + article.content
                                logger.info(f"Inserted header image at beginning of content: {image['wp_media_id']}")
                            else:
                                # Insert body images throughout the content
                                # Find a good position - after a paragraph
                                paragraphs = article.content.split('\n\n')
                                if len(paragraphs) > 3:  # If we have enough paragraphs
                                    # Insert after the 3rd paragraph or at 1/3 of the content
                                    insert_pos = min(3, len(paragraphs) // 3)
                                    
                                    image_html = (
                                        f'\n\n<!-- wp:image {{ "id": {image["wp_media_id"]}, "sizeSlug": "large" }} -->\n'
                                        f'<figure class="wp-block-image size-large"><img src="{wp_url}" '
                                        f'alt="{alt_text}" class="wp-image-{image["wp_media_id"]}"/>'
                                        f'<figcaption>{image.get("caption", "")}</figcaption></figure>\n'
                                        '<!-- /wp:image -->\n\n'
                                    )
                                    
                                    # Insert the image HTML after the selected paragraph
                                    paragraphs.insert(insert_pos, image_html)
                                    article.content = '\n\n'.join(paragraphs)
                                    logger.info(f"Inserted image within content at position {insert_pos}: {image['wp_media_id']}")
                                else:
                                    # If not enough paragraphs, append at the end
                                    article.content += (
                                        f'\n\n<!-- wp:image {{ "id": {image["wp_media_id"]}, "sizeSlug": "large" }} -->\n'
                                        f'<figure class="wp-block-image size-large"><img src="{wp_url}" '
                                        f'alt="{alt_text}" class="wp-image-{image["wp_media_id"]}"/>'
                                        f'<figcaption>{image.get("caption", "")}</figcaption></figure>\n'
                                        '<!-- /wp:image -->'
                                    )
                                    logger.info(f"Appended image at end of content: {image['wp_media_id']}")
                    else:
                        logger.warning(f"Missing WordPress URL for media ID: {image.get('wp_media_id')}")

            # Set featured image
            featured_image = None
            if uploaded_images:
                # First try to find an image marked as header
                featured_image = next(
                    (img for img in uploaded_images if img.get('placement') == 'header'),
                    None
                )
                
                # If no header image, use the first image
                if not featured_image and uploaded_images:
                    featured_image = uploaded_images[0]
                    logger.info("Using first image as featured image")

            # Process SEO data if available
            seo_data = {}
            if hasattr(article, 'seo_data') and article.seo_data:
                logger.info(f"Processing SEO data for article")

                # Extract SEO metadata
                meta_title = article.title
                if len(meta_title) > 60:
                    meta_title = meta_title[:57] + "..."

                meta_description = ""
                if 'variations' in article.seo_data and article.seo_data['variations']:
                    # Use the first paragraph of content as meta description
                    paragraphs = [p for p in article.content.split('\n\n') if p and not p.startswith('#')]
                    if paragraphs:
                        meta_description = paragraphs[0][:160]

                    # If no paragraphs found, use the first variation
                    if not meta_description and article.seo_data['variations']:
                        meta_description = f"Learn about {article.seo_data['variations'][0]}"

                # Extract keywords from SEO data
                keywords = []
                if 'variations' in article.seo_data and article.seo_data['variations']:
                    keywords.extend(article.seo_data['variations'])

                # Add SEO data to post metadata
                seo_data = {
                    'meta': {
                        '_yoast_wpseo_title': meta_title,
                        '_yoast_wpseo_metadesc': meta_description,
                        '_yoast_wpseo_focuskw': ', '.join(keywords[:5])
                    }
                }

                # Generate tags from SEO data if not already provided
                if not hasattr(article, 'tags') or not article.tags:
                    article.tags = self._generate_tags_from_seo(article.seo_data)
                    logger.info(f"Generated tags from SEO data: {article.tags}")
            
            # Ensure article has categories
            if not hasattr(article, 'categories') or not article.categories:
                logger.info("No categories specified, using default")
                article.categories = ['Uncategorized']
                
            # Ensure article has tags
            if not hasattr(article, 'tags') or not article.tags:
                # Extract tags from title and content
                logger.info("No tags specified, generating from title")
                article.tags = self._extract_tags_from_title(article.title)

            # Convert markdown headings to WordPress blocks
            content = self._convert_markdown_to_gutenberg(article.content)
            
            # Create post data
            post_data = {
                'title': article.title,
                'content': content,
                'status': 'publish',
                'meta': seo_data.get('meta', {})
            }
            
            # Add featured image if available
            if featured_image and featured_image.get('wp_media_id'):
                post_data['featured_media'] = featured_image['wp_media_id']
                logger.info(f"Setting featured image ID: {featured_image['wp_media_id']}")

            # Publish post
            logger.info("Creating WordPress post")
            post_id = await self._create_post(post_data)
            if not post_id:
                logger.error("Failed to create post")
                return False

            # Add categories and tags
            logger.info(f"Adding categories and tags to post {post_id}")
            async with aiohttp.ClientSession() as session:
                await self._add_categories_and_tags(session, post_id, article)

            logger.info(f"Successfully published article with ID: {post_id}")
            return True

        except Exception as e:
            logger.error(f"Error publishing article: {e}")
            return False
            
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
    
    def _extract_tags_from_title(self, title: str) -> List[str]:
        """Extract potential tags from the article title"""
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'}
        
        # Clean the title
        clean_title = title.lower()
        
        # Split into words and filter out stop words
        words = [word for word in clean_title.split() if word not in stop_words and len(word) > 3]
        
        # Create potential tags (single words and pairs)
        tags = []
        
        # Add individual words as tags
        tags.extend(words[:5])  # Limit to 5 single-word tags
        
        # Add pairs of consecutive words as tags
        if len(words) >= 2:
            for i in range(len(words) - 1):
                tag_pair = f"{words[i]} {words[i+1]}"
                if len(tag_pair) <= 50:  # WordPress tag length limit
                    tags.append(tag_pair)
        
        # Clean and return tags
        return self._clean_tags(tags[:10])  # Limit to 10 tags total

    def _generate_tags_from_seo(self, seo_data: Dict) -> List[str]:
        """Generate tags from SEO data"""
        tags = []

        # Extract from variations
        if 'variations' in seo_data and seo_data['variations']:
            tags.extend(seo_data['variations'])

        # Extract from questions (use first 2 words of each question)
        if 'questions' in seo_data and seo_data['questions']:
            for question in seo_data['questions']:
                # Remove question words and get first 2-3 meaningful words
                cleaned = question.replace('What is', '').replace('How to', '').replace('Why is', '').strip()
                words = cleaned.split()
                if len(words) >= 2:
                    tags.append(' '.join(words[:3]))

        # Clean and return tags
        return self._clean_tags(tags)

    async def _create_post(self, post_data: Dict) -> int:
        """Create a new post in WordPress and return the post ID"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.wp_api_url}/posts",
                    headers=self.headers,
                    json=post_data
                ) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        post_id = result.get('id')
                        logger.info(f"Post created with ID: {post_id}")
                        return post_id
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to create post: {response.status} - {error_text}")
                        return 0
        except Exception as e:
            logger.error(f"Error creating post: {e}")
            return 0

    async def _verify_post_published(self, session: aiohttp.ClientSession, post_id: int) -> bool:
        """Verify a post was actually published"""
        try:
            async with session.get(
                f"{self.wp_api_url}/posts/{post_id}",
                headers=self.headers
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def _upload_image(self, session: aiohttp.ClientSession, image_data: bytes, alt_text: str) -> Optional[int]:
        """Upload image using Application Password"""
        try:
            if not image_data or len(image_data) < 1000:
                logger.warning(f"Image data too small: {len(image_data) if image_data else 0} bytes")
                return None
                
            # Detect MIME type from image data
            mime_type = 'image/jpeg'  # Default
            if image_data.startswith(b'\x89PNG'):
                mime_type = 'image/png'
            elif image_data.startswith(b'GIF'):
                mime_type = 'image/gif'

            # Create filename with proper extension
            extension = mime_type.split('/')[-1]
            filename = f"article-image-{int(time.time())}.{extension}"
            
            logger.info(f"Uploading image: {filename} ({len(image_data)} bytes, {mime_type})")

            headers = {
                **self.headers,
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": mime_type
            }

            # Upload to WordPress
            try:
                async with session.post(
                    f"{self.wp_api_url}/media",
                    headers=headers,
                    data=image_data,
                    timeout=60  # Longer timeout for image uploads
                ) as upload_response:
                    if upload_response.status in [200, 201]:
                        result = await upload_response.json()
                        media_id = result.get('id')
                        logger.info(f"Image uploaded successfully with ID: {media_id}")
                        
                        # Update alt text if provided
                        if alt_text and media_id:
                            await self._update_media_alt_text(session, media_id, alt_text)
                            
                        return media_id
                    else:
                        error_text = await upload_response.text()
                        logger.error(f"Failed to upload image: {upload_response.status} - {error_text}")
                        return None
            except asyncio.TimeoutError:
                logger.error("Image upload timed out")
                return None
            except Exception as upload_error:
                logger.error(f"Error during image upload request: {upload_error}")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing image upload: {e}")
            return None
            
    async def _update_media_alt_text(self, session: aiohttp.ClientSession, media_id: int, alt_text: str) -> bool:
        """Update alt text for an uploaded media item"""
        try:
            update_data = {
                'alt_text': alt_text
            }
            
            async with session.post(
                f"{self.wp_api_url}/media/{media_id}",
                headers=self.headers,
                json=update_data
            ) as response:
                if response.status == 200:
                    logger.info(f"Updated alt text for media ID: {media_id}")
                    return True
                else:
                    logger.warning(f"Failed to update alt text: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error updating media alt text: {e}")
            return False

    async def _add_categories_and_tags(self, session: aiohttp.ClientSession, post_id: int, article: Article):
        """Add categories and tags to published post"""
        try:
            # Add default category if none specified
            categories = getattr(article, 'categories', [1])  # Where 1 is default category ID

            tags = getattr(article, 'tags', [])
            
            # Update post with categories and tags
            update_data = {
                'categories': await self._ensure_categories(session, categories),
                'tags': await self._ensure_tags(session, tags)
            }
            
            async with session.post(
                f"{self.wp_api_url}/posts/{post_id}",
                headers=self.headers,
                json=update_data
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to update post categories/tags: {await response.text()}")

        except Exception as e:
            logger.error(f"Error adding categories and tags: {e}")

    async def _ensure_categories(self, session: aiohttp.ClientSession, categories: list) -> list:
        """Get or create categories and return their IDs"""
        try:
            category_ids = []
            for category in categories:
                # Try to find existing category
                async with session.get(
                    f"{self.wp_api_url}/categories",
                    headers=self.headers,
                    params={"search": category}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data:
                            category_ids.append(data[0]['id'])
                            continue

                # Create new category if not found
                async with session.post(
                    f"{self.wp_api_url}/categories",
                    headers=self.headers,
                    json={"name": category}
                ) as response:
                    if response.status == 201:
                        data = await response.json()
                        category_ids.append(data['id'])

            return category_ids
        except Exception as e:
            logger.error(f"Error getting category IDs: {e}")
            return [1]  # Return default category ID

    async def _ensure_tags(self, session: aiohttp.ClientSession, tags: list) -> list:
        """Create tags and return their IDs"""
        try:
            # Clean and validate tags before processing
            cleaned_tags = self._clean_tags(tags)
            logger.info(f"Processing {len(cleaned_tags)} tags: {cleaned_tags}")

            tag_ids = []
            for tag in cleaned_tags:
                # Skip empty or invalid tags
                if not tag or len(tag) < 2 or tag.startswith('...'):
                    logger.warning(f"Skipping invalid tag: '{tag}'")
                    continue

                # Limit tag length to 50 characters
                if len(tag) > 50:
                    tag = tag[:50]
                    logger.warning(f"Truncated long tag to: '{tag}'")

                logger.info(f"Processing tag: {tag}")

                # First try to find if tag already exists
                async with session.get(
                    f"{self.wp_api_url}/tags",
                    headers=self.headers,
                    params={"search": tag}
                ) as get_response:
                    if get_response.status == 200:
                        data = await get_response.json()
                        if data:
                            logger.info(f"Found existing tag: {tag} with ID: {data[0]['id']}")
                            tag_ids.append(data[0]['id'])
                            continue

                # If tag doesn't exist, create it
                async with session.post(
                    f"{self.wp_api_url}/tags",
                    headers=self.headers,
                    json={"name": tag}
                ) as response:
                    if response.status == 201:
                        data = await response.json()
                        logger.info(f"Created new tag: {tag} with ID: {data['id']}")
                        tag_ids.append(data['id'])
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to create tag '{tag}': {response.status} - {error_text}")

            logger.info(f"Processed {len(cleaned_tags)} tags, created/found {len(tag_ids)} tag IDs")
            return tag_ids
        except Exception as e:
            logger.error(f"Error creating tags: {e}")
            return []

    def _clean_tags(self, tags: list) -> list:
        """Clean and validate tags"""
        if not tags:
            return []

        cleaned_tags = []
        for tag in tags:
            # Skip if tag is None or empty
            if not tag:
                continue

            # Convert to string if not already
            tag = str(tag).strip()

            # Skip empty tags
            if not tag:
                continue

            # Skip tags that are too short
            if len(tag) < 3:
                continue

            # Skip tags that start with problematic patterns
            if tag.startswith('...') or tag == '':
                continue

            # Skip if tag is the full article title or too long
            if len(tag.split()) > 5 or len(tag) > 50:
                # Try to extract meaningful parts from long tags
                words = tag.split()
                if len(words) > 2:
                    # Use the first 2-3 words if they make sense
                    potential_tag = ' '.join(words[:3])
                    if len(potential_tag) <= 50 and len(potential_tag) >= 3:
                        cleaned_tags.append(potential_tag)
                continue

            # Add the cleaned tag
            cleaned_tags.append(tag)

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = [x for x in cleaned_tags if not (x in seen or seen.add(x))]

        # Limit to a reasonable number of tags
        return unique_tags[:10]
