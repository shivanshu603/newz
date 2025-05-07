import asyncio
import aiohttp
from bs4 import BeautifulSoup, Tag
from PIL import Image
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Union
from utils import logger
import random
import json
from urllib.parse import urljoin, quote_plus
import hashlib
import os
from fake_useragent import UserAgent
import re
from image_tools import verify_image, optimize_image
from collections import Counter

# Optional NLP support
try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = spacy.load('en_core_web_sm')
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

# Optional CLIP support
CLIP_AVAILABLE = False
try:
    import torch
    from PIL import Image
    import clip
    CLIP_AVAILABLE = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
except (ImportError, ModuleNotFoundError):
    logger.warning("CLIP not available. Some image processing features will be limited.")

class APIConfig:
    PIXABAY_API_KEY = os.getenv('PIXABAY_API_KEY', '')
    PEXELS_API_KEY = os.getenv('PEXELS_API_KEY', '')
    UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY', '')

class ImageScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = None
        self.headers = {'User-Agent': self.ua.random}

    async def __aenter__(self):
        try:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            logger.info("ImageScraper session initialized")
            return self
        except Exception as e:
            logger.error(f"Failed to initialize ImageScraper session: {e}")
            # Create a session anyway to avoid NoneType errors
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_images(self, topic: str, num_images: int = 3) -> Optional[Dict[str, Union[str, List[str]]]]:
        # This method is required by the system and should call get_images internally
        return await self.get_images(topic, num_images)

    async def get_images(self, topic: str, num_images: int = 3) -> Optional[Dict[str, Union[str, List[str]]]]:
        try:
            # Validate input
            if not topic or not isinstance(topic, str):
                logger.error(f"Invalid topic provided: {topic}")
                return None
                
            # Limit number of images to a reasonable value
            num_images = min(max(1, num_images), 5)  # Between 1 and 5
            
            logger.info(f"Fetching {num_images} images for topic: {topic}")
            
            # Make sure session is initialized
            if not self.session:
                logger.warning("Session not initialized, creating new session")
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            
            # Try API-based image fetching first
            logger.info("Attempting to fetch images from APIs")
            image_results = await self._get_images_from_apis(topic, num_images)
            if image_results and image_results.get('images'):
                logger.info(f"Successfully fetched {len(image_results['images'])} images from APIs")
                return image_results
                
            # Fallback to scraping if APIs fail
            logger.info("API fetching failed, falling back to web scraping")
            image_results = await self._search_images(topic, num_images)
            if image_results and image_results.get('images'):
                logger.info(f"Successfully scraped {len(image_results['images'])} images")
                return image_results
                
            # If we get here, all methods failed
            logger.warning(f"Failed to find any images for topic: {topic}")
            
            # Return empty result rather than None to avoid errors
            return {'topic': topic, 'images': []}
            
        except aiohttp.ClientError as e:
            logger.error(f"Network error while fetching images: {e}")
            # Return empty result rather than None
            return {'topic': topic, 'images': []}
            
        except Exception as e:
            logger.error(f"Unexpected error in image scraping: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return empty result rather than None
            return {'topic': topic, 'images': []}

    async def _get_images_from_apis(self, topic: str, num_images: int) -> Optional[Dict[str, Union[str, List[str]]]]:
        # Try Unsplash API
        if APIConfig.UNSPLASH_ACCESS_KEY:
            try:
                if not self.session:
                    logger.error("Session not initialized for Unsplash API request")
                    return None
                    
                async with self.session.get(
                    f"https://api.unsplash.com/search/photos?query={quote_plus(topic)}&per_page={num_images}",
                    headers={"Authorization": f"Client-ID {APIConfig.UNSPLASH_ACCESS_KEY}"}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and isinstance(data, dict) and 'results' in data and data['results'] is not None:
                            images = []
                            for photo in data['results']:
                                if not photo or not isinstance(photo, dict):
                                    continue
                                    
                                # Safely get the urls dictionary
                                urls = None
                                if 'urls' in photo and photo['urls'] is not None and isinstance(photo['urls'], dict):
                                    urls = photo['urls']
                                
                                if urls and isinstance(urls, dict) and 'regular' in urls:
                                    regular_url = urls['regular']
                                    if regular_url and isinstance(regular_url, str):
                                        images.append(regular_url)
                                else:
                                    logger.debug(f"Photo urls missing or invalid: {photo}")
                                    
                            if images:
                                return {'topic': topic, 'images': images[:num_images]}
                        else:
                            logger.error("Unsplash API returned no results or invalid data")
                    else:
                        logger.error(f"Unsplash API returned status {response.status}")
            except Exception as e:
                logger.error(f"Unsplash API error: {e}")
                import traceback
                logger.debug(f"Unsplash API error traceback: {traceback.format_exc()}")

        # Try Pixabay API
        if APIConfig.PIXABAY_API_KEY:
            try:
                if not self.session:
                    logger.error("Session not initialized for Pixabay API request")
                    return None
                    
                async with self.session.get(
                    f"https://pixabay.com/api/?key={APIConfig.PIXABAY_API_KEY}&q={quote_plus(topic)}&image_type=photo&per_page={num_images}"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and isinstance(data, dict) and 'hits' in data and data['hits'] is not None:
                            images = []
                            for hit in data['hits']:
                                if not hit or not isinstance(hit, dict):
                                    continue
                                    
                                # Safely get the webformatURL
                                if 'webformatURL' in hit and hit['webformatURL'] is not None and isinstance(hit['webformatURL'], str):
                                    url = hit['webformatURL']
                                    images.append(url)
                                else:
                                    logger.debug(f"Hit webformatURL missing or invalid: {hit}")
                                    
                            if images:
                                return {'topic': topic, 'images': images[:num_images]}
                        else:
                            logger.error("Pixabay API returned no hits or invalid data")
                    else:
                        logger.error(f"Pixabay API returned status {response.status}")
            except Exception as e:
                logger.error(f"Pixabay API error: {e}")
                import traceback
                logger.debug(f"Pixabay API error traceback: {traceback.format_exc()}")

        # Try Pexels API
        if APIConfig.PEXELS_API_KEY:
            try:
                if not self.session:
                    logger.error("Session not initialized for Pexels API request")
                    return None
                    
                async with self.session.get(
                    f"https://api.pexels.com/v1/search?query={quote_plus(topic)}&per_page={num_images}",
                    headers={"Authorization": APIConfig.PEXELS_API_KEY}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and isinstance(data, dict) and 'photos' in data and data['photos'] is not None:
                            images = []
                            for photo in data['photos']:
                                if not photo or not isinstance(photo, dict):
                                    continue
                                    
                                # Safely get the src dictionary
                                src = None
                                if 'src' in photo and photo['src'] is not None and isinstance(photo['src'], dict):
                                    src = photo['src']
                                
                                if src and isinstance(src, dict) and 'medium' in src:
                                    medium_url = src['medium']
                                    if medium_url and isinstance(medium_url, str):
                                        images.append(medium_url)
                                else:
                                    logger.debug(f"Photo src missing or invalid: {photo}")
                                    
                            if images:
                                return {'topic': topic, 'images': images[:num_images]}
                        else:
                            logger.error("Pexels API returned no photos or invalid data")
                    else:
                        logger.error(f"Pexels API returned status {response.status}")
            except Exception as e:
                logger.error(f"Pexels API error: {e}")
                import traceback
                logger.debug(f"Pexels API error traceback: {traceback.format_exc()}")

        return None

    async def _search_images(self, topic: str, num_images: int) -> Optional[Dict[str, Union[str, List[str]]]]:
        search_query = self._prepare_search_query(topic)
        try:
            if not self.session:
                logger.error("Session not initialized for image search")
                return None
                
            async with self.session.get(f"https://www.google.com/search?q={search_query}&tbm=isch",
                                        headers=self.headers) as response:
                if response.status != 200:
                    logger.warning(f"Image search returned status {response.status}")
                    return None
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                images = []
                
                for img_tag in soup.find_all(['img', 'div']):
                    if not isinstance(img_tag, Tag):
                        continue
                        
                    img_url = None
                    
                    # Safely check for image attributes
                    for attr in ['src', 'data-src', 'data-original']:
                        # Skip if tag is None
                        if img_tag is None:
                            break
                            
                        # Check if the tag has a get method
                        if hasattr(img_tag, 'get') and callable(img_tag.get):
                            # Try to get the attribute
                            attr_value = img_tag.get(attr)
                            if attr_value:
                                img_url = attr_value
                                break
                    
                    # Only add valid image URLs
                    if img_url and isinstance(img_url, str) and img_url.startswith(('http://', 'https://')):
                        # Skip tiny images (data URIs, icons, etc.)
                        if len(img_url) > 10 and not img_url.startswith('data:'):
                            images.append(img_url)
                            
                    # Break if we have enough images
                    if len(images) >= num_images:
                        break

                if images:
                    return {'topic': topic, 'images': images[:num_images]}
                else:
                    logger.warning(f"No images found for topic: {topic}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error during image scraping: {e}")
            import traceback
            logger.debug(f"Image scraping error traceback: {traceback.format_exc()}")
            return None

    def _prepare_search_query(self, topic: str) -> str:
        """
        Prepare and clean the search query string for Google Images scraping.
        """
        import urllib.parse
        cleaned = topic.strip()
        # Replace multiple spaces with single space
        cleaned = ' '.join(cleaned.split())
        # URL encode the cleaned string
        encoded = urllib.parse.quote_plus(cleaned)
        return encoded
