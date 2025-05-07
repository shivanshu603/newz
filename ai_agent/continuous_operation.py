import asyncio
import random
import base64
import os
import io
import signal
import traceback
import logging
import sys
from typing import List, Dict, Optional, Set, Any, Union

from datetime import datetime, timezone
from utils import logger
from blog_generator_new import BlogGenerator
from trending_topic_discoverer import TrendingTopicDiscoverer
from wordpress_publisher import WordPressPublisher
from config import Config
import feedparser
import aiohttp
from blog_publisher import BlogPublisher
import re
import time
import hashlib
from models import Article
from bs4 import BeautifulSoup
from seo_analyzer import SEOAnalyzer
from keyword_researcher import KeywordResearcher
from keyword_researcher_enhanced import KeywordResearcherEnhanced
from image_scraper import ImageScraper
from enhanced_seo import EnhancedSEOAnalyzer
from utils.network_resilience import NetworkResilience
import torch
from utils.rag_helper import RAGHelper
import gc
from news_monitor import NewsMonitor

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    force=True
)

class ContinuousOperator:
    def __init__(self):
        """Initialize with strict requirements"""
        try:
            # Create logs directory if it doesn't exist
            log_dir = Config.LOG_DIR
            os.makedirs(log_dir, exist_ok=True)

            # Setup UTF-8 encoding for stdout/stderr
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

            # Setup logging
            Config.setup_logging()
            
            # Verify required packages
            self._verify_requirements()
            
            # Initialize core components first
            logger.info("Initializing BlogGenerator...")
            self.blog_generator = BlogGenerator()
            
            logger.info("Initializing KeywordResearcher...")
            self.keyword_researcher = KeywordResearcher()
            
            logger.info("Initializing SEOAnalyzer...")
            self.seo_analyzer = SEOAnalyzer()
            
            logger.info("Initializing RAGHelper...")
            self.rag_helper = RAGHelper()
            
            logger.info("Initializing ImageScraper...")
            self.image_scraper = ImageScraper()
            
            # Initialize publisher with config validation
            if not all([Config.WORDPRESS_SITE_URL, Config.WORDPRESS_USERNAME, Config.WORDPRESS_PASSWORD]):
                raise ValueError("Missing WordPress configuration")
            self.publisher = WordPressPublisher(
                wp_url=Config.WORDPRESS_SITE_URL,
                wp_username=Config.WORDPRESS_USERNAME,
                wp_password=Config.WORDPRESS_PASSWORD
            )
            
            # Initialize NewsMonitor
            logger.info("Initializing NewsMonitor...")
            self.news_monitor = NewsMonitor()
            
            # Verify all critical components are initialized and have required methods
            required_components = [
                ('blog_generator', 'generate_article'),
                ('keyword_researcher', 'find_keywords'), 
                ('seo_analyzer', 'analyze_keyword'),
                ('rag_helper', 'get_context'),
                ('image_scraper', 'fetch_images'),
                ('publisher', 'publish_article'),
                ('news_monitor', 'monitor_sources')
            ]
            
            for component, method in required_components:
                component_instance = getattr(self, component, None)
                if not component_instance:
                    raise RuntimeError(f"Component {component} not initialized")
                if not hasattr(component_instance, method):
                    raise RuntimeError(f"Component {component} missing required method {method}")
            print("All components verified successfully")
            
            # Initialize remaining components
            self.trend_discoverer = TrendingTopicDiscoverer()
            
            # Initialize queues and tracking
            self.topic_queue = asyncio.Queue()
            self.processed_topics: Set[str] = set()
            self.stats = {
                'topics_collected': 0,
                'articles_generated': 0,
                'articles_published': 0,
                'failures': 0
            }
            self.client_session = None

            logger.info("Continuous operation system initialized successfully")

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _verify_requirements(self):
        """Verify all required packages and resources are available"""
        try:
            required_packages = Config.MODEL_REQUIREMENTS.get('required_packages', [])
            if not required_packages:
                logger.warning("No required packages specified in config")
                return

            # Check if each required package is installed
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    logger.error(f"Required package not found: {package}")
                    raise

            logger.info("All required packages are available")
        except Exception as e:
            logger.error(f"Error verifying requirements: {e}")
            raise

    async def initialize(self) -> None:
        """Async initialization method with retries and component verification"""
        max_retries = 3
        retry_delay = 5
        components = {
            'blog_generator': lambda: BlogGenerator(),
            'rag_helper': lambda: RAGHelper(),
            'trend_discoverer': lambda: TrendingTopicDiscoverer(),
            'seo_analyzer': lambda: SEOAnalyzer(),
            'keyword_researcher': lambda: KeywordResearcher(),
            'image_scraper': lambda: ImageScraper(),
            'publisher': lambda: WordPressPublisher(
                wp_url=Config.WORDPRESS_SITE_URL,
                wp_username=Config.WORDPRESS_USERNAME,
                wp_password=Config.WORDPRESS_PASSWORD
            ),
            'news_monitor': lambda: NewsMonitor()
        }

        for attempt in range(max_retries):
            try:
                print(f"\nInitialization Attempt {attempt + 1}/{max_retries}")
                print("="*40)
                
                # Initialize all components with verification
                for name, init_func in components.items():
                    attr_name = name.lower()
                    print(f"Initializing {name}...")
                    try:
                        # Initialize component with memory check
                        try:
                            from utils.memory_manager import check_memory_availability
                            if not check_memory_availability():
                                logger.warning(f"Low memory detected, using minimal configuration for {name}")
                                if hasattr(init_func, 'low_memory'):
                                    component = init_func(low_memory=True)
                                else:
                                    component = init_func()
                            else:
                                component = init_func()
                        except ImportError:
                            component = init_func()
                            

                        # Store component
                        setattr(self, attr_name, component)
                        
                        # Verify component is properly initialized
                        component_instance = getattr(self, attr_name)
                        if not component_instance:
                            raise RuntimeError(f"Failed to store {name} component")
                            

                        print(f"✅ {name} initialized successfully")
                    except Exception as e:
                        print(f"❌ Failed to initialize {name}: {str(e)}")
                        if "memory" in str(e).lower() or "paging" in str(e).lower():
                            logger.warning(f"Memory-related error initializing {name}, trying fallback")
                            if hasattr(init_func, 'low_memory'):
                                component = init_func(low_memory=True)
                                setattr(self, attr_name, component)
                                print(f"✅ {name} initialized in low-memory mode")
                                continue
                        raise

                # Test critical component functionality
                print("\nTesting component functionality...")
                
                # Test blog generator with increased timeout
                print("Testing Blog Generator...")
                # Skip test_generation to allow real topics generation as per user request
                print("Skipping BlogGenerator test_generation as per configuration to allow real topic generation.")
                print("✅ Blog Generator test skipped")


                # Test RAG helper
                print("Testing RAG Helper...")
                if not self.rag_helper:
                    raise RuntimeError("RAGHelper not initialized")
                rag_test = await self.rag_helper.get_context("Latest Technology Trends")  # Using a more concrete topic
                if not isinstance(rag_test, dict):
                    raise RuntimeError("RAGHelper test failed") 
                print("✅ RAG Helper test passed")

                # Test trend discoverer with timeout
                print("Testing Trend Discoverer...")
                if not self.trend_discoverer:
                    raise RuntimeError("TrendDiscoverer not initialized")
                trends = await asyncio.wait_for(
                    self.trend_discoverer.get_trending_topics(),
                    timeout=60  # 60 second timeout
                )
                if not isinstance(trends, list):
                    raise RuntimeError("TrendDiscoverer test failed")
                print("✅ Trend Discoverer test passed")

                # Test news monitor with timeout
                print("Testing News Monitor...")
                if not self.news_monitor:
                    raise RuntimeError("NewsMonitor not initialized")
                await asyncio.wait_for(
                    self.news_monitor.initialize(),
                    timeout=60  # 60 second timeout
                )
                print("✅ News Monitor initialized successfully")

                print("\nAll components initialized and verified successfully!")
                return

            except Exception as e:
                logger.error(f"Initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise


    async def run_continuous_operation(self):
        """Main operation loop with sequential topic processing and resource management"""
        logger.info("Starting continuous operation with sequential processing...")
        logger.debug(f"Initial stats: {self.stats}")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\nStarting Operation Cycle #{iteration}")
                
                try:
                    # Start news monitoring in background
                    asyncio.create_task(self.news_monitor.monitor_sources())
                    
                    # Initialize and start trend discoverer
                    await self.trend_discoverer.start()
                    
                    # Get trending topics from multiple sources
                    logger.info("Fetching trending topics from news sources...")
                    try:
                        rss_topics = await self.news_monitor.get_rss_topics()
                        api_topics = await self.trend_discoverer.get_trending_topics()
                        
                        # Combine topics without using set to avoid unhashable errors
                        all_topics = []
                        if rss_topics:
                            all_topics.extend(rss_topics)
                        if api_topics:
                            all_topics.extend(api_topics)
                        
                        logger.info(f"Found {len(all_topics)} topics to process sequentially")
                        
                        for topic in all_topics:
                            try:
                                # Standardize topic format
                                topic_name = topic['name'] if isinstance(topic, dict) and 'name' in topic else topic
                                if not isinstance(topic_name, str) or len(topic_name) < 3:
                                    logger.warning(f"Invalid topic format: {topic}")
                                    continue
                                
                                logger.info(f"Processing topic: {topic_name}")
                                
                                # Get context for the topic
                                context = await self.rag_helper.get_context(topic_name)
                                if not context:
                                    logger.warning(f"Failed to get context for topic: {topic_name}")
                                    continue

                                # Generate article with context and proper formatting
                                article = await self.blog_generator.generate_article({
                                    'name': topic_name,
                                    'context': context
                                })

                                if not article or not isinstance(article, Article):
                                    logger.error(f"Article generation failed for: {topic_name}")
                                    continue

                                logger.info(f"Article generated successfully for: {topic_name}")
                                
                                # Clean and validate content while preserving headings
                                article = await self._validate_and_clean_content(article)
                                
                                # Ensure proper WordPress formatting with preserved headings
                                if hasattr(article, 'content'):
                                    # Create publisher instance with proper heading preservation
                                    blog_publisher = BlogPublisher(
                                        wp_url=Config.WORDPRESS_SITE_URL,
                                        wp_username=Config.WORDPRESS_USERNAME,
                                        wp_password=Config.WORDPRESS_PASSWORD
                                    )

                                    # Add categories if not present
                                    if not hasattr(article, 'categories') or not article.categories:
                                        # Determine categories based on topic and content
                                        categories = self._determine_categories(topic_name, context)
                                        article.categories = categories if categories else [1]  # Default to category ID 1 if none found

                                    # Add images if none present
                                    if not hasattr(article, 'images') or not article.images:
                                        async with ImageScraper() as scraper:
                                            images_result = await scraper.fetch_images(topic_name, num_images=1)
                                        if images_result and isinstance(images_result, dict):
                                            image_urls = images_result.get('images', [])
                                            article.images = []
                                            for url in image_urls:
                                                try:
                                                    media_id = await blog_publisher.upload_image_from_url(url)
                                                    if media_id:
                                                        article.images.append({'id': str(media_id), 'url': url})
                                                except Exception as e:
                                                    logger.error(f"Failed to upload image {url}: {e}")

                                    # Publish article with preserved headings
                                    logger.info(f"Publishing article: {topic_name}")
                                    await blog_publisher.publish_article(article)
                                    logger.info(f"Published article: {topic_name}")

                                # Update stats
                                self.stats['articles_published'] += 1

                                # Optional: small delay between articles
                                await asyncio.sleep(1)
                                
                            except Exception as e:
                                logger.error(f"Error processing topic {topic}: {e}")
                                self.stats['failures'] += 1
                                continue
                    
                    except Exception as e:
                        logger.error(f"Error fetching topics: {e}")
                        self.stats['failures'] += 1
                    
                    logger.info(f"Completed processing cycle.")
                    logger.info(f"Next cycle in 30 seconds (debug mode)")
                    await asyncio.sleep(30)  # Reduced for debugging
                    

                except Exception as e:
                    logger.error(f"Error in operation cycle: {e}")
                    await asyncio.sleep(60)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Fatal error in continuous operation: {e}")
            raise
        finally:
            await self.cleanup()

    async def _validate_and_clean_content(self, article: Article) -> Article:
        """Validate and clean article content to ensure proper structure"""
        try:
            if not article or not article.content:
                raise ValueError("Invalid article or empty content")

            content = article.content
            
            # Split content into sections while preserving markdown headings
            sections = []
            current_section = []
            lines = content.split('\n')
            
            for line in lines:
                # If this is a heading, start a new section
                if re.match(r'^#{1,6}\s', line):
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                current_section.append(line)
            
            # Add the last section
            if current_section:
                sections.append('\n'.join(current_section))
            
            # Clean each section individually while preserving headings
            cleaned_sections = []
            for section in sections:
                lines = section.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    # Preserve markdown headings
                    if re.match(r'^#{1,6}\s', line):
                        cleaned_lines.append(line)
                        cleaned_lines.append('')  # Add space after heading
                        continue
                        
                    # Clean up the line
                    cleaned = line.strip()
                    if cleaned:
                        # Remove common artifacts and filler phrases
                        cleaned = re.sub(r'(?:honestly|you know|look|i think|without a doubt),?\s*', '', cleaned, flags=re.IGNORECASE)
                        cleaned = re.sub(r'(?:to my utter dismay|with remarkable skill|when pigs fly),?\s*', '', cleaned, flags=re.IGNORECASE)
                        cleaned = re.sub(r'http:www\S*', '', cleaned)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        
                        if cleaned:
                            cleaned_lines.append(cleaned)
                
                if cleaned_lines:
                    cleaned_sections.append('\n\n'.join(cleaned_lines))
            
            # Join sections with proper spacing
            content = '\n\n'.join(cleaned_sections)
            
            # Ensure proper markdown heading hierarchy
            content = self._fix_heading_hierarchy(content)
            
            # Fix any remaining formatting issues
            content = re.sub(r'\n{3,}', '\n\n', content)  # Fix multiple newlines
            content = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\n\2', content)  # Add paragraph breaks
            
            # Update article
            article.content = content.strip()
            
            # Clean title and meta description
            if article.title:
                article.title = re.sub(r'(?i)create\s+(?:a\s+)?(?:section|article)\s+about\s*', '', article.title)
                article.title = re.sub(r'\s+', ' ', article.title).strip()
            
            if article.meta_description:
                article.meta_description = re.sub(r'(?i)create\s+(?:a\s+)?(?:section|article)\s+about\s*', '', article.meta_description)
                article.meta_description = re.sub(r'http:www\S*', '', article.meta_description)
                article.meta_description = re.sub(r'\s+', ' ', article.meta_description).strip()
            
            return article
            
        except Exception as e:
            logger.error(f"Error validating article content: {e}")
            return article

    def _fix_heading_hierarchy(self, content: str) -> str:
        """Fix heading hierarchy and clean up heading formatting while preserving markdown headings"""
        try:
            # Split content into lines
            lines = content.split('\n')
            processed_lines = []
            
            # Track if we've seen an H1 heading
            has_h1 = False
            
            for line in lines:
                # Check if this is a markdown heading
                markdown_heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
                
                if markdown_heading_match:
                    # This is a markdown heading
                    hashes = markdown_heading_match.group(1)
                    heading_text = markdown_heading_match.group(2).strip()
                    
                    # Preserve the heading level but ensure proper hierarchy
                    if len(hashes) == 1:  # This is an H1
                        if not has_h1:
                            has_h1 = True
                            processed_lines.append(f"# {heading_text}")
                        else:
                            # Convert additional H1s to H2s
                            processed_lines.append(f"## {heading_text}")
                    else:
                        # Keep other heading levels as they are
                        processed_lines.append(line)
                else:
                    # Handle other heading formats (convert to markdown)
                    wiki_heading_match = re.match(r'^=+\s*(.+?)\s*=+$', line)
                    if wiki_heading_match:  # Wiki-style H1
                        heading_text = wiki_heading_match.group(1).strip()
                        if not has_h1:
                            has_h1 = True
                            processed_lines.append(f"# {heading_text}")
                        else:
                            processed_lines.append(f"## {heading_text}")
                    elif re.match(r'^h([1-6]):\s*(.+)$', line, re.IGNORECASE):  # h1: style
                        match = re.match(r'^h([1-6]):\s*(.+)$', line, re.IGNORECASE)
                        if match:  # Add null check
                            level = int(match.group(1))
                            heading_text = match.group(2).strip()
                        
                            # Convert to markdown with proper level
                            if level == 1 and not has_h1:
                                has_h1 = True
                                processed_lines.append(f"# {heading_text}")
                            elif level == 1:
                                processed_lines.append(f"## {heading_text}")
                            else:
                                # Add appropriate number of # for the heading level
                                processed_lines.append(f"{'#' * level} {heading_text}")
                        else:
                            # If match is None, keep the line as is
                            processed_lines.append(line)
                    else:
                        # Not a heading, keep as is
                        processed_lines.append(line)
            
            # Join lines and fix spacing around headings
            content = '\n'.join(processed_lines)
            
            # Ensure proper spacing around headings
            content = re.sub(r'(?<!\n\n)(^|\n)#', r'\n\n#', content)
            content = re.sub(r'#[^#\n]+\n(?!\n)', r'\g<0>\n', content)
            
            # Fix multiple consecutive newlines
            content = re.sub(r'\n{3,}', '\n\n', content)
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error fixing heading hierarchy: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return content

    def _determine_categories(self, topic: str, context: dict) -> List[int]:
        """Determine appropriate categories based on topic and context"""
        try:
            # Define category mappings
            category_mappings = {
                'technology': 2,    # Technology category ID
                'business': 3,      # Business category ID
                'health': 4,        # Health category ID
                'science': 5,       # Science category ID
                'entertainment': 6, # Entertainment category ID
                'sports': 7,        # Sports category ID
                'politics': 8,      # Politics category ID
                'world': 9,         # World News category ID
            }
            
            # Keywords for each category
            category_keywords = {
                'technology': {'tech', 'software', 'app', 'digital', 'ai', 'cyber', 'computer', 'mobile', 'internet', 'google', 'apple', 'microsoft'},
                'business': {'business', 'economy', 'market', 'finance', 'industry', 'company', 'startup', 'stock', 'trade', 'investment'},
                'health': {'health', 'medical', 'healthcare', 'disease', 'treatment', 'medicine', 'wellness', 'hospital', 'doctor', 'patient'},
                'science': {'science', 'research', 'study', 'scientist', 'discovery', 'space', 'physics', 'biology', 'chemistry', 'scientific'},
                'entertainment': {'entertainment', 'movie', 'film', 'music', 'game', 'tv', 'show', 'celebrity', 'actor', 'actress'},
                'sports': {'sports', 'game', 'player', 'team', 'match', 'tournament', 'championship', 'athletes', 'league', 'soccer', 'football'},
                'politics': {'politics', 'government', 'election', 'political', 'president', 'minister', 'policy', 'congress', 'senate', 'law'},
                'world': {'world', 'international', 'global', 'country', 'nation', 'foreign', 'diplomatic', 'embassy', 'overseas', 'continent'}
            }
            
            # Analyze topic and context
            categories = set()
            topic_lower = topic.lower()
            
            # Check topic against category keywords
            for category, keywords in category_keywords.items():
                if any(keyword in topic_lower for keyword in keywords):
                    categories.add(category_mappings[category])
            
            # Check context content if available
            if context and isinstance(context, dict):
                context_text = ' '.join(str(v) for v in context.values()).lower()
                for category, keywords in category_keywords.items():
                    if any(keyword in context_text for keyword in keywords):
                        categories.add(category_mappings[category])
            
            # Return list of unique category IDs
            return list(categories) if categories else [1]  # Default to general category (ID: 1)
            
        except Exception as e:
            logger.error(f"Error determining categories: {e}")
            return [1]  # Return default category on error

async def main():
    """Main entry point with enhanced logging and continuous operation"""
    print("Starting continuous operation system...")
    logger.info("Starting continuous operation system...")
    
    try:
        logger.info("Creating ContinuousOperator instance...")
        operator = ContinuousOperator()
        
        logger.info("Initializing components...")
        await operator.initialize()
        
        logger.info("Starting continuous operation loop...")
        await operator.run_continuous_operation()
        
    except Exception as e:
        logger.error(f"Critical error in main operation: {e}")
        logger.error(traceback.format_exc())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        logger.info("Resources cleaned up. Exiting.")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
