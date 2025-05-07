from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import json
import aiohttp
import asyncio
from urllib.parse import quote
from blog_generator import BlogGenerator
from keyword_researcher import KeywordResearcher
from seo_analyzer import SEOAnalyzer
from seo.advanced_keyword_metrics import AdvancedKeywordMetrics
from wordpress_publisher import WordPressPublisher
from utils import logger
from image_scraper import ImageScraper
from image_tools import optimize_image, verify_image
from datetime import datetime
from models import Article

class SEOOptimizedPublisher:
    def __init__(self):
        self.blog_generator = BlogGenerator()
        self.keyword_researcher = KeywordResearcher()
        self.seo_analyzer = SEOAnalyzer()
        self.metrics_analyzer = AdvancedKeywordMetrics()
        self.wp_publisher = WordPressPublisher()
        self.image_scraper = ImageScraper()
        
    async def publish_optimized_article(self, topic: Dict) -> bool:
        """Generate, optimize and publish article with proper error handling"""
        try:
            # Ensure SEO analyzer is initialized
            if not await self._initialize_seo_analyzer():
                return False

            # 1. Research keywords with error handling
            try:
                keywords = await self.keyword_researcher.find_keywords(topic['name'])
                if not keywords or not keywords.get('primary_keywords'):
                    logger.error("No keywords found for topic")
                    return False
                topic['keywords'] = keywords['primary_keywords']
            except Exception as e:
                logger.error(f"Error researching keywords: {e}")
                return False

            # 2. Generate article with retries
            article = await self._generate_article(topic)
            if not article:
                return False

            # Safe article access with null checks
            content = getattr(article, 'content', '')
            title = getattr(article, 'title', '')
            
            if not content or not title:
                logger.error("Article missing required content or title")
                return False

            # 3. Analyze SEO metrics with fallback
            try:
                if not hasattr(self.seo_analyzer, 'safe_analyze_keyword'):
                    seo_analysis = {}
                else:
                    seo_analysis = await self.seo_analyzer.safe_analyze_keyword(topic['name'])
                metrics = await self.metrics_analyzer.calculate_comprehensive_metrics(topic['name'])
            except Exception as e:
                logger.error(f"Error in SEO analysis: {e}")
                seo_analysis = {}
                metrics = {}

            # 4. Optimize content based on analysis
            if article and article.content:
                article.content = await self._optimize_content(
                    article.content,
                    article
                )
                
                # Get optimal images for the content
                if article.title and article.content:
                    images = await self._get_relevant_images(article.title, article.content)
                    article.images = images

                    # Add image alt text and captions
                    article.content = self._insert_images_into_content(article.content, images)
            
            # 5. Calculate final SEO score
            final_score = await self._calculate_final_score(article, seo_analysis)
            
            # 6. Only publish if meets threshold
            if final_score >= 0.7:  # 70% score threshold
                # Add schema markup
                article.content = self._add_schema_markup(article, metrics)
                
                # Publish to WordPress
                success = await self.wp_publisher.publish_article(article)
                
                if success:
                    logger.info(f"Successfully published optimized article: {article.title}")
                    logger.info(f"SEO Score: {final_score:.2f}")
                    return True
            
            logger.warning(f"Article did not meet SEO threshold: {final_score:.2f}")
            return False
            
        except Exception as e:
            logger.error(f"Error in optimized publishing: {e}")
            return False

    async def _initialize_seo_analyzer(self) -> bool:
        """Initialize SEO analyzer with proper async handling"""
        try:
            if hasattr(self.seo_analyzer, '_verify_initialization'):
                if not self.seo_analyzer._verify_initialization():
                    if hasattr(self.seo_analyzer, 'ensure_methods'):
                        ensure_method = self.seo_analyzer.ensure_methods
                        if asyncio.iscoroutinefunction(ensure_method):
                            await ensure_method()
                        else:
                            ensure_method()
            return True
        except Exception as e:
            logger.error(f"Error initializing SEO analyzer: {e}")
            return False

    async def _generate_article(self, topic: Dict) -> Optional[Article]:
        """Generate article with retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                article = await self.blog_generator.generate_article(topic)
                if article and article.content:
                    return article
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Article generation attempt {attempt + 1} failed: {e}")
        return None

    async def _optimize_content(self, content: str, article: Optional[Article]) -> str:
        """Optimize content with proper None checks"""
        try:
            if not article:
                return content
                
            # Now safe to use article attributes
            if not hasattr(article, 'content') or not article.content:
                logger.error("Article missing content")
                return content
                
            optimized = await self._apply_optimizations(article.content)
            return optimized
                
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            return content

    async def _scrape_serp_data(self, keyword: str) -> Dict:
        """Scrape SERP data without API"""
        try:
            url = f"https://www.google.com/search?q={quote(keyword)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._parse_serp_html(html)
            return {}
        except Exception as e:
            logger.error(f"Error scraping SERP: {e}")
            return {}

    def _add_schema_markup(self, article: Article, metrics: Dict) -> str:
        """Add SEO schema markup to content"""
        try:
            schema = {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": article.title,
                "description": self._generate_meta_description(article.content),
                "keywords": ", ".join(metrics.get('keywords', [])),
                "articleBody": article.content,
                "datePublished": article.published_at.isoformat(),
                "author": {
                    "@type": "Person",
                    "name": "AI Content Generator"
                }
            }
            
            schema_script = f"""
<script type="application/ld+json">
{json.dumps(schema, indent=2)}
</script>
"""
            return article.content + schema_script
            
        except Exception as e:
            logger.error(f"Error adding schema markup: {e}")
            return article.content

    async def _calculate_final_score(self, article: Article, seo_analysis: Dict) -> float:
        """Calculate final SEO score for article"""
        try:
            # Get scores from different aspects
            content_score = self._analyze_content_quality(article.content)
            keyword_score = self._analyze_keyword_usage(article.content, seo_analysis)
            technical_score = self._analyze_technical_seo(article.content)
            
            # Weighted average
            weights = {
                'content': 0.4,
                'keywords': 0.3, 
                'technical': 0.3
            }
            
            final_score = (
                content_score * weights['content'] +
                keyword_score * weights['keywords'] +
                technical_score * weights['technical']
            )
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating SEO score: {e}")
            return 0.0

    async def _get_relevant_images(self, title: str, content: str) -> List[Dict]:
        """Get contextually relevant images for the article"""
        try:
            # Analyze content for image context
            image_contexts = self._analyze_image_needs(title, content)
            
            # Get images for each context
            all_images = []
            for context in image_contexts:
                images = await self.image_scraper.get_relevant_images(
                    topic=context['query'],
                    content=context['description'],
                    num_images=1
                )
                if images:
                    images[0]['placement'] = context['placement']
                    images[0]['alt_text'] = context['alt_text']
                    all_images.extend(images)

            return all_images

        except Exception as e:
            logger.error(f"Error getting relevant images: {e}")
            return []

    def _analyze_image_needs(self, title: str, content: str) -> List[Dict]:
        """Analyze content to determine optimal image contexts and placements"""
        try:
            contexts = []
            
            # Header image based on title
            contexts.append({
                'query': title,
                'description': 'Featured image representing main topic',
                'placement': 'header',
                'alt_text': f"Featured image for {title}"
            })

            # Analyze content sections for image opportunities
            sections = content.split('\n\n')
            for i, section in enumerate(sections):
                if len(section) > 200 and i % 3 == 0:  # Every third substantial section
                    # Extract main topic of section
                    section_topic = self._extract_section_topic(section)
                    contexts.append({
                        'query': section_topic,
                        'description': section[:200],
                        'placement': f'section_{i}',
                        'alt_text': f"Illustration of {section_topic}"
                    })

            return contexts

        except Exception as e:
            logger.error(f"Error analyzing image needs: {e}")
            return []

    def _insert_images_into_content(self, content: str, images: List[Dict]) -> str:
        """Insert images into content at optimal positions"""
        try:
            sections = content.split('\n\n')
            enhanced_sections = []

            # Add header image
            header_image = next((img for img in images if img['placement'] == 'header'), None)
            if header_image:
                enhanced_sections.append(self._format_image_html(header_image))

            # Insert section images
            for i, section in enumerate(sections):
                enhanced_sections.append(section)
                section_image = next(
                    (img for img in images if img['placement'] == f'section_{i}'), 
                    None
                )
                if section_image:
                    enhanced_sections.append(self._format_image_html(section_image))

            return '\n\n'.join(enhanced_sections)

        except Exception as e:
            logger.error(f"Error inserting images: {e}")
            return content

    def _format_image_html(self, image: Dict) -> str:
        """Format image HTML with proper attributes"""
        return f"""
<figure class="wp-block-image size-large">
    <img src="{image['url']}" alt="{image['alt_text']}" class="wp-image"/>
    <figcaption>{image.get('caption', '')}</figcaption>
</figure>
"""

    async def optimize_article(self, article: Optional[Article]) -> Optional[Article]:
        """Optimize article with proper None checks"""
        if not article:
            logger.error("No article provided for optimization")
            return None
            
        try:
            # Validate article attributes before accessing
            if not article.content or not article.title:
                logger.error("Article missing required content or title")
                return None

            # Now safe to access attributes
            optimized_content = await self._optimize_content(article.content, article)
            optimized_title = await self._optimize_title(article.title)
            
            article.content = optimized_content
            article.title = optimized_title
            
            return article
            
        except Exception as e:
            logger.error(f"Error optimizing article: {e}")
            return None

    def _validate_article(self, article: Optional[Article]) -> bool:
        """Validate article with proper None checks"""
        if not article:
            return False
            
        # Now safe to access attributes
        has_content = bool(article.content)
        has_title = bool(article.title)
        has_images = bool(article.images)
        
        return all([has_content, has_title, has_images])
