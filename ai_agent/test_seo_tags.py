import asyncio
import os
from dotenv import load_dotenv
from blog_generator import BlogGenerator
from blog_publisher import BlogPublisher
from models import Article
from utils import logger
from config import Config
from seo_analyzer import SEOAnalyzer
from keyword_researcher import KeywordResearcher

# Load environment variables
load_dotenv()

async def test_seo_and_tags():
    """Test the SEO and tag functionality"""
    try:
        logger.info("Starting SEO and tag test")
        
        # Initialize the components
        blog_generator = BlogGenerator()
        publisher = BlogPublisher(
            wp_url=Config.WORDPRESS_SITE_URL,
            wp_username=Config.WORDPRESS_USERNAME,
            wp_password=Config.WORDPRESS_PASSWORD
        )
        seo_analyzer = SEOAnalyzer()
        keyword_researcher = KeywordResearcher()
        
        # Verify WordPress connection
        logger.info("Verifying WordPress connection...")
        if not await publisher._verify_connection():
            logger.error("Could not connect to WordPress. Please check credentials.")
            return
        
        logger.info("WordPress connection verified, generating article...")
        
        # Create a test article
        article = Article(
            title="The Benefits of Artificial Intelligence in Healthcare",
            content="Artificial intelligence is transforming healthcare in numerous ways. From improved diagnostics to personalized treatment plans, AI is helping doctors provide better care to patients. Machine learning algorithms can analyze medical images with high accuracy, often detecting issues that human doctors might miss. Additionally, AI-powered systems can process vast amounts of medical literature to stay current with the latest research and treatment options.\n\nOne of the most promising applications is in predictive analytics. By analyzing patterns in patient data, AI can help identify individuals at risk for certain conditions before symptoms appear. This enables preventive interventions that can save lives and reduce healthcare costs. Hospitals are also using AI for administrative tasks, freeing up medical professionals to focus more on patient care.",
            keywords=[],  # Intentionally empty to test keyword generation
            tags=[],      # Intentionally empty to test tag generation
            images=[]
        )
        
        logger.info("Article created, analyzing with SEO analyzer...")
        
        # Analyze with SEO analyzer
        seo_analysis = await seo_analyzer.analyze_keyword(article.title)
        logger.info(f"SEO analysis complete with {len(seo_analysis.get('variations', []))} variations")
        
        # Get keyword research data
        keyword_data = await keyword_researcher.find_keywords(article.title)
        logger.info(f"Keyword research complete with {len(keyword_data.get('primary_keywords', []))} primary keywords")
        
        # Combine keywords
        primary_keywords = keyword_data.get('primary_keywords', [article.title])
        secondary_keywords = keyword_data.get('secondary_keywords', []) + seo_analysis.get('variations', [])
        
        # Remove duplicates
        all_keywords = primary_keywords + secondary_keywords
        unique_keywords = []
        for kw in all_keywords:
            if kw and isinstance(kw, str) and kw.lower() not in [k.lower() for k in unique_keywords]:
                unique_keywords.append(kw)
        
        # Set article keywords and tags
        article.keywords = unique_keywords[:15]  # Limit to 15 keywords
        article.tags = article.keywords[:10]     # Use top 10 keywords as tags
        
        # Create SEO data
        article.seo_data = {
            # Yoast SEO fields
            '_yoast_wpseo_focuskw': article.keywords[0] if article.keywords else article.title,
            '_yoast_wpseo_metadesc': article.content[:160] + "...",
            '_yoast_wpseo_title': article.title,
            
            # Open Graph fields
            '_yoast_wpseo_opengraph-title': article.title,
            '_yoast_wpseo_opengraph-description': article.content[:160] + "...",
            
            # Twitter fields
            '_yoast_wpseo_twitter-title': article.title,
            '_yoast_wpseo_twitter-description': article.content[:160] + "...",
            
            # Additional SEO fields
            'keywords': ', '.join(article.keywords[:10]),
            'focus_keyword': article.keywords[0] if article.keywords else article.title,
            'secondary_keywords': ', '.join(article.keywords[1:6] if len(article.keywords) > 1 else [])
        }
        
        # Log article details
        logger.info(f"Article enhanced with:")
        logger.info(f"- {len(article.keywords)} keywords: {article.keywords[:5]}")
        logger.info(f"- {len(article.tags)} tags: {article.tags[:5]}")
        logger.info(f"- SEO data: {', '.join(f'{k}: {v[:20]}...' if isinstance(v, str) and len(v) > 20 else f'{k}: {v}' for k, v in list(article.seo_data.items())[:5])}")
        
        # Publish the article
        logger.info("Publishing article...")
        success = await publisher.publish_article(article)
        
        if success:
            logger.info("Article published successfully with SEO data and tags!")
        else:
            logger.error("Failed to publish article")
        
    except Exception as e:
        logger.error(f"Error in SEO and tag test: {e}")

if __name__ == "__main__":
    asyncio.run(test_seo_and_tags())