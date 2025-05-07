import asyncio
import os
import pytest
from blog_generator_new import BlogGenerator
from utils.rag_helper import RAGHelper
from models import Article
import json
from config import Config
import logging
from utils import logger

@pytest.mark.asyncio
async def test_rag_helper_context():
    """Test RAG helper context gathering"""
    rag = RAGHelper()
    try:
        topic = "Artificial Intelligence in Healthcare"
        context = await rag.get_context(topic)
        
        assert context is not None, "Context should not be None"
        assert 'summary' in context, "Context should contain summary"
        assert 'facts' in context, "Context should contain facts"
        assert len(context['facts']) > 0, "Should have found some facts"
        
    finally:
        await rag.close()

@pytest.mark.asyncio
async def test_prompt_generation():
    """Test enhanced prompt generation"""
    generator = RAGHelper()
    try:
        topic = {
            'name': 'Machine Learning Applications',
            'context': 'Current trends and applications in ML'
        }
        
        prompt_data = await generator.get_context(topic['name'])
        
        assert prompt_data is not None, "Prompt data should not be None"
        assert 'prompt' in prompt_data, "Should contain generated prompt"
        assert 'keywords' in prompt_data, "Should contain keywords"
        assert len(prompt_data['keywords']['primary']) > 0, "Should have primary keywords"
        
    finally:
        await generator.cleanup()

@pytest.mark.asyncio
async def test_blog_generation():
    """Test full blog generation with RAG enhancement"""
    generator = BlogGenerator()
    try:
        topic = {
            'name': 'Future of Cloud Computing',
            'context': 'Emerging trends in cloud technology'
        }
        
        article = await generator.generate_article(topic)
        
        assert article is not None, "Article should not be None"
        assert isinstance(article, Article), "Should return Article instance"
        assert len(article.content.split()) >= Config.MIN_ARTICLE_LENGTH, "Article should meet minimum length"
        assert article.meta_keywords, "Should have meta keywords"
        assert article.seo_title, "Should have SEO title"
        assert len(article.tags) > 0, "Should have tags"
        
    finally:
        await generator.cleanup()

@pytest.mark.asyncio
async def test_content_quality():
    """Test quality of generated content"""
    generator = BlogGenerator()
    try:
        topic = {
            'name': 'Cybersecurity Best Practices',
            'context': 'Modern cybersecurity guidelines'
        }
        
        article = await generator.generate_article(topic)
        assert article is not None, "Article should not be None"
        
        # Check structural quality
        paragraphs = article.content.split('\n\n')
        assert len(paragraphs) >= 5, "Should have at least 5 paragraphs"
        
        # Check keyword presence
        assert any(kw.lower() in article.content.lower() 
                  for kw in article.keywords), "Keywords should appear in content"
        
        # Check metadata
        assert article.meta_description, "Should have meta description"
        assert len(article.meta_description) <= 155, "Meta description should be proper length"
        
        # Check images
        assert article.images, "Should have images"
        assert len(article.images) <= 3, "Should have reasonable number of images"
        
    finally:
        await generator.cleanup()

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in blog generation"""
    generator = BlogGenerator()
    try:
        # Test with invalid topic
        invalid_topic = {
            'name': '',  # Empty topic
            'context': 'Invalid test'
        }
        
        article = await generator.generate_article(invalid_topic)
        assert article is None, "Should handle invalid topic gracefully"
        
        # Test with very long topic
        long_topic = {
            'name': 'x' * 1000,  # Unreasonably long topic
            'context': 'Test long topic'
        }
        
        article = await generator.generate_article(long_topic)
        assert article is None, "Should handle too-long topic gracefully"
        
    finally:
        await generator.cleanup()

@pytest.mark.asyncio
async def test_memory_management():
    """Test memory management during generation"""
    from utils.memory_manager import MemoryManager
    
    manager = MemoryManager()
    generator = BlogGenerator()
    try:
        # Test memory monitoring
        stats = manager.get_memory_stats()
        assert stats, "Should get memory statistics"
        
        # Test generation with memory monitoring
        topic = {
            'name': 'Big Data Analytics',
            'context': 'Modern data analysis techniques'
        }
        
        # Monitor memory during generation
        memory_usage = None
        article = await generator.generate_article(topic)
        memory_usage = manager.monitor_memory_usage()
        
        assert article is not None, "Should generate article"
        assert memory_usage is None or isinstance(memory_usage, tuple), "Should monitor memory"
        
    finally:
        await generator.cleanup()

@pytest.mark.asyncio
async def test_network_resilience():
    """Test network resilience during research"""
    from utils.network_resilience import NetworkResilience
    
    network = NetworkResilience()
    try:
        # Test with valid URL
        result = await network.get_json('https://api.github.com/zen')
        assert result is not None, "Should handle valid request"
        
        # Test with invalid URL
        result = await network.get_json('https://invalid.example.com')
        assert result is None, "Should handle invalid request gracefully"
        
    finally:
        await network.close()

@pytest.mark.asyncio
async def test_content_structure():
    """Test content structure and formatting"""
    generator = BlogGenerator()
    try:
        # Test different topic types
        topics = [
            {
                'name': 'Docker Containerization',
                'context': 'Container technology guide'
            },
            {
                'name': 'How to Learn Python',
                'context': 'Programming tutorial'
            },
            {
                'name': 'Best Gaming Laptops 2025',
                'context': 'Product review'
            }
        ]
        
        for topic in topics:
            article = await generator.generate_article(topic)
            assert article is not None, f"Should generate article for {topic['name']}"
            
            # Check content structure
            content = article.content
            assert '# ' in content or '## ' in content, "Should have markdown headings"
            
            # Check sections
            sections = [s for s in content.split('\n') if s.startswith('#')]
            assert len(sections) >= 3, "Should have at least 3 sections"
            
            # Check formatting
            assert not content.endswith('\n\n'), "Should not have trailing newlines"
            assert len(content.split('\n\n')) >= 5, "Should have proper paragraph separation"
            
    finally:
        await generator.cleanup()

@pytest.mark.asyncio
async def test_seo_optimization():
    """Test SEO optimization in generated content"""
    generator = BlogGenerator()
    try:
        topic = {
            'name': 'SEO Best Practices',
            'context': 'Search engine optimization guide'
        }
        
        article = await generator.generate_article(topic)
        assert article is not None, "Should generate article"
        
        # Check meta tags
        assert article.meta_description, "Should have meta description"
        assert len(article.meta_description) <= 155, "Meta description should be proper length"
        assert article.meta_keywords, "Should have meta keywords"
        
        # Check title
        assert article.seo_title, "Should have SEO title"
        assert len(article.seo_title) <= 60, "SEO title should be proper length"
        
        # Check keyword density
        content_lower = article.content.lower()
        keyword_count = content_lower.count(topic['name'].lower())
        words = len(content_lower.split())
        keyword_density = keyword_count / words
        
        assert 0.01 <= keyword_density <= 0.03, "Should have proper keyword density"
        
    finally:
        await generator.cleanup()

@pytest.mark.asyncio
async def test_content_humanization():
    """Test content humanization"""
    generator = BlogGenerator()
    try:
        topic = {
            'name': 'Writing Natural Content',
            'context': 'Content writing guide'
        }
        
        article = await generator.generate_article(topic)
        assert article is not None, "Should generate article"
        
        # Check for common filler phrases
        filler_phrases = Config.FILLER_PHRASES
        content_lower = article.content.lower()
        
        filler_count = sum(content_lower.count(phrase.lower()) for phrase in filler_phrases)
        assert filler_count == 0, "Should not contain filler phrases"
        
        # Check sentence variety
        sentences = content_lower.split('.')
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        assert 10 <= avg_length <= 25, "Should have reasonable sentence lengths"
        
        # Check paragraph structure
        paragraphs = [p for p in article.content.split('\n\n') if p.strip()]
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        
        assert all(10 <= l <= 150 for l in paragraph_lengths), "Paragraphs should be reasonable length"
        
    finally:
        await generator.cleanup()

@pytest.mark.asyncio
async def test_image_handling():
    """Test image handling in generated articles"""
    generator = BlogGenerator()
    try:
        topic = {
            'name': 'Web Design Trends',
            'context': 'Modern web design guide'
        }
        
        article = await generator.generate_article(topic)
        assert article is not None, "Should generate article"
        
        # Check images
        assert article.images, "Should have images"
        assert len(article.images) <= 3, "Should have reasonable number of images"
        
        # Check image properties
        for image in article.images:
            assert image.get('url'), "Image should have URL"
            assert image.get('alt_text'), "Image should have alt text"
            assert image.get('caption'), "Image should have caption"
            
            # Check dimensions if available
            if 'dimensions' in image:
                width = image['dimensions'].get('width', 0)
                height = image['dimensions'].get('height', 0)
                assert width >= Config.MIN_IMAGE_WIDTH, "Image should meet minimum width"
                assert height >= Config.MIN_IMAGE_HEIGHT, "Image should meet minimum height"
                
    finally:
        await generator.cleanup()

def run_tests():
    """Run all tests"""
    try:
        # Configure asyncio for Windows if needed
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        # Run tests
        pytest.main([__file__, '-v'])
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise

if __name__ == "__main__":
    run_tests()