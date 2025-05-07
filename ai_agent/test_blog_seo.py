import asyncio
import pytest
from blog_generator_new import BlogGenerator
from content_optimizer import ContentOptimizer
from seo_validator import SEOValidator

@pytest.mark.asyncio
async def test_blog_generation_with_seo():
    """Test blog generation with SEO optimization"""
    generator = BlogGenerator()
    optimizer = ContentOptimizer()
    
    test_topic = {
        'name': 'Artificial Intelligence in Healthcare 2025',
        'context': 'technology healthcare ai'
    }
    
    # Generate initial article
    article = await generator.generate_article(test_topic)
    assert article is not None, "Article generation failed"
    
    # Validate SEO metrics
    assert article.seo_score >= 0.5, f"SEO score too low: {article.seo_score}"
    assert article.readability_score >= 0.6, f"Readability score too low: {article.readability_score}"
    
    # Validate heading structure
    assert article.heading_structure.get('h1') == 1, "Should have exactly one H1"
    assert article.heading_structure.get('h2', 0) >= 3, "Should have at least 3 H2 sections"
    
    # Validate keyword integration
    assert len(article.keywords) > 0, "No keywords found"
    assert len(article.semantic_keywords) > 0, "No semantic keywords found"
    
    # Test content optimization
    optimization_result = optimizer.optimize_content(article.content, {
        'primary': ['artificial intelligence', 'healthcare', 'AI'],
        'secondary': ['machine learning', 'medical diagnosis', 'patient care'],
        'long_tail': ['AI-powered healthcare solutions', 'medical image analysis AI']
    })
    
    assert optimization_result['seo_score'] >= article.seo_score, "Optimization didn't improve SEO score"
    assert optimization_result['readability_score'] >= 0.7, "Optimization didn't achieve target readability"

@pytest.mark.asyncio
async def test_seo_validation():
    """Test SEO validation functionality"""
    validator = SEOValidator()
    
    test_content = """
    <h1>Artificial Intelligence in Healthcare</h1>
    
    <h2>Introduction to AI in Medicine</h2>
    AI is revolutionizing healthcare through advanced algorithms and machine learning.
    
    <h2>Key Applications</h2>
    Medical diagnosis and patient care are being transformed by AI technologies.
    
    <h2>Future Trends</h2>
    AI-powered healthcare solutions will continue to evolve.
    """
    
    test_keywords = {
        'primary': ['artificial intelligence', 'healthcare', 'AI'],
        'secondary': ['machine learning', 'medical diagnosis'],
        'questions': ['How is AI used in healthcare?']
    }
    
    result = validator.validate(test_content, test_keywords)
    
    assert result['score'] > 0, "Should return a positive SEO score"
    assert isinstance(result['suggestions'], list), "Should return optimization suggestions"
    assert result['details']['primary_keyword_usage']['found_in_title'], "Primary keyword should be in title"

if __name__ == "__main__":
    asyncio.run(test_blog_generation_with_seo())
    asyncio.run(test_seo_validation())