import pytest
import asyncio
from content_optimizer import ContentOptimizer

@pytest.mark.asyncio
async def test_optimize_content():
    optimizer = ContentOptimizer()
    content = "This is a sample blog post."
    primary_keyword = "sample"
    result = await optimizer.optimize_content(content, primary_keyword)

    assert 'optimized_content' in result
    assert 'optimization_results' in result

@pytest.mark.asyncio
async def test_optimize_meta_tags():
    optimizer = ContentOptimizer()
    content = "This is a sample blog post."
    primary_keyword = "sample"
    result = await optimizer.optimize_meta_tags(content, primary_keyword)

    assert 'title' in result
    assert 'meta_description' in result

if __name__ == '__main__':
    pytest.main()
