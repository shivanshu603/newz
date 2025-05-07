import asyncio
from typing import Dict, List, Set
import re
from datetime import datetime
from utils import logger
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seo_analyzer import SEOAnalyzer

class SEOHelper:
    """
    A wrapper for the SEOAnalyzer class that provides backward compatibility
    with the blog_generator_new.py expectations.

    This class delegates all functionality to the SEOAnalyzer class.
    """

    def __init__(self):
        logger.info("Initializing SEO Helper (wrapper for SEOAnalyzer)")
        self.analyzer = SEOAnalyzer()
        
    async def analyze(self, keyword: str) -> Dict:
        """
        Delegate to the SEOAnalyzer class for keyword analysis

        Args:
            keyword: The main keyword to analyze

        Returns:
            Dict containing keyword analysis
        """
        try:
            logger.info(f"SEOHelper delegating analysis to SEOAnalyzer for: {keyword}")
            return await self.analyzer.analyze_keyword(keyword)
        except Exception as e:
            logger.error(f"Error in SEOHelper analyze method: {e}")
            # Return minimal fallback data
            return {
                'keyword': keyword,
                'variations': [
                    f"best {keyword}",
                    f"how to use {keyword}",
                    f"{keyword} guide",
                    f"{keyword} tutorial",
                    f"what is {keyword}"
                ],
                'metrics': {
                    'difficulty': 0.5,
                    'volume': 'medium',
                    'competition': 'moderate'
                }
            }
    
    # All other methods are delegated to the SEOAnalyzer class
    # This class is just a wrapper for backward compatibility