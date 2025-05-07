from typing import List
from utils import logger
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter


class KeywordFinder:
    def __init__(self):
        self.stop_words = set([
            'the', 'and', 'of', 'to', 'in', 'for', 'is', 'on', 'that', 'by', 
            'this', 'with', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at',
            'as', 'your', 'all', 'have', 'new', 'more', 'an', 'was', 'we', 'will',
            'home', 'can', 'us', 'about', 'if', 'page', 'my', 'has', 'search', 'free',
            'but', 'our', 'one', 'other', 'do', 'no', 'information', 'time', 'they'
        ])

    def get_related_keywords(self, topic: str) -> List[str]:
        """Get related keywords for a given topic"""
        try:
            # Get search suggestions
            suggestions = self._get_search_suggestions(topic)
            
            # Analyze SERP for top keywords
            serp_keywords = self._analyze_serp(topic)
            
            # Combine and rank keywords
            keywords = self._rank_keywords(suggestions + serp_keywords)
            
            return keywords[:10]  # Return top 10 keywords
        except Exception as e:
            logger.error(f"Error finding keywords: {str(e)}")
            return []

    def _get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions with proper type handling"""
        try:
            url = f"https://suggestqueries.google.com/complete/search?output=toolbar&q={query}"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'xml')
            
            # Use get() method and convert to list of strings
            suggestions: List[str] = []
            for suggestion in soup.find_all('suggestion'):
                if suggestion and suggestion.get('data'):
                    suggestions.append(str(suggestion.get('data')))
            return suggestions

        except Exception as e:
            logger.error(f"Error getting search suggestions: {e}")
            return []

    def _analyze_serp(self, query: str) -> List[str]:
        """Analyze SERP for top keywords"""
        try:
            url = f"https://www.google.com/search?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract keywords from titles and descriptions
            titles = [h3.text for h3 in soup.find_all('h3')]
            descriptions = [div.text for div in soup.find_all('div', class_='VwiC3b')]
            
            # Extract keywords
            all_text = ' '.join(titles + descriptions)
            words = re.findall(r'\b\w+\b', all_text.lower())
            filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            # Get most common keywords
            keyword_counts = Counter(filtered_words)
            return [kw for kw, cnt in keyword_counts.most_common(10)]
        except Exception as e:
            logger.error(f"Error analyzing SERP: {str(e)}")
            return []

    def _rank_keywords(self, keywords: List[str]) -> List[str]:
        """Rank keywords based on relevance and popularity"""
        try:
            # Simple ranking based on frequency
            keyword_counts = Counter(keywords)
            return [kw for kw, cnt in keyword_counts.most_common()]
        except Exception as e:
            logger.error(f"Error ranking keywords: {str(e)}")
            return []
