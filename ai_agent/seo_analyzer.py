import asyncio
import re
import random
import json
import time
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator, Union
from collections import Counter
from datetime import datetime
from bs4 import BeautifulSoup, Tag
from urllib.parse import quote_plus, urlparse
from utils import logger
from utils.request_limiter import RequestLimiter
from utils.network_resilience import NetworkResilience
from textblob import TextBlob
import os
import aiohttp
import warnings
import logging
from fake_useragent import UserAgent
from keyword_researcher import KeywordResearcher
from config import Config

logging.captureWarnings(True)


class DummyDoc:
    """Simple document class that implements iteration"""
    def __init__(self, text):
        self.text = text
        self.sents = [text]

    def __iter__(self) -> Iterator[str]:
        """Make DummyDoc iterable"""
        yield self.text

class SEOAnalyzer:
    """
    Advanced SEO Analyzer that implements sophisticated keyword research and SEO analysis
    without relying on external API keys.

    Features:
    - Search engine autocomplete scraping
    - SERP analysis for keyword extraction
    - Competitor keyword analysis
    - Search volume estimation
    - Keyword difficulty scoring
    - Intent classification
    - Trend analysis
    - LSI keyword generation
    - Long-tail keyword discovery
    - Question-based keyword generation
    - Content optimization
    """
    def __init__(self):
        """Initialize SEO analyzer with scraping capabilities"""
        self.session = None
        self.search_engines = {
            'bing': 'https://www.bing.com/search?q=',
            'yahoo': 'https://search.yahoo.com/search?p=',
            'duckduckgo': 'https://duckduckgo.com/html/?q='
        }
        self.headers_list = [
            {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/89.0'},
            {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'},
            {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/91.0.864.59'}
        ]
        self.delay = 2  # Delay between requests

    async def _init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def analyze_keyword(self, keyword: str) -> Dict[str, Any]:
        """Analyze keyword using web scraping"""
        try:
            await self._init_session()
            
            # Scrape search results from multiple engines
            results: Dict[str, List[Dict[str, str]]] = {}
            for engine, url in self.search_engines.items():
                engine_results = await self._scrape_search_results(
                    url + quote_plus(keyword)
                )
                if engine_results:
                    results[engine] = engine_results
                await asyncio.sleep(self.delay)

            # Get related keywords from search suggestions
            related = await self._get_search_suggestions(keyword)
            questions = await self._extract_questions_from_results(results)
            
            # Extract and analyze data
            analysis = {
                'variations': await self._extract_keyword_variations(results, keyword),
                'questions': questions,
                'intent': self._determine_intent(keyword, results),
                'metrics': self._calculate_metrics(results),
                'competition': len(results.get('bing', [])),
                'related_searches': related
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing keyword: {e}")
            return self._get_fallback_analysis(keyword)

    async def _scrape_search_results(self, url: str) -> List[Dict]:
        """Scrape search results from a given URL"""
        try:
            headers = random.choice(self.headers_list)

            # Ensure session is initialized
            if not self.session:
                await self._init_session()

            if not self.session:
                logger.error("Failed to initialize session")
                return []

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    # Extract search results (adjust selectors as needed)
                    for result in soup.select('.result,.g'):
                        title = result.select_one('h3,h2')
                        snippet = result.select_one('.snippet,.desc')
                        if title:
                            results.append({
                                'title': title.get_text(strip=True),
                                'snippet': snippet.get_text(strip=True) if snippet else ''
                            })
                    return results
                return []
        except Exception as e:
            logger.error(f"Error scraping results: {e}")
            return []

    async def _get_search_suggestions(self, keyword: str) -> List[str]:
        """Get search suggestions using autocomplete"""
        try:
            suggestions: Set[str] = set()
            
            # Use only Google and Bing for suggestions
            sources = [
                f"http://suggestqueries.google.com/complete/search?output=toolbar&q={quote_plus(keyword)}",
                f"https://api.bing.com/osjson.aspx?query={quote_plus(keyword)}"
            ]
            
            for source in sources:
                try:
                    headers = random.choice(self.headers_list)
                    if not self.session:
                        await self._init_session()
                    if self.session:
                        async with self.session.get(source, headers=headers) as response:
                            if response.status == 200:
                                if 'google' in source:
                                    data = await response.text()
                                    soup = BeautifulSoup(data, 'xml')
                                    for suggestion in soup.find_all('suggestion'):
                                        if isinstance(suggestion, Tag):
                                            data_attr = suggestion.get('data')
                                            if data_attr:
                                                suggestions.add(data_attr)
                                else:  # Bing
                                    try:
                                        json_data = await response.json()
                                        if isinstance(json_data, list) and len(json_data) > 1:
                                            suggestions.update(json_data[1])
                                    except json.JSONDecodeError:
                                        continue
                except Exception as e:
                    logger.error(f"Error getting suggestions from {source}: {e}")
                await asyncio.sleep(self.delay)
                
            return list(suggestions)
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {e}")
            return []

    def _get_fallback_analysis(self, keyword: str) -> Dict[str, Any]:
        """Get fallback analysis when main analysis fails"""
        return {
            'variations': [keyword],
            'questions': [],
            'intent': 'informational',
            'metrics': {},
            'competition': 0,
            'related_searches': []
        }

    async def _extract_keyword_variations(self, results: Dict[str, List[Dict]], keyword: str) -> List[str]:
        """Extract keyword variations from results"""
        variations = set([keyword])
        
        # Add basic variations
        variations.add(f"how to {keyword}")
        variations.add(f"what is {keyword}")
        variations.add(f"best {keyword}")
        
        # Process each result
        for engine_results in results.values():
            if not engine_results:  # Skip if results is None or empty
                continue
                
            for result in engine_results:
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                if title:
                    variations.add(title.lower())
                if snippet:
                    variations.add(snippet.lower())
        
        return list(variations)

    async def _extract_questions_from_results(self, results: Dict[str, List[Dict]]) -> List[str]:
        """Extract questions from search results"""
        questions: Set[str] = set()
        
        # Process each result
        for engine_results in results.values():
            if not engine_results:  # Skip if results is None or empty
                continue
                
            for result in engine_results:
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                
                # Look for question patterns
                for text in [title, snippet]:
                    if not text:  # Skip if text is None or empty
                        continue
                    
                    questions.update(self._extract_questions_from_text(text))
        
        return list(questions)

    def _extract_questions_from_text(self, text: str) -> Set[str]:
        """Extract questions from a single text"""
        questions = set()
        if not text:
            return questions

        # Question patterns
        patterns = [
            r'(?:^|(?<=[.!?])\s+)(what|how|why|when|where|which|who)\s+(?:[^.!?])+[?]',
            r'(?:^|(?<=[.!?])\s+)(can|should|will|does|do)\s+(?:[^.!?])+[?]'
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            questions.update(match.group(0).strip() for match in matches)

        return questions

    def _determine_intent(self, keyword: str, serp_data: Dict[str, Any]) -> str:
        """Determine search intent from keyword and SERP data"""
        try:
            # Check patterns in keyword
            keyword = keyword.lower()
            
            if any(w in keyword for w in ['how', 'tutorial', 'guide', 'steps']):
                return 'how-to'
            elif any(w in keyword for w in ['what', 'who', 'when', 'where', 'why']):
                return 'informational'
            elif any(w in keyword for w in ['buy', 'price', 'cost', 'deal', 'cheap']):
                return 'transactional'
            elif any(w in keyword for w in ['best', 'top', 'vs', 'versus', 'review']):
                return 'commercial'
            
            return 'informational'  # Default intent
            
        except Exception as e:
            logger.error(f"Error determining intent: {e}")
            return 'informational'

    def _calculate_metrics(self, serp_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate SEO metrics from SERP data with proper null checks"""
        try:
            # Define default metrics
            default_metrics = {
                'competition': 0.5,
                'search_volume': 0,
                'keyword_difficulty': 50,
                'content_gaps': []
            }

            # Handle None case for serp_data
            if not serp_data:
                return default_metrics

            # Check if serp_data is a dictionary
            if not isinstance(serp_data, dict):
                return default_metrics

            # Safely get bing results with null checks
            bing_results = serp_data.get('bing', [])
            if not isinstance(bing_results, list):
                bing_results = []

            # Calculate competition
            competition = len(bing_results) / 10 if bing_results else 0.5

            return {
                'competition': competition,
                'search_volume': 0,  # Would need paid API for real data
                'keyword_difficulty': 50,  # Default medium difficulty
                'content_gaps': []
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'competition': 0.5,
                'search_volume': 0,
                'keyword_difficulty': 50,
                'content_gaps': []
            }

    async def _get_related_searches(self, keyword: str) -> Dict[str, List[str]]:
        """Get related searches for a keyword"""
        try:
            related = set()
            questions = set()
            
            # Extract from search suggestions
            suggestions = await self._get_search_suggestions(keyword)
            
            # Categorize suggestions
            for suggestion in suggestions:
                if any(suggestion.lower().startswith(q) for q in ['what', 'how', 'why', 'when', 'where', 'which', 'who']):
                    questions.add(suggestion)
                else:
                    related.add(suggestion)
            
            return {
                'related': list(related),
                'questions': list(questions)
            }
        except Exception as e:
            logger.error(f"Error getting related searches: {e}")
            return {'related': [], 'questions': []}

    async def analyze_serp_data(self, keyword: str) -> Dict[str, Any]:
        """Analyze SERP data for a keyword"""
        try:
            if not keyword:
                return {}

            # Get search results
            results = {}
            for engine, url in self.search_engines.items():
                results[engine] = await self._scrape_search_results(
                    url + quote_plus(keyword)
                )
                await asyncio.sleep(self.delay)

            # Analyze results
            analysis = {
                'organic_results': [],
                'featured_snippets': [],
                'related_searches': set(),
                'questions': set()
            }

            # Process each result
            for engine_results in results.values():
                for result in engine_results:
                    analysis['organic_results'].append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', '')
                    })
                    
                    # Extract questions
                    text = f"{result.get('title', '')} {result.get('snippet', '')}"
                    for question in self._extract_questions_from_text(text):
                        analysis['questions'].add(question)

            # Add related searches
            suggestions = await self._get_search_suggestions(keyword)
            analysis['related_searches'] = set(suggestions)

            # Convert sets to lists for JSON serialization
            analysis['related_searches'] = list(analysis['related_searches'])
            analysis['questions'] = list(analysis['questions'])

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing SERP data: {e}")
            return {
                'organic_results': [],
                'featured_snippets': [],
                'related_searches': [],
                'questions': []
            }

    async def _get_serp_results(self, query: str, engine: str) -> List[Dict]:
        """Get SERP results with proper error handling"""
        try:
            if engine not in self.search_engines:
                logger.error(f"Unknown search engine: {engine}")
                return []

            engine_config = self.search_engines.get(engine)
            if not engine_config:
                return []

            if not self.session:
                await self._init_session()
                if not self.session:
                    return []

            headers = random.choice(self.headers_list)
            
            try:
                url = engine_config
                async with self.session.get(url + quote_plus(query), headers=headers, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Error status {response.status} from {engine}")
                        return []

                    html = await response.text()
                    return self._parse_serp_results(html, engine)
            except Exception as e:
                logger.error(f"Error fetching SERP from {engine}: {e}")
                return []

        except Exception as e:
            logger.error(f"Error in SERP results: {e}")
            return []

    async def analyze_content(self, content: str, topic: str) -> Dict:
        """Analyze content for SEO optimization"""
        try:
            # Get basic keyword analysis
            keyword_analysis = await self.analyze_keyword(topic)
            
            # Analyze content structure
            structure_analysis = self._analyze_content_structure(content)
            
            # Calculate keyword density
            keyword_density = self._calculate_keyword_density(content, topic)
            
            return {
                'variations': keyword_analysis.get('variations', []),
                'questions': keyword_analysis.get('questions', []),
                'density': keyword_density,
                'structure': structure_analysis,
                'keywords': keyword_analysis.get('keywords', []),
                'seo_score': self._calculate_seo_score(content, topic, keyword_density),
                'recommendations': self._generate_recommendations(content, topic, keyword_density)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {}

    def _analyze_content_structure(self, content: str) -> Dict:
        """Analyze content structure for SEO"""
        try:
            paragraphs = content.split('\n\n')
            headings = [line for line in content.split('\n') if line.strip().startswith('#')]
            
            return {
                'num_paragraphs': len(paragraphs),
                'num_headings': len(headings),
                'avg_paragraph_length': sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")
            return {}

    def _calculate_keyword_density(self, content: str, topic: str) -> float:
        """Calculate keyword density in content"""
        try:
            words = content.lower().split()
            topic_words = topic.lower().split()
            
            # Count occurrences of topic words
            topic_count = sum(1 for word in words if any(topic_word in word for topic_word in topic_words))
            
            # Calculate density
            density = topic_count / len(words) if words else 0
            return round(density * 100, 2)  # Return as percentage
            
        except Exception as e:
            logger.error(f"Error calculating keyword density: {e}")
            return 0.0

    def _calculate_seo_score(self, content: str, topic: str, keyword_density: float) -> int:
        """Calculate overall SEO score"""
        try:
            score = 100
            
            # Check keyword density
            if keyword_density < Config.MIN_KEYWORD_DENSITY * 100:
                score -= 20
            elif keyword_density > Config.MAX_KEYWORD_DENSITY * 100:
                score -= 10
                
            # Check content length
            word_count = len(content.split())
            if word_count < Config.MIN_ARTICLE_LENGTH:
                score -= 20
            
            # Check headings
            headings = [line for line in content.split('\n') if line.strip().startswith('#')]
            if len(headings) < Config.MIN_HEADINGS:
                score -= 15
                
            return max(0, score)  # Ensure score doesn't go below 0
            
        except Exception as e:
            logger.error(f"Error calculating SEO score: {e}")
            return 0

    def _generate_recommendations(self, content: str, topic: str, keyword_density: float) -> List[str]:
        """Generate SEO recommendations"""
        try:
            recommendations = []
            
            # Check keyword density
            if keyword_density < Config.MIN_KEYWORD_DENSITY * 100:
                recommendations.append(f"Increase keyword density (currently {keyword_density}%)")
            elif keyword_density > Config.MAX_KEYWORD_DENSITY * 100:
                recommendations.append(f"Reduce keyword density (currently {keyword_density}%)")
                
            # Check content length
            word_count = len(content.split())
            if word_count < Config.MIN_ARTICLE_LENGTH:
                recommendations.append(f"Increase content length (currently {word_count} words)")
                
            # Check headings
            headings = [line for line in content.split('\n') if line.strip().startswith('#')]
            if len(headings) < Config.MIN_HEADINGS:
                recommendations.append(f"Add more headings (currently {len(headings)})")
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
