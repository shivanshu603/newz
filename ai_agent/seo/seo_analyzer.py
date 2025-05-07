import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from urllib.parse import quote_plus
from typing import Dict, List, Optional, Set, Any, Union
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
from utils.keyword_ranker import KeywordRanker
import asyncio
import aiohttp
from bs4 import BeautifulSoup, Tag
import json
import random  # Add missing import
from textblob import TextBlob

from utils import logger

class SEOAnalyzer:
    def __init__(self):
        """Initialize SEO analyzer with transformer model"""
        try:
            # Initialize transformer model with proper error handling
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.tfidf = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3)
            )
            logger.info("Transformer model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing transformer model: {e}")
            self.model = None
            self.tfidf = None

        # Initialize variation generators
        self.variation_generators = {
            'modifiers': ['best', 'top', 'guide to', 'tutorial', 'how to'],
            'prefixes': ['the', 'a complete', 'ultimate', 'essential'],
            'suffixes': ['guide', 'tutorial', 'tips', 'strategies'],
            'intent': ['how to', 'what is', 'why', 'when to']
        }

        self.error_log = []  # Add error log list
        
        # Initialize HTTP headers for requests
        self.headers_list = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-GB,en;q=0.9'
            }
        ]
        
        self.ensure_methods()
        logger.info("SEO Analyzer initialized with all required attributes")

    def _verify_initialization(self) -> bool:
        """Verify if all required attributes are initialized"""
        required_attrs = [
            'model', 'tfidf', 'variation_generators'
        ]
        return all(hasattr(self, attr) for attr in required_attrs)

    def ensure_methods(self) -> bool:
        """Ensure critical methods exist"""
        try:
            required_methods = {
                '_generate_variations': self._default_generate_variations,
                '_analyze_competition': self._default_analyze_competition,
                '_analyze_content_gaps': self._default_analyze_content_gaps,
                '_analyze_semantic_relevance': self._default_analyze_semantic_relevance,
                '_analyze_search_intent': self._default_analyze_search_intent,
                '_analyze_serp_features': self._default_analyze_serp_features
            }
            
            for method_name, default_impl in required_methods.items():
                if not hasattr(self, method_name):
                    setattr(self, method_name, default_impl.__get__(self, self.__class__))
                    logger.info(f"Bound method: {method_name}")
            
            logger.info("SEO Analyzer methods verified")
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring methods: {e}")
            return False

    def _default_generate_variations(self, keyword: str) -> List[str]:
        """Generate keyword variations based on context and related terms."""
        logger.info(f"Generating variations for: {keyword}")
        variations = set()
        try:
            if not keyword or not isinstance(keyword, str):
                return [keyword] if keyword else []
            
            keyword = keyword.lower().strip()
            words = keyword.split()
            
            # Add original variations
            variations.add(keyword)
            variations.add(keyword.title())
            variations.add(' '.join(w.capitalize() for w in words))
            
            # Use synonyms or related terms to generate variations
            # Example: Fetch synonyms from a predefined list or an API
            synonyms = self.fetch_synonyms(keyword) if hasattr(self, 'fetch_synonyms') else []
            for synonym in synonyms:
                variations.add(synonym)
            
            return list(variations)
            
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            return [keyword] if keyword else []

    def _generate_variations(self, keyword: Union[str, Dict[str, Any]]) -> List[str]:
        """Generate keyword variations with robust input handling"""
        try:
            # Handle input types
            if isinstance(keyword, dict):
                keyword = keyword.get('keyword', '')
            elif not isinstance(keyword, str):
                return []
                
            if not keyword:
                return []
                
            # Clean and normalize keyword
            keyword = keyword.lower().strip()
            words = keyword.split()
            
            # Generate all variations
            variations = {
                keyword,
                keyword.title(),
                ' '.join(w.capitalize() for w in words)
            }
            
            # Add modified variations
            for modifier in self.variation_generators['modifiers']:
                variations.add(f"{modifier} {keyword}")
            
            for prefix in self.variation_generators['prefixes']:
                variations.add(f"{prefix} {keyword}")
                
            for suffix in self.variation_generators['suffixes']:
                variations.add(f"{keyword} {suffix}")
                
            for intent in self.variation_generators['intent']:
                variations.add(f"{intent} {keyword}")
                
            return list(variations)
            
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            return []

    async def analyze_keyword(self, keyword: str) -> Dict[str, Any]:
        """Main method for keyword analysis with proper async handling"""
        try:
            if not self._verify_initialization():
                raise ValueError("SEOAnalyzer not properly initialized")

            # Generate variations first
            variations = self._generate_variations(keyword)
            
            # Execute async methods sequentially since some may depend on others
            competition = await self._analyze_competition(keyword)
            gaps = await self._analyze_content_gaps(keyword)
            semantic = await self._analyze_semantic_relevance(keyword)
            intent = await self._analyze_search_intent(keyword)
            serp = await self._analyze_serp_features(keyword)
            
            # Process results safely
            metrics = {
                'competition': competition if not isinstance(competition, Exception) else {},
                'content_gaps': gaps if not isinstance(gaps, Exception) else [],
                'semantic': semantic if not isinstance(semantic, Exception) else {},
                'intent': intent if not isinstance(intent, Exception) else {},
                'serp': serp if not isinstance(serp, Exception) else {}
            }
            
            return {
                'keyword': keyword,
                'variations': list(variations),  # Convert set to list
                'metrics': metrics
            }

        except Exception as e:
            logger.error(f"Error analyzing keyword: {e}")
            return {
                'keyword': keyword,
                'variations': [keyword],
                'metrics': {
                    'difficulty': 0.5,
                    'volume': 'medium',
                    'competition': 'moderate'
                }
            }

    def _extract_content_from_serp(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract content from SERP results"""
        try:
            content = {}
            
            # Extract titles and links safely
            for result in soup.find_all('div', class_='g'):
                title_elem = result.find('h3')
                link_elem = result.find('a')
                
                if title_elem and link_elem and isinstance(link_elem, Tag):
                    href = link_elem.get('href', '')
                    if href:
                        content[href] = {
                            'title': title_elem.get_text(strip=True),
                            'url': href
                        }
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting SERP content: {e}")
            return {}

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            if self.model is None:
                return 0.0
                
            # Get embeddings
            embedding1 = self.model.encode([text1])[0]
            embedding2 = self.model.encode([text2])[0]
            
            # Calculate cosine similarity
            similarity = float(np.dot(embedding1, embedding2) / 
                            (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def _default_analyze_competition(self, keyword: str):
        """Default implementation for competition analysis"""
        return {'difficulty': 0.5, 'competition': 'moderate'}

    def _extract_content_topics(self, content: str) -> List[str]:
        """Extract key topics from content using TF-IDF"""
        try:
            if not content or not isinstance(content, str):
                return []
                
            # Clean content
            content = re.sub(r'<[^>]+>', ' ', content)  # Remove HTML tags
            content = re.sub(r'\s+', ' ', content).strip()
            
            if not content:
                return []
                
            # Get top keywords using TF-IDF
            if self.tfidf is None:
                return []
                
            tfidf_matrix = self.tfidf.fit_transform([content])
            feature_names = self.tfidf.get_feature_names_out()
            
            # Get top 10 keywords by TF-IDF score
            feature_indices = tfidf_matrix.sum(axis=0).argsort()[0, -10:][::-1]
            top_keywords = []
            
            for i in feature_indices:
                feature = feature_names[i]
                if isinstance(feature, str) and feature.isalpha() and len(feature) > 3:
                    top_keywords.append(str(feature))
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"Error extracting content topics: {e}")
            return []

    async def _default_analyze_content_gaps(self, keyword: str) -> List[str]:
        """Analyze content gaps by comparing with top competitors"""
        try:
            gaps = []
            competitors = await self._get_top_competitors(keyword)
            
            # Get our own content topics for comparison
            our_topics = set(self._extract_content_topics(keyword))
            
            for comp in competitors[:5]:  # Analyze top 5 competitors
                if comp.get('content'):
                    comp_topics = set(self._extract_content_topics(comp['content']))
                    # Find topics they cover that we don't
                    missing_topics = comp_topics - our_topics
                    gaps.extend(missing_topics)
                    
            # Return unique gaps sorted by relevance
            unique_gaps = list(set(gaps))
            return sorted(unique_gaps, 
                        key=lambda x: self.calculate_semantic_similarity(keyword, x),
                        reverse=True)[:10]  # Return top 10 most relevant gaps
            
        except Exception as e:
            logger.error(f"Error in content gap analysis: {e}")
            return []

    async def _default_analyze_semantic_relevance(self, keyword: str):
        """Analyze semantic relevance using transformer model"""
        try:
            if not self.model:
                return {'relevance': 0.5, 'related_terms': []}
                
            # Get related terms from search results
            results = await self._fetch_serp_data(keyword)
            titles = [r['title'] for r in results if 'title' in r]
            descriptions = [r.get('description', '') for r in results]
            
            # Extract key phrases
            all_text = ' '.join(titles + descriptions)
            key_phrases = self._extract_content_topics(all_text)
            
            # Calculate semantic relevance scores
            related_terms = []
            for phrase in set(key_phrases):
                if phrase.lower() != keyword.lower():
                    score = self.calculate_semantic_similarity(keyword, phrase)
                    if score > 0.3:  # Only include reasonably relevant terms
                        related_terms.append((phrase, score))
            
            # Sort by relevance score
            related_terms.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'relevance': 1.0 if related_terms else 0.5,
                'related_terms': [term[0] for term in related_terms[:10]]  # Top 10
            }
            
        except Exception as e:
            logger.error(f"Error analyzing semantic relevance: {e}")
            return {'relevance': 0.5, 'related_terms': []}

    async def _get_top_competitors(self, keyword: str) -> List[Dict]:
        """Get top competitors for a keyword"""
        try:
            results = await self._fetch_serp_data(query=keyword)
            return [{
                'url': r.get('url', ''),
                'title': r.get('title', ''),
                'content': r.get('content', '')
            } for r in results[:5]]  # Return top 5 results
        except Exception as e:
            logger.error(f"Error getting competitors: {e}")
            return []

    async def _default_analyze_search_intent(self, keyword: str):
        """Default implementation for search intent analysis"""
        return {'intent': 'informational', 'confidence': 0.5}

    async def _default_analyze_serp_features(self, keyword: str) -> Dict:
        """Enhanced SERP feature analysis with more detailed results"""
        try:
            features = {
                'featured_snippet': {'exists': False, 'type': None, 'content': None},
                'people_also_ask': [],
                'knowledge_panel': {'exists': False, 'type': None},
                'local_pack': {'exists': False, 'count': 0},
                'video_results': {'exists': False, 'count': 0},
                'shopping_results': {'exists': False, 'count': 0}
            }
            
            results = await self._fetch_serp_data(keyword)
            
            for result in results:
                # Check for featured snippet
                if result.get('featured_snippet'):
                    features['featured_snippet'] = {
                        'exists': True,
                        'type': result.get('featured_type', 'unknown'),
                        'content': result.get('featured_content', '')
                    }
                
                # Check for PAA questions
                if result.get('questions'):
                    features['people_also_ask'].extend(result['questions'])
                
                # Check for knowledge panel
                if result.get('knowledge_panel'):
                    features['knowledge_panel'] = {
                        'exists': True,
                        'type': result.get('panel_type', 'generic')
                    }
                
                # Check for local pack
                if result.get('local_results'):
                    features['local_pack'] = {
                        'exists': True,
                        'count': len(result['local_results'])
                    }
                
                # Check for video results
                if result.get('video_results'):
                    features['video_results'] = {
                        'exists': True,
                        'count': len(result['video_results'])
                    }
                
                # Check for shopping results
                if result.get('shopping_results'):
                    features['shopping_results'] = {
                        'exists': True,
                        'count': len(result['shopping_results'])
                    }
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing SERP features: {e}")
            return {
                'featured_snippet': False,
                'people_also_ask': [],
                'knowledge_panel': False,
                'local_pack': False,
                'video_results': False
            }

    async def _get_search_data(self, keyword: str) -> Dict[str, Any]:
        """Get comprehensive search data"""
        try:
            # Pass keyword as query parameter
            results = await self._fetch_serp_data(query=keyword)  # Fix: Add query parameter
            
            related_searches = await self._get_related_searches(keyword)
            serp_features = self._extract_serp_features(results)
            
            return {
                'related_searches': related_searches,
                'features': serp_features,
                'results': results
            }
        except Exception as e:
            logger.error(f"Error getting search data: {e}")
            raise

    def _parse_serp_results(self, html: str, engine: str) -> List[Dict]:
        """Parse SERP HTML into structured results"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []

            if engine == 'google':
                # Updated Google search results container class to 'tF2Cxc'
                google_results = soup.find_all('div', class_='tF2Cxc')
                logger.debug(f"Found {len(google_results)} Google result containers")
                for result in google_results:
                    title_elem = result.find('h3')
                    # Find the first 'a' tag anywhere inside the result div
                    link_elem = result.find('a')
                    if title_elem and link_elem:
                        url = link_elem.get('href', '')
                        if not url:
                            # Try data-href attribute as fallback
                            url = link_elem.get('data-href', '')
                        if url:
                            results.append({
                                'title': title_elem.get_text(strip=True),
                                'url': url,
                                'engine': engine
                            })
            elif engine == 'bing':
                bing_results = soup.find_all('li', class_='b_algo')
                logger.debug(f"Found {len(bing_results)} Bing result containers")
                for result in bing_results:
                    title = result.find('h2')
                    link = result.find('a')
                    if title and link:
                        results.append({
                            'title': title.get_text(strip=True),
                            'url': link.get('href', ''),
                            'engine': engine
                        })
            elif engine == 'yahoo':
                yahoo_results = soup.find_all('div', class_='dd')
                logger.debug(f"Found {len(yahoo_results)} Yahoo result containers")
                for result in yahoo_results:
                    title = result.find('h3')
                    link = result.find('a')
                    if title and link:
                        results.append({
                            'title': title.get_text(strip=True),
                            'url': link.get('href', ''),
                            'engine': engine
                        })

            return results

        except Exception as e:
            logger.error(f"Error parsing {engine} results: {e}")
            return []

    async def _fetch_serp_data(self, query: Union[str, bytes]) -> List[Dict]:
        """Fetch SERP data with robust query handling and enhanced logging"""
        try:
            if not hasattr(self, 'search_engines'):
                self.search_engines = {
                    'google': 'https://www.google.com/search?q={}',
                    'bing': 'https://www.bing.com/search?q={}',
                    'yahoo': 'https://search.yahoo.com/search?p={}'
                }

            # Ensure query is properly encoded
            if isinstance(query, bytes):
                query_str = query.decode('utf-8')
            else:
                query_str = str(query)

            results = []
            for engine, url_template in self.search_engines.items():
                try:
                    url = url_template.format(quote_plus(query_str))
                    headers = random.choice(self.headers_list)
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=headers) as response:
                            if response.status != 200:
                                logger.error(f"Non-200 response from {engine}: {response.status}")
                                continue
                            html = await response.text()
                            if engine == 'google':
                                logger.debug(f"Google HTML snippet: {html[:1000]}")
                            parsed_results = self._parse_serp_results(html, engine)
                            if not parsed_results:
                                logger.warning(f"No results parsed from {engine} for query '{query_str}'")
                            results.extend(parsed_results)

                    await asyncio.sleep(random.uniform(1, 2))  # Respect rate limits
                except Exception as e:
                    logger.error(f"Error fetching from {engine}: {e}", exc_info=True)
                    continue
            
            return results

        except Exception as e:
            logger.error(f"Error fetching SERP data: {e}", exc_info=True)
            return []

    async def _get_related_searches(self, query: str) -> Dict[str, Any]:
        """Get related searches for a query"""
        try:
            results = await self._fetch_serp_data(query=query)
            related = set()
            questions = set()
            
            for result in results:
                # Extract related searches
                if result.get('related_searches'):
                    related.update(result['related_searches'])
                # Extract questions
                if result.get('questions'):
                    questions.update(result['questions'])
                    
            return {
                'related_searches': list(related),
                'questions': list(questions)
            }
            
        except Exception as e:
            logger.error(f"Error getting related searches: {e}")
            return {'related_searches': [], 'questions': []}

    async def _analyze_serp_features(self, keyword: str) -> Dict:
        """Analyze SERP features for keyword"""
        try:
            serp_results = await self._fetch_serp_data(query=keyword)  # Fix: Add query parameter
            
            features = {
                'featured_snippet': False,
                'people_also_ask': [],
                'knowledge_panel': False,
                'local_pack': False,
                'video_results': False
            }

            for result in serp_results:
                # Check for featured snippet
                if result.get('featured_snippet'):
                    features['featured_snippet'] = True
                    
                # Check for PAA questions
                if result.get('questions'):
                    features['people_also_ask'].extend(result['questions'])

            return features

        except Exception as e:
            logger.error(f"Error analyzing SERP features: {e}")
            return {
                'featured_snippet': False,
                'people_also_ask': [],
                'knowledge_panel': False
            }

    def _optimize_keyword_density(self, content: str, primary_keywords: List[str], 
                                secondary_keywords: List[str], lsi_keywords: List[str]) -> str:
        """Optimize keyword density in content with robust input handling"""
        try:
            # Validate inputs
            if not content or not isinstance(content, str):
                return content
                
            content = str(content)  # Ensure string type
            word_count = len(content.split())
            if word_count < 100:
                return content
                
            # Ensure keywords are strings
            primary_keywords = [str(kw) for kw in primary_keywords] if primary_keywords else []
            secondary_keywords = [str(kw) for kw in secondary_keywords] if secondary_keywords else []
            lsi_keywords = [str(kw) for kw in lsi_keywords] if lsi_keywords else []
                
            # Process keywords in order of importance
            for keyword in primary_keywords:
                content = self._add_keyword_naturally(content, keyword, target_density=0.02)
            for keyword in secondary_keywords:
                content = self._add_keyword_naturally(content, keyword, target_density=0.01)
            for keyword in lsi_keywords:
                content = self._add_keyword_naturally(content, keyword, target_density=0.005)
                
            return content
        except Exception as e:
            logger.error(f"Error optimizing keyword density: {e}")
            return content

    def _optimize_heading_hierarchy(self, content: str, keywords: List[str]) -> str:
        """Optimize heading structure with keywords"""
        try:
            lines = content.split('\n')
            enhanced_lines = []
            current_level = 1
            
            for line in lines:
                if line.startswith('#'):
                    match = re.match(r'^(#+)', line)
                    if match and match.group():  # Check if group() returns a value
                        level = len(match.group())
                        if level - current_level > 1:
                            level = current_level + 1
                        current_level = level
                        
                        # Add keywords to headings naturally
                        if keywords and level <= 2:
                            heading_text = line.lstrip('#').strip()
                            if not any(kw.lower() in heading_text.lower() for kw in keywords):
                                keyword = keywords[0]
                                heading_text = f"{heading_text}: {keyword}"
                                line = f"{'#' * level} {heading_text}"
                                
                enhanced_lines.append(line)
            
            return '\n'.join(enhanced_lines)
        except Exception as e:
            logger.error(f"Error optimizing headings: {e}")
            return content

    def _add_schema_friendly_structures(self, content: str) -> str:
        """Add schema markup to content"""
        try:
            if "## FAQ" not in content and "## Frequently Asked Questions" not in content:
                faq_section = "\n## Frequently Asked Questions\n\n"
                content += faq_section
            return content
        except Exception as e:
            logger.error(f"Error adding schema structures: {e}")
            return content

    def integrate_statistics(self, content: str, stats: List[str]) -> str:
        """Add statistics to content"""
        try:
            if not stats:
                return content
                
            paragraphs = content.split('\n\n')
            for i, stat in enumerate(stats):
                if i < len(paragraphs):
                    paragraphs[i] += f"\n\nAccording to research, {stat}"
            
            return '\n\n'.join(paragraphs)
        except Exception as e:
            logger.error(f"Error integrating statistics: {e}")
            return content

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text content"""
        try:
            if not text:
                return []
            
            # Clean text
            clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = clean_text.split()
            
            # Remove stopwords
            stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
            keywords = [w for w in words if w not in stopwords and len(w) > 3]
            
            # Return unique keywords
            return list(set(keywords))
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

async def main():
    """Main execution function with proper async handling"""
    analyzer = None
    try:
        analyzer = SEOAnalyzer()
        test_keyword = "vampire  diaries"
        
        print(f"\nAnalyzing keyword: {test_keyword}")
        print("-" * 50)
        
        results = await analyzer.analyze_keyword(test_keyword)
        
        # Print results
        print("\nAnalysis Results:")
        print(f"Keyword: {results['keyword']}")
        print("\nVariations:")
        for var in results.get('variations', [])[:5]:
            print(f"- {var}")
            
        print("\nMetrics:")
        metrics = results.get('metrics', {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                print(f"{metric_name}:")
                for k, v in metric_value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{metric_name}: {metric_value}")
                
    except Exception as e:
        print(f"Error in analysis: {e}")
        
    finally:
        if analyzer:
            # Clean up any remaining tasks
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()
            
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
