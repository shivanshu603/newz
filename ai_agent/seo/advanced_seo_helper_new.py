import asyncio
import re
import random
import json
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import Counter
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
import hashlib

from utils import logger

class AdvancedSEOHelper:
    """
    Advanced SEO Helper that implements sophisticated keyword research and SEO analysis
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
    """
    
    def __init__(self):
        logger.info("Advanced SEO Helper initialized")
        
        # User agents for scraping
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        ]
        
        # Autocomplete endpoints
        self.autocomplete_endpoints = {
            'google': 'https://suggestqueries.google.com/complete/search?client=firefox&q={}',
            'bing': 'https://api.bing.com/osjson.aspx?query={}',
            'youtube': 'https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q={}'
        }
        
        # Intent classification patterns
        self.intent_patterns = {
            'informational': [
                'what', 'how', 'why', 'when', 'where', 'who', 'which', 
                'guide', 'tutorial', 'learn', 'examples', 'ideas', 'tips'
            ],
            'navigational': [
                'login', 'sign in', 'website', 'official', 'download', 
                'app', 'address', 'location', 'hours', 'near me'
            ],
            'transactional': [
                'buy', 'purchase', 'order', 'shop', 'deal', 'discount', 'coupon',
                'cheap', 'price', 'cost', 'subscription', 'free shipping'
            ],
            'commercial': [
                'best', 'top', 'review', 'vs', 'versus', 'compare', 'comparison',
                'alternative', 'difference between', 'pros and cons'
            ]
        }
        
        # Keyword modifiers for different types
        self.keyword_modifiers = {
            'general': [
                'best', 'top', 'guide', 'tutorial', 'tips', 'ideas', 'examples',
                'how to', 'what is', 'why', 'when', 'where', 'review'
            ],
            'commercial': [
                'buy', 'cheap', 'discount', 'deal', 'affordable', 'best', 'top',
                'premium', 'professional', 'review', 'vs', 'alternative'
            ],
            'local': [
                'near me', 'in [city]', 'local', 'nearby', 'closest', 'delivery',
                'open now', 'best in', 'available in'
            ],
            'question': [
                'how to', 'what is', 'why is', 'when to', 'where to', 'who is',
                'can you', 'should I', 'will', 'does', 'is it'
            ]
        }
        
        # Cache for keyword data
        self.keyword_cache = {}
        self.cache_expiry = 24 * 60 * 60  # 24 hours in seconds
        
        # Initialize session
        self.session = None
        
    async def initialize(self):
        """Initialize the session for HTTP requests"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
    async def analyze(self, keyword: str) -> Dict:
        """
        Comprehensive keyword analysis using multiple techniques
        
        Args:
            keyword: The main keyword to analyze
            
        Returns:
            Dict containing complete keyword analysis
        """
        try:
            await self.initialize()
            
            # Check cache first
            cache_key = self._get_cache_key(keyword)
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                logger.info(f"Using cached SEO data for: {keyword}")
                return cached_data
            
            logger.info(f"Performing comprehensive SEO analysis for: {keyword}")
            
            # Perform analysis in parallel
            tasks = [
                self._get_autocomplete_suggestions(keyword),
                self._analyze_serp(keyword),
                self._classify_intent(keyword),
                self._generate_variations(keyword),
                self._generate_questions(keyword),
                self._generate_lsi_keywords(keyword),
                self._estimate_metrics(keyword)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Combine results
            autocomplete_suggestions, serp_data, intent, variations, questions, lsi_keywords, metrics = results
            
            # Combine all keyword variations
            all_variations = list(set(variations + autocomplete_suggestions))
            
            # Create final analysis
            analysis = {
                'keyword': keyword,
                'variations': all_variations,
                'lsi_keywords': lsi_keywords,
                'questions': questions,
                'intent': intent,
                'serp_features': serp_data.get('serp_features', {}),
                'competitors': serp_data.get('competitors', []),
                'metrics': metrics,
                'long_tail': self._generate_long_tail(keyword, all_variations),
                'keyword_clusters': self._cluster_keywords(all_variations + lsi_keywords)
            }
            
            # Cache the results
            self._add_to_cache(cache_key, analysis)
            
            logger.info(f"Completed SEO analysis for: {keyword} with {len(all_variations)} variations")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in SEO analysis: {e}")
            # Return basic fallback data
            return {
                'keyword': keyword,
                'variations': self._basic_variations(keyword),
                'metrics': {
                    'difficulty': 0.5,
                    'volume': 'medium',
                    'competition': 'moderate'
                }
            }
    
    async def _get_autocomplete_suggestions(self, keyword: str) -> List[str]:
        """Get keyword suggestions from search engine autocomplete"""
        suggestions = set()
        try:
            # Try multiple search engines
            for name, endpoint in self.autocomplete_endpoints.items():
                try:
                    url = endpoint.format(quote_plus(keyword))
                    headers = {'User-Agent': random.choice(self.user_agents)}

                    # Ensure session is initialized
                    if not self.session:
                        await self.initialize()

                    async with self.session.get(url, headers=headers, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json(content_type=None)
                            
                            # Different engines return different formats
                            if name == 'google' or name == 'youtube':
                                if isinstance(data, list) and len(data) > 1:
                                    suggestions.update(data[1])
                            elif name == 'bing':
                                if isinstance(data, list) and len(data) > 1:
                                    suggestions.update(data[1])
                    
                    # Add a small delay to avoid rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Error getting autocomplete from {name}: {e}")
            
            # Clean suggestions
            clean_suggestions = []
            for suggestion in suggestions:
                if suggestion and isinstance(suggestion, str) and suggestion != keyword:
                    clean_suggestions.append(suggestion)
            
            logger.info(f"Found {len(clean_suggestions)} autocomplete suggestions for: {keyword}")
            return clean_suggestions
            
        except Exception as e:
            logger.error(f"Error in autocomplete suggestions: {e}")
            return []
    
    async def _analyze_serp(self, keyword: str) -> Dict:
        """Analyze search engine results page for the keyword"""
        try:
            # Simulate SERP analysis without actually scraping Google
            # In a real implementation, you would carefully scrape the SERP
            
            serp_data = {
                'serp_features': self._predict_serp_features(keyword),
                'competitors': [],
                'top_domains': []
            }
            
            return serp_data
            
        except Exception as e:
            logger.error(f"Error in SERP analysis: {e}")
            return {'serp_features': {}, 'competitors': []}
    
    def _predict_serp_features(self, keyword: str) -> Dict:
        """Predict which SERP features might appear for this keyword"""
        features = {
            'featured_snippet': False,
            'knowledge_panel': False,
            'local_pack': False,
            'image_pack': False,
            'video_results': False,
            'news_results': False,
            'shopping_results': False,
            'people_also_ask': False
        }
        
        # Simple heuristics to predict SERP features
        keyword_lower = keyword.lower()
        
        # Featured snippet likely for how/what/why questions
        if any(keyword_lower.startswith(q) for q in ['how ', 'what ', 'why ']):
            features['featured_snippet'] = True
            features['people_also_ask'] = True
        
        # Knowledge panel for entities (brands, people, places)
        if any(re.search(r'\b' + entity + r'\b', keyword_lower) for entity in ['apple', 'google', 'amazon', 'microsoft']):
            features['knowledge_panel'] = True
        
        # Local pack for local intent
        if any(term in keyword_lower for term in ['near me', 'in city', 'location', 'nearby']):
            features['local_pack'] = True
        
        # Image pack for visual searches
        if any(term in keyword_lower for term in ['images', 'pictures', 'photos', 'how to', 'diy']):
            features['image_pack'] = True
        
        # Video results for how-to and entertainment
        if any(term in keyword_lower for term in ['video', 'youtube', 'watch', 'how to']):
            features['video_results'] = True
        
        # News for current events
        if any(term in keyword_lower for term in ['news', 'latest', 'update', 'today', 'breaking']):
            features['news_results'] = True
        
        # Shopping for commercial intent
        if any(term in keyword_lower for term in ['buy', 'price', 'cheap', 'best', 'review']):
            features['shopping_results'] = True
        
        return features
    
    def _classify_intent(self, keyword: str) -> str:
        """Classify search intent of the keyword"""
        keyword_lower = keyword.lower()
        
        # Check each intent pattern
        intent_scores = {intent: 0 for intent in self.intent_patterns.keys()}
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in keyword_lower:
                    intent_scores[intent] += 1
        
        # Get the intent with the highest score
        if max(intent_scores.values()) > 0:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Default to informational if no patterns match
        return 'informational'
    
    async def _generate_variations(self, keyword: str) -> List[str]:
        """Generate keyword variations using multiple techniques"""
        variations = [keyword]
        
        # Add common modifiers
        for category, modifiers in self.keyword_modifiers.items():
            for modifier in modifiers:
                # Add prefix variations
                if not keyword.lower().startswith(modifier.lower()):
                    variations.append(f"{modifier} {keyword}")
                
                # Add suffix variations
                if not keyword.lower().endswith(modifier.lower()):
                    variations.append(f"{keyword} {modifier}")
        
        # Add year variations
        current_year = datetime.now().year
        variations.append(f"{keyword} {current_year}")
        variations.append(f"{keyword} {current_year + 1}")
        
        # Add "vs" variations for commercial intent
        if self._classify_intent(keyword) == 'commercial':
            competitors = ['alternative', 'vs competition', 'competitors', 'options']
            for competitor in competitors:
                variations.append(f"{keyword} {competitor}")
        
        # Clean and deduplicate
        clean_variations = []
        seen = set()
        for var in variations:
            var_lower = var.lower()
            if var_lower not in seen and var_lower != keyword.lower():
                clean_variations.append(var)
                seen.add(var_lower)
        
        return clean_variations
    
    async def _generate_questions(self, keyword: str) -> List[str]:
        """Generate question-based keywords"""
        questions = []
        
        # Basic question patterns
        question_patterns = [
            "what is {keyword}",
            "how does {keyword} work",
            "why is {keyword} important",
            "when should I use {keyword}",
            "where can I find {keyword}",
            "who needs {keyword}",
            "which {keyword} is best",
            "are {keyword} worth it",
            "can {keyword} be used for",
            "how to choose {keyword}",
            "what are the benefits of {keyword}",
            "how much does {keyword} cost",
            "is {keyword} better than",
            "what are the types of {keyword}",
            "how to install {keyword}",
            "how to fix {keyword} problems",
            "what are common {keyword} issues",
            "how to optimize {keyword}",
            "what are the alternatives to {keyword}",
            "how to compare {keyword} options"
        ]
        
        # Generate questions
        for pattern in question_patterns:
            questions.append(pattern.format(keyword=keyword))
        
        # Add more specific questions based on intent
        intent = self._classify_intent(keyword)
        
        if intent == 'commercial':
            commercial_questions = [
                "what is the best {keyword} for beginners",
                "which {keyword} offers the best value",
                "what is the most reliable {keyword}",
                "how to find discounts on {keyword}",
                "is premium {keyword} worth the cost"
            ]
            for q in commercial_questions:
                questions.append(q.format(keyword=keyword))
        
        elif intent == 'informational':
            info_questions = [
                "what are the latest trends in {keyword}",
                "how has {keyword} evolved",
                "what is the future of {keyword}",
                "how to learn more about {keyword}",
                "what are expert opinions on {keyword}"
            ]
            for q in info_questions:
                questions.append(q.format(keyword=keyword))
        
        return questions
    
    async def _generate_lsi_keywords(self, keyword: str) -> List[str]:
        """Generate LSI (Latent Semantic Indexing) keywords"""
        lsi_keywords = []
        
        # Extract main terms from keyword
        main_terms = keyword.lower().split()
        
        # Add synonyms and related terms for each main term
        for term in main_terms:
            if len(term) > 3:  # Only process meaningful words
                # Add common variations
                lsi_keywords.append(f"{term}s")  # Plural
                lsi_keywords.append(term.rstrip('s'))  # Singular
                
                # Add common prefixes
                lsi_keywords.append(f"best {term}")
                lsi_keywords.append(f"top {term}")
                lsi_keywords.append(f"popular {term}")
                
                # Add common suffixes
                lsi_keywords.append(f"{term} guide")
                lsi_keywords.append(f"{term} tutorial")
                lsi_keywords.append(f"{term} tips")
        
        # Add industry-specific terms
        industry_terms = [
            'software', 'app', 'tool', 'platform', 'service',
            'online', 'digital', 'automated', 'smart', 'ai',
            'professional', 'enterprise', 'business', 'personal'
        ]
        
        for term in industry_terms:
            lsi_keywords.append(f"{keyword} {term}")
        
        # Clean and deduplicate
        clean_lsi = []
        seen = set()
        for kw in lsi_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen and kw_lower != keyword.lower():
                clean_lsi.append(kw)
                seen.add(kw_lower)
        
        return clean_lsi
    
    def _generate_long_tail(self, keyword: str, variations: List[str]) -> List[str]:
        """Generate long-tail keyword variations"""
        long_tail = []
        
        # Add more specific variations
        specifics = [
            'for beginners', 'for professionals', 'step by step', 
            'in 2024', 'with examples', 'free', 'premium',
            'open source', 'alternatives', 'vs competitors'
        ]
        
        for specific in specifics:
            long_tail.append(f"{keyword} {specific}")
        
        # Add question-based long-tail
        questions = [
            f"how to get started with {keyword}",
            f"what are the benefits of {keyword}",
            f"how to choose the right {keyword}",
            f"common mistakes to avoid with {keyword}",
            f"is {keyword} worth the investment",
            f"how to optimize {keyword} for better results"
        ]
        long_tail.extend(questions)
        
        # Add intent-specific long-tail
        intent = self._classify_intent(keyword)
        
        if intent == 'commercial':
            commercial_long_tail = [
                f"best {keyword} for small business",
                f"enterprise-grade {keyword} solutions",
                f"{keyword} pricing comparison",
                f"affordable {keyword} alternatives",
                f"{keyword} free trial"
            ]
            long_tail.extend(commercial_long_tail)
        
        elif intent == 'informational':
            info_long_tail = [
                f"{keyword} complete guide 2024",
                f"understanding {keyword} fundamentals",
                f"{keyword} for beginners pdf",
                f"learn {keyword} online course",
                f"{keyword} certification"
            ]
            long_tail.extend(info_long_tail)
        
        # Clean and deduplicate
        clean_long_tail = []
        seen = set()
        for kw in long_tail:
            kw_lower = kw.lower()
            if kw_lower not in seen and kw_lower != keyword.lower():
                clean_long_tail.append(kw)
                seen.add(kw_lower)
        
        return clean_long_tail
    
    def _cluster_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Cluster keywords into related groups"""
        clusters = {}
        
        # Simple clustering based on common words
        for keyword in keywords:
            words = set(keyword.lower().split())
            
            # Find the most relevant cluster
            best_cluster = None
            best_overlap = 0
            
            for cluster_name, cluster_keywords in clusters.items():
                cluster_words = set(cluster_name.lower().split())
                overlap = len(words.intersection(cluster_words))
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cluster = cluster_name
            
            # Add to existing cluster or create new one
            if best_overlap >= 1:  # At least one word in common
                clusters[best_cluster].append(keyword)
            else:
                clusters[keyword] = [keyword]
        
        return clusters
    
    async def _estimate_metrics(self, keyword: str) -> Dict:
        """Estimate keyword metrics"""
        # Estimate difficulty based on keyword length and complexity
        words = keyword.split()
        word_count = len(words)
        
        # Short keywords are typically more competitive
        if word_count <= 2:
            difficulty = 0.7
            volume = 'high'
            competition = 'high'
        elif word_count <= 4:
            difficulty = 0.5
            volume = 'medium'
            competition = 'moderate'
        else:
            difficulty = 0.3
            volume = 'low'
            competition = 'low'
        
        # Adjust for commercial intent
        intent = self._classify_intent(keyword)
        if intent == 'transactional' or intent == 'commercial':
            difficulty += 0.1
            competition = 'high' if competition != 'high' else competition
        
        # Adjust for trending topics
        if any(trend in keyword.lower() for trend in ['ai', 'crypto', 'nft', '2024', 'covid']):
            volume = 'high'
        
        # Ensure difficulty is within bounds
        difficulty = max(0.1, min(0.9, difficulty))
        
        return {
            'difficulty': difficulty,
            'volume': volume,
            'competition': competition,
            'keyword_density': 1.0,
            'readability': 70
        }
    
    def _basic_variations(self, keyword: str) -> List[str]:
        """Generate basic keyword variations as fallback"""
        return [
            f"best {keyword}",
            f"how to use {keyword}",
            f"{keyword} guide",
            f"{keyword} tutorial",
            f"what is {keyword}"
        ]
    
    def _get_cache_key(self, keyword: str) -> str:
        """Generate a cache key for the keyword"""
        return hashlib.md5(keyword.lower().encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get data from cache if available and not expired"""
        if cache_key in self.keyword_cache:
            timestamp, data = self.keyword_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return data
        return None
    
    def _add_to_cache(self, cache_key: str, data: Dict) -> None:
        """Add data to cache with current timestamp"""
        self.keyword_cache[cache_key] = (time.time(), data)
        
        # Clean up cache if it gets too large
        if len(self.keyword_cache) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(self.keyword_cache.items(), key=lambda x: x[1][0])
            self.keyword_cache = dict(sorted_cache[-500:])  # Keep newest 500 entries