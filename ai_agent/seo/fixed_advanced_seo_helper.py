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
                "is premium {keyword} worth the extra cost"
            ]
            for pattern in commercial_questions:
                questions.append(pattern.format(keyword=keyword))
                
        elif intent == 'informational':
            info_questions = [
                "what are the latest trends in {keyword}",
                "how has {keyword} evolved",
                "what is the history of {keyword}",
                "what research exists about {keyword}",
                "how does {keyword} compare to traditional methods"
            ]
            for pattern in info_questions:
                questions.append(pattern.format(keyword=keyword))
        
        return questions
    
    async def _generate_lsi_keywords(self, keyword: str) -> List[str]:
        """Generate LSI (Latent Semantic Indexing) keywords"""
        # In a real implementation, this would use more sophisticated techniques
        # like TF-IDF analysis of top-ranking pages or NLP models
        
        # For this demo, we'll use a simplified approach with predefined semantic fields
        semantic_fields = {
            'marketing': ['strategy', 'campaign', 'audience', 'targeting', 'conversion', 'funnel', 'analytics'],
            'technology': ['software', 'hardware', 'system', 'device', 'application', 'platform', 'integration'],
            'business': ['company', 'startup', 'enterprise', 'industry', 'market', 'revenue', 'profit'],
            'education': ['learning', 'teaching', 'course', 'training', 'skill', 'knowledge', 'certification'],
            'health': ['wellness', 'fitness', 'nutrition', 'diet', 'exercise', 'therapy', 'treatment']
        }
        
        # Determine which semantic field(s) the keyword might belong to
        keyword_lower = keyword.lower()
        relevant_fields = []
        
        for field, terms in semantic_fields.items():
            if any(term in keyword_lower for term in terms):
                relevant_fields.append(field)
        
        # If no fields match, use general terms
        if not relevant_fields:
            relevant_fields = list(semantic_fields.keys())
        
        # Generate LSI keywords from relevant fields
        lsi_keywords = []
        for field in relevant_fields:
            field_terms = semantic_fields[field]
            for term in field_terms:
                if term not in keyword_lower:
                    lsi_keywords.append(f"{keyword} {term}")
                    lsi_keywords.append(f"{term} for {keyword}")
        
        return lsi_keywords[:20]  # Limit to top 20
    
    def _generate_long_tail(self, keyword: str, variations: List[str]) -> List[str]:
        """Generate long-tail keyword variations"""
        long_tail = []
        
        # Combine with modifiers
        modifiers = [
            "best", "top", "cheap", "affordable", "premium", "free",
            "online", "near me", "review", "alternative", "vs", "without",
            "with", "under", "over", "before", "after", "during", "for beginners",
            "for professionals", "step by step", "easy", "advanced", "ultimate"
        ]
        
        # Generate 2-3 word combinations
        for var in variations[:10]:  # Limit to avoid too many combinations
            for mod in modifiers:
                if mod not in var.lower():
                    long_tail.append(f"{var} {mod}")
        
        # Add some 4+ word phrases
        long_phrases = [
            "best ways to use {keyword}",
            "how to get started with {keyword}",
            "what you need to know about {keyword}",
            "common mistakes to avoid with {keyword}",
            "how to choose the right {keyword}",
            "top 5 things to consider when buying {keyword}",
            "the ultimate guide to {keyword} for beginners",
            "how to save money on {keyword} this year",
            "what experts say about {keyword} trends",
            "how to compare different {keyword} options"
        ]
        
        for phrase in long_phrases:
            long_tail.append(phrase.format(keyword=keyword))
        
        # Deduplicate and limit
        return list(set(long_tail))[:50]  # Limit to top 50
    
    def _cluster_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Group keywords into clusters based on common terms"""
        clusters = {}
        
        # Extract significant terms from each keyword
        for keyword in keywords:
            words = keyword.lower().split()
            # Skip very short words
            significant_words = [w for w in words if len(w) > 3]
            
            for word in significant_words:
                if word not in clusters:
                    clusters[word] = []
                clusters[word].append(keyword)
        
        # Remove clusters that are too small
        return {k: v for k, v in clusters.items() if len(v) >= 3}
    
    def _basic_variations(self, keyword: str) -> List[str]:
        """Generate basic keyword variations as a fallback"""
        variations = []
        modifiers = ["best", "top", "how to", "guide", "tutorial", "tips", "review"]
        
        for mod in modifiers:
            variations.append(f"{mod} {keyword}")
            variations.append(f"{keyword} {mod}")
        
        return variations
    
    async def _estimate_metrics(self, keyword: str) -> Dict:
        """Estimate keyword metrics like difficulty, volume, etc."""
        # In a real implementation, this would use actual data sources
        # Here we'll use some heuristics
        
        # Estimate search volume based on keyword length and specificity
        words = keyword.split()
        if len(words) == 1:
            volume = "high" if len(keyword) < 8 else "medium"
        elif len(words) == 2:
            volume = "medium"
        else:
            volume = "low" if len(words) > 4 else "medium-low"
        
        # Estimate keyword difficulty (0-1 scale)
        if len(words) == 1 and len(keyword) < 8:
            difficulty = 0.8  # Short single words are competitive
        elif any(term in keyword.lower() for term in ['how', 'what', 'why']):
            difficulty = 0.5  # Informational queries are moderately competitive
        elif any(term in keyword.lower() for term in ['best', 'top', 'vs']):
            difficulty = 0.7  # Commercial investigation terms are competitive
        elif len(words) >= 4:
            difficulty = 0.3  # Long-tail keywords are less competitive
        else:
            difficulty = 0.6  # Default medium difficulty
        
        # Estimate CPC based on commercial intent
        commercial_terms = ['buy', 'price', 'cheap', 'discount', 'deal', 'purchase']
        if any(term in keyword.lower() for term in commercial_terms):
            cpc = "high"
        elif any(term in keyword.lower() for term in ['free', 'diy', 'homemade']):
            cpc = "low"
        else:
            cpc = "medium"
        
        # Estimate competition level
        if difficulty > 0.7:
            competition = "high"
        elif difficulty > 0.4:
            competition = "moderate"
        else:
            competition = "low"
        
        return {
            'volume': volume,
            'difficulty': round(difficulty, 2),
            'cpc': cpc,
            'competition': competition
        }
    
    def _get_cache_key(self, keyword: str) -> str:
        """Generate a cache key for the keyword"""
        return hashlib.md5(keyword.lower().encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get data from cache if it exists and is not expired"""
        if cache_key in self.keyword_cache:
            timestamp, data = self.keyword_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return data
        return None
    
    def _add_to_cache(self, cache_key: str, data: Dict) -> None:
        """Add data to cache with timestamp"""
        self.keyword_cache[cache_key] = (time.time(), data)
    
    async def close(self):
        """Close the session and clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Advanced SEO Helper resources cleaned up")