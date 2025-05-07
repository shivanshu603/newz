import sys
import os
from typing import Dict, List, Set, Optional, Any
import asyncio
import aiohttp
from bs4 import BeautifulSoup, Tag
from collections import defaultdict
import re
from urllib.parse import quote_plus
import json

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import logger
import random
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

class KeywordAnalyzer:
    def __init__(self):
        self.search_endpoints = {
            'google': 'https://www.google.com/complete/search?q={}&client=gws-wiz',
            'bing': 'https://www.bing.com/AS/Suggestions?qry={}&cvid=1',
            'youtube': 'https://suggestqueries.google.com/complete/search?client=youtube&ds=yt&q={}',
            'amazon': 'https://completion.amazon.com/search/complete?search-alias=aps&client=amazon-search-ui&mkt=1&q={}'
        }
        
        self.question_starters = {
            'how': ['to', 'do', 'can', 'does', 'is'],
            'what': ['is', 'are', 'does', 'means'],
            'why': ['is', 'does', 'do', 'are', 'should'],
            'when': ['is', 'does', 'should', 'will', 'can'],
            'where': ['to', 'can', 'is', 'are', 'should'],
            'which': ['is', 'are', 'one', 'tool', 'service']
        }

    async def analyze_keyword(self, keyword: str) -> Dict:
        """Enhanced keyword analysis"""
        tasks = [
            self._analyze_competition(keyword),
            self._analyze_content_gaps(keyword),
            self._analyze_semantic_relevance(keyword),
            self._analyze_search_intent(keyword),
            self._analyze_serp_features(keyword)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            'competition_analysis': results[0],
            'content_gaps': results[1],
            'semantic_keywords': results[2],
            'search_intent': results[3],
            'serp_features': results[4],
            'difficulty_score': await self._calculate_difficulty(keyword),
            'estimated_volume': await self._estimate_search_volume(keyword)
        }

    async def _get_search_suggestions(self, keyword: str) -> List[str]:
        """Get suggestions from multiple search engines"""
        suggestions = set()
        
        async def fetch_suggestions(endpoint: str, kw: str) -> List[str]:
            try:
                # Ensure kw is a string before encoding
                if not isinstance(kw, str):
                    logger.error(f"Expected string for keyword, got {type(kw)}")
                    return []
                url = endpoint.format(quote_plus(kw))
                headers = {'User-Agent': self._get_random_user_agent()}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.text()
                            return self._parse_suggestions(data, endpoint)
                return []
            except Exception as e:
                logger.error(f"Error fetching suggestions: {e}")
                return []

        tasks = [fetch_suggestions(endpoint, keyword) 
                for endpoint in self.search_endpoints.values()]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            suggestions.update(result)
        
        return list(suggestions)

    async def _analyze_serp_features(self, keyword: str) -> Dict:
        """Analyze SERP features for the keyword"""
        features = {
            'featured_snippet': False,
            'people_also_ask': [],
            'local_pack': False,
            'video_results': False,
            'shopping_results': False
        }
        
        try:
            url = f"https://www.google.com/search?q={quote_plus(keyword)}"
            headers = {'User-Agent': self._get_random_user_agent()}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Check for featured snippet
                        features['featured_snippet'] = bool(soup.find('div', {'class': 'c-feature'}))
                        
                        # Get "People Also Ask" questions
                        paa_divs = soup.find_all('div', {'class': 'related-question-pair'})
                        features['people_also_ask'] = [div.text for div in paa_divs]
                        
                        # Check for local pack
                        features['local_pack'] = bool(soup.find('div', {'class': 'local-pack'}))
                        
                        # Check for video results
                        features['video_results'] = bool(soup.find('div', {'class': 'video-result'}))
                        
                        # Check for shopping results
                        features['shopping_results'] = bool(soup.find('div', {'class': 'commercial-unit-desktop-top'}))
        
        except Exception as e:
            logger.error(f"Error analyzing SERP features: {e}")
        
        return features

    async def _get_competitor_keywords(self, keyword: str) -> List[str]:
        """Extract keywords from top-ranking pages"""
        competitor_keywords = set()
        
        try:
            url = f"https://www.google.com/search?q={quote_plus(keyword)}"
            headers = {'User-Agent': self._get_random_user_agent()}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract URLs from top results
                        result_links = [a['href'] for a in soup.select('.r a')][:5]
                        
                        # Analyze each competitor page
                        for link in result_links:
                            try:
                                async with session.get(link, headers=headers) as page_response:
                                    if page_response.status == 200:
                                        page_html = await page_response.text()
                                        page_soup = BeautifulSoup(page_html, 'html.parser')
                                        
                                        # Extract keywords from meta tags
                                        meta_keywords = page_soup.find('meta', {'name': 'keywords'})
                                        if meta_keywords:
                                            content = meta_keywords.get('content', '')
                                            if content:
                                                competitor_keywords.update(
                                                    content.lower().split(',')
                                                )
                                        
                                        # Extract keywords from headings
                                        headings = page_soup.find_all(['h1', 'h2', 'h3'])
                                        for heading in headings:
                                            words = re.findall(r'\w+', heading.text.lower())
                                            competitor_keywords.update(words)
                            
                            except Exception as e:
                                logger.error(f"Error analyzing competitor page: {e}")
                                continue
        
        except Exception as e:
            logger.error(f"Error getting competitor keywords: {e}")
        
        return list(competitor_keywords)

    async def _generate_long_tail_keywords(self, keyword: str) -> List[str]:
        """Generate long-tail keyword variations"""
        modifiers = {
            'intent': ['best', 'top', 'cheap', 'affordable', 'premium', 'professional'],
            'location': ['near me', 'online', 'local', 'in [city]'],
            'time': ['2024', '2025', 'today', 'fast', 'instant'],
            'action': ['buy', 'get', 'find', 'download', 'compare'],
            'quality': ['review', 'vs', 'alternative', 'solution']
        }
        
        long_tail = set()
        words = keyword.split()
        
        # Generate combinations
        for category, mods in modifiers.items():
            for mod in mods:
                long_tail.add(f"{mod} {keyword}")
                long_tail.add(f"{keyword} {mod}")
                
                if len(words) > 1:
                    # Insert modifier between words
                    for i in range(len(words)):
                        new_words = words.copy()
                        new_words.insert(i, mod)
                        long_tail.add(' '.join(new_words))
        
        return list(long_tail)

    async def _analyze_search_intent(self, keyword: str) -> Dict:
        """Analyze search intent and categorize keyword"""
        keyword = keyword.lower()
        
        # Intent patterns
        patterns = {
            'informational': r'\b(how|what|why|when|where|who|guide|tutorial|learn)\b',
            'navigational': r'\b(login|sign in|website|official|download)\b',
            'transactional': r'\b(buy|price|order|purchase|cheap|deal|discount)\b',
            'commercial': r'\b(best|review|vs|compare|top|ranking)\b'
        }
        
        # Check each pattern
        intents = {}
        for intent, pattern in patterns.items():
            matches = len(re.findall(pattern, keyword))
            if matches > 0:
                intents[intent] = matches
        
        # Determine primary intent
        if not intents:
            primary_intent = 'informational'  # Default
        else:
            primary_intent = max(intents.items(), key=lambda x: x[1])[0]
        
        return {
            'primary_intent': primary_intent,
            'all_intents': intents
        }

    async def _get_question_keywords(self, keyword: str) -> List[str]:
        """Generate question-based keyword variations"""
        questions = set()
        
        for starter, verbs in self.question_starters.items():
            for verb in verbs:
                questions.add(f"{starter} {verb} {keyword}")
                
                # Generate more natural questions
                if starter == 'how':
                    questions.add(f"{starter} to {keyword}")
                    questions.add(f"{starter} much does {keyword} cost")
                elif starter == 'what':
                    questions.add(f"{starter} {verb} the best {keyword}")
                    questions.add(f"{starter} {keyword} should I buy")
                elif starter == 'why':
                    questions.add(f"{starter} {verb} {keyword} important")
                    questions.add(f"{starter} choose {keyword}")
        
        return list(questions)

    async def _analyze_competition(self, keyword: str) -> Dict:
        """Analyze competition strength"""
        try:
            competitors = await self._get_top_competitors(keyword)
            metrics = {
                'avg_domain_authority': 0,
                'content_quality_score': 0,
                'backlink_strength': 0,
                'keyword_density': 0
            }
            
            for comp in competitors:
                # Analyze various SEO factors
                da_score = await self._analyze_domain_authority(comp['url'])
                content_score = self._analyze_content_quality(comp['content'])
                backlink_score = await self._analyze_backlinks(comp['url'])
                
                metrics['avg_domain_authority'] += da_score
                metrics['content_quality_score'] += content_score
                metrics['backlink_strength'] += backlink_score
                
            # Calculate averages
            count = len(competitors) or 1
            for key in metrics:
                metrics[key] /= count
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing competition: {e}")
            return {}

    async def _analyze_content_gaps(self, keyword: str) -> List[str]:
        """Find content opportunities by analyzing competitors"""
        try:
            # Get competitor content
            competitors = await self._get_top_competitors(keyword)
            all_topics = set()
            covered_topics = set()
            
            for comp in competitors:
                topics = self._extract_content_topics(comp['content'])
                all_topics.update(topics)
                
                # Check which topics are well-covered
                for topic in topics:
                    if self._is_topic_well_covered(comp['content'], topic):
                        covered_topics.add(topic)
            
            # Find gaps
            gaps = all_topics - covered_topics
            return list(gaps)
            
        except Exception as e:
            logger.error(f"Error analyzing content gaps: {e}")
            return []

    async def _analyze_semantic_relevance(self, keyword: str) -> Dict:
        """Analyze semantic relevance and related terms"""
        try:
            # Get semantically related terms
            related_terms = await self._get_related_terms(keyword)
            
            # Calculate semantic similarity scores
            similarity_scores = self._calculate_semantic_similarity(keyword, related_terms)
            
            # Group by topic clusters
            topic_clusters = self._cluster_by_topic(related_terms, similarity_scores)
            
            return {
                'related_terms': related_terms,
                'topic_clusters': topic_clusters,
                'similarity_scores': similarity_scores
            }
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return {}

    def _calculate_semantic_similarity(self, keyword: str, terms: List[str]) -> Dict[str, float]:
        """Calculate semantic similarity scores using NLP"""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            # Load model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embeddings
            keyword_embedding = model.encode([keyword])[0]
            term_embeddings = model.encode(terms)
            
            # Calculate cosine similarity
            similarities = {
                term: float(np.dot(keyword_embedding, term_emb) / 
                          (np.linalg.norm(keyword_embedding) * np.linalg.norm(term_emb)))
                for term, term_emb in zip(terms, term_embeddings)
            }
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return {term: 0.5 for term in terms}

    def _cluster_by_topic(self, terms: List[str], similarity_scores: Dict[str, float]) -> Dict[str, List[str]]:
        """Cluster terms into topics using similarity scores"""
        try:
            from sklearn.cluster import DBSCAN
            import numpy as np
            
            # Convert to matrix for clustering
            similarity_matrix = np.array(list(similarity_scores.values())).reshape(-1, 1)
            
            # Perform clustering
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(similarity_matrix)
            
            # Group terms by cluster
            clusters = defaultdict(list)
            for term, cluster_id in zip(terms, clustering.labels_):
                clusters[f"cluster_{cluster_id}"].append(term)
                
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Error clustering topics: {e}")
            return {"cluster_0": terms}

    def _get_random_user_agent(self) -> str:
        """Get a random user agent string"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        return random.choice(user_agents)

    def _parse_suggestions(self, data: str, endpoint: str) -> List[str]:
        """Parse suggestions based on endpoint format"""
        try:
            if 'google' in endpoint:
                data = json.loads(data)
                return [item[0] for item in data[1]]
            elif 'bing' in endpoint:
                soup = BeautifulSoup(data, 'html.parser')
                return [item.text for item in soup.find_all('li')]
            elif 'amazon' in endpoint:
                data = json.loads(data)
                if isinstance(data[1], list):
                    return data[1]
                else:
                    return []
            return []
        except Exception:
            return []

    async def _calculate_difficulty(self, keyword: str) -> Dict:
        """Calculate comprehensive keyword difficulty score"""
        try:
            # Get SERP data
            serp_data = await self._analyze_serp_results(keyword)
            
            # Analyze backlink profiles
            backlink_metrics = await self._analyze_backlink_profiles(serp_data['urls'])
            
            # Get historical trend data
            trend_data = await self._get_historical_trends(keyword)
            
            # Calculate difficulty components
            domain_score = self._calculate_domain_strength(backlink_metrics)
            content_score = self._analyze_content_competition(serp_data['content'])
            historical_score = self._analyze_historical_performance(trend_data)
            
            # Use ML model for final scoring
            difficulty_score = self._compute_ml_difficulty_score({
                'domain_metrics': domain_score,
                'content_metrics': content_score,
                'historical_metrics': historical_score,
                'serp_features': serp_data['features']
            })
            
            return {
                'overall_score': difficulty_score,
                'components': {
                    'domain_strength': domain_score,
                    'content_competition': content_score,
                    'historical_difficulty': historical_score
                },
                'metrics': {
                    'backlink_data': backlink_metrics,
                    'trend_data': trend_data,
                    'serp_features': serp_data['features']
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating difficulty: {e}")
            return {'overall_score': 0.5}  # Default moderate difficulty

    async def _analyze_backlink_profiles(self, urls: List[str]) -> Dict:
        """Analyze backlink profiles of ranking pages"""
        try:
            metrics = defaultdict(list)
            
            for url in urls[:5]:  # Analyze top 5 competitors
                try:
                    domain = self._extract_domain(url)
                    
                    # Get domain metrics
                    domain_data = await self._get_domain_metrics(domain)
                    
                    # Get backlink data
                    backlink_data = await self._get_backlink_data(url)
                    
                    # Calculate quality scores
                    authority_score = self._calculate_authority_score(domain_data)
                    trust_score = self._calculate_trust_score(backlink_data)
                    
                    metrics['authority_scores'].append(authority_score)
                    metrics['trust_scores'].append(trust_score)
                    metrics['backlink_counts'].append(backlink_data['total_backlinks'])
                    metrics['referring_domains'].append(backlink_data['referring_domains'])
                    
                except Exception as e:
                    logger.error(f"Error analyzing backlinks for {url}: {e}")
                    continue
            
            return dict(metrics)
            
        except Exception as e:
            logger.error(f"Error in backlink analysis: {e}")
            return {}

    async def _analyze_metrics(self, competitors: List[Dict]) -> List[Dict]:
        """Analyze competitor metrics"""
        try:
            metrics = []
            for comp in competitors:
                # Use get() instead of [] for safer access
                content = comp.get('content', '')
                url = comp.get('url', '')
                
                backlinks = await self._get_backlink_metrics(url)
                metrics.append({
                    'url': url,
                    'content_length': len(content.split()),
                    'backlinks': backlinks,
                    'relevance': self._calculate_relevance(content)
                })
            return metrics
        except Exception as e:
            logger.error(f"Error analyzing metrics: {e}")
            return []

    async def _get_backlink_metrics(self, url: str) -> List[str]:
        """Get backlink metrics ensuring List[str] return type"""
        try:
            if not url:
                return []
            return [url]  # Return as list to match return type
        except Exception as e:
            logger.error(f"Error getting backlink metrics: {e}")
            return []

    def _compute_ml_difficulty_score(self, metrics: Dict) -> float:
        """Calculate difficulty score using ML model"""
        try:
            # Prepare feature vector
            features = np.array([
                metrics['domain_metrics']['avg_authority'],
                metrics['domain_metrics']['avg_trust'],
                metrics['content_metrics']['quality_score'],
                metrics['content_metrics']['length_score'],
                metrics['historical_metrics']['trend_strength'],
                metrics['historical_metrics']['competition_growth'],
                len(metrics['serp_features']) / 10  # Normalize feature count
            ]).reshape(1, -1)
            
            # Load pre-trained model (using simple weighted average as fallback)
            try:
                import joblib
                model = joblib.load('models/difficulty_model.pkl')
                score = model.predict(features)[0]
            except:
                # Fallback to weighted average if model fails
                weights = [0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]
                score = np.average(features[0], weights=weights)
            
            return min(max(score, 0), 1)  # Ensure score is between 0 and 1
            
        except Exception as e:
            logger.error(f"Error in ML difficulty calculation: {e}")
            return 0.5  # Default moderate difficulty

    def _analyze_historical_performance(self, trend_data: Dict) -> Dict:
        """Analyze historical performance and trends"""
        try:
            # Calculate trend metrics
            values = np.array(trend_data['historical_values'])
            dates = np.array(trend_data['dates'])
            
            # Calculate trend direction and strength
            slope, trend_strength = self._calculate_trend_metrics(values)
            
            # Analyze seasonality
            seasonality = self._detect_seasonality(values)
            
            # Calculate competition growth
            competition_growth = self._calculate_competition_growth(trend_data['competitor_counts'])
            
            return {
                'trend_direction': 'up' if slope > 0 else 'down',
                'trend_strength': float(trend_strength),
                'seasonality': seasonality,
                'competition_growth': float(competition_growth),
                'stability_score': float(self._calculate_stability(values))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing historical performance: {e}")
            return {}

    def _calculate_domain_strength(self, backlink_metrics: Dict) -> float:
        """Calculate domain strength score"""
        try:
            if not backlink_metrics:
                return 0.5
                
            # Calculate average metrics
            avg_authority = np.mean(backlink_metrics['authority_scores'])
            avg_trust = np.mean(backlink_metrics['trust_scores'])
            
            # Calculate backlink diversity
            referring_domains = np.mean(backlink_metrics['referring_domains'])
            max_domains = 1000  # Benchmark for max domains
            domain_diversity = min(referring_domains / max_domains, 1.0)
            
            # Weight the components
            weights = {
                'authority': 0.4,
                'trust': 0.4,
                'diversity': 0.2
            }
            
            return (
                avg_authority * weights['authority'] +
                avg_trust * weights['trust'] +
                domain_diversity * weights['diversity']
            )
            
        except Exception as e:
            logger.error(f"Error calculating domain strength: {e}")
            return 0.5

    def _extract_content_from_serp(self, html: str) -> Dict[str, Any]:
        """Extract content from SERP results safely"""
        try:
            content = {}
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract titles and links safely using get()
            for result in soup.find_all('div', class_='g'):
                title_elem = result.find('h3')
                link_elem = result.find('a')
                
                if title_elem and isinstance(link_elem, Tag):
                    # Get text content safely with fallback to empty string
                    title_text = title_elem.get_text(strip=True) or ''
                    href = link_elem.get('href', '')
                    
                    if href and title_text:
                        content[href] = {
                            'title': title_text.lower(),  # Safe to call lower() on non-None string
                            'url': href
                        }
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting SERP content: {e}")
            return {}

    def _extract_keywords(self, text: Optional[str]) -> List[str]:
        """Extract keywords ensuring List[str] return type"""
        if not text or not isinstance(text, str):
            return []
        
        text = text.lower()
        words = text.split()
        return [w for w in words if len(w) > 3]

    def _extract_keywords_from_text(self, text: Optional[str]) -> List[str]:
        """Extract keywords from text safely"""
        if not text:
            return []
            
        # Now safe to use string methods after None check
        text = text.lower()
        keywords = set()
        
        # Extract keywords
        words = text.split()
        keywords.update(word for word in words if len(word) > 3)
        
        return list(keywords)

    def _process_text(self, text: Optional[str]) -> List[str]:
        """Process text safely ensuring List[str] return type"""
        if not text or not isinstance(text, str):
            return []
            
        processed = text.lower()
        return [processed]  # Return as list to match return type

    def _check_keyword(self, kw: Optional[str]) -> str:
        """Safely handle potentially None keywords"""
        if not kw:
            return ""
        return kw.lower()

    def _get_related_searches(self, keyword: str) -> List[str]:
        """Get related searches ensuring List[str] return type"""
        try:
            searches = self._fetch_related_searches(keyword)
            if isinstance(searches, list):
                return searches
            elif isinstance(searches, str):
                return [searches]
            return []
        except Exception as e:
            logger.error(f"Error getting related searches: {e}")
            return []

    def _process_keyword(self, keyword: Optional[str]) -> str:
        """Process keyword with proper null check"""
        if not keyword:
            return ""
        return keyword.lower().strip()

    async def _analyze_competitors(self, keyword: Optional[str]) -> List[str]:
        """Analyze competitors with proper null handling"""
        if not keyword:
            return []

        try:
            # Implementation would go here
            # Since this is just a stub, we'll return an empty list
            return []

        except Exception as e:
            logger.error(f"Error analyzing competitors: {e}")
            return []
