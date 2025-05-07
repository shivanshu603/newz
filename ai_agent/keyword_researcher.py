from typing import Dict, List, Optional, Tuple, Any, Set, Union, TypedDict, NamedTuple
from datetime import datetime, timedelta
import aiohttp
import re
import os
import time
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from utils import logger
import asyncio
import random
from urllib.parse import quote_plus, quote, urljoin, urlparse
import numpy as np
import json
import hashlib
from utils.request_limiter import RequestThrottler
from utils.keyword_ranker import KeywordRanker
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN

# Add type definitions
class KeywordGroup(TypedDict):
    """Properly typed keyword group structure"""
    primary: List[str]
    secondary: List[str]
    related: List[str]
    clusters: Dict[str, List[str]]

class KeywordData(TypedDict):
    """Properly typed keyword data structure"""
    keywords: List[str]
    embeddings: np.ndarray
    similarities: np.ndarray

class KeywordResult(TypedDict):
    """Properly typed keyword result structure"""
    primary_keywords: List[str]
    secondary_keywords: List[str]
    questions: List[str]
    semantic_groups: Dict[str, List[str]]
    related_terms: List[str]
    long_tail: List[str]  # Add long_tail field

class KeywordResearcher:
    def __init__(self):
        self.headers_list = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            },
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        ]

        self.search_engines = {
            'bing': {
                'url': 'https://www.bing.com/search',
                'params': {'q': '{query}', 'setlang': 'en'},
                'selectors': {
                    'results': '.b_algo',
                    'title': 'h2',
                    'link': 'a',
                    'snippet': '.b_caption p'
                }
            },
            'yahoo': {
                'url': 'https://search.yahoo.com/search',
                'params': {'p': '{query}', 'ei': 'UTF-8'},
                'selectors': {
                    'results': '#web .algo',
                    'title': 'h3',
                    'link': '.compTitle a',
                    'snippet': '.compText'
                }
            },
            'aol': {
                'url': 'https://search.aol.com/aol/search',
                'params': {'q': '{query}', 'ei': 'UTF-8'},
                'selectors': {
                    'results': '.algo',
                    'title': 'h3',
                    'link': 'a',
                    'snippet': '.compText'
                }
            }
        }

        self.min_delay = 2
        self.max_delay = 5

        self.keyword_ranker = KeywordRanker()

        self.topic_groups = {
            'character': ['who', 'actor', 'cast', 'played by', 'role'],
            'plot': ['story', 'episode', 'season', 'ending', 'happens'],
            'theme': ['about', 'meaning', 'represents', 'symbolism'],
            'facts': ['real', 'based on', 'true', 'history', 'origin']
        }

        self.topic_categories = {
            'show_info': ['episode', 'season', 'series', 'show', 'watch', 'streaming'],
            'characters': ['cast', 'actor', 'character', 'plays', 'role', 'stars'],
            'plot': ['story', 'plot', 'happens', 'ending', 'what happens'],
            'reviews': ['review', 'rating', 'worth', 'good', 'best', 'popular'],
            'platform': ['netflix', 'amazon', 'hulu', 'streaming', 'watch online']
        }

        self.question_patterns = {
            'what': ['what happens', 'what is', 'what about'],
            'how': ['how to', 'how does', 'how many'],
            'when': ['when does', 'when is', 'when will'],
            'where': ['where to', 'where can', 'where is'],
            'why': ['why did', 'why is', 'why does']
        }

        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
            logger.info("NLP models loaded successfully")
            self.models_loaded = True
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
            self.models_loaded = False

        self.custom_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        self.domain_terms = {'episode', 'season', 'series', 'show', 'character', 'cast'}

        # Remove predefined patterns and templates
        self.domain_patterns = {}
        self.entity_types = {}

        # Increase model utilization weights
        self.keyword_weights = {
            'semantic_relevance': 0.45,  # Increased weight for semantic understanding
            'contextual_fit': 0.35,     # New weight for contextual relevance
            'frequency': 0.20           # Reduced weight for raw frequency
        }

        # Dynamic threshold based on semantic similarity
        self.thresholds = {
            'semantic_similarity': 0.65,  # Base similarity threshold
            'cluster_density': 0.4,      # Cluster cohesion threshold
            'min_topic_relevance': 0.3   # Minimum topic relevance score
        }

    async def get_keywords(self, topic: str) -> KeywordResult:
        """Alias for find_keywords to maintain backward compatibility"""
        return await self.find_keywords(topic)

    async def find_keywords(self, topic: str) -> KeywordResult:
        """Enhanced keyword research using deep semantic analysis"""
        # Skip processing if input looks like a phone number
        if re.match(r'^[\d\s\-+()]{7,}$', topic.strip()):
            return self._create_empty_result(topic)
        try:
            # Get initial keyword candidates
            seed_keywords = await self._get_seed_keywords(topic)
            serp_results = await self._get_serp_data(topic)

            # Generate topic embedding once
            topic_embedding = self.sentence_model.encode([topic])[0]

            # Extract and analyze keywords using transformers
            keyword_data = await self._extract_and_analyze_keywords(
                topic, 
                topic_embedding,
                serp_results, 
                seed_keywords
            )

            # Get semantic clusters and related terms
            semantic_groups = self._create_semantic_clusters(
                keyword_data['keywords'],
                keyword_data['embeddings'],
                topic_embedding
            )

            return KeywordResult(
                primary_keywords=semantic_groups['primary'][:10],
                secondary_keywords=semantic_groups['secondary'][:20],
                questions=await self._extract_semantic_questions(serp_results, topic_embedding),
                semantic_groups=semantic_groups['clusters'],
                related_terms=semantic_groups['related'][:20],
                long_tail=[]  # Placeholder for long_tail field
            )

        except Exception as e:
            logger.error(f"Error in keyword research: {e}")
            return self._create_empty_result(topic)

    async def _extract_and_analyze_keywords(self, topic: str, topic_embedding: np.ndarray, 
                                         serp_results: List[Dict], seed_keywords: Set[str]) -> KeywordData:
        """Extract and analyze keywords using transformer models"""
        try:
            # Extract text content from SERP results
            content = self._extract_content_from_serp(serp_results)

            # Use spaCy for initial phrase extraction
            doc = self.nlp(content)
            candidates = set()

            # Extract noun phrases and named entities
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 4:  # Limit phrase length
                    candidates.add(chunk.text.lower().strip())

            for ent in doc.ents:
                if len(ent.text.split()) <= 4:
                    candidates.add(ent.text.lower().strip())

            # Add seed keywords
            candidates.update(seed_keywords)

            # Generate embeddings for all candidates
            candidate_embeddings = self.sentence_model.encode(list(candidates))

            # Calculate semantic similarity with topic
            similarities = np.dot(candidate_embeddings, topic_embedding)

            # Create properly typed KeywordData
            return KeywordData(
                keywords=list(candidates),
                embeddings=candidate_embeddings,
                similarities=similarities
            )

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return KeywordData(
                keywords=[],
                embeddings=np.array([]),
                similarities=np.array([])
            )

    def _create_semantic_clusters(self, keywords: List[str], 
                                embeddings: np.ndarray, 
                                topic_embedding: np.ndarray) -> KeywordGroup:
        """Create semantic clusters using transformer embeddings"""
        try:
            # Calculate similarities with topic
            topic_similarities = np.dot(embeddings, topic_embedding)

            # Create initial groups based on similarity
            primary = []
            secondary = []
            related = []

            # Cluster embeddings using DBSCAN for better natural grouping
            clusters = DBSCAN(
                eps=0.3,
                min_samples=2,
                metric='cosine'
            ).fit(embeddings)

            # Organize clusters
            cluster_groups = defaultdict(list)
            for keyword, similarity, cluster_label in zip(
                keywords, topic_similarities, clusters.labels_
            ):
                # Assign to primary/secondary based on similarity
                if similarity > self.thresholds['semantic_similarity']:
                    primary.append(keyword)
                elif similarity > self.thresholds['min_topic_relevance']:
                    secondary.append(keyword)
                else:
                    related.append(keyword)

                # Add to semantic cluster if not noise (-1)
                if cluster_label != -1:
                    cluster_groups[f"cluster_{cluster_label}"].append(keyword)

            # Calculate cluster themes using centroid keywords
            themed_clusters = {}
            for cluster_name, cluster_keywords in cluster_groups.items():
                cluster_theme = self._determine_cluster_theme(cluster_keywords)
                themed_clusters[cluster_theme] = cluster_keywords

            return KeywordGroup(
                primary=primary,
                secondary=secondary,
                related=related,
                clusters=themed_clusters
            )

        except Exception as e:
            logger.error(f"Error creating semantic clusters: {e}")
            return KeywordGroup(
                primary=[],
                secondary=[],
                related=[],
                clusters={}
            )

    def _determine_cluster_theme(self, keywords: List[str]) -> str:
        """Determine cluster theme using semantic analysis"""
        try:
            # Get embeddings for keywords
            embeddings = self.sentence_model.encode(keywords)
            
            # Calculate centroid
            centroid = np.mean(embeddings, axis=0)
            
            # Find keyword closest to centroid
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            central_keyword = keywords[np.argmin(distances)]
            
            # Clean up theme name
            theme = central_keyword.replace('-', ' ').title()
            return f"Topic: {theme}"

        except Exception as e:
            logger.error(f"Error determining cluster theme: {e}")
            return "General"

    async def _extract_semantic_questions(self, serp_results: List[Dict], 
                                       topic_embedding: np.ndarray) -> List[str]:
        """Extract and rank questions based on semantic relevance"""
        try:
            questions = []
            
            for result in serp_results:
                text = f"{result['title']} {result['snippet']}"
                doc = self.nlp(text)
                
                # Extract sentences that start with question words
                for sent in doc.sents:
                    lower_sent = sent.text.lower()
                    if any(lower_sent.startswith(q) for q in ['what', 'how', 'why', 'when', 'where', 'which', 'who']):
                        questions.append(sent.text.strip())

            if not questions:
                return []

            # Calculate semantic similarity with topic
            question_embeddings = self.sentence_model.encode(questions)
            similarities = np.dot(question_embeddings, topic_embedding)

            # Sort questions by relevance
            ranked_questions = [q for _, q in sorted(
                zip(similarities, questions),
                key=lambda x: x[0],
                reverse=True
            )]

            return ranked_questions

        except Exception as e:
            logger.error(f"Error extracting questions: {e}")
            return []

    def _extract_content_from_serp(self, serp_results: List[Dict]) -> str:
        """Extract and clean content from SERP results"""
        try:
            content_parts = []
            for result in serp_results:
                title = result.get('title', '').strip()
                snippet = result.get('snippet', '').strip()
                if title:
                    content_parts.append(title)
                if snippet:
                    content_parts.append(snippet)
            
            return ' '.join(content_parts)

        except Exception as e:
            logger.error(f"Error extracting SERP content: {e}")
            return ""

    async def _get_serp_data(self, topic: str) -> List[Dict]:
        """Get SERP results from all search engines"""
        try:
            all_results = []
            for engine in self.search_engines:
                results = await self._get_serp_results(topic, engine)
                all_results.extend(results)
            return all_results
        except Exception as e:
            logger.error(f"Error getting SERP data: {e}")
            return []

    async def fetch_serp_data(self, keyword: str) -> Dict:
        """Fetch SERP data from multiple search engines"""
        try:
            results = {
                'organic_results': [],
                'related_searches': set(),
                'questions': set(),
                'featured_snippets': []
            }

            # Fetch from multiple engines concurrently
            tasks = []
            for engine in self.search_engines:
                tasks.append(self._fetch_engine_results(engine, keyword))
            
            engine_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results from each engine
            for result in engine_results:
                if isinstance(result, dict):  # Skip any failed requests
                    results['organic_results'].extend(result.get('organic_results', []))
                    results['related_searches'].update(result.get('related_searches', []))
                    results['questions'].update(result.get('questions', []))
                    if result.get('featured_snippet'):
                        results['featured_snippets'].append(result['featured_snippet'])
            
            # Convert sets to lists for JSON serialization
            results['related_searches'] = list(results['related_searches'])
            results['questions'] = list(results['questions'])
            
            return results

        except Exception as e:
            logger.error(f"Error fetching SERP data: {e}")
            return {
                'organic_results': [],
                'related_searches': [],
                'questions': [],
                'featured_snippets': []
            }

    async def _fetch_engine_results(self, engine: str, keyword: str) -> Dict:
        """Fetch search results from a specific engine"""
        try:
            results = await self._get_serp_results(keyword, engine)
            
            # Extract additional data from results
            related_searches = set()
            questions = set()
            featured_snippet = None

            for result in results:
                # Extract questions from titles and snippets
                text = f"{result['title']} {result['snippet']}"
                questions.update(self._extract_questions(text))

                # Look for featured snippets
                if 'featured' in result.get('type', '').lower():
                    featured_snippet = result

            return {
                'organic_results': results,
                'related_searches': related_searches,
                'questions': questions,
                'featured_snippet': featured_snippet
            }

        except Exception as e:
            logger.error(f"Error fetching results from {engine}: {e}")
            return {}

    def _extract_questions(self, text: str) -> Set[str]:
        """Extract questions from text"""
        try:
            questions = set()
            # Match question patterns
            patterns = [
                r'(?:^|(?<=[.!?])\s+)(what|how|why|when|where|which|who)\s+(?:[^.!?])+[?]',
                r'(?:^|(?<=[.!?])\s+)(can|should|will|does|do)\s+(?:[^.!?])+[?]'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    question = match.group(0).strip()
                    if self._is_valid_question(question):
                        questions.add(question)
            
            return questions

        except Exception as e:
            logger.error(f"Error extracting questions: {e}")
            return set()

    def _create_empty_result(self, topic: str) -> KeywordResult:
        """Create empty result structure with basic variations"""
        try:
            # Create basic variations of the topic
            basic_variations = [
                topic,
                f"best {topic}",
                f"{topic} reviews",
                f"{topic} specs",
                f"{topic} features"
            ]

            return KeywordResult(
                primary_keywords=basic_variations,
                secondary_keywords=[],
                semantic_groups={
                    'characters': [],
                    'locations': [],
                    'organizations': [],
                    'concepts': []
                },
                questions=[
                    f"what is {topic}",
                    f"how does {topic} work",
                    f"why choose {topic}"
                ],
                related_terms=[],
                long_tail=[]  # Placeholder for long_tail field
            )
        except Exception as e:
            logger.error(f"Error creating empty result: {e}")
            return KeywordResult(
                primary_keywords=[topic],
                secondary_keywords=[],
                semantic_groups={},
                questions=[],
                related_terms=[],
                long_tail=[]  # Placeholder for long_tail field
            )

    async def _get_seed_keywords(self, topic: str) -> Set[str]:
        """Get initial seed keywords from various sources"""
        try:
            seeds = set()
            headers = random.choice(self.headers_list)
            
            url = f"https://suggestqueries.google.com/complete/search?output=toolbar&q={quote_plus(topic)}"
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url) as response:
                    text = await response.text()
                    soup = BeautifulSoup(text, 'xml')
                    suggestions = [suggestion.get('data', '') for suggestion in soup.find_all('suggestion')]
                    seeds.update(filter(None, suggestions))
            
            seeds.add(topic)
            seeds.add(f"what is {topic}")
            seeds.add(f"how to {topic}")
            seeds.add(f"best {topic}")
            seeds.add(f"{topic} guide")
            
            return seeds
            
        except Exception as e:
            logger.error(f"Error getting seed keywords: {e}")
            return {topic}

    async def _get_serp_results(self, query: str, engine: str) -> List[Dict]:
        """Get SERP results with proper error handling"""
        try:
            if engine not in self.search_engines:
                logger.error(f"Unknown search engine: {engine}")
                return []

            engine_config = self.search_engines[engine]
            
            params = engine_config['params'].copy()
            params = {k: v.format(query=query) if isinstance(v, str) else v 
                     for k, v in params.items()}

            await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))

            headers = random.choice(self.headers_list)

            url = engine_config['url']
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Error status {response.status} from {engine}")
                        return []

                    html = await response.text()
                    return self._parse_serp_results(html, engine)

        except Exception as e:
            logger.error(f"Error fetching SERP from {engine}: {e}")
            return []

    def _parse_serp_results(self, html: str, engine: str) -> List[Dict]:
        """Parse SERP results based on engine"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []

            if engine == 'bing':
                for result in soup.select('#b_results .b_algo'):
                    title_elem = result.select_one('h2')
                    link_elem = result.select_one('a')
                    snippet_elem = result.select_one('.b_caption p')
                    
                    if title_elem and link_elem and snippet_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'url': link_elem.get('href', ''),
                            'snippet': snippet_elem.get_text(strip=True)
                        })

            elif engine == 'yahoo':
                for result in soup.select('#web .algo'):
                    title_elem = result.select_one('h3')
                    link_elem = result.select_one('.compTitle a')
                    snippet_elem = result.select_one('.compText')
                    
                    if title_elem and link_elem and snippet_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'url': link_elem.get('href', ''),
                            'snippet': snippet_elem.get_text(strip=True)
                        })

            elif engine == 'duckduckgo':
                for result in soup.select('.result'):
                    title_elem = result.select_one('.result__title')
                    link_elem = result.select_one('.result__url')
                    snippet_elem = result.select_one('.result__snippet')
                    
                    if title_elem and link_elem and snippet_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'url': link_elem.get_text(strip=True),
                            'snippet': snippet_elem.get_text(strip=True)
                        })

            return results

        except Exception as e:
            logger.error(f"Error parsing SERP results from {engine}: {e}")
            return []

    def _basic_keyword_processing(self, topic: str, serp_results: List[Dict]) -> KeywordResult:
        """Basic keyword processing without NLP models"""
        try:
            keywords = set()
            questions = set()
            
            # Process SERP results
            for result in serp_results:
                text = f"{result['title']} {result['snippet']}"
                
                # Extract basic keywords
                words = text.lower().split()
                for i in range(len(words)):
                    # Single words
                    if len(words[i]) > 3 and words[i] not in self.custom_stopwords:
                        keywords.add(words[i])
                    
                    # Bigrams
                    if i < len(words) - 1:
                        bigram = f"{words[i]} {words[i+1]}"
                        if len(bigram) > 7:
                            keywords.add(bigram)
                    
                    # Trigrams
                    if i < len(words) - 2:
                        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                        if len(trigram) > 10:
                            keywords.add(trigram)
                
                # Extract questions
                for pattern in [r'(?i)(?:what|how|why|when|where|which|who)\s+[^.!?]+\?']:
                    matches = re.findall(pattern, text)
                    questions.update(matches)
            
            # Separate primary and secondary keywords
            primary = {kw for kw in keywords if topic.lower() in kw.lower()}
            secondary = keywords - primary
            
            return KeywordResult(
                primary_keywords=list(primary)[:15],
                secondary_keywords=list(secondary)[:30],
                semantic_groups={
                    'characters': [],
                    'locations': [],
                    'organizations': [],
                    'concepts': []
                },
                questions=list(questions)[:10],
                related_terms=list(keywords - primary - secondary)[:20],
                long_tail=[]  # Placeholder for long_tail field
            )
            
        except Exception as e:
            logger.error(f"Error in basic keyword processing: {e}")
            return self._create_empty_result(topic)

async def test_keyword_researcher(topic: str):
    """Test the keyword researcher with a specific topic"""
    print(f"\n{'='*80}")
    print(f"TESTING KEYWORD RESEARCHER WITH TOPIC: {topic}")
    print(f"{'='*80}\n")

    researcher = KeywordResearcher()

    print("\nGathering keyword data...")
    try:
        # Get comprehensive keyword data
        keyword_data = await researcher.find_keywords(topic)
        
        # Print categorized results
        print("\nPrimary Keywords:")
        for i, kw in enumerate(keyword_data.get('primary_keywords', [])[:10], 1):
            print(f"  {i}. {kw}")

        print("\nSecondary Keywords:")
        for i, kw in enumerate(keyword_data.get('secondary_keywords', [])[:10], 1):
            print(f"  {i}. {kw}")

        print("\nQuestion Keywords:")
        for i, q in enumerate(keyword_data.get('questions', [])[:5], 1):
            print(f"  {i}. {q}")

        print("\nTopic Groups:")
        for group, keywords in keyword_data.get('semantic_groups', {}).items():
            if keywords:
                print(f"\n  {group.replace('_', ' ').title()}:")
                for i, kw in enumerate(keywords[:5], 1):
                    print(f"    {i}. {kw}")

        print("\nRelated Terms:")
        for i, term in enumerate(keyword_data.get('related_terms', [])[:5], 1):
            print(f"  {i}. {term}")

        # Save results to file
        filename = f"keyword_research_{topic.replace(' ', '_')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(keyword_data, f, indent=2)
        print(f"\nFull results saved to {filename}")

    except Exception as e:
        print(f"✗ Error extracting keywords: {e}")

    print(f"\n{'='*80}")
    print(f"TEST COMPLETED")
    print(f"{'='*80}\n")

if __name__== "__main__":
    import sys
    import asyncio
    import platform

    import logging
    from utils import logger

    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if len(sys.argv) > 1:
        topic = sys.argv[1]
    else:
        topic = input("Enter a topic to research: ")

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(test_keyword_researcher(topic))
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Test failed to complete.")
