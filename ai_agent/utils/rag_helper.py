import os
import sys
import re
import json
# Add parent directory to path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keyword_researcher import KeywordResearcher
from seo_analyzer import SEOAnalyzer
import time
import asyncio
import aiohttp
import random
import feedparser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import logger
import wikipedia
wikipedia.set_lang("en")  # Set default language to English
logger.info("Wikipedia package successfully imported and configured")
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Set, Union, Any, Tuple
from urllib.parse import quote_plus, urlparse
from itertools import combinations
import hashlib
from bs4 import BeautifulSoup
from collections import defaultdict
from transformers.pipelines import pipeline
import torch
import networkx as nx
from utils import logger
from config import Config
from dateutil import parser as date_parser

class CognitiveAnalyzer:
    """Handles advanced cognitive analysis of collected information"""
    def __init__(self):
        try:
            self.analyzer = pipeline('text-generation',
                                  model=Config.DEFAULT_MODEL,
                                  device=0 if torch.cuda.is_available() else -1)
            self.graph = nx.Graph()
            self.verifier = pipeline("text-classification", 
                                  model="google/tapas-base-finetuned-tabfact")
        except Exception as e:
            logger.error(f"Error initializing cognitive analyzer: {e}")
            self.analyzer = None
            self.verifier = None

    async def analyze_context(self, context: Dict) -> Dict:
        """Perform deep cognitive analysis of collected information"""
        if not self.analyzer:
            return context
            
        try:
            # Extract semantic relationships
            context['relationships'] = await self._extract_relationships(context)
            
            # Build temporal understanding
            context['timeline'] = await self._build_timeline(context)
            
            # Perform domain classification
            context['domain'] = await self._classify_domain(context)
            
            # Verify facts
            context['verified_facts'] = await self._verify_facts(context)
            
            # Build knowledge graph
            self._build_knowledge_graph(context)
            
            return context
        except Exception as e:
            logger.error(f"Error in cognitive analysis: {e}")
            return context

    async def _extract_relationships(self, context: Dict) -> List[Dict]:
        """Extract semantic relationships between entities"""
        if not self.analyzer or not context.get('entities'):
            return []
            
        try:
            prompt = f"Extract relationships between these entities: {', '.join(context['entities'])}"
            result = self.analyzer(prompt, max_length=2000)
            relationships = []
            
            if result and isinstance(result, list) and len(result) > 0:
                first_result = result[0] if result else None
                if first_result:
                    rel_text = first_result.get('generated_text', '') if isinstance(first_result, dict) else str(first_result)
                    for rel in rel_text.split(';'):
                        if '->' in rel:
                            parts = rel.split('->')
                            if len(parts) == 2:
                                relationships.append({
                                    'source': parts[0].strip(),
                                    'target': parts[1].strip(),
                                    'type': 'related_to'
                                })
            return relationships
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []

    async def _build_timeline(self, context: Dict) -> List[Dict]:
        """Build temporal understanding of events"""
        timeline = []
        try:
            # Extract dates from text
            dates = []
            for fact in context.get('facts', []):
                try:
                    date = date_parser.parse(fact, fuzzy=True)
                    dates.append({
                        'date': date.isoformat(),
                        'text': fact
                    })
                except:
                    continue
                    
            # Sort by date
            timeline = sorted(dates, key=lambda x: x['date'])
        except Exception as e:
            logger.error(f"Error building timeline: {e}")
            
        return timeline

    async def _classify_domain(self, context: Dict) -> str:
        """Classify the domain of the topic"""
        if not self.analyzer or not context.get('summary'):
            return 'general'
            
        try:
            prompt = f"Classify this text into one domain: {context['summary']}"
            result = self.analyzer(prompt, max_length=50)
            
            if result and isinstance(result, list) and len(result) > 0:
                first_result = result[0] if result else None
                if first_result:
                    return first_result.get('generated_text', 'general').lower() if isinstance(first_result, dict) else str(first_result).lower()
            return 'general'
        except Exception as e:
            logger.error(f"Error classifying domain: {e}")
            return 'general'

    async def _verify_facts(self, context: Dict) -> List[Dict]:
        """Verify facts against sources"""
        if not self.verifier or not context.get('facts'):
            return []
            
        verified = []
        try:
            for fact in context['facts']:
                result = self.verifier(f"Verify: {fact}", 
                                     context['sources'])
                if result and isinstance(result, list) and len(result) > 0:
                    first_result = result[0] if result else None
                    if first_result:
                        verified.append({
                            'fact': fact,
                            'supported': first_result.get('label', '') == 'SUPPORTS',
                            'confidence': first_result.get('score', 0.0)
                        })
        except Exception as e:
            logger.error(f"Error verifying facts: {e}")
            
        return verified

    def _build_knowledge_graph(self, context: Dict):
        """Build knowledge graph from context"""
        try:
            # Add entities as nodes
            for entity in context.get('entities', []):
                self.graph.add_node(entity)
                
            # Add relationships as edges
            for rel in context.get('relationships', []):
                self.graph.add_edge(rel['source'], rel['target'], 
                                  type=rel['type'])
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")

class RAGHelper:
    def __init__(self):
        """Initialize RAG helper with research capabilities"""
        self.session = None
        # Initialize single multi-task model
        try:
            self.analyzer = pipeline('text-generation',
                                  model=Config.DEFAULT_MODEL,
                                  device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            logger.error(f"Error initializing analyzer model: {e}")
            self.analyzer = None

        # Initialize keyword research and SEO analysis components
        self.keyword_researcher = KeywordResearcher()
        self.seo_analyzer = SEOAnalyzer()

        self.news_sources = [
            'https://newsapi.org/v2/everything',
            'https://api.nytimes.com/svc/search/v2/articlesearch.json',
            'https://api.theguardian.com/search'
        ]

    def _remove_emojis_and_special_chars(self, text: str) -> str:
        """Remove emojis and unwanted special characters from text"""
        if not text:
            return ""
            
        try:
            # First, try to use emoji package if available
            try:
                import emoji
                text = emoji.replace_emoji(text, replace='')
            except ImportError:
                # If emoji package is not installed, fallback to comprehensive regex for emojis
                emoji_pattern = re.compile(
                    "["
                    "\U0001F600-\U0001F64F"  # emoticons
                    "\U0001F300-\U0001F5FF"  # symbols & pictographs
                    "\U0001F680-\U0001F6FF"  # transport & map symbols
                    "\U0001F700-\U0001F77F"  # alchemical symbols
                    "\U0001F780-\U0001F7FF"  # Geometric Shapes
                    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                    "\U0001FA00-\U0001FA6F"  # Chess Symbols
                    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    "\U00002702-\U000027B0"  # Dingbats
                    "\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)
                text = emoji_pattern.sub(r'', text)

            # Remove hashtags and mentions
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'@\w+', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove non-ASCII characters
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            
            # Remove other unwanted special characters except basic punctuation and alphanumerics
            text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'\"()-]', '', text)

            # Fix common formatting issues
            text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Fix punctuation spacing
            text = re.sub(r'([.,;:!?])\s+([.,;:!?])', r'\1\2', text)  # Fix consecutive punctuation
            text = re.sub(r'\.{2,}', '...', text)  # Standardize ellipses
            text = re.sub(r',,+', ',', text)  # Remove multiple commas
            
            # Remove excessive capitalization (all caps sentences)
            words = text.split()
            for i, word in enumerate(words):
                if len(word) > 3 and word.isupper():
                    words[i] = word.capitalize()
            text = ' '.join(words)

            return text.strip()
        except Exception as e:
            logger.error(f"Error removing emojis and special chars: {e}")
            return text

    async def get_context(self, topic: str) -> Dict:
        """Get comprehensive context for topic"""
        try:
            # Clean topic text
            topic = re.sub(r'\([^)]*\)', '', topic).strip()  # Remove year/date markers
            topic = re.sub(r'^\d+\s+', '', topic)  # Remove leading numbers
            
            # Extract product type and category
            product_match = re.search(r'(Apple Watch|iPhone|iPad|MacBook|AirPods)', topic)
            product_type = product_match.group(1) if product_match else None
            
            # Extract accessory types
            accessory_types = []
            if 'accessory' in topic.lower() or 'accessories' in topic.lower():
                accessory_matches = re.findall(r'(bands?|chargers?|cases|protectors?|stands?|docks?)', topic.lower())
                accessory_types = list(set(accessory_matches))
            
            # Build keyword data
            keyword_data = {
                'primary': [],
                'secondary': [],
                'semantic_groups': {},
                'questions': []
            }
            
            # Add primary keywords
            if product_type:
                keyword_data['primary'].append(f"best {product_type} accessories")
                if accessory_types:
                    keyword_data['primary'].extend([f"{product_type} {acc}" for acc in accessory_types[:2]])
            
            # Add secondary keywords
            secondary_terms = [
                'premium', 'affordable', 'durable', 'stylish', 'wireless',
                'protective', 'charging', 'official', 'third-party'
            ]
            keyword_data['secondary'] = random.sample(secondary_terms, min(5, len(secondary_terms)))
            
            # Add semantic groups
            if accessory_types:
                semantic_groups = {
                    'protection': ['cases', 'screen protectors', 'covers', 'bumpers'],
                    'charging': ['chargers', 'wireless charging', 'power banks', 'charging stands'],
                    'style': ['bands', 'straps', 'designer accessories', 'custom designs'],
                    'functionality': ['docks', 'stands', 'adapters', 'storage']
                }
                keyword_data['semantic_groups'] = semantic_groups
            
            # Add common questions
            if product_type and accessory_types:
                keyword_data['questions'] = [
                    f"What are the best {product_type} {acc}s in 2025?" for acc in accessory_types[:3]
                ]
            
            # Add related entities
            entities = []
            if product_type == 'Apple Watch':
                entities.extend(['watchOS', 'Series 9', 'Ultra 2'])
            
            return {
                'seo_keywords': keyword_data,
                'entities': entities,
                'product_type': product_type,
                'accessory_types': accessory_types,
                'key_points': [
                    f"Latest {product_type} accessories for 2025",
                    "Premium and budget-friendly options",
                    "Quality and durability considerations",
                    "Style and functionality balance",
                    "Brand reliability and warranty"
                ] if product_type else []
            }
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return {}

    async def _get_wikipedia_info(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get information from Wikipedia"""
        if wikipedia is None:
            logger.warning("Wikipedia module not available")
            return None

        try:
            # Search Wikipedia
            search_results = wikipedia.search(topic)
            if not search_results:
                return None

            try:
                # Get page content
                page = wikipedia.page(search_results[0], auto_suggest=False)
                content = page.content
                summary = wikipedia.summary(search_results[0], sentences=5)

                # Extract facts and entities
                facts = self._extract_facts(content)
                entities = self._extract_entities(content)

                return {
                    'summary': summary,
                    'key_points': self._extract_key_points(content),
                    'facts': facts,
                    'sources': [page.url],
                    'entities': entities
                }
            except Exception as e:
                # Check if it's a DisambiguationError
                if wikipedia and hasattr(e, 'options') and hasattr(wikipedia, 'exceptions') and isinstance(e, wikipedia.exceptions.DisambiguationError):
                    if e.options:
                        try:
                            page = wikipedia.page(e.options[0], auto_suggest=False)
                            return {
                                'summary': wikipedia.summary(e.options[0], sentences=3),
                                'key_points': self._extract_key_points(page.content),
                                'facts': self._extract_facts(page.content),
                                'sources': [page.url],
                                'entities': self._extract_entities(page.content)
                            }
                        except Exception:
                            logger.error(f"Error processing disambiguation option: {e.options[0]}")
                            pass
            except:
                pass

            return None

        except Exception as e:
            logger.error(f"Error getting Wikipedia info: {e}")
            return None

    async def _get_news_info(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get recent news information about the topic using RSS feeds only"""
        try:
            # Get RSS feed information (already implemented)
            rss_info = await self._get_rss_feeds(topic)
            if not rss_info:
                return None

            # Process RSS results to match expected format
            return {
                'latest_developments': rss_info.get('latest_developments', []),
                'sources': rss_info.get('sources', []),
                'facts': rss_info.get('facts', [])
            }

        except Exception as e:
            logger.error(f"Error getting news info from RSS: {e}")
            return None

    async def _get_research_papers(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get relevant research paper information"""
        try:
            # Use arXiv API
            url = f"http://export.arxiv.org/api/query?search_query={quote_plus(topic)}&max_results=5"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                    
                text = await response.text()
                soup = BeautifulSoup(text, 'xml')
                
                papers = []
                for entry in soup.find_all('entry'):
                    title_elem = entry.find('title')
                    summary_elem = entry.find('summary')
                    id_elem = entry.find('id')

                    papers.append({
                        'title': title_elem.text if title_elem else '',
                        'summary': summary_elem.text if summary_elem else '',
                        'url': id_elem.text if id_elem else ''
                    })
                    
                return {
                    'research': papers,
                    'sources': [p['url'] for p in papers],
                    'facts': [p['summary'] for p in papers]
                }

        except Exception as e:
            logger.error(f"Error getting research papers: {e}")
            return None

    async def _scrape_web_content(self, topic: str) -> Optional[Dict[str, Any]]:
        """Scrape web content for the given topic"""
        try:
            if not self.session:
                return None
                
            search_url = f"https://www.google.com/search?q={quote_plus(topic)}"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    
                    # Extract search results
                    results = []
                    for result in soup.find_all('div', class_='tF2Cxc'):
                        title = result.find('h3')
                        description = result.find('div', class_='VwiC3b')
                        link = result.find('a')
                        
                        if title and description and link:
                            title_text = title.text if title and hasattr(title, 'text') else ''
                            desc_text = description.text if description and hasattr(description, 'text') else ''
                            url = link.get('href') if link and hasattr(link, 'get') else ''
                            
                            if title_text and desc_text and url:
                                results.append({
                                    'title': title_text,
                                    'summary': desc_text,
                                    'url': url
                                })

                    if not results:
                        return None

                    return {
                        'key_points': [r['title'] for r in results[:5] if 'title' in r and r['title']],
                        'facts': [r['summary'] for r in results[:5] if 'summary' in r and r['summary']],
                        'sources': [r['url'] for r in results[:5] if 'url' in r and r['url']]
                    }

            return None

        except Exception as e:
            logger.error(f"Error scraping web content: {e}")
            return None

    def _generate_prompt(self, topic: str, context: Dict[str, Any]) -> str:
        """Basic prompt generation using context"""
        prompt = f"Write comprehensive article about {topic}"
        
        # Add structure based on available context
        if context.get('timeline'):
            prompt += " with chronological timeline"
        if context.get('relationships'):
            prompt += " including relationship analysis"
        if context.get('verified_facts'):
            prompt += " highlighting verified facts"
        
        # Add content sections
        if context.get('key_points'):
            prompt += "\n\nKey points:\n" + "\n".join(
                f"- {point}" for point in context['key_points'][:5])
        
        if context.get('facts'):
            prompt += "\n\nImportant facts:\n" + "\n".join(
                f"- {fact}" for fact in context['facts'][:10])
        
        return prompt

    def generate_enhanced_prompt(self, topic: str, context: Dict[str, Any] = None) -> str:
        """Generate focused prompt using only the most relevant information"""
        if context is None:
            context = {}
            
        # Extract and prioritize only the most relevant information
        key_points = context.get('key_points', [])[:3]  # Top 3 most relevant points
        facts = [f for f in context.get('facts', []) if self._calculate_relevance(f, topic) > 0.4][:3]  # Filtered facts
        sources = context.get('sources', [])[:2]  # Top 2 sources
        
        # Build concise, focused prompt
        prompt = f"""Write a comprehensive article about: {topic}

Most Important Information:
{chr(10).join(f"- {point}" for point in key_points)}

Key Facts:
{chr(10).join(f"- {fact}" for fact in facts)}

Article Requirements:
- Length: 800-1000 words
- Structure: Introduction, Analysis, Conclusion
- Tone: Professional and authoritative
- Sources: {', '.join(sources) if sources else 'None provided'}
- Focus: Stay strictly on topic and use only verified information"""

        return prompt

    async def _get_reddit_info(self, topic: str) -> List[Dict[str, Any]]:
        """Get information from Reddit"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Get Reddit OAuth token
            auth = aiohttp.BasicAuth(Config.REDDIT_CLIENT_ID, Config.REDDIT_CLIENT_SECRET)
            token_url = "https://www.reddit.com/api/v1/access_token"
            data = {"grant_type": "client_credentials"}
            
            async with self.session.post(token_url, auth=auth, data=data) as response:
                if response.status != 200:
                    return []
                    
                token_data = await response.json()
                access_token = token_data.get('access_token')
                
                if not access_token:
                    return []

            # Search Reddit
            headers = {"Authorization": f"Bearer {access_token}"}
            # Ensure topic is properly encoded
            try:
                encoded_topic = quote_plus(str(topic).encode('utf-8')) if topic else ''
                search_url = f"https://oauth.reddit.com/r/all/search?q={encoded_topic}&sort=relevance&limit=5"
            except Exception as e:
                logger.error(f"Error encoding search topic: {e}")
                return []
            
            async with self.session.get(search_url, headers=headers) as response:
                if response.status != 200:
                    return []
                    
                data = await response.json()
                posts = data.get('data', {}).get('children', [])
                
                results = []
                for post in posts:
                    if not isinstance(post, dict):
                        continue
                        
                    post_data = post.get('data', {})
                    if not isinstance(post_data, dict):
                        continue
                        
                    # Safely get all fields with type checking
                    title = str(post_data.get('title', '')) if post_data.get('title') else ''
                    permalink = str(post_data.get('permalink', '')) if post_data.get('permalink') else ''
                    selftext = str(post_data.get('selftext', '')) if post_data.get('selftext') else ''
                    
                    # Only add if we have valid data
                    if title and permalink:
                        results.append({
                            'title': title,
                            'url': f"https://reddit.com{permalink}",
                            'summary': selftext[:200],
                            'source': 'Reddit',
                            'engagement': {
                                'upvotes': int(post_data.get('ups', 0)),
                                'comments': int(post_data.get('num_comments', 0))
                            }
                        })
                
                return results

        except Exception as e:
            logger.error(f"Error getting Reddit info: {e}")
            return []

    async def _get_twitter_info(self, topic: str) -> List[Dict[str, Any]]:
        """Get information from Twitter"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            headers = {
                "Authorization": f"Bearer {Config.TWITTER_API_KEY}",
                "User-Agent": "v2RecentSearchPython"
            }

            # Search Twitter API v2
            search_url = "https://api.twitter.com/2/tweets/search/recent"
            params = {
                'query': topic,
                'max_results': 10,
                'tweet.fields': 'created_at,public_metrics,entities'
            }
            
            async with self.session.get(search_url, headers=headers, params=params) as response:
                if response.status != 200:
                    return []
                    
                data = await response.json()
                tweets = data.get('data', [])
                
                results = []
                for tweet in tweets:
                    if tweet:
                        results.append({
                            'title': tweet.get('text', '')[:100],
                            'url': f"https://twitter.com/i/web/status/{tweet.get('id')}",
                            'summary': tweet.get('text'),
                            'source': 'Twitter',
                            'engagement': tweet.get('public_metrics', {})
                        })
                
                return results

        except Exception as e:
            logger.error(f"Error getting Twitter info: {e}")
            return []

    def _extract_facts(self, text: str) -> List[str]:
        """Extract key facts from text"""
        facts = []
        sentences = text.split('.')
        
        for sentence in sentences:
            # Look for fact patterns
            if re.search(r'(is|was|were|has|had|will|can|could|should|would|may|might)\s', sentence):
                fact = sentence.strip()
                if len(fact) > 10 and len(fact) < 200:
                    facts.append(fact)

        return facts[:10]  # Return top 10 facts

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text[:10000])  # Limit text length for performance
            
            entities = []
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    entities.append(ent.text)
                    
            return list(set(entities))  # Return unique entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text"""
        points = []
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            # Look for important sentences
            if re.search(r'(important|significant|key|main|critical|essential)', para.lower()):
                points.append(para.strip())
                
        return points[:5]  # Return top 5 key points

    def _merge_context(self, base: Dict[str, Any], new: Dict[str, Any]) -> None:
        """Merge with basic deduplication"""
        # Simple text similarity deduplication
        seen_facts = set()
        for fact in base.get('facts', []):
            seen_facts.add(fact.lower().strip())
        
        for fact in new.get('facts', []):
            norm_fact = fact.lower().strip()
            if norm_fact not in seen_facts:
                base.setdefault('facts', []).append(fact)
                seen_facts.add(norm_fact)
        
        # Other merges remain same
        for key, value in new.items():
            if key == 'facts':
                continue
            if isinstance(value, list):
                base.setdefault(key, []).extend(value)
            elif isinstance(value, dict):
                base.setdefault(key, {}).update(value)
            else:
                base.setdefault(key, value)

    async def _detect_topic_type(self, topic: str) -> str:
        """Detect the type of topic"""
        topic_lower = topic.lower()
        
        # Check patterns
        patterns = {
            'news': r'(news|latest|update|breaking)',
            'person': r'(who is|biography|life of)',
            'event': r'(event|conference|festival|ceremony)',
            'product': r'(product|device|gadjet|tool)',
            'company': r'(company|organization|firm)',
            'technology': r'(technology|software|hardware|app)',
            'tutorial': r'(how to|guide|tutorial|steps)',
            'review': r'(review|comparison|vs|versus)',
            'analysis': r'(analysis|study|research|report)'
        }
        
        for type_name, pattern in patterns.items():
            if re.search(pattern, topic_lower):
                return type_name
                
        return 'general'

    async def close(self):
        """Cleanup resources"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            self.session = None
        except Exception as e:
            logger.error(f"Error closing RAG helper: {e}")

    async def _get_rss_feeds(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get highly relevant RSS feed information with strict topic filtering"""
        if not self.session:
            return None
            
        # Use only topic-specific RSS feeds with proper query parameters
        rss_feeds = [
            f'https://news.google.com/rss/search?q={quote_plus(topic)}&num=3',
            f'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml?query={quote_plus(topic)}',
            f'https://content.guardianapis.com/search?q={quote_plus(topic)}&api-key={Config.GUARDIAN_API_KEY}&format=rss'
        ]
        
        results = []
        for feed_url in rss_feeds:
            try:
                async with self.session.get(feed_url, timeout=10) as response:
                    if response.status == 200:
                        text = await response.text()
                        feed = feedparser.parse(text)
                        
                        # Filter entries to only those containing the exact topic
                        for entry in feed.entries:
                            if not (hasattr(entry, 'title') and hasattr(entry, 'description')):
                                continue
                                
                            # Check if topic appears in title or description using exact match
                            title_contains = re.search(r'\b' + re.escape(topic.lower()) + r'\b', entry.title.lower())
                            desc_contains = re.search(r'\b' + re.escape(topic.lower()) + r'\b', entry.description.lower())
                            
                            if title_contains or desc_contains:
                                # Extract only the relevant parts with context
                                summary = self._extract_relevant_text(entry.description, topic, 
                                    window=3)  # Include 3 sentences around matches
                                
                                results.append({
                                    'title': entry.title[:100],
                                    'summary': summary[:150] + '...' if len(summary) > 150 else summary,
                                    'url': entry.link if hasattr(entry, 'link') else '',
                                    'relevance': 1.0 if title_contains else 0.7  # Higher score for title matches
                                })
                                
                                if len(results) >= 5:  # Limit to 5 most relevant results
                                    break
            except Exception as e:
                logger.warning(f"Skipping RSS feed {feed_url}: {str(e)}")
                continue

        if not results:
            return None

        return {
            'key_points': [r['title'] for r in results],
            'facts': [r['summary'] for r in results],
            'sources': [r['url'] for r in results if r['url']]
        }

    def _extract_relevant_text(self, text: str, topic: str, window: int = 1) -> str:
        """Extract sentences containing the topic with surrounding context"""
        if not text:
            return ""
            
        sentences = re.split(r'(?<=[.!?])\s+', text)
        relevant_indices = []
        
        # Find indices of sentences containing the topic
        for i, sentence in enumerate(sentences):
            if re.search(r'\b' + re.escape(topic.lower()) + r'\b', sentence.lower()):
                # Add window around the match
                start = max(0, i - window)
                end = min(len(sentences), i + window + 1)
                relevant_indices.extend(range(start, end))
                
        # Get unique sorted indices
        relevant_indices = sorted(set(relevant_indices))
        
        # Extract and join the relevant sentences
        relevant = [sentences[i] for i in relevant_indices if i < len(sentences)]
        return ' '.join(relevant) if relevant else text[:150] + '...'

    def _calculate_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score between text and query"""
        try:
            # Simple word overlap relevance score
            text_words = set(text.lower().split())
            query_words = set(query.lower().split())
            overlap = len(text_words.intersection(query_words))
            return overlap / (len(text_words) + len(query_words) - overlap)
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0

    async def generate_with_context(self, topic: str, section: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Generate content for a specific section using RAG approach"""
        try:
            if not self.analyzer:
                logger.error("Text generation model not initialized")
                return None

            # Generate focused prompt for the section
            section_prompt = f"Write a detailed section about '{section}' for the topic '{topic}'"
            
            if context:
                # Add relevant context from knowledge base
                if context.get('key_points'):
                    relevant_points = [p for p in context['key_points'] if self._calculate_relevance(p, section) > 0.4]
                    if relevant_points:
                        section_prompt += "\n\nKey points to include:\n" + "\n".join(f"- {p}" for p in relevant_points[:3])
                
                if context.get('facts'):
                    relevant_facts = [f for f in context['facts'] if self._calculate_relevance(f, section) > 0.4]
                    if relevant_facts:
                        section_prompt += "\n\nRelevant facts:\n" + "\n".join(f"- {f}" for f in relevant_facts[:3])

            # Generate content with the model
            result = self.analyzer(section_prompt, 
                                max_length=1000,
                                num_return_sequences=1,
                                temperature=0.7)
            
            if result and isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '') if isinstance(result[0], dict) else str(result[0])
                return self._remove_emojis_and_special_chars(generated_text)
            
            return None

        except Exception as e:
            logger.error(f"Error generating content with context: {e}")
            return None
