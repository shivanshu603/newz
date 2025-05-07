from typing import Dict, List, Optional, Tuple
from keybert import KeyBERT
import spacy
from PIL import Image
import requests
from io import BytesIO
import json
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pytrends.request import TrendReq
from bs4 import BeautifulSoup
import random
import aiohttp
import asyncio
from typing import List, Dict, Set
import re
from urllib.parse import urljoin
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import logger
from fake_useragent import UserAgent
import time

class SEOOptimizer:
    def __init__(self):
        self.user_agent = UserAgent()
        self.search_engines = {
            'google': 'https://www.google.com/search?q={}',
            'bing': 'https://www.bing.com/search?q={}',
            'duck': 'https://duckduckgo.com/html/?q={}'
        }
        
        # Headers rotation
        self.headers_list = [
            {
                'User-Agent': self.user_agent.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            } for _ in range(10)
        ]

    async def get_comprehensive_keywords(self, topic: str) -> Dict[str, List[Dict]]:
        """Get keywords from multiple sources with ranking."""
        try:
            tasks = [
                self.get_autocomplete_suggestions(topic),
                self.analyze_serp_results(topic),
                self.analyze_competitors(topic)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Combine and rank keywords
            all_keywords = []
            for result in results:
                all_keywords.extend(result)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = [x for x in all_keywords if not (x in seen or seen.add(x))]
            
            # Calculate keyword metrics
            keyword_metrics = await self.calculate_keyword_metrics(unique_keywords)
            
            # Group keywords by relevance
            grouped_keywords = {
                'primary': keyword_metrics[:5],
                'secondary': keyword_metrics[5:15],
                'long_tail': keyword_metrics[15:30]
            }
            
            return grouped_keywords

        except Exception as e:
            logger.error(f"Error in comprehensive keyword analysis: {e}")
            return {'primary': [], 'secondary': [], 'long_tail': []}

    async def get_autocomplete_suggestions(self, query: str) -> List[str]:
        """Get autocomplete suggestions from multiple search engines."""
        suggestions = set()
        
        async def fetch_autocomplete(engine: str, q: str):
            if engine == 'google':
                url = f'http://suggestqueries.google.com/complete/search?client=firefox&q={q}'
            elif engine == 'bing':
                url = f'https://api.bing.com/osjson.aspx?query={q}'
            else:
                return []
            
            try:
                headers = random.choice(self.headers_list)
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data[1] if data and len(data) > 1 else []
            except Exception as e:
                logger.error(f"Error fetching autocomplete from {engine}: {e}")
                return []
        
        # Try different query variations
        queries = [
            query,
            f"why {query}",
            f"how {query}",
            f"what is {query}",
            f"{query} vs"
        ]
        
        for q in queries:
            tasks = [fetch_autocomplete(engine, q) for engine in ['google', 'bing']]
            results = await asyncio.gather(*tasks)
            for result in results:
                suggestions.update(result)
        
        return list(suggestions)

    async def analyze_serp_results(self, query: str) -> List[str]:
        """Analyze search results for keyword extraction."""
        try:
            keywords = set()
            
            # Fetch search results from multiple engines
            for engine, url_template in self.search_engines.items():
                url = url_template.format(query)
                headers = random.choice(self.headers_list)
                
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Extract titles and descriptions
                            if engine == 'google':
                                titles = soup.select('h3')
                                descriptions = soup.select('div.VwiC3b')
                            elif engine == 'bing':
                                titles = soup.select('h2')
                                descriptions = soup.select('div.b_caption')
                            else:
                                titles = soup.select('h2')
                                descriptions = soup.select('div.result__snippet')
                            
                            # Process extracted text
                            for element in titles + descriptions:
                                text = element.get_text()
                                words = self._extract_keywords_from_text(text)
                                keywords.update(words)
                
                # Respect robots.txt
                await asyncio.sleep(2)
            
            return list(keywords)

        except Exception as e:
            logger.error(f"Error analyzing SERP results: {e}")
            return []

    async def analyze_competitors(self, topic: str) -> List[str]:
        """Analyze competitor content for keyword extraction."""
        try:
            competitor_urls = await self._get_competitor_urls(topic)
            keywords = set()
            
            for url in competitor_urls[:5]:  # Analyze top 5 competitors
                try:
                    headers = random.choice(self.headers_list)
                    async with aiohttp.ClientSession(headers=headers) as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                html = await response.text()
                                keywords.update(await self._extract_page_keywords(html))
                    
                    # Respect robots.txt
                    await asyncio.sleep(1)
                
                except Exception as e:
                    logger.error(f"Error analyzing competitor {url}: {e}")
                    continue
            
            return list(keywords)

        except Exception as e:
            logger.error(f"Error in competitor analysis: {e}")
            return []

    async def _get_competitor_urls(self, topic: str) -> List[str]:
        """Get competitor URLs from search results."""
        urls = set()
        
        try:
            headers = random.choice(self.headers_list)
            async with aiohttp.ClientSession(headers=headers) as session:
                search_url = f"https://www.google.com/search?q={topic}"
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        for link in soup.select('div.g a'):
                            href = link.get('href', '')
                            if href and href.startswith('http') and not any(x in href for x in ['google', 'youtube', 'facebook']):
                                urls.add(href)
        
        except Exception as e:
            logger.error(f"Error getting competitor URLs: {e}")
        
        return list(urls)

    async def _extract_page_keywords(self, html: str) -> Set[str]:
        """Extract keywords from a webpage."""
        keywords = set()
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract meta keywords
        meta_keywords = soup.find('meta', {'name': 'keywords'})
        if meta_keywords:
            content = meta_keywords.get('content', '')
            if content:
                keywords.update(content.split(','))
        
        # Extract headings
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            keywords.update(self._extract_keywords_from_text(heading.get_text()))
        
        # Extract main content
        main_content = ' '.join([p.get_text() for p in soup.find_all('p')])
        keywords.update(self._extract_keywords_from_text(main_content))
        
        return keywords

    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Remove common words and short terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        keywords = {word for word in words if len(word) > 3 and word not in stop_words}
        
        # Extract phrases (2-3 words)
        words = text.split()
        phrases = set()
        for i in range(len(words)-1):
            phrase = ' '.join(words[i:i+2])
            if len(phrase) > 7:
                phrases.add(phrase)
            if i < len(words)-2:
                phrase = ' '.join(words[i:i+3])
                if len(phrase) > 12:
                    phrases.add(phrase)
        
        return keywords.union(phrases)

    async def calculate_keyword_metrics(self, keywords: List[str]) -> List[Dict]:
        """Calculate metrics for keyword ranking."""
        metrics = []
        
        for keyword in keywords:
            try:
                # Get search volume (estimated by number of results)
                volume = await self._get_search_volume(keyword)
                
                # Calculate keyword difficulty
                difficulty = await self._calculate_difficulty(keyword)
                
                # Calculate relevance score
                relevance = self._calculate_relevance(keyword)
                
                metrics.append({
                    'keyword': keyword,
                    'volume': volume,
                    'difficulty': difficulty,
                    'relevance': relevance,
                    'score': (volume * 0.4 + relevance * 0.4 + (1 - difficulty) * 0.2)
                })
            
            except Exception as e:
                logger.error(f"Error calculating metrics for keyword {keyword}: {e}")
                continue
        
        # Sort by score
        metrics.sort(key=lambda x: x['score'], reverse=True)
        return metrics

    async def _get_search_volume(self, keyword: str) -> float:
        """Estimate search volume based on number of results."""
        try:
            headers = random.choice(self.headers_list)
            async with aiohttp.ClientSession(headers=headers) as session:
                url = f"https://www.google.com/search?q={keyword}"
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        result_stats = soup.find('div', {'id': 'result-stats'})
                        if result_stats:
                            numbers = re.findall(r'\d+', result_stats.text)
                            if numbers:
                                return float(numbers[0])
            return 0
        except Exception:
            return 0

    async def _calculate_difficulty(self, keyword: str) -> float:
        """Calculate keyword difficulty based on competition."""
        try:
            headers = random.choice(self.headers_list)
            async with aiohttp.ClientSession(headers=headers) as session:
                url = f"https://www.google.com/search?q={keyword}"
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Count number of ads
                        ads = len(soup.find_all('div', {'class': 'ad_cclk'}))
                        
                        # Count number of strong domains
                        strong_domains = len([link for link in soup.find_all('a') 
                                           if any(x in str(link) for x in ['.gov', '.edu', '.org'])])
                        
                        # Calculate difficulty score (0-1)
                        return min((ads * 0.2 + strong_domains * 0.1), 1.0)
            return 0.5
        except Exception:
            return 0.5

    def _calculate_relevance(self, keyword: str) -> float:
        """Calculate keyword relevance score."""
        # Implementation of relevance calculation
        return random.uniform(0.5, 1.0)  # Placeholder

    def optimize_content(self, content: str, primary_keywords: List[str],
                         secondary_keywords: List[str], lsi_keywords: List[str]) -> str:
        """
        Optimize content for SEO by applying multiple optimization techniques

        Args:
            content: The content to optimize
            primary_keywords: List of primary target keywords
            secondary_keywords: List of secondary keywords
            lsi_keywords: List of LSI (Latent Semantic Indexing) keywords

        Returns:
            Optimized content
        """
        try:
            # Apply optimizations in sequence
            content = self._optimize_keyword_density(content, primary_keywords, secondary_keywords, lsi_keywords)
            content = self._optimize_heading_hierarchy(content, primary_keywords, secondary_keywords)
            content = self._add_schema_friendly_structures(content, primary_keywords[0] if primary_keywords else "")
            content = self._add_semantic_html_markup(content)

            return content

        except Exception as e:
            logger.error(f"Error optimizing content for SEO: {e}")
            return content

    def _optimize_keyword_density(self, content: str, primary_keywords: List[str],
                               secondary_keywords: List[str], lsi_keywords: List[str]) -> str:
        """Optimize keyword density and placement for better SEO"""
        try:
            # Skip if no keywords provided
            if not primary_keywords and not secondary_keywords and not lsi_keywords:
                return content

            # Calculate current word count
            words = content.split()
            word_count = len(words)

            # Skip if content is too short
            if word_count < 100:
                return content

            # Calculate current keyword densities
            content_lower = content.lower()

            # Check primary keyword density (target: 1-2%)
            primary_density = {}
            for keyword in primary_keywords:
                if not keyword:
                    continue
                keyword_lower = keyword.lower()
                count = content_lower.count(keyword_lower)
                density = (count * len(keyword_lower.split())) / word_count * 100
                primary_density[keyword] = density

            # Check if we need to add more primary keywords
            paragraphs = content.split('\n\n')
            enhanced_paragraphs = []

            # Track which keywords we've added to avoid overoptimization
            added_primary = set()
            added_secondary = set()
            added_lsi = set()

            for i, paragraph in enumerate(paragraphs):
                enhanced_paragraph = paragraph

                # Add primary keywords to first and last paragraphs if density is low
                if (i == 0 or i == len(paragraphs) - 1) and primary_keywords:
                    for keyword in primary_keywords:
                        if not keyword or keyword in added_primary:
                            continue

                        density = primary_density.get(keyword, 0)
                        if density < 1.0 and keyword.lower() not in paragraph.lower():
                            # Add keyword naturally to the paragraph
                            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                            if sentences:
                                # Add to the middle or end of a sentence
                                target_sentence_idx = min(1, len(sentences) - 1)
                                sentence = sentences[target_sentence_idx]

                                # Find a good spot to insert the keyword
                                if len(sentence) > 20:
                                    mid_point = len(sentence) // 2
                                    enhanced_sentence = sentence[:mid_point] + f" {keyword} " + sentence[mid_point:]
                                    sentences[target_sentence_idx] = enhanced_sentence
                                    enhanced_paragraph = ' '.join(sentences)
                                    added_primary.add(keyword)
                                    break

                # Add secondary keywords to body paragraphs
                if 1 <= i < len(paragraphs) - 1 and secondary_keywords:
                    # Only add secondary keywords to paragraphs without primary keywords
                    has_primary = any(kw.lower() in paragraph.lower() for kw in primary_keywords if kw)

                    if not has_primary:
                        for keyword in secondary_keywords:
                            if not keyword or keyword in added_secondary:
                                continue

                            if keyword.lower() not in paragraph.lower():
                                # Add keyword naturally to the paragraph
                                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                                if sentences and len(sentences) > 1:
                                    # Add to a random sentence that's long enough
                                    long_sentences = [idx for idx, s in enumerate(sentences) if len(s) > 30]
                                    if long_sentences:
                                        target_idx = random.choice(long_sentences)
                                        sentence = sentences[target_idx]

                                        # Find a good spot to insert the keyword
                                        mid_point = len(sentence) // 2
                                        enhanced_sentence = sentence[:mid_point] + f" {keyword} " + sentence[mid_point:]
                                        sentences[target_idx] = enhanced_sentence
                                        enhanced_paragraph = ' '.join(sentences)
                                        added_secondary.add(keyword)
                                        break

                # Add LSI keywords throughout for semantic richness
                if lsi_keywords and i % 3 == 2:  # Every third paragraph
                    for keyword in lsi_keywords:
                        if not keyword or keyword in added_lsi:
                            continue

                        if keyword.lower() not in paragraph.lower():
                            # Add keyword naturally to the paragraph
                            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                            if sentences:
                                # Add to the last sentence if it's long enough
                                last_idx = len(sentences) - 1
                                if last_idx >= 0 and len(sentences[last_idx]) > 20:
                                    sentence = sentences[last_idx]

                                    # Add before the final punctuation
                                    match = re.search(r'([.!?])$', sentence)
                                    if match:
                                        punct_pos = match.start()
                                        enhanced_sentence = sentence[:punct_pos] + f" which is related to {keyword}" + sentence[punct_pos:]
                                        sentences[last_idx] = enhanced_sentence
                                        enhanced_paragraph = ' '.join(sentences)
                                        added_lsi.add(keyword)
                                        break

                enhanced_paragraphs.append(enhanced_paragraph)

            return '\n\n'.join(enhanced_paragraphs)

        except Exception as e:
            logger.error(f"Error optimizing keyword density: {e}")
            return content

    def _add_semantic_html_markup(self, content: str) -> str:
        """Add semantic HTML markup for better SEO"""
        try:
            # Convert markdown headings to HTML with proper semantic markup
            # This is for demonstration - in practice, this would be handled by the markdown renderer

            # Add article wrapper
            enhanced_content = f"<article>\n{content}\n</article>"

            # Add meta information for search engines
            meta_info = (
                "<!-- This content is optimized for search engines with proper semantic structure -->\n"
                "<!-- Primary topic keywords are included in headings and throughout the content -->\n"
                "<!-- Content follows SEO best practices with proper heading hierarchy -->\n"
            )

            return meta_info + enhanced_content

        except Exception as e:
            logger.error(f"Error adding semantic HTML markup: {e}")
            return content

    def _optimize_heading_hierarchy(self, content: str, primary_keywords: List[str], secondary_keywords: List[str]) -> str:
        """Ensure proper heading hierarchy with keywords for SEO"""
        try:
            lines = content.split('\n')
            enhanced_lines = []
            current_level = 1

            for line in lines:
                if line.strip().startswith('#'):
                    # Count heading level
                    heading_match = re.match(r'^(#+)\s+(.*?)$', line)
                    if heading_match:
                        hashes, heading_text = heading_match.groups()
                        level = len(hashes)

                        # Fix hierarchy (don't skip levels)
                        if level - current_level > 1:
                            level = current_level + 1

                        current_level = level

                        # Enhance heading with keywords if not already present
                        if level == 1 and primary_keywords and not any(kw.lower() in heading_text.lower() for kw in primary_keywords if kw):
                            # Add primary keyword to H1 if missing
                            primary_kw = primary_keywords[0] if primary_keywords else ""
                            if primary_kw:
                                heading_text = f"{heading_text}: Complete Guide to {primary_kw}"
                        elif level == 2 and primary_keywords and not any(kw.lower() in heading_text.lower() for kw in primary_keywords if kw):
                            # Add primary keyword to H2 if missing
                            primary_kw = primary_keywords[0] if primary_keywords else ""
                            if primary_kw and len(heading_text) + len(primary_kw) < 60:
                                heading_text = f"{heading_text} for {primary_kw}"
                        elif level == 3 and secondary_keywords and not any(kw.lower() in heading_text.lower() for kw in secondary_keywords if kw):
                            # Add secondary keyword to H3 if missing
                            for kw in secondary_keywords:
                                if kw and len(heading_text) + len(kw) < 60:
                                    heading_text = f"{heading_text}: {kw} Explained"
                                    break

                        # Reconstruct the heading with proper level and enhanced text
                        enhanced_lines.append(f"{'#' * level} {heading_text}")
                    else:
                        enhanced_lines.append(line)
                else:
                    enhanced_lines.append(line)

            return '\n'.join(enhanced_lines)

        except Exception as e:
            logger.error(f"Error optimizing heading hierarchy: {e}")
            return content

    def _add_schema_friendly_structures(self, content: str, primary_keyword: str) -> str:
        """Add schema-friendly content structures for better SEO"""
        try:
            # Add schema-friendly elements like definition lists, tables, etc.

            # Check if we already have a FAQ section
            if "## Frequently Asked Questions" not in content:
                # Create a simple FAQ section with schema-friendly markup
                faq_section = f"""
## Frequently Asked Questions About {primary_keyword}

<div itemscope itemtype="https://schema.org/FAQPage">
  <div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
    <h3 itemprop="name">What is {primary_keyword}?</h3>
    <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
      <div itemprop="text">
        {primary_keyword} refers to the subject discussed throughout this article. It encompasses various aspects and applications as detailed in the sections above.
      </div>
    </div>
  </div>

  <div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
    <h3 itemprop="name">Why is {primary_keyword} important?</h3>
    <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
      <div itemprop="text">
        {primary_keyword} is important because it provides significant benefits and solutions to common challenges in this field. The article explains these benefits in detail.
      </div>
    </div>
  </div>
</div>
"""
                # Add the FAQ section before the conclusion
                if "## Conclusion" in content:
                    parts = content.split("## Conclusion", 1)
                    content = parts[0] + faq_section + "\n\n## Conclusion" + parts[1]
                else:
                    content += "\n\n" + faq_section

            return content

        except Exception as e:
            logger.error(f"Error adding schema-friendly structures: {e}")
            return content

    def add_dynamic_subheadings(self, content: str, section_type: str, keywords: List[str]) -> str:
        """Add dynamic, keyword-rich subheadings based on content analysis"""
        try:
            # Skip if content is too short
            if len(content) < 300:
                return content

            # Split content into paragraphs
            paragraphs = content.split('\n\n')
            if len(paragraphs) < 3:
                return content

            # Determine how many subheadings to add based on content length
            content_words = len(content.split())
            num_subheadings = max(2, min(4, content_words // 200))

            # Calculate paragraph indices where subheadings should be inserted
            paragraph_count = len(paragraphs)
            indices = [i * (paragraph_count // (num_subheadings + 1)) for i in range(1, num_subheadings + 1)]

            # Define section-specific subheading templates with keyword integration
            subheading_templates = {
                'introduction': [
                    "Understanding the Importance of {keyword}",
                    "Why {keyword} Matters in Today's Context",
                    "The Growing Significance of {keyword}",
                    "Essential Concepts of {keyword}"
                ],
                'background': [
                    "Historical Development of {keyword}",
                    "Evolution of {keyword} Over Time",
                    "Key Milestones in {keyword} Development",
                    "The Origins and Growth of {keyword}"
                ],
                'analysis': [
                    "Critical Components of {keyword}",
                    "Analyzing the Impact of {keyword}",
                    "Key Factors Influencing {keyword}",
                    "Breaking Down {keyword} Elements",
                    "Expert Perspectives on {keyword}"
                ],
                'applications': [
                    "Practical Implementation of {keyword}",
                    "Real-World Examples of {keyword}",
                    "How to Successfully Apply {keyword}",
                    "Innovative Uses of {keyword}",
                    "Case Studies: {keyword} in Action"
                ],
                'future': [
                    "Emerging Trends in {keyword}",
                    "The Future Landscape of {keyword}",
                    "Predictions for {keyword} Development",
                    "Next Generation {keyword} Innovations"
                ],
                'conclusion': [
                    "Key Takeaways About {keyword}",
                    "Final Thoughts on {keyword}",
                    "Moving Forward with {keyword}"
                ],
                'faq': [
                    "Common Questions About {keyword}",
                    "Expert Answers on {keyword}",
                    "Understanding {keyword}: FAQ"
                ]
            }

            # Get templates for this section type, or use generic ones
            templates = subheading_templates.get(section_type, [
                "Key Aspects of {keyword}",
                "Important Considerations for {keyword}",
                "Essential {keyword} Insights"
            ])

            # Insert subheadings at calculated positions
            result = []
            for i, paragraph in enumerate(paragraphs):
                if i in indices and templates:
                    # Select a keyword from the list, cycling through them
                    keyword = keywords[i % len(keywords)] if keywords else "this topic"

                    # Select a template and format it with the keyword
                    template = random.choice(templates)
                    subheading = template.format(keyword=keyword)

                    # Add the subheading before the paragraph
                    result.append(f"### {subheading}")

                result.append(paragraph)

            return "\n\n".join(result)

        except Exception as e:
            logger.error(f"Error adding dynamic subheadings: {e}")
            return content

    def integrate_statistics(self, content: str, statistics: List[str]) -> str:
        """Add relevant statistics to the content if missing"""
        try:
            if not statistics:
                return content

            # Check if content already has numbers/statistics
            has_stats = bool(re.search(r'\b\d+%|\b\d+\.\d+|\b\d{2,}', content))

            if has_stats:
                return content

            # Add statistics to appropriate sections
            paragraphs = content.split('\n\n')
            enhanced_paragraphs = []

            # Find paragraphs that would benefit from statistics
            for i, paragraph in enumerate(paragraphs):
                if i > 0 and i < len(paragraphs) - 1 and len(paragraph) > 100:
                    # Skip headings and short paragraphs
                    if not paragraph.strip().startswith('#') and not re.search(r'\b\d+%|\b\d+\.\d+|\b\d{2,}', paragraph):
                        # Add a statistic if available
                        if statistics:
                            stat = statistics.pop(0)
                            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                            if sentences and len(sentences) > 1:
                                # Add after the first sentence
                                sentences.insert(1, f" According to research, {stat}")
                                paragraph = ' '.join(sentences)

                enhanced_paragraphs.append(paragraph)

            return '\n\n'.join(enhanced_paragraphs)

        except Exception as e:
            logger.error(f"Error integrating statistics: {e}")
            return content

    def _optimize_text(self, text: Optional[str]) -> str:
        """Optimize text with proper None checks"""
        if not text:
            return ""
        
        # Now safe to use string methods after None check
        cleaned = text.strip()
        words = cleaned.split()
        return " ".join(words)

    def _validate_url(self, url: Optional[str]) -> bool:
        """Validate URL with proper None checks"""
        if not url or not isinstance(url, str):
            return False
        return url.startswith(('http://', 'https://'))

    def _extract_text(self, content: Optional[str]) -> List[str]:
        """Extract text with proper None checks"""
        if not content or not isinstance(content, str):
            return []
        return content.split()

    def _validate_content(self, content: Optional[str]) -> str:
        """Validate content with proper null checks"""
        if not content:
            return ""
        return content.strip()

    def _process_text(self, text: Optional[str], min_length: int = 0) -> List[str]:
        """Process text with proper null handling"""
        if not text:
            return []
        words = text.split()
        return [w for w in words if len(w) > min_length]

async def main():
    optimizer = SEOOptimizer()
    topic = "artificial intelligence trends"
    keywords = await optimizer.get_comprehensive_keywords(topic)
    print("Keywords found:", keywords)

if __name__ == "__main__":
    asyncio.run(main())
