from typing import Dict, List, Optional
import re
import nltk
from utils import logger
from seo_validator import SEOValidator
from content_formatter import ContentFormatter
from textblob import TextBlob

class ContentOptimizer:
    KEYWORD_MAPPING = {
        'primary': {
            'headings': ['h1', 'h2'],
            'density': (0.8, 1.2),
            'position': ['first_paragraph', 'conclusion']
        },
        'secondary': {
            'headings': ['h2', 'h3'],
            'density': (0.3, 0.8),
            'position': ['middle_sections']
        },
        'long_tail': {
            'headings': ['h3', 'body'],
            'density': (0.1, 0.3),
            'position': ['throughout']
        }
    }

    def __init__(self):
        self.seo_validator = SEOValidator()
        self.content_formatter = ContentFormatter()
        self.initialize_nltk()

    async def enhance_section(self, content: str, context: dict) -> str:
        """Async method to enhance a content section by optimizing it."""
        try:
            result = self.optimize_content(content, context)
            return result.get('optimized_content', content)
        except Exception as e:
            from utils import logger
            logger.error(f"Error in enhance_section: {e}")
            return content


    def initialize_nltk(self):
        """Initialize required NLTK resources"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.error(f"Error initializing NLTK: {e}")

    def optimize_content(self, content: str, keyword_data: Dict) -> Dict:
        """Optimize content with comprehensive SEO and readability improvements"""
        try:
            # Initial SEO validation
            seo_report = self.seo_validator.validate(content, keyword_data)
            
            # If SEO score is too low, apply optimizations
            if seo_report['score'] < 0.7:
                content = self._apply_seo_optimizations(content, keyword_data, seo_report)
                
            # Format content structure
            content = self.content_formatter.format_article(content, keyword_data)
            
            # Optimize readability
            content = self._improve_readability(content)
            
            # Final SEO validation
            final_seo_report = self.seo_validator.validate(content, keyword_data)
            
            return {
                'optimized_content': content,
                'seo_score': final_seo_report['score'],
                'seo_suggestions': final_seo_report['suggestions'],
                'readability_score': self._calculate_readability_score(content)
            }
            
        except Exception as e:
            logger.error(f"Error in content optimization: {e}")
            return {
                'optimized_content': content,
                'seo_score': 0.0,
                'seo_suggestions': ['Error during optimization'],
                'readability_score': 0.0
            }

    def _apply_seo_optimizations(self, content: str, keyword_data: Dict, seo_report: Dict) -> str:
        """Apply SEO optimizations based on validation report"""
        try:
            # Fix heading hierarchy
            if any("heading" in suggestion.lower() for suggestion in seo_report['suggestions']):
                content = self.content_formatter._enforce_heading_structure(content)
            
            # Optimize keyword placement
            content = self._optimize_keyword_placement(content, keyword_data)
            
            # Add missing semantic variations
            if any("semantic" in suggestion.lower() for suggestion in seo_report['suggestions']):
                content = self._add_semantic_variations(content, keyword_data)
            
            # Enhance meta elements
            content = self._optimize_meta_elements(content, keyword_data)
            
            return content
            
        except Exception as e:
            logger.error(f"Error applying SEO optimizations: {e}")
            return content

    def _optimize_keyword_placement(self, content: str, keyword_data: Dict) -> str:
        """Optimize keyword placement according to mapping rules"""
        try:
            sections = re.split(r'(<h[1-6]>.*?</h[1-6]>)', content, flags=re.IGNORECASE | re.DOTALL)
            optimized_sections = []
            
            current_section = ''
            for section in sections:
                if not section.strip():
                    continue
                    
                # Handle headings
                heading_match = re.match(r'<h(\d)>(.*?)</h\d>', section, re.IGNORECASE)
                if heading_match:
                    level = int(heading_match.group(1))
                    text = heading_match.group(2)
                    
                    # Apply keyword mapping rules
                    if level == 1 and keyword_data.get('primary'):
                        if not any(kw.lower() in text.lower() for kw in keyword_data['primary']):
                            text = f"{keyword_data['primary'][0]}: {text}"
                    elif level in [2, 3] and keyword_data.get('secondary'):
                        if not any(kw.lower() in text.lower() for kw in keyword_data['secondary']):
                            text = f"{text} - {keyword_data['secondary'][0]}"
                            
                    section = f'<h{level}>{text}</h{level}>'
                    
                else:
                    # Handle body content
                    for kw_type, rules in self.KEYWORD_MAPPING.items():
                        keywords = keyword_data.get(kw_type, [])
                        if keywords and rules.get('position', []) == ['throughout']:
                            # Add keywords naturally if missing
                            for keyword in keywords:
                                if keyword.lower() not in section.lower():
                                    section = self._insert_keyword_naturally(section, keyword)
                
                optimized_sections.append(section)
            
            return ''.join(optimized_sections)
            
        except Exception as e:
            logger.error(f"Error optimizing keyword placement: {e}")
            return content

    def _add_semantic_variations(self, content: str, keyword_data: Dict) -> str:
        """Add semantic variations of keywords"""
        try:
            # Get all keywords
            all_keywords = []
            for key in ['primary', 'secondary', 'long_tail']:
                all_keywords.extend(keyword_data.get(key, []))
            
            # Generate semantic variations
            variations = set()
            for keyword in all_keywords:
                # Add simple variations
                words = keyword.split()
                if len(words) > 1:
                    variations.update([
                        ' '.join(words[::-1]),  # Reverse order
                        words[0] + ' and ' + ' '.join(words[1:]),  # Add conjunction
                        ' '.join(words[1:]) + ' ' + words[0]  # Move first word to end
                    ])
            
            # Add variations naturally to content
            for variation in variations:
                if variation.lower() not in content.lower():
                    content = self._insert_keyword_naturally(content, variation)
            
            return content
            
        except Exception as e:
            logger.error(f"Error adding semantic variations: {e}")
            return content

    def _optimize_meta_elements(self, content: str, keyword_data: Dict) -> str:
        """Optimize meta elements like title and description"""
        try:
            # Optimize title (h1)
            title_match = re.search(r'<h1>(.*?)</h1>', content, re.IGNORECASE)
            if title_match and keyword_data.get('primary'):
                title = title_match.group(1)
                if not any(kw.lower() in title.lower() for kw in keyword_data['primary']):
                    new_title = f"{keyword_data['primary'][0]}: {title}"
                    content = content.replace(title_match.group(0), f"<h1>{new_title}</h1>")
            
            return content
            
        except Exception as e:
            logger.error(f"Error optimizing meta elements: {e}")
            return content

    def _improve_readability(self, content: str) -> str:
        """Improve content readability"""
        try:
            # Split into paragraphs
            paragraphs = re.split(r'\n\n+', content)
            improved_paragraphs = []
            
            for para in paragraphs:
                # Skip headings
                if re.match(r'<h[1-6]>', para):
                    improved_paragraphs.append(para)
                    continue
                
                # Analyze paragraph
                blob = TextBlob(para)
                sentence_list = list(blob.sentences)  # Convert to list for proper indexing
                
                # If paragraph is too long, split it
                if len(blob.words) > 100 and len(sentence_list) > 1:
                    mid = len(sentence_list) // 2
                    first_half = ' '.join(str(s) for s in sentence_list[:mid])
                    second_half = ' '.join(str(s) for s in sentence_list[mid:])
                    para = f"{first_half}\n\n{second_half}"
                
                improved_paragraphs.append(para)
            
            return '\n\n'.join(improved_paragraphs)
            
        except Exception as e:
            logger.error(f"Error improving readability: {e}")
            return content

    def _calculate_readability_score(self, content: str) -> float:
        """Calculate content readability score"""
        try:
            # Remove HTML tags
            clean_content = re.sub(r'<[^>]+>', '', content)
            
            # Calculate metrics
            blob = TextBlob(clean_content)
            sentences = len(blob.sentences)
            words = len(blob.words)
            
            if sentences == 0 or words == 0:
                return 0.0
            
            # Calculate average sentence length
            avg_sentence_length = words / sentences
            
            # Penalize very long sentences
            if avg_sentence_length > 25:
                readability_score = max(0.0, 1.0 - ((avg_sentence_length - 25) / 25))
            else:
                readability_score = 1.0
            
            return round(readability_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating readability score: {e}")
            return 0.0

    def _insert_keyword_naturally(self, text: str, keyword: str) -> str:
        """Insert keyword naturally into text"""
        try:
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                return text
            
            # Find best sentence for insertion
            best_sentence_idx = 0
            max_similarity = 0
            
            for idx, sentence in enumerate(sentences):
                # Calculate similarity based on shared words
                sentence_words = set(nltk.word_tokenize(sentence.lower()))
                keyword_words = set(nltk.word_tokenize(keyword.lower()))
                similarity = len(sentence_words & keyword_words) / len(sentence_words | keyword_words) if sentence_words else 0
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_sentence_idx = idx
            
            # Insert keyword
            if max_similarity < 0.3:  # If no good match, append to end
                sentences.append(f"Speaking of {keyword}, it's worth noting its significance.")
            else:
                # Insert naturally into best matching sentence
                sentences[best_sentence_idx] = f"In terms of {keyword}, {sentences[best_sentence_idx].lower()}"
            
            return ' '.join(sentences)
            
        except Exception as e:
            logger.error(f"Error inserting keyword naturally: {e}")
            return text

    async def plan_content(self, topic: str, outline: Dict) -> Dict:
        """Enhanced content planning with improved RAG context gathering"""
        try:
            # Gather initial context from RAG system
            base_context = await self.rag_helper.get_context(topic)
            
            # Analyze topic relevance and create focused context
            topic_context = await self._analyze_topic_relevance(topic, base_context)
            
            # Structure the content plan with layered context
            content_plan = {
                'topic': topic,
                'main_context': topic_context,
                'sections': await self._plan_sections(outline, topic_context),
                'metadata': await self._generate_metadata(topic, topic_context)
            }
            
            return content_plan
        except Exception as e:
            logger.error(f"Error in content planning: {e}")
            raise

    async def _analyze_topic_relevance(self, topic: str, context: Dict) -> Dict:
        """Analyze and enhance topic relevance"""
        try:
            # Extract key concepts and validate against context
            key_concepts = await self.concept_extractor.extract(topic)
            validated_context = await self.context_validator.validate(
                key_concepts,
                context
            )
            
            return validated_context
        except Exception as e:
            logger.error(f"Error analyzing topic relevance: {e}")
            raise
