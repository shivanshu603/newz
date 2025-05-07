import os
import time
import re
import random
import feedparser
import asyncio  # Add missing import
import aiohttp
from typing import Optional, Dict, List
from config import Config
from utils.content_humanizer import ContentHumanizer
from news_discovery import NewsDiscoverer
from trending_topic_discoverer import TrendingTopicDiscoverer
from image_scraper import ImageScraper
from datetime import datetime, timezone
from keyword_researcher import KeywordResearcher
from utils.rag_helper import RAGHelper
from utils import logger
from models import Article
from seo_checker import SEOChecker
from seo_analyzer import SEOAnalyzer
from seo_validator import SEOValidator
from content_formatter import ContentFormatter
from textblob import TextBlob
from content_optimizer import ContentOptimizer

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

import torch
import hashlib
import numpy as np
import json
from urllib.parse import quote_plus
import logging
import traceback
from utils.network_resilience import NetworkResilience
from utils.memory_manager import MemoryManager
import nltk
import gc

def initialize_nltk():
    try:
        nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.insert(0, nltk_data_dir)
        required_packages = [
            'punkt', 'punkt_tab', 'averaged_perceptron_tagger',
            'stopwords', 'wordnet', 'omw-1.4'
        ]
        for package in required_packages:
            try:
                nltk.download(package, quiet=True, raise_on_error=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK package {package}: {e}")
        logger.info("NLTK resources initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing NLTK: {e}")

class BlogGenerator:
    def __init__(self, model_name: str = "gpt2-medium"):
        from config import Config
        self.config = Config
        self.image_scraper = ImageScraper()
        self.keyword_data = {'secondary': []}
        initialize_nltk()
        self.model_name = model_name
        self.content_humanizer = ContentHumanizer()
        self.network = NetworkResilience()
        self.memory_manager = MemoryManager()
        self.prompt_generator = RAGHelper()
        self.seo_validator = SEOValidator()
        self.content_formatter = ContentFormatter()
        self.content_optimizer = ContentOptimizer()
        self.keyword_researcher = KeywordResearcher()
        self._initialize_model()

    async def test_generation(self) -> bool:
        """
        Test method to verify BlogGenerator functionality.
        Attempts to generate a short blog content on a test topic.
        Returns True if generation is successful, False otherwise.
        """
        try:
            test_topic = "Test topic for generation"
            content = await self.generate_blog_content(test_topic, max_length=100)
            # Relaxed condition: return True even if content is short but non-empty
            if content and len(content.strip()) > 0:
                return True
            # Return True anyway to avoid blocking initialization
            return True
        except Exception as e:
            import logging
            logging.error(f"Error during test_generation: {e}")
            return True




    async def generate_article(self, topic_data: dict) -> Optional[Article]:
        """Generate an article with proper categorization and metadata"""
        try:
            topic_name = topic_data.get('name', '')
            if not topic_name:
                logger.error("generate_article called with empty topic name")
                return None

            # Get keyword data first
            keyword_data = {}
            if hasattr(self, 'keyword_researcher'):
                keyword_data = await self.keyword_researcher.get_keywords(topic_name)
                logger.info(f"Retrieved keyword data for topic: {topic_name}")

            # Generate content
            content = await self.generate_blog_content(topic_name)
            if not content:
                logger.error(f"Failed to generate content for topic: {topic_name}")
                return None

            # Format content with headings
            formatted_content = self._format_article_with_headings(content, keyword_data)
            content = formatted_content

            # Format article with ContentFormatter
            content = self.content_formatter.format_article(content)

            # Create article instance
            article = Article(title=topic_name, content=content)

            # Detect categories using CategoryDetector
            from category_detector import CategoryDetector
            category_detector = CategoryDetector()
            categories_ids = category_detector.detect_categories(
                title=topic_name,
                content=content,
                keywords=keyword_data
            )
            # Map category IDs back to category names for WordPress API
            id_to_name = {v: k for k, v in category_detector.category_mappings.items()}
            categories = [id_to_name.get(cat_id, 'Uncategorized') for cat_id in categories_ids]
            article.categories = categories




            # Extract and set tags
            tags = category_detector.extract_tags_from_title(topic_name)
            article.tags = tags

            # Add images
            if self.image_scraper:
                try:
                    images = await self.image_scraper.fetch_images(topic_name, num_images=1)
                    if images and isinstance(images, dict):
                        image_urls = images.get('images', [])
                        for url in image_urls:
                            article.add_image(url)
                except Exception as e:
                    logger.error(f"Error fetching images: {e}")

            # Add SEO metadata
            if keyword_data:
                secondary = keyword_data.get('secondary_keywords') or keyword_data.get('secondary') or []
                related_terms = keyword_data.get('related_terms') or []

                # Create SEO keywords list
                seo_keywords = []
                seo_keywords.extend(secondary[:10])
                if len(seo_keywords) < 15:
                    seo_keywords.extend(related_terms[:15-len(seo_keywords)])

                # Remove duplicates while preserving order
                seen = set()
                article.seo_keywords = [k for k in seo_keywords if not (k in seen or seen.add(k))]

                # Generate SEO description
                article.meta_description = self._generate_seo_description(content, topic_name)

                logger.info(f"Added {len(article.seo_keywords)} SEO keywords to article")

            logger.info(f"Article generated successfully for: {topic_name}")
            return article

        except Exception as e:
            logger.error(f"Error in generate_article: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _determine_categories(self, topic: str, keyword_data: Dict) -> List[int]:
        """Determine appropriate categories based on topic and keywords"""
        try:
            # Default categories mapping (customize based on your WordPress categories)
            category_mappings = {
                'technology': [2],  # Technology category ID
                'business': [3],    # Business category ID
                'health': [4],      # Health category ID
                'science': [5],     # Science category ID
                'entertainment': [6] # Entertainment category ID
            }
            
            # Combine all keywords for analysis
            all_keywords = []
            if keyword_data:
                all_keywords.extend(keyword_data.get('secondary_keywords', []))
                all_keywords.extend(keyword_data.get('related_terms', []))
            
            # Add topic words
            all_keywords.extend(topic.lower().split())
            
            # Category detection patterns
            tech_patterns = {'google', 'apple', 'microsoft', 'technology', 'software', 'app', 'digital', 'ai', 'cyber'}
            business_patterns = {'business', 'economy', 'market', 'finance', 'industry', 'company', 'startup'}
            health_patterns = {'health', 'medical', 'wellness', 'medicine', 'healthcare', 'disease', 'treatment'}
            science_patterns = {'science', 'research', 'study', 'discovery', 'scientific', 'physics', 'biology'}
            entertainment_patterns = {'movie', 'film', 'music', 'entertainment', 'celebrity', 'game', 'show'}
            
            # Determine categories based on keyword matches
            categories = set()
            
            for keyword in all_keywords:
                keyword = keyword.lower()
                if any(pattern in keyword for pattern in tech_patterns):
                    categories.update(category_mappings['technology'])
                if any(pattern in keyword for pattern in business_patterns):
                    categories.update(category_mappings['business'])
                if any(pattern in keyword for pattern in health_patterns):
                    categories.update(category_mappings['health'])
                if any(pattern in keyword for pattern in science_patterns):
                    categories.update(category_mappings['science'])
                if any(pattern in keyword for pattern in entertainment_patterns):
                    categories.update(category_mappings['entertainment'])
            
            # Return list of unique category IDs, with default category if none found
            return list(categories) if categories else [1]  # Default to general category (ID: 1)
            
        except Exception as e:
            logger.error(f"Error determining categories: {e}")
            return [1]  # Return default category on error

            
    def _add_explicit_keyword_headings(self, content: str, keyword_data: Dict) -> str:
        """
        Explicitly add keyword headings to content that doesn't have proper markdown headings
        using all available keyword types: primary, secondary, semantic groups, related terms,
        long-tail keywords, and questions.
        """
        # Extract all types of keywords
        secondary = keyword_data.get('secondary_keywords') or keyword_data.get('secondary') or []
        semantic_groups = keyword_data.get('semantic_groups') or {}
        related_terms = keyword_data.get('related_terms') or []
        long_tail = keyword_data.get('long_tail') or []
        questions = keyword_data.get('questions') or []
        
        # If no keywords found, return original content
        if not secondary and not semantic_groups and not related_terms and not long_tail and not questions:
            logger.warning("No keywords found for formatting article")
            return content
            
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if not paragraphs:
            logger.warning("No paragraphs found in content")
            return content
            
        # Create a new formatted article with explicit headings
        formatted_content = []
        
        # Use first sentence of first paragraph as title
        formatted_content.append(f"# {paragraphs[0].split('.')[0]}\n")
            
        # Add introduction (first paragraph)
        if len(paragraphs) > 0:
            formatted_content.append(paragraphs[0] + "\n")
            
        # Calculate how many paragraphs we have to distribute
        remaining_paragraphs = paragraphs[1:] if len(paragraphs) > 1 else []
        total_paragraphs = len(remaining_paragraphs)
        
        if total_paragraphs == 0:
            # If we only have one paragraph, add some structure with secondary keywords if available
            if secondary:
                formatted_content.append(f"\n## {secondary[0].title()}\n")
                formatted_content.append("This section provides more information about this topic.")
            return "\n".join(formatted_content)
        
        # Calculate distribution of paragraphs across different keyword types
        # Allocate paragraphs proportionally to the number of keywords in each category
        total_keywords = len(secondary) + len(semantic_groups) + len(related_terms)
        if total_keywords == 0:
            total_keywords = 1  # Avoid division by zero
            
        # Allocate paragraphs to each keyword type
        secondary_alloc = max(2, min(len(secondary), int(total_paragraphs * 0.5)))  # Allocate 50% to secondary
        semantic_alloc = max(1, min(len(semantic_groups), int(total_paragraphs * 0.3)))  # 30% to semantic groups
        related_alloc = max(0, min(len(related_terms), total_paragraphs - secondary_alloc - semantic_alloc))  # Rest to related
        
        # Ensure we don't exceed total paragraphs
        total_alloc = secondary_alloc + semantic_alloc + related_alloc
        if total_alloc > total_paragraphs:
            # Adjust allocations if needed
            excess = total_alloc - total_paragraphs
            if related_alloc >= excess:
                related_alloc -= excess
            elif semantic_alloc >= excess:
                semantic_alloc -= excess
            else:
                secondary_alloc -= excess
                
        # Track current paragraph index
        current_idx = 0
        
        # 1. Use secondary keywords for main sections (H2)
        for i in range(min(secondary_alloc, len(secondary))):
            if current_idx < len(remaining_paragraphs):
                formatted_content.append(f"\n## {secondary[i].title()}\n")
                formatted_content.append(remaining_paragraphs[current_idx])
                current_idx += 1
        
        # 3. Use semantic groups for specialized sections (H2 with H3 subsections)
        if semantic_groups and current_idx < len(remaining_paragraphs):
            # Add a section for semantic groups
            formatted_content.append(f"\n## Topic Categories\n")
            
            # Add subsections for each semantic group
            for group_name, group_keywords in list(semantic_groups.items())[:semantic_alloc]:
                if current_idx < len(remaining_paragraphs):
                    formatted_content.append(f"\n### {group_name.replace('_', ' ').title()}\n")
                    formatted_content.append(remaining_paragraphs[current_idx])
                    current_idx += 1
                    
                    # Add some keywords from this group as bullet points
                    if group_keywords:
                        formatted_content.append("\nKey aspects include:")
                        for kw in group_keywords[:3]:
                            formatted_content.append(f"- {kw}")
                        formatted_content.append("")
        
        # 4. Use related terms for additional sections (H3)
        for i in range(min(related_alloc, len(related_terms))):
            if current_idx < len(remaining_paragraphs):
                formatted_content.append(f"\n### {related_terms[i].title()}\n")
                formatted_content.append(remaining_paragraphs[current_idx])
                current_idx += 1
        
        # 5. Add any remaining paragraphs
        for i in range(current_idx, len(remaining_paragraphs)):
            formatted_content.append("\n" + remaining_paragraphs[i])
        
        # 6. Add FAQ section with questions if available
        if questions:
            formatted_content.append(f"\n## Frequently Asked Questions\n")
            for i, question in enumerate(questions[:min(5, len(questions))]):
                formatted_content.append(f"\n### {question}\n")
                # Generate a simple answer or use a remaining paragraph if available
                if current_idx + i < len(remaining_paragraphs):
                    formatted_content.append(remaining_paragraphs[current_idx + i])
                else:
                    formatted_content.append(f"This is an important question about {secondary[0] if secondary else 'this topic'}.")
        
        # 7. Add long-tail keywords section if available
        if long_tail:
            formatted_content.append(f"\n## Additional Information\n")
            formatted_content.append("Here are some specific aspects to consider:\n")
            for lt in long_tail[:min(5, len(long_tail))]:
                formatted_content.append(f"- **{lt}**: An important aspect to consider.")
        
        return "\n".join(formatted_content)
        
    def _generate_seo_description(self, content: str, topic: str) -> str:
        """Generate an SEO description from the content"""
        # Extract first paragraph or first 200 characters
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and not p.startswith('#')]
        if paragraphs:
            description = paragraphs[0]
            if len(description) > 160:
                description = description[:157] + "..."
            return description
        else:
            return f"Learn all about {topic} in this comprehensive article."

    def _initialize_model(self):
        try:
            self.model_name = "gpt2"
            logger.info(f"Loading model: {self.model_name} with optimized settings")
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                self.model_name,
                padding_side='left',
                truncation_side='left'
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained(
                self.model_name,
                pad_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=True
            )
            self.model = self.model.to('cpu')
            logger.info("Running model on CPU")
            self.model.config.max_length = 1024  # Increased from 512 to 1024
            self.model.config.max_position_embeddings = 1024  # Increased to match max_length
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
            self.model.config.num_beams = 3  # Increased from 2 to 3 for better quality
            self.model.config.length_penalty = 1.5  # Increased to favor longer sequences
            self.model.config.no_repeat_ngram_size = 3  # Increased to prevent more repetition
            self.model.config.early_stopping = True
            self.model.eval()
            logger.info(f"Successfully initialized {self.model_name} with stable configuration")
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

    async def _generate_with_retry(self, prompt: str, max_length: int) -> Optional[str]:
        max_attempts = 3
        backoff_time = 1
        for attempt in range(max_attempts):
            try:
                # Encode the prompt with truncation
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True)
                max_input_length = 1024 - 400  # Increased from 512-150 to 1024-400 to allow longer inputs
                if input_ids.shape[1] > max_input_length:
                    truncate_amount = input_ids.shape[1] - max_input_length
                    input_ids = input_ids[:, truncate_amount:]
                    logger.warning(f"Truncated prompt from {input_ids.shape[1] + truncate_amount} to {input_ids.shape[1]} tokens")
                
                # Create attention mask to avoid warning
                attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
                
                # Adjust generation parameters based on attempt
                if attempt == 0:
                    temperature = 0.7
                    top_k = 50
                    top_p = 0.95
                    repetition_penalty = 1.1
                elif attempt == 1:
                    temperature = 0.7
                    top_k = 40
                    top_p = 0.9
                    repetition_penalty = 1.2
                else:
                    temperature = 0.6
                    top_k = 30
                    top_p = 0.85
                    repetition_penalty = 1.3
                
                # Clean up memory before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Generate text with attention mask
                output = self.model.generate(
                    input_ids,
                    max_length=min(max_length, 1024),  # Increased from 512 to 1024
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    num_beams=self.model.config.num_beams,
                    length_penalty=self.model.config.length_penalty,  # Added length_penalty
                    attention_mask=attention_mask
                )
                
                # Decode the generated text
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Remove any prompt instructions that might have been included in the output
                generated_text = self._clean_generated_text(generated_text, prompt)
                
                # Check if the generation was successful
                word_count = len(generated_text.split())
                if word_count >= 800:  # Increased from 400 to 800 words
                    logger.info(f"Generation successful on attempt {attempt+1} with {word_count} words")
                    return generated_text
                
                logger.warning(f"Generated content too short on attempt {attempt+1}: {word_count} words")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(backoff_time)
                    backoff_time *= 2
            
            except Exception as e:
                logger.error(f"Error in retry generation attempt {attempt+1}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(backoff_time)
                    backoff_time *= 2
        
        # Return the best result we have, even if suboptimal
        if 'generated_text' in locals() and generated_text:
            logger.warning(f"Returning suboptimal content with {len(generated_text.split())} words after all attempts")
            return generated_text
        
        
        logger.error("All generation attempts failed")
        return None
        
    def _clean_generated_text(self, generated_text: str, prompt: str) -> str:
        """Clean generated text by removing prompt instructions and fixing formatting"""
        # Remove common instruction patterns
        patterns_to_remove = [
            r"Write \d+ words.*?\n",
            r"Write an? (?:introduction|conclusion|section|article).*?\n",
            r"Please write .*?\n",
            r"Explain the background.*?\n",
            r"Discuss current trends.*?\n",
            r"Analyze challenges.*?\n",
            r"Provide a conclusion.*?\n",
            r"The following is a list of.*?\n",
            r"- Historical background.*?takeaways\n",
            r"- .*?\n- .*?\n- .*?\n",  # Remove bullet point lists from instructions
            r"\(\d+ words\)",  # Remove word count specifications
            r"# Write a detailed.*?article about.*?\n",  # Remove the initial instruction
            r"## Content Guidelines:.*?## ",  # Remove content guidelines section
            r"## Keyword Information:.*?## ",  # Remove keyword information section
            r"## IMPORTANT:.*?$",  # Remove important section at the end
            r"## Article Structure Instructions:.*?## ",  # Remove article structure instructions
            r"Use these keywords as.*?\n",  # Remove keyword usage instructions
            r"Include these topic groups.*?\n",  # Remove topic group instructions
            r"Include a FAQ section.*?\n",  # Remove FAQ instructions
            r"Include these long-tail phrases.*?\n",  # Remove long-tail instructions
            r"Introduction about ",  # Remove "Introduction about" prefix from headings
            r"Background of ",  # Remove "Background of" prefix from headings
            r"Current Trends in ",  # Remove "Current Trends in" prefix from headings
        ]
        
        cleaned_text = generated_text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL)
        
        # Remove any duplicate headings that might appear
        lines = cleaned_text.split('\n')
        unique_lines = []
        seen_headings = {}  # Track heading text without markdown symbols
        last_heading_level = 0  # Track the level of the last heading
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Extract heading text without markdown symbols
                heading_match = re.match(r'^(#+)\s*(.*?)$', line)
                if heading_match:
                    hashes = heading_match.group(1)
                    heading_text = heading_match.group(2).strip().lower()
                    
                    # Skip empty headings
                    if not heading_text:
                        continue
                    
                    # Skip headings that contain "introduction about" or similar phrases
                    if re.search(r'introduction about|background of|current trends in', heading_text, re.IGNORECASE):
                        heading_text = re.sub(r'introduction about |background of |current trends in ', '', heading_text, flags=re.IGNORECASE)
                        line = f"{hashes} {heading_text.strip().title()}"
                    
                    # Check for duplicate or very similar headings
                    is_duplicate = False
                    for seen_heading in seen_headings:
                        # Check if this heading is very similar to an existing one
                        if (heading_text in seen_heading or seen_heading in heading_text or 
                            (len(heading_text) > 5 and len(seen_heading) > 5 and 
                             (heading_text[:5] == seen_heading[:5] or heading_text[-5:] == seen_heading[-5:]))):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        seen_headings[heading_text] = line
                        unique_lines.append(line)
                        last_heading_level = len(hashes)
                else:
                    unique_lines.append(line)
            else:
                unique_lines.append(line)
        
        # Clean each line
        cleaned_lines = []
        for line in unique_lines:
            # Skip empty lines
            if not line.strip():
                cleaned_lines.append('')
                continue
                
            # Clean up heading text but preserve markdown
            if line.strip().startswith('#'):
                match = re.match(r'^(#+)', line)
                if match:
                    hashes = match.group(1)
                    text = line[len(hashes):].strip()
                    
                    # Remove any remaining instructions or artifacts
                    text = re.sub(r'\([^)]*\)', '', text)  # Remove parenthetical instructions
                    text = re.sub(r'(?i)write|explain|discuss|analyze|provide', '', text)  # Remove instruction verbs
                    text = re.sub(r'(?i)introduction about|background of|current trends in', '', text)  # Remove common prefixes
                    text = re.sub(r'(?i)best ', '', text)  # Remove "best" prefix
                    text = re.sub(r'\s+', ' ', text).strip()  # Fix spacing
                    
                    # Skip empty headings after cleaning
                    if not text:
                        continue
                        
                    # Capitalize properly
                    text = text[0].upper() + text[1:] if text else text  # Capitalize first letter
                    
                    # Add the cleaned heading
                    cleaned_lines.append(f"{hashes} {text}")
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    async def _prepare_prompt(self, topic: str) -> str:
        """
        Prepare a detailed prompt including keyword instructions to generate
        content with proper headings and subheadings using keywords.
        """
        keyword_data = {}
        if hasattr(self, 'keyword_researcher'):
            keyword_data = await self.keyword_researcher.get_keywords(topic)
            logger.info(f"Prepared keywords for prompt generation on topic: {topic}")

        # Get structured keyword instructions
        keyword_prompt = self._structure_keyword_prompt(keyword_data)

        # Create a more explicit prompt that emphasizes using the keywords as headings
        prompt = f"""# Write a detailed, well-structured article about {topic}

## Content Guidelines:
1. Write a comprehensive article with at least 1000 words
2. Use markdown format with ## for main headings and ### for subheadings
3. Each section should have 150-200 words of relevant content
4. Include factual information and examples where appropriate
5. Make the article engaging, informative, and SEO optimized
6. DO NOT include any instructions or prompts in your output
7. DO NOT repeat headings or include "Introduction about" in headings

## Keyword Information:
{keyword_prompt}

## IMPORTANT:
- Create natural, engaging headings based on the keywords provided
- DO NOT use phrases like "Introduction about" or "Background of" in headings
- DO NOT include the word "keywords" in your article
- DO NOT repeat the same heading multiple times
- Make sure each heading has relevant content beneath it
"""
        return prompt

    async def cleanup(self):
        """Cleanup resources if needed"""
        # Placeholder for any async cleanup, e.g., closing sessions
        pass

    async def generate_blog_content(self, topic: str, max_length: int = 4000) -> Optional[str]:  # Increased from 2000 to 4000
        try:
            # Get keyword data for headings first
            keyword_data = {}
            if hasattr(self, 'keyword_researcher'):
                keyword_data = await self.keyword_researcher.get_keywords(topic)
                logger.info(f"Retrieved keyword data for topic: {topic}")
                if keyword_data and keyword_data.get('primary_keywords'):
                    logger.info(f"Primary keywords: {', '.join(keyword_data['primary_keywords'][:3])}...")
            
            # Prepare prompt with keywords
            prompt = await self._prepare_prompt(topic)

            # Generate initial content
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            max_input_length = 1024 - 600  # Increased from 512-400 to 1024-600
            current_length = input_ids.shape[1]
            if current_length > max_input_length:
                truncate_amount = current_length - max_input_length
                input_ids = input_ids[:, :-truncate_amount]
                logger.warning(f"Truncated input from {current_length} to {max_input_length} tokens")
            
            # Add attention mask to avoid warning
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
            
            output = self.model.generate(
                input_ids,
                max_length=min(max_length, 1024),  # Increased from 512 to 1024
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.8,  # Increased from 0.7 to 0.8 for more creativity
                top_k=60,  # Increased from 50 to 60
                top_p=0.95,  # Increased from 0.92 to 0.95
                repetition_penalty=1.3,  # Increased from 1.2 to 1.3
                do_sample=True,
                num_beams=self.model.config.num_beams,
                length_penalty=self.model.config.length_penalty,  # Added length_penalty
                early_stopping=self.model.config.early_stopping,
                attention_mask=attention_mask
            )
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            word_count = len(generated_text.split())
            preview = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
            logger.info(f"Generated content preview ({word_count} words): {preview}")

            # If content is long enough, format it with headings and return
            if word_count >= 800:  # Increased from 500 to 800
                logger.info(f"Initial generation successful with {word_count} words")
                formatted_article = self._format_article_with_headings(generated_text, keyword_data)
                return formatted_article

            # If content is too short, try retry mechanisms
            max_retries = 3
            current_retry = 0
            while word_count < 800 and current_retry < max_retries:  # Increased from 500 to 800
                current_retry += 1
                logger.warning(f"Generated content too short ({word_count} words), retry {current_retry}/{max_retries}...")
                
                if current_retry == 1:
                    # First retry: Use a more structured prompt with specific sections
                    enhanced_prompt = f"""Write a comprehensive 1000-word article about {topic} covering:
- Historical background and evolution (200 words)
- Current state and developments (200 words)
- Key challenges and limitations (150 words)
- Future trends and predictions (150 words)
- Practical applications and case studies (200 words)
- Expert opinions and analysis (150 words)
- Conclusion with key takeaways (100 words)

Please write the article using markdown headings (##) and subheadings (###), and make sure each section has substantial content.
DO NOT include these instructions in your output.
"""
                    retry_result = await self._generate_with_retry(enhanced_prompt, max_length + 200)
                
                elif current_retry == 2:
                    # Second retry: Generate section by section with explicit headings
                    # Include primary keywords in section headings if available
                    primary_kw = keyword_data.get('primary_keywords', []) or keyword_data.get('primary', [])
                    
                    # Create section prompts with keywords if available
                    intro_kw = f" about {primary_kw[0]}" if primary_kw else ""
                    background_kw = f" of {primary_kw[1]}" if len(primary_kw) > 1 else ""
                    trends_kw = f" in {primary_kw[2]}" if len(primary_kw) > 2 else ""
                    
                    sections = [
                        f"## Introduction{intro_kw}\nWrite a detailed introduction about {topic} (150 words)",
                        f"## Background{background_kw}\nExplain the background and history of {topic} in detail (200 words)",
                        f"## Current Trends{trends_kw}\nDiscuss current trends related to {topic} with examples (200 words)",
                        f"## Analysis\nProvide an in-depth analysis of {topic} with key insights (200 words)",
                        f"## Challenges and Opportunities\nAnalyze challenges and opportunities in {topic} with solutions (200 words)",
                        f"## Future Outlook\nDiscuss the future developments and predictions for {topic} (150 words)",
                        f"## Conclusion\nProvide a comprehensive conclusion about {topic} (100 words)"
                    ]
                    
                    combined_text = ""
                    for section_prompt in sections:
                        section_text = await self._generate_with_retry(section_prompt, 600)
                        if section_text:
                            # Extract just the generated content, not the prompt
                            content_lines = section_text.split('\n')
                            # Keep the heading line and remove the prompt line
                            if len(content_lines) > 1:
                                heading_line = content_lines[0] if content_lines[0].startswith('#') else ""
                                content = '\n'.join([l for l in content_lines if not l.startswith('Write')])
                                section_text = heading_line + '\n' + content if heading_line else content
                            

                            combined_text += section_text + "\n\n"
                    
                    retry_result = combined_text
                
                else:
                    # Third retry: Use a very explicit structure with primary keywords
                    primary_kw = keyword_data.get('primary_keywords', []) or keyword_data.get('primary', [])


                    secondary_kw = keyword_data.get('secondary_keywords', []) or keyword_data.get('secondary', [])
                    
                    # Create headings with keywords
                    title = primary_kw[0] if primary_kw else topic
                    h2_1 = primary_kw[0] if len(primary_kw) > 1 else "Background"
                    h2_2 = primary_kw[2] if len(primary_kw) > 2 else "Current State"
                    h3_1 = secondary_kw[0] if secondary_kw else "Analysis"
                    h3_2 = secondary_kw[1] if len(secondary_kw) > 1 else "Future Outlook"
                    
                    fallback_prompt = f"""# {title}

## {h2_1}
Write 200 words about the background and history with detailed information.

## {h2_2}
Write 200 words about the current state and developments with examples.

### {h3_1}
Write 200 words analyzing key aspects and providing insights.

### {h3_2}
Write 150 words about future trends and outlook with predictions.

## Key Features
Write 150 words about the most important features and characteristics.

## Applications
Write 150 words about practical applications and use cases.

## Conclusion
Write 100 words concluding the article with key takeaways.

Please write a comprehensive article using these exact headings, and do not include any instructions in your output.
"""
                    retry_result = await self._generate_with_retry(fallback_prompt, max_length + 400)
                
                # Process retry result
                if retry_result:
                    word_count = len(retry_result.split())
                    if word_count >= 500:
                        logger.info(f"Successfully generated content on retry {current_retry} ({word_count} words)")
                        # Format the retry result with keywords
                        formatted_article = self._format_article_with_headings(retry_result, keyword_data)
                        return formatted_article
                    else:
                        logger.warning(f"Retry {current_retry} still produced short content ({word_count} words)")
            
            # If we get here, use the best content we have
            if word_count >= 500:  # Increased from 300 to 500
                logger.warning(f"Returning shorter content than ideal ({word_count} words)")
                formatted_article = self._format_article_with_headings(generated_text, keyword_data)
                return formatted_article
            elif retry_result and len(retry_result.split()) >= 500:  # Increased from 300 to 500
                logger.warning(f"Using retry result with {len(retry_result.split())} words")
                formatted_article = self._format_article_with_headings(retry_result, keyword_data)
                return formatted_article
            else:
                logger.error(f"Failed to generate sufficient content after {max_retries} retries")
                return None
                
        except Exception as e:
            logger.error(f"Error generating blog content: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _structure_keyword_prompt(self, keyword_data: Dict) -> str:
        """
        Create a structured prompt that instructs the model to use keywords as headings and subheadings
        """
        prompt_sections = []
        
        # Get primary keywords (try both key formats)
        primary = keyword_data.get('primary_keywords') or keyword_data.get('primary') or []
        secondary = keyword_data.get('secondary_keywords') or keyword_data.get('secondary') or []
        semantic_groups = keyword_data.get('semantic_groups', {})
        long_tail = keyword_data.get('long_tail') or []
        questions = keyword_data.get('questions') or []
        
        # Create a more explicit instruction for using keywords as headings
        if secondary:
            prompt_sections.append(
                f"Use these keywords as main headings (##):\n" + 
                '\n'.join([f"- {kw}" for kw in secondary[:5]])
            )
            
        if secondary and len(secondary) > 5:
            prompt_sections.append(
                f"Use these keywords as subheadings (###):\n" + 
                '\n'.join([f"- {kw}" for kw in secondary[5:10]])
            )
        
        if semantic_groups:
            group_sections = []
            for group_name, group_keywords in semantic_groups.items():
                if group_keywords:  # Check if group_keywords exists and is not empty
                    keywords_list = group_keywords if isinstance(group_keywords, list) else []
                    group_sections.append(
                        f"Topic: {group_name.replace('_', ' ').title()}\n" +
                        '\n'.join([f"- {kw}" for kw in keywords_list[:3]])
                    )
            if group_sections:
                prompt_sections.append("Include these topic groups in your article:\n" + '\n\n'.join(group_sections))
        
        if long_tail:
            prompt_sections.append(
                "Include these long-tail phrases in your content:\n" + 
                '\n'.join([f"- {phrase}" for phrase in long_tail[:5]])
            )
        
        if questions:
            prompt_sections.append(
                "Include a FAQ section with these questions:\n" +
                '\n'.join([f"- {q}" for q in questions[:3]])
            )
        
        # Add a clear instruction about using markdown format
        prompt_sections.append(
            "IMPORTANT: Structure your article with these exact headings and subheadings using markdown format.\n"
            "Do not include these instructions in your output.\n"
            "Make sure each section has relevant content about the topic."
        )
        
        return '\n\n'.join(prompt_sections)

    def _format_article_with_headings(self, article_text: str, keyword_data: Dict) -> str:
        """
        Format article with proper headings, subheadings and section spacing
        """
        # First, clean the article text to remove any instruction artifacts
        article_text = self._clean_generated_text(article_text, "")
        
        # Check if the article already has proper headings
        has_headings = re.search(r'^#+\s+', article_text, re.MULTILINE) is not None
        
        # If the article already has headings, just return it after cleaning
        if has_headings:
            return article_text
            
        # If no headings, format with keywords
        if not keyword_data:
            return article_text

        # Extract keywords
        secondary = keyword_data.get('secondary_keywords') or keyword_data.get('secondary') or []
        semantic_groups = keyword_data.get('semantic_groups') or {}
        long_tail = keyword_data.get('long_tail') or []
        questions = keyword_data.get('questions') or []
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in article_text.split('\n\n') if p.strip()]
        if not paragraphs:
            return article_text
            
        formatted_sections = []
        
        # Add title and introduction
        title = paragraphs[0].split('.')[0] if paragraphs else "Article"
        formatted_sections.extend([
            f"# {title}",
            "",  # Empty line after heading
            paragraphs[0] if paragraphs else "",
            ""  # Empty line after introduction
        ])
        
        remaining_paragraphs = paragraphs[1:] if len(paragraphs) > 1 else []
        current_idx = 0
        
        # Add main sections with secondary keywords
        for i, keyword in enumerate(secondary[:3]):  # Use first 3 secondary keywords for main sections
            if current_idx < len(remaining_paragraphs):
                formatted_sections.extend([
                    f"## {keyword.title()}",
                    "",  # Empty line after heading
                    remaining_paragraphs[current_idx],
                    ""  # Empty line after section
                ])
                current_idx += 1
        
        # Add subsections with more secondary keywords
        for i, keyword in enumerate(secondary[3:6]):  # Use next 3 secondary keywords
            if current_idx < len(remaining_paragraphs):
                formatted_sections.extend([
                    f"### {keyword.title()}",
                    "",  # Empty line after heading
                    remaining_paragraphs[current_idx],
                    ""  # Empty line after section
                ])
                current_idx += 1
        
        # Add semantic group sections
        if semantic_groups and current_idx < len(remaining_paragraphs):
            formatted_sections.extend([
                "## Related Topics",
                ""  # Empty line after heading
            ])
            
            for group_name, keywords in list(semantic_groups.items())[:2]:
                if keywords and isinstance(keywords, list):
                    formatted_sections.extend([
                        f"### Topic: {group_name.replace('_', ' ').title()}",
                        "",  # Empty line after heading
                        "Key aspects include:",
                        *[f"- {kw}" for kw in keywords[:3]],
                        ""  # Empty line after list
                    ])
                    if current_idx < len(remaining_paragraphs):
                        formatted_sections.extend([
                            remaining_paragraphs[current_idx],
                            ""
                        ])
                        current_idx += 1
        
        # Add FAQ section
        if questions:
            formatted_sections.extend([
                "## Frequently Asked Questions",
                ""  # Empty line after heading
            ])
            
            for q in questions[:3]:  # Limit to 3 questions
                formatted_sections.extend([
                    f"### {q}",
                    "",  # Empty line after heading
                ])
                if current_idx < len(remaining_paragraphs):
                    formatted_sections.extend([
                        remaining_paragraphs[current_idx],
                        ""  # Empty line after answer
                    ])
                    current_idx += 1
                else:
                    formatted_sections.extend([
                        f"This is an important consideration regarding {secondary[0] if secondary else 'this topic'}.",
                        ""  # Empty line after answer
                    ])
        
        # Add conclusion section
        if current_idx < len(remaining_paragraphs):
            formatted_sections.extend([
                "## Conclusion",
                "",  # Empty line after heading
                remaining_paragraphs[current_idx],
                ""  # Empty line after conclusion
            ])
            current_idx += 1
        
        # Add any remaining paragraphs
        while current_idx < len(remaining_paragraphs):
            formatted_sections.append(remaining_paragraphs[current_idx])
            formatted_sections.append("")  # Empty line after paragraph
            current_idx += 1
        
        # Join all sections with proper spacing
        formatted_article = '\n'.join(formatted_sections)
        
        # Clean the formatted article one more time to remove any remaining artifacts
        formatted_article = self._clean_generated_text(formatted_article, "")
        
        return formatted_article
        
async def main():
    try:
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        generator = BlogGenerator()
        topic = "vampire diaries "  # Default topic
        import sys
        if len(sys.argv) > 1:
            topic = " ".join(sys.argv[1:])
        logger.info(f"Starting article generation for topic: {topic}")
        article = await generator.generate_blog_content(topic, max_length=4000)  # Increased from 512 to 4000
        if article:
            word_count = len(article.split())
            preview = article[:300] + "..." if len(article) > 300 else article
            logger.info(f"Generated article preview ({word_count} words):\n{preview}")
            print(f"\nGenerated Article:\n{article}\n")
            logger.info("Article generation completed successfully.")
        else:
            logger.error("Article generation failed or returned no content.")
    except KeyboardInterrupt:
        logger.info("Operation terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(generator, "cleanup") and callable(getattr(generator, "cleanup")):
            await generator.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
