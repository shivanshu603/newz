import random
import re
from typing import List, Dict, Optional
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from utils import logger
import time

class ContentHumanizer:
    def __init__(self):
        """Initialize with fallback tokenization"""
        self.initialized = False
        try:
            # Initialize basic tokenization without punkt_tab dependency
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.initialized = True
        except Exception as e:
            logger.error(f"Error initializing ContentHumanizer: {e}")

    async def process_content(self, text: str) -> Optional[str]:
        """Process and humanize content with fallback methods"""
        try:
            if not text:
                return None
                
            # Basic sentence and paragraph handling if NLTK initialization failed
            if not self.initialized:
                return self._basic_text_processing(text)

            # Enhanced processing with NLTK when available
            processed_text = self._enhance_readability(text)
            processed_text = self._improve_structure(processed_text)
            processed_text = self._fix_formatting(processed_text)
            
            return processed_text

        except Exception as e:
            logger.error(f"Error humanizing text: {e}")
            # Fallback to basic processing
            return self._basic_text_processing(text)

    def _basic_text_processing(self, text: str) -> str:
        """Basic text processing without NLTK dependencies"""
        try:
            # Basic cleanup
            text = text.strip()
            
            # Fix spacing
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
            
            # Fix basic punctuation
            text = re.sub(r'\s+([.,!?])', r'\1', text)
            text = re.sub(r'([.,!?])([A-Z])', r'\1 \2', text)
            
            # Add paragraph breaks
            sentences = text.split('. ')
            paragraphs = []
            current_para = []
            
            for sentence in sentences:
                current_para.append(sentence)
                if len(current_para) >= 3:  # Group ~3 sentences per paragraph
                    paragraphs.append('. '.join(current_para) + '.')
                    current_para = []
                    
            if current_para:
                paragraphs.append('. '.join(current_para) + '.')
                
            return '\n\n'.join(paragraphs)

        except Exception as e:
            logger.error(f"Error in basic text processing: {e}")
            return text

    def _enhance_readability(self, text: str) -> str:
        """Enhance text readability"""
        try:
            # Break up long sentences
            sentences = text.split('. ')
            improved = []
            
            for sentence in sentences:
                if len(sentence.split()) > 25:  # Break up sentences longer than 25 words
                    parts = sentence.split(', ')
                    if len(parts) > 1:
                        improved.extend([p.strip() + '.' for p in parts])
                    else:
                        improved.append(sentence + '.')
                else:
                    improved.append(sentence + '.')
                    
            return ' '.join(improved)
            
        except Exception as e:
            logger.error(f"Error enhancing readability: {e}")
            return text

    def _improve_structure(self, text: str) -> str:
        """Improve text structure with proper markdown formatting"""
        try:
            # Ensure proper heading structure
            lines = text.split('\n')
            structured_lines = []
            in_list = False
            
            for line in lines:
                # Fix heading formatting
                if re.match(r'^#{1,6}\s', line):
                    line = re.sub(r'^(#{1,6})\s*(.+)', r'\1 \2', line)
                
                # Fix list formatting
                if re.match(r'^\s*[-*]\s', line):
                    if not in_list:
                        structured_lines.append('')  # Add space before list
                    in_list = True
                    line = re.sub(r'^\s*([-*])\s*', r'* ', line)
                else:
                    if in_list:
                        structured_lines.append('')  # Add space after list
                    in_list = False
                
                structured_lines.append(line)
            
            return '\n'.join(structured_lines)
            
        except Exception as e:
            logger.error(f"Error improving structure: {e}")
            return text

    def _fix_formatting(self, text: str) -> str:
        """Fix formatting while preserving markdown structure"""
        try:
            # Fix markdown formatting
            text = re.sub(r'(\*\*|__)\s+', r'\1', text)  # Fix bold
            text = re.sub(r'\s+(\*\*|__)', r'\1', text)
            text = re.sub(r'(\*|_)\s+', r'\1', text)  # Fix italic
            text = re.sub(r'\s+(\*|_)', r'\1', text)
            
            # Handle markdown structure
            lines = text.split('\n')
            formatted_lines = []
            prev_was_heading = False
            
            for line in lines:
                is_heading = re.match(r'^#{1,6}\s', line)
                
                # Add spacing before headings
                if is_heading and formatted_lines and not prev_was_heading:
                    formatted_lines.append('')
                
                # Fix heading formatting
                if is_heading:
                    line = re.sub(r'^(#{1,6})\s*(.+?)\s*$', r'\1 \2', line)
                    formatted_lines.append(line)
                    formatted_lines.append('')  # Add space after heading
                else:
                    # Fix list formatting
                    if re.match(r'^\s*[-*]\s+', line):
                        line = re.sub(r'^\s*[-*]\s+', '* ', line)
                    formatted_lines.append(line)
                
                prev_was_heading = is_heading
            
            text = '\n'.join(formatted_lines)
            
            # Fix empty lines and spacing
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'([.!?])([A-Z])', r'\1\n\n\2', text)  # Add paragraph breaks
            
            return text
            
        except Exception as e:
            logger.error(f"Error fixing formatting: {e}")
            return text

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Nothing to clean up currently
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def humanize(self, content: str, topic: str = None) -> str:
        """Humanize content while preserving markdown structure"""
        try:
            # Split content into sections by headings
            sections = []
            current_section = []
            lines = content.split('\n')
            
            for line in lines:
                # If this is a heading, start a new section
                if re.match(r'^#{1,6}\s', line):
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = [line]  # Start new section with heading
                else:
                    current_section.append(line)
            
            # Add the last section
            if current_section:
                sections.append('\n'.join(current_section))
            
            # Process each section while preserving headings
            humanized_sections = []
            doc_state = {
                "current_paragraph": 0,
                "topic": topic
            }
            
            for section in sections:
                section_lines = section.split('\n')
                humanized_lines = []
                
                for line in section_lines:
                    # Preserve headings
                    if re.match(r'^#{1,6}\s', line):
                        humanized_lines.append(line)
                        humanized_lines.append('')  # Add space after heading
                        continue
                    
                    # Skip empty lines
                    if not line.strip():
                        humanized_lines.append(line)
                        continue
                    
                    # Process regular text
                    processed = line
                    processed = self._vary_sentence_structure(processed)
                    processed = self._add_natural_elements(processed, doc_state)
                    processed = self._adjust_formality(processed)
                    processed = self._add_transitions(processed, doc_state["current_paragraph"], len(sections))
                    processed = self._add_burstiness(processed)
                    processed = self._inject_colloquialisms(processed, doc_state)
                    processed = self._add_personal_touch(processed, doc_state["current_paragraph"], doc_state)
                    processed = self._add_sensory_details(processed)
                    processed = self._add_minor_imperfections(processed)
                    
                    humanized_lines.append(processed)
                    doc_state["current_paragraph"] += 1
                
                if humanized_lines:
                    humanized_sections.append('\n'.join(humanized_lines))
            
            # Join sections with proper spacing
            humanized = '\n\n'.join(humanized_sections)
            
            # Clean up final formatting while preserving markdown
            humanized = self._clean_artifacts(humanized)
            humanized = self._fix_formatting(humanized)
            
            return humanized.strip()
            
        except Exception as e:
            logger.error(f"Error in humanize: {e}")
            return content

    def _clean_artifacts(self, content: str) -> str:
        """Clean up generation artifacts while preserving markdown"""
        try:
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Preserve markdown headings
                if re.match(r'^#{1,6}\s', line):
                    cleaned_lines.append(line)
                    continue
                    
                # Clean up regular text
                cleaned = line.strip()
                if cleaned:
                    # Remove common artifacts
                    cleaned = re.sub(r'(?i)(?:honestly|you know|look|i think),?\s*', '', cleaned)
                    cleaned = re.sub(r'(?i)(?:without a doubt|to my complete horror|when pigs fly),?\s*', '', cleaned)
                    cleaned = re.sub(r'http:www\S*', '', cleaned)
                    cleaned = re.sub(r'\s+', ' ', cleaned)
                    
                    if cleaned:
                        cleaned_lines.append(cleaned)
                else:
                    cleaned_lines.append(line)  # Preserve empty lines
            
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            logger.error(f"Error cleaning artifacts: {e}")
            return content
