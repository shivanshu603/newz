from typing import Dict, List, Optional
import re
from utils import logger

class ContentFormatter:
    def format_article(self, content: str, seo_keywords: Dict = None) -> str:
        """Format an article for optimal readability and structure"""
        try:
            # Normalize line endings
            content = content.replace('\r\n', '\n')
            
            # Ensure proper heading formatting with emphasis
            content = self._format_headings_with_emphasis(content)
            
            # Format paragraphs and spacing
            content = self._improve_paragraphs(content)
            
            # Add SEO keywords if provided
            if seo_keywords:
                content = self._integrate_keywords(content, seo_keywords)
            
            return content.strip()
        except Exception as e:
            logger.error(f"Error formatting content: {e}")
            return content

    def _format_headings_with_emphasis(self, content: str) -> str:
        """Format headings with proper emphasis and spacing"""
        try:
            lines = content.split('\n')
            formatted_lines = []
            last_heading_level = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    formatted_lines.append(line)
                    continue
                    
                # Check if line is a heading
                heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
                if heading_match:
                    level = len(heading_match.group(1))
                    text = heading_match.group(2).strip()
                    
                    # Add extra line break before higher-level headings
                    if level <= last_heading_level and formatted_lines:
                        formatted_lines.append('')
                    
                    # Format heading with proper emphasis
                    if level == 1:
                        formatted_lines.extend([
                            '',
                            f"# {text.upper()}",  # Main title in uppercase
                            ''
                        ])
                    elif level == 2:
                        formatted_lines.extend([
                            '',
                            f"## {text.title()}",  # Section headers in title case
                            ''
                        ])
                    else:
                        formatted_lines.extend([
                            '',
                            f"{'#' * level} {text.capitalize()}",  # Subsections capitalized
                            ''
                        ])
                    
                    last_heading_level = level
                else:
                    formatted_lines.append(line)
            
            return '\n'.join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error formatting headings: {e}")
            return content

    def _improve_paragraphs(self, content: str) -> str:
        """Improve paragraph structure and readability"""
        try:
            sections = re.split(r'\n(#{1,6}\s+[^\n]+)', content)
            formatted_sections = []
            
            for i, section in enumerate(sections):
                if i % 2 == 0:  # Content section
                    # Split into paragraphs
                    paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
                    formatted_paragraphs = []
                    
                    for paragraph in paragraphs:
                        if paragraph.startswith(('- ', '* ')):
                            # Preserve list formatting
                            formatted_paragraphs.append(paragraph)
                        else:
                            # Format paragraph text
                            sentences = [s.strip() for s in paragraph.split('. ')]
                            formatted_sentences = []
                            
                            for sentence in sentences:
                                if sentence:
                                    # Ensure proper capitalization and punctuation
                                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence
                                    if not sentence[-1] in '.!?':
                                        sentence += '.'
                                    formatted_sentences.append(sentence)
                            
                            formatted_paragraphs.append('. '.join(formatted_sentences))
                    
                    formatted_sections.append('\n\n'.join(formatted_paragraphs))
                else:  # Heading section
                    formatted_sections.append(section)
            
            return '\n\n'.join(formatted_sections)
            
        except Exception as e:
            logger.error(f"Error improving paragraphs: {e}")
            return content

    def _integrate_keywords(self, content: str, seo_keywords: Dict) -> str:
        """Integrate SEO keywords naturally into the content"""
        try:
            # Insert primary keywords into first paragraph if missing
            primary_keywords = seo_keywords.get('primary', [])
            if primary_keywords:
                paragraphs = content.split('\n\n')
                if paragraphs:
                    first_para = paragraphs[0]
                    for kw in primary_keywords:
                        if kw.lower() not in first_para.lower():
                            first_para += f" {kw}"
                    paragraphs[0] = first_para
                    content = '\n\n'.join(paragraphs)
            
            # Add secondary keywords in subheadings if missing
            secondary_keywords = seo_keywords.get('secondary', [])
            for kw in secondary_keywords:
                if kw.lower() not in content.lower():
                    content += f"\n\n## {kw}\nDetails about {kw}."
            
            return content
        except Exception as e:
            logger.error(f"Error integrating keywords: {e}")
            return content


    def _clean_content(self, content: str) -> str:
        """Remove unwanted artifacts and clean up content"""
        # Remove random number sequences
        content = re.sub(r'\b\d+(?:\s+\d+)*\b', '', content)
        
        # Remove URLs and unnecessary links
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        
        # Remove common spam-like phrases
        spam_phrases = [
            "Click here to download",
            "Subscribe To Our Newsletter",
            "Please consider supporting us",
            "Download Now",
            "Free View In iTunes"
        ]
        for phrase in spam_phrases:
            content = content.replace(phrase, '')
        
        return content.strip()

    def _fix_formatting(self, content: str) -> str:
        """Fix quotations, emphasis, and other formatting"""
        # Fix quote formatting
        content = re.sub(r'"([^"]*)"', r'"\1"', content)
        
        # Remove multiple punctuation
        content = re.sub(r'([.!?]){2,}', r'\1', content)
        
        # Fix spacing around punctuation
        content = re.sub(r'\s+([.,!?])', r'\1', content)
        
        # Remove random capitalization
        content = re.sub(r'\b[A-Z]+\b', lambda m: m.group(0).capitalize(), content)
        
        return content
