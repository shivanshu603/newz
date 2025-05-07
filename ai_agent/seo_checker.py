from typing import Dict, List, Optional, Iterator, Any
from functools import cached_property
from textblob import TextBlob
import re
from utils import logger
from bs4 import BeautifulSoup

class SEOChecker:
    def __init__(self):
        self.min_word_count = 600
        self.max_word_count = 2500
        self.optimal_keyword_density = 0.02  # 2%
        self.min_headings = 3
        
        self.readability_targets = {
            'min_score': 60,  # Flesch reading ease
            'max_sentence_length': 20,
            'max_paragraph_length': 150
        }
        
        self.meta_requirements = {
            'title_length': (40, 60),
            'description_length': (140, 160),
            'min_keywords': 5
        }
        
        self._cache: Dict[str, Any] = {}

    def check_article_seo(self, content: str, meta: Dict) -> Dict:
        """Comprehensive SEO check of article content and metadata"""
        try:
            seo_score = 0
            issues = []
            recommendations = []

            # Content checks
            content_stats = self._analyze_content(content)
            seo_score += self._score_content(content_stats)
            
            # Check headings
            heading_score, heading_issues = self._check_headings(content)
            seo_score += heading_score
            issues.extend(heading_issues)
            
            # Check keyword optimization
            keyword_score, keyword_issues = self._check_keywords(content, meta.get('keywords', []))
            seo_score += keyword_score
            issues.extend(keyword_issues)
            
            # Check readability
            readability_score, readability_issues = self._check_readability(content)
            seo_score += readability_score
            issues.extend(readability_issues)
            
            # Check meta tags
            meta_score, meta_issues = self._check_meta_tags(meta)
            seo_score += meta_score
            issues.extend(meta_issues)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(content_stats, issues)

            return {
                'score': min(100, seo_score),
                'issues': issues,
                'recommendations': recommendations,
                'stats': content_stats
            }

        except Exception as e:
            logger.error(f"Error in SEO check: {e}")
            return {
                'score': 0,
                'issues': ['Error performing SEO check'],
                'recommendations': [],
                'stats': {}
            }

    def _analyze_content(self, content: str) -> Dict:
        """Analyze content statistics"""
        try:
            # Clean HTML tags
            text = BeautifulSoup(content, 'html.parser').get_text()
            
            # Basic stats
            words = text.split()
            sentences = TextBlob(text).sentences
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            
            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'paragraph_count': len(paragraphs),
                'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
                'avg_paragraph_length': len(words) / len(paragraphs) if paragraphs else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {}

    def _score_content(self, stats: Dict) -> float:
        """Score content based on statistics"""
        score = 0
        
        # Word count score (0-25)
        word_count = stats.get('word_count', 0)
        if word_count >= self.min_word_count:
            score += min(25, (word_count / self.min_word_count) * 20)
            
        # Sentence length score (0-25)
        avg_sentence_length = stats.get('avg_sentence_length', 0)
        if 10 <= avg_sentence_length <= 20:
            score += 25
        elif avg_sentence_length < 10:
            score += 15
        else:
            score += max(0, 25 - (avg_sentence_length - 20))
            
        # Paragraph length score (0-25)
        avg_paragraph_length = stats.get('avg_paragraph_length', 0)
        if avg_paragraph_length <= 150:
            score += 25
        else:
            score += max(0, 25 - ((avg_paragraph_length - 150) / 10))
            
        return score

    def _check_headings(self, content: str) -> tuple[float, List[str]]:
        """Check heading structure and hierarchy"""
        score = 0
        issues = []
        
        soup = BeautifulSoup(content, 'html.parser')
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        # Check number of headings
        if len(headings) < self.min_headings:
            issues.append(f"Too few headings (found {len(headings)}, minimum {self.min_headings})")
        else:
            score += 20
            
        # Check heading hierarchy
        last_level = 0
        for heading in headings:
            if heading and heading.name:
                current_level = int(heading.name[1])
                if current_level - last_level > 1:
                    issues.append(f"Skipped heading level (from h{last_level} to h{current_level})")
                last_level = current_level
            
        if not issues:
            score += 20
            
        return score, issues

    def _check_keywords(self, content: str, keywords: Optional[List[str]]) -> tuple[float, List[str]]:
        """Check keyword optimization with proper null checks"""
        score = 0
        issues = []

        if not keywords:
            keywords = []

        # Clean content
        text = BeautifulSoup(content, 'html.parser').get_text().lower()
        words = text.split()
        
        for keyword in keywords:
            keyword = keyword.lower()
            count = text.count(keyword)
            density = count / len(words) if words else 0
            
            if density > self.optimal_keyword_density * 1.5:
                issues.append(f"Keyword '{keyword}' appears too frequently")
            elif density < self.optimal_keyword_density * 0.5:
                issues.append(f"Keyword '{keyword}' appears too rarely")
            else:
                score += 10
                
        return min(40, score), issues

    def _check_readability(self, content: str) -> tuple[float, List[str]]:
        """Check content readability"""
        score = 0
        issues = []
        
        text = BeautifulSoup(content, 'html.parser').get_text()
        blob = TextBlob(text)
        
        # Calculate Flesch reading ease
        word_count = len(text.split())
        sentence_count = len(blob.sentences)
        syllable_count = sum(self._count_syllables(word) for word in text.split())
        
        if sentence_count > 0 and word_count > 0:
            flesch_score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
            
            if flesch_score < self.readability_targets['min_score']:
                issues.append(f"Content might be too difficult to read (score: {flesch_score:.1f})")
            else:
                score += 20
                
        # Check sentence lengths
        sentences = getattr(blob, 'sentences', [])
        if sentences:
            long_sentences = [s for s in sentences if len(getattr(s, 'words', [])) > self.readability_targets['max_sentence_length']]
            if long_sentences:
                issues.append(f"Found {len(long_sentences)} sentences that are too long")
            else:
                score += 20
        else:
            score += 20  # No sentences to check, assume it's fine
            
        return score, issues

    def _check_meta_tags(self, meta: Dict) -> tuple[float, List[str]]:
        """Check meta tag optimization"""
        score = 0
        issues = []
        
        if not meta:
            return 0, ["Missing meta data"]
        
        # Check title length
        title = meta.get('title', '')
        min_title, max_title = self.meta_requirements['title_length']
        if not min_title <= len(title) <= max_title:
            issues.append(f"Title length should be between {min_title} and {max_title} characters")
        else:
            score += 20
            
        # Check description length
        description = meta.get('description', '')
        min_desc, max_desc = self.meta_requirements['description_length']
        if not min_desc <= len(description) <= max_desc:
            issues.append(f"Description length should be between {min_desc} and {max_desc} characters")
        else:
            score += 20
            
        # Check keywords
        keywords = meta.get('keywords', [])
        if len(keywords) < self.meta_requirements['min_keywords']:
            issues.append(f"Too few keywords (found {len(keywords)}, minimum {self.meta_requirements['min_keywords']})")
        else:
            score += 20
            
        return score, issues

    def _generate_recommendations(self, stats: Dict, issues: List[str]) -> List[str]:
        """Generate SEO improvement recommendations"""
        recommendations = []
        
        # Word count recommendations
        word_count = stats.get('word_count', 0)
        if word_count < self.min_word_count:
            recommendations.append(f"Add more content to reach at least {self.min_word_count} words")
        elif word_count > self.max_word_count:
            recommendations.append(f"Consider splitting content into multiple articles")
            
        # Extract action items from issues
        for issue in issues:
            if "too few" in issue.lower():
                recommendations.append(f"Add more {issue.split()[2]}")
            elif "too long" in issue.lower():
                recommendations.append(f"Shorten {issue.split()[1]}")
                
        return recommendations

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_char_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_is_vowel:
                count += 1
            prev_char_is_vowel = is_vowel
            
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2:
            count += 1
        if count == 0:
            count = 1
            
        return count

    def _cache_results(self, key: str, data: Dict) -> None:
        """Cache results safely"""
        if not hasattr(self, '_cache'):
            self._cache = {}
        if isinstance(self._cache, dict):
            self._cache[key] = data

    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached results safely"""
        if not hasattr(self, '_cache') or not isinstance(self._cache, dict):
            return None
        return self._cache.get(key)

    def _get_cached_property(self, name: str) -> Optional[Dict]:
        """Safely get cached property"""
        if not hasattr(self, '_cache'):
            return None
        cache = getattr(self, '_cache', {})
        if not isinstance(cache, dict):
            return None
        return cache.get(name)

    def _check_cached_property(self, prop_name: str) -> bool:
        """Check cached property safely"""
        if not hasattr(self, '_cache'):
            return False
        cache = getattr(self, '_cache', {})
        if not isinstance(cache, dict):
            return False
        return bool(cache.get(prop_name))

    def _cache_property_iter(self) -> Iterator[str]:
        """Make cached_property iterable"""
        if hasattr(self, '_cache'):
            yield from self._cache.keys()
        else:
            return iter(())

    # Allow iteration over cached properties
    def __iter__(self) -> Iterator[str]:
        return self._cache_property_iter()
