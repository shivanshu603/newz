from textblob import TextBlob
from typing import Dict, Tuple
import re
from utils import logger

class QualityMetrics:
    def __init__(self):
        # Use direct values instead of Config
        self.min_word_count = 300  # Minimum article length
        self.min_coherence_score = 0.3  # Coherence threshold
        self.min_readability_score = 30.0  # Readability threshold
        self.max_retries = 3  # Maximum generation attempts

    def evaluate_content(self, content: str) -> Tuple[bool, Dict]:
        """Evaluate content quality"""
        metrics = {
            'word_count': len(content.split()),
            'coherence_score': self._calculate_coherence(content),
            'readability_score': self._calculate_readability(content)
        }
        
        is_acceptable = (
            metrics['word_count'] >= self.min_word_count and
            metrics['coherence_score'] >= self.min_coherence_score and
            metrics['readability_score'] >= self.min_readability_score
        )
        
        return is_acceptable, metrics

    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence"""
        try:
            blob = TextBlob(text)
            sentences = blob.sentences
            if not sentences:
                return 0.0
            
            # Calculate sentence-to-sentence similarity
            scores = []
            for i in range(len(sentences) - 1):
                similarity = self._sentence_similarity(str(sentences[i]), str(sentences[i + 1]))
                scores.append(similarity)
            
            return sum(scores) / len(scores) if scores else 0.0
        except:
            return 0.0

    def _sentence_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two sentences"""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def _calculate_readability(self, text: str) -> float:
        """Calculate Flesch reading ease score"""
        try:
            blob = TextBlob(text)
            words = len(text.split())
            sentences = len(blob.sentences)
            syllables = sum(self._count_syllables(word) for word in text.split())
            
            if words == 0 or sentences == 0:
                return 0.0
                
            return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        except:
            return 0.0

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
            
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
            
        return count
