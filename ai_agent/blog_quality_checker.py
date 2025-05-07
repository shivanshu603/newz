from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from models import Blog
from config import Config
from utils import logger
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BlogQualityChecker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.min_quality_score = Config.BLOG_QUALITY_THRESHOLD

    def is_quality_acceptable(self, content: str) -> bool:
        """Check if blog content meets quality standards."""
        return self.check_blog_quality(content)

    def check_blog_quality(self, content: str) -> bool:
        """Check if blog content meets quality standards including human-like writing, structure, and engagement"""
        try:

            
            # Check basic metrics
            if not self._check_word_count(content, Config.BLOG_MIN_WORDS):
                logger.info("Blog post failed word count check")
                return False
                
            # Check readability
            if not self._check_readability(content):
                logger.info("Blog post failed readability check")
                return False
                
            # Check uniqueness
            if not self._check_uniqueness(content):
                logger.info("Blog post failed uniqueness check")
                return False
                
            # Check structure and engagement
            if not self._check_structure(content):
                logger.info("Blog post failed structure check")
                return False
                
            if not self._check_engagement(content):
                logger.info("Blog post failed engagement check")
                return False
                
            logger.info("Blog post passed all quality checks")
            return True

        except Exception as e:
            logger.error(f"Error checking blog quality: {str(e)}")
            return False

    def _check_word_count(self, content: str, min_words: int) -> bool:
        """Verify the blog meets minimum word count"""
        word_count = len(content.split())
        logger.debug(f"Word count: {word_count} (minimum: {min_words})")
        if word_count >= min_words:
            logger.debug("Word count check passed.")
            return True
        else:
            logger.debug("Word count check failed.")
            return False

    def _check_structure(self, content: str) -> bool:
        """Check blog structure including paragraphs, headings, and lists"""
        # Check for proper paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            logger.debug("Insufficient number of paragraphs")
            return False
            
        # Check for bullet points or numbered lists
        has_lists = any(line.strip().startswith(('- ', '* ', '1. ', '• ')) 
                      for line in content.split('\n'))
        if not has_lists:
            logger.debug("No bullet points or numbered lists found")
            return False
            
        # Check for question-and-answer format
        has_questions = any(line.strip().endswith('?') 
                           for line in content.split('\n'))
        if not has_questions:
            logger.debug("No question-and-answer format detected")
            return False
            
        return True

    def _check_engagement(self, content: str) -> bool:
        """Check blog engagement metrics"""
        # Check for natural language patterns
        doc = self.nlp(content)
        pos_tags = [token.pos_ for token in doc]
        verb_count = pos_tags.count('VERB')
        adj_count = pos_tags.count('ADJ')
        
        if verb_count < 5 or adj_count < 3:

            logger.debug("Insufficient verbs or adjectives for natural language")
            return False
            
        # Check for topic initiation
        sentences = [sent.text for sent in doc.sents]
        topic_intros = sum(1 for sent in sentences 
                         if any(word in sent.lower() 
                              for word in ['introduction', 'overview', 'what is']))
        if topic_intros < 1:
            logger.debug("No clear topic introduction found")
            return False
            
        # Check for engagement phrases
        engagement_phrases = ['you might', 'consider this', 'important to', 'key aspect']
        has_engagement = any(phrase in content.lower() 
                              for phrase in engagement_phrases)
        if not has_engagement:
            logger.debug("Insufficient engagement phrases found")
            return False
            
        return True

    def _check_readability(self, content: str) -> bool:
        """Check the blog's readability score"""
        doc = self.nlp(content)
        # Calculate Flesch-Kincaid grade level
        num_sentences = len(list(doc.sents))
        num_words = len(list(doc))
        num_syllables = sum([self._count_syllables(word.text) for word in doc])
        
        if num_sentences == 0 or num_words == 0:
            logger.debug("Blog has no sentences or words")
            return False
            
        readability = 206.835 - (1.015 * (num_words / num_sentences)) - (84.6 * (num_syllables / num_words))
        logger.debug(f"Blog readability score: {readability} (minimum: 60)")
        return readability >= 50  # Minimum readability score


    def _check_uniqueness(self, content: str) -> bool:
        """Check if the blog content is unique"""
        # Compare with existing blogs
        existing_blogs = self._get_recent_blogs()
        if not existing_blogs:
            logger.debug("No existing blogs found for uniqueness comparison")
            return True
            
        # Calculate similarity scores
        vectorizer = TfidfVectorizer()
        texts = [content] + [b.content for b in existing_blogs]
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Log similarity scores
        logger.debug(f"Blog uniqueness similarity scores: {similarity_scores[0]}")
        
        # Check if any similarity score is too high
        return all(score < 0.8 for score in similarity_scores[0])


    def _get_recent_blogs(self) -> List[Blog]:
        """Get recently generated blogs for comparison"""
        from models import Session  # Import Session from models
        session = Session()
        try:
            blogs = session.query(Blog).order_by(Blog.generated_at.desc()).limit(10).all()
            session.commit()
            return blogs
        except Exception as e:
            session.rollback()
            logger.error(f"Error fetching recent blogs: {str(e)}")
            return []
        finally:
            session.close()

    def _count_syllables(self, word: str) -> bool:
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count += 1
        return count

    def analyze_quality(self, content: str) -> dict:
        """Analyze the quality of blog content"""
        try:
            doc = self.nlp(content)
            
            # Calculate readability
            num_sentences = len(list(doc.sents))
            num_words = len(list(doc))
            num_syllables = sum([self._count_syllables(word.text) for word in doc])
            readability = 206.835 - (1.015 * (num_words / num_sentences)) - (84.6 * (num_syllables / num_words)) if num_sentences > 0 else 0
            
            # Calculate uniqueness
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([content])
            uniqueness = tfidf_matrix.mean()
            
            return {
                'readability_score': readability,
                'uniqueness_score': float(uniqueness),
                'word_count': num_words,
                'sentence_count': num_sentences,
                'syllable_count': num_syllables
            }
        except Exception as e:
            logger.error(f"Error analyzing quality: {str(e)}")
            return {
                'readability_score': 0,
                'uniqueness_score': 0,
                'word_count': 0,
                'sentence_count': 0,
                'syllable_count': 0
            }

def check_quality(content: str) -> bool:
    """Check blog content quality using BlogQualityChecker"""
    checker = BlogQualityChecker()
    return checker.check_blog_quality(content)


if __name__ == "__main__":
    checker = BlogQualityChecker()
    # Test with a sample blog
    test_content = {
        'title': "Test Blog",
        'content': "This is a test blog content to check the quality checker functionality.",
        'topic': "Testing"
    }
    quality_ok = checker.check_blog_quality(test_content)
    print(f"Blog quality check result: {quality_ok}")
