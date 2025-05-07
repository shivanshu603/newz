import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from config import Config
from utils import logger

class ContentAnalyzer:
    def __init__(self):
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
            raise
        
    async def __aenter__(self):
        if not self.nlp:
            self.nlp = spacy.load("en_core_web_sm")
        return self

        
    async def __aexit__(self, exc_type, exc, tb):
        self.nlp = None
        if exc:
            logger.error(f"Error in ContentAnalyzer: {str(exc)}")
        return

        self.stop_words = set(stopwords.words('english'))
        self.quality_threshold = Config.BLOG_QUALITY_THRESHOLD

    def enhance_content(self, content: str) -> str:
        """Enhance content quality"""
        try:
            # Improve readability
            content = self._improve_readability(content)
            
            # Check and improve structure
            content = self._improve_structure(content)
            
            # Verify uniqueness
            if not self._check_uniqueness(content):
                content = self._make_unique(content)
                
            return content
            
        except Exception as e:
            logger.error(f"Error enhancing content: {str(e)}")
            return content

    def _improve_readability(self, content: str) -> str:
        """Improve content readability"""
        doc = self.nlp(content)
        sentences = [sent.text for sent in doc.sents]
        
        # Ensure sentence length variation
        sentences = [self._adjust_sentence_length(sent) for sent in sentences]
        
        return ' '.join(sentences)

    def _improve_structure(self, content: str) -> str:
        """Improve content structure"""
        # Add subheadings
        content = self._add_subheadings(content)
        
        # Add bullet points where appropriate
        content = self._add_bullet_points(content)
        
        return content

    def _check_uniqueness(self, content: str) -> bool:
        """Check content uniqueness"""
        # Compare with existing content
        return True  # Placeholder for uniqueness check

    def _make_unique(self, content: str) -> str:
        """Make content more unique"""
        # Implement uniqueness enhancement
        return content

    def _adjust_sentence_length(self, sentence: str) -> str:
        """Adjust sentence length for better readability"""
        if len(sentence.split()) > 25:
            return self._split_long_sentence(sentence)
        return sentence

    def _split_long_sentence(self, sentence: str) -> str:
        """Split long sentences into shorter ones"""
        doc = self.nlp(sentence)
        return ' '.join([sent.text for sent in doc.sents])

    def _add_subheadings(self, content: str) -> str:
        """Add subheadings to content"""
        # Implement subheading addition
        return content

    def _add_bullet_points(self, content: str) -> str:
        """Add bullet points where appropriate"""
        # Implement bullet point addition
        return content
