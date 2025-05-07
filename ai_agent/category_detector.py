from typing import List, Dict, Set
from utils import logger
import re

class CategoryDetector:
    def __init__(self):
        # Define category mappings (customize based on your WordPress categories)
        self.category_mappings = {
            'technology': 2,
            'business': 3,
            'health': 4,
            'science': 5,
            'entertainment': 6,
            'sports': 7,
            'politics': 8,
            'world': 9,
            'finance': 10
        }
        
        # Define keyword sets for each category as dictionaries
        self.category_keywords = {
            'technology': {
                'keywords': [
                    'tech', 'software', 'app', 'digital', 'ai', 'cyber', 'computer', 
                    'mobile', 'internet', 'google', 'apple', 'microsoft', 'android',
                    'artificial intelligence', 'machine learning', 'blockchain', 'cloud',
                    'programming', 'developer', 'code', 'startup', 'innovation'
                ]
            },
            'business': {
                'keywords': [
                    'business', 'economy', 'market', 'finance', 'industry', 'company',
                    'startup', 'stock', 'trade', 'investment', 'ceo', 'corporate',
                    'enterprise', 'management', 'strategy', 'revenue', 'profit'
                ]
            },
            'health': {
                'keywords': [
                    'health', 'medical', 'healthcare', 'disease', 'treatment', 'medicine',
                    'wellness', 'hospital', 'doctor', 'patient', 'therapy', 'mental health',
                    'diet', 'nutrition', 'fitness', 'vaccine', 'research'
                ]
            },
            'science': {
                'keywords': [
                    'science', 'research', 'study', 'discovery', 'scientific', 'physics',
                    'biology', 'chemistry', 'space', 'climate', 'environment', 'technology',
                    'innovation', 'experiment', 'laboratory', 'scientist'
                ]
            },
            'entertainment': {
                'keywords': [
                    'entertainment', 'movie', 'film', 'music', 'game', 'tv', 'show',
                    'celebrity', 'actor', 'actress', 'hollywood', 'series', 'streaming',
                    'media', 'arts', 'culture', 'performance'
                ]
            },
            'sports': {
                'keywords': [
                    'sports', 'game', 'player', 'team', 'match', 'tournament',
                    'championship', 'athletes', 'league', 'soccer', 'football',
                    'basketball', 'baseball', 'tennis', 'olympic'
                ]
            },
            'politics': {
                'keywords': [
                    'politics', 'government', 'election', 'political', 'president',
                    'minister', 'policy', 'congress', 'senate', 'law', 'legislation',
                    'democrat', 'republican', 'parliament', 'diplomatic'
                ]
            },
            'world': {
                'keywords': [
                    'world', 'international', 'global', 'country', 'nation', 'foreign',
                    'diplomatic', 'embassy', 'overseas', 'continent', 'europe', 'asia',
                    'africa', 'america', 'middle east', 'united nations'
                ]
            },
            'finance': {
                'keywords': [
                    'finance', 'banking', 'investment', 'stock market', 'trading',
                    'cryptocurrency', 'bitcoin', 'forex', 'economy', 'market',
                    'wall street', 'financial', 'money', 'debt', 'loan'
                ]
            }
        }

    def detect_categories(self, title: str, content: str = None, keywords: Dict = None) -> List[int]:
        """
        Detect appropriate categories based on title, content, and keywords
        Returns a list of category IDs
        """
        try:
            # Combine all text for analysis
            text_to_analyze = title.lower()
            if content:
                text_to_analyze += ' ' + content.lower()
            if keywords:
                # Add all types of keywords
                for key_type in ['primary_keywords', 'secondary_keywords', 'related_terms']:
                    if key_type in keywords:
                        text_to_analyze += ' ' + ' '.join(keywords[key_type]).lower()

            # Store detected categories
            detected_categories = set()

            # Check text against each category's keywords
            for category, keywords in self.category_keywords.items():
                if self._text_matches_category(text_to_analyze, keywords):
                    category_id = self.category_mappings.get(category)
                    if category_id:
                        detected_categories.add(category_id)

            # Convert to list and add default category if none found
            category_ids = list(detected_categories)
            if not category_ids:
                category_ids = [1]  # Default to Uncategorized
            
            # Ensure all category IDs are integers
            category_ids = [int(cat_id) for cat_id in category_ids]
            
            logger.info(f"Detected category IDs: {category_ids}")
            return category_ids

        except Exception as e:
            logger.error(f"Error detecting categories: {e}")
            return [1]  # Return default category on error

    def _text_matches_category(self, text: str, keywords: Dict[str, List[str]]) -> bool:
        """
        Check if text matches category keywords
        Returns True if sufficient keyword matches are found
        """
        try:
            # Count keyword matches
            match_count = sum(1 for keyword in keywords['keywords'] if keyword in text)
            
            # Require at least 2 keyword matches for a category match
            return match_count >= 2

        except Exception as e:
            logger.error(f"Error matching text to category: {e}")
            return False

    def extract_tags_from_title(self, title: str) -> List[str]:
        """
        Extract relevant tags from article title
        Returns a list of tags
        """
        try:
            words = title.lower().split()
            tags = []

            # Single word tags (words longer than 3 characters)
            tags.extend([w for w in words if len(w) > 3])

            # Two-word combinations
            for i in range(len(words) - 1):
                tags.append(f"{words[i]} {words[i+1]}")

            # Remove duplicates while preserving order
            unique_tags = []
            seen = set()
            for tag in tags:
                if tag not in seen:
                    seen.add(tag)
                    unique_tags.append(tag)

            return unique_tags[:10]  # Limit to 10 tags

        except Exception as e:
            logger.error(f"Error extracting tags: {e}")
            return []
