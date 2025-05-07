import re
from typing import Dict, List, Optional, TypedDict
from keyword_researcher import KeywordResearcher
from utils import logger

class KeywordResult(TypedDict):
    """Properly typed keyword result structure"""
    primary_keywords: List[str]
    secondary_keywords: List[str]
    questions: List[str]
    semantic_groups: Dict[str, List[str]]
    related_terms: List[str]
    long_tail: List[str]

class KeywordResearcherEnhanced(KeywordResearcher):
    def __init__(self):
        super().__init__()
        self.relevance_threshold = 0.6
        self.min_keyword_length = 4
        
    async def research_keywords(self, topic: str) -> Dict:
        """Enhanced keyword research that returns Dict"""
        try:
            # Convert KeywordResult to Dict
            results = await super().find_keywords(topic)
            return dict(results)  # Convert TypedDict to regular dict
            
        except Exception as e:
            logger.error(f"Error in enhanced keyword research: {e}")
            return {
                'primary_keywords': [],
                'secondary_keywords': [],
                'questions': [],
                'semantic_groups': {},
                'related_terms': [],
                'long_tail': []
            }

    async def get_keywords(self, topic: str) -> KeywordResult:
        """Alias for find_keywords to maintain backward compatibility"""
        return await self.find_keywords(topic)

    async def find_keywords(self, topic: str) -> KeywordResult:
        """Enhanced keyword research using deep semantic analysis"""
        # Skip processing if input looks like a phone number
        if re.match(r'^[\d\s\-+()]{7,}$', topic.strip()):
            return self._create_empty_result(topic)
        return await super().find_keywords(topic)
            
    def _is_relevant_enhanced(self, keyword: str, topic: str) -> bool:
        """Enhanced relevance checking"""
        # Add your enhanced relevance logic here
        return True  # Placeholder
