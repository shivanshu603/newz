from typing import List, Dict, Set
from collections import Counter
import math

class KeywordRanker:
    def __init__(self):
        self.document_frequencies = {}
        self.total_documents = 0
        self.corpus = []

    def compute_tfidf(self, keywords: List[str], topic: str) -> Dict[str, float]:
        """
        Compute TF-IDF scores for keywords relative to the topic
        """
        # Treat each keyword as a document
        self.corpus = [kw.lower() for kw in keywords]
        self.total_documents = len(self.corpus)
        
        # Compute document frequencies
        for doc in self.corpus:
            words = set(doc.split())
            for word in words:
                self.document_frequencies[word] = self.document_frequencies.get(word, 0) + 1

        # Compute TF-IDF scores
        scores = {}
        topic_words = set(topic.lower().split())
        
        for keyword in keywords:
            score = 0
            kw_words = keyword.lower().split()
            
            # Term frequency in keyword
            tf = Counter(kw_words)
            
            # Calculate TF-IDF score
            for word in set(kw_words):
                if word in topic_words:  # Boost words that appear in topic
                    idf = math.log(self.total_documents / self.document_frequencies[word])
                    score += (tf[word] / len(kw_words)) * idf * 1.5  # 50% boost for topic words
                else:
                    idf = math.log(self.total_documents / self.document_frequencies[word])
                    score += (tf[word] / len(kw_words)) * idf
                    
            scores[keyword] = score
            
        return scores

    def compute_semantic_similarity(self, keywords: List[str], topic: str) -> Dict[str, float]:
        """
        Compute semantic similarity scores using character n-grams
        """
        scores = {}
        topic_ngrams = self._get_ngrams(topic.lower(), 3)
        
        for keyword in keywords:
            keyword_ngrams = self._get_ngrams(keyword.lower(), 3)
            
            # Calculate Jaccard similarity
            if topic_ngrams and keyword_ngrams:
                intersection = len(topic_ngrams & keyword_ngrams)
                union = len(topic_ngrams | keyword_ngrams)
                score = intersection / union if union > 0 else 0
            else:
                score = 0
                
            # Boost exact matches and contains relationships
            if topic.lower() in keyword.lower():
                score *= 1.5
            elif keyword.lower() in topic.lower():
                score *= 1.3
                
            scores[keyword] = score
            
        return scores

    def _get_ngrams(self, text: str, n: int) -> Set[str]:
        """Generate character n-grams from text"""
        return set(text[i:i+n] for i in range(len(text)-n+1))

    def rank_keywords(self, keywords: List[str], topic: str, 
                     tfidf_weight: float = 0.6, 
                     semantic_weight: float = 0.4) -> List[str]:
        """
        Rank keywords using a hybrid approach combining TF-IDF and semantic similarity
        """
        if not keywords:
            return []

        # Get TF-IDF scores
        tfidf_scores = self.compute_tfidf(keywords, topic)
        
        # Get semantic similarity scores
        semantic_scores = self.compute_semantic_similarity(keywords, topic)
        
        # Normalize scores
        max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1
        max_semantic = max(semantic_scores.values()) if semantic_scores else 1
        
        # Combine scores
        final_scores = {}
        for keyword in keywords:
            tfidf_score = tfidf_scores.get(keyword, 0) / max_tfidf
            semantic_score = semantic_scores.get(keyword, 0) / max_semantic
            
            final_scores[keyword] = (
                tfidf_score * tfidf_weight + 
                semantic_score * semantic_weight
            )
            
        # Sort by final score
        ranked_keywords = sorted(
            keywords,
            key=lambda k: final_scores[k],
            reverse=True
        )
        
        return ranked_keywords
