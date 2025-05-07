from typing import List, Dict
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import numpy as np

class LSIAnalyzer:
    def __init__(self):
        self.dictionary = None
        self.lsi_model = None
        self.num_topics = 10

    def train_lsi_model(self, documents: List[str]):
        """Train LSI model on documents"""
        # Tokenize documents
        texts = [[word for word in doc.lower().split()] for doc in documents]
        
        # Create dictionary
        self.dictionary = Dictionary(texts)
        
        # Create corpus
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        # Train LSI model
        self.lsi_model = LsiModel(
            corpus=corpus, 
            id2word=self.dictionary,
            num_topics=self.num_topics
        )

    def find_related_terms(self, keyword: str, top_n: int = 10) -> List[str]:
        """Find LSI-related terms for a keyword"""
        if not self.lsi_model or not self.dictionary:
            return []

        # Convert keyword to bow
        bow = self.dictionary.doc2bow(keyword.lower().split())
        
        # Get LSI representation
        lsi_rep = self.lsi_model[bow]
        
        # Find similar terms
        similar_terms = []
        for topic_id, score in sorted(lsi_rep, key=lambda x: abs(x[1]), reverse=True):
            terms = self.lsi_model.show_topic(topic_id, top_n)
            similar_terms.extend([term for term, _ in terms])
            
        return list(dict.fromkeys(similar_terms))[:top_n]  # Remove duplicates

    def calculate_coherence(self, keywords: List[str]) -> float:
        """Calculate coherence score for keywords"""
        if not self.lsi_model or not self.dictionary:
            return 0.0
            
        coherence_model = CoherenceModel(
            model=self.lsi_model,
            texts=[keywords],
            dictionary=self.dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
