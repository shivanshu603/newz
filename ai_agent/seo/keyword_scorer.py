from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict

class KeywordScorer:
    def __init__(self):
        self.difficulty_weights = {
            'domain_authority': 0.4,
            'competition': 0.3,
            'content_quality': 0.3
        }
        self.intent_multipliers = {
            'transactional': 1.5,
            'commercial': 1.3,
            'informational': 1.0,
            'navigational': 0.8
        }

    def calculate_keyword_score(self, keyword_data: Dict) -> float:
        """Calculate comprehensive keyword score"""
        base_score = self._calculate_base_score(keyword_data)
        intent_multiplier = self.intent_multipliers.get(keyword_data.get('intent', 'informational'), 1.0)
        trend_bonus = self._calculate_trend_bonus(keyword_data.get('trend_data', {}))
        
        final_score = base_score * intent_multiplier * (1 + trend_bonus)
        return min(100, final_score)  # Cap at 100

    def cluster_keywords(self, keywords: List[str], num_clusters: int = 5) -> Dict[str, List[str]]:
        """Cluster keywords using K-means"""
        # Convert keywords to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([' '.join(k.split('-')) for k in keywords])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Group keywords by cluster
        clustered_keywords = defaultdict(list)
        for keyword, cluster_id in zip(keywords, clusters):
            clustered_keywords[f"cluster_{cluster_id}"].append(keyword)
            
        return dict(clustered_keywords)

    def _calculate_base_score(self, data: Dict) -> float:
        """Calculate base keyword score"""
        volume_score = np.log1p(data.get('search_volume', 0)) / 10
        competition_score = 1 - data.get('difficulty', 0.5)
        relevance_score = data.get('relevance', 0.5)
        
        return (volume_score * 0.4 + competition_score * 0.3 + relevance_score * 0.3) * 100

    def _calculate_trend_bonus(self, trend_data: Dict) -> float:
        """Calculate bonus score based on trend data"""
        if not trend_data:
            return 0
        
        growth_rate = trend_data.get('growth_rate', 0)
        consistency = trend_data.get('consistency', 0)
        seasonality = trend_data.get('seasonality', 1.0)
        
        trend_score = (growth_rate * 0.5 + consistency * 0.3 + seasonality * 0.2)
        return max(0, min(0.5, trend_score))  # Cap bonus at 50%
