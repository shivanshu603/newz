from collections import defaultdict
import numpy as np
from typing import Dict, List, Set
import re
from datetime import datetime, timedelta

class ClickstreamAnalyzer:
    def __init__(self):
        self.user_patterns = defaultdict(list)
        self.search_history = defaultdict(int)
        self.timestamp_data = defaultdict(list)

    def track_search(self, query: str, user_id: str = 'anonymous'):
        """Track a search query"""
        timestamp = datetime.now()
        self.search_history[query.lower()] += 1
        self.user_patterns[user_id].append((query, timestamp))
        self.timestamp_data[query].append(timestamp)

    def get_trending_searches(self, hours: int = 24) -> List[Dict]:
        """Get trending searches in the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        trend_scores = defaultdict(float)

        for query, timestamps in self.timestamp_data.items():
            recent_searches = len([t for t in timestamps if t > cutoff])
            total_searches = len(timestamps)
            
            if total_searches > 0:
                trend_score = (recent_searches / total_searches) * np.log1p(total_searches)
                trend_scores[query] = trend_score

        # Sort by trend score
        trending = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)
        return [{'keyword': k, 'trend_score': s} for k, s in trending[:20]]

    def get_user_journey(self, user_id: str) -> List[Dict]:
        """Analyze user search journey"""
        searches = self.user_patterns.get(user_id, [])
        journey = []
        
        for i, (query, timestamp) in enumerate(searches):
            journey.append({
                'step': i + 1,
                'query': query,
                'timestamp': timestamp,
                'type': self._classify_search_type(query)
            })
        return journey

    def _classify_search_type(self, query: str) -> str:
        """Classify search type based on patterns"""
        query = query.lower()
        
        if any(word in query for word in ['how', 'what', 'why', 'when']):
            return 'informational'
        elif any(word in query for word in ['buy', 'price', 'purchase', 'cheap']):
            return 'transactional'
        elif any(word in query for word in ['best', 'review', 'compare', 'vs']):
            return 'commercial'
        elif re.search(r'\b(login|sign|website|official)\b', query):
            return 'navigational'
        return 'other'
