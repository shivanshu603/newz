from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sqlite3
import json

class Feedback(BaseModel):
    article_id: str
    score: float
    comments: Optional[str] = None
    aspects: Optional[dict] = None

class FeedbackCollector:
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize feedback database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS feedback
                    (article_id TEXT, score REAL, comments TEXT, 
                     aspects TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

    async def store_feedback(self, feedback: Feedback):
        """Store feedback in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""INSERT INTO feedback (article_id, score, comments, aspects)
                        VALUES (?, ?, ?, ?)""",
                     (feedback.article_id, feedback.score, feedback.comments,
                      json.dumps(feedback.aspects or {})))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            return False

    async def get_feedback_stats(self):
        """Get feedback statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""SELECT AVG(score) as avg_score, 
                        COUNT(*) as total_feedback
                        FROM feedback""")
            stats = c.fetchone()
            conn.close()
            return {
                "average_score": stats[0],
                "total_feedback": stats[1]
            }
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {}
