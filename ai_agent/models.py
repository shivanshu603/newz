from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import enum
import json
import re
import time
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey,
    create_engine, Index, Boolean, Float, Enum, UniqueConstraint, JSON
)
from utils import logger
from config import Config

# Initialize Base
Base = declarative_base()

# Define enums first
class SourceType(enum.Enum):
    NEWS = "news"
    SOCIAL = "social"
    BLOG = "blog"
    RSS = "rss"

class ContentStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PUBLISHED = "published"
    FAILED = "failed"

# Article Model (Define this first since it's referenced by other models)
class ArticleModel(Base):
    """Database model for articles"""
    __tablename__ = 'articles'

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    url = Column(String(500), unique=True)
    source_id = Column(Integer, ForeignKey('sources.id'), nullable=False)
    published_at = Column(DateTime(timezone=True), nullable=False)
    discovered_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    author = Column(String(100))
    images = Column(JSON)
    tags = Column(JSON)
    importance_score = Column(Float, default=0.0)
    
    # New SEO and readability fields
    seo_score = Column(Float, default=0.0)
    readability_score = Column(Float, default=0.0)
    seo_suggestions = Column(JSON)  # Store list of suggestions
    semantic_keywords = Column(JSON)  # Store list of semantic keywords
    heading_structure = Column(JSON)  # Store heading hierarchy data
    keyword_density = Column(JSON)  # Store keyword density metrics
    
    # Meta fields
    meta_description = Column(String(500))
    meta_keywords = Column(String(500))
    seo_title = Column(String(200))
    seo_description = Column(String(500))
    
    __table_args__ = (
        Index('idx_seo_score', 'seo_score'),
        Index('idx_readability_score', 'readability_score'),
    )

# Source Model
class Source(Base):
    __tablename__ = 'sources'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    url = Column(String(500), nullable=False, unique=True)
    type = Column(Enum(SourceType), nullable=False)
    active = Column(Boolean, default=True)
    
    # Relationship to ArticleModel
    articles = relationship("ArticleModel", backref="source")

# Article dataclass for business logic
class Article:
    def __init__(self, title: str, content: str, categories: List[int] = None, tags: List[str] = None, 
                 meta_description: str = None, slug: str = None, images: List[Dict] = None):
        self.title = title
        self.content = content
        self.categories = categories or [1]  # Default to Uncategorized (ID: 1)
        self.tags = tags or []
        self.meta_description = meta_description
        self.slug = slug or self._generate_slug(title)
        self.images = images or []
        self.featured_image_id = None
        self.seo_keywords = []
        
    def _generate_slug(self, title: str) -> str:
        """Generate URL-friendly slug from title"""
        slug = title.lower()
        # Remove special characters and replace spaces with hyphens
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'\s+', '-', slug)
        return slug.strip('-')
        
    def add_image(self, image_url: str, media_id: int = None):
        """Add image to article"""
        image = {'url': image_url}
        if media_id:
            image['id'] = media_id
        self.images.append(image)
        
    def set_featured_image(self, media_id: int):
        """Set featured image ID"""
        self.featured_image_id = media_id
        
    def add_category(self, category_id: int):
        """Add category ID to article"""
        if not self.categories:
            self.categories = []
        if category_id not in self.categories:
            self.categories.append(category_id)
            
    def add_tag(self, tag: str):
        """Add tag to article"""
        if not self.tags:
            self.tags = []
        # Clean and format tag
        tag = tag.lower().strip()
        if tag and tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary format"""
        return {
            'title': self.title,
            'content': self.content,
            'categories': self.categories,
            'tags': self.tags,
            'meta_description': self.meta_description,
            'slug': self.slug,
            'images': self.images,
            'featured_image_id': self.featured_image_id,
            'seo_keywords': self.seo_keywords
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Article':
        """Create article from dictionary data"""
        # Convert ISO format string back to datetime
        if isinstance(data.get('published_at'), str):
            data['published_at'] = datetime.fromisoformat(data['published_at'])
            
        return cls(**data)

    def validate(self) -> bool:
        """Validate article has required fields and meets quality standards"""
        try:
            if not self.title or not self.content:
                return False
                
            if len(self.content.split()) < 500:  # Minimum 500 words
                return False
                
            if not self.meta_description:
                return False
                
            # Add SEO validation
            if self.seo_score < 0.5:  # Minimum SEO score threshold
                return False
                
            if self.readability_score < 0.6:  # Minimum readability score threshold
                return False
                
            # Validate heading structure
            if not self.heading_structure or 'h1' not in self.heading_structure:
                return False
                
            return True
        except Exception:
            return False

@dataclass
class TrendingTopic:
    """Data class for trending topics"""
    name: str
    frequency: int
    first_seen: datetime = field(default_factory=datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=datetime.now(timezone.utc))

    source: str = "unknown"
    score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'frequency': self.frequency,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'source': self.source,
            'score': self.score
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrendingTopic':
        return cls(
            name=data['name'],
            frequency=data['frequency'],
            first_seen=datetime.fromisoformat(data['first_seen']),
            last_seen=datetime.fromisoformat(data['last_seen']),
            source=data.get('source', 'unknown'),
            score=data.get('score', 0.0)
        )

class TrendingTopicModel(Base):
    """Database model for trending topics"""
    __tablename__ = 'trending_topics'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    frequency = Column(Integer, default=1)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    source = Column(String(100))
    score = Column(Float, default=0.0)

    __table_args__ = (
        Index('idx_topic_name', 'name'),
        Index('idx_topic_score', 'score'),
    )

# Topic Models
class ProcessedTopic(Base):
    __tablename__ = 'processed_topics'
    
    id = Column(Integer, primary_key=True)
    topic_name = Column(String(500), unique=True, index=True)
    processed_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    source = Column(String(100))
    hash_signature = Column(String(64), unique=True)
    is_published = Column(Boolean, default=False)

class QueuedTopic(Base):
    __tablename__ = 'queued_topics'
    
    id = Column(Integer, primary_key=True)
    topic_name = Column(String(500))
    priority = Column(Float)
    queued_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    data = Column(Text)  # JSON serialized topic data

# Initialize database engine
# Ensure the database directory exists
import os
from pathlib import Path

# Extract the database path from the URL if it's a SQLite database
if Config.DATABASE_URL.startswith('sqlite'):
    # Parse the path from the URL
    db_path = Config.DATABASE_URL.split(':///')[-1]
    # Create the directory if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if db_dir:
        Path(db_dir).mkdir(parents=True, exist_ok=True)

engine = create_async_engine(
    Config.DATABASE_URL,
    # SQLite with aiosqlite doesn't support these pool parameters
    # pool_size=5,
    # max_overflow=10,
    # pool_timeout=30,
    pool_recycle=3600
)

async def init_db():
    """Initialize database"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
