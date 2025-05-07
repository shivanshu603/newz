import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from utils import logger
import yaml

class Config:
    """Configuration management for the blog generation system"""
    
    # File paths and directories
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = BASE_DIR / "cache"
    MODELS_DIR = BASE_DIR / "models"
    MODEL_CACHE_DIR = CACHE_DIR / "models"  # For cached model files
    LOGS_DIR = BASE_DIR / "logs"
    LOG_DIR = LOGS_DIR  # Backward compatibility alias
    TEMP_DIR = BASE_DIR / "temp"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, CACHE_DIR, MODELS_DIR, MODEL_CACHE_DIR, LOGS_DIR, TEMP_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # API Keys and credentials (load from environment variables)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    NYTIMES_API_KEY = os.getenv('NYTIMES_API_KEY')
    GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    UNSPLASH_API_KEY = os.getenv('UNSPLASH_API_KEY')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')

    # Database configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///data/blog_generator.db')

    # WordPress settings
    WORDPRESS_SITE_URL = os.getenv('WORDPRESS_SITE_URL', 'https://example.com')  # Default value added
    WORDPRESS_USERNAME = os.getenv('WORDPRESS_USERNAME')
    WORDPRESS_PASSWORD = os.getenv('WORDPRESS_PASSWORD')
    
    # Article generation settings
    MIN_ARTICLE_LENGTH = 1000
    MAX_ARTICLE_LENGTH = 3000
    MIN_SECTION_LENGTH = 100
    MAX_SECTION_LENGTH = 500
    MIN_PARAGRAPH_LENGTH = 50
    MAX_PARAGRAPH_LENGTH = 150
    
    # SEO settings
    MIN_KEYWORD_DENSITY = 0.01
    MAX_KEYWORD_DENSITY = 0.03
    META_DESCRIPTION_LENGTH = 155
    SEO_TITLE_LENGTH = 60
    MAX_TAGS = 5
    MAX_CATEGORIES = 3
    
    # Image settings
    MIN_IMAGE_WIDTH = 800
    MIN_IMAGE_HEIGHT = 600
    MAX_IMAGES_PER_ARTICLE = 3
    SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp']
    
    # Network settings
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    MAX_CONCURRENT_REQUESTS = 5
    TRENDING_WINDOW_HOURS = 24  # Time window for trending topics analysis
    
    # Model settings
    DEFAULT_MODEL = "gpt2"  # Start with smaller base model
    FALLBACK_MODEL = "distilgpt2"  # Even smaller fallback
    MODEL_BATCH_SIZE = 2  # Reduced from 4
    MAX_TOKENS = 5000  # Reduced from 1000
    TEMPERATURE = 0.75
    TOP_P = 0.95
    
    MODEL_REQUIREMENTS = {
        'required_packages': [
            'torch',
            'transformers',
            'sentencepiece',
            'spacy',
            'nltk'
        ],
        'optional_packages': [
            'clip',
            'diffusers'
        ]
    }
    
    # Memory management
    MIN_MEMORY_GB = 1.0  # Reduced from 2.0GB
    MIN_GPU_MEMORY_GB = 1.0  # Reduced from 2.0GB
    MEMORY_THRESHOLD_GB = 0.5  # Reduced from 1.0GB
    MAX_ARTICLE_COMPLEXITY = "medium"  # New setting to control resource usage
    
    # RAG settings
    MAX_CONTEXT_LENGTH = 2000
    MIN_SIMILARITY_SCORE = 0.7
    TOP_K_RESULTS = 5
    CACHE_TIMEOUT = 3600  # 1 hour

    # News sources configuration
    NEWS_SOURCES = [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "https://www.theguardian.com/world/rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://feeds.npr.org/1001/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
        "https://www.wired.com/feed/rss",
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
    ]
    
    # Content quality settings
    MIN_READABILITY_SCORE = 60  # Flesch reading ease
    MAX_SENTENCE_LENGTH = 25
    MIN_SENTENCE_LENGTH = 10
    FILLER_PHRASES = [
        "it goes without saying",
        "as a matter of fact",
        "at the end of the day",
        "needless to say",
        "for what it's worth"
    ]
    
    # Humanization settings
    MAX_REPEATED_PHRASES = 3
    MIN_UNIQUE_WORDS_RATIO = 0.4
    TRANSITION_WORDS = [
        "however", "moreover", "furthermore", "consequently",
        "additionally", "meanwhile", "nevertheless", "therefore"
    ]

    SECTION_MAX_TOKENS = 1000

    
    @classmethod
    def load_config(cls, config_file: str = "config.json") -> None:
        """Load configuration from file"""
        try:
            config_path = cls.BASE_DIR / config_file
            if not config_path.exists():
                return
                
            with open(config_path) as f:
                if config_file.endswith('.json'):
                    config_data = json.load(f)
                elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file}")
                    
            # Update class attributes
            for key, value in config_data.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
                    
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config from {config_file}: {e}")

    @classmethod
    def save_config(cls, config_file: str = "config.json") -> None:
        """Save current configuration to file"""
        try:
            config_path = cls.BASE_DIR / config_file
            
            # Get all uppercase attributes
            config_data = {
                key: value for key, value in cls.__dict__.items()
                if key.isupper() and not key.startswith('_')
            }
            
            # Convert paths to strings
            for key, value in config_data.items():
                if isinstance(value, Path):
                    config_data[key] = str(value)
            
            with open(config_path, 'w') as f:
                if config_file.endswith('.json'):
                    json.dump(config_data, f, indent=4)
                elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    yaml.safe_dump(config_data, f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file}")
                    
            logger.info(f"Saved configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving config to {config_file}: {e}")

    @classmethod
    def get_api_key(cls, service: str) -> Optional[str]:
        """Get API key for a service with validation"""
        key_map = {
            'openai': cls.OPENAI_API_KEY,
            'newsapi': cls.NEWSAPI_KEY,
            'nytimes': cls.NYTIMES_API_KEY,
            'guardian': cls.GUARDIAN_API_KEY,
            'google': cls.GOOGLE_API_KEY,
            'unsplash': cls.UNSPLASH_API_KEY,
            'reddit': cls.REDDIT_CLIENT_ID,
            'reddit_secret': cls.REDDIT_CLIENT_SECRET,
            'twitter': cls.TWITTER_API_KEY
        }
        
        key = key_map.get(service.lower())
        if not key:
            logger.warning(f"No API key found for service: {service}")
        return key

    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate configuration settings"""
        validation = {
            'api_keys': all([cls.OPENAI_API_KEY, cls.NEWSAPI_KEY, cls.GOOGLE_API_KEY]),
            'paths': all([cls.DATA_DIR.exists(), cls.CACHE_DIR.exists(), cls.MODELS_DIR.exists(), cls.LOGS_DIR.exists()]),
            'article_settings': all([cls.MIN_ARTICLE_LENGTH < cls.MAX_ARTICLE_LENGTH, cls.MIN_SECTION_LENGTH < cls.MAX_SECTION_LENGTH, cls.MIN_PARAGRAPH_LENGTH < cls.MAX_PARAGRAPH_LENGTH]),
            'seo_settings': all([cls.MIN_KEYWORD_DENSITY < cls.MAX_KEYWORD_DENSITY, cls.META_DESCRIPTION_LENGTH > 0, cls.SEO_TITLE_LENGTH > 0]),
            'image_settings': all([cls.MIN_IMAGE_WIDTH > 0, cls.MIN_IMAGE_HEIGHT > 0, cls.MAX_IMAGES_PER_ARTICLE > 0])
        }
        
        return validation

    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get path for a model file"""
        return cls.MODELS_DIR / model_name

    @classmethod
    def get_cache_path(cls, cache_key: str) -> Path:
        """Get path for a cache file"""
        return cls.CACHE_DIR / f"{cache_key}.json"

    @classmethod
    def get_log_path(cls, log_name: str) -> Path:
        """Get path for a log file"""
        return cls.LOGS_DIR / f"{log_name}.log"

    @classmethod
    def setup_logging(cls):
        """Configure logging system with UTF-8 support"""
        log_file = cls.get_log_path('ai_agent')
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Create stream handler that works with UTF-8
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, stream_handler]
        )

# Load configuration on module import
Config.load_config()
