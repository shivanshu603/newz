import json
import logging
import os
from pathlib import Path

def setup_logging():
    """Configure logging for the application with enhanced terminal output."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # More concise format for console output
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create handlers
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # Set log level based on environment variable or default to INFO
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )

    # Ensure the logger is created only once
    logger = logging.getLogger("ai_agent")

    # Log startup information
    logger.info("=" * 60)
    logger.info("BLOG GENERATION SYSTEM STARTING")
    logger.info("=" * 60)
    logger.info(f"Log level set to: {log_level_name}")
    logger.info(f"Logs will be saved to: {log_dir / 'app.log'}")
    logger.info("=" * 60)


from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Remove the duplicate logging configuration


# Set up logging
setup_logging()

# Create and export logger instance
logger = logging.getLogger("ai_agent")

def load_config():
    """Load configuration from environment variables and config.json"""
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load WordPress credentials from environment variables
        wordpress_config = {
            'url': os.getenv('WORDPRESS_URL'),
            'username': os.getenv('WORDPRESS_USERNAME'),
            'password': os.getenv('WORDPRESS_PASSWORD')
        }
        
        # Load other settings from config.json
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
            
        # Merge configurations
        config['wordpress'] = wordpress_config
        return config
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None

def get_trending_topics():
    """Fetch trending topics from reliable news sources with fallback"""
    try:
        # Try BBC News first
        url = "https://www.bbc.com/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract headlines from BBC
        headlines = soup.find_all('h3', class_='gs-c-promo-heading__title')
        topics = [h.text.strip() for h in headlines[:10]]  # Get top 10 topics
        
        if not topics:
            # Fallback to AP News if BBC fails
            url = "https://apnews.com/"
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = soup.find_all('h2', class_='PagePromo-title')
            topics = [h.text.strip() for h in headlines[:10]]
            
        if not topics:
            # Final fallback to hardcoded topics
            topics = [
                "Artificial Intelligence", 
                "Climate Change", 
                "Global Economy", 
                "Space Exploration", 
                "Healthcare Innovations", 
                "Renewable Energy", 
                "Cybersecurity", 
                "Education Technology", 
                "Sustainable Living", 
                "Digital Transformation"
            ]
            
        return topics
        
    except Exception as e:
        logger.error(f"Error fetching trending topics: {e}")
        # Return hardcoded topics if all else fails
        return [
            "Artificial Intelligence", 
            "Climate Change", 
            "Global Economy", 
            "Space Exploration", 
            "Healthcare Innovations"
        ]

def validate_config(config):
    """Validate the loaded configuration"""
    required_keys = ['wordpress', 'google', 'news_sources']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    return True
