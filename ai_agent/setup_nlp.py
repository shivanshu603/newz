import os
import nltk
import spacy
import logging
from pathlib import Path
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Download required NLTK data"""
    try:
        # Create NLTK data directory
        nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.insert(0, nltk_data_dir)
        
        # Required NLTK packages
        required_packages = [
            'punkt',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words',
            'stopwords',
            'wordnet',
            'omw-1.4'
        ]
        
        # Download packages
        for package in required_packages:
            try:
                nltk.download(package, quiet=True, raise_on_error=True)
                logger.info(f"Successfully downloaded NLTK package: {package}")
            except Exception as e:
                logger.error(f"Failed to download NLTK package {package}: {e}")
                
    except Exception as e:
        logger.error(f"Error setting up NLTK: {e}")
        raise

def setup_spacy():
    """Download and set up spaCy models"""
    try:
        # Required spaCy models
        models = ['en_core_web_sm']
        
        for model in models:
            try:
                # Check if model is already installed
                spacy.load(model)
                logger.info(f"spaCy model {model} is already installed")
            except OSError:
                logger.info(f"Downloading spaCy model: {model}")
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
                
    except Exception as e:
        logger.error(f"Error setting up spaCy: {e}")
        raise

def setup_directories():
    """Create necessary directories"""
    try:
        # Create required directories
        directories = [
            "data",
            "logs",
            "models",
            "cache",
            "temp"
        ]
        
        base_dir = Path(__file__).parent
        for dir_name in directories:
            dir_path = base_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        raise

def main():
    """Main setup function"""
    try:
        logger.info("Starting NLP setup...")
        
        # Setup directories
        setup_directories()
        
        # Setup NLTK
        logger.info("Setting up NLTK...")
        setup_nltk()
        
        # Setup spaCy
        logger.info("Setting up spaCy...")
        setup_spacy()
        
        logger.info("NLP setup completed successfully")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()