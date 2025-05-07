import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config

def setup_model():
    """Download and setup the language model."""
    print(f"Setting up model: {Config.MODEL_NAME}")
    
    try:
        # Create model cache directory if it doesn't exist
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        
        # Download model and tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODEL_CACHE_DIR
        )
        
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODEL_CACHE_DIR
        )
        
        print("Model setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error setting up model: {e}")
        return False

if __name__ == "__main__":
    setup_model()