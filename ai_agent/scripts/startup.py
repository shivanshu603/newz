import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from models import init_db
from utils import setup_logging

async def setup_environment():
    """Prepare environment for application startup"""
    try:
        # Create required directories
        for directory in [Config.LOG_DIR, Config.DATA_DIR, Config.MODEL_CACHE_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        # Initialize database
        await init_db()
        
        # Setup logging
        setup_logging()
        
        return True
        
    except Exception as e:
        print(f"Environment setup failed: {e}")
        return False

if __name__ == "__main__":
    if asyncio.run(setup_environment()):
        print("Environment setup completed successfully")
    else:
        sys.exit(1)
