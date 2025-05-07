import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import Config
from utils import setup_logging

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitor overall system health and components"""
    
    def __init__(self):
        self.components: Dict[str, bool] = {
            "news_monitor": False,
            "blog_generator": False,
            "wordpress": False,
            "database": False
        }
        self.last_check = datetime.now()
        self.error_count = 0
        self.max_errors = 3
        
    async def check_filesystem(self) -> bool:
        """Verify required directories and files exist"""
        required_dirs = [
            Config.LOG_DIR,
            Config.DATA_DIR,
            Config.MODEL_CACHE_DIR
        ]
        
        required_files = [
            Path("config.py"),
            Path("models.py"),
            Path(".env")
        ]
        
        try:
            # Check directories
            for dir_path in required_dirs:
                if not Path(dir_path).exists():
                    logger.error(f"Required directory missing: {dir_path}")
                    return False
                    
            # Check files
            for file_path in required_files:
                if not file_path.exists():
                    logger.error(f"Required file missing: {file_path}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Filesystem check failed: {e}")
            return False
            
    async def check_database(self) -> bool:
        """Verify database connection and tables"""
        try:
            from models import init_db
            await init_db()
            self.components["database"] = True
            return True
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            self.components["database"] = False
            return False
            
    async def monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Check filesystem
                if not await self.check_filesystem():
                    self.error_count += 1
                
                # Check database
                if not await self.check_database():
                    self.error_count += 1
                
                # Check other components through ServiceManager
                from app import service_manager
                health = service_manager.health_check()
                
                self.components.update({
                    "news_monitor": health["monitor"] == "running",
                    "blog_generator": health["blog_generator"] == "running",
                    "wordpress": health["wordpress"] == "connected"
                })
                
                # Reset error count if all good
                if all(self.components.values()):
                    self.error_count = 0
                    
                # Take action if too many errors
                if self.error_count >= self.max_errors:
                    logger.critical("Too many system errors, attempting recovery...")
                    await self.attempt_recovery()
                    
                self.last_check = datetime.now()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)  # Short sleep on error
                
    async def attempt_recovery(self):
        """Try to recover system from errors"""
        try:
            # Restart services
            from app import service_manager
            await service_manager.shutdown()
            await service_manager.initialize()
            
            # Clear error count if successful
            if all(service_manager.health_check().values()):
                self.error_count = 0
                logger.info("System recovery successful")
                
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
