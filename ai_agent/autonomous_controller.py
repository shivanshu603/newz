import logging
from typing import Optional
from fastapi import FastAPI
import logging
from utils import setup_logging
from news_monitor import NewsMonitor
from blog_publisher import BlogPublisher


logger = logging.getLogger(__name__)

class AutonomousController:
    """Main controller for autonomous blog publishing."""
    
    def __init__(self):
        """Initialize the autonomous controller."""
        setup_logging()
        self.news_monitor = NewsMonitor()
        self.blog_publisher = BlogPublisher()
        self.app = FastAPI()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes."""
        @self.app.get("/")
        async def root():
            return {"message": "Autonomous Blog Publisher Running"}
            
        @self.app.get("/status")
        async def status():
            return {
                "news_monitor": self.news_monitor.is_running,
                "blog_publisher": self.blog_publisher.is_running
            }
            
    def start(self):
        """Start the autonomous publishing system."""
        try:
            self.news_monitor.start()
            self.blog_publisher.start()
            logger.info("Autonomous publishing system started successfully")
        except Exception as e:
            logger.error(f"Error starting autonomous system: {e}")
            raise
            
    def stop(self):
        """Stop the autonomous publishing system."""
        try:
            self.news_monitor.stop()
            self.blog_publisher.stop()
            logger.info("Autonomous publishing system stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping autonomous system: {e}")
            raise
