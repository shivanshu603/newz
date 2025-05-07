from dotenv import load_dotenv
import os
import logging
import asyncio
from typing import Dict, Optional, List

# Load environment variables from .env file
load_dotenv('.env')

# Debug: Print loaded environment variables
print("Loaded environment variables:")
print(f"WORDPRESS_USERNAME: {os.getenv('WORDPRESS_USERNAME')}")
print(f"WORDPRESS_PASSWORD: {os.getenv('WORDPRESS_PASSWORD')}")

from datetime import datetime
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Depends

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy.exc import SQLAlchemyError

from config import Config
from utils import logger
from models import init_db
from news_monitor import NewsMonitor
from news_discovery import NewsDiscoverer
from blog_generator import BlogGenerator
from wordpress_integration import WordPressClient
from system_monitor import SystemMonitor
from autonomous_controller import AutonomousController

# Configure application
Config.setup_logging()
os.makedirs(Config.LOG_DIR, exist_ok=True)
os.makedirs(Config.DATA_DIR, exist_ok=True)

class ServiceManager:
    """Manage application services and their lifecycle."""

    def __init__(self):
        self.monitor: Optional[NewsMonitor] = None
        self.discoverer: Optional[NewsDiscoverer] = None
        self.blog_gen: BlogGenerator = BlogGenerator()
        self.wp: Optional[WordPressClient] = None
        self.is_healthy = False
        self.last_health_check = datetime.utcnow()
        self.background_tasks: List[asyncio.Task] = []
        self.system_monitor = SystemMonitor()
        self.autonomous_controller = AutonomousController()

    async def initialize(self) -> None:
        """Initialize all services."""
        try:
            # Validate configuration
            if not Config.validate_config():
                raise ValueError("Invalid configuration")

            # Initialize database
            await init_db()

            # Initialize services
            self.monitor = NewsMonitor()
            self.discoverer = NewsDiscoverer()
            self.wp = WordPressClient()

            # Verify WordPress connection
            if not await self.wp.connect():
                raise ConnectionError("Could not connect to WordPress")

            # Start background tasks
            self.background_tasks.append(
                asyncio.create_task(self._run_news_monitor())
            )
            self.background_tasks.append(
                asyncio.create_task(self._run_health_check())
            )
            self.background_tasks.append(
                asyncio.create_task(self.system_monitor.monitor_loop())
            )

            self.background_tasks.append(
                asyncio.create_task(self.system_monitor.monitor_loop())
            )
            self.background_tasks.append(
                asyncio.create_task(self._run_content_pipeline())
            )
            self.background_tasks.append(
                asyncio.create_task(self.autonomous_controller.start())
            )

            self.is_healthy = True
            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            self.is_healthy = False
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown all services."""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Shutdown services
            if self.monitor:
                await self.monitor.stop()
            if self.discoverer:
                await self.discoverer.__aexit__(None, None, None)
            if self.wp:
                await self.wp.close()
            if self.discoverer:
                await self.discoverer.__aexit__(None, None, None)


            logger.info("Services shut down successfully")

        except Exception as e:
            logger.error(f"Error during service shutdown: {e}")
            raise

    async def _run_news_monitor(self) -> None:
        """Run the news monitoring service and handle content generation/publishing."""
        try:
            while True:
                if self.is_healthy and self.monitor:
                    # Monitor for new articles
                    articles = await self.monitor.monitor_sources()
                    
                    # Process each new article
                    for article in articles:
                        try:
                            # Generate blog content
                            blog_post: Dict[str, Any] = await self.blog_gen.generate_blog_post(article['title'])
                            
                            if blog_post["status"] == "success":
                                # Publish to WordPress
                                post_id = await self.wp.publish_post(
                                    title=blog_post["title"],
                                    content=blog_post["content"],
                                    meta_description=blog_post.get("meta_description", ""),
                                    keywords=blog_post.get("keywords", "").split(", "),
                                    categories=["News", "AI"],
                                    tags=["news", "ai", "automation"],
                                    images=blog_post.get("images", [])
                                )
                                
                                if post_id:
                                    logger.info(f"Successfully published post {post_id}: {blog_post['title']}")
                                else:
                                    logger.error("Failed to publish post")
                            
                        except Exception as e:
                            logger.error(f"Error processing article: {e}")
                            self.is_healthy = False
                            
                await asyncio.sleep(Config.NEWS_CHECK_INTERVAL)
        except Exception as e:
            logger.error(f"News monitoring error: {e}")
            self.is_healthy = False

    async def _run_content_pipeline(self) -> None:
        """Continuous content generation and publishing pipeline."""
        try:
            while True:
                if self.is_healthy:
                    # Get trending topics
                    topics = await self.discoverer.get_trending_topics() if self.discoverer else []
                    
                    # Process each topic
                    for topic in topics:
                        try:
                            # Generate blog content
                            blog_post: Dict[str, Any] = await self.blog_gen.generate_blog_post(topic)
                            
                            if blog_post["status"] == "success":
                                # Publish to WordPress
                                post_id: Optional[int] = await self.wp.publish_post(
                                    title=blog_post["title"],
                                    content=blog_post["content"],
                                    meta_description=blog_post.get("meta_description", ""),
                                    keywords=blog_post.get("keywords", "").split(", "),
                                    categories=["Trending", "AI"],
                                    tags=["trending", "ai", "automation"],
                                    images=blog_post.get("images", [])
                                )
                                
                                if post_id:
                                    logger.info(f"Successfully published trending post {post_id}: {blog_post['title']}")
                                else:
                                    logger.error("Failed to publish trending post")
                            
                        except Exception as e:
                            logger.error(f"Error processing trending topic: {e}")
                            self.is_healthy = False
                            
                await asyncio.sleep(Config.TRENDING_CHECK_INTERVAL)
        except Exception as e:
            logger.error(f"Content pipeline error: {e}")
            self.is_healthy = False

    async def _run_health_check(self) -> None:
        """Periodic health check of services."""
        try:
            while True:
                try:
                    # Check WordPress connection
                    if self.wp and not await self.wp.connect():
                        raise ConnectionError("WordPress connection lost")

                    # Check monitor status
                    if self.monitor and not self.monitor.is_running:
                        raise ConnectionError("News monitor not running")

                    self.is_healthy = True
                    self.last_health_check = datetime.utcnow()

                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    self.is_healthy = False

                await asyncio.sleep(60)  # Check every minute
        except asyncio.CancelledError:
            logger.info("Health check stopped")

    def health_check(self) -> Dict[str, str]:
        """Get health status of all services."""
        return {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "last_check": self.last_health_check.isoformat(),
            "monitor": "running" if self.monitor and self.monitor.is_running else "stopped",
            "discoverer": "running" if self.discoverer else "stopped",
            "blog_generator": "running" if self.blog_gen else "stopped",
            "wordpress": "connected" if self.wp and self.wp.is_connected else "disconnected"
        }

# Initialize service manager
service_manager = ServiceManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    try:
        logger.info("Initializing application")
        await service_manager.initialize()
        yield
    finally:
        logger.info("Shutting down application")
        await service_manager.shutdown()

# Initialize FastAPI app
app = FastAPI(
    title="AI Blog Publisher",
    description="Automated blog content generation and publishing system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """Check system health status."""
    health_status = service_manager.health_check()
    if health_status["status"] == "healthy":
        return health_status
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=health_status
    )

@app.get("/content", status_code=status.HTTP_200_OK)
async def get_content(topic: Optional[str] = None):
    """
    Retrieve generated content.

    Args:
        topic: Optional topic for content generation
    """
    try:
        if not service_manager.is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is not healthy"
            )

        # Generate content
        blog_post = await service_manager.blog_gen.generate_blog_post(topic)

        if blog_post["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Content generation failed"
            )

        return blog_post

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/publish", status_code=status.HTTP_201_CREATED)
async def publish_post(
    background_tasks: BackgroundTasks,
    topic: Optional[str] = None,
    categories: List[str] = ["AI", "Technology"],
    tags: List[str] = ["ai", "blog", "automation"]
):
    """
    Generate and publish a blog post.

    Args:
        topic: Optional topic for the blog post
        categories: List of WordPress categories
        tags: List of post tags
    """
    try:
        if not service_manager.is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is not healthy"
            )

        # Generate content
        blog_post = await service_manager.blog_gen.generate_blog_post(topic)

        if blog_post["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Content generation failed"
            )

        # Publish to WordPress
        post_id = await service_manager.wp.publish_post(
            title=blog_post["title"],
            content=blog_post["content"],
            meta_description=blog_post.get("meta_description", ""),
            keywords=blog_post.get("keywords", "").split(", "),
            categories=categories,
            tags=tags,
            images=blog_post.get("images", [])
        )

        if not post_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to publish to WordPress"
            )

        # Schedule background tasks
        background_tasks.add_task(
            service_manager.blog_gen.update_post_metrics,
            post_id,
            blog_post["content"]
        )

        return {
            "status": "success",
            "message": "Blog post published successfully",
            "post_id": post_id,
            "title": blog_post["title"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing post: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/topics", status_code=status.HTTP_200_OK)
async def get_trending_topics():
    """Get current trending topics."""
    try:
        if not service_manager.is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is not healthy"
            )

        topics = await service_manager.discoverer.get_trending_topics() if service_manager.discoverer else []
        return {"topics": topics}

    except Exception as e:
        logger.error(f"Error getting trending topics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    import sys
    from pathlib import Path

    # Ensure all required directories exist
    for directory in [Config.LOG_DIR, Config.DATA_DIR, Config.MODEL_CACHE_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Configure logging
    Config.setup_logging()
    logger.info("Starting AI Blog Publisher")

    try:
        # Start the server
        uvicorn.run(
            "app:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True,
            workers=1,
            timeout_keep_alive=60,
            loop="asyncio"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
