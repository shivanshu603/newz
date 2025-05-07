# AI Blog Publisher System Architecture

## System Overview

This is an automated blog publishing system that monitors news sources, generates AI-powered blog content, and publishes it to WordPress. The system is built using FastAPI and includes various components for content generation, optimization, and publishing.

## Core Components

### 1. Application Entry Point (app.py)
- Main FastAPI application with async support
- ServiceManager class for lifecycle management:
  * Initializes and manages all core services
  * Handles graceful startup/shutdown
  * Runs background tasks
- REST API endpoints:
  * GET / - Health check
  * GET /content - Generate content
  * POST /publish - Publish blog posts
  * GET /topics - Get trending topics
- Background task processing
- Comprehensive error handling and logging

### 2. Configuration Management (config.py)
- Environment-based configuration system
- Core settings management:
  * Base paths and directories
  * WordPress credentials
  * Database configuration
  * News monitoring settings
  * Blog generation parameters
- Configuration validation
- Logging setup with file and console handlers
- Default fallbacks for optional settings

### 3. News Monitoring System
#### NewsMonitor (news_monitor.py)
- Asynchronous news source monitoring
- Features:
  * Source validation and management
  * Content scraping and processing
  * Article extraction and cleaning
  * Database integration
- Components:
  * BeautifulSoup for HTML processing
  * NLTK for text processing
  * Async HTTP client for fetching
- Error handling and retry logic

#### NewsDiscoverer (news_discovery.py)
- Trending topic identification
- Source aggregation
- Topic categorization
- Relevance scoring

### 4. Content Generation System
#### BlogGenerator (blog_generator.py)
- AI-powered content generation:
  * Uses transformers pipeline (default: gpt2)
  * Chunk-based text generation
  * Content optimization integration
- SEO Integration:
  * Keyword research
  * Meta description generation
  * Content optimization
- Quality Assurance:
  * Content validation
  * Readability checks
  * Plagiarism detection
- NLTK integration for text processing

### 5. WordPress Integration (wordpress_integration.py)
- WordPress XML-RPC Client:
  * Secure authentication
  * Connection management
  * Health checking
- Post Management:
  * Post creation and publishing
  * Metadata handling
  * Custom fields support
- Media Handling:
  * Image upload support
  * MIME type detection
  * Media attachment to posts
- Error handling and logging

## Data Flow

1. **News Discovery Flow**:
   ```
   News Sources -> NewsMonitor (Async Scraping) -> Content Processing -> Database Storage -> NewsDiscoverer -> Trending Topics
   ```
   - NewsMonitor continuously checks configured sources
   - Content is scraped and cleaned using BeautifulSoup
   - Articles are stored in database with metadata
   - NewsDiscoverer analyzes trends and patterns
   - Trending topics are ranked and prioritized

2. **Content Generation Flow**:
   ```
   Topic Selection -> BlogGenerator (AI Model) -> Content Optimization -> SEO Enhancement -> Quality Validation -> Final Content
   ```
   - Topics are selected from trending or user input
   - AI model generates initial content in chunks
   - Content is optimized for readability
   - SEO parameters are integrated
   - Quality checks ensure content standards

3. **Publishing Flow**:
   ```
   Final Content -> WordPress Client -> Media Upload -> Post Creation -> Metadata Addition -> Published Post
   ```
   - Content is prepared for WordPress format
   - Images are processed and uploaded
   - Post is created with metadata
   - SEO fields are populated
   - Post is published or scheduled

4. **Monitoring Flow**:
   ```
   Health Checks -> Service Status -> Error Detection -> Auto Recovery -> Logging
   ```
   - Regular health checks of all services
   - Status monitoring and reporting
   - Error detection and handling
   - Automatic recovery attempts
   - Comprehensive logging of all operations

## Directory Structure

```
/
├── app.py                      # Main application entry point
├── config.py                   # Configuration management
├── blog_generator.py           # Content generation logic
├── news_monitor.py            # News monitoring service
├── news_discovery.py          # Trending topics discovery
├── wordpress_integration.py    # WordPress API integration
├── content_optimizer.py       # Content optimization
├── seo_optimizer.py           # SEO optimization
├── models.py                  # Database models
├── utils.py                   # Utility functions
├── data/                      # Data storage
│   └── images/                # Image storage
├── logs/                      # Application logs
└── model_cache/              # AI model cache
```

## Service Dependencies

1. **External Services**:
   - WordPress site (configured via WP_URL)

2. **Internal Services**:
   - Database (SQLite/PostgreSQL)
   - File system (for logs and cache)
   - Background task processor

## Startup Sequence

1. **Configuration Loading**:
   - Load environment variables
   - Validate configuration
   - Setup logging

2. **Service Initialization**:
   - Initialize database
   - Start news monitor
   - Initialize blog generator
   - Connect to WordPress

3. **Background Tasks**:
   - Start news monitoring loop
   - Start health check loop
   - Initialize task queues

## API Endpoints

1. **Health Check**:
   ```
   GET /
   Returns system health status
   ```

2. **Content Generation**:
   ```
   GET /content?topic={topic}
   Generate blog content for optional topic
   ```

3. **Post Publishing**:
   ```
   POST /publish
   Generate and publish blog post
   ```

4. **Trending Topics**:
   ```
   GET /topics
   Get current trending topics
   ```

## Error Handling

- Service health monitoring
- Graceful degradation
- Automatic retries
- Comprehensive logging
- Error reporting

## Security Considerations

- API authentication
- WordPress credentials protection
- Rate limiting
- Input validation
- CORS configuration

## Monitoring and Maintenance

- Health check endpoint
- Service status monitoring
- Log rotation
- Cache management
- Database backups

## Scaling Considerations

- Async processing
- Task queuing
- Resource pooling
- Cache optimization
- Database indexing

This documentation provides a high-level overview of the system architecture. Each component has its own specific implementation details and can be further customized based on specific requirements.
