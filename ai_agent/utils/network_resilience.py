import aiohttp
import asyncio
import logging
import time
import random
import sys
import os
from typing import Optional, Any, Callable, Dict, Union, Type, List, Tuple
from functools import wraps
from aiohttp import ClientTimeout

# Add parent directory to path to allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils import logger

# Define the decorator outside the class
def resilient_request(retries=None, delay=None):
    """Decorator for making any async function network-resilient"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            max_attempts = retries or self.max_retries
            base_wait = delay or self.base_delay

            for attempt in range(max_attempts):
                try:
                    return await func(self, *args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    await self._handle_retry(attempt, e, base_wait)
                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
            return None
        return wrapper
    return decorator

class NetworkResilience:
    """Network resilience handler for reliable network operations"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """Initialize network resilience handler"""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.session = None
        self._backoff_factor = 2
        self._jitter = 0.1

    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close network resources"""
        if self.session:
            await self.session.close()
            self.session = None

    async def request(self,
                     method: str,
                     url: str,
                     **kwargs) -> Optional[Union[Dict[str, Any], str]]:
        """Make a resilient HTTP request"""
        await self._ensure_session()

        for attempt in range(self.max_retries):
            try:
                # Add default timeout if not specified
                if 'timeout' not in kwargs:
                    kwargs['timeout'] = aiohttp.ClientTimeout(total=30)

                if not self.session:
                    await self._ensure_session()
                    if not self.session:
                        logger.error("Failed to create session")
                        return None

                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 429:  # Rate limit
                        retry_after = response.headers.get('Retry-After', str(self.base_delay))
                        await asyncio.sleep(float(retry_after))
                        continue

                    response.raise_for_status()

                    # Handle different content types
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        return await response.json()
                    else:
                        return await response.text()

            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Network error after {self.max_retries} attempts: {e}")
                    raise

                await self._handle_retry(attempt, e)

            except Exception as e:
                logger.error(f"Unexpected error during request: {e}")
                raise

        return None

    async def _handle_retry(self, attempt: int, error: Exception, base_delay: float = None):
        """Handle retry logic with exponential backoff"""
        delay = base_delay or self.base_delay
        wait_time = delay * (self._backoff_factor ** attempt)

        # Add jitter
        jitter = wait_time * self._jitter * (2 * random.random() - 1)
        wait_time += jitter

        logger.warning(f"Request failed (attempt {attempt + 1}): {error}. Retrying in {wait_time:.2f}s")
        await asyncio.sleep(wait_time)

    @resilient_request()
    async def get_json(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """GET request expecting JSON response"""
        result = await self.request('GET', url, **kwargs)
        if isinstance(result, dict):
            return result
        return None

    @resilient_request()
    async def post_json(self, url: str, data: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """POST request with JSON data"""
        kwargs['json'] = data
        result = await self.request('POST', url, **kwargs)
        if isinstance(result, dict):
            return result
        return None

    async def batch_request(self,
                          urls: list[str],
                          method: str = 'GET',
                          concurrency: int = 5,
                          **kwargs) -> list[Optional[Dict[str, Any]]]:
        """Make multiple requests in parallel with rate limiting"""
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_with_semaphore(url: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                result = await self.request(method, url, **kwargs)
                if isinstance(result, dict):
                    return result
                return None

        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert to the expected return type
        return [r if isinstance(r, dict) else None for r in results]

    def is_rate_limited(self, response: aiohttp.ClientResponse) -> bool:
        """Check if response indicates rate limiting"""
        return (
            response.status == 429 or
            'X-RateLimit-Remaining' in response.headers and 
            int(response.headers['X-RateLimit-Remaining']) == 0
        )

    def get_retry_after(self, response: aiohttp.ClientResponse) -> float:
        """Get retry delay from response headers"""
        if 'Retry-After' in response.headers:
            return float(response.headers['Retry-After'])
        elif 'X-RateLimit-Reset' in response.headers:
            reset_time = int(response.headers['X-RateLimit-Reset'])
            return max(0, reset_time - time.time())
        return self.base_delay

    def should_retry(self, response: aiohttp.ClientResponse) -> bool:
        """Determine if request should be retried based on response"""
        return (
            response.status in {429, 500, 502, 503, 504} or  # Standard retry status codes
            (response.status >= 500 and response.status < 600)  # All 5xx errors
        )
