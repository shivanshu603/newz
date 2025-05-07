import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque, defaultdict
from utils import logger

class RequestThrottler:
    def __init__(self, requests_per_second: float, burst_size: int = 1):
        self.rate = 1.0 / requests_per_second
        self.burst_size = burst_size
        self.last_request = {}
        self.locks = {}
        self._cleanup_task = None
        self._request_times = {}
        self._tokens = {}
        self._max_tokens = burst_size

    async def wait(self, engine: str = 'default'):
        """Alias for acquire method to maintain compatibility"""
        await self.acquire(engine)

    async def acquire(self, key: str = 'default'):
        """Throttle requests with improved error handling"""
        try:
            # Normal throttling logic
            if key not in self._request_times:
                self._request_times[key] = deque(maxlen=self.burst_size)
                self._tokens[key] = self._max_tokens
                
            now = datetime.now()
            self._request_times[key].append(now)
            
            # Check if we're being rate limited
            if len(self._request_times[key]) >= self.burst_size:
                time_diff = (now - self._request_times[key][0]).total_seconds()
                if time_diff < self.rate * self.burst_size:
                    # Exponential backoff
                    delay = min(300, self.rate * (2 ** (len(self._request_times[key]) - self.burst_size)))
                    logger.warning(f"Rate limit reached for {key}, backing off for {delay:.1f}s")
                    await asyncio.sleep(delay)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error in request throttling: {e}")
            # Default delay if error occurs
            await asyncio.sleep(2)
            return False

    def cleanup(self):
        """Clean up old request records"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)
        
        for key in list(self._request_times.keys()):
            while self._request_times[key] and self._request_times[key][0] < cutoff:
                self._request_times[key].popleft()
            if not self._request_times[key]:
                del self._request_times[key]
                if key in self._tokens:
                    del self._tokens[key]

class RequestLimiter:
    def __init__(self, requests_per_second: float = 1.0):
        """Initialize request limiter with rate limit"""
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time: Dict[str, float] = {}
        self.lock = asyncio.Lock()

    async def wait(self, domain: Optional[str] = None) -> None:
        """Wait if needed to respect rate limits"""
        try:
            async with self.lock:
                now = time.time()
                key = domain or 'default'
                
                if key in self.last_request_time:
                    elapsed = now - self.last_request_time[key]
                    if elapsed < self.min_interval:
                        delay = self.min_interval - elapsed
                        await asyncio.sleep(delay)
                
                self.last_request_time[key] = time.time()
                
        except Exception as e:
            logger.error(f"Error in request limiter wait: {e}")
            # Default delay as fallback
            await asyncio.sleep(1.0)

    def reset(self) -> None:
        """Reset the limiter's state"""
        self.last_request_time.clear()
