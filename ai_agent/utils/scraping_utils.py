import aiohttp
import asyncio
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import random
from datetime import datetime, timedelta

class ScrapingUtils:
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/89.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/91.0.864.59'
        ]
        self.request_delays = {}
        self.min_delay = 2
        self.session = None

    async def get_with_retry(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Make GET request with retry logic and rate limiting"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        domain = url.split('/')[2]
        last_request = self.request_delays.get(domain)
        if last_request:
            time_since = (datetime.now() - last_request).total_seconds()
            if time_since < self.min_delay:
                await asyncio.sleep(self.min_delay - time_since)

        headers = {'User-Agent': random.choice(self.user_agents)}
        
        for attempt in range(max_retries):
            try:
                async with self.session.get(url, headers=headers) as response:
                    self.request_delays[domain] = datetime.now()
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:  # Too Many Requests
                        await asyncio.sleep(30 * (attempt + 1))
                    else:
                        await asyncio.sleep(5 * (attempt + 1))
            except Exception as e:
                await asyncio.sleep(5 * (attempt + 1))
        return None

    @staticmethod
    def extract_text_with_selector(html: str, selector: str) -> List[str]:
        """Extract text content using CSS selector"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            return [element.get_text(strip=True) for element in soup.select(selector)]
        except Exception:
            return []

    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
