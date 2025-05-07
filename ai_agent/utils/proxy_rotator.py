from typing import List
import random
from utils import logger

class ProxyRotator:
    def __init__(self, proxy_list: List[str]):
        self.proxies = proxy_list if proxy_list else ['']  # Empty string for no proxy
        self.current_index = 0

    def get_next_proxy(self) -> str:
        """Get next proxy using round-robin with fallback"""
        try:
            if not self.proxies:
                return ''
            
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            return proxy
        except Exception as e:
            logger.error(f"Error getting proxy: {e}")
            return ''
