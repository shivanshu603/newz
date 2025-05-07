import os
import psutil
import torch
import gc
from typing import Dict, Optional, Tuple
import logging
from utils import logger

class MemoryManager:
    """Memory management utility for AI operations"""
    
    def __init__(self, min_memory_gb: float = 2.0, min_gpu_memory_gb: float = 2.0):
        """Initialize memory manager with minimum requirements"""
        self.min_memory_gb = min_memory_gb
        self.min_gpu_memory_gb = min_gpu_memory_gb
        self.optimized_mode = False
        self._process = psutil.Process(os.getpid())

    def check_memory_availability(self) -> bool:
        """Check if sufficient memory is available"""
        try:
            # Get system memory info
            sys_memory = psutil.virtual_memory()
            available_memory_gb = sys_memory.available / (1024 ** 3)  # Convert to GB

            # Check GPU memory if CUDA is available
            if torch.cuda.is_available():
                gpu_memory = self._get_gpu_memory()
                if gpu_memory < self.min_gpu_memory_gb:
                    logger.warning(f"Low GPU memory: {gpu_memory:.2f}GB available")
                    return False

            # Check system memory
            if available_memory_gb < self.min_memory_gb:
                logger.warning(f"Low system memory: {available_memory_gb:.2f}GB available")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking memory availability: {e}")
            return False

    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return 0.0
        except Exception as e:
            logger.error(f"Error getting GPU memory: {e}")
            return 0.0

    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics"""
        try:
            stats = {
                'system_total': psutil.virtual_memory().total / (1024 ** 3),
                'system_available': psutil.virtual_memory().available / (1024 ** 3),
                'system_used': psutil.virtual_memory().used / (1024 ** 3),
                'process_memory': self._process.memory_info().rss / (1024 ** 3)
            }
            
            if torch.cuda.is_available():
                stats['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                stats['gpu_allocated'] = torch.cuda.memory_allocated() / (1024 ** 3)
                stats['gpu_reserved'] = torch.cuda.memory_reserved() / (1024 ** 3)
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

    def cleanup_memory(self) -> bool:
        """Perform memory cleanup"""
        try:
            # Clear Python garbage collector
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up memory: {e}")
            return False

    def enable_optimized_mode(self):
        """Enable optimized mode for low-memory environments"""
        self.optimized_mode = True
        self.min_memory_gb = max(1.0, self.min_memory_gb * 0.5)  # Reduce minimum requirements
        self.min_gpu_memory_gb = max(1.0, self.min_gpu_memory_gb * 0.5)

    def disable_optimized_mode(self):
        """Disable optimized mode"""
        self.optimized_mode = False
        self.min_memory_gb = min(2.0, self.min_memory_gb * 2.0)  # Restore minimum requirements
        self.min_gpu_memory_gb = min(2.0, self.min_gpu_memory_gb * 2.0)

    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size based on available memory"""
        try:
            stats = self.get_memory_stats()
            available_memory = stats.get('system_available', 0)
            
            if available_memory < 2:  # Less than 2GB available
                return 1
            elif available_memory < 4:  # Less than 4GB available
                return 2
            elif available_memory < 8:  # Less than 8GB available
                return 4
            else:
                return 8
                
        except Exception as e:
            logger.error(f"Error getting recommended batch size: {e}")
            return 1  # Conservative default

    def is_low_memory_mode(self) -> bool:
        """Check if system is in low memory condition"""
        try:
            stats = self.get_memory_stats()
            available_memory = stats.get('system_available', 0)
            
            return available_memory < self.min_memory_gb or self.optimized_mode
            
        except Exception as e:
            logger.error(f"Error checking low memory mode: {e}")
            return True  # Conservative default

    def monitor_memory_usage(self, threshold_gb: float = 1.0) -> Optional[Tuple[float, float]]:
        """Monitor memory usage and return (used, available) if above threshold"""
        try:
            stats = self.get_memory_stats()
            used = stats.get('process_memory', 0)
            available = stats.get('system_available', 0)
            
            if used > threshold_gb:
                return (used, available)
                
            return None
            
        except Exception as e:
            logger.error(f"Error monitoring memory usage: {e}")
            return None

    def estimate_memory_requirement(self, text_length: int) -> float:
        """Estimate memory requirement for processing text"""
        try:
            # Rough estimation: 
            # - Each character takes ~2 bytes
            # - Factor in overhead and intermediate calculations
            base_memory = text_length * 2  # Basic text storage
            processing_memory = base_memory * 3  # Processing overhead
            total_memory_bytes = base_memory + processing_memory
            
            return total_memory_bytes / (1024 ** 3)  # Convert to GB
            
        except Exception as e:
            logger.error(f"Error estimating memory requirement: {e}")
            return 0.0

    def can_process_text(self, text_length: int) -> bool:
        """Check if text of given length can be processed with available memory"""
        try:
            required_memory = self.estimate_memory_requirement(text_length)
            stats = self.get_memory_stats()
            available_memory = stats.get('system_available', 0)
            
            return available_memory > required_memory
            
        except Exception as e:
            logger.error(f"Error checking text processing capability: {e}")
            return False
