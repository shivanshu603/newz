import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
import sys
import traceback
from typing import Optional, Dict, Any
from functools import wraps
import time
import json

class CustomLogger:
    """Enhanced logging utility for the blog generation system"""
    
    def __init__(self, name: str = "BlogGenerator"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Set up logging handlers"""
        # Ensure log directory exists
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # File handler with rotation by size (UTF-8 encoded)
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Daily rotating handler for analytical logs (UTF-8 encoded)
        analytics_handler = TimedRotatingFileHandler(
            log_dir / "analytics.log",
            when="midnight",
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        analytics_handler.setLevel(logging.INFO)
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        try:
            # Force UTF-8 encoding if possible
            if hasattr(console_handler.stream, 'reconfigure'):
                console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
            console_handler.setLevel(logging.INFO)

            # Override emit to handle encoding errors gracefully
            original_emit = console_handler.emit

            def safe_emit(record):
                try:
                    original_emit(record)
                except UnicodeEncodeError:
                    # Encode message with errors='replace' to avoid crash
                    msg = console_handler.format(record)
                    stream = console_handler.stream
                    try:
                        stream.write(msg.encode(stream.encoding or 'utf-8', errors='replace').decode(stream.encoding or 'utf-8'))
                        stream.write(console_handler.terminator)
                        console_handler.flush()
                    except Exception:
                        # If still fails, fallback to ignoring errors
                        stream.write(msg.encode(stream.encoding or 'utf-8', errors='ignore').decode(stream.encoding or 'utf-8'))
                        stream.write(console_handler.terminator)
                        console_handler.flush()

            console_handler.emit = safe_emit

        except Exception as e:
            # Fallback for terminals that don't support encoding changes
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.ERROR)
            self.logger.error(f"Could not configure console encoding: {e}")
        
        # Create formatters and add them to the handlers
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        analytics_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s'
        )
        
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        analytics_handler.setFormatter(analytics_format)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(analytics_handler)
        
        # Store handlers for later use
        self.file_handler = file_handler
        self.console_handler = console_handler
        self.analytics_handler = analytics_handler


    def log_execution_time(self, func):
        """Decorator to log function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.info(
                    f"Function {func.__name__} executed in {execution_time:.2f} seconds",
                    extra={'execution_time': execution_time}
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self.error(
                    f"Error in {func.__name__}: {str(e)}",
                    extra={
                        'execution_time': execution_time,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                )
                raise
        return wrapper

    def log_memory_usage(self, func):
        """Decorator to log memory usage"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import psutil
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                result = func(*args, **kwargs)
                
                end_memory = process.memory_info().rss / 1024 / 1024
                memory_diff = end_memory - start_memory
                
                self.info(
                    f"Memory usage for {func.__name__}: {memory_diff:.2f}MB",
                    extra={
                        'start_memory': start_memory,
                        'end_memory': end_memory,
                        'memory_diff': memory_diff
                    }
                )
                return result
            except Exception as e:
                self.error(f"Error monitoring memory in {func.__name__}: {str(e)}")
                return func(*args, **kwargs)
        return wrapper

    def exception_handler(self, func):
        """Decorator for standardized exception handling"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.error(
                    f"Exception in {func.__name__}: {str(e)}",
                    extra={
                        'error_type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    }
                )
                raise
        return wrapper

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with optional extra data"""
        self._log(logging.INFO, message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message with optional extra data"""
        self._log(logging.ERROR, message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with optional extra data"""
        self._log(logging.WARNING, message, extra)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with optional extra data"""
        self._log(logging.DEBUG, message, extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message with optional extra data"""
        self._log(logging.CRITICAL, message, extra)

    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """Internal method to handle logging with extra data"""
        if extra:
            extra_str = json.dumps(extra)
            self.logger.log(level, message, extra={'extra': extra_str})
        else:
            self.logger.log(level, message, extra={'extra': '{}'})

    def set_level(self, level: int):
        """Set logging level"""
        self.logger.setLevel(level)
        self.file_handler.setLevel(level)
        self.console_handler.setLevel(level)

    def disable_console_output(self):
        """Disable console output"""
        self.logger.removeHandler(self.console_handler)

    def enable_console_output(self):
        """Enable console output"""
        if self.console_handler not in self.logger.handlers:
            self.logger.addHandler(self.console_handler)

    def cleanup(self):
        """Clean up logging handlers"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

# Singleton logger instance
_logger_instance = None

def get_logger():
    """Get the singleton logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = CustomLogger()
        # Ensure no duplicate handlers
        if len(_logger_instance.logger.handlers) > 3:  # We expect 3 handlers
            for handler in _logger_instance.logger.handlers[3:]:
                handler.close()
                _logger_instance.logger.removeHandler(handler)
    return _logger_instance

# Create and export the logger instance
logger = get_logger()
__all__ = ['logger']
