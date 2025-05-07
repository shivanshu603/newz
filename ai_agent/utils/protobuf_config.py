import os
import sys
from utils import logger

def configure_protobuf():
    """Configure protobuf settings to avoid descriptor creation issues"""
    try:
        # Set environment variable for pure Python implementation
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        
        # Attempt to import protobuf to verify configuration
        import google.protobuf
        version = google.protobuf.__version__
        logger.info(f"Using protobuf version: {version}")
        
        # Verify version compatibility
        major, minor, *_ = version.split('.')
        if int(major) >= 4:
            logger.warning("Protobuf version 4+ detected, forcing pure Python implementation")
            
        return True
    except ImportError as e:
        logger.error(f"Error configuring protobuf: {e}")
        return False
