import socket
from utils import logger

def safe_socket_shutdown(sock: socket.socket) -> None:
    """Safely shutdown a socket with error handling"""
    try:
        # Only attempt shutdown if it's a valid socket
        if sock and isinstance(sock, socket.socket):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError as sock_err:
                if sock_err.winerror != 10038:  # Not a socket error
                    logger.error(f"Socket shutdown error: {sock_err}")
    except Exception as e:
        logger.warning(f"Error during socket cleanup: {e}")
