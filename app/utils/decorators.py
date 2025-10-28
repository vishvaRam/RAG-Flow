import time
import asyncio
from functools import wraps
from app.core.logging import logger


def log_time(operation_name: str):
    """Decorator to log execution time"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                logger.info(f"⏱️  {operation_name} took {duration:.2f}ms")
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                logger.error(f"⏱️  {operation_name} failed after {duration:.2f}ms: {e}")
                raise
        return wrapper
    return decorator
