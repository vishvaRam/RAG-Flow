from contextlib import asynccontextmanager
from typing import AsyncGenerator
import asyncpg
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


class DatabaseManager:
    """Manages raw asyncpg connection pools and SQLAlchemy async engines."""

    def __init__(self):
        self.pool: asyncpg.Pool | None = None
        self.engine: AsyncEngine | None = None

    async def initialize(self) -> None:
        logger.info("Initializing Postgres pool and SQLAlchemy async engine...")
        self.pool = await asyncpg.create_pool(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            min_size=settings.DB_MIN_CONNECTIONS,
            max_size=settings.DB_MAX_CONNECTIONS,
            command_timeout=60,
        )
        self.engine = create_async_engine(
            settings.sqlalchemy_async_url,
            pool_pre_ping=True,
            pool_size=settings.DB_MIN_CONNECTIONS,
            max_overflow=settings.DB_MAX_CONNECTIONS,
        )
        logger.info("✓ Database pools successfully initialized")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
        if self.engine:
            await self.engine.dispose()
        logger.info("✓ Database pools successfully closed")

    @asynccontextmanager
    async def acquire_pg(self) -> AsyncGenerator[asyncpg.Connection, None]:
        if not self.pool:
            raise RuntimeError("Postgres pool has not been initialized.")
        async with self.pool.acquire() as connection:
            yield connection


db_manager = DatabaseManager()
