import secrets
import string
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from app.core.config import get_settings
from app.core.logging import logger
from app.models.schemas import (
    ChatMessageCreateDB,
    ChatMessageReadDB,
    SessionSummaryDB,
)

settings = get_settings()


def generate_message_id(length: int = 14) -> str:
    """Generate a random alphanumeric ID string."""
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


class PostgreSQLService:
    """
    Async PostgreSQL database service for managing frontend UI chat history and summaries.
    """

    def __init__(self):
        self.pool: asyncpg.Pool | None = None
        self._initializing = False

    async def _ensure_pool(self):
        """Ensure connection pool is initialized lazily."""
        if self.pool is None and not self._initializing:
            self._initializing = True
            try:
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
                logger.info("✓ PostgreSQL asyncpg connection pool created successfully")
            except Exception as error:
                logger.error(
                    f"✗ Error creating PostgreSQL connection pool: {error}",
                    exc_info=True,
                )
                self._initializing = False
                raise
            finally:
                self._initializing = False

    @asynccontextmanager
    async def acquire_connection(self):
        """Context manager for acquiring and auto-releasing pool connections."""
        await self._ensure_pool()
        connection = await self.pool.acquire()  # type: ignore
        try:
            yield connection
        finally:
            await self.pool.release(connection)  # type: ignore

    async def execute_query(
        self, query: str, *args, fetch: bool = True, fetchone: bool = False
    ) -> list[dict[str, Any]] | None:
        """Execute a parameterized SQL query asynchronously."""
        async with self.acquire_connection() as conn:
            if fetchone:
                result = await conn.fetchrow(query, *args)
                return dict(result) if result else None
            elif fetch:
                results = await conn.fetch(query, *args)
                return [dict(row) for row in results]
            else:
                await conn.execute(query, *args)
                return None

    async def insert_message(
        self, message: ChatMessageCreateDB, message_id: str | None = None
    ) -> ChatMessageReadDB:
        """Insert a chat message (user or assistant) into the history table."""
        if message_id is None:
            message_id = generate_message_id()

        query = f"""
        INSERT INTO {settings.HISTORY_TABLE} 
        (id, session_id, sender_id, sender_type, message, created_at)
        VALUES ($1, $2, $3, $4, $5, EXTRACT(EPOCH FROM NOW())::bigint)
        RETURNING id, session_id, sender_id, sender_type, message, 
                  created_at, updated_at, deleted_at
        """

        result = await self.execute_query(
            query,
            message_id,
            message.session_id,
            message.sender_id,
            message.sender_type,
            message.message,
            fetchone=True,
        )

        return ChatMessageReadDB(**result)  # type: ignore

    async def get_chat_history(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> list[ChatMessageReadDB]:
        """Retrieve paginated chat history for a session."""
        query = f"""
        SELECT id, session_id, sender_id, sender_type, message, 
               created_at, updated_at, deleted_at
        FROM {settings.HISTORY_TABLE}
        WHERE session_id = $1 AND deleted_at IS NULL
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        """

        results = await self.execute_query(query, session_id, limit, offset)
        return [ChatMessageReadDB(**row) for row in results]  # type: ignore

    async def count_session_messages(self, session_id: str) -> int:
        """Count total active messages in a session."""
        query = f"""
        SELECT COUNT(*) as count
        FROM {settings.HISTORY_TABLE}
        WHERE session_id = $1 AND deleted_at IS NULL
        """
        result = await self.execute_query(query, session_id, fetchone=True)
        return result["count"] if result else 0  # type: ignore

    async def get_session_summary(self, session_id: str) -> SessionSummaryDB | None:
        """Retrieve the latest session summary."""
        query = f"""
        SELECT id, session_id, summary, messages_count, created_at, updated_at
        FROM {settings.SUMMARY_TABLE}
        WHERE session_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """
        result = await self.execute_query(query, session_id, fetchone=True)
        return SessionSummaryDB(**result) if result else None  # type: ignore

    async def save_session_summary(
        self, session_id: str, summary: str, messages_count: int
    ) -> SessionSummaryDB:
        """Insert a new session summary."""
        summary_id = generate_message_id()
        query = f"""
        INSERT INTO {settings.SUMMARY_TABLE}
        (id, session_id, summary, messages_count, created_at)
        VALUES ($1, $2, $3, $4, EXTRACT(EPOCH FROM NOW())::bigint)
        RETURNING id, session_id, summary, messages_count, created_at, updated_at
        """
        result = await self.execute_query(
            query, summary_id, session_id, summary, messages_count, fetchone=True
        )
        return SessionSummaryDB(**result)  # type: ignore

    async def close_all_connections(self):
        """Gracefully close all pooled database connections."""
        if self.pool:
            await self.pool.close()
            logger.info("✓ PostgreSQL connection pool closed")


db_service = PostgreSQLService()
