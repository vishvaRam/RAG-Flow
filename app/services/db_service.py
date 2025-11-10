import os
import secrets
import string
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
import asyncpg
from langfuse import observe
from app.core.config import get_settings
from app.models.schemas import (
    ChatMessageCreateDB,
    ChatMessageReadDB,
    UserMessageCreateDB,
    AssistantMessageCreateDB,
    SessionSummaryDB,
)


settings = get_settings()


def generate_message_id(length: int = 14) -> str:
    """
    Generate a cryptographically secure random string ID.
    
    Args:
        length: Length of the ID string (default: 14)
        
    Returns:
        Random alphanumeric string
    """
    # Use uppercase letters and digits for better readability
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


class PostgreSQLService:
    """
    Async PostgreSQL database service with connection pooling for RAG applications.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5433,
        database: str = "rag_database",
        user: str = "postgres",
        password: str = "admin",
        min_connections: int = 2,
        max_connections: int = 10
    ):
        """
        Initialize PostgreSQL connection pool parameters.
        Pool will be created lazily on first use.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool: Optional[asyncpg.Pool] = None
        self._initializing = False
    
    async def _ensure_pool(self):
        """Ensure connection pool is initialized (lazy initialization)"""
        if self.pool is None and not self._initializing:
            self._initializing = True
            try:
                self.pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    min_size=self.min_connections,
                    max_size=self.max_connections,
                    command_timeout=60
                )
                print("✓ PostgreSQL connection pool created successfully")
            except Exception as error:
                print(f"✗ Error creating connection pool: {error}")
                self._initializing = False
                raise
            finally:
                self._initializing = False
    
    @asynccontextmanager
    async def acquire_connection(self):
        """
        Context manager for acquiring database connections.
        Automatically returns connection to pool after use.
        """
        await self._ensure_pool()
        
        connection = await self.pool.acquire()
        try:
            yield connection
        finally:
            await self.pool.release(connection)
    
    async def execute_query(
        self, 
        query: str, 
        *args,
        fetch: bool = True,
        fetchone: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a SQL query asynchronously.
        
        Args:
            query: SQL query string
            *args: Query parameters
            fetch: Whether to fetch all results
            fetchone: Whether to fetch single result
            
        Returns:
            Query results if fetch=True, None otherwise
        """
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

    @observe()
    async def insert_chat_message(
        self, 
        message: ChatMessageCreateDB,
        message_id: Optional[str] = None
    ) -> ChatMessageReadDB:
        """
        Insert a new chat message into the database.
        
        Args:
            message: ChatMessageCreateDB Pydantic model
            message_id: Optional custom message ID (generates one if not provided)
            
        Returns:
            ChatMessageReadDB with created message data
        """
        if message_id is None:
            message_id = generate_message_id()
            
        query = """
        INSERT INTO chat_messages_history 
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
            fetchone=True
        )

        return ChatMessageReadDB(**result)

    @observe()
    async def insert_user_message(
        self,
        message: UserMessageCreateDB,
        message_id: Optional[str] = None
    ) -> ChatMessageReadDB:
        """
        Insert a user message into the database.
        
        Args:
            message: UserMessageCreateDB Pydantic model
            message_id: Optional custom message ID (generates one if not provided)
            
        Returns:
            ChatMessageReadDB with created message data
        """
        if message_id is None:
            message_id = generate_message_id()
            
        query = """
        INSERT INTO chat_messages_history 
        (id, session_id, sender_id, sender_type, message, created_at)
        VALUES ($1, $2, $3, 'user', $4, EXTRACT(EPOCH FROM NOW())::bigint)
        RETURNING id, session_id, sender_id, sender_type, message, 
                  created_at, updated_at, deleted_at
        """
        
        result = await self.execute_query(
            query,
            message_id,
            message.session_id,
            message.sender_id,
            message.message,
            fetchone=True
        )

        return ChatMessageReadDB(**result)

    @observe()
    async def insert_assistant_message(
        self,
        message: AssistantMessageCreateDB,
        message_id: Optional[str] = None
    ) -> ChatMessageReadDB:
        """
        Insert an assistant message into the database.
        
        Args:
            message: AssistantMessageCreateDB Pydantic model
            message_id: Optional custom message ID (generates one if not provided)
            
        Returns:
            ChatMessageReadDB with created message data
        """
        if message_id is None:
            message_id = generate_message_id()
            
        query = """
        INSERT INTO chat_messages_history 
        (id, session_id, sender_id, sender_type, message, created_at)
        VALUES ($1, $2, $3, 'assistant', $4, EXTRACT(EPOCH FROM NOW())::bigint)
        RETURNING id, session_id, sender_id, sender_type, message, 
                  created_at, updated_at, deleted_at
        """
        
        result = await self.execute_query(
            query,
            message_id,
            message.session_id,
            message.sender_id,
            message.message,
            fetchone=True
        )

        return ChatMessageReadDB(**result)

    @observe()
    async def update_chat_message(
        self, 
        message_id: str, 
        new_message: str
    ) -> Optional[ChatMessageReadDB]:
        """
        Update an existing chat message.
        
        Args:
            message_id: Message ID to update (string)
            new_message: New message content
            
        Returns:
            Updated ChatMessageReadDB or None if not found
        """
        query = """
        UPDATE chat_messages_history
        SET message = $1, updated_at = EXTRACT(EPOCH FROM NOW())::bigint
        WHERE id = $2 AND deleted_at IS NULL
        RETURNING id, session_id, sender_id, sender_type, message, 
                  created_at, updated_at, deleted_at
        """
        
        result = await self.execute_query(
            query, 
            new_message, 
            message_id, 
            fetchone=True
        )
        return ChatMessageReadDB(**result) if result else None

    @observe()
    async def soft_delete_message(self, message_id: str) -> bool:
        """
        Soft delete a chat message.
        
        Args:
            message_id: Message ID to delete (string)
            
        Returns:
            True if deleted, False if not found
        """
        query = """
        UPDATE chat_messages_history
        SET deleted_at = EXTRACT(EPOCH FROM NOW())::bigint
        WHERE id = $1 AND deleted_at IS NULL
        """
        
        await self.execute_query(query, message_id, fetch=False)
        return True

    @observe()
    async def get_message_by_id(
        self, 
        message_id: str
    ) -> Optional[ChatMessageReadDB]:
        """
        Get a specific message by ID.
        
        Args:
            message_id: Message ID (string)
            
        Returns:
            ChatMessageReadDB or None if not found
        """
        query = """
        SELECT id, session_id, sender_id, sender_type, message, 
               created_at, updated_at, deleted_at
        FROM chat_messages_history
        WHERE id = $1
        """
        
        result = await self.execute_query(query, message_id, fetchone=True)
        return ChatMessageReadDB(**result) if result else None
    
    @observe()
    async def get_chat_history(
        self, 
        session_id: str, 
        limit: Optional[int] = 50,
        offset: Optional[int] = 0
    ) -> List[ChatMessageReadDB]:
        """
        Retrieve chat history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            offset: Number of messages to skip
            
        Returns:
            List of ChatMessageReadDB objects
        """
        query = """
        SELECT id, session_id, sender_id, sender_type, message, 
               created_at, updated_at, deleted_at
        FROM chat_messages_history
        WHERE session_id = $1 AND deleted_at IS NULL
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        """
        
        results = await self.execute_query(query, session_id, limit, offset)
        return [ChatMessageReadDB(**row) for row in results]
    
    async def count_session_messages(self, session_id: str) -> int:
        """Count total messages in a session."""
        query = """
        SELECT COUNT(*) as count
        FROM chat_messages_history
        WHERE session_id = $1 AND deleted_at IS NULL
        """
        result = await self.execute_query(query, session_id, fetchone=True)
        return result['count'] if result else 0
    
    async def get_session_summary(self, session_id: str) -> Optional[SessionSummaryDB]:
        """Get the latest summary for a session."""
        query = """
        SELECT id, session_id, summary, messages_count, created_at, updated_at
        FROM chat_session_summaries
        WHERE session_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """
        result = await self.execute_query(query, session_id, fetchone=True)
        return SessionSummaryDB(**result) if result else None
    
    async def save_session_summary(
        self,
        session_id: str,
        summary: str,
        messages_count: int
    ) -> SessionSummaryDB:
        """Save or update session summary."""
        summary_id = generate_message_id(14)
        query = """
        INSERT INTO chat_session_summaries
        (id, session_id, summary, messages_count, created_at)
        VALUES ($1, $2, $3, $4, EXTRACT(EPOCH FROM NOW())::bigint)
        RETURNING id, session_id, summary, messages_count, created_at, updated_at
        """
        result = await self.execute_query(
            query,
            summary_id,
            session_id,
            summary,
            messages_count,
            fetchone=True
        )
        return SessionSummaryDB(**result)
    
    async def get_conversation_context(
        self,
        session_id: str,
        window_size: int = 10
    ) -> Tuple[Optional[str], List[ChatMessageReadDB], int]:
        """
        Get conversation context with summary and recent messages.
        
        Args:
            session_id: Session identifier
            window_size: Number of recent messages to retrieve in full
            
        Returns:
            Tuple of (summary, recent_messages, total_count)
        """
        # Get total message count
        total_count = await self.count_session_messages(session_id)
        
        # Get recent messages
        recent_messages = await self.get_chat_history(session_id, limit=window_size)
        recent_messages.reverse()  # Chronological order
        
        # Get summary if exists
        summary_obj = await self.get_session_summary(session_id)
        summary = summary_obj.summary if summary_obj else None
        
        return summary, recent_messages, total_count
    
    async def close_all_connections(self):
        """Close all connections in the pool."""
        if self.pool:
            await self.pool.close()
            print("✓ All PostgreSQL connections closed")


# Module-level singleton instance
db_service = PostgreSQLService(
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    database=settings.DB_NAME,
    user=settings.DB_USER,
    password=settings.DB_PASSWORD,
    min_connections=settings.DB_MIN_CONNECTIONS,
    max_connections=settings.DB_MAX_CONNECTIONS
)
