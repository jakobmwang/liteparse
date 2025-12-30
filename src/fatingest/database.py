# src/fatingest/database.py
"""
Async PostgreSQL ledger for documents and chunks.
"""
import asyncpg
import logging
import os
from typing import Any


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://litepipe:__CHANGE_ME__@localhost:5432/litepipe"
)

logger = logging.getLogger()


class Database:
    """
    PostgreSQL connection pool for document and chunk ledger.
    """
    
    def __init__(self):
        self.pool = None
    
    
    async def connect(self) -> None:
        """
        Create connection pool if not exists.
        """
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database connected")
    
    
    async def disconnect(self) -> None:
        """
        Close connection pool.
        """
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database disconnected")
    
    
    async def ensure_schema(self) -> None:
        """
        Create tables if they do not exist.
        """
        await self.connect()
        
        schema = """
        CREATE TABLE IF NOT EXISTS documents (
            document_id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            cas_key TEXT NOT NULL,
            meta JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(source, source_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_documents_cas_key ON documents(cas_key);
        CREATE INDEX IF NOT EXISTS idx_documents_scope ON documents(scope);
        
        CREATE TABLE IF NOT EXISTS chunks (
            document_id BIGINT NOT NULL,
            chunk_idx INT NOT NULL,
            chunk_hash TEXT NOT NULL,
            markdown TEXT NOT NULL,
            meta JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (document_id, chunk_idx),
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_chunks_hash_scope ON chunks(chunk_hash, (
            SELECT scope FROM documents WHERE documents.document_id = chunks.document_id
        ));
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(schema)
            logger.info("Database schema ensured")
    
    
    async def execute(self, query: str, *args) -> str:
        """
        Execute query and return status.
        """
        await self.connect()
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)
    
    
    async def fetch(self, query: str, *args) -> list[dict[str, Any]]:
        """
        Fetch multiple rows as list of dicts.
        """
        await self.connect()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    
    async def fetchrow(self, query: str, *args) -> dict[str, Any] | None:
        """
        Fetch single row as dict, or None if not found.
        """
        await self.connect()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    
    async def fetchval(self, query: str, *args) -> Any:
        """
        Fetch single value.
        """
        await self.connect()
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)