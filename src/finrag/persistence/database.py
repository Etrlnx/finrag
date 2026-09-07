"""Database connection and session management for FinRAG."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from finrag.persistence.models import Base

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/finrag",
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    poolclass=NullPool,  # Use NullPool for serverless/container environments
)

# Create session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session():
    """Get a synchronous session for migrations/scripts."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    sync_url = DATABASE_URL.replace("postgresql+asyncpg", "postgresql")
    sync_engine = create_engine(sync_url)
    sync_session = sessionmaker(bind=sync_engine)
    return sync_session()


# For backwards compatibility / testing
def create_tables() -> None:
    """Create tables synchronously (for testing)."""
    sync_url = DATABASE_URL.replace("postgresql+asyncpg", "postgresql")
    from sqlalchemy import create_engine
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)