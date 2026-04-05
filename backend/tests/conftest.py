from __future__ import annotations

import asyncio
import sys
import os
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.database import Base, get_db

# Use SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with test_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Mock Redis functions
mock_redis_store: dict[str, str] = {}
mock_redis_queue: list[str] = []


async def mock_enqueue_task(task_id: str):
    mock_redis_queue.append(task_id)


async def mock_dequeue_task(timeout: int = 0):
    if mock_redis_queue:
        return mock_redis_queue.pop(0)
    return None


async def mock_cache_get(key: str):
    return mock_redis_store.get(key)


async def mock_cache_set(key: str, value: str, ttl: int = 60):
    mock_redis_store[key] = value


async def mock_get_redis():
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.llen = AsyncMock(return_value=len(mock_redis_queue))
    return mock


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Create tables before each test and drop after."""
    import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    mock_redis_store.clear()
    mock_redis_queue.clear()


@pytest_asyncio.fixture
async def client():
    async def mock_build_export(dataset_id: str):
        pass  # no-op in tests

    with patch("core.redis_client.enqueue_task", mock_enqueue_task), \
         patch("core.redis_client.dequeue_task", mock_dequeue_task), \
         patch("core.redis_client.cache_get", mock_cache_get), \
         patch("core.redis_client.cache_set", mock_cache_set), \
         patch("core.redis_client.get_redis", mock_get_redis), \
         patch("routes.tasks.enqueue_task", mock_enqueue_task), \
         patch("routes.annotators.dequeue_task", mock_dequeue_task), \
         patch("routes.metrics.cache_get", mock_cache_get), \
         patch("routes.metrics.cache_set", mock_cache_set), \
         patch("routes.metrics.get_redis", mock_get_redis), \
         patch("routes.exports._build_export", mock_build_export):
        from main import app
        app.dependency_overrides[get_db] = override_get_db
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac
        app.dependency_overrides.clear()
