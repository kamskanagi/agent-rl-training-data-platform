from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://rl_user:rl_pass@localhost:5432/rl_platform",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    import models  # noqa: F401 — ensure all models are registered on Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    if engine.dialect.name == "postgresql":
        await _upgrade_existing_schema()


# create_all only creates missing tables; it never alters existing ones. Until the
# project adopts Alembic, these idempotent statements bring older databases up to date.
_SCHEMA_UPGRADES = [
    "ALTER TABLE annotators ADD COLUMN IF NOT EXISTS api_token_hash VARCHAR(64)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_annotators_api_token_hash ON annotators (api_token_hash)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_feedback_task_annotator ON feedback_items (task_id, annotator_id)",
]


async def _upgrade_existing_schema() -> None:
    import logging

    from sqlalchemy import text

    for stmt in _SCHEMA_UPGRADES:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(stmt))
        except Exception as exc:  # e.g. existing duplicate feedback blocks the unique index
            logging.getLogger(__name__).warning("schema upgrade skipped: %s (%s)", stmt, exc)
