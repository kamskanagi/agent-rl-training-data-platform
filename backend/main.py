from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from core.database import async_session, init_db
from core.redis_client import close_redis, get_redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: initializing database tables...")
    await init_db()
    logger.info("Database tables created.")

    r = await get_redis()
    pong = await r.ping()
    logger.info(f"Redis ping: {pong}")

    yield

    await close_redis()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="RL Training Data Platform",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    db_status = "ok"
    redis_status = "ok"

    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    try:
        r = await get_redis()
        await r.ping()
    except Exception:
        redis_status = "error"

    status = "ok" if db_status == "ok" and redis_status == "ok" else "error"
    return {"status": status, "db": db_status, "redis": redis_status}
