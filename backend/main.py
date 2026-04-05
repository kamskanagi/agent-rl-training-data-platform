from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes.tasks import router as tasks_router
from routes.feedback import router as feedback_router
from routes.annotators import router as annotators_router
from routes.metrics import router as metrics_router
from routes.exports import router as exports_router

app.include_router(tasks_router)
app.include_router(feedback_router)
app.include_router(annotators_router)
app.include_router(metrics_router)
app.include_router(exports_router)


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
