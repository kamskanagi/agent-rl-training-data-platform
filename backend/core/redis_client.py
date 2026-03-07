from __future__ import annotations

import json
import os

import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

ANNOTATION_QUEUE = "rl:queue:annotation"

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


async def close_redis():
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None


async def enqueue_task(task_id: str) -> None:
    r = await get_redis()
    await r.lpush(ANNOTATION_QUEUE, task_id)


async def dequeue_task(timeout: int = 0) -> str | None:
    r = await get_redis()
    result = await r.brpop(ANNOTATION_QUEUE, timeout=timeout)
    if result:
        return result[1]
    return None


async def cache_get(key: str) -> str | None:
    r = await get_redis()
    return await r.get(key)


async def cache_set(key: str, value: str, ttl: int = 60) -> None:
    r = await get_redis()
    await r.set(key, value, ex=ttl)
