from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.redis_client import cache_get, cache_set, get_redis, ANNOTATION_QUEUE
from models import Annotator, FeedbackItem, Task, TaskStatus, TrainingRun
from schemas import PlatformMetrics, TrainingRunResponse

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

PLATFORM_CACHE_KEY = "rl:cache:platform_metrics"


@router.get("/platform", response_model=PlatformMetrics)
async def platform_metrics(db: AsyncSession = Depends(get_db)):
    cached = await cache_get(PLATFORM_CACHE_KEY)
    if cached:
        return PlatformMetrics(**json.loads(cached))

    total_tasks = (await db.execute(select(func.count(Task.id)))).scalar() or 0
    pending_tasks = (
        await db.execute(
            select(func.count(Task.id)).where(Task.status == TaskStatus.PENDING)
        )
    ).scalar() or 0
    completed_tasks = (
        await db.execute(
            select(func.count(Task.id)).where(Task.status == TaskStatus.COMPLETED)
        )
    ).scalar() or 0
    total_feedback = (
        await db.execute(select(func.count(FeedbackItem.id)))
    ).scalar() or 0
    total_annotators = (
        await db.execute(select(func.count(Annotator.id)))
    ).scalar() or 0
    avg_quality = (
        await db.execute(select(func.avg(Task.quality_score)))
    ).scalar()
    avg_iaa = (await db.execute(select(func.avg(Task.iaa)))).scalar()

    r = await get_redis()
    queue_depth = await r.llen(ANNOTATION_QUEUE)

    metrics = PlatformMetrics(
        total_tasks=total_tasks,
        pending_tasks=pending_tasks,
        completed_tasks=completed_tasks,
        total_feedback=total_feedback,
        total_annotators=total_annotators,
        avg_quality_score=round(avg_quality, 4) if avg_quality else None,
        avg_iaa=round(avg_iaa, 4) if avg_iaa else None,
        queue_depth=queue_depth,
    )

    await cache_set(PLATFORM_CACHE_KEY, metrics.model_dump_json(), ttl=60)
    return metrics


@router.get("/training", response_model=list[TrainingRunResponse])
async def list_training_runs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TrainingRun).order_by(TrainingRun.created_at.desc())
    )
    return result.scalars().all()


@router.get("/training/{run_id}", response_model=TrainingRunResponse)
async def get_training_run(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run
