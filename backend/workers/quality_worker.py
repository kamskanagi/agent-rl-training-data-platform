from __future__ import annotations

import asyncio
import json

from sqlalchemy import select

from core.database import async_session, init_db
from core.logging import setup_logging, get_logger
from core.metrics import quality_score_histogram, tasks_completed_total
from core.redis_client import get_redis, close_redis
from core.scoring import consensus_reward, fleiss_kappa, load_reliability, quality_score
from models import FeedbackItem, Task, TaskStatus

setup_logging()
logger = get_logger(__name__)

QUALITY_EVENT_CHANNEL = "rl:events:feedback"


async def process_task(task_id: str) -> None:
    async with async_session() as db:
        task = await db.get(Task, task_id)
        if not task:
            logger.warning("task_not_found", task_id=task_id)
            return

        result = await db.execute(
            select(FeedbackItem).where(
                FeedbackItem.task_id == task_id,
                FeedbackItem.flagged == False,  # noqa: E712
            )
        )
        all_feedback = list(result.scalars().all())

        if not all_feedback:
            logger.info("task_no_feedback", task_id=task_id)
            return

        # Recompute IAA from rankings
        rankings = [fb.ranking for fb in all_feedback if fb.ranking]
        if len(rankings) >= 2:
            task.iaa = round(fleiss_kappa(rankings), 4)

        # Recompute consensus reward and quality score
        reliability = await load_reliability(db, all_feedback)
        task.consensus_reward = round(consensus_reward(all_feedback, reliability), 4)
        task.quality_score = quality_score(task.iaa, all_feedback, task.min_annotations)

        # Auto-complete when enough annotations
        if len(all_feedback) >= task.min_annotations and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.COMPLETED
            tasks_completed_total.inc()
            logger.info("task_completed", task_id=task_id, quality_score=task.quality_score)

        # Flag low-quality tasks
        if task.quality_score is not None and task.quality_score < 0.3 and len(all_feedback) >= task.min_annotations:
            task.status = TaskStatus.FLAGGED
            logger.info("task_flagged", task_id=task_id, quality_score=task.quality_score)

        # Record quality score in Prometheus histogram
        if task.quality_score is not None:
            quality_score_histogram.observe(task.quality_score)

        await db.commit()
        logger.info(
            "task_processed",
            task_id=task_id,
            iaa=task.iaa,
            quality_score=task.quality_score,
            consensus_reward=task.consensus_reward,
            status=task.status,
        )


async def listen_pubsub():
    """Listen for feedback events via Redis Pub/Sub."""
    r = await get_redis()
    pubsub = r.pubsub()
    await pubsub.subscribe(QUALITY_EVENT_CHANNEL)
    logger.info("pubsub_subscribed", channel=QUALITY_EVENT_CHANNEL)

    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            data = json.loads(message["data"])
            task_id = data.get("task_id")
            if task_id:
                logger.info("feedback_event_received", task_id=task_id)
                await process_task(task_id)
        except Exception:
            logger.exception("Error processing pubsub message")


SWEEP_INTERVAL_SECONDS = 60


async def sweep_pending():
    """Fallback for missed Pub/Sub events: periodically re-score pending tasks that
    have feedback. (This used to pop the annotation queue, which stole tasks from
    annotators; that queue now belongs to the annotators alone.)"""
    logger.info("sweep_started", interval_seconds=SWEEP_INTERVAL_SECONDS)
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        try:
            async with async_session() as db:
                result = await db.execute(
                    select(Task.id)
                    .where(Task.status == TaskStatus.PENDING)
                    .where(Task.id.in_(select(FeedbackItem.task_id)))
                )
                task_ids = list(result.scalars().all())
            for task_id in task_ids:
                await process_task(task_id)
        except Exception:
            logger.exception("sweep_failed")


async def main():
    logger.info("worker_starting")
    await init_db()
    logger.info("worker_ready")

    # Run both listeners concurrently
    await asyncio.gather(
        listen_pubsub(),
        sweep_pending(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("worker_stopped")
