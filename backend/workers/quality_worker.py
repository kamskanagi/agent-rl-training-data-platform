from __future__ import annotations

import asyncio
import json
import math

from itertools import combinations

from sqlalchemy import select

from core.database import async_session, init_db
from core.logging import setup_logging, get_logger
from core.metrics import quality_score_histogram, tasks_completed_total
from core.redis_client import get_redis, close_redis, ANNOTATION_QUEUE
from models import FeedbackItem, Task, TaskStatus

setup_logging()
logger = get_logger(__name__)

QUALITY_EVENT_CHANNEL = "rl:events:feedback"


def _fleiss_kappa(rankings: list[list[int]]) -> float:
    if len(rankings) < 2:
        return 0.0
    n_raters = len(rankings)
    n_items = len(rankings[0])
    if n_items < 2:
        return 0.0

    pairs = list(combinations(range(n_items), 2))
    if not pairs:
        return 0.0

    total_agreement = 0
    total_pairs = len(pairs) * n_raters * (n_raters - 1) / 2

    for i, j in pairs:
        prefer_i = sum(1 for r in rankings if r[i] < r[j])
        prefer_j = n_raters - prefer_i
        total_agreement += prefer_i * (prefer_i - 1) / 2 + prefer_j * (prefer_j - 1) / 2

    if total_pairs == 0:
        return 0.0

    p_o = total_agreement / total_pairs
    p_e = 0.5
    kappa = (p_o - p_e) / (1.0 - p_e)
    return max(-1.0, min(1.0, kappa))


def _consensus_reward(feedbacks: list[FeedbackItem]) -> float:
    rewards: list[float] = []
    for fb in feedbacks:
        if fb.scalar_reward is not None:
            rewards.append(fb.scalar_reward)
        elif fb.binary_label is not None:
            rewards.append(1.0 if fb.binary_label else 0.0)
        elif fb.ranking is not None and len(fb.ranking) >= 2:
            best = min(fb.ranking)
            worst = max(fb.ranking)
            if worst != best:
                rewards.append(1.0 - (best - 1) / (worst - 1))
            else:
                rewards.append(0.5)
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def _quality_score(
    iaa: float | None,
    feedbacks: list[FeedbackItem],
    min_annotations: int,
) -> float:
    iaa_val = max(0.0, iaa) if iaa is not None else 0.0

    rewards: list[float] = []
    for fb in feedbacks:
        if fb.scalar_reward is not None:
            rewards.append(fb.scalar_reward)
        elif fb.binary_label is not None:
            rewards.append(1.0 if fb.binary_label else 0.0)

    if len(rewards) >= 2:
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0

    non_flagged = [fb for fb in feedbacks if not fb.flagged]
    coverage = min(1.0, len(non_flagged) / max(1, min_annotations))

    return round(0.4 * iaa_val + 0.4 * (1.0 - min(1.0, stdev)) + 0.2 * coverage, 4)


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
            task.iaa = round(_fleiss_kappa(rankings), 4)

        # Recompute consensus reward and quality score
        task.consensus_reward = round(_consensus_reward(all_feedback), 4)
        task.quality_score = _quality_score(task.iaa, all_feedback, task.min_annotations)

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


async def poll_queue():
    """Fallback: poll the annotation queue for task IDs."""
    r = await get_redis()
    logger.info("queue_polling_started", queue=ANNOTATION_QUEUE)

    while True:
        result = await r.brpop(ANNOTATION_QUEUE, timeout=5)
        if result:
            task_id = result[1]
            logger.info("task_dequeued", task_id=task_id)
            await process_task(task_id)


async def main():
    logger.info("worker_starting")
    await init_db()
    logger.info("worker_ready")

    # Run both listeners concurrently
    await asyncio.gather(
        listen_pubsub(),
        poll_queue(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("worker_stopped")
