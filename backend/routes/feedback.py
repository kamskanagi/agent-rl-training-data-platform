from __future__ import annotations

import json
import math
from itertools import combinations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.metrics import feedback_submitted_total
from core.redis_client import get_redis
from models import FeedbackItem, Task, TaskStatus
from schemas import FeedbackResponse, FeedbackSubmit

QUALITY_EVENT_CHANNEL = "rl:events:feedback"

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


def _fleiss_kappa(rankings: list[list[int]]) -> float:
    """Compute Fleiss' kappa from a list of ranking vectors.

    Each ranking is a list of ints representing the preference order.
    We convert to pairwise agreement matrices.
    """
    if len(rankings) < 2:
        return 0.0

    n_raters = len(rankings)
    n_items = len(rankings[0])

    if n_items < 2:
        return 0.0

    # Count pairwise agreements: for each pair of items, how many raters
    # agree on the ordering
    pairs = list(combinations(range(n_items), 2))
    if not pairs:
        return 0.0

    total_agreement = 0
    total_pairs = len(pairs) * n_raters * (n_raters - 1) / 2

    for i, j in pairs:
        prefer_i = sum(1 for r in rankings if r[i] < r[j])
        prefer_j = n_raters - prefer_i
        # Agreement for this pair
        total_agreement += prefer_i * (prefer_i - 1) / 2 + prefer_j * (prefer_j - 1) / 2

    if total_pairs == 0:
        return 0.0

    p_o = total_agreement / total_pairs  # observed agreement
    p_e = 0.5  # expected agreement under random (binary choice)

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return max(-1.0, min(1.0, kappa))


def _consensus_reward(feedbacks: list[FeedbackItem]) -> float:
    """Reliability-weighted average reward across feedback types."""
    rewards: list[float] = []
    for fb in feedbacks:
        if fb.scalar_reward is not None:
            rewards.append(fb.scalar_reward)
        elif fb.binary_label is not None:
            rewards.append(1.0 if fb.binary_label else 0.0)
        elif fb.ranking is not None and len(fb.ranking) >= 2:
            # Normalise ranking to 0-1 (lower rank = better = higher reward)
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
    """quality_score = 0.4*IAA + 0.4*(1-reward_stdev) + 0.2*coverage"""
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


@router.post("/", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    payload: FeedbackSubmit, db: AsyncSession = Depends(get_db)
):
    task = await db.get(Task, payload.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    feedback = FeedbackItem(
        task_id=payload.task_id,
        annotator_id=payload.annotator_id,
        ranking=payload.ranking,
        scalar_reward=payload.scalar_reward,
        binary_label=payload.binary_label,
        critique_text=payload.critique_text,
        criterion_scores=payload.criterion_scores,
        confidence=payload.confidence,
    )
    db.add(feedback)
    await db.flush()

    feedback_submitted_total.labels(annotation_type=task.annotation_type).inc()

    # Recompute IAA and quality score
    result = await db.execute(
        select(FeedbackItem).where(
            FeedbackItem.task_id == payload.task_id,
            FeedbackItem.flagged == False,  # noqa: E712
        )
    )
    all_feedback = list(result.scalars().all())

    # Fleiss' kappa from rankings
    rankings = [fb.ranking for fb in all_feedback if fb.ranking]
    if len(rankings) >= 2:
        task.iaa = round(_fleiss_kappa(rankings), 4)

    task.consensus_reward = round(_consensus_reward(all_feedback), 4)
    task.quality_score = _quality_score(task.iaa, all_feedback, task.min_annotations)

    # Auto-complete
    if len(all_feedback) >= task.min_annotations and task.status == TaskStatus.PENDING:
        task.status = TaskStatus.COMPLETED

    await db.flush()
    await db.refresh(feedback)

    # Publish event for quality worker
    try:
        r = await get_redis()
        await r.publish(
            QUALITY_EVENT_CHANNEL,
            json.dumps({"task_id": payload.task_id, "feedback_id": feedback.id}),
        )
    except Exception:
        pass  # Worker will pick up via queue fallback

    return feedback


@router.get("/task/{task_id}", response_model=list[FeedbackResponse])
async def get_task_feedback(task_id: str, db: AsyncSession = Depends(get_db)):
    task = await db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    result = await db.execute(
        select(FeedbackItem)
        .where(FeedbackItem.task_id == task_id)
        .order_by(FeedbackItem.created_at.desc())
    )
    return result.scalars().all()


@router.post("/{feedback_id}/flag", response_model=FeedbackResponse)
async def flag_feedback(feedback_id: str, db: AsyncSession = Depends(get_db)):
    fb = await db.get(FeedbackItem, feedback_id)
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")
    fb.flagged = True
    await db.flush()
    await db.refresh(fb)
    return fb
