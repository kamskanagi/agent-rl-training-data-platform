from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import current_annotator
from core.database import get_db
from core.metrics import feedback_submitted_total
from core.redis_client import get_redis, requeue_task
from core.scoring import (
    consensus_reward,
    fleiss_kappa,
    load_reliability,
    quality_score,
    update_reliability,
)
from models import Annotator, FeedbackItem, Task, TaskStatus
from schemas import FeedbackResponse, FeedbackSubmit

QUALITY_EVENT_CHANNEL = "rl:events:feedback"

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post("/", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    payload: FeedbackSubmit,
    annotator: Annotator = Depends(current_annotator),
    db: AsyncSession = Depends(get_db),
):
    if payload.annotator_id is not None and payload.annotator_id != annotator.id:
        raise HTTPException(status_code=403, detail="annotator_id does not match your token")

    task = await db.get(Task, payload.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    _validate_ranking(payload.ranking, task)

    existing = await db.execute(
        select(FeedbackItem.id).where(
            FeedbackItem.task_id == task.id, FeedbackItem.annotator_id == annotator.id
        )
    )
    if existing.first():
        raise HTTPException(status_code=409, detail="You already submitted feedback for this task")

    feedback = FeedbackItem(
        task_id=payload.task_id,
        annotator_id=annotator.id,
        ranking=payload.ranking,
        scalar_reward=payload.scalar_reward,
        binary_label=payload.binary_label,
        critique_text=payload.critique_text,
        criterion_scores=payload.criterion_scores,
        confidence=payload.confidence,
    )
    db.add(feedback)
    try:
        await db.flush()
    except IntegrityError:  # a concurrent duplicate slipped past the check above
        raise HTTPException(status_code=409, detail="You already submitted feedback for this task")

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
        task.iaa = round(fleiss_kappa(rankings), 4)

    # Everyone on this task has a new agreement sample, so refresh their reliability
    # before using it to weight the consensus.
    await update_reliability(db, {fb.annotator_id for fb in all_feedback})
    reliability = await load_reliability(db, all_feedback)
    task.consensus_reward = round(consensus_reward(all_feedback, reliability), 4)
    task.quality_score = quality_score(task.iaa, all_feedback, task.min_annotations)

    # Auto-complete
    if len(all_feedback) >= task.min_annotations and task.status == TaskStatus.PENDING:
        task.status = TaskStatus.COMPLETED

    await db.flush()
    await db.refresh(feedback)

    # Still needs annotations: put it back in the queue for the next annotator.
    if task.status == TaskStatus.PENDING:
        try:
            await requeue_task(task.id)
        except Exception:
            pass  # Redis down: the task can be re-queued later; the feedback is saved

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


def _validate_ranking(ranking: list[int] | None, task: Task) -> None:
    """A ranking gives each response a distinct rank 1..n (1 = best), indexed by response."""
    if ranking is None:
        return
    n = len(task.responses or [])
    if len(ranking) < 2 or (n and len(ranking) != n) or sorted(ranking) != list(range(1, len(ranking) + 1)):
        raise HTTPException(
            status_code=422,
            detail=f"ranking must give each of the task's {n or len(ranking)} responses "
                   f"a distinct rank from 1 (best) to {n or len(ranking)}",
        )


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
    # A flagged item counts as zero agreement for its author, and it changes the
    # comparison set for everyone else on the task.
    result = await db.execute(select(FeedbackItem.annotator_id).where(FeedbackItem.task_id == fb.task_id))
    await update_reliability(db, set(result.scalars().all()))
    await db.refresh(fb)
    return fb
