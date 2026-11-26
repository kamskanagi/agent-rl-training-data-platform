from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.metrics import feedback_submitted_total
from core.redis_client import get_redis
from core.scoring import consensus_reward, fleiss_kappa, load_reliability, quality_score
from models import FeedbackItem, Task, TaskStatus
from schemas import FeedbackResponse, FeedbackSubmit

QUALITY_EVENT_CHANNEL = "rl:events:feedback"

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


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
        task.iaa = round(fleiss_kappa(rankings), 4)

    reliability = await load_reliability(db, all_feedback)
    task.consensus_reward = round(consensus_reward(all_feedback, reliability), 4)
    task.quality_score = quality_score(task.iaa, all_feedback, task.min_annotations)

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
