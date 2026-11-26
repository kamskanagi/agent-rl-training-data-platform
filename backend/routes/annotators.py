from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import current_annotator, issue_token
from core.database import get_db
from core.redis_client import dequeue_task, requeue_task
from models import Annotator, FeedbackItem, Task, TaskAssignment, TaskStatus
from schemas import AnnotatorCreate, AnnotatorCreated, AnnotatorResponse, TaskResponse

router = APIRouter(prefix="/api/annotators", tags=["annotators"])

# How many queue entries next-task inspects before giving up.
MAX_QUEUE_SCAN = 50


@router.post("/", response_model=AnnotatorCreated, status_code=201)
async def create_annotator(
    payload: AnnotatorCreate, db: AsyncSession = Depends(get_db)
):
    # Check for duplicate email
    existing = await db.execute(
        select(Annotator).where(Annotator.email == payload.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    annotator = Annotator(
        email=payload.email,
        name=payload.name,
        role=payload.role,
        expertise_tags=payload.expertise_tags,
    )
    token = issue_token(annotator)
    db.add(annotator)
    await db.flush()
    await db.refresh(annotator)
    return AnnotatorCreated(
        **AnnotatorResponse.model_validate(annotator).model_dump(), api_token=token
    )


@router.get("/", response_model=list[AnnotatorResponse])
async def list_annotators(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Annotator).order_by(Annotator.created_at.desc()))
    return result.scalars().all()


@router.get("/me", response_model=AnnotatorResponse)
async def get_me(annotator: Annotator = Depends(current_annotator)):
    return annotator


@router.get("/{annotator_id}", response_model=AnnotatorResponse)
async def get_annotator(annotator_id: str, db: AsyncSession = Depends(get_db)):
    annotator = await db.get(Annotator, annotator_id)
    if not annotator:
        raise HTTPException(status_code=404, detail="Annotator not found")
    return annotator


@router.get("/{annotator_id}/next-task", response_model=TaskResponse)
async def next_task(
    annotator_id: str,
    annotator: Annotator = Depends(current_annotator),
    db: AsyncSession = Depends(get_db),
):
    if annotator_id != annotator.id:
        raise HTTPException(status_code=403, detail="You can only fetch tasks for yourself")

    done = await db.execute(
        select(FeedbackItem.task_id).where(FeedbackItem.annotator_id == annotator.id)
    )
    already_annotated = set(done.scalars().all())

    # Pop until we find a pending task this annotator hasn't done. Pending tasks that
    # were skipped go back on the queue for other annotators.
    skipped: list[str] = []
    found: Task | None = None
    for _ in range(MAX_QUEUE_SCAN):
        task_id = await dequeue_task(timeout=1)
        if not task_id:
            break
        task = await db.get(Task, task_id)
        if not task or task.status != TaskStatus.PENDING:
            continue  # deleted or finished: drop it from the queue
        if task.id in already_annotated:
            if task.id in skipped:
                break  # cycled through the whole queue; re-queued once below
            skipped.append(task.id)
            continue
        found = task
        break

    for task_id in skipped:
        await requeue_task(task_id)

    if not found:
        raise HTTPException(status_code=404, detail="No tasks in queue")

    db.add(TaskAssignment(task_id=found.id, annotator_id=annotator.id))
    await db.flush()
    return found
