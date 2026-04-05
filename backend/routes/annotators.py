from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.redis_client import dequeue_task
from models import Annotator, Task, TaskAssignment
from schemas import AnnotatorCreate, AnnotatorResponse, TaskResponse

router = APIRouter(prefix="/api/annotators", tags=["annotators"])


@router.post("/", response_model=AnnotatorResponse, status_code=201)
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
    db.add(annotator)
    await db.flush()
    await db.refresh(annotator)
    return annotator


@router.get("/", response_model=list[AnnotatorResponse])
async def list_annotators(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Annotator).order_by(Annotator.created_at.desc()))
    return result.scalars().all()


@router.get("/{annotator_id}", response_model=AnnotatorResponse)
async def get_annotator(annotator_id: str, db: AsyncSession = Depends(get_db)):
    annotator = await db.get(Annotator, annotator_id)
    if not annotator:
        raise HTTPException(status_code=404, detail="Annotator not found")
    return annotator


@router.get("/{annotator_id}/next-task", response_model=TaskResponse)
async def next_task(annotator_id: str, db: AsyncSession = Depends(get_db)):
    annotator = await db.get(Annotator, annotator_id)
    if not annotator:
        raise HTTPException(status_code=404, detail="Annotator not found")

    task_id = await dequeue_task(timeout=1)
    if not task_id:
        raise HTTPException(status_code=404, detail="No tasks in queue")

    task = await db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    assignment = TaskAssignment(task_id=task.id, annotator_id=annotator_id)
    db.add(assignment)
    await db.flush()

    return task
