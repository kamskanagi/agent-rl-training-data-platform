from __future__ import annotations

import json
import os
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import async_session, get_db
from models import Dataset, FeedbackItem, Task, TaskStatus
from schemas import DatasetCreate, DatasetResponse

router = APIRouter(prefix="/api/exports", tags=["exports"])

EXPORT_DIR = os.getenv("EXPORT_DIR", "/exports")


def _task_to_rl_example(task: Task, feedbacks: list[FeedbackItem]) -> dict | None:
    """Convert task + feedback into DPO chosen/rejected format."""
    responses = task.responses or []
    if len(responses) < 2:
        return None

    # Compute average reward per response from rankings
    response_rewards: dict[int, list[float]] = {i: [] for i in range(len(responses))}

    for fb in feedbacks:
        if fb.flagged:
            continue
        if fb.ranking and len(fb.ranking) == len(responses):
            n = len(responses)
            for idx, rank in enumerate(fb.ranking):
                # Lower rank = better = higher reward
                response_rewards[idx].append(1.0 - (rank - 1) / max(1, n - 1))
        elif fb.scalar_reward is not None and len(responses) == 2:
            response_rewards[0].append(fb.scalar_reward)
            response_rewards[1].append(1.0 - fb.scalar_reward)

    avg_rewards = {}
    for idx, rewards in response_rewards.items():
        avg_rewards[idx] = sum(rewards) / len(rewards) if rewards else 0.5

    sorted_indices = sorted(avg_rewards, key=lambda i: avg_rewards[i], reverse=True)
    chosen_idx = sorted_indices[0]
    rejected_idx = sorted_indices[-1]

    chosen_resp = responses[chosen_idx]
    rejected_resp = responses[rejected_idx]

    return {
        "id": task.id,
        "prompt": task.prompt,
        "chosen": chosen_resp.get("text", ""),
        "rejected": rejected_resp.get("text", ""),
        "reward_chosen": round(avg_rewards[chosen_idx], 4),
        "reward_rejected": round(avg_rewards[rejected_idx], 4),
        "task_type": task.annotation_type,
        "quality_score": task.quality_score,
        "iaa": task.iaa,
        "num_annotators": len(set(fb.annotator_id for fb in feedbacks if not fb.flagged)),
        "tags": task.tags or [],
        "evaluation_criteria": task.evaluation_criteria or [],
        "all_responses": [
            {
                "model_id": r.get("model_id", f"model-{i}"),
                "text": r.get("text", ""),
                "avg_reward": round(avg_rewards.get(i, 0.5), 4),
            }
            for i, r in enumerate(responses)
        ],
    }


async def _build_export(dataset_id: str):
    """Background task: build JSONL export file."""
    async with async_session() as db:
        dataset = await db.get(Dataset, dataset_id)
        if not dataset:
            return

        filters = dataset.filters or {}
        query = select(Task).where(Task.status == TaskStatus.COMPLETED)

        if filters.get("min_quality"):
            query = query.where(Task.quality_score >= filters["min_quality"])
        if filters.get("min_iaa"):
            query = query.where(Task.iaa >= filters["min_iaa"])
        if filters.get("annotation_type"):
            query = query.where(Task.annotation_type == filters["annotation_type"])

        result = await db.execute(query)
        tasks = result.scalars().all()

        os.makedirs(EXPORT_DIR, exist_ok=True)
        export_path = os.path.join(EXPORT_DIR, f"{dataset_id}.jsonl")

        examples = []
        for task in tasks:
            fb_result = await db.execute(
                select(FeedbackItem).where(FeedbackItem.task_id == task.id)
            )
            feedbacks = list(fb_result.scalars().all())
            example = _task_to_rl_example(task, feedbacks)
            if example:
                examples.append(example)

        with open(export_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        dataset.export_path = export_path
        dataset.task_count = len(examples)
        dataset.exported_at = datetime.utcnow()
        await db.commit()


@router.post("/datasets", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    payload: DatasetCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    dataset = Dataset(
        name=payload.name,
        filters=payload.filters,
        export_format=payload.export_format,
    )
    db.add(dataset)
    await db.flush()
    await db.refresh(dataset)

    background_tasks.add_task(_build_export, dataset.id)
    return dataset


@router.get("/datasets", response_model=list[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
    return result.scalars().all()


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    dataset = await db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    dataset = await db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.export_path or not os.path.exists(dataset.export_path):
        raise HTTPException(status_code=404, detail="Export not ready")
    return FileResponse(
        dataset.export_path,
        media_type="application/jsonl",
        filename=f"{dataset.name}.jsonl",
    )
