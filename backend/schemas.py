from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ── Task ──────────────────────────────────────────────────────────────

class TaskCreate(BaseModel):
    prompt: str
    responses: list[dict] | None = None
    annotation_type: str = "ranking"
    min_annotations: int = 3
    tags: list[str] | None = None
    evaluation_criteria: list[str] | None = None


class TaskUpdate(BaseModel):
    prompt: str | None = None
    status: str | None = None
    min_annotations: int | None = None
    tags: list[str] | None = None
    evaluation_criteria: list[str] | None = None


class TaskResponse(BaseModel):
    id: str
    prompt: str
    responses: list[dict] | None = None
    annotation_type: str
    status: str
    min_annotations: int
    quality_score: float | None = None
    iaa: float | None = None
    consensus_reward: float | None = None
    tags: list[str] | None = None
    evaluation_criteria: list[str] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int
    page: int
    page_size: int


# ── Feedback ──────────────────────────────────────────────────────────

class FeedbackSubmit(BaseModel):
    task_id: str
    annotator_id: str
    ranking: list[int] | None = None
    scalar_reward: float | None = None
    binary_label: bool | None = None
    critique_text: str | None = None
    criterion_scores: dict[str, float] | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class FeedbackResponse(BaseModel):
    id: str
    task_id: str
    annotator_id: str
    ranking: list[int] | None = None
    scalar_reward: float | None = None
    binary_label: bool | None = None
    critique_text: str | None = None
    criterion_scores: dict[str, float] | None = None
    confidence: float | None = None
    flagged: bool = False
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


# ── Annotator ─────────────────────────────────────────────────────────

class AnnotatorCreate(BaseModel):
    email: str
    name: str
    role: str = "annotator"
    expertise_tags: list[str] | None = None


class AnnotatorResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    expertise_tags: list[str] | None = None
    reliability_score: float
    avg_agreement_rate: float | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


# ── Metrics ───────────────────────────────────────────────────────────

class PlatformMetrics(BaseModel):
    total_tasks: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    total_feedback: int = 0
    total_annotators: int = 0
    avg_quality_score: float | None = None
    avg_iaa: float | None = None
    queue_depth: int = 0


# ── Dataset / Export ──────────────────────────────────────────────────

class DatasetCreate(BaseModel):
    name: str
    filters: dict | None = None
    export_format: str = "jsonl"


class DatasetResponse(BaseModel):
    id: str
    name: str
    filters: dict | None = None
    task_count: int
    reward_distribution: dict | None = None
    export_path: str | None = None
    export_format: str
    exported_at: datetime | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


# ── Training Run ──────────────────────────────────────────────────────

class TrainingRunResponse(BaseModel):
    id: str
    dataset_id: str
    algorithm: str
    config: dict | None = None
    reward_history: list | None = None
    kl_history: list | None = None
    loss_history: list | None = None
    status: str
    created_at: datetime | None = None

    model_config = {"from_attributes": True}
