from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class AnnotationType(str, enum.Enum):
    RANKING = "ranking"
    SCALAR = "scalar"
    BINARY = "binary"
    CRITIQUE = "critique"


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FLAGGED = "flagged"


class TrainingStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    responses: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    annotation_type: Mapped[str] = mapped_column(
        Enum(AnnotationType, name="annotation_type_enum"), default=AnnotationType.RANKING
    )
    status: Mapped[str] = mapped_column(
        Enum(TaskStatus, name="task_status_enum"), default=TaskStatus.PENDING
    )
    min_annotations: Mapped[int] = mapped_column(Integer, default=3)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    iaa: Mapped[float | None] = mapped_column(Float, nullable=True)
    consensus_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    tags: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    evaluation_criteria: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    feedback_items: Mapped[list[FeedbackItem]] = relationship(back_populates="task")
    assignments: Mapped[list[TaskAssignment]] = relationship(back_populates="task")


class FeedbackItem(Base):
    __tablename__ = "feedback_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("tasks.id"), nullable=False)
    annotator_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("annotators.id"), nullable=False
    )
    ranking: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scalar_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    binary_label: Mapped[bool | None] = mapped_column(nullable=True)
    critique_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    criterion_scores: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    flagged: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    task: Mapped[Task] = relationship(back_populates="feedback_items")
    annotator: Mapped[Annotator] = relationship(back_populates="feedback_items")


class Annotator(Base):
    __tablename__ = "annotators"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="annotator")
    expertise_tags: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reliability_score: Mapped[float] = mapped_column(Float, default=1.0)
    avg_agreement_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    feedback_items: Mapped[list[FeedbackItem]] = relationship(back_populates="annotator")
    assignments: Mapped[list[TaskAssignment]] = relationship(back_populates="annotator")


class TaskAssignment(Base):
    __tablename__ = "task_assignments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("tasks.id"), nullable=False)
    annotator_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("annotators.id"), nullable=False
    )
    assigned_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    time_spent_sec: Mapped[int | None] = mapped_column(Integer, nullable=True)

    task: Mapped[Task] = relationship(back_populates="assignments")
    annotator: Mapped[Annotator] = relationship(back_populates="assignments")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    filters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    task_count: Mapped[int] = mapped_column(Integer, default=0)
    reward_distribution: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    export_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    export_format: Mapped[str] = mapped_column(String(50), default="jsonl")
    exported_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id"), nullable=False
    )
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reward_history: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    kl_history: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    loss_history: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum(TrainingStatus, name="training_status_enum"), default=TrainingStatus.QUEUED
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    dataset: Mapped[Dataset] = relationship()
