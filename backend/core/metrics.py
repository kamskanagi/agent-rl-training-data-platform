from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import Info

# ── Custom metrics ────────────────────────────────────────────────────

feedback_submitted_total = Counter(
    "rl_feedback_submitted_total",
    "Total number of feedback submissions",
    ["annotation_type"],
)

tasks_created_total = Counter(
    "rl_tasks_created_total",
    "Total number of tasks created",
)

tasks_completed_total = Counter(
    "rl_tasks_completed_total",
    "Total number of tasks that reached completed status",
)

exports_created_total = Counter(
    "rl_exports_created_total",
    "Total number of dataset exports created",
    ["export_format"],
)

annotation_queue_depth = Gauge(
    "rl_annotation_queue_depth",
    "Current depth of the annotation queue",
)

active_annotators = Gauge(
    "rl_active_annotators",
    "Number of annotators who submitted feedback in the last hour",
)

quality_score_histogram = Histogram(
    "rl_task_quality_score",
    "Distribution of task quality scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

feedback_latency_seconds = Histogram(
    "rl_feedback_processing_seconds",
    "Time taken to process a feedback submission",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


# ── Instrumentator setup ─────────────────────────────────────────────

def _request_size(info: Info) -> None:
    """Track request body sizes."""
    pass  # Default instrumentator already tracks this


def setup_instrumentator() -> Instrumentator:
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/health", "/metrics"],
        env_var_name="ENABLE_METRICS",
    )
    return instrumentator
