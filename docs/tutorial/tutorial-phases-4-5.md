# Building an RL Training Data Platform: Phases 4 and 5

> Upgrade the backend with a real-time quality worker, Parquet and HuggingFace exports, a seed script, and full production observability via Prometheus and Grafana.

**What you'll build:** A background worker that continuously recomputes inter-annotator agreement and quality scores using Redis Pub/Sub, two new binary export formats (Parquet and HuggingFace Datasets), a seed script that populates the database with 50 synthetic tasks and realistic feedback, and a complete observability stack with structured JSON logging, Prometheus custom metrics, and a pre-wired Grafana dashboard.

**Tech stack:** Python 3.12, asyncio, Redis Pub/Sub, pyarrow, structlog, prometheus-client, prometheus-fastapi-instrumentator, Prometheus 2.51, Grafana 10.4

**Prerequisites:** Phases 1, 2, and 3 complete — all backend routes and the React frontend functional, Docker Compose stack running.

**Time estimate:** 2-3 hours

**Difficulty:** Intermediate

**Final repo:** https://github.com/kamskanagi/agent-rl-training-data-platform

---

## Table of Contents

1. [Phase 4: Quality Worker, Exports, and Seed Data](#phase-4-quality-worker-exports-and-seed-data)
   - [Step 4.1: Add pyarrow to Dependencies](#step-41-add-pyarrow-to-dependencies)
   - [Step 4.2: Replace the Worker Placeholder with a Real Consumer](#step-42-replace-the-worker-placeholder-with-a-real-consumer)
   - [Step 4.3: Publish Feedback Events from the Feedback Route](#step-43-publish-feedback-events-from-the-feedback-route)
   - [Step 4.4: Add Parquet and HuggingFace Export Formats](#step-44-add-parquet-and-huggingface-export-formats)
   - [Step 4.5: Create the Seed Script](#step-45-create-the-seed-script)
   - [Step 4.6: Update the Test Fixture to Mock Redis in the Feedback Route](#step-46-update-the-test-fixture-to-mock-redis-in-the-feedback-route)
   - [Verify Phase 4](#verify-phase-4)
2. [Phase 5: Observability — Structured Logging and Prometheus Metrics](#phase-5-observability--structured-logging-and-prometheus-metrics)
   - [Step 5.1: Add Observability Dependencies](#step-51-add-observability-dependencies)
   - [Step 5.2: Structured Logging Module](#step-52-structured-logging-module)
   - [Step 5.3: Prometheus Metrics Module](#step-53-prometheus-metrics-module)
   - [Step 5.4: Wire Logging and Metrics into main.py](#step-54-wire-logging-and-metrics-into-mainpy)
   - [Step 5.5: Add Prometheus Counters to Route Modules](#step-55-add-prometheus-counters-to-route-modules)
   - [Step 5.6: Add Structured Logging and Metrics to the Quality Worker](#step-56-add-structured-logging-and-metrics-to-the-quality-worker)
   - [Step 5.7: Prometheus Configuration](#step-57-prometheus-configuration)
   - [Step 5.8: Grafana Provisioning and Dashboard](#step-58-grafana-provisioning-and-dashboard)
   - [Step 5.9: Add Prometheus and Grafana to Docker Compose](#step-59-add-prometheus-and-grafana-to-docker-compose)
   - [Verify Phase 5](#verify-phase-5)
3. [Updated Project Structure](#updated-project-structure)
4. [Environment Variables Reference](#environment-variables-reference)
5. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
6. [Next Steps](#next-steps)

---

## Phase 4: Quality Worker, Exports, and Seed Data

### What We're Building

Phase 4 replaces the quality worker stub with a process that actually does work: it listens on a Redis Pub/Sub channel for feedback events and recomputes inter-annotator agreement (IAA), consensus reward, and quality scores in real time. We also extend the exports route with two production-grade formats — Apache Parquet (columnar binary) and HuggingFace Datasets (JSONL + metadata JSON). Finally, a seed script populates a fresh database with 50 synthetic tasks and realistic annotator feedback so the frontend and export endpoints have something meaningful to display.

### Prerequisites

Phases 1, 2, and 3 complete. The Docker stack must start cleanly with `docker compose up --build`.

---

### Step 4.1: Add pyarrow to Dependencies

**Why:** The Parquet export format requires Apache Arrow's Python library. pyarrow is a binary dependency — it is not pulled in transitively by any existing package — so we must pin it explicitly. We require version 15 or later because earlier versions have a breaking API change in how empty-schema tables are constructed.

**Edit `backend/requirements.txt`** — add the pyarrow line at the end:

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
sqlalchemy[asyncio]>=2.0.25
asyncpg>=0.29.0
redis>=5.0.0
pydantic>=2.5.0
python-dotenv>=1.0.0
httpx>=0.27.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
aiosqlite>=0.20.0
pyarrow>=15.0.0
```

---

### Step 4.2: Replace the Worker Placeholder with a Real Consumer

**Why:** The Phase 1 worker was a placeholder that did nothing. The real worker needs two entry points for task processing:

1. **Redis Pub/Sub** — the feedback route publishes an event on the `rl:events:feedback` channel the moment a feedback item is saved. The worker picks this up with near-zero latency.
2. **Queue polling** — a fallback `BRPOP` loop on the annotation queue handles any events that were missed if the worker was temporarily down. `BRPOP` blocks for up to 5 seconds before timing out, which is CPU-friendly.

Both listeners run concurrently in the same asyncio event loop with `asyncio.gather`. The actual quality-scoring logic — Fleiss' kappa, consensus reward, and the weighted quality formula — lives in pure Python helper functions that are shared with the feedback route.

**Replace `backend/workers/quality_worker.py`** with the full implementation below:

```python
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
```

> **What's happening here:**
>
> **`_fleiss_kappa` — pairwise agreement kappa:** Fleiss' kappa measures how much raters agree beyond what is expected by chance. For ranking tasks we reduce the problem to pairwise binary choices: for every pair of responses `(i, j)`, each rater either prefers `i` over `j` or vice versa. We count how often raters agree on each pair, then compute `kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)`. Since this is a binary choice, the expected agreement by chance is always 0.5. A kappa of 1.0 means perfect agreement; 0.0 means agreement no better than chance; negative values mean raters systematically disagree.
>
> **`_consensus_reward` — normalizing mixed feedback types:** Not all tasks use the same annotation type. Scalar feedback is already a 0-1 float; binary labels are converted to 1.0/0.0; rankings are linearly scaled so rank 1 becomes reward 1.0 and the worst rank becomes 0.0. This normalization lets us average across annotation types.
>
> **`_quality_score` — the weighted formula:** `quality = 0.4 * IAA + 0.4 * (1 - reward_stdev) + 0.2 * coverage`. The three components capture three dimensions of quality: annotator agreement (IAA), internal consistency of reward signals (low stdev = consistent), and completeness (coverage = fraction of required annotations received).
>
> **Auto-complete and auto-flag:** When `len(all_feedback) >= task.min_annotations`, the task transitions from `PENDING` to `COMPLETED`. If the quality score is below 0.3 at that point, it is immediately marked `FLAGGED` instead. These state transitions happen in the worker so the feedback route stays free of side-effect logic.
>
> **`listen_pubsub` and `poll_queue` running with `asyncio.gather`:** Both coroutines run in the same event loop. `listen_pubsub` is the fast path — it reacts within milliseconds of a feedback submission. `poll_queue` is a safety net: if the worker was restarted while events were in flight, the queue retains those task IDs and `brpop` will drain them. The two paths are not mutually exclusive — a task may be processed twice if both fire, but the recomputation is idempotent.
>
> **`setup_logging()` and `get_logger()`** are imported from `core.logging`, which we will create in Phase 5. For now, note that these calls are present in the worker so that both phases can be committed together. The implementation is explained in Step 5.2.

---

### Step 4.3: Publish Feedback Events from the Feedback Route

**Why:** The worker needs to know when new feedback arrives. We add a `redis.publish()` call at the end of `submit_feedback`. The call is wrapped in a `try/except` with a silent pass — if Redis is temporarily unavailable, the worker will still pick up the task via the queue polling fallback. This design keeps the HTTP request from failing due to a Redis outage.

**Replace `backend/routes/feedback.py`** with the full updated version:

```python
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

    # Recompute IAA and quality score inline (fast path)
    result = await db.execute(
        select(FeedbackItem).where(
            FeedbackItem.task_id == payload.task_id,
            FeedbackItem.flagged == False,  # noqa: E712
        )
    )
    all_feedback = list(result.scalars().all())

    rankings = [fb.ranking for fb in all_feedback if fb.ranking]
    if len(rankings) >= 2:
        task.iaa = round(_fleiss_kappa(rankings), 4)

    task.consensus_reward = round(_consensus_reward(all_feedback), 4)
    task.quality_score = _quality_score(task.iaa, all_feedback, task.min_annotations)

    if len(all_feedback) >= task.min_annotations and task.status == TaskStatus.PENDING:
        task.status = TaskStatus.COMPLETED

    await db.flush()
    await db.refresh(feedback)

    # Publish event for quality worker (best-effort)
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
```

> **What's happening here:**
>
> **Dual-write design — inline and async:** The route still recomputes IAA and quality score synchronously before returning the response. This means the HTTP caller always gets back an up-to-date `quality_score` on the task. The Pub/Sub publish is an additional notification to the worker so it can persist a more thorough recomputation (including flagging logic) without blocking the request.
>
> **`feedback_submitted_total.labels(annotation_type=...).inc()`** — this Prometheus counter tracks how many feedback submissions arrived per annotation type. We call `.labels()` before `.inc()` to attach the dimension. The `feedback_submitted_total` counter is imported from `core.metrics`, which is created in Phase 5 Step 5.3. The import will resolve once that module exists.
>
> **Silent `except Exception: pass`** — we do not propagate Redis errors to the caller. The annotator's submission must never fail because of an infrastructure issue. The worker's `brpop` fallback will catch any events that were not delivered via Pub/Sub.

---

### Step 4.4: Add Parquet and HuggingFace Export Formats

**Why:** The DPO training pipeline used in Phase 2 exported only JSONL. Parquet is the standard format for large ML datasets: it is columnar (fast to read selected columns), compressed by default, and natively supported by pandas, HuggingFace Datasets, and PyTorch data loaders. The HuggingFace Datasets format adds a `dataset_info.json` metadata file alongside the JSONL, making the directory directly loadable with `datasets.load_dataset("path/to/dir")`. We also compute a reward distribution summary at export time so the caller can inspect the range without downloading the full file.

**Replace `backend/routes/exports.py`** with the full updated version:

```python
from __future__ import annotations

import json
import os
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import async_session, get_db
from core.metrics import exports_created_total
from models import Dataset, FeedbackItem, Task, TaskStatus
from schemas import DatasetCreate, DatasetResponse

router = APIRouter(prefix="/api/exports", tags=["exports"])

EXPORT_DIR = os.getenv("EXPORT_DIR", "/exports")


def _write_parquet(examples: list[dict], path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not examples:
        # Write empty parquet with the expected schema
        schema = pa.schema([
            ("id", pa.string()),
            ("prompt", pa.string()),
            ("chosen", pa.string()),
            ("rejected", pa.string()),
            ("reward_chosen", pa.float64()),
            ("reward_rejected", pa.float64()),
        ])
        table = pa.table({}, schema=schema)
        pq.write_table(table, path)
        return

    rows = {
        "id": [ex["id"] for ex in examples],
        "prompt": [ex["prompt"] for ex in examples],
        "chosen": [ex["chosen"] for ex in examples],
        "rejected": [ex["rejected"] for ex in examples],
        "reward_chosen": [ex["reward_chosen"] for ex in examples],
        "reward_rejected": [ex["reward_rejected"] for ex in examples],
        "task_type": [ex.get("task_type", "") for ex in examples],
        "quality_score": [ex.get("quality_score") for ex in examples],
        "iaa": [ex.get("iaa") for ex in examples],
        "num_annotators": [ex.get("num_annotators", 0) for ex in examples],
    }
    table = pa.table(rows)
    pq.write_table(table, path)


def _write_huggingface(examples: list[dict], path: str, name: str) -> None:
    """Write in HuggingFace Datasets format (JSONL + dataset_info.json)."""
    os.makedirs(path, exist_ok=True)

    data_path = os.path.join(path, "train.jsonl")
    with open(data_path, "w") as f:
        for ex in examples:
            row = {
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
                "reward_chosen": ex["reward_chosen"],
                "reward_rejected": ex["reward_rejected"],
            }
            f.write(json.dumps(row) + "\n")

    info = {
        "dataset_name": name,
        "description": "DPO preference dataset exported from RL Training Data Platform",
        "features": {
            "prompt": {"dtype": "string", "_type": "Value"},
            "chosen": {"dtype": "string", "_type": "Value"},
            "rejected": {"dtype": "string", "_type": "Value"},
            "reward_chosen": {"dtype": "float64", "_type": "Value"},
            "reward_rejected": {"dtype": "float64", "_type": "Value"},
        },
        "splits": {
            "train": {"num_examples": len(examples)},
        },
        "task_categories": ["text-generation"],
        "tags": ["dpo", "rlhf", "preference"],
    }
    with open(os.path.join(path, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def _task_to_rl_example(task: Task, feedbacks: list[FeedbackItem]) -> dict | None:
    """Convert task + feedback into DPO chosen/rejected format."""
    responses = task.responses or []
    if len(responses) < 2:
        return None

    response_rewards: dict[int, list[float]] = {i: [] for i in range(len(responses))}

    for fb in feedbacks:
        if fb.flagged:
            continue
        if fb.ranking and len(fb.ranking) == len(responses):
            n = len(responses)
            for idx, rank in enumerate(fb.ranking):
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
    """Background task: build the export file in the requested format."""
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

        export_format = dataset.export_format or "jsonl"

        if export_format == "parquet":
            export_path = os.path.join(EXPORT_DIR, f"{dataset_id}.parquet")
            _write_parquet(examples, export_path)
        elif export_format == "huggingface":
            export_path = os.path.join(EXPORT_DIR, f"{dataset_id}_hf")
            _write_huggingface(examples, export_path, dataset.name)
        else:
            with open(export_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

        # Compute reward distribution summary
        rewards = [ex["reward_chosen"] for ex in examples if "reward_chosen" in ex]
        reward_dist = None
        if rewards:
            reward_dist = {
                "min": round(min(rewards), 4),
                "max": round(max(rewards), 4),
                "mean": round(sum(rewards) / len(rewards), 4),
                "count": len(rewards),
            }

        dataset.export_path = export_path
        dataset.task_count = len(examples)
        dataset.reward_distribution = reward_dist
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

    exports_created_total.labels(export_format=payload.export_format).inc()
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

    fmt = dataset.export_format or "jsonl"

    if fmt == "parquet":
        return FileResponse(
            dataset.export_path,
            media_type="application/octet-stream",
            filename=f"{dataset.name}.parquet",
        )
    elif fmt == "huggingface":
        hf_data = os.path.join(dataset.export_path, "train.jsonl")
        if not os.path.exists(hf_data):
            raise HTTPException(status_code=404, detail="Export not ready")
        return FileResponse(
            hf_data,
            media_type="application/jsonl",
            filename=f"{dataset.name}_train.jsonl",
        )
    else:
        return FileResponse(
            dataset.export_path,
            media_type="application/jsonl",
            filename=f"{dataset.name}.jsonl",
        )
```

> **What's happening here:**
>
> **`_write_parquet` — deferred import:** `import pyarrow` happens inside the function, not at module import time. This keeps the route loadable in test environments where pyarrow may not be installed. The function is only called when the export format is `"parquet"`, so test coverage for JSONL and HuggingFace exports is unaffected.
>
> **Empty-table schema for Parquet:** When `examples` is empty, we cannot infer a schema from the data. pyarrow requires an explicit schema in this case. We define the minimum columns (`id`, `prompt`, `chosen`, `rejected`, `reward_chosen`, `reward_rejected`) so the file is a valid Parquet file with zero rows.
>
> **`_write_huggingface` — directory layout:** The HuggingFace Datasets library expects a directory containing at least `train.jsonl` and `dataset_info.json`. The `dataset_info.json` describes column types using the HuggingFace `Value` feature descriptor. After running this function, the directory can be loaded directly with `datasets.load_dataset("path/to/dir", data_files="train.jsonl")`.
>
> **`reward_distribution` — summary statistics:** The export background task computes min, max, mean, and count over the `reward_chosen` column. This is stored on the `Dataset` row so the API caller can assess data quality without downloading the file. A dataset where min ≈ max ≈ 0.5 suggests annotators had no strong preferences, which is a signal to re-examine task design.
>
> **`exports_created_total.labels(export_format=...).inc()`** — similar to the feedback counter, this is a labelled Prometheus counter. We label by `export_format` so the Grafana dashboard can break down exports by type.

---

### Step 4.5: Create the Seed Script

**Why:** An empty database makes it impossible to exercise the full pipeline — the frontend shows blank charts, the export endpoints return empty files, and quality metrics have nothing to aggregate. The seed script generates a statistically realistic dataset: 8 annotators with varying reliability scores, 50 tasks spread across all annotation types and statuses, feedback counts that match each status (pending tasks have 0-1 feedback, completed tasks have at least `min_annotations`), pre-computed quality scores, 3 sample datasets, and 1 training run with synthetic reward curves.

The script uses `random.seed(42)` for reproducibility — every run on a fresh database produces the same rows.

**Create `scripts/seed.py`:**

```python
"""Seed script: populate the database with 50 synthetic tasks and realistic feedback.

Usage:
    python scripts/seed.py                   # default: DATABASE_URL from env
    DATABASE_URL=... python scripts/seed.py  # explicit connection
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import uuid

# Ensure backend modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import text
from core.database import async_session, init_db, engine
from models import (
    Annotator,
    AnnotationType,
    Dataset,
    FeedbackItem,
    Task,
    TaskAssignment,
    TaskStatus,
    TrainingRun,
    TrainingStatus,
)

random.seed(42)

# ── Prompt templates ──────────────────────────────────────────────────

PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in simple terms.",
    "Write a Python function that {task}.",
    "Compare and contrast {topic_a} with {topic_b}.",
    "Summarize the key points of {topic}.",
    "What are the pros and cons of {topic}?",
    "Generate a creative story about {topic}.",
    "Debug the following code:\n```python\n{code}\n```",
    "Translate the following to {language}: {text}",
    "Provide step-by-step instructions for {task}.",
    "What would happen if {scenario}?",
]

TOPICS = [
    "machine learning", "quantum computing", "climate change", "blockchain",
    "neural networks", "reinforcement learning", "natural language processing",
    "computer vision", "distributed systems", "cybersecurity", "gene editing",
    "autonomous vehicles", "microservices", "containerization", "edge computing",
    "federated learning", "transformer architecture", "attention mechanism",
    "gradient descent", "backpropagation", "transfer learning", "data augmentation",
    "model compression", "knowledge distillation", "prompt engineering",
]

TASKS = [
    "sorts a list using quicksort", "implements binary search",
    "calculates the Fibonacci sequence", "validates email addresses",
    "parses JSON from a string", "implements a simple LRU cache",
    "finds the shortest path in a graph", "merges two sorted arrays",
    "computes the edit distance between two strings",
    "implements a basic tokenizer",
]

MODEL_NAMES = ["gpt-4", "claude-3", "llama-70b", "gemini-pro", "mixtral-8x7b"]

ANNOTATOR_NAMES = [
    ("Alice Chen", "alice@example.com"),
    ("Bob Martinez", "bob@example.com"),
    ("Carol Wang", "carol@example.com"),
    ("David Kim", "david@example.com"),
    ("Eve Johnson", "eve@example.com"),
    ("Frank Brown", "frank@example.com"),
    ("Grace Lee", "grace@example.com"),
    ("Henry Taylor", "henry@example.com"),
]

TAGS = [
    ["coding", "python"], ["explanation", "beginner"], ["comparison", "analysis"],
    ["creative", "writing"], ["debugging", "python"], ["translation"],
    ["tutorial", "step-by-step"], ["reasoning", "logic"],
    ["math", "algorithms"], ["nlp", "text"],
]

EVALUATION_CRITERIA = [
    ["accuracy", "clarity", "completeness"],
    ["helpfulness", "harmlessness", "honesty"],
    ["relevance", "coherence", "fluency"],
    ["correctness", "efficiency", "readability"],
]


def _uid() -> str:
    return str(uuid.uuid4())


def _generate_prompt(idx: int) -> str:
    template = PROMPT_TEMPLATES[idx % len(PROMPT_TEMPLATES)]
    return template.format(
        topic=random.choice(TOPICS),
        topic_a=random.choice(TOPICS),
        topic_b=random.choice(TOPICS),
        task=random.choice(TASKS),
        code="def fib(n): return fib(n-1) + fib(n-2)",
        language=random.choice(["Spanish", "French", "Japanese", "German"]),
        text="Hello, how are you today?",
        scenario=f"{random.choice(TOPICS)} became mainstream overnight",
    )


def _generate_responses(n: int = 2) -> list[dict]:
    models = random.sample(MODEL_NAMES, min(n, len(MODEL_NAMES)))
    return [
        {
            "model_id": model,
            "text": f"This is a synthetic response from {model}. "
            f"It demonstrates a {random.choice(['detailed', 'concise', 'thorough', 'creative'])} "
            f"approach to answering the question with "
            f"{random.choice(['examples', 'analogies', 'step-by-step reasoning', 'code snippets'])}.",
        }
        for model in models
    ]


async def seed():
    await init_db()

    async with async_session() as db:
        # Guard: skip if data already exists
        result = await db.execute(text("SELECT count(*) FROM tasks"))
        count = result.scalar()
        if count and count > 0:
            print(f"Database already has {count} tasks. Skipping seed.")
            return

        # ── Create annotators ─────────────────────────────────────────
        annotators = []
        for name, email in ANNOTATOR_NAMES:
            a = Annotator(
                id=_uid(),
                email=email,
                name=name,
                role=random.choice(["annotator", "annotator", "annotator", "senior_annotator"]),
                expertise_tags=random.choice(TAGS),
                reliability_score=round(random.uniform(0.7, 1.0), 2),
            )
            db.add(a)
            annotators.append(a)
        await db.flush()
        print(f"Created {len(annotators)} annotators.")

        # ── Create 50 tasks ───────────────────────────────────────────
        tasks = []
        for i in range(50):
            n_responses = random.choice([2, 2, 2, 3, 4])
            annotation_type = random.choice(list(AnnotationType))
            status = random.choices(
                [TaskStatus.PENDING, TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS, TaskStatus.FLAGGED],
                weights=[0.15, 0.55, 0.2, 0.1],
            )[0]

            task = Task(
                id=_uid(),
                prompt=_generate_prompt(i),
                responses=_generate_responses(n_responses),
                annotation_type=annotation_type,
                status=status,
                min_annotations=random.choice([3, 3, 3, 5]),
                tags=random.choice(TAGS),
                evaluation_criteria=random.choice(EVALUATION_CRITERIA),
            )
            db.add(task)
            tasks.append(task)
        await db.flush()
        print(f"Created {len(tasks)} tasks.")

        # ── Create feedback ───────────────────────────────────────────
        feedback_count = 0
        for task in tasks:
            if task.status in (TaskStatus.PENDING,):
                n_feedback = random.randint(0, 1)
            elif task.status == TaskStatus.IN_PROGRESS:
                n_feedback = random.randint(1, task.min_annotations - 1)
            else:
                n_feedback = random.randint(task.min_annotations, task.min_annotations + 2)

            task_annotators = random.sample(annotators, min(n_feedback, len(annotators)))

            for annotator in task_annotators:
                n_resp = len(task.responses) if task.responses else 2

                fb = FeedbackItem(
                    id=_uid(),
                    task_id=task.id,
                    annotator_id=annotator.id,
                    confidence=round(random.uniform(0.5, 1.0), 2),
                    flagged=random.random() < 0.05,
                )

                if task.annotation_type == AnnotationType.RANKING:
                    fb.ranking = random.sample(range(1, n_resp + 1), n_resp)
                elif task.annotation_type == AnnotationType.SCALAR:
                    fb.scalar_reward = round(random.uniform(0.0, 1.0), 4)
                elif task.annotation_type == AnnotationType.BINARY:
                    fb.binary_label = random.choice([True, False])
                elif task.annotation_type == AnnotationType.CRITIQUE:
                    fb.critique_text = random.choice([
                        "Response A is more detailed and accurate.",
                        "Response B provides better examples.",
                        "Both responses are adequate but A has fewer errors.",
                        "Neither response fully addresses the question.",
                        "Response A is clearly superior in clarity and depth.",
                    ])
                    fb.scalar_reward = round(random.uniform(0.3, 0.9), 4)

                db.add(fb)
                feedback_count += 1

                assignment = TaskAssignment(
                    id=_uid(),
                    task_id=task.id,
                    annotator_id=annotator.id,
                    time_spent_sec=random.randint(30, 600),
                )
                db.add(assignment)

        await db.flush()
        print(f"Created {feedback_count} feedback items.")

        # ── Compute quality scores for completed tasks ────────────────
        from itertools import combinations
        import math

        for task in tasks:
            result = await db.execute(
                FeedbackItem.__table__.select().where(
                    FeedbackItem.task_id == task.id,
                    FeedbackItem.flagged == False,  # noqa: E712
                )
            )
            all_fb_rows = result.fetchall()

            if not all_fb_rows:
                continue

            rankings = [row.ranking for row in all_fb_rows if row.ranking]
            if len(rankings) >= 2:
                n_raters = len(rankings)
                n_items = len(rankings[0])
                if n_items >= 2:
                    pairs = list(combinations(range(n_items), 2))
                    total_agreement = 0
                    total_pairs = len(pairs) * n_raters * (n_raters - 1) / 2
                    for ii, jj in pairs:
                        prefer_i = sum(1 for r in rankings if r[ii] < r[jj])
                        prefer_j = n_raters - prefer_i
                        total_agreement += prefer_i * (prefer_i - 1) / 2 + prefer_j * (prefer_j - 1) / 2
                    if total_pairs > 0:
                        p_o = total_agreement / total_pairs
                        kappa = (p_o - 0.5) / 0.5
                        task.iaa = round(max(-1.0, min(1.0, kappa)), 4)

            rewards = []
            for row in all_fb_rows:
                if row.scalar_reward is not None:
                    rewards.append(row.scalar_reward)
                elif row.binary_label is not None:
                    rewards.append(1.0 if row.binary_label else 0.0)
            if rewards:
                task.consensus_reward = round(sum(rewards) / len(rewards), 4)

            iaa_val = max(0.0, task.iaa) if task.iaa is not None else 0.0
            stdev = 0.0
            if len(rewards) >= 2:
                mean = sum(rewards) / len(rewards)
                variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
                stdev = math.sqrt(variance)
            non_flagged = [r for r in all_fb_rows if not r.flagged]
            coverage = min(1.0, len(non_flagged) / max(1, task.min_annotations))
            task.quality_score = round(0.4 * iaa_val + 0.4 * (1.0 - min(1.0, stdev)) + 0.2 * coverage, 4)

        await db.flush()
        print("Quality scores computed.")

        # ── Create sample datasets ────────────────────────────────────
        for name, fmt, filters in [
            ("DPO Training Set v1", "jsonl", {"min_quality": 0.5}),
            ("High Agreement Set", "parquet", {"min_quality": 0.6, "min_iaa": 0.3}),
            ("Full Export", "huggingface", None),
        ]:
            ds = Dataset(
                id=_uid(),
                name=name,
                filters=filters,
                export_format=fmt,
                task_count=0,
            )
            db.add(ds)
        await db.flush()
        print("Created 3 sample datasets.")

        # ── Create sample training run ────────────────────────────────
        datasets_result = await db.execute(Dataset.__table__.select().limit(1))
        first_ds = datasets_result.fetchone()
        if first_ds:
            tr = TrainingRun(
                id=_uid(),
                dataset_id=first_ds.id,
                algorithm="DPO",
                config={"learning_rate": 1e-5, "beta": 0.1, "epochs": 3},
                reward_history=[round(random.uniform(0.3, 0.8) + i * 0.02, 4) for i in range(20)],
                kl_history=[round(random.uniform(0.01, 0.05) + i * 0.001, 4) for i in range(20)],
                loss_history=[round(0.8 - i * 0.03 + random.uniform(-0.02, 0.02), 4) for i in range(20)],
                status=TrainingStatus.COMPLETED,
            )
            db.add(tr)
            await db.flush()
            print("Created 1 sample training run.")

        # ── Update annotator reliability scores ───────────────────────
        for annotator in annotators:
            fb_result = await db.execute(
                FeedbackItem.__table__.select().where(
                    FeedbackItem.annotator_id == annotator.id
                )
            )
            fb_rows = fb_result.fetchall()
            if fb_rows:
                non_flagged = [r for r in fb_rows if not r.flagged]
                annotator.reliability_score = round(len(non_flagged) / max(1, len(fb_rows)), 2)

        await db.commit()
        print("\nSeed complete!")
        print(f"  - {len(annotators)} annotators")
        print(f"  - {len(tasks)} tasks")
        print(f"  - {feedback_count} feedback items")
        print(f"  - 3 datasets")
        print(f"  - 1 training run")


if __name__ == "__main__":
    asyncio.run(seed())
```

> **What's happening here:**
>
> **`sys.path.insert(0, ...)`** — the seed script lives in `scripts/` but needs to import from `backend/`. Inserting the backend directory at the front of `sys.path` makes `from core.database import ...` work without installing the backend as a package.
>
> **Idempotency guard:** The first thing `seed()` does is count existing tasks. If any exist, it prints a message and returns. This means you can safely run the script multiple times — it only populates an empty database.
>
> **Status-aware feedback counts:** The number of feedback items per task is chosen to match the task's status. `PENDING` tasks get 0-1 items (not yet annotated), `IN_PROGRESS` tasks get fewer than `min_annotations` (partially annotated), and `COMPLETED`/`FLAGGED` tasks get `min_annotations` or more. This mirrors what the live system would produce.
>
> **5% flagged feedback:** `random.random() < 0.05` flags approximately 5% of feedback items as problematic. This gives the quality scoring logic a realistic signal — tasks with many flagged items will have lower coverage scores.
>
> **Inline quality score computation:** The seed script replicates the quality scoring formulas inline rather than calling the route or worker. This is intentional: the script bypasses the full application stack for speed and runs as a standalone command. The formulas are identical to the worker.
>
> **Synthetic reward curves:** `reward_history`, `kl_history`, and `loss_history` are generated with a simple linear trend plus noise. This gives the Training page in the dashboard a realistic-looking chart to display rather than an empty graph.

---

### Step 4.6: Update the Test Fixture to Mock Redis in the Feedback Route

**Why:** The feedback route now calls `get_redis()` to publish a Pub/Sub message. The test client uses SQLite and has no Redis connection. Without mocking this call, every feedback submission test will either fail or produce a connection error. We add `routes.feedback.get_redis` to the existing `patch` context in `conftest.py` so the publish call is intercepted by the same `mock_get_redis` fixture already used by the metrics route.

**Edit `backend/tests/conftest.py`** — add `routes.feedback.get_redis` to the patch context (the new line is marked):

```python
from __future__ import annotations

import asyncio
import sys
import os
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.database import Base, get_db

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with test_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


mock_redis_store: dict[str, str] = {}
mock_redis_queue: list[str] = []


async def mock_enqueue_task(task_id: str):
    mock_redis_queue.append(task_id)


async def mock_dequeue_task(timeout: int = 0):
    if mock_redis_queue:
        return mock_redis_queue.pop(0)
    return None


async def mock_cache_get(key: str):
    return mock_redis_store.get(key)


async def mock_cache_set(key: str, value: str, ttl: int = 60):
    mock_redis_store[key] = value


async def mock_get_redis():
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.llen = AsyncMock(return_value=len(mock_redis_queue))
    return mock


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Create tables before each test and drop after."""
    import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    mock_redis_store.clear()
    mock_redis_queue.clear()


@pytest_asyncio.fixture
async def client():
    async def mock_build_export(dataset_id: str):
        pass  # no-op in tests

    with patch("core.redis_client.enqueue_task", mock_enqueue_task), \
         patch("core.redis_client.dequeue_task", mock_dequeue_task), \
         patch("core.redis_client.cache_get", mock_cache_get), \
         patch("core.redis_client.cache_set", mock_cache_set), \
         patch("core.redis_client.get_redis", mock_get_redis), \
         patch("routes.tasks.enqueue_task", mock_enqueue_task), \
         patch("routes.annotators.dequeue_task", mock_dequeue_task), \
         patch("routes.metrics.cache_get", mock_cache_get), \
         patch("routes.metrics.cache_set", mock_cache_set), \
         patch("routes.metrics.get_redis", mock_get_redis), \
         patch("routes.feedback.get_redis", mock_get_redis), \   # ← Add this line
         patch("routes.exports._build_export", mock_build_export):
        from main import app
        app.dependency_overrides[get_db] = override_get_db
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac
        app.dependency_overrides.clear()
```

> **What's happening here:**
>
> **`patch("routes.feedback.get_redis", mock_get_redis)`** — Python's `unittest.mock.patch` replaces the name `get_redis` in the `routes.feedback` module's namespace for the duration of the `with` block. When `submit_feedback` calls `await get_redis()`, it receives the mock instead of the real Redis client. The mock's `publish` method is an `AsyncMock` that returns `None` without doing anything, which is exactly the behaviour we want in tests.
>
> **Why we patch the route module, not `core.redis_client`:** The route imports `get_redis` with `from core.redis_client import get_redis`. At import time, the name `get_redis` in `routes.feedback` is bound to the function object. Patching `core.redis_client.get_redis` would replace the original in the source module but would not affect the already-bound name in `routes.feedback`. We must patch the name where it is used.

---

### Verify Phase 4

**Run the test suite:**

```bash
cd backend && python -m pytest tests/ -v
```

All existing tests should still pass. If you see `ConnectionRefusedError` on a feedback test, the `routes.feedback.get_redis` patch is missing from `conftest.py`.

**Seed the database (Docker stack must be running):**

```bash
docker compose up -d
DATABASE_URL=postgresql+asyncpg://rl_user:rl_pass@localhost:5432/rl_platform \
  python scripts/seed.py
```

Expected output:

```
Created 8 annotators.
Created 50 tasks.
Created ~200 feedback items.
Quality scores computed.
Created 3 sample datasets.
Created 1 sample training run.

Seed complete!
  - 8 annotators
  - 50 tasks
  - ~200 feedback items
  - 3 datasets
  - 1 training run
```

**Test Parquet export via the API:**

```bash
# Create a parquet dataset
curl -s -X POST http://localhost:8000/api/exports/datasets \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Parquet", "export_format": "parquet", "filters": {"min_quality": 0.4}}' \
  | python -m json.tool

# Poll until exported_at is not null (allow a few seconds for the background task)
sleep 3
curl -s http://localhost:8000/api/exports/datasets | python -m json.tool
```

**Test HuggingFace export:**

```bash
curl -s -X POST http://localhost:8000/api/exports/datasets \
  -H "Content-Type: application/json" \
  -d '{"name": "HF Export", "export_format": "huggingface"}' \
  | python -m json.tool
```

**Acceptance criteria checklist:**

- [ ] `pytest` passes with zero failures
- [ ] Seed script populates 50 tasks, 8 annotators, quality scores computed
- [ ] `POST /api/exports/datasets` with `export_format: "parquet"` produces a `.parquet` file
- [ ] `POST /api/exports/datasets` with `export_format: "huggingface"` produces a directory with `train.jsonl` and `dataset_info.json`
- [ ] `reward_distribution` field on the dataset response is non-null after export
- [ ] Frontend Overview page shows non-zero KPI cards after seeding

---

## Phase 5: Observability — Structured Logging and Prometheus Metrics

### What We're Building

Phase 5 adds production observability to every layer of the stack. "Observability" means being able to answer "what is the system doing right now?" without attaching a debugger. We achieve this in three ways:

1. **Structured logging** with `structlog` — every log line is a machine-readable JSON object (in production) with a fixed set of keys, making log aggregation and filtering trivial in tools like Datadog, Loki, or CloudWatch.
2. **Prometheus metrics** — custom counters, gauges, and histograms that track domain-specific business events (feedback submitted, tasks completed, quality score distribution) alongside HTTP-level metrics (request rate, latency percentiles, error rate) provided automatically by `prometheus-fastapi-instrumentator`.
3. **Grafana dashboard** — a pre-built 8-panel dashboard provisioned automatically on first start, so there is no manual click-through required to see the platform's behaviour.

### Prerequisites

Phase 4 complete. Docker Compose stack running.

---

### Step 5.1: Add Observability Dependencies

**Why:** Two new libraries are required. `structlog` provides structured logging with a processor pipeline; it integrates with Python's standard `logging` module so existing `logging.getLogger()` calls in third-party libraries also go through the structlog formatter. `prometheus-fastapi-instrumentator` auto-instruments every FastAPI route with request count, latency, and in-flight request metrics without requiring per-route boilerplate.

**Edit `backend/requirements.txt`** — add the two new lines:

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
sqlalchemy[asyncio]>=2.0.25
asyncpg>=0.29.0
redis>=5.0.0
pydantic>=2.5.0
python-dotenv>=1.0.0
httpx>=0.27.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
aiosqlite>=0.20.0
pyarrow>=15.0.0
prometheus-fastapi-instrumentator>=7.0.0
structlog>=24.1.0
```

---

### Step 5.2: Structured Logging Module

**Why:** Python's built-in `logging` module formats messages as flat strings. When you grep logs for a task ID, you are pattern-matching inside unstructured text. structlog adds a processor pipeline that attaches a dictionary of key-value pairs to every log call, then renders the final output either as JSON (production) or a human-friendly coloured console format (development). The format is controlled by the `LOG_FORMAT` environment variable so the same code works in both contexts.

**Create `backend/core/logging.py`:**

```python
from __future__ import annotations

import logging
import os
import sys

import structlog


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Processors applied to every log call before rendering
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # In production (LOG_FORMAT=json), output JSON; otherwise dev console
    log_format = os.getenv("LOG_FORMAT", "console")

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
```

> **What's happening here:**
>
> **`shared_processors` pipeline:** Each processor in the list is a callable that receives the logger, method name, and event dictionary, and returns a (possibly modified) event dictionary. The processors run in order:
> - `merge_contextvars` — merges any context variables bound with `structlog.contextvars.bind_contextvars()`. This is how you attach a request ID to every log line within a request without passing it as a parameter.
> - `add_logger_name` — adds `"logger": "module.name"` to the event.
> - `add_log_level` — adds `"level": "info"` to the event.
> - `TimeStamper(fmt="iso")` — adds `"timestamp": "2024-01-15T10:23:45.123Z"`.
> - `StackInfoRenderer` — renders stack traces if `stack_info=True` is passed.
> - `UnicodeDecoder` — ensures byte strings in event values are decoded to unicode.
>
> **`ProcessorFormatter` bridge:** structlog integrates with Python's standard logging by wrapping log records through a `ProcessorFormatter`. This means `logging.getLogger("sqlalchemy").info("...")` also goes through the structlog pipeline and produces consistent output. Without this bridge, structlog and stdlib logging would produce two different formats.
>
> **`JSONRenderer` vs `ConsoleRenderer`:** In Docker (`LOG_FORMAT=json`), every log line is a single JSON object. This is what log aggregation tools expect. In local development (`LOG_FORMAT=console`), structlog uses a coloured, human-readable format with aligned columns. Setting `LOG_FORMAT=json` in `docker-compose.yml` (which we do in Step 5.9) ensures production-format logs inside the container.
>
> **`cache_logger_on_first_use=True`:** After the first call to `structlog.get_logger(name)`, the logger is cached. This avoids re-running the processor binding logic on every log call.
>
> **Silencing uvicorn.access and sqlalchemy.engine:** These two loggers are extremely verbose at `INFO` level. Setting them to `WARNING` suppresses the per-request access log and every SQL statement without hiding error-level messages.

---

### Step 5.3: Prometheus Metrics Module

**Why:** We want to track six domain-specific metrics beyond the HTTP-level metrics that `prometheus-fastapi-instrumentator` provides automatically. Defining all metrics in one module prevents name collisions (Prometheus panics if two `Counter` objects have the same metric name) and gives routes a single import to add instrumentation.

**Create `backend/core/metrics.py`:**

```python
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

def setup_instrumentator() -> Instrumentator:
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/health", "/metrics"],
        env_var_name="ENABLE_METRICS",
    )
    return instrumentator
```

> **What's happening here:**
>
> **`Counter` vs `Gauge` vs `Histogram`:**
> - `Counter` — only goes up. Use it for events: feedback submissions, tasks created, exports. The Prometheus `rate()` function derives events-per-second from counters.
> - `Gauge` — can go up and down. Use it for current state: queue depth, number of active annotators.
> - `Histogram` — tracks a distribution. Each observation falls into a bucket based on its value. Prometheus stores the counts per bucket, which lets you query percentiles with `histogram_quantile(0.95, ...)`.
>
> **Labelled counters (`["annotation_type"]`, `["export_format"]`):** Labels create a separate time series per unique label value. `feedback_submitted_total{annotation_type="ranking"}` and `feedback_submitted_total{annotation_type="scalar"}` are distinct series. The Grafana dashboard uses `rate(rl_feedback_submitted_total[5m])` with `legendFormat: "{{annotation_type}}"` to plot each type as a separate line.
>
> **`quality_score_histogram` buckets:** We explicitly define buckets at 0.1 intervals across the full 0.0-1.0 range, because the default Prometheus buckets (designed for latency in seconds) would be meaningless for a 0-1 quality score. This enables the "Quality Score Distribution" histogram panel in Grafana.
>
> **`setup_instrumentator` — excluded handlers:** `/health` and `/metrics` are excluded from HTTP instrumentation. Including them would pollute latency percentiles (health checks are near-zero latency) and create infinite recursion (Prometheus scraping `/metrics` would record a metric for that scrape).
>
> **`should_group_status_codes=True`:** Groups 200, 201, 204 into `2xx`, 400, 404, 422 into `4xx`, etc. This keeps the number of time series manageable when there are many distinct status codes in use.

---

### Step 5.4: Wire Logging and Metrics into main.py

**Why:** Logging must be configured before the application creates any loggers. We call `setup_logging()` at module level (before the `app` object is created) so that even import-time log messages from FastAPI and SQLAlchemy go through the structlog formatter. The Prometheus instrumentator must be attached to the `app` object and its `/metrics` endpoint exposed before any requests arrive.

**Replace `backend/main.py`** with the full updated version:

```python
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from core.database import async_session, init_db
from core.logging import setup_logging, get_logger
from core.metrics import setup_instrumentator
from core.redis_client import close_redis, get_redis

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting_up", action="initializing database tables")
    await init_db()
    logger.info("database_ready")

    r = await get_redis()
    pong = await r.ping()
    logger.info("redis_ready", ping=pong)

    yield

    await close_redis()
    logger.info("shutdown_complete")


app = FastAPI(
    title="RL Training Data Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics — instrument all routes and expose /metrics
instrumentator = setup_instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")

from routes.tasks import router as tasks_router
from routes.feedback import router as feedback_router
from routes.annotators import router as annotators_router
from routes.metrics import router as metrics_router
from routes.exports import router as exports_router

app.include_router(tasks_router)
app.include_router(feedback_router)
app.include_router(annotators_router)
app.include_router(metrics_router)
app.include_router(exports_router)


@app.get("/health")
async def health():
    db_status = "ok"
    redis_status = "ok"

    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    try:
        r = await get_redis()
        await r.ping()
    except Exception:
        redis_status = "error"

    status = "ok" if db_status == "ok" and redis_status == "ok" else "error"
    return {"status": status, "db": db_status, "redis": redis_status}
```

> **What's happening here:**
>
> **`setup_logging()` at module level:** Called before anything else in the file. Python evaluates the module top-to-bottom at import time. By placing `setup_logging()` before `app = FastAPI(...)`, we guarantee the structlog handler is installed before FastAPI registers its own internal loggers.
>
> **`get_logger(__name__)` replacing `logging.getLogger(__name__)`:** This returns a structlog `BoundLogger` instead of a stdlib `Logger`. The key difference is that structlog loggers accept keyword arguments: `logger.info("redis_ready", ping=pong)` produces `{"event": "redis_ready", "ping": "True", "level": "info", ...}` in JSON format. Compare this to `logger.info(f"Redis ping: {pong}")` which embeds the value in an unstructured string.
>
> **`instrumentator.instrument(app).expose(app, endpoint="/metrics")`:** This one line does two things. `.instrument(app)` registers Starlette middleware that intercepts every request and records its duration and status into the Prometheus histograms/counters. `.expose(app, endpoint="/metrics")` adds a GET `/metrics` route that responds with the Prometheus text format. Prometheus scrapes this endpoint every 15 seconds.
>
> **`import logging` removed:** The stdlib `import logging` and `logging.basicConfig()` from Phase 3 are removed. structlog's `setup_logging()` configures the stdlib root logger directly, making `logging.basicConfig()` redundant and potentially conflicting.

---

### Step 5.5: Add Prometheus Counters to Route Modules

**Why:** The `tasks_created_total` counter needs to increment every time a task is created. The `feedback_submitted_total` counter needs the annotation type label. The `exports_created_total` counter needs the export format label. These counters were already imported into the routes in Phase 4 — here we confirm their placement and explain the pattern.

The three route files already contain the counter imports and `.inc()` calls from the Phase 4 work above. For completeness, here is where each counter fires:

**In `backend/routes/tasks.py`** — after flushing the new task to the database:

```python
from core.metrics import tasks_created_total

# Inside create_task():
await db.flush()
await db.refresh(task)
tasks_created_total.inc()       # ← fires here
await enqueue_task(task.id)
return task
```

**In `backend/routes/feedback.py`** — after flushing the new feedback item:

```python
from core.metrics import feedback_submitted_total

# Inside submit_feedback():
await db.flush()
feedback_submitted_total.labels(annotation_type=task.annotation_type).inc()   # ← fires here
```

**In `backend/routes/exports.py`** — inside `create_dataset()`, after flushing the new dataset:

```python
from core.metrics import exports_created_total

# Inside create_dataset():
await db.refresh(dataset)
exports_created_total.labels(export_format=payload.export_format).inc()   # ← fires here
background_tasks.add_task(_build_export, dataset.id)
```

> **What's happening here:**
>
> **Why counters increment after the flush, not after commit:** SQLAlchemy's `flush()` writes the INSERT to the database transaction but does not commit. If the transaction later rolls back, the counter will still have been incremented — this is an accepted trade-off. Incrementing after commit would require an additional await and complicates error handling. For counters tracking business events, slight over-counting is preferable to under-counting.
>
> **`.labels(annotation_type=task.annotation_type)`:** This call looks up (or creates) the child counter for this label combination. The first time a new label value is seen, Prometheus registers a new time series. Subsequent calls return the cached child counter.

---

### Step 5.6: Add Structured Logging and Metrics to the Quality Worker

**Why:** The worker is a long-running process in its own container. Without structured logging, diagnosing issues requires reading unformatted print statements. With structlog, every event (worker start, pubsub subscribe, task processed, task flagged) produces a JSON log line with a consistent schema. We also call `tasks_completed_total.inc()` and `quality_score_histogram.observe()` from the worker so the Prometheus counters reflect background processing, not just HTTP-triggered events.

The `backend/workers/quality_worker.py` file already contains the full implementation with logging and metrics calls as written in Step 4.2. The key calls are:

```python
from core.logging import setup_logging, get_logger
from core.metrics import quality_score_histogram, tasks_completed_total

setup_logging()
logger = get_logger(__name__)

# In process_task():
logger.info("task_processed",
    task_id=task_id,
    iaa=task.iaa,
    quality_score=task.quality_score,
    consensus_reward=task.consensus_reward,
    status=task.status,
)

# When a task auto-completes:
tasks_completed_total.inc()
logger.info("task_completed", task_id=task_id, quality_score=task.quality_score)

# After computing quality score:
if task.quality_score is not None:
    quality_score_histogram.observe(task.quality_score)
```

> **What's happening here:**
>
> **`setup_logging()` called at module level in the worker:** The worker runs as a completely separate Python process (`python -m workers.quality_worker`). It does not share memory with the API process. We must call `setup_logging()` independently in the worker so its logs also go through the structlog formatter.
>
> **Structured event names as strings:** Event names like `"task_processed"`, `"task_completed"`, `"task_flagged"` are snake_case identifiers rather than human sentences. This convention allows tools like Datadog or Loki to `filter event="task_flagged"` without regex parsing. The additional key-value pairs provide the context needed for investigation.
>
> **`quality_score_histogram.observe(task.quality_score)`** — the worker calls this every time it recomputes a quality score. Over time, this histogram accumulates thousands of observations and the Grafana panel can show the full distribution of quality scores across the platform.

---

### Step 5.7: Prometheus Configuration

**Why:** Prometheus needs to know which HTTP endpoint to scrape and how often. We provide this via a YAML config file. The `api` service is identified by its Docker Compose service name — within the Docker network, `api:8000` resolves to the API container's internal IP without any extra configuration.

First, create the directory structure:

```bash
mkdir -p monitoring/prometheus
```

**Create `monitoring/prometheus/prometheus.yml`:**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "rl-api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics
```

> **What's happening here:**
>
> **`scrape_interval: 15s`:** Prometheus polls the `/metrics` endpoint every 15 seconds. This is the standard default. Decreasing it to 5s gives higher resolution but increases storage and network usage. 15s is fine for a platform that processes human feedback, where sub-second precision is not needed.
>
> **`targets: ["api:8000"]`:** Within Docker Compose's default bridge network, each service is reachable by its service name as a hostname. The Prometheus container and the API container share this network, so `api:8000` resolves correctly. No ports need to be explicitly linked between the containers.
>
> **`metrics_path: /metrics`:** This must match the `endpoint` parameter passed to `.expose()` in `main.py`. The default Prometheus path is also `/metrics`, so this line is technically redundant but makes the intent explicit.

---

### Step 5.8: Grafana Provisioning and Dashboard

**Why:** Grafana supports "provisioning" — a mechanism to load data sources, dashboards, and alert rules from YAML and JSON files at startup. Without provisioning, every new deployment requires a human to click through the Grafana UI to add the Prometheus data source and import the dashboard. With provisioning, the full setup is declarative and version-controlled.

Create the directory structure:

```bash
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards
```

**Create `monitoring/grafana/provisioning/datasources/prometheus.yml`:**

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

> **What's happening here:**
> - `access: proxy` — Grafana fetches data from Prometheus server-side. The browser never directly contacts Prometheus. This is required when Prometheus is on an internal Docker network and not exposed to the host.
> - `url: http://prometheus:9090` — Grafana resolves the Prometheus container by its Docker Compose service name.
> - `isDefault: true` — dashboards that do not specify a data source use this one by default.
> - `editable: false` — provisioned data sources cannot be modified through the UI, preventing accidental misconfiguration.

**Create `monitoring/grafana/provisioning/dashboards/dashboards.yml`:**

```yaml
apiVersion: 1

providers:
  - name: "RL Platform"
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: false
```

> **What's happening here:**
> - `type: file` — tells Grafana to load dashboards from JSON files on disk rather than from the database.
> - `path: /var/lib/grafana/dashboards` — the directory inside the Grafana container where dashboard JSON files are stored. This maps to `./monitoring/grafana/dashboards` on the host via a Docker volume mount (configured in Step 5.9).
> - `disableDeletion: false` — if a JSON file is removed from disk, Grafana removes the corresponding dashboard from the UI on next restart.

**Create `monitoring/grafana/dashboards/rl-platform.json`:**

This is the complete 8-panel dashboard JSON. Copy the full content below exactly:

```json
{
  "annotations": { "list": [] },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 1,
  "id": null,
  "links": [],
  "panels": [
    {
      "title": "Request Rate (req/s)",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
      "targets": [
        {
          "expr": "rate(http_requests_total[5m])",
          "legendFormat": "{{method}} {{handler}} {{status}}",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "reqps",
          "custom": { "lineWidth": 2, "fillOpacity": 10 }
        }
      }
    },
    {
      "title": "Request Latency (p50 / p95 / p99)",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
      "targets": [
        {
          "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p50",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p95",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p99",
          "refId": "C"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "s",
          "custom": { "lineWidth": 2, "fillOpacity": 10 }
        }
      }
    },
    {
      "title": "Feedback Submissions (by type)",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
      "targets": [
        {
          "expr": "rate(rl_feedback_submitted_total[5m])",
          "legendFormat": "{{annotation_type}}",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "ops",
          "custom": { "lineWidth": 2, "fillOpacity": 15, "drawStyle": "bars" }
        }
      }
    },
    {
      "title": "Annotator Throughput",
      "type": "stat",
      "gridPos": { "h": 8, "w": 6, "x": 12, "y": 8 },
      "targets": [
        {
          "expr": "sum(increase(rl_feedback_submitted_total[1h]))",
          "legendFormat": "Submissions / hr",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "thresholds": {
            "steps": [
              { "color": "red", "value": null },
              { "color": "yellow", "value": 10 },
              { "color": "green", "value": 50 }
            ]
          }
        }
      }
    },
    {
      "title": "Tasks Created / Completed",
      "type": "stat",
      "gridPos": { "h": 8, "w": 6, "x": 18, "y": 8 },
      "targets": [
        {
          "expr": "sum(increase(rl_tasks_created_total[1h]))",
          "legendFormat": "Created / hr",
          "refId": "A"
        },
        {
          "expr": "sum(increase(rl_tasks_completed_total[1h]))",
          "legendFormat": "Completed / hr",
          "refId": "B"
        }
      ]
    },
    {
      "title": "Annotation Queue Depth",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 16 },
      "targets": [
        {
          "expr": "rl_annotation_queue_depth",
          "legendFormat": "Queue depth",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "custom": { "lineWidth": 2, "fillOpacity": 20 },
          "thresholds": {
            "steps": [
              { "color": "green", "value": null },
              { "color": "yellow", "value": 50 },
              { "color": "red", "value": 200 }
            ]
          }
        }
      }
    },
    {
      "title": "Quality Score Distribution",
      "type": "histogram",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 16 },
      "targets": [
        {
          "expr": "rl_task_quality_score_bucket",
          "legendFormat": "{{le}}",
          "refId": "A",
          "format": "heatmap"
        }
      ]
    },
    {
      "title": "Exports Created (by format)",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 24 },
      "targets": [
        {
          "expr": "increase(rl_exports_created_total[1h])",
          "legendFormat": "{{export_format}}",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "custom": { "drawStyle": "bars", "fillOpacity": 50 }
        }
      }
    },
    {
      "title": "HTTP Error Rate",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 24 },
      "targets": [
        {
          "expr": "sum(rate(http_requests_total{status=~\"4..|5..\"}[5m])) by (status)",
          "legendFormat": "{{status}}",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "reqps",
          "custom": { "lineWidth": 2, "fillOpacity": 10 }
        }
      }
    }
  ],
  "schemaVersion": 39,
  "tags": ["rl-platform"],
  "templating": { "list": [] },
  "time": { "from": "now-1h", "to": "now" },
  "title": "RL Training Data Platform",
  "uid": "rl-platform-overview",
  "version": 1
}
```

> **What's happening here — the 8 panels:**
>
> 1. **Request Rate** — `rate(http_requests_total[5m])`. The `http_requests_total` counter is created by `prometheus-fastapi-instrumentator`. `rate()` computes the per-second increase over a 5-minute window, which smooths out spikes. Broken down by method, handler, and status.
>
> 2. **Request Latency p50/p95/p99** — `histogram_quantile(Q, rate(http_request_duration_seconds_bucket[5m]))`. This is the standard Prometheus pattern for latency percentiles from a histogram. p99 > 500ms is a common alert threshold.
>
> 3. **Feedback Submissions by type** — `rate(rl_feedback_submitted_total[5m])` broken down by `annotation_type` label. The bar chart style shows spikes clearly.
>
> 4. **Annotator Throughput** — `sum(increase(rl_feedback_submitted_total[1h]))` gives total submissions in the last hour. The threshold steps colour the stat green when throughput is healthy (>50/hr), yellow when low (10-50), and red when very low (<10).
>
> 5. **Tasks Created/Completed** — two `increase()` series side-by-side. If created consistently exceeds completed, the queue is growing.
>
> 6. **Annotation Queue Depth** — `rl_annotation_queue_depth` gauge. Thresholds turn the fill yellow at 50 and red at 200 items, signalling that the worker is falling behind.
>
> 7. **Quality Score Distribution** — the histogram bucket metric `rl_task_quality_score_bucket` rendered as a heatmap. This shows whether quality is concentrated at high values (good) or spreading toward low values (signal to review task design).
>
> 8. **HTTP Error Rate** — `rate(http_requests_total{status=~"4..|5.."}[5m])`. The regex `4..|5..` matches any 4xx or 5xx status code. A sudden spike here means a code change broke request handling.

---

### Step 5.9: Add Prometheus and Grafana to Docker Compose

**Why:** The monitoring stack must run inside Docker Compose alongside the application so Prometheus can reach the API on the internal Docker network (`api:8000`). Running Prometheus on the host would require exposing the API port and managing host networking, which is fragile and inconsistent across platforms.

**Replace `docker-compose.yml`** with the full updated version:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: rl-postgres
    environment:
      POSTGRES_USER: rl_user
      POSTGRES_PASSWORD: rl_pass
      POSTGRES_DB: rl_platform
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rl_user -d rl_platform"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: rl-redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build:
      context: ./backend
    container_name: rl-api
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://rl_user:rl_pass@postgres:5432/rl_platform
      REDIS_URL: redis://redis:6379/0
      EXPORT_DIR: /exports
      LOG_FORMAT: json
    volumes:
      - exports:/exports
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 5s
      retries: 5

  frontend:
    build:
      context: ./frontend
    container_name: rl-frontend
    ports:
      - "3000:80"
    environment:
      VITE_API_URL: http://localhost:8000
    depends_on:
      api:
        condition: service_healthy

  worker:
    build:
      context: ./backend
    container_name: rl-worker
    command: python -m workers.quality_worker
    environment:
      DATABASE_URL: postgresql+asyncpg://rl_user:rl_pass@postgres:5432/rl_platform
      REDIS_URL: redis://redis:6379/0
      EXPORT_DIR: /exports
      LOG_FORMAT: json
    volumes:
      - exports:/exports
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: rl-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=7d"
    depends_on:
      api:
        condition: service_healthy

  grafana:
    image: grafana/grafana:10.4.0
    container_name: rl-grafana
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  pgdata:
  exports:
  prometheus_data:
  grafana_data:
```

> **What's happening here:**
>
> **Prometheus service:**
> - `prom/prometheus:v2.51.0` — pinning to a specific patch version prevents surprise breaking changes on rebuild.
> - `./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml` — the config file is bind-mounted. Editing the YAML file on the host takes effect after `docker compose restart prometheus`.
> - `--storage.tsdb.retention.time=7d` — Prometheus keeps 7 days of metrics. Increase this for longer retention, but note that each additional day adds roughly 500MB to 2GB of storage depending on the number of time series.
> - `prometheus_data:/prometheus` — persists the time-series database across container restarts.
> - `depends_on: api: condition: service_healthy` — Prometheus only starts after the API health check passes. If Prometheus starts before the API, the first scrape would fail and produce a gap in the metrics timeline.
>
> **Grafana service:**
> - `prom/grafana:10.4.0` — Grafana 10 introduced breaking changes in panel configuration. We pin to 10.4 to ensure the dashboard JSON remains valid.
> - `./monitoring/grafana/provisioning:/etc/grafana/provisioning` — the provisioning directory is bind-mounted. Grafana reads all YAML files under `provisioning/datasources/` and `provisioning/dashboards/` at startup.
> - `./monitoring/grafana/dashboards:/var/lib/grafana/dashboards` — the dashboard JSON directory is bind-mounted at the path specified in `dashboards.yml`. Grafana watches this directory and reloads changed JSON files.
> - Port `3001:3000` — Grafana's internal port is 3000, which conflicts with the frontend's host port. We remap it to 3001 on the host.
> - `GF_SECURITY_ADMIN_PASSWORD: admin` — the default admin credentials. Change this in production.
>
> **`LOG_FORMAT: json` in api and worker:** Both the API and worker receive this environment variable. `setup_logging()` reads `LOG_FORMAT` and selects `JSONRenderer` when the value is `"json"`. This produces machine-readable log lines suitable for log aggregation.

---

### Verify Phase 5

**Rebuild and start the full stack:**

```bash
docker compose up --build
```

Wait for all health checks to pass (approximately 30-60 seconds on first build).

**Verify the metrics endpoint:**

```bash
curl -s http://localhost:8000/metrics | grep rl_
```

Expected output includes lines like:

```
# HELP rl_feedback_submitted_total Total number of feedback submissions
# TYPE rl_feedback_submitted_total counter
# HELP rl_tasks_created_total Total number of tasks created
# TYPE rl_tasks_created_total counter
# HELP rl_task_quality_score Distribution of task quality scores
# TYPE rl_task_quality_score histogram
```

**Verify Prometheus is scraping:**

1. Open `http://localhost:9090` in your browser.
2. Click "Status" → "Targets".
3. The `rl-api` target should show state `UP` with a recent "Last Scrape" timestamp.

**Generate some metrics by submitting feedback:**

```bash
# Create a task
TASK_ID=$(curl -s -X POST http://localhost:8000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Test prompt","responses":[{"model_id":"a","text":"Response A"},{"model_id":"b","text":"Response B"}],"annotation_type":"ranking","min_annotations":2}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['id'])")

# Submit feedback twice
curl -s -X POST http://localhost:8000/api/feedback/ \
  -H "Content-Type: application/json" \
  -d "{\"task_id\":\"$TASK_ID\",\"annotator_id\":\"test-1\",\"ranking\":[1,2]}" > /dev/null

curl -s -X POST http://localhost:8000/api/feedback/ \
  -H "Content-Type: application/json" \
  -d "{\"task_id\":\"$TASK_ID\",\"annotator_id\":\"test-2\",\"ranking\":[1,2]}" > /dev/null

# Check metrics
curl -s http://localhost:8000/metrics | grep rl_feedback_submitted_total
```

Expected:

```
rl_feedback_submitted_total{annotation_type="ranking"} 2.0
```

**Verify Grafana dashboard:**

1. Open `http://localhost:3001` in your browser.
2. Log in with `admin` / `admin`.
3. Click the hamburger menu → "Dashboards".
4. You should see "RL Training Data Platform" in the list.
5. Click it — all 8 panels should load with data.

**Verify structured JSON logging:**

```bash
docker compose logs api --tail=20
```

In production mode (`LOG_FORMAT=json`), each log line should be valid JSON:

```json
{"event": "database_ready", "level": "info", "logger": "main", "timestamp": "2024-01-15T10:23:45.123Z"}
{"event": "redis_ready", "ping": "True", "level": "info", "logger": "main", "timestamp": "2024-01-15T10:23:45.456Z"}
```

**Acceptance criteria checklist:**

- [ ] `curl http://localhost:8000/metrics` returns Prometheus text with `rl_` prefixed custom metrics
- [ ] Prometheus UI at `http://localhost:9090/targets` shows `rl-api` target as `UP`
- [ ] Grafana UI at `http://localhost:3001` shows the "RL Training Data Platform" dashboard with 8 panels
- [ ] `docker compose logs api` produces JSON-formatted log lines
- [ ] `docker compose logs worker` produces JSON-formatted log lines
- [ ] After submitting feedback, `rl_feedback_submitted_total` counter increments in `/metrics`
- [ ] `pytest` still passes with zero failures

---

## Updated Project Structure

After completing Phases 4 and 5, the project directory looks like this:

```
agent-rl-training-data-platform/
├── docker-compose.yml                        # Updated: prometheus and grafana services added
│
├── backend/
│   ├── main.py                               # Updated: structlog + Prometheus instrumentator
│   ├── models.py
│   ├── schemas.py
│   ├── requirements.txt                      # Updated: pyarrow, prometheus-fastapi-instrumentator, structlog
│   ├── Dockerfile
│   ├── pytest.ini
│   ├── core/
│   │   ├── database.py
│   │   ├── redis_client.py
│   │   ├── logging.py                        # NEW: structlog setup
│   │   └── metrics.py                        # NEW: Prometheus custom metrics
│   ├── routes/
│   │   ├── tasks.py                          # Updated: tasks_created_total counter
│   │   ├── feedback.py                       # Updated: Redis publish + feedback_submitted_total
│   │   ├── annotators.py
│   │   ├── metrics.py
│   │   └── exports.py                        # Updated: Parquet + HuggingFace formats + reward_distribution
│   ├── workers/
│   │   └── quality_worker.py                 # Replaced: full Redis consumer with structlog + metrics
│   └── tests/
│       ├── conftest.py                       # Updated: routes.feedback.get_redis patched
│       ├── test_tasks.py
│       ├── test_feedback.py
│       ├── test_annotators.py
│       ├── test_metrics.py
│       └── test_exports.py
│
├── scripts/
│   ├── init.sql
│   └── seed.py                               # NEW: 50 tasks, 8 annotators, quality scores
│
├── frontend/
│   └── ...                                   # Unchanged from Phase 3
│
└── monitoring/                               # NEW: entire directory
    ├── prometheus/
    │   └── prometheus.yml
    └── grafana/
        ├── provisioning/
        │   ├── datasources/
        │   │   └── prometheus.yml
        │   └── dashboards/
        │       └── dashboards.yml
        └── dashboards/
            └── rl-platform.json
```

---

## Environment Variables Reference

| Variable | Service | Description | Example |
|----------|---------|-------------|---------|
| `DATABASE_URL` | api, worker | Async PostgreSQL connection string | `postgresql+asyncpg://rl_user:rl_pass@postgres:5432/rl_platform` |
| `REDIS_URL` | api, worker | Redis connection URL | `redis://redis:6379/0` |
| `EXPORT_DIR` | api, worker | Directory where export files are written | `/exports` |
| `LOG_FORMAT` | api, worker | `json` for machine-readable output, `console` for human-readable | `json` |
| `LOG_LEVEL` | api, worker | Python logging level | `INFO` |
| `GF_SECURITY_ADMIN_USER` | grafana | Grafana admin username | `admin` |
| `GF_SECURITY_ADMIN_PASSWORD` | grafana | Grafana admin password | `admin` |
| `GF_USERS_ALLOW_SIGN_UP` | grafana | Disable public sign-up | `"false"` |

---

## Common Issues and Troubleshooting

**`ModuleNotFoundError: No module named 'pyarrow'` when creating a Parquet export**

The `pyarrow` package was added to `requirements.txt` but the Docker image was not rebuilt. Run `docker compose up --build` to trigger a fresh pip install.

**`prometheus_client.registry.CollectorRegistry: Duplicated timeseries` error on startup**

This happens when the Python process is reloaded (e.g., by a code watcher) without a full restart. Prometheus metric objects are singletons registered in a global registry. On reload, the module re-executes and tries to register the same metric name twice. The fix is to always use `docker compose restart api` rather than hot-reloading inside a running container during development.

**Grafana dashboard shows "No data" for all panels**

Either Prometheus is not yet scraping, or the API has not received any requests. Check `http://localhost:9090/targets` first. If the target is `DOWN`, check that the API container is healthy (`docker compose ps`). If the target is `UP` but panels are empty, generate traffic with `curl http://localhost:8000/api/tasks` and wait 15-30 seconds for Prometheus to scrape and Grafana to render.

**`routes.feedback.get_redis` patch causes `AttributeError` in tests**

This error means the `feedback` module was imported before the patch context was entered. Ensure all `from main import app` statements are inside the `with patch(...)` block in `conftest.py`.

**Seed script fails with `sqlalchemy.exc.OperationalError`**

The seed script connects directly to PostgreSQL using `DATABASE_URL`. If you run it on the host with `localhost:5432`, ensure port 5432 is published in `docker-compose.yml` (it is by default). If you run it inside the Docker network, use `postgres:5432`. The most common cause is running the script before `docker compose up` has started.

**Worker container exits immediately with `KeyError: 'data'` from pubsub**

This is a Redis Pub/Sub protocol detail: the first message from `pubsub.listen()` is a subscription confirmation with `type="subscribe"`, not `type="message"`. The worker correctly filters these with `if message["type"] != "message": continue`. If you see this error, check that you are running the current version of `quality_worker.py` from Step 4.2 and not an older version.

---

## Next Steps

With Phases 4 and 5 complete, the platform collects human feedback, computes quality scores in real time, exports training data in three formats, and exposes full observability through Prometheus and Grafana. Possible extensions from here:

- **Alerting rules in Prometheus** — add a `rules/` directory with alerting rules (e.g., alert when `rl_annotation_queue_depth > 200` for more than 5 minutes) and configure Alertmanager to send notifications to Slack or PagerDuty.
- **Grafana alert contacts** — wire Grafana alert channels to the same notification endpoints, and add threshold-based alerts directly on the dashboard panels.
- **Distributed tracing with OpenTelemetry** — add `opentelemetry-instrumentation-fastapi` and export traces to Jaeger or Tempo. Traces complement metrics by showing exactly which function call inside a request is slow.
- **Worker horizontal scaling** — the quality worker uses `brpop` on a single queue, which means running multiple worker instances will split the load automatically. Add a `replicas: 2` field under the `worker` service in Docker Compose and verify that tasks are processed correctly.
- **Annotator reliability scoring** — the seed script sets static reliability scores. A production implementation would recompute these from actual flagging history: `reliability = non_flagged_submissions / total_submissions`. The `Annotator.reliability_score` field already exists; add a periodic background task to update it.
- **HuggingFace Hub push** — extend the export route with an optional `push_to_hub` flag that uses the `huggingface_hub` library to upload the dataset to a private Hub repository after local export.
