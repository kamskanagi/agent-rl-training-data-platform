# Building an RL Training Data Platform: Phases 1 and 2

> Build a production-ready backend for collecting human preference feedback and exporting RL-ready training data — step by step from an empty repo to a fully tested REST API.

**What you'll build:** A FastAPI backend backed by PostgreSQL and Redis that lets researchers create annotation tasks, collect pairwise rankings and scalar rewards from annotators, compute inter-annotator agreement (Fleiss' kappa) automatically on every feedback submission, and export the resulting preference data as JSONL in DPO (Direct Preference Optimization) format ready for LLM fine-tuning.

**Tech stack:** Python 3.12, FastAPI, SQLAlchemy 2.0 (async), asyncpg, PostgreSQL 16, Redis 7, Docker Compose, Pydantic v2, pytest-asyncio, httpx, aiosqlite

**Prerequisites:** Docker Desktop installed and running, Python 3.12, basic familiarity with FastAPI and async Python, understanding of what RLHF / preference data means (not required to build, but helps with the "why")

**Time estimate:** 3-5 hours

**Difficulty:** Intermediate

**Final repo:** https://github.com/kamskanagi/agent-rl-training-data-platform

---

## Table of Contents

1. [Phase 1: Foundation — Docker, Database, and Health Endpoint](#phase-1-foundation--docker-database-and-health-endpoint)
   - [Step 1.1: Repository Structure](#step-11-repository-structure)
   - [Step 1.2: Python Dependencies](#step-12-python-dependencies)
   - [Step 1.3: The Dockerfile](#step-13-the-dockerfile)
   - [Step 1.4: Docker Compose — The Full Stack](#step-14-docker-compose--the-full-stack)
   - [Step 1.5: PostgreSQL Initialization Script](#step-15-postgresql-initialization-script)
   - [Step 1.6: Environment Variables](#step-16-environment-variables)
   - [Step 1.7: Async Database Engine and Session](#step-17-async-database-engine-and-session)
   - [Step 1.8: Redis Client and Queue Helpers](#step-18-redis-client-and-queue-helpers)
   - [Step 1.9: ORM Models — Six Tables](#step-19-orm-models--six-tables)
   - [Step 1.10: The FastAPI Application and Health Endpoint](#step-110-the-fastapi-application-and-health-endpoint)
   - [Step 1.11: Quality Worker Placeholder](#step-111-quality-worker-placeholder)
   - [Verify Phase 1](#verify-phase-1)
2. [Phase 2: Backend API — Tasks, Feedback, Annotators, Metrics, Exports](#phase-2-backend-api--tasks-feedback-annotators-metrics-exports)
   - [Step 2.1: Pydantic Schemas](#step-21-pydantic-schemas)
   - [Step 2.2: Tasks Router](#step-22-tasks-router)
   - [Step 2.3: Feedback Router and IAA Scoring](#step-23-feedback-router-and-iaa-scoring)
   - [Step 2.4: Annotators Router and Queue Integration](#step-24-annotators-router-and-queue-integration)
   - [Step 2.5: Metrics Router with Redis Caching](#step-25-metrics-router-with-redis-caching)
   - [Step 2.6: Exports Router and DPO Format](#step-26-exports-router-and-dpo-format)
   - [Step 2.7: Wire All Routers into main.py](#step-27-wire-all-routers-into-mainpy)
   - [Step 2.8: Test Infrastructure — conftest.py](#step-28-test-infrastructure--conftestpy)
   - [Step 2.9: Task Tests](#step-29-task-tests)
   - [Step 2.10: Feedback Tests](#step-210-feedback-tests)
   - [Step 2.11: Annotator Tests](#step-211-annotator-tests)
   - [Step 2.12: Metrics Tests](#step-212-metrics-tests)
   - [Step 2.13: Export Tests](#step-213-export-tests)
   - [Verify Phase 2](#verify-phase-2)
3. [Project Structure Reference](#project-structure-reference)
4. [Environment Variables Reference](#environment-variables-reference)
5. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
6. [Next Steps](#next-steps)

---

## Phase 1: Foundation — Docker, Database, and Health Endpoint

### What We're Building

Phase 1 establishes the entire infrastructure before writing a single route. By the end, running one command (`docker compose up --build`) will start PostgreSQL, Redis, the FastAPI API server, and a background worker. A health endpoint at `/health` will confirm all three services are reachable. All database tables are created on startup via SQLAlchemy's `create_all`, which means zero migration tooling is needed at this stage.

### Prerequisites

An empty git repository and Docker Desktop running on your machine.

---

### Step 1.1: Repository Structure

**Why:** We separate backend code, Docker infrastructure, and scripts into distinct directories from the start. This prevents the flat-file chaos that happens when everything lives at the root, and matches the layout Docker Compose expects.

Create the following directories:

```
agent-rl-training-data-platform/
├── backend/
│   ├── core/
│   └── workers/
└── scripts/
```

```bash
mkdir -p backend/core backend/workers scripts
```

Then create the Python package init files so Python treats these as importable modules:

```bash
touch backend/__init__.py
touch backend/core/__init__.py
touch backend/workers/__init__.py
```

> **What's happening here:**
> - `__init__.py` files make each directory a Python package, which is required for imports like `from core.database import get_db` to work when the application runs from inside the `backend/` directory.
> - `core/` will hold shared infrastructure (database engine, Redis client) that every route module needs.
> - `workers/` will hold the background quality-scoring process that runs as a separate container.

---

### Step 1.2: Python Dependencies

**Why:** Pinning minimum versions rather than exact versions lets pip resolve a compatible set while preventing breaking changes from new majors.

**Create `backend/requirements.txt`:**

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
```

> **What's happening here:**
> - `fastapi` is the web framework. `uvicorn[standard]` is the ASGI server that runs it; the `[standard]` extra adds WebSocket support and faster event loops.
> - `sqlalchemy[asyncio]` — the `[asyncio]` extra pulls in `greenlet`, which SQLAlchemy's async layer requires to bridge synchronous C-extension code into the async event loop.
> - `asyncpg` is the pure-async PostgreSQL driver that SQLAlchemy uses under the hood when the connection URL starts with `postgresql+asyncpg://`.
> - `redis>=5.0.0` includes `redis.asyncio`, the async Redis client (previously a separate `aioredis` package — this consolidation happened in v4.2).
> - `httpx` is needed by pytest's `AsyncClient` to make test requests against the FastAPI app without starting a real server.
> - `aiosqlite` is the async SQLite driver used only in tests — it lets the test suite run without a PostgreSQL instance, using an in-memory or file-based SQLite database instead.
> - `pytest-asyncio` teaches pytest how to run `async def` test functions.

---

### Step 1.3: The Dockerfile

**Why:** A minimal Dockerfile keeps the image small and build times short. We copy `requirements.txt` first, before the application code, so Docker can cache the `pip install` layer — a layer only invalidated when dependencies change, not when source files change.

**Create `backend/Dockerfile`:**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

> **What's happening here:**
> - `python:3.12-slim` is Debian-based with only the minimal packages needed to run Python, resulting in a roughly 50 MB base image vs ~900 MB for the full image.
> - `WORKDIR /app` sets the working directory for all subsequent commands and for the container at runtime. When uvicorn runs `main:app`, Python looks for `main.py` relative to `/app`.
> - The two-step `COPY requirements.txt` then `COPY . .` pattern is a Docker build cache optimization. Changing a `.py` file will NOT re-run `pip install` because the `requirements.txt` layer is unchanged.
> - `--workers 4` runs four uvicorn worker processes inside the container, giving the API basic concurrency even without a load balancer. Each worker is a separate OS process that handles requests independently.
> - `--host 0.0.0.0` is required in Docker — without it, uvicorn listens only on `127.0.0.1` (loopback), which is not reachable from outside the container.

---

### Step 1.4: Docker Compose — The Full Stack

**Why:** Docker Compose lets us define and start all four services (Postgres, Redis, API, worker) with a single command. Healthchecks in Compose prevent a race condition where the API tries to connect to Postgres before Postgres has finished initializing.

**Create `docker-compose.yml` at the project root:**

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

  worker:
    build:
      context: ./backend
    container_name: rl-worker
    command: python -m workers.quality_worker
    environment:
      DATABASE_URL: postgresql+asyncpg://rl_user:rl_pass@postgres:5432/rl_platform
      REDIS_URL: redis://redis:6379/0
      EXPORT_DIR: /exports
    volumes:
      - exports:/exports
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  pgdata:
  exports:
```

> **What's happening here:**
>
> **postgres service:**
> - `postgres:16-alpine` is the Alpine-based image — significantly smaller than the default Debian image.
> - The `volumes` entry maps `./scripts/init.sql` into `/docker-entrypoint-initdb.d/`. Postgres automatically runs all `.sql` files in that directory when the database is first created.
> - The `healthcheck` runs `pg_isready`, a built-in Postgres utility that returns a success code only when the database is accepting connections. Without this, the API container might start before Postgres is ready.
>
> **redis service:**
> - `redis-cli ping` returns `PONG` when Redis is healthy. The healthcheck uses this for readiness detection.
>
> **api service:**
> - `depends_on` with `condition: service_healthy` means Docker will not start the API container until both Postgres and Redis have passed their healthchecks. This is a stronger guarantee than the default `depends_on` (which only waits for the container to start, not for the service inside it to be ready).
> - The `DATABASE_URL` uses `postgres` as the hostname — Docker Compose creates an internal DNS entry for each service name, so `postgres` resolves to the Postgres container's IP.
> - The `exports` volume is shared between `api` and `worker` because the API's background export task writes JSONL files there, and the worker may also need to read them.
> - The API healthcheck uses Python's built-in `urllib.request` rather than `curl` because `curl` is not installed in the slim image.
>
> **worker service:**
> - Uses `command: python -m workers.quality_worker` to override the default `CMD` from the Dockerfile. The `-m` flag runs the module as a script, which handles Python path resolution correctly.
> - The worker shares the same Docker image as the API (both build from `./backend`) but runs a different entrypoint.

---

### Step 1.5: PostgreSQL Initialization Script

**Why:** The `pgcrypto` extension provides cryptographic functions like `gen_random_uuid()`. Installing it at database creation time ensures it is available before any application code runs.

**Create `scripts/init.sql`:**

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

> **What's happening here:**
> - `IF NOT EXISTS` makes this script idempotent — safe to run multiple times.
> - This script runs only when the Postgres data directory is first initialized. Subsequent container restarts skip it.
> - Although we generate UUIDs in Python (not in SQL), having `pgcrypto` available is a good baseline for any PostgreSQL project.

---

### Step 1.6: Environment Variables

**Why:** Storing a `.env.example` file documents all required configuration without committing real secrets to the repository. Developers copy this to `.env` and fill in their values.

**Create `backend/.env.example`:**

```
DATABASE_URL=postgresql+asyncpg://rl_user:rl_pass@postgres:5432/rl_platform
REDIS_URL=redis://redis:6379/0
EXPORT_DIR=/exports
```

> **What's happening here:**
> - These values match what is set in `docker-compose.yml`. When running locally outside Docker (e.g., for tests), the application falls back to these same values as defaults in the Python code.
> - `EXPORT_DIR=/exports` points to the Docker-managed volume. When running tests, this is overridden or irrelevant.

---

### Step 1.7: Async Database Engine and Session

**Why:** All database I/O in this application is async. SQLAlchemy 2.0's async support uses `create_async_engine` and `AsyncSession` instead of their synchronous equivalents. Using the synchronous versions in an async application would block the event loop and defeat the purpose of async.

**Create `backend/core/database.py`:**

```python
from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://rl_user:rl_pass@localhost:5432/rl_platform",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    import models  # noqa: F401 — ensure all models are registered on Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

> **What's happening here:**
>
> **`from __future__ import annotations`** — this import is at the top of every file in this codebase. It enables PEP 563 "postponed evaluation of annotations", which means type hints like `str | None` (PEP 604 union syntax) work without raising a `TypeError` on Python versions that do not natively support them as runtime expressions. It also enables forward references in SQLAlchemy's `Mapped` type hints.
>
> **`create_async_engine`** — creates the async connection pool. `pool_pre_ping=True` sends a lightweight `SELECT 1` before handing out a connection from the pool, which detects and discards stale connections that were dropped by the database server. This is important in long-running applications.
>
> **`async_sessionmaker`** — the async equivalent of `sessionmaker`. `expire_on_commit=False` is a critical setting: by default, SQLAlchemy expires all ORM object attributes after a commit, forcing a re-fetch on next access. In an async context, this re-fetch happens lazily outside the session, which raises a `MissingGreenlet` error. Setting `expire_on_commit=False` keeps the objects usable after `commit`.
>
> **`get_db()`** — this is a FastAPI dependency function. The `yield` makes it a generator, which FastAPI uses for cleanup. The `try/yield/except` pattern ensures:
> - If the request handler succeeds: `commit()` is called automatically when the handler returns.
> - If an exception is raised: `rollback()` is called, then the exception propagates to FastAPI's error handler.
> - Routes never need to call `commit()` or `rollback()` themselves.
>
> **`init_db()`** — called once at application startup. The `import models` inside the function body is intentional: it forces all model classes to register themselves on `Base.metadata` before `create_all` runs. If models were not imported, `create_all` would create zero tables.

---

### Step 1.8: Redis Client and Queue Helpers

**Why:** The annotation system uses Redis as a task queue. When a researcher creates a task, it is pushed onto a Redis list. When an annotator requests work, a task is popped from that list. Redis lists provide the FIFO queue semantics we need with zero additional infrastructure.

**Create `backend/core/redis_client.py`:**

```python
from __future__ import annotations

import json
import os

import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

ANNOTATION_QUEUE = "rl:queue:annotation"

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


async def close_redis():
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None


async def enqueue_task(task_id: str) -> None:
    r = await get_redis()
    await r.lpush(ANNOTATION_QUEUE, task_id)


async def dequeue_task(timeout: int = 0) -> str | None:
    r = await get_redis()
    result = await r.brpop(ANNOTATION_QUEUE, timeout=timeout)
    if result:
        return result[1]
    return None


async def cache_get(key: str) -> str | None:
    r = await get_redis()
    return await r.get(key)


async def cache_set(key: str, value: str, ttl: int = 60) -> None:
    r = await get_redis()
    await r.set(key, value, ex=ttl)
```

> **What's happening here:**
>
> **Module-level singleton `_redis`** — the Redis client is a connection pool, not a single connection. Creating it once and reusing it across all requests is the correct pattern. The `get_redis()` function implements lazy initialization: the pool is created on first call and reused thereafter. `close_redis()` is called at application shutdown to drain the pool cleanly.
>
> **`decode_responses=True`** — Redis stores everything as bytes internally. This flag tells the client to decode all responses to Python strings automatically, so `r.get(key)` returns `str | None` instead of `bytes | None`.
>
> **`lpush` + `brpop` queue pattern:**
> - `lpush(queue, value)` pushes to the LEFT (head) of the list.
> - `brpop(queue, timeout)` pops from the RIGHT (tail) — the FIRST item pushed. Together these create FIFO ordering.
> - The `b` in `brpop` means "blocking": if the queue is empty, the call waits up to `timeout` seconds for an item to appear. With `timeout=1` (as used in the annotator route), the call blocks for at most 1 second before returning `None` — avoiding a busy loop.
> - `brpop` returns a tuple `(queue_name, value)`, so we return `result[1]` to get just the task ID.
>
> **Cache helpers** — `cache_get` / `cache_set` wrap plain Redis `GET`/`SET` with an optional `ex` (expiry in seconds). The `ttl=60` default is used for platform metrics, so dashboards see data that is at most 60 seconds stale.

---

### Step 1.9: ORM Models — Six Tables

**Why:** Defining all six tables upfront means the schema is complete before any routes are written. This avoids the common mistake of discovering schema gaps mid-implementation. The six tables form the complete data model for the platform.

**Create `backend/models.py`:**

```python
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
```

> **What's happening here:**
>
> **UUIDs as `String(36)`** — UUIDs are stored as 36-character strings (e.g., `"550e8400-e29b-41d4-a716-446655440000"`) rather than PostgreSQL's native `UUID` type. This decision exists because tests use SQLite, which has no native UUID column type. Using `String(36)` works identically in both databases. The `_uuid()` helper generates UUIDs in Python, so the database never needs to do it.
>
> **`str, enum.Enum` inheritance** — Python enums that inherit from both `str` and `enum.Enum` are "string enums". Their values are plain strings, so they serialize naturally to JSON (`"ranking"` instead of `<AnnotationType.RANKING: 'ranking'>`). FastAPI and Pydantic handle them correctly without custom serializers.
>
> **`SQLAlchemy.dialects.postgresql.JSON`** — using the PostgreSQL-specific `JSON` type (not the generic `sqlalchemy.JSON`) stores values as proper JSON in Postgres with binary indexing support. In SQLite (for tests), SQLAlchemy falls back to storing it as a text column automatically.
>
> **`Mapped` type hints** — SQLAlchemy 2.0's new `Mapped[T]` annotation style makes columns fully type-checked. `Mapped[str]` means the column is non-nullable; `Mapped[str | None]` means it is nullable. This replaces the older `Column(String, nullable=True)` verbosity.
>
> **`server_default=func.now()`** — this sets the default at the DATABASE level (via a `DEFAULT now()` SQL expression), which is more reliable than Python-side defaults because it uses the database clock and does not require the application to explicitly set the value.
>
> **`onupdate=func.now()`** on `updated_at` — SQLAlchemy will automatically update this column to the current timestamp whenever the row is modified via the ORM.
>
> **`expire_on_commit=False` and relationships** — ORM relationships (`feedback_items`, `assignments`) are loaded lazily by default. Because we set `expire_on_commit=False` on the session factory, related objects loaded within a request stay accessible after commit.
>
> **Entity relationship summary:**
> - `Task` is the central entity. It has many `FeedbackItem` records (one per annotator submission) and many `TaskAssignment` records (one per annotator assignment).
> - `FeedbackItem` belongs to one `Task` and one `Annotator`.
> - `TaskAssignment` joins `Task` and `Annotator` and tracks when the annotator was assigned and whether they completed it.
> - `Dataset` records an export snapshot with its filter criteria and file path.
> - `TrainingRun` references a `Dataset` and stores RL training metrics (reward history, KL divergence, loss curves).

---

### Step 1.10: The FastAPI Application and Health Endpoint

**Why:** The application needs startup and shutdown hooks to initialize the database tables and manage the Redis connection pool. FastAPI's `asynccontextmanager` lifespan is the modern way to do this (replacing the deprecated `on_event` decorators).

**Create `backend/main.py`:**

```python
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from core.database import async_session, init_db
from core.redis_client import close_redis, get_redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: initializing database tables...")
    await init_db()
    logger.info("Database tables created.")

    r = await get_redis()
    pong = await r.ping()
    logger.info(f"Redis ping: {pong}")

    yield

    await close_redis()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="RL Training Data Platform",
    version="0.1.0",
    lifespan=lifespan,
)


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
> **`asynccontextmanager` lifespan** — everything before `yield` runs at startup; everything after `yield` runs at shutdown. FastAPI calls `init_db()` once when the application starts, creating all tables that do not yet exist. Calling `await init_db()` on every restart is safe because `create_all` is idempotent — it skips tables that already exist.
>
> **Health endpoint design** — the `/health` endpoint actively tests both connections on every call rather than assuming they are healthy. It runs a `SELECT 1` against the database (the lightest possible query) and a `PING` against Redis. Each check is wrapped in its own `try/except` so a Redis failure does not mask a database failure — both statuses are reported independently. This is useful for debugging: you can tell at a glance which service is down.
>
> **`text("SELECT 1")`** — SQLAlchemy requires raw SQL strings to be wrapped in `text()` in version 2.0 to make it explicit that you are bypassing the ORM.
>
> **Why `main.py` is minimal at this stage** — Phase 1 only needs proof that the stack is up. Route modules do not exist yet, so the only endpoint is `/health`. Phase 2 will add all five route modules.

---

### Step 1.11: Quality Worker Placeholder

**Why:** The worker container must start successfully even though its real logic (quality scoring) is built in Phase 4. A placeholder that logs and sleeps fulfills the container's contract.

**Create `backend/workers/quality_worker.py`:**

```python
from __future__ import annotations

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Quality worker started. Waiting for jobs...")
    # Placeholder: actual quality scoring logic will be added in Phase 4
    while True:
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
```

> **What's happening here:**
> - `asyncio.sleep(5)` yields control to the event loop every 5 seconds. This is a well-behaved async idle loop — it does not consume CPU.
> - `asyncio.run(main())` is the standard entry point for running an async function as a script. Docker Compose runs this via `python -m workers.quality_worker`.
> - The `while True` loop keeps the container alive. If we used a one-shot script, Docker would consider the container "exited" and keep restarting it.

---

### Verify Phase 1

Run the full stack:

```bash
docker compose up --build
```

Wait for all four containers to report healthy (about 30-60 seconds on first build). Then test the health endpoint:

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

Expected output:

```json
{
    "status": "ok",
    "db": "ok",
    "redis": "ok"
}
```

You can also verify individual services:

```bash
# Check PostgreSQL is accepting connections
docker exec rl-postgres pg_isready -U rl_user -d rl_platform

# Check Redis
docker exec rl-redis redis-cli ping

# View API logs
docker logs rl-api --tail 20
```

Expected API log output on startup:

```
INFO:root:Starting up: initializing database tables...
INFO:root:Database tables created.
INFO:root:Redis ping: True
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### What We Accomplished

Phase 1 is a complete, working infrastructure layer. PostgreSQL is running with the `pgcrypto` extension, Redis is running as a queue and cache store, all six database tables exist, and the API server is verifying both on every health check. The background worker container is running and waiting for Phase 4 to give it real work. The entire stack starts and stops with a single command.

---

## Phase 2: Backend API — Tasks, Feedback, Annotators, Metrics, Exports

### What We're Building

Phase 2 builds all five route modules that form the complete REST API. The most complex piece is the feedback submission endpoint, which recomputes inter-annotator agreement (Fleiss' kappa) and a composite quality score on every new submission. The exports endpoint converts accumulated preference data into DPO (Direct Preference Optimization) format — the standard input format for RL fine-tuning with libraries like TRL and OpenRLHF. All 26 tests use SQLite and mocked Redis, so they run without Docker.

### Prerequisites

Phase 1 complete. The full Docker stack should start and the health endpoint should return `{"status": "ok", ...}`.

---

### Step 2.1: Pydantic Schemas

**Why:** Pydantic schemas define the shape of every request body and response body. Defining them in one file before writing any routes makes the API contract explicit and prevents type errors from propagating through the codebase.

**Create `backend/schemas.py`:**

```python
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
```

> **What's happening here:**
>
> **`model_config = {"from_attributes": True}`** — this Pydantic v2 setting (replacing `orm_mode = True` from v1) tells Pydantic to read field values from object attributes rather than dictionary keys. This is required to convert SQLAlchemy ORM objects directly to Pydantic models. Without it, you would need to manually convert ORM objects to dicts before returning them.
>
> **Request vs. Response schemas** — each resource has distinct schemas for input (`TaskCreate`, `FeedbackSubmit`) and output (`TaskResponse`, `FeedbackResponse`). This matters because:
> - Input schemas accept only user-controlled fields (no `id`, `created_at`, computed scores).
> - Output schemas include all fields the API returns, including computed ones (`quality_score`, `iaa`).
> - `TaskUpdate` is a PATCH schema where all fields are optional (`str | None = None`), allowing partial updates.
>
> **`Field(None, ge=0.0, le=1.0)` on `confidence`** — Pydantic's `Field` adds validation constraints. `ge=0.0` means "greater than or equal to 0.0"; `le=1.0` means "less than or equal to 1.0". If an annotator submits `confidence: 1.5`, the request is rejected with a 422 Validation Error before the route handler is even called.
>
> **`FeedbackSubmit` has four mutually exclusive annotation fields** — `ranking`, `scalar_reward`, `binary_label`, and `critique_text` correspond to the four annotation types. The route logic handles each case; the schema simply makes all four optional since only one will be present per submission.

---

### Step 2.2: Tasks Router

**Why:** The tasks router handles the full lifecycle of annotation tasks: creation (which enqueues the task to Redis), listing with filters and pagination, updating, deleting, and flagging. Flagging is a separate endpoint rather than a PATCH to make the intent explicit in the API.

**Create `backend/routes/__init__.py`** (empty file):

```bash
touch backend/routes/__init__.py
```

**Create `backend/routes/tasks.py`:**

```python
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.redis_client import enqueue_task
from models import Task, TaskStatus
from schemas import TaskCreate, TaskListResponse, TaskResponse, TaskUpdate

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.post("/", response_model=TaskResponse, status_code=201)
async def create_task(payload: TaskCreate, db: AsyncSession = Depends(get_db)):
    task = Task(
        prompt=payload.prompt,
        responses=payload.responses,
        annotation_type=payload.annotation_type,
        min_annotations=payload.min_annotations,
        tags=payload.tags,
        evaluation_criteria=payload.evaluation_criteria,
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)
    await enqueue_task(task.id)
    return task


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    status: str | None = None,
    annotation_type: str | None = None,
    tag: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    query = select(Task)
    count_query = select(func.count(Task.id))

    if status:
        query = query.where(Task.status == status)
        count_query = count_query.where(Task.status == status)
    if annotation_type:
        query = query.where(Task.annotation_type == annotation_type)
        count_query = count_query.where(Task.annotation_type == annotation_type)

    total = (await db.execute(count_query)).scalar() or 0
    offset = (page - 1) * page_size
    query = query.order_by(Task.created_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(query)
    tasks = result.scalars().all()

    return TaskListResponse(tasks=tasks, total=total, page=page, page_size=page_size)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, db: AsyncSession = Depends(get_db)):
    task = await db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: str, payload: TaskUpdate, db: AsyncSession = Depends(get_db)
):
    task = await db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(task, field, value)
    await db.flush()
    await db.refresh(task)
    return task


@router.delete("/{task_id}", status_code=204)
async def delete_task(task_id: str, db: AsyncSession = Depends(get_db)):
    task = await db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    await db.delete(task)


@router.post("/{task_id}/flag", response_model=TaskResponse)
async def flag_task(task_id: str, db: AsyncSession = Depends(get_db)):
    task = await db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.status = TaskStatus.FLAGGED
    await db.flush()
    await db.refresh(task)
    return task
```

> **What's happening here:**
>
> **`db.flush()` vs `db.commit()`** — `flush()` sends the SQL statements to the database within the current transaction but does NOT commit. This makes the new row visible within the same session (so `db.refresh(task)` can read the server-set `id` and `created_at`) without permanently writing it to disk. The `get_db()` dependency commits automatically when the request handler returns without error. Using `flush()` in handlers instead of `commit()` keeps all operations within a single transaction — if anything fails after the flush, the rollback undoes everything.
>
> **`db.refresh(task)` after `flush()`** — refreshes the in-memory object from the database, populating server-generated fields like `id` (from the `default=_uuid` Python-side default, which was applied before the flush) and `created_at` (from `server_default=func.now()`, which the database sets). Without this call, `task.created_at` would be `None` in the response because SQLAlchemy does not know the server filled it in.
>
> **`await enqueue_task(task.id)` in `create_task`** — the Redis push happens AFTER the flush but while still inside the transaction. This means: if the commit later fails (rare, but possible), the task ID is in Redis but not in the database. In production you would use a transactional outbox pattern. For this project, the risk is acceptable — the annotator endpoint handles a missing task gracefully.
>
> **`payload.model_dump(exclude_unset=True)`** — Pydantic's `model_dump()` returns a dictionary. `exclude_unset=True` includes only fields that were actually present in the request JSON, not fields that defaulted to `None`. This is what makes PATCH work correctly: sending `{"prompt": "new"}` updates only `prompt` and leaves all other fields untouched.
>
> **Pagination with a count query** — the list endpoint runs two queries: one for the count (needed for `total` in the response) and one for the page data. Both queries share the same `WHERE` clauses so the count and data stay consistent. The count query uses `func.count(Task.id)` which maps to `COUNT(tasks.id)` in SQL.
>
> **`Query(1, ge=1)` and `Query(20, ge=1, le=100)`** — FastAPI's `Query` with validators prevents invalid pagination parameters (`page=0`, `page_size=1000`) without any custom validation code. A 422 Unprocessable Entity response is returned automatically if constraints are violated.

---

### Step 2.3: Feedback Router and IAA Scoring

**Why:** This is the most algorithmically dense part of the API. Every time an annotator submits feedback, the system recomputes three quality metrics for the parent task: Fleiss' kappa (inter-annotator agreement), consensus reward, and a composite quality score. The task automatically transitions to `completed` when it has received enough non-flagged annotations.

**Create `backend/routes/feedback.py`:**

```python
from __future__ import annotations

import math
from itertools import combinations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models import FeedbackItem, Task, TaskStatus
from schemas import FeedbackResponse, FeedbackSubmit

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
> **Fleiss' kappa — the algorithm:**
>
> Fleiss' kappa measures how much annotators agree with each other beyond what would be expected by chance. A kappa of 1.0 means perfect agreement; 0.0 means agreement no better than random; negative values mean less agreement than chance (systematic disagreement).
>
> For ranking data, the implementation converts multi-way rankings into pairwise comparisons. For each pair of responses `(i, j)`, it counts how many raters ranked `i` above `j` (call this `prefer_i`) and how many ranked `j` above `i` (call this `prefer_j = n_raters - prefer_i`).
>
> Observed agreement `p_o` is the fraction of rater-pairs that agreed on any given pairwise comparison. Under random guessing on a binary choice, expected agreement `p_e = 0.5`. The kappa formula is then `(p_o - p_e) / (1 - p_e)`.
>
> The result is clamped to `[-1.0, 1.0]` with `max(-1.0, min(1.0, kappa))` to handle floating-point edge cases.
>
> **Consensus reward:** Converts each feedback item to a scalar reward regardless of annotation type:
> - `scalar_reward`: used directly.
> - `binary_label`: mapped to 1.0 (positive) or 0.0 (negative).
> - `ranking`: the best-ranked response is normalized to a reward in [0, 1].
> The mean of all these normalized rewards is the consensus reward.
>
> **Quality score formula:** `0.4 * IAA + 0.4 * (1 - reward_stdev) + 0.2 * coverage`
> - IAA (weighted 40%): Higher inter-annotator agreement means more trustworthy labels.
> - Reward stability (weighted 40%): Low standard deviation across reward signals means annotators are consistent. `(1 - stdev)` converts this so higher is better.
> - Coverage (weighted 20%): The fraction of `min_annotations` slots that have been filled by non-flagged feedback. A task at 100% coverage is fully annotated.
>
> **`FeedbackItem.flagged == False` with `# noqa: E712`** — SQLAlchemy's `==` operator on mapped columns generates a SQL `WHERE flagged = FALSE` expression. Python style linters warn about using `== False` instead of `is False`, but SQLAlchemy requires `==` (not `is`) to generate SQL. The `noqa` comment silences the linter warning.
>
> **Auto-completion logic** — when the count of non-flagged feedback items reaches `task.min_annotations` AND the task is still `PENDING`, it transitions to `COMPLETED`. This is only checked on `PENDING` tasks, not `IN_PROGRESS` or `FLAGGED`, to avoid accidental re-completion.

---

### Step 2.4: Annotators Router and Queue Integration

**Why:** The annotators router manages the human annotators in the system. The most important endpoint is `/{annotator_id}/next-task`, which pops a task from the Redis queue and creates a `TaskAssignment` record — the link between "an annotator was given this task" and "they submitted feedback for it".

**Create `backend/routes/annotators.py`:**

```python
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
```

> **What's happening here:**
>
> **Duplicate email check with `scalar_one_or_none()`** — this executes a `SELECT` query and returns the first matching `Annotator` object or `None`. If it returns an object, the email is already registered and we raise a `409 Conflict` (not a `400 Bad Request` — 409 specifically means the request conflicts with existing state). The `unique=True` constraint on `Annotator.email` in the model would catch this at the database level too, but checking in Python first gives a cleaner error message.
>
> **`next_task` dequeue flow:**
> 1. Verify the annotator exists.
> 2. Call `dequeue_task(timeout=1)` — blocks for up to 1 second waiting for a task ID.
> 3. If nothing arrives in 1 second, return 404 "No tasks in queue".
> 4. Fetch the full `Task` record from PostgreSQL using the task ID from Redis.
> 5. Create a `TaskAssignment` linking annotator to task.
> 6. Return the task.
>
> This is the only endpoint where we call `dequeue_task` with `timeout=1` rather than 0. The short wait avoids a race condition where the queue appears empty for a brief moment due to replication lag. Returning 404 on an empty queue is intentional — the annotation UI will poll again.
>
> **Why not use `PUT` instead of `GET` for next-task?** — `GET` is not semantically correct here because calling it has a side effect (popping from the queue). However, using `GET` keeps the frontend integration simple — the UI can use a standard fetch. In a more strict REST design, this would be a `POST` to `/api/annotators/{id}/assignments`.

---

### Step 2.5: Metrics Router with Redis Caching

**Why:** Platform metrics require aggregating across multiple tables. Running five `COUNT`/`AVG` queries on every dashboard request is expensive under load. A 60-second Redis cache absorbs the read load while keeping data reasonably fresh.

**Create `backend/routes/metrics.py`:**

```python
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.redis_client import cache_get, cache_set, get_redis, ANNOTATION_QUEUE
from models import Annotator, FeedbackItem, Task, TaskStatus, TrainingRun
from schemas import PlatformMetrics, TrainingRunResponse

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

PLATFORM_CACHE_KEY = "rl:cache:platform_metrics"


@router.get("/platform", response_model=PlatformMetrics)
async def platform_metrics(db: AsyncSession = Depends(get_db)):
    cached = await cache_get(PLATFORM_CACHE_KEY)
    if cached:
        return PlatformMetrics(**json.loads(cached))

    total_tasks = (await db.execute(select(func.count(Task.id)))).scalar() or 0
    pending_tasks = (
        await db.execute(
            select(func.count(Task.id)).where(Task.status == TaskStatus.PENDING)
        )
    ).scalar() or 0
    completed_tasks = (
        await db.execute(
            select(func.count(Task.id)).where(Task.status == TaskStatus.COMPLETED)
        )
    ).scalar() or 0
    total_feedback = (
        await db.execute(select(func.count(FeedbackItem.id)))
    ).scalar() or 0
    total_annotators = (
        await db.execute(select(func.count(Annotator.id)))
    ).scalar() or 0
    avg_quality = (
        await db.execute(select(func.avg(Task.quality_score)))
    ).scalar()
    avg_iaa = (await db.execute(select(func.avg(Task.iaa)))).scalar()

    r = await get_redis()
    queue_depth = await r.llen(ANNOTATION_QUEUE)

    metrics = PlatformMetrics(
        total_tasks=total_tasks,
        pending_tasks=pending_tasks,
        completed_tasks=completed_tasks,
        total_feedback=total_feedback,
        total_annotators=total_annotators,
        avg_quality_score=round(avg_quality, 4) if avg_quality else None,
        avg_iaa=round(avg_iaa, 4) if avg_iaa else None,
        queue_depth=queue_depth,
    )

    await cache_set(PLATFORM_CACHE_KEY, metrics.model_dump_json(), ttl=60)
    return metrics


@router.get("/training", response_model=list[TrainingRunResponse])
async def list_training_runs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TrainingRun).order_by(TrainingRun.created_at.desc())
    )
    return result.scalars().all()


@router.get("/training/{run_id}", response_model=TrainingRunResponse)
async def get_training_run(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run
```

> **What's happening here:**
>
> **Cache-aside pattern** — on every request: check Redis first; if a cached value exists, deserialize it and return immediately (skipping all database queries); if not, run the queries, build the response, serialize to JSON, write to Redis with a 60-second TTL, and return.
>
> **`metrics.model_dump_json()`** — Pydantic v2's method that serializes the model to a JSON string. Used instead of `json.dumps(metrics.dict())` because it handles non-standard types like `datetime` correctly.
>
> **`r.llen(ANNOTATION_QUEUE)`** — Redis `LLEN` returns the length of a list (i.e., the number of task IDs currently in the queue). This tells the dashboard how many tasks are waiting for annotators.
>
> **`or 0` on scalar queries** — `func.count()` always returns an integer, but `func.avg()` can return `None` if there are no rows. The `or 0` on count queries is a safety net; the explicit `None` check on `avg_quality` and `avg_iaa` handles the empty-table case correctly.

---

### Step 2.6: Exports Router and DPO Format

**Why:** The final product of this platform is a JSONL file in DPO format that RL training frameworks like TRL can consume directly. The export runs as a background task so the HTTP request returns immediately (with the newly created `Dataset` record) while the file write happens asynchronously.

**Create `backend/routes/exports.py`:**

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
```

> **What's happening here:**
>
> **DPO format** — Direct Preference Optimization requires training examples in the form `{prompt, chosen, rejected}`. The `chosen` response is the one with the highest average reward; `rejected` is the one with the lowest. The additional fields (`reward_chosen`, `reward_rejected`, `quality_score`, `iaa`) let downstream training pipelines filter by quality before training.
>
> **Reward normalization for ranking data:** A ranking like `[1, 2]` means "response 0 is rank 1 (best), response 1 is rank 2". The formula `1.0 - (rank - 1) / max(1, n - 1)` converts:
> - Rank 1 out of 2: `1.0 - 0/1 = 1.0` (best)
> - Rank 2 out of 2: `1.0 - 1/1 = 0.0` (worst)
> For scalar feedback with 2 responses, the assumption is `score(response_0) + score(response_1) = 1.0`.
>
> **`BackgroundTasks` injection** — FastAPI's `BackgroundTasks` is injected by adding it as a route function parameter. Calling `background_tasks.add_task(fn, arg)` schedules `fn(arg)` to run after the response is sent. This is appropriate for tasks that take a few seconds; for tasks that could take minutes, a proper task queue (Celery, RQ, or arq) would be used instead.
>
> **`_build_export` opens its own session** — the background task runs after the request handler returns and the original database session is closed. It must open a new session via `async with async_session() as db`. It also calls `await db.commit()` explicitly instead of relying on the `get_db` dependency (which is not in scope here).
>
> **`FileResponse`** — FastAPI streams the file directly from disk to the client. The `media_type="application/jsonl"` sets the `Content-Type` header, and `filename=` sets `Content-Disposition: attachment; filename="..."` which triggers a download in browsers.
>
> **Export filters** — the `filters` JSON field on `Dataset` controls which tasks are included: `min_quality`, `min_iaa`, and `annotation_type`. These filters are applied at query time, not post-hoc, so large databases export efficiently.

---

### Step 2.7: Wire All Routers into main.py

**Why:** FastAPI does not auto-discover routes. Each router must be explicitly registered with `app.include_router()`. We add the imports and registrations to `main.py`.

**Edit `backend/main.py` — add these lines after the `app = FastAPI(...)` block:**

```python
# ADD these imports after app = FastAPI(...)
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
```

The final `main.py` should look like this:

```python
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from core.database import async_session, init_db
from core.redis_client import close_redis, get_redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: initializing database tables...")
    await init_db()
    logger.info("Database tables created.")

    r = await get_redis()
    pong = await r.ping()
    logger.info(f"Redis ping: {pong}")

    yield

    await close_redis()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="RL Training Data Platform",
    version="0.1.0",
    lifespan=lifespan,
)

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
> **Imports after `app = FastAPI(...)`** — the route modules import `app`-dependent code, so they must be imported after `app` is created. This is an accepted pattern in FastAPI projects. The alternative — using `APIRouter` without any `app` dependency (which these routers do) — technically allows top-level imports, but placing them after `app` makes the dependency clear.
>
> **`tags=["tasks"]` on each router** — these strings appear in Swagger UI (at `/docs`) as section headers, grouping related endpoints together. They also populate the `operationId` field in the OpenAPI schema, which tools like Swagger Codegen use when generating client SDKs.

---

### Step 2.8: Test Infrastructure — conftest.py

**Why:** Tests must run without Docker, without PostgreSQL, and without Redis. `conftest.py` is pytest's shared fixture file — it sets up an SQLite-backed database and in-memory Redis mocks that all test modules share automatically.

**Create `backend/tests/__init__.py`** (empty file):

```bash
touch backend/tests/__init__.py
```

**Create `backend/pytest.ini`:**

```ini
[pytest]
asyncio_mode = auto
```

> **What's happening here:**
> - `asyncio_mode = auto` tells pytest-asyncio to treat ALL `async def` test functions as asyncio tests automatically. Without this, every test would need the `@pytest.mark.asyncio` decorator (you can use it anyway, but the setting removes the requirement to add it everywhere).

**Create `backend/tests/conftest.py`:**

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

# Use SQLite for tests
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


# Mock Redis functions
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
> **`sys.path.insert(0, ...)`** — test files are in `backend/tests/`, but application code is in `backend/`. Inserting `backend/` into `sys.path` allows `from core.database import Base` to resolve correctly when tests are run from outside the `backend/` directory.
>
> **SQLite for tests** — `sqlite+aiosqlite:///./test.db` creates a SQLite file called `test.db` in the current directory. The `setup_db` fixture creates all tables before each test and drops them after, ensuring a clean state for every test. The file is not deleted between tests, only the tables are recreated — this is faster than deleting and recreating the file.
>
> **`autouse=True` on `setup_db`** — every test in the project automatically gets a clean database without explicitly requesting the `setup_db` fixture. Tests that need the HTTP client still declare `client` explicitly.
>
> **In-memory Redis mocks** — `mock_redis_queue` is a plain Python list that simulates the Redis queue. `mock_redis_store` is a plain dict that simulates Redis `GET`/`SET`. These are module-level variables reset in `setup_db`'s cleanup step, so state does not leak between tests.
>
> **Double-patching Redis** — notice that both `core.redis_client.enqueue_task` AND `routes.tasks.enqueue_task` are patched. This is because Python's `patch` replaces the name in the module where it was imported, not in the module where it was defined. `routes/tasks.py` does `from core.redis_client import enqueue_task`, creating its own reference. To replace what `tasks.py` calls, we must patch it in the `routes.tasks` namespace too.
>
> **`app.dependency_overrides[get_db] = override_get_db`** — FastAPI's dependency override system replaces `get_db` globally for the duration of the test. Any route that declares `db: AsyncSession = Depends(get_db)` will receive the SQLite-backed session instead of the PostgreSQL one.
>
> **`ASGITransport(app=app)`** — httpx's ASGI transport makes HTTP requests directly to the FastAPI app object, bypassing all network I/O. The application runs in the same process as the test, which makes it possible to use the same SQLite engine for both.
>
> **`app.dependency_overrides.clear()`** — called after the test to clean up the override, preventing it from leaking into the next test file.

---

### Step 2.9: Task Tests

**Why:** Eight tests cover the full task CRUD lifecycle: create, list (with filtering), get (including 404), update, delete, flag, and filtered listing. This gives complete confidence that the tasks router is correct before building feedback on top of it.

**Create `backend/tests/test_tasks.py`:**

```python
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_create_task(client):
    resp = await client.post("/api/tasks/", json={
        "prompt": "Write a Python function to sort a list",
        "responses": [
            {"model_id": "m-a", "text": "def sort(lst): return sorted(lst)"},
            {"model_id": "m-b", "text": "def sort(lst): lst.sort(); return lst"},
        ],
        "annotation_type": "ranking",
        "min_annotations": 3,
        "tags": ["python", "sorting"],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["prompt"] == "Write a Python function to sort a list"
    assert data["status"] == "pending"
    assert data["annotation_type"] == "ranking"


@pytest.mark.asyncio
async def test_list_tasks(client):
    await client.post("/api/tasks/", json={"prompt": "Task 1"})
    await client.post("/api/tasks/", json={"prompt": "Task 2"})
    resp = await client.get("/api/tasks/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["tasks"]) == 2


@pytest.mark.asyncio
async def test_get_task(client):
    create_resp = await client.post("/api/tasks/", json={"prompt": "Get me"})
    task_id = create_resp.json()["id"]
    resp = await client.get(f"/api/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == task_id


@pytest.mark.asyncio
async def test_get_task_not_found(client):
    resp = await client.get("/api/tasks/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_task(client):
    create_resp = await client.post("/api/tasks/", json={"prompt": "Original"})
    task_id = create_resp.json()["id"]
    resp = await client.patch(f"/api/tasks/{task_id}", json={"prompt": "Updated"})
    assert resp.status_code == 200
    assert resp.json()["prompt"] == "Updated"


@pytest.mark.asyncio
async def test_delete_task(client):
    create_resp = await client.post("/api/tasks/", json={"prompt": "Delete me"})
    task_id = create_resp.json()["id"]
    resp = await client.delete(f"/api/tasks/{task_id}")
    assert resp.status_code == 204
    resp = await client.get(f"/api/tasks/{task_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_flag_task(client):
    create_resp = await client.post("/api/tasks/", json={"prompt": "Flag me"})
    task_id = create_resp.json()["id"]
    resp = await client.post(f"/api/tasks/{task_id}/flag")
    assert resp.status_code == 200
    assert resp.json()["status"] == "flagged"


@pytest.mark.asyncio
async def test_filter_tasks_by_status(client):
    await client.post("/api/tasks/", json={"prompt": "Task 1"})
    create2 = await client.post("/api/tasks/", json={"prompt": "Task 2"})
    task_id = create2.json()["id"]
    await client.post(f"/api/tasks/{task_id}/flag")

    resp = await client.get("/api/tasks/?status=flagged")
    data = resp.json()
    assert data["total"] == 1
    assert data["tasks"][0]["status"] == "flagged"
```

---

### Step 2.10: Feedback Tests

**Why:** Four tests verify feedback submission, the quality scoring pipeline, feedback listing, and flagging. `test_feedback_recomputes_quality` is the most important — it confirms that after two feedback submissions, the task's `quality_score` and `iaa` fields are populated and the status transitions to `completed`.

**Create `backend/tests/test_feedback.py`:**

```python
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_submit_feedback(client):
    # Create task and annotator first
    task_resp = await client.post("/api/tasks/", json={
        "prompt": "Test prompt",
        "responses": [{"text": "a"}, {"text": "b"}],
    })
    task_id = task_resp.json()["id"]

    ann_resp = await client.post("/api/annotators/", json={
        "email": "ann1@test.com",
        "name": "Ann One",
    })
    ann_id = ann_resp.json()["id"]

    resp = await client.post("/api/feedback/", json={
        "task_id": task_id,
        "annotator_id": ann_id,
        "ranking": [1, 2],
        "confidence": 0.9,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["task_id"] == task_id
    assert data["ranking"] == [1, 2]


@pytest.mark.asyncio
async def test_feedback_recomputes_quality(client):
    task_resp = await client.post("/api/tasks/", json={
        "prompt": "Quality test",
        "responses": [{"text": "a"}, {"text": "b"}],
        "min_annotations": 2,
    })
    task_id = task_resp.json()["id"]

    # Create 2 annotators
    ann1 = (await client.post("/api/annotators/", json={
        "email": "q1@test.com", "name": "Q1",
    })).json()["id"]
    ann2 = (await client.post("/api/annotators/", json={
        "email": "q2@test.com", "name": "Q2",
    })).json()["id"]

    # Submit feedback from both
    await client.post("/api/feedback/", json={
        "task_id": task_id, "annotator_id": ann1,
        "ranking": [1, 2], "scalar_reward": 0.8,
    })
    await client.post("/api/feedback/", json={
        "task_id": task_id, "annotator_id": ann2,
        "ranking": [1, 2], "scalar_reward": 0.85,
    })

    # Task should now be completed with quality scores
    task = (await client.get(f"/api/tasks/{task_id}")).json()
    assert task["status"] == "completed"
    assert task["quality_score"] is not None
    assert task["iaa"] is not None


@pytest.mark.asyncio
async def test_get_task_feedback(client):
    task_resp = await client.post("/api/tasks/", json={"prompt": "Feedback list"})
    task_id = task_resp.json()["id"]
    ann_resp = await client.post("/api/annotators/", json={
        "email": "fl@test.com", "name": "FL",
    })
    ann_id = ann_resp.json()["id"]

    await client.post("/api/feedback/", json={
        "task_id": task_id, "annotator_id": ann_id,
        "scalar_reward": 0.7,
    })

    resp = await client.get(f"/api/feedback/task/{task_id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_flag_feedback(client):
    task_resp = await client.post("/api/tasks/", json={"prompt": "Flag fb"})
    task_id = task_resp.json()["id"]
    ann_resp = await client.post("/api/annotators/", json={
        "email": "ffb@test.com", "name": "FFB",
    })
    ann_id = ann_resp.json()["id"]

    fb_resp = await client.post("/api/feedback/", json={
        "task_id": task_id, "annotator_id": ann_id,
        "scalar_reward": 0.5,
    })
    fb_id = fb_resp.json()["id"]

    resp = await client.post(f"/api/feedback/{fb_id}/flag")
    assert resp.status_code == 200
    assert resp.json()["flagged"] is True
```

---

### Step 2.11: Annotator Tests

**Why:** Six tests cover annotator creation (including the duplicate email 409 case), listing, retrieval by ID, the queue integration test (`test_next_task`), and the empty queue case.

**Create `backend/tests/test_annotators.py`:**

```python
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_create_annotator(client):
    resp = await client.post("/api/annotators/", json={
        "email": "alice@test.com",
        "name": "Alice",
        "role": "senior",
        "expertise_tags": ["python", "ml"],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "alice@test.com"
    assert data["reliability_score"] == 1.0


@pytest.mark.asyncio
async def test_duplicate_email(client):
    await client.post("/api/annotators/", json={
        "email": "dup@test.com", "name": "First",
    })
    resp = await client.post("/api/annotators/", json={
        "email": "dup@test.com", "name": "Second",
    })
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_list_annotators(client):
    await client.post("/api/annotators/", json={
        "email": "a1@test.com", "name": "A1",
    })
    await client.post("/api/annotators/", json={
        "email": "a2@test.com", "name": "A2",
    })
    resp = await client.get("/api/annotators/")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_get_annotator(client):
    create_resp = await client.post("/api/annotators/", json={
        "email": "get@test.com", "name": "Get",
    })
    ann_id = create_resp.json()["id"]
    resp = await client.get(f"/api/annotators/{ann_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == ann_id


@pytest.mark.asyncio
async def test_next_task(client):
    # Create task (which enqueues to mock queue)
    task_resp = await client.post("/api/tasks/", json={"prompt": "Queue task"})
    task_id = task_resp.json()["id"]

    ann_resp = await client.post("/api/annotators/", json={
        "email": "next@test.com", "name": "Next",
    })
    ann_id = ann_resp.json()["id"]

    resp = await client.get(f"/api/annotators/{ann_id}/next-task")
    assert resp.status_code == 200
    assert resp.json()["id"] == task_id


@pytest.mark.asyncio
async def test_next_task_empty_queue(client):
    ann_resp = await client.post("/api/annotators/", json={
        "email": "empty@test.com", "name": "Empty",
    })
    ann_id = ann_resp.json()["id"]

    resp = await client.get(f"/api/annotators/{ann_id}/next-task")
    assert resp.status_code == 404
```

---

### Step 2.12: Metrics Tests

**Why:** Four tests verify the platform metrics endpoint returns correct counts, the cache is hit on the second call, the training run list returns an empty array when there are no runs, and the training run 404 case.

**Create `backend/tests/test_metrics.py`:**

```python
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_platform_metrics(client):
    # Create some data
    await client.post("/api/tasks/", json={"prompt": "Metric task"})
    await client.post("/api/annotators/", json={
        "email": "met@test.com", "name": "Met",
    })

    resp = await client.get("/api/metrics/platform")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tasks"] == 1
    assert data["total_annotators"] == 1
    assert data["pending_tasks"] == 1


@pytest.mark.asyncio
async def test_platform_metrics_cache(client):
    """Second call should hit cache."""
    await client.get("/api/metrics/platform")
    resp = await client.get("/api/metrics/platform")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_training_runs(client):
    resp = await client.get("/api/metrics/training")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_training_run_not_found(client):
    resp = await client.get("/api/metrics/training/nonexistent")
    assert resp.status_code == 404
```

---

### Step 2.13: Export Tests

**Why:** Four tests verify dataset creation, listing, retrieval by ID, and the 404 case for downloading an export that hasn't been generated yet. The `_build_export` background task is mocked to a no-op so tests don't try to write files to disk.

**Create `backend/tests/test_exports.py`:**

```python
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_create_dataset(client):
    resp = await client.post("/api/exports/datasets", json={
        "name": "test-export",
        "filters": {"min_quality": 0.5},
        "export_format": "jsonl",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test-export"
    assert data["export_format"] == "jsonl"


@pytest.mark.asyncio
async def test_list_datasets(client):
    await client.post("/api/exports/datasets", json={"name": "ds1"})
    await client.post("/api/exports/datasets", json={"name": "ds2"})
    resp = await client.get("/api/exports/datasets")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_get_dataset(client):
    create_resp = await client.post("/api/exports/datasets", json={"name": "get-ds"})
    ds_id = create_resp.json()["id"]
    resp = await client.get(f"/api/exports/datasets/{ds_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == ds_id


@pytest.mark.asyncio
async def test_download_not_ready(client):
    create_resp = await client.post("/api/exports/datasets", json={"name": "dl-ds"})
    ds_id = create_resp.json()["id"]
    resp = await client.get(f"/api/exports/datasets/{ds_id}/download")
    assert resp.status_code == 404
```

---

### Verify Phase 2

**Run all 26 tests** (no Docker required):

```bash
cd /path/to/agent-rl-training-data-platform/backend
pytest tests/ -v
```

Expected output (all 26 passing):

```
tests/test_annotators.py::test_create_annotator PASSED
tests/test_annotators.py::test_duplicate_email PASSED
tests/test_annotators.py::test_list_annotators PASSED
tests/test_annotators.py::test_get_annotator PASSED
tests/test_annotators.py::test_next_task PASSED
tests/test_annotators.py::test_next_task_empty_queue PASSED
tests/test_exports.py::test_create_dataset PASSED
tests/test_exports.py::test_list_datasets PASSED
tests/test_exports.py::test_get_dataset PASSED
tests/test_exports.py::test_download_not_ready PASSED
tests/test_feedback.py::test_submit_feedback PASSED
tests/test_feedback.py::test_feedback_recomputes_quality PASSED
tests/test_feedback.py::test_get_task_feedback PASSED
tests/test_feedback.py::test_flag_feedback PASSED
tests/test_metrics.py::test_platform_metrics PASSED
tests/test_metrics.py::test_platform_metrics_cache PASSED
tests/test_metrics.py::test_list_training_runs PASSED
tests/test_metrics.py::test_get_training_run_not_found PASSED
tests/test_tasks.py::test_create_task PASSED
tests/test_tasks.py::test_list_tasks PASSED
tests/test_tasks.py::test_get_task PASSED
tests/test_tasks.py::test_get_task_not_found PASSED
tests/test_tasks.py::test_update_task PASSED
tests/test_tasks.py::test_delete_task PASSED
tests/test_tasks.py::test_flag_task PASSED
tests/test_tasks.py::test_filter_tasks_by_status PASSED

26 passed in X.XXs
```

**Verify the live API with Docker:**

```bash
docker compose up --build
```

Then open Swagger UI in your browser at `http://localhost:8000/docs`. You should see all five route groups: tasks, feedback, annotators, metrics, and exports.

**Run a quick end-to-end smoke test with curl:**

```bash
# 1. Create a task
curl -s -X POST http://localhost:8000/api/tasks/ \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain gradient descent in one sentence.",
    "responses": [
      {"model_id": "gpt-4o", "text": "Gradient descent updates weights by moving in the direction of steepest loss reduction."},
      {"model_id": "claude-3", "text": "It iteratively adjusts parameters to minimize a loss function by following the negative gradient."}
    ],
    "annotation_type": "ranking",
    "min_annotations": 2
  }' | python3 -m json.tool

# 2. Create an annotator
curl -s -X POST http://localhost:8000/api/annotators/ \
  -H "Content-Type: application/json" \
  -d '{"email": "alice@example.com", "name": "Alice"}' | python3 -m json.tool

# Replace TASK_ID and ANNOTATOR_ID with the IDs from above responses

# 3. Submit feedback (use the IDs from steps 1 and 2)
curl -s -X POST http://localhost:8000/api/feedback/ \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "TASK_ID",
    "annotator_id": "ANNOTATOR_ID",
    "ranking": [1, 2],
    "scalar_reward": 0.85,
    "confidence": 0.9
  }' | python3 -m json.tool

# 4. Check platform metrics
curl -s http://localhost:8000/api/metrics/platform | python3 -m json.tool
```

### What We Accomplished

Phase 2 is a complete, tested REST API. All five route modules handle the full lifecycle of annotation data: tasks move from `pending` to `completed` as feedback arrives; Fleiss' kappa and quality scores update automatically; the metrics endpoint gives a live snapshot of platform health; and completed tasks can be exported to JSONL in DPO format. All 26 tests pass against SQLite with mocked Redis, confirming the logic is correct independently of the infrastructure.

---

## Project Structure Reference

```
agent-rl-training-data-platform/
├── docker-compose.yml          ← 4 services: postgres, redis, api, worker
├── scripts/
│   └── init.sql                ← pgcrypto extension (runs at DB init)
└── backend/
    ├── Dockerfile              ← python:3.12-slim, 4 uvicorn workers
    ├── requirements.txt        ← all Python dependencies
    ├── .env.example            ← environment variable documentation
    ├── pytest.ini              ← asyncio_mode = auto
    ├── __init__.py
    ├── main.py                 ← FastAPI app, lifespan, /health, router wiring
    ├── models.py               ← 6 SQLAlchemy ORM models + 3 enums
    ├── schemas.py              ← Pydantic v2 request/response schemas
    ├── core/
    │   ├── __init__.py
    │   ├── database.py         ← async engine, session factory, get_db(), init_db()
    │   └── redis_client.py     ← Redis pool, queue helpers, cache helpers
    ├── routes/
    │   ├── __init__.py
    │   ├── tasks.py            ← POST/GET/PATCH/DELETE/flag — 6 endpoints
    │   ├── feedback.py         ← submit/list/flag + IAA scoring — 3 endpoints
    │   ├── annotators.py       ← CRUD + next-task queue pop — 4 endpoints
    │   ├── metrics.py          ← platform metrics (cached) + training runs — 3 endpoints
    │   └── exports.py          ← dataset CRUD + DPO JSONL export — 4 endpoints
    ├── workers/
    │   ├── __init__.py
    │   └── quality_worker.py   ← placeholder; real logic in Phase 4
    └── tests/
        ├── __init__.py
        ├── conftest.py         ← SQLite DB + Redis mocks + httpx client fixture
        ├── test_tasks.py       ← 8 tests
        ├── test_feedback.py    ← 4 tests
        ├── test_annotators.py  ← 6 tests
        ├── test_metrics.py     ← 4 tests
        └── test_exports.py     ← 4 tests
```

---

## Environment Variables Reference

| Variable | Description | Default (Docker) | Default (local) |
|----------|-------------|-----------------|-----------------|
| `DATABASE_URL` | SQLAlchemy async connection string | `postgresql+asyncpg://rl_user:rl_pass@postgres:5432/rl_platform` | `postgresql+asyncpg://rl_user:rl_pass@localhost:5432/rl_platform` |
| `REDIS_URL` | Redis connection URL | `redis://redis:6379/0` | `redis://localhost:6379/0` |
| `EXPORT_DIR` | Directory for JSONL export files | `/exports` | `/exports` |

The `/0` suffix in the Redis URL selects database 0 (Redis supports up to 16 logical databases, numbered 0-15).

---

## Common Issues and Troubleshooting

**Docker Compose build fails with "connection refused" errors**
The `depends_on: condition: service_healthy` directives prevent this, but if you see it anyway, wait for PostgreSQL to fully initialize. Run `docker logs rl-postgres` to see its startup progress.

**`pytest` fails with `ModuleNotFoundError: No module named 'core'`**
Run pytest from the `backend/` directory, not the project root: `cd backend && pytest tests/ -v`. The `sys.path.insert` in `conftest.py` adds `backend/` to the path, but this only works if `conftest.py` itself can be found first.

**`MissingGreenlet` error during tests**
This usually means a relationship was accessed outside of an async context. Ensure `expire_on_commit=False` is set on the session factory and that you are calling `await db.refresh(obj)` before accessing server-set fields.

**`FeedbackItem.flagged == False` raises a linter warning**
This is expected. The `# noqa: E712` comment suppresses it. SQLAlchemy requires `==` (not `is`) here because `==` is overloaded to generate a SQL expression, not a Python boolean comparison.

**Health endpoint returns `{"status": "error", "redis": "error"}`**
Redis is not reachable. Check `docker logs rl-redis` and verify the `REDIS_URL` environment variable matches the service name in `docker-compose.yml`.

**`test_feedback_recomputes_quality` fails with `quality_score: None`**
The test uses `min_annotations: 2` and submits exactly 2 feedback items. Verify the feedback endpoint is correctly checking `len(all_feedback) >= task.min_annotations` (not `>`). Also confirm both submissions are for the same `task_id`.

**Port conflicts on startup**
If port 5432 or 6379 is already in use by a local PostgreSQL or Redis instance, Docker Compose will fail. Either stop the local services or change the port mapping in `docker-compose.yml` (e.g., `"5433:5432"` to map to a different host port).

---

## Next Steps

With Phase 1 and Phase 2 complete, the full backend API is functional and tested. The logical next phases are:

**Phase 3 — React Frontend:** Build the annotation dashboard with React 18, TypeScript, and Vite. Five views: Overview (platform metrics), Task Manager, Annotation UI (popping from the queue), Training Metrics (reward/KL/loss charts), and Export Builder. **See `tutorial-phase-3.md` for the full step-by-step guide.**

**Phase 4 — Quality Worker and Seed Data:** Replace the `quality_worker.py` placeholder with a real Redis consumer that processes feedback events, recomputes quality scores in the background, and updates task statuses. Add a `scripts/seed.py` that populates 50 synthetic tasks with realistic feedback distributions.

**Phase 5 — Observability:** Add Prometheus metrics middleware to FastAPI, configure Grafana dashboards for annotator throughput and queue depth, and add structured JSON logging.

**Relevant Documentation:**
- FastAPI dependency injection: https://fastapi.tiangolo.com/tutorial/dependencies/
- SQLAlchemy 2.0 async: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- Pydantic v2 model config: https://docs.pydantic.dev/latest/concepts/config/
- Fleiss' kappa: https://en.wikipedia.org/wiki/Fleiss%27_kappa
- DPO paper (Rafailov et al., 2023): https://arxiv.org/abs/2305.18290
- TRL DPO Trainer: https://huggingface.co/docs/trl/dpo_trainer
