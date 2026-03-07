# RL Training Data Platform — SPEC v1.0
> Phase-by-Phase Local Development with Claude Code

---

## Quick Stats

| | | | |
|:---:|:---:|:---:|:---:|
| **5** Phases | **7** Docker Services | **15+** API Routes | **4** Export Formats |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Local Development Setup](#3-local-development-setup)
4. [Development Phases](#4-development-phases)
   - [Phase 1 — Foundation](#phase-1--foundation--docker-db-and-bare-api)
   - [Phase 2 — Backend API](#phase-2--backend-api--tasks-feedback-annotators)
   - [Phase 3 — Frontend](#phase-3--frontend--react-dashboard)
   - [Phase 4 — Worker + Exports](#phase-4--worker--exports--dataset-pipeline)
   - [Phase 5 — Observability](#phase-5--observability--prometheus--grafana--polish)
5. [API Quick Reference](#5-api-quick-reference)
6. [Development Rules](#6-development-rules)
7. [Phase Completion Summary](#7-phase-completion-summary)

---

## 1. Project Overview

This document is the authoritative specification for building the **RL Training Data Platform** — a full-stack system for collecting human preference feedback, managing annotation workflows, and exporting RL-ready training datasets for LLM fine-tuning. It is designed to be built **locally, phase by phase, using Claude Code** as the primary development assistant.

### 1.1 Goals

- Let researchers create coding, reasoning, math, and instruction-following tasks
- Assign tasks to annotators via a Redis-backed priority queue
- Collect pairwise rankings, scalar rewards, and critiques as reward signals
- Compute inter-annotator agreement (Fleiss' κ) and consensus reward automatically
- Export clean, filtered datasets in JSONL, Parquet, or HuggingFace format
- Visualise platform health, annotator throughput, and training run metrics

### 1.2 Why This Exists

This platform solves a real bottleneck in RLHF/RLAIF pipelines: getting clean, high-quality human preference data from annotators into a format that RL trainers can consume directly. It replaces ad-hoc spreadsheet workflows, eliminates noisy reward signals, and gives researchers full observability into annotation quality before fine-tuning starts.

### 1.3 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend API | FastAPI + Python 3.12 | Async REST API, 4 uvicorn workers |
| Database | PostgreSQL 16 | Tasks, feedback, annotators, datasets |
| Cache / Queue | Redis 7 | Annotation queue, metrics cache, export queue |
| ORM | SQLAlchemy 2.0 (async) | Async models, migrations via Alembic |
| Frontend | React + TypeScript + Vite | Dashboard, annotation UI, task creator |
| Charts | Recharts | Training metrics, reward histograms |
| Workers | Python asyncio | Background quality scoring, IAA compute |
| Containers | Docker + Compose | Full local stack, single command startup |
| Observability | Prometheus + Grafana | Metrics dashboards, alerting |

---

## 2. Architecture

### 2.1 Component Diagram

```
React + TypeScript Dashboard  (localhost:3000)
  Overview | Tasks | Annotate | Training | Exports
               |
               | REST + JSON
               v
FastAPI  (localhost:8000)  —  4 uvicorn workers
  /tasks  /feedback  /metrics  /annotators  /exports
      |                  |                      |
  Postgres           Redis :6379           Quality Worker
  :5432          (queues + cache)       (async IAA scoring)
      |
  Prometheus :9090  →  Grafana :3001
```

### 2.2 Data Flow

1. **Task Creation** — Researcher POSTs task → DB persisted → Redis annotation queue
2. **Annotator Assignment** — Worker pops from queue → creates TaskAssignment → serves to annotator
3. **Feedback Submission** — Annotator submits ranking/scalar → DB stored → background quality job enqueued
4. **Quality Scoring** — Worker computes Fleiss' κ, consensus reward, quality_score → updates Task
5. **Dataset Export** — Researcher creates Dataset with filters → background job builds JSONL/Parquet → download ready
6. **Training** — Dataset consumed by TRL/OpenRLHF → training metrics POSTed back → visualised in Grafana

### 2.3 Database Schema

| Table | Key Columns |
|---|---|
| `tasks` | prompt, responses[], annotation_type, min_annotations, quality_score, iaa, consensus_reward |
| `feedback_items` | ranking[], scalar_reward, binary_label, critique_text, criterion_scores, confidence |
| `annotators` | email, role, expertise_tags, reliability_score, avg_agreement_rate |
| `task_assignments` | task_id ↔ annotator_id, assigned_at, completed_at, time_spent_sec |
| `datasets` | filters, task_count, reward_distribution, export_path, export_format |
| `training_runs` | dataset_id, algorithm, config, reward_history[], kl_history[], loss_history[] |

---

## 3. Local Development Setup

### 3.1 Prerequisites

- Docker Desktop 4.x+ (includes Compose v2)
- Node.js 20+ and npm 10+
- Python 3.12+
- Claude Code CLI — installed and authenticated
- Git

### 3.2 Repository Structure

```
rl-platform/
├── backend/
│   ├── main.py               # FastAPI app + lifespan
│   ├── models.py             # SQLAlchemy ORM models
│   ├── schemas.py            # Pydantic request/response
│   ├── core/
│   │   ├── database.py       # Async engine + session
│   │   └── redis_client.py   # Queue helpers
│   ├── routes/
│   │   ├── tasks.py
│   │   ├── feedback.py
│   │   ├── metrics.py
│   │   ├── annotators.py
│   │   └── exports.py
│   ├── workers/
│   │   └── quality_worker.py # Background IAA + scoring
│   ├── tests/                # Phase 2+
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/            # Overview, Tasks, Annotate, Training, Exports
│   │   ├── components/       # Shared UI components
│   │   └── api/client.ts     # Typed API client
│   ├── Dockerfile
│   └── package.json
├── infra/
│   ├── prometheus.yml
│   └── grafana/
├── scripts/
│   ├── seed.py               # Dev data seeding
│   └── init.sql
├── docker-compose.yml
├── docker-compose.dev.yml    # Hot-reload overrides
└── SPEC.md
```

### 3.3 First-Time Setup

```bash
git clone <repo> && cd rl-platform
cp backend/.env.example backend/.env
docker compose up --build
# wait for all health checks (~30s)
python scripts/seed.py          # optional: seed 50 sample tasks
```

Then open:
- `http://localhost:3000` — Frontend dashboard
- `http://localhost:8000/docs` — Swagger UI

### 3.4 Dev Workflow with Claude Code

> **Tip:** Start Claude Code from the project root. Keep `docker compose up` running in a separate terminal. For each phase, use the suggested prompts below as starting points.

```bash
claude   # launch interactive Claude Code session
```

- Begin each session: *"Read the SPEC.md and existing files in `backend/` and `frontend/src/` to understand the current state, then implement Phase N."*
- Use `/test` to run pytest after each phase
- Commit after each phase: `git commit -am 'Phase N complete'`

---

## 4. Development Phases

> **Convention:** Each phase is self-contained and leaves the app in a working, testable state. Never start a new phase until the previous one passes its acceptance criteria.

---

### Phase 1 — Foundation: Docker, DB, and Bare API

**Goal:** Get a running FastAPI server, PostgreSQL, and Redis all wired together in Docker. No business logic yet — just infrastructure.

#### Deliverables

| File / Path | Description | Depends On |
|---|---|---|
| `docker-compose.yml` | postgres, redis, api, worker services with health checks | — |
| `backend/Dockerfile` | Python 3.12-slim, uvicorn entrypoint | — |
| `backend/main.py` | FastAPI app with lifespan (DB init + Redis ping) | DB, Redis |
| `backend/core/database.py` | Async SQLAlchemy engine, AsyncSession, Base | Postgres |
| `backend/core/redis_client.py` | aioredis client, queue helpers (enqueue/dequeue/cache) | Redis |
| `backend/models.py` | All 6 ORM models with indexes and relationships | database.py |
| `scripts/init.sql` | `CREATE EXTENSION pgcrypto;` initial grants | Postgres |
| `GET /health` | Returns `{status: ok, db: ok, redis: ok}` | All |

#### Claude Code Prompt

```
Set up the full Docker Compose stack for the RL Training Data Platform.
Create docker-compose.yml with postgres:16-alpine, redis:7-alpine, and a
FastAPI api service. The FastAPI app (backend/main.py) should use
asynccontextmanager lifespan to create all DB tables on startup and ping
Redis. backend/core/database.py should use asyncpg with SQLAlchemy 2.0
async. backend/core/redis_client.py should expose enqueue_task,
dequeue_task, cache_get, cache_set helpers. backend/models.py should
define Task, FeedbackItem, Annotator, TaskAssignment, Dataset, TrainingRun
with all columns and relationships. Add GET /health that checks both DB and
Redis. Everything should start with: docker compose up --build
```

#### Acceptance Criteria

- [ ] `docker compose up` starts all services with no errors
- [ ] `curl http://localhost:8000/health` returns `{"status":"ok","db":"ok","redis":"ok"}`
- [ ] `docker compose ps` shows all containers healthy
- [ ] SQLAlchemy creates all tables on first startup
- [ ] Redis responds to PING via `docker exec rl-redis redis-cli ping`

---

### Phase 2 — Backend API: Tasks, Feedback, Annotators

**Goal:** Implement all core REST routes with full request/response validation, Redis queue integration, and Pydantic schemas. Backend fully usable via Swagger UI.

#### Deliverables

| File / Path | Description | Depends On |
|---|---|---|
| `backend/schemas.py` | Pydantic v2: TaskCreate, FeedbackSubmit, AnnotatorCreate, PlatformMetrics | Phase 1 |
| `routes/tasks.py` | POST, GET list (paginated+filtered), GET by id, PATCH, DELETE, POST /flag | schemas.py |
| `routes/feedback.py` | POST (IAA + consensus reward compute), GET /task/{id}, POST /flag | tasks.py |
| `routes/annotators.py` | CRUD + GET `/{id}/next-task` (pops Redis queue, creates TaskAssignment) | redis_client |
| `routes/metrics.py` | GET /platform (cached 60s), GET /training/{run_id}, GET /training list | Redis cache |
| `routes/exports.py` | POST /datasets (background export), GET list, GET /{id}/download | Background |
| `backend/tests/` | pytest with httpx AsyncClient + SQLite in-memory | All routes |

#### Quality Scoring Logic

- **Fleiss' κ** — computed from all non-flagged ranking feedback items after each submission
- **Consensus reward** — reliability-weighted average across scalar/binary/ranking types
- **Quality score** — `0.4 × IAA + 0.4 × (1 − reward_stdev) + 0.2 × annotation_coverage`
- **Auto-completion** — `task.status = COMPLETED` when `feedback_count ≥ min_annotations`

#### Claude Code Prompt

```
Implement all FastAPI routes for the RL Training Data Platform.
Create backend/schemas.py with Pydantic v2 schemas.
routes/tasks.py (full CRUD + flag + Redis enqueue on create),
routes/feedback.py (submit with Fleiss kappa IAA and
reliability-weighted consensus reward, recompute quality_score =
0.4*iaa + 0.4*(1-stdev) + 0.2*coverage after each submission),
routes/annotators.py (CRUD + GET /{id}/next-task that pops from Redis
queue and creates TaskAssignment),
routes/metrics.py (GET /platform with Redis 60s cache),
routes/exports.py (POST triggers background task to build JSONL with
chosen/rejected DPO format).
Wire all routes into main.py. Add pytest tests in backend/tests/ using
httpx.AsyncClient.
```

#### Acceptance Criteria

- [ ] `POST /api/tasks/` creates task and enqueues to Redis, returns 201
- [ ] `POST /api/feedback/` with ranking payload recomputes IAA and quality_score on task
- [ ] `GET /api/annotators/{id}/next-task` pops from Redis queue, creates TaskAssignment
- [ ] `GET /api/metrics/platform` returns all KPIs, cached on repeat calls
- [ ] `pytest backend/tests/ -v` passes all tests
- [ ] Swagger UI at `http://localhost:8000/docs` shows all routes with working Try It Out

---

### Phase 3 — Frontend: React Dashboard

**Goal:** Build the full React + TypeScript frontend with all 5 dashboard views connected to the live API. Use Vite for dev server with hot reload.

#### Deliverables

| File / Path | Description | Depends On |
|---|---|---|
| `frontend/package.json` | React 18, TypeScript 5, Vite, Recharts, React Query | — |
| `frontend/src/api/client.ts` | Typed fetch helpers, base URL from env, error handling | Phase 2 |
| `src/pages/Overview.tsx` | KPI cards, feedback velocity AreaChart, reward histogram BarChart | client.ts |
| `src/pages/Tasks.tsx` | Paginated task table, status/type filters, quality/IAA columns | client.ts |
| `src/pages/Annotate.tsx` | Pairwise ranking UI, criterion sliders, confidence, submit | client.ts |
| `src/pages/Training.tsx` | LineCharts for reward/KL/loss, training run table | client.ts |
| `src/pages/Exports.tsx` | Dataset builder with filter sliders, format selector, download | client.ts |
| `frontend/Dockerfile` | Vite build → nginx:alpine serve on port 80 | package.json |

#### API Client Conventions

- `useQuery` (React Query) for all GET endpoints — auto-refetch every 30s on Overview
- `useMutation` for POST/PATCH operations with optimistic updates
- All API errors shown as inline banners, not console.log
- Loading skeletons on all data-fetching components
- TypeScript strict mode — no implicit any

#### Claude Code Prompt

```
Build the React + TypeScript frontend for the RL Training Data Platform.
Use Vite, React 18, TypeScript 5, and Recharts. Create
frontend/src/api/client.ts with typed fetch helpers for all backend
endpoints (VITE_API_URL from env). Build 5 pages:
- Overview.tsx: KPI cards + AreaChart feedback velocity + BarChart reward histogram
- Tasks.tsx: filterable paginated table with status/type/quality/IAA
- Annotate.tsx: pairwise ranking UI with criterion scores and confidence slider
- Training.tsx: LineChart for reward/KL/loss with run selector
- Exports.tsx: dataset builder form with filter sliders + download button
Add nav sidebar and CreateTaskModal. Wire to FastAPI backend at
localhost:8000. Dark theme: #0a0c0f background, #f59e0b amber accent,
IBM Plex Mono for data values.
```

#### Acceptance Criteria

- [ ] All 5 pages render with real data from the backend API
- [ ] Overview page shows live feedback count and updates on refresh
- [ ] Creating a task from the modal adds it to the task table and Redis queue
- [ ] Annotate page submits feedback and shows success confirmation
- [ ] `docker compose up` serves frontend at `http://localhost:3000`
- [ ] `npm run build` succeeds with no TypeScript errors

---

### Phase 4 — Worker + Exports + Dataset Pipeline

**Goal:** Implement the quality scoring background worker and full dataset export pipeline. Exports must be TRL/OpenRLHF-compatible and downloadable from the UI.

#### Deliverables

| File / Path | Description | Depends On |
|---|---|---|
| `workers/quality_worker.py` | Async Redis consumer: dequeues QUALITY_QUEUE, rescores task quality in DB | Phase 1 |
| `routes/exports.py` | `_task_to_rl_example()`: task+feedback → DPO chosen/rejected schema | Phase 2 |
| `routes/exports.py` | `_build_export()`: JSONL, Parquet (pandas), HF Dataset background export | pandas |
| `scripts/seed.py` | Seeds 50 tasks with synthetic feedback for testing export pipeline | Phase 2 |
| `GET .../download` | FileResponse serving completed export from EXPORT_DIR volume | Phase 3 |
| `Exports.tsx` | Poll export status, show progress, enable download when ready | Phase 3 |

#### Export Schema (DPO format)

```json
{
  "id": "uuid",
  "prompt": "The task prompt text",
  "chosen": "Best-ranked model response",
  "rejected": "Worst-ranked model response",
  "reward_chosen": 0.891,
  "reward_rejected": 0.312,
  "task_type": "coding",
  "quality_score": 0.912,
  "iaa": 0.824,
  "num_annotators": 5,
  "tags": ["python", "concurrency"],
  "evaluation_criteria": ["correctness", "code quality", "efficiency"],
  "all_responses": [
    { "model_id": "m-a", "text": "...", "avg_reward": 0.891 },
    { "model_id": "m-b", "text": "...", "avg_reward": 0.312 }
  ]
}
```

#### Claude Code Prompt

```
Implement the background quality worker and full dataset export pipeline.
In workers/quality_worker.py, create an async Redis consumer listening
on rl:queue:quality that dequeues job payloads and rescores task
quality_score using weighted formula of IAA, reward stdev, and annotation
coverage. In routes/exports.py, implement _task_to_rl_example() that
converts Task + FeedbackItems into a DPO dict with chosen, rejected,
reward_chosen, reward_rejected. Implement _build_export() as a FastAPI
BackgroundTask that writes JSONL (default), Parquet (pandas), or HF
Dataset format to EXPORT_DIR and updates Dataset.export_path when done.
Add seed script at scripts/seed.py creating 50 synthetic tasks with
3–5 feedback items each.
```

#### Acceptance Criteria

- [ ] `python scripts/seed.py` populates 50 tasks with feedback
- [ ] `POST /api/exports/datasets` triggers background JSONL export
- [ ] Polling GET shows `exported_at` timestamp when complete
- [ ] `GET /api/exports/datasets/{id}/download` returns valid JSONL file
- [ ] Each JSONL line has `chosen`, `rejected`, `reward_chosen`, `reward_rejected`
- [ ] Worker process logs quality score updates as tasks receive feedback

---

### Phase 5 — Observability: Prometheus + Grafana + Polish

**Goal:** Wire Prometheus metrics, build a Grafana dashboard, add structured logging, and finalize error handling.

#### Deliverables

| File / Path | Description | Depends On |
|---|---|---|
| `backend/main.py` | Add prometheus-fastapi-instrumentator, `/metrics` endpoint | Phase 2 |
| `infra/prometheus.yml` | Scrape `api:8000/metrics` every 15s | Docker Compose |
| `infra/grafana/datasources/` | Prometheus datasource YAML for auto-provisioning | Grafana |
| `infra/grafana/dashboards/` | Platform dashboard: feedback rate, queue depth, IAA, task status | Prometheus |
| `backend/` (all routes) | structlog structured logging: `request_id`, `route`, `duration_ms` | Phase 2 |
| `backend/` error handling | Global exception handler: `{error, detail, request_id}` JSON | All routes |
| `docker-compose.dev.yml` | Volume mounts for hot-reload on backend and frontend | All phases |
| `README.md` | Full local setup, architecture diagram, API reference table | — |

#### Key Grafana Panels

- Feedback submissions / minute (`rate[5m]`)
- Annotation queue depth over time
- Task status breakdown (stacked bar)
- Average IAA by task type
- P95 API response time by route (`histogram_quantile`)
- Quality score distribution histogram

#### Claude Code Prompt

```
Add production observability to the RL Training Data Platform.
In backend/main.py, integrate prometheus-fastapi-instrumentator to
expose /metrics. Create infra/prometheus.yml scraping api:8000/metrics
every 15s. Create infra/grafana/datasources/prometheus.yml for
auto-provisioning. Create infra/grafana/dashboards/platform.json with
panels for: feedback rate, queue depth, task status breakdown, avg IAA,
API p95 latency. Add structlog structured logging to all routes with
request_id and duration_ms. Add global FastAPI exception handler returning
{error, detail, request_id}. Create docker-compose.dev.yml with volume
mounts for hot reload. Update README.md with full architecture, local
setup, API reference.
```

#### Acceptance Criteria

- [ ] `http://localhost:8000/metrics` returns Prometheus-formatted metrics
- [ ] Prometheus UI at `:9090` shows `rl-platform-api` target as UP
- [ ] Grafana at `:3001` shows Platform dashboard with live data
- [ ] All API errors return JSON with `error`, `detail`, `request_id`
- [ ] Structured log lines include `route`, `method`, `duration_ms`, `status_code`
- [ ] `docker compose -f docker-compose.yml -f docker-compose.dev.yml up` enables hot reload

---

## 5. API Quick Reference

### 5.1 Core Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/tasks/` | Create task + enqueue for annotation |
| `GET` | `/api/tasks/` | List tasks (filter: status, type, tag, page) |
| `GET` | `/api/tasks/{id}` | Get full task with responses and criteria |
| `PATCH` | `/api/tasks/{id}` | Update task metadata or status |
| `POST` | `/api/tasks/{id}/flag` | Flag task for review |
| `POST` | `/api/feedback/` | Submit ranking/scalar/binary/critique |
| `GET` | `/api/feedback/task/{id}` | All feedback for a task |
| `POST` | `/api/feedback/{id}/flag` | Flag suspicious feedback item |
| `POST` | `/api/annotators/` | Register new annotator |
| `GET` | `/api/annotators/{id}/next-task` | Pop next queued task assignment |
| `GET` | `/api/metrics/platform` | Platform KPIs (cached 60s) |
| `GET` | `/api/metrics/training/{id}` | Training run time-series |
| `POST` | `/api/exports/datasets` | Build filtered dataset (async) |
| `GET` | `/api/exports/datasets` | List all datasets |
| `GET` | `/api/exports/datasets/{id}/download` | Download JSONL/Parquet export |

### 5.2 Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://rl_user:rl_pass@postgres:5432/rl_platform` | Override for remote DB |
| `REDIS_URL` | `redis://redis:6379/0` | Override for ElastiCache |
| `EXPORT_DIR` | `/exports` | Docker volume mount path |
| `VITE_API_URL` | `http://localhost:8000` | Frontend API base URL |
| `GF_SECURITY_ADMIN_PASSWORD` | `admin` | **Change in production!** |

---

## 6. Development Rules

### 6.1 General

1. Complete and test each phase before moving to the next. Never skip ahead.
2. Commit after each phase: `git commit -am 'Phase N complete: <summary>'`
3. All database schema changes must have an Alembic migration (Phase 2+).
4. All new endpoints must have at least one pytest test.
5. No hardcoded secrets — use environment variables only.

### 6.2 Backend

- Use `async/await` throughout — no synchronous DB calls in async endpoints
- Use `Depends(get_db)` for all DB sessions — never create sessions manually in routes
- All UUIDs stored as `String` (not native UUID) for SQLite test compatibility
- Background tasks via FastAPI's `BackgroundTasks` for Phase 4; move to worker queue if >1s
- `from __future__ import annotations` at top of all files using PEP 604 unions

### 6.3 Frontend

- TypeScript strict mode — no implicit `any`
- API calls only through `api/client.ts` — never `fetch()` inline in components
- Use React Query for all server state — no `useState` for API data
- All score/reward values displayed with consistent precision (e.g. `0.83`, not `0.829999`)

### 6.4 Claude Code Session Tips

> **Before starting a phase:** Always begin with: *"Read SPEC.md and the existing files in `backend/` and `frontend/src/` to understand the current state. Then implement Phase N as described."* This ensures Claude Code doesn't re-implement already-working code.

- Paste the relevant phase section of this SPEC into the prompt for context
- After each file is created, ask Claude Code to run tests before proceeding
- Use `/diff` to review changes before accepting
- If a phase produces errors, fix them in the same session before committing

---

## 7. Phase Completion Summary

| # | Phase | Key Outcome | Est. Effort | Status |
|:---:|---|---|:---:|:---:|
| 1 | Foundation | Docker, DB, Redis, `/health` live | 2–3 hrs | ☐ |
| 2 | Backend API | All routes, IAA scoring, Swagger working | 4–6 hrs | ☐ |
| 3 | Frontend | React dashboard, 5 views, live API | 4–6 hrs | ☐ |
| 4 | Worker + Exports | Quality worker, JSONL/Parquet, seed data | 3–4 hrs | ☐ |
| 5 | Observability | Grafana, Prometheus, structured logging | 2–3 hrs | ☐ |

**Total estimated: ~15–25 hours across 5 Claude Code sessions**

---

> **After all phases complete:** Run `docker compose up && python scripts/seed.py`, then open `localhost:3000` for the dashboard, `localhost:3001` for Grafana, and `localhost:8000/docs` for the full API. Export a JSONL dataset and validate it loads in HuggingFace `datasets` or TRL `DPOTrainer`.
