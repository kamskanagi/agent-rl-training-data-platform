# RL Training Data Platform

A production-ready platform for collecting human preference feedback and exporting RL-ready training data. Built for teams fine-tuning LLMs with RLHF, DPO, and related alignment techniques.

## Motivation

The quality of an RLHF-tuned model is only as good as the human preference data it learns from. Commercial annotation platforms exist, but they are opaque, expensive, and hard to customize for research workflows. Open-source alternatives tend to focus on the annotation UI alone, leaving teams to cobble together their own pipelines for quality control, agreement metrics, and export formatting.

This project exists to close that gap. It provides the full pipeline — from task creation to annotator assignment, real-time quality scoring, and export in formats that plug directly into training frameworks — as a single, self-hosted stack that researchers and ML engineers can run, inspect, and extend.

It was also built as a hands-on learning project. Rather than reading about how RLHF data pipelines work in theory, you build one end-to-end: designing the data model, implementing inter-annotator agreement algorithms, wiring up background workers, and connecting it all to a dashboard with observability built in.

## What You Will Learn

Building this project takes you through a broad set of real-world engineering skills:

- **Async Python and FastAPI** — structuring a production API with async SQLAlchemy, Pydantic v2 validation, and dependency injection
- **Data modeling for ML pipelines** — designing schemas that capture pairwise rankings, scalar rewards, binary labels, and free-text critiques in a single flexible model
- **Inter-annotator agreement** — implementing Fleiss' kappa from scratch and understanding why agreement metrics matter for training data quality
- **Background processing** — building a Redis-backed worker that listens on Pub/Sub and a fallback queue, recomputes quality scores, and manages task lifecycle
- **Export pipeline design** — converting raw annotations into DPO-ready JSONL, Parquet, and HuggingFace Datasets format with reward distribution statistics
- **React dashboards** — building a multi-page TypeScript frontend with React Query, Recharts visualizations, and a dark-themed design system
- **Observability from day one** — instrumenting an API with Prometheus custom metrics, provisioning Grafana dashboards, and adding structured JSON logging with structlog
- **Docker Compose orchestration** — running a multi-service stack (API, worker, PostgreSQL, Redis, frontend, Prometheus, Grafana) with health checks and dependency ordering
- **Testing async APIs** — writing pytest-asyncio tests with SQLite, mocked Redis, and HTTPX's async test client

## What It Does

- **Annotation management** — Create tasks with multiple model responses, assign them to annotators, and collect pairwise rankings, scalar rewards, binary labels, or free-text critiques
- **Quality scoring** — Automatically computes inter-annotator agreement (Fleiss' kappa), consensus rewards, and composite quality scores on every feedback submission
- **Background processing** — Redis-backed quality worker recomputes scores in real time via Pub/Sub, auto-completes tasks, and flags low-quality annotations
- **Multi-format export** — Export preference data as JSONL (DPO format), Parquet, or HuggingFace Datasets format ready for `datasets.load_dataset()`
- **Dashboard** — React frontend with 5 pages: Overview (KPI cards + charts), Tasks (filterable table), Annotate (pairwise ranking UI), Training (metric charts), Exports (dataset builder)
- **Observability** — Prometheus metrics, Grafana dashboards, and structured JSON logging with structlog

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12, FastAPI, SQLAlchemy 2.0 (async), Pydantic v2 |
| Database | PostgreSQL 16, asyncpg |
| Cache/Queue | Redis 7 |
| Frontend | React 18, TypeScript 5, Vite, React Router v6, Recharts |
| Monitoring | Prometheus, Grafana, structlog |
| Infrastructure | Docker Compose, nginx |

## Quick Start

```bash
# Clone and start all services
git clone https://github.com/kamskanagi/agent-rl-training-data-platform.git
cd agent-rl-training-data-platform
docker compose up --build
```

Once running:

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin/admin) |
| Health Check | http://localhost:8000/health |
| Metrics | http://localhost:8000/metrics |

## Seed Data

Populate the database with 50 synthetic tasks, 8 annotators, and realistic feedback:

```bash
DATABASE_URL=postgresql+asyncpg://rl_user:rl_pass@localhost:5432/rl_platform \
  python scripts/seed.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (DB + Redis) |
| `GET` | `/metrics` | Prometheus metrics |
| **Tasks** | | |
| `POST` | `/api/tasks/` | Create a task |
| `GET` | `/api/tasks/` | List tasks (filterable by status, type, tag) |
| `GET` | `/api/tasks/{id}` | Get task details |
| `PATCH` | `/api/tasks/{id}` | Update a task |
| `DELETE` | `/api/tasks/{id}` | Delete a task |
| `POST` | `/api/tasks/{id}/flag` | Flag a task |
| **Feedback** | | |
| `POST` | `/api/feedback/` | Submit feedback |
| `GET` | `/api/feedback/task/{id}` | Get feedback for a task |
| `POST` | `/api/feedback/{id}/flag` | Flag feedback |
| **Annotators** | | |
| `POST` | `/api/annotators/` | Register an annotator |
| `GET` | `/api/annotators/` | List annotators |
| `GET` | `/api/annotators/{id}` | Get annotator details |
| `GET` | `/api/annotators/{id}/next` | Get next task from queue |
| **Metrics** | | |
| `GET` | `/api/metrics/` | Platform-wide metrics |
| `GET` | `/api/metrics/training-runs` | List training runs |
| `GET` | `/api/metrics/training-runs/{id}` | Get training run details |
| **Exports** | | |
| `POST` | `/api/exports/datasets` | Create a dataset export |
| `GET` | `/api/exports/datasets` | List datasets |
| `GET` | `/api/exports/datasets/{id}` | Get dataset details |
| `GET` | `/api/exports/datasets/{id}/download` | Download exported data |

## Project Structure

```
agent-rl-training-data-platform/
├── backend/
│   ├── core/
│   │   ├── database.py          # Async SQLAlchemy engine and session
│   │   ├── redis_client.py      # Redis client, queue, and cache helpers
│   │   ├── logging.py           # Structured logging with structlog
│   │   └── metrics.py           # Prometheus custom metrics
│   ├── routes/
│   │   ├── tasks.py             # Task CRUD endpoints
│   │   ├── feedback.py          # Feedback submission + IAA scoring
│   │   ├── annotators.py        # Annotator management + queue
│   │   ├── metrics.py           # Platform metrics + training runs
│   │   └── exports.py           # Dataset export (JSONL, Parquet, HF)
│   ├── workers/
│   │   └── quality_worker.py    # Redis consumer for quality scoring
│   ├── tests/                   # pytest-asyncio test suite
│   ├── models.py                # SQLAlchemy ORM models
│   ├── schemas.py               # Pydantic request/response schemas
│   └── main.py                  # FastAPI app entrypoint
├── frontend/
│   └── src/
│       ├── pages/               # Overview, Tasks, Annotate, Training, Exports
│       ├── components/          # CreateTaskModal
│       └── api/client.ts        # Typed API client
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/
│       ├── provisioning/        # Datasource and dashboard configs
│       └── dashboards/          # Pre-built dashboard JSON
├── scripts/
│   ├── init.sql                 # PostgreSQL init script
│   └── seed.py                  # Synthetic data seeder
├── docs/tutorial/               # Step-by-step build tutorials
└── docker-compose.yml           # Full stack orchestration
```

## Running Tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

Tests use SQLite (via aiosqlite) and mock Redis, so no external services are needed.

## Export Formats

| Format | Description |
|--------|-------------|
| **JSONL** | DPO-ready format with `prompt`, `chosen`, `rejected`, reward scores, and metadata |
| **Parquet** | Columnar format via PyArrow for efficient large-scale processing |
| **HuggingFace** | `train.jsonl` + `dataset_info.json`, compatible with `datasets.load_dataset()` |

## Tutorials

Step-by-step build tutorials are available in `docs/tutorial/`:

- [Phases 1-2](docs/tutorial/tutorial-phases-1-2.md) — Foundation + Backend API
- [Phase 3](docs/tutorial/tutorial-phase-3.md) — Frontend React Dashboard
- [Phases 4-5](docs/tutorial/tutorial-phases-4-5.md) — Worker, Exports, Seed Data, Observability

## License

MIT
