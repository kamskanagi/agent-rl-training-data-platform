# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL Training Data Platform — a full-stack system for collecting human preference feedback, managing annotation workflows, and exporting RL-ready training datasets (DPO format) for LLM fine-tuning. Built phase-by-phase per SPEC.md.

## Tech Stack

- **Backend:** FastAPI + Python 3.12, SQLAlchemy 2.0 (async with asyncpg), Alembic migrations
- **Database:** PostgreSQL 16, Redis 7 (queues + cache)
- **Frontend:** React 18 + TypeScript 5 + Vite, Recharts, React Query
- **Infrastructure:** Docker Compose, Prometheus, Grafana
- **Workers:** Python asyncio background quality scoring

## Commands

```bash
# Start full stack
docker compose up --build

# Start with hot-reload (after Phase 5)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run backend tests
pytest backend/tests/ -v

# Run single test
pytest backend/tests/test_tasks.py -v -k "test_name"

# Seed dev data (50 tasks with synthetic feedback)
python scripts/seed.py

# Frontend build check
cd frontend && npm run build
```

## Architecture

```
React Dashboard (:3000) → FastAPI (:8000, 4 uvicorn workers) → PostgreSQL (:5432) + Redis (:6379)
                                                               → Quality Worker (async Redis consumer)
Prometheus (:9090) → Grafana (:3001)
```

**Data flow:** Task created → enqueued to Redis → annotator pops from queue → submits feedback → background worker computes Fleiss' κ + consensus reward + quality_score → dataset export (JSONL/Parquet/HF) → consumed by TRL/OpenRLHF.

**Quality score formula:** `0.4 × IAA + 0.4 × (1 − reward_stdev) + 0.2 × annotation_coverage`

## Key Conventions

### Backend
- `async/await` throughout — no synchronous DB calls in async endpoints
- `Depends(get_db)` for all DB sessions — never create sessions manually in routes
- UUIDs stored as `String` (not native UUID) for SQLite test compatibility
- `from __future__ import annotations` at top of all files using PEP 604 unions
- Background tasks via FastAPI's `BackgroundTasks`; move to worker queue if >1s

### Frontend
- TypeScript strict mode — no implicit `any`
- API calls only through `api/client.ts` — never inline `fetch()` in components
- React Query for all server state — no `useState` for API data
- Scores/rewards displayed with consistent precision (e.g., `0.83` not `0.829999`)
- Dark theme: `#0a0c0f` background, `#f59e0b` amber accent, IBM Plex Mono for data values

## Development Phases

The project follows 5 sequential phases defined in SPEC.md. Each phase must be complete and tested before starting the next. Always read SPEC.md and existing code before implementing a phase.

1. **Foundation** — Docker, DB, Redis, `/health` endpoint
2. **Backend API** — All routes, IAA scoring, Swagger
3. **Frontend** — React dashboard, 5 views
4. **Worker + Exports** — Quality worker, JSONL/Parquet pipeline, seed data
5. **Observability** — Prometheus, Grafana, structured logging

## Environment Variables

| Variable | Default |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://rl_user:rl_pass@postgres:5432/rl_platform` |
| `REDIS_URL` | `redis://redis:6379/0` |
| `EXPORT_DIR` | `/exports` |
| `VITE_API_URL` | `http://localhost:8000` |

## DB Schema (6 tables)

`tasks`, `feedback_items`, `annotators`, `task_assignments`, `datasets`, `training_runs` — see SPEC.md §2.3 for column details.
