# Building an RL Training Data Platform: Phase 3

> Build the full React + TypeScript dashboard for the RL Training Data Platform — five pages wired to the live FastAPI backend, with real-time data, dark theme, and Recharts visualizations.

**What you'll build:** A React 18 + TypeScript 5 frontend with five dashboard pages (Overview, Tasks, Annotate, Training, Exports), a typed API client, a dark theme with amber accents and IBM Plex Mono data values, a navigation sidebar, and a task creation modal — all connected to the FastAPI backend from Phase 2 and containerized with Docker.

**Tech stack:** React 18, TypeScript 5, Vite, React Query (TanStack Query v5), React Router v6, Recharts, nginx (Docker production serve)

**Prerequisites:** Phase 1 and Phase 2 complete, Node.js 20+, npm, the backend API running (or Docker Compose)

**Time estimate:** 2-4 hours

**Difficulty:** Intermediate

---

## Table of Contents

1. [Phase 3: Frontend — React Dashboard](#phase-3-frontend--react-dashboard)
   - [Step 3.1: Backend CORS Middleware](#step-31-backend-cors-middleware)
   - [Step 3.2: Frontend Project Scaffolding](#step-32-frontend-project-scaffolding)
   - [Step 3.3: Vite and TypeScript Configuration](#step-33-vite-and-typescript-configuration)
   - [Step 3.4: HTML Entry Point and Fonts](#step-34-html-entry-point-and-fonts)
   - [Step 3.5: API Client — Typed Fetch Helpers](#step-35-api-client--typed-fetch-helpers)
   - [Step 3.6: Global Styles and Dark Theme](#step-36-global-styles-and-dark-theme)
   - [Step 3.7: Application Shell — main.tsx and App.tsx](#step-37-application-shell--maintsx-and-apptsx)
   - [Step 3.8: Overview Page — KPI Cards and Charts](#step-38-overview-page--kpi-cards-and-charts)
   - [Step 3.9: Tasks Page — Filterable Paginated Table](#step-39-tasks-page--filterable-paginated-table)
   - [Step 3.10: Annotate Page — Pairwise Ranking UI](#step-310-annotate-page--pairwise-ranking-ui)
   - [Step 3.11: Training Page — Metric Line Charts](#step-311-training-page--metric-line-charts)
   - [Step 3.12: Exports Page — Dataset Builder](#step-312-exports-page--dataset-builder)
   - [Step 3.13: CreateTaskModal Component](#step-313-createtaskmodal-component)
   - [Step 3.14: Frontend Dockerfile and nginx](#step-314-frontend-dockerfile-and-nginx)
   - [Step 3.15: Docker Compose — Add Frontend Service](#step-315-docker-compose--add-frontend-service)
   - [Verify Phase 3](#verify-phase-3)
2. [Updated Project Structure](#updated-project-structure)
3. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
4. [Next Steps](#next-steps)

---

## Phase 3: Frontend — React Dashboard

### What We're Building

Phase 3 adds the entire user-facing frontend. By the end, `docker compose up --build` will serve a React dashboard at `http://localhost:3000` that talks to the FastAPI backend at `http://localhost:8000`. Every page displays real data, mutations invalidate the React Query cache automatically, and the dark theme uses the exact design tokens specified in the project spec.

### Prerequisites

Phases 1 and 2 complete — all backend routes functional, Docker services running.

---

### Step 3.1: Backend CORS Middleware

**Why:** The frontend runs on port 3000 (Vite dev) or port 80 (Docker nginx), while the backend runs on port 8000. Browsers block cross-origin requests by default. Adding CORS middleware to FastAPI tells the browser "yes, requests from these origins are allowed." Without this, every `fetch()` call from the dashboard will fail silently with a CORS error in the browser console.

**Update `backend/main.py`** — add the CORS import and middleware after the `app` is created:

```python
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
> - `CORSMiddleware` intercepts preflight `OPTIONS` requests and adds `Access-Control-Allow-Origin` headers to responses.
> - We allow two origins: `http://localhost:3000` (Docker nginx) and `http://localhost:5173` (Vite's default dev server port). Adding both means the dashboard works whether you're running in Docker or doing local `npm run dev`.
> - `allow_credentials=True` enables cookies and authorization headers in cross-origin requests — needed if you later add authentication.
> - `allow_methods=["*"]` and `allow_headers=["*"]` are permissive for development. In production, you'd restrict these to only the methods and headers your frontend actually uses.

---

### Step 3.2: Frontend Project Scaffolding

**Why:** We create the project structure and `package.json` manually rather than using `create-react-app` or `create-vite` because we need precise control over dependencies and configuration. This also avoids hundreds of boilerplate files we'd immediately delete.

Create the directory structure:

```bash
mkdir -p frontend/src/{api,pages,components} frontend/public
```

**Create `frontend/package.json`:**

```json
{
  "name": "rl-training-dashboard",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@tanstack/react-query": "^5.59.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.0",
    "recharts": "^2.12.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.5",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "typescript": "^5.5.4",
    "vite": "^5.4.0"
  }
}
```

> **What's happening here:**
> - `"type": "module"` tells Node to treat `.js` files as ESM — required by Vite.
> - `"build": "tsc -b && vite build"` runs the TypeScript compiler first (to catch type errors) before Vite bundles the production output. If `tsc` finds errors, the build fails early.
> - **React Query** (`@tanstack/react-query` v5) handles all server state — fetching, caching, background refetching, and cache invalidation after mutations. This means we never use `useState` + `useEffect` for API data.
> - **Recharts** is the charting library — it renders SVG-based charts as React components, which means charts rerender reactively when data changes.
> - **React Router** v6 handles client-side navigation between the five pages without full-page reloads.

Install dependencies:

```bash
cd frontend && npm install
```

---

### Step 3.3: Vite and TypeScript Configuration

**Why:** Vite needs minimal config — just the React plugin and a dev server port. TypeScript strict mode catches entire categories of bugs at compile time (null safety, implicit any, unused variables) that would otherwise only surface at runtime.

**Create `frontend/vite.config.ts`:**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
  },
})
```

**Create `frontend/tsconfig.json`:**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"]
}
```

**Create `frontend/src/vite-env.d.ts`:**

```typescript
/// <reference types="vite/client" />
```

> **What's happening here:**
> - `server.port: 3000` matches the port we'll expose in Docker Compose.
> - `server.host: true` binds to `0.0.0.0` so the dev server is reachable from Docker containers and other machines on the network.
> - `"strict": true` enables all TypeScript strict checks: `strictNullChecks`, `noImplicitAny`, `strictFunctionTypes`, etc.
> - `"noUnusedLocals"` and `"noUnusedParameters"` prevent dead code from accumulating — if you declare a variable or parameter but never use it, the build fails.
> - `"moduleResolution": "bundler"` uses Vite's resolution strategy, which supports `import.meta.env` and `.tsx` extensions without explicit paths.
> - `vite-env.d.ts` provides type definitions for Vite-specific features like `import.meta.env.VITE_API_URL`.

---

### Step 3.4: HTML Entry Point and Fonts

**Why:** The HTML file is the single entry point — Vite injects the bundled JS and CSS at build time. Loading fonts from Google Fonts with `preconnect` reduces latency by establishing the connection early.

**Create `frontend/index.html`:**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>RL Training Data Platform</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

> **What's happening here:**
> - Two fonts are loaded: **IBM Plex Mono** for data values (scores, IDs, metrics) and **Inter** for body text. The spec requires IBM Plex Mono specifically because monospaced numbers align vertically in tables and KPI cards, making comparisons easier to scan.
> - `<link rel="preconnect">` opens a TCP + TLS connection to Google Fonts before the browser discovers the font CSS — this shaves ~100ms off font loading.
> - `<script type="module" src="/src/main.tsx">` — Vite transforms this at dev time (serving `.tsx` directly via ESBuild) and at build time (replacing it with the bundled output).

---

### Step 3.5: API Client — Typed Fetch Helpers

**Why:** Centralizing all API calls in a single file prevents fetch logic from scattering across components. TypeScript interfaces mirror the backend Pydantic schemas, so the compiler catches mismatches between what the frontend sends and what the backend expects. Every component imports from `api/client.ts` — no inline `fetch()` calls anywhere.

**Create `frontend/src/api/client.ts`:**

```typescript
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(res.status, body);
  }
  return res.json();
}

// ── Types ─────────────────────────────────────────────────────────────

export interface Task {
  id: string;
  prompt: string;
  responses: { model_id: string; text: string }[] | null;
  annotation_type: string;
  status: string;
  min_annotations: number;
  quality_score: number | null;
  iaa: number | null;
  consensus_reward: number | null;
  tags: string[] | null;
  evaluation_criteria: string[] | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface TaskListResponse {
  tasks: Task[];
  total: number;
  page: number;
  page_size: number;
}

export interface TaskCreate {
  prompt: string;
  responses?: { model_id: string; text: string }[];
  annotation_type?: string;
  min_annotations?: number;
  tags?: string[];
  evaluation_criteria?: string[];
}

export interface FeedbackItem {
  id: string;
  task_id: string;
  annotator_id: string;
  ranking: number[] | null;
  scalar_reward: number | null;
  binary_label: boolean | null;
  critique_text: string | null;
  criterion_scores: Record<string, number> | null;
  confidence: number | null;
  flagged: boolean;
  created_at: string | null;
}

export interface FeedbackSubmit {
  task_id: string;
  annotator_id: string;
  ranking?: number[];
  scalar_reward?: number;
  binary_label?: boolean;
  critique_text?: string;
  criterion_scores?: Record<string, number>;
  confidence?: number;
}

export interface Annotator {
  id: string;
  email: string;
  name: string;
  role: string;
  expertise_tags: string[] | null;
  reliability_score: number;
  avg_agreement_rate: number | null;
  created_at: string | null;
}

export interface PlatformMetrics {
  total_tasks: number;
  pending_tasks: number;
  completed_tasks: number;
  total_feedback: number;
  total_annotators: number;
  avg_quality_score: number | null;
  avg_iaa: number | null;
  queue_depth: number;
}

export interface Dataset {
  id: string;
  name: string;
  filters: Record<string, unknown> | null;
  task_count: number;
  reward_distribution: Record<string, unknown> | null;
  export_path: string | null;
  export_format: string;
  exported_at: string | null;
  created_at: string | null;
}

export interface DatasetCreate {
  name: string;
  filters?: Record<string, unknown>;
  export_format?: string;
}

export interface TrainingRun {
  id: string;
  dataset_id: string;
  algorithm: string;
  config: Record<string, unknown> | null;
  reward_history: number[] | null;
  kl_history: number[] | null;
  loss_history: number[] | null;
  status: string;
  created_at: string | null;
}

// ── API Functions ────────────────────────────────────────────────────

export const api = {
  // Tasks
  getTasks: (params?: { page?: number; page_size?: number; status?: string; annotation_type?: string }) => {
    const qs = new URLSearchParams();
    if (params?.page) qs.set('page', String(params.page));
    if (params?.page_size) qs.set('page_size', String(params.page_size));
    if (params?.status) qs.set('status', params.status);
    if (params?.annotation_type) qs.set('annotation_type', params.annotation_type);
    return request<TaskListResponse>(`/api/tasks/?${qs}`);
  },
  getTask: (id: string) => request<Task>(`/api/tasks/${id}`),
  createTask: (data: TaskCreate) =>
    request<Task>('/api/tasks/', { method: 'POST', body: JSON.stringify(data) }),
  updateTask: (id: string, data: Partial<TaskCreate>) =>
    request<Task>(`/api/tasks/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  deleteTask: (id: string) =>
    request<void>(`/api/tasks/${id}`, { method: 'DELETE' }),
  flagTask: (id: string) =>
    request<Task>(`/api/tasks/${id}/flag`, { method: 'POST' }),

  // Feedback
  submitFeedback: (data: FeedbackSubmit) =>
    request<FeedbackItem>('/api/feedback/', { method: 'POST', body: JSON.stringify(data) }),
  getTaskFeedback: (taskId: string) =>
    request<FeedbackItem[]>(`/api/feedback/task/${taskId}`),
  flagFeedback: (id: string) =>
    request<FeedbackItem>(`/api/feedback/${id}/flag`, { method: 'POST' }),

  // Annotators
  getAnnotators: () => request<Annotator[]>('/api/annotators/'),
  createAnnotator: (data: { email: string; name: string; role?: string; expertise_tags?: string[] }) =>
    request<Annotator>('/api/annotators/', { method: 'POST', body: JSON.stringify(data) }),
  getNextTask: (annotatorId: string) =>
    request<Task>(`/api/annotators/${annotatorId}/next-task`),

  // Metrics
  getPlatformMetrics: () => request<PlatformMetrics>('/api/metrics/platform'),
  getTrainingRuns: () => request<TrainingRun[]>('/api/metrics/training'),
  getTrainingRun: (id: string) => request<TrainingRun>(`/api/metrics/training/${id}`),

  // Exports
  createDataset: (data: DatasetCreate) =>
    request<Dataset>('/api/exports/datasets', { method: 'POST', body: JSON.stringify(data) }),
  getDatasets: () => request<Dataset[]>('/api/exports/datasets'),
  getDataset: (id: string) => request<Dataset>(`/api/exports/datasets/${id}`),
  downloadDataset: (id: string) => `${BASE_URL}/api/exports/datasets/${id}/download`,
};
```

> **What's happening here:**
> - `import.meta.env.VITE_API_URL` reads the environment variable at build time. In Docker, the nginx reverse proxy handles routing, but during local development the frontend calls the backend directly on port 8000.
> - `ApiError` extends `Error` with a `status` code, so mutation `onError` handlers can distinguish between 404 (not found) and 500 (server error) without parsing the message string.
> - The generic `request<T>` function handles all HTTP calls: it sets `Content-Type: application/json`, checks `res.ok`, and returns the parsed JSON body typed as `T`. Every API function passes through this single function.
> - TypeScript interfaces mirror the backend Pydantic schemas field-for-field. Nullable fields use `| null` (matching Python's `Optional` / `None`), and optional request fields use `?` (matching Pydantic's default values).
> - `downloadDataset` returns a URL string instead of calling `request()` — the browser navigates to this URL directly to trigger a file download, and `FileResponse` on the backend sends the correct `Content-Disposition` header.

---

### Step 3.6: Global Styles and Dark Theme

**Why:** A single CSS file with CSS custom properties (variables) ensures the dark theme is consistent across all pages. Defining colors, fonts, and component styles once prevents duplication and makes theme changes a single-line edit. The spec mandates `#0a0c0f` background, `#f59e0b` amber accent, and IBM Plex Mono for data values.

**Create `frontend/src/index.css`:**

```css
:root {
  --bg-primary: #0a0c0f;
  --bg-secondary: #12151a;
  --bg-card: #181c23;
  --bg-hover: #1e2330;
  --border: #2a2f3a;
  --text-primary: #e8eaed;
  --text-secondary: #9aa0ab;
  --text-muted: #6b7280;
  --accent: #f59e0b;
  --accent-hover: #d97706;
  --accent-dim: rgba(245, 158, 11, 0.15);
  --success: #22c55e;
  --error: #ef4444;
  --info: #3b82f6;
  --font-mono: 'IBM Plex Mono', monospace;
  --font-sans: 'Inter', -apple-system, sans-serif;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: var(--font-sans);
  background: var(--bg-primary);
  color: var(--text-primary);
  -webkit-font-smoothing: antialiased;
}

/* Layout */
.app-layout {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  width: 220px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  padding: 24px 0;
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
}

.sidebar-logo {
  padding: 0 20px 24px;
  border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: 14px;
  font-weight: 600;
  color: var(--accent);
  letter-spacing: -0.5px;
}

.sidebar-logo span {
  color: var(--text-secondary);
  font-weight: 400;
}

.sidebar nav {
  padding: 16px 0;
  flex: 1;
}

.sidebar a {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 20px;
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.15s;
}

.sidebar a:hover {
  color: var(--text-primary);
  background: var(--bg-hover);
}

.sidebar a.active {
  color: var(--accent);
  background: var(--accent-dim);
  border-right: 2px solid var(--accent);
}

.main-content {
  margin-left: 220px;
  flex: 1;
  padding: 32px;
  min-width: 0;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 28px;
}

.page-header h1 {
  font-size: 22px;
  font-weight: 600;
}

/* Cards */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 28px;
}

.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
}

.kpi-card .label {
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
}

.kpi-card .value {
  font-family: var(--font-mono);
  font-size: 28px;
  font-weight: 600;
  color: var(--text-primary);
}

.kpi-card .value.accent {
  color: var(--accent);
}

/* Charts */
.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 28px;
}

.chart-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
}

.chart-card h3 {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 16px;
  color: var(--text-secondary);
}

/* Tables */
.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

th {
  text-align: left;
  padding: 10px 14px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
  font-weight: 600;
}

td {
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  color: var(--text-secondary);
}

tr:hover td {
  background: var(--bg-hover);
}

td .mono {
  font-family: var(--font-mono);
  font-size: 13px;
}

/* Status badges */
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.badge.pending {
  background: rgba(59, 130, 246, 0.15);
  color: #60a5fa;
}

.badge.in_progress {
  background: rgba(245, 158, 11, 0.15);
  color: #fbbf24;
}

.badge.completed {
  background: rgba(34, 197, 94, 0.15);
  color: #4ade80;
}

.badge.flagged {
  background: rgba(239, 68, 68, 0.15);
  color: #f87171;
}

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  border: none;
  cursor: pointer;
  transition: all 0.15s;
}

.btn-primary {
  background: var(--accent);
  color: #000;
}

.btn-primary:hover {
  background: var(--accent-hover);
}

.btn-secondary {
  background: var(--bg-hover);
  color: var(--text-primary);
  border: 1px solid var(--border);
}

.btn-secondary:hover {
  border-color: var(--text-muted);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Forms */
input, select, textarea {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  color: var(--text-primary);
  font-size: 14px;
  font-family: var(--font-sans);
  width: 100%;
  outline: none;
  transition: border-color 0.15s;
}

input:focus, select:focus, textarea:focus {
  border-color: var(--accent);
}

label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
  margin-bottom: 6px;
}

.form-group {
  margin-bottom: 16px;
}

/* Modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 28px;
  width: 90%;
  max-width: 540px;
  max-height: 85vh;
  overflow-y: auto;
}

.modal h2 {
  font-size: 18px;
  margin-bottom: 20px;
}

.modal-actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
  margin-top: 24px;
}

/* Filter bar */
.filter-bar {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.filter-bar select, .filter-bar input {
  width: auto;
  min-width: 160px;
}

/* Banner */
.banner {
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
  margin-bottom: 16px;
}

.banner.success {
  background: rgba(34, 197, 94, 0.15);
  color: #4ade80;
  border: 1px solid rgba(34, 197, 94, 0.3);
}

.banner.error {
  background: rgba(239, 68, 68, 0.15);
  color: #f87171;
  border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Skeleton loading */
.skeleton {
  background: linear-gradient(90deg, var(--bg-hover) 25%, var(--bg-card) 50%, var(--bg-hover) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 6px;
  height: 20px;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Range slider */
input[type="range"] {
  -webkit-appearance: none;
  height: 6px;
  background: var(--border);
  border-radius: 3px;
  border: none;
  padding: 0;
}

input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
}

/* Pagination */
.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: 20px;
}

.pagination button {
  padding: 6px 12px;
}

.pagination .page-info {
  font-size: 13px;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

/* Annotate page — response cards */
.response-cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 20px;
}

.response-card {
  background: var(--bg-secondary);
  border: 2px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  cursor: pointer;
  transition: border-color 0.15s;
}

.response-card:hover {
  border-color: var(--text-muted);
}

.response-card.selected {
  border-color: var(--accent);
}

.response-card.chosen {
  border-color: var(--success);
}

.response-card.rejected {
  border-color: var(--error);
}

.response-card .model-label {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 8px;
}

.response-card .response-text {
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-primary);
  white-space: pre-wrap;
}

/* Criterion score sliders */
.criteria-sliders {
  display: grid;
  gap: 14px;
  margin-bottom: 20px;
}

.criterion-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.criterion-row label {
  min-width: 140px;
  margin-bottom: 0;
}

.criterion-row input[type="range"] {
  flex: 1;
}

.criterion-row .criterion-value {
  font-family: var(--font-mono);
  font-size: 13px;
  min-width: 36px;
  text-align: right;
  color: var(--accent);
}

/* Empty state */
.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: var(--text-muted);
}

.empty-state p {
  font-size: 15px;
  margin-bottom: 16px;
}
```

> **What's happening here:**
> - **CSS custom properties** (`--bg-primary`, `--accent`, etc.) defined in `:root` are the single source of truth for the design system. Every component references these variables, so changing the theme means editing one block.
> - The color hierarchy has four background levels: `primary` (darkest, page bg) → `secondary` (sidebar) → `card` (content cards) → `hover` (interactive states). This layering creates depth without borders everywhere.
> - **Status badges** use semi-transparent backgrounds (`rgba(...)`) that blend with any parent background. The color-coding is consistent: blue = pending, amber = in progress, green = completed, red = flagged.
> - **Skeleton loading** uses a CSS shimmer animation — a gradient that slides infinitely across the placeholder. This is shown while React Query fetches data, giving the user a visual signal that content is loading.
> - `.response-card.chosen` (green border) and `.response-card.rejected` (red border) visually distinguish the annotator's ranking on the Annotate page.

---

### Step 3.7: Application Shell — main.tsx and App.tsx

**Why:** `main.tsx` bootstraps the React tree with all required providers (React Query, Router). `App.tsx` defines the layout — a fixed sidebar with navigation links and a main content area that swaps between pages via React Router. The sidebar also contains the "New Task" button that opens the `CreateTaskModal`.

**Create `frontend/src/main.tsx`:**

```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
```

**Create `frontend/src/App.tsx`:**

```tsx
import { Routes, Route, NavLink } from 'react-router-dom';
import { useState } from 'react';
import Overview from './pages/Overview';
import Tasks from './pages/Tasks';
import Annotate from './pages/Annotate';
import Training from './pages/Training';
import Exports from './pages/Exports';
import CreateTaskModal from './components/CreateTaskModal';

function App() {
  const [showCreateTask, setShowCreateTask] = useState(false);

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          RL Platform <span>v0.1</span>
        </div>
        <nav>
          <NavLink to="/" end>Overview</NavLink>
          <NavLink to="/tasks">Tasks</NavLink>
          <NavLink to="/annotate">Annotate</NavLink>
          <NavLink to="/training">Training</NavLink>
          <NavLink to="/exports">Exports</NavLink>
        </nav>
        <div style={{ padding: '16px 20px' }}>
          <button className="btn btn-primary" style={{ width: '100%' }} onClick={() => setShowCreateTask(true)}>
            + New Task
          </button>
        </div>
      </aside>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/tasks" element={<Tasks />} />
          <Route path="/annotate" element={<Annotate />} />
          <Route path="/training" element={<Training />} />
          <Route path="/exports" element={<Exports />} />
        </Routes>
      </main>

      {showCreateTask && <CreateTaskModal onClose={() => setShowCreateTask(false)} />}
    </div>
  );
}

export default App;
```

> **What's happening here:**
> - `QueryClient` is configured with `retry: 1` (retry once on failure, not the default 3 times) and `refetchOnWindowFocus: false` (don't spam the API every time the user switches tabs). Individual queries can override these defaults — the Overview page sets `refetchInterval: 30_000` for live updates.
> - `NavLink` (from React Router) automatically adds the `.active` CSS class to the link matching the current URL. The `end` prop on the Overview link prevents it from being active for all routes (since every route starts with `/`).
> - The `CreateTaskModal` is conditionally rendered at the top level, outside of `<Routes>`, so it can be opened from any page via the sidebar button.

---

### Step 3.8: Overview Page — KPI Cards and Charts

**Why:** The Overview page is the landing page — it should give a researcher an instant read on platform health. KPI cards show counts and averages at a glance, the area chart shows feedback velocity over time (are annotators active?), and the bar chart shows the distribution of quality scores (are they high enough for training?).

**Create `frontend/src/pages/Overview.tsx`:**

```tsx
import { useQuery } from '@tanstack/react-query';
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api, type PlatformMetrics, type Task } from '../api/client';

function fmt(n: number | null | undefined): string {
  if (n == null) return '--';
  return n.toFixed(2);
}

export default function Overview() {
  const { data: metrics, isLoading } = useQuery<PlatformMetrics>({
    queryKey: ['metrics'],
    queryFn: api.getPlatformMetrics,
    refetchInterval: 30_000,
  });

  const { data: taskList } = useQuery({
    queryKey: ['tasks', 'recent'],
    queryFn: () => api.getTasks({ page: 1, page_size: 50 }),
    refetchInterval: 30_000,
  });

  // Build velocity data from tasks by date
  const velocityData = buildVelocityData(taskList?.tasks ?? []);
  const rewardData = buildRewardHistogram(taskList?.tasks ?? []);

  if (isLoading) {
    return (
      <div>
        <div className="page-header"><h1>Overview</h1></div>
        <div className="kpi-grid">
          {Array.from({ length: 6 }).map((_, i) => (
            <div className="kpi-card" key={i}><div className="skeleton" style={{ height: 40 }} /></div>
          ))}
        </div>
      </div>
    );
  }

  const kpis = [
    { label: 'Total Tasks', value: metrics?.total_tasks ?? 0 },
    { label: 'Pending', value: metrics?.pending_tasks ?? 0 },
    { label: 'Completed', value: metrics?.completed_tasks ?? 0 },
    { label: 'Feedback Items', value: metrics?.total_feedback ?? 0 },
    { label: 'Avg Quality', value: fmt(metrics?.avg_quality_score), accent: true },
    { label: 'Avg IAA', value: fmt(metrics?.avg_iaa), accent: true },
    { label: 'Queue Depth', value: metrics?.queue_depth ?? 0 },
    { label: 'Annotators', value: metrics?.total_annotators ?? 0 },
  ];

  return (
    <div>
      <div className="page-header">
        <h1>Overview</h1>
      </div>

      <div className="kpi-grid">
        {kpis.map((kpi) => (
          <div className="kpi-card" key={kpi.label}>
            <div className="label">{kpi.label}</div>
            <div className={`value ${kpi.accent ? 'accent' : ''}`}>{kpi.value}</div>
          </div>
        ))}
      </div>

      <div className="chart-grid">
        <div className="chart-card">
          <h3>Feedback Velocity</h3>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={velocityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
              <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip contentStyle={{ background: '#181c23', border: '1px solid #2a2f3a', borderRadius: 8, color: '#e8eaed' }} />
              <Area type="monotone" dataKey="count" stroke="#f59e0b" fill="rgba(245,158,11,0.2)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Quality Score Distribution</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={rewardData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
              <XAxis dataKey="range" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip contentStyle={{ background: '#181c23', border: '1px solid #2a2f3a', borderRadius: 8, color: '#e8eaed' }} />
              <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function buildVelocityData(tasks: Task[]) {
  const counts: Record<string, number> = {};
  for (const t of tasks) {
    if (!t.created_at) continue;
    const date = t.created_at.slice(0, 10);
    counts[date] = (counts[date] || 0) + 1;
  }
  return Object.entries(counts)
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-14)
    .map(([date, count]) => ({ date: date.slice(5), count }));
}

function buildRewardHistogram(tasks: Task[]) {
  const buckets = ['0-.2', '.2-.4', '.4-.6', '.6-.8', '.8-1'];
  const counts = [0, 0, 0, 0, 0];
  for (const t of tasks) {
    if (t.quality_score == null) continue;
    const idx = Math.min(Math.floor(t.quality_score * 5), 4);
    counts[idx]++;
  }
  return buckets.map((range, i) => ({ range, count: counts[i] }));
}
```

> **What's happening here:**
> - Two `useQuery` hooks run in parallel: one fetches platform metrics from the cached Redis endpoint, the other fetches recent tasks. Both refetch every 30 seconds (`refetchInterval: 30_000`), giving the Overview page near-real-time updates without WebSockets.
> - While loading, the page renders skeleton placeholders (shimmer animation) instead of a spinner — this prevents layout shift when data arrives because the skeletons occupy the same space as the real cards.
> - `buildVelocityData` groups tasks by their `created_at` date and counts how many were created each day over the last 14 days. The `AreaChart` renders this as a filled line chart showing annotation activity trends.
> - `buildRewardHistogram` buckets quality scores into 5 ranges (0–0.2, 0.2–0.4, etc.) and counts how many tasks fall in each range. The `BarChart` shows this distribution — ideally, most bars should be in the 0.6–1.0 range.
> - `fmt()` formats nullable numbers to 2 decimal places, showing `--` for null values. This ensures scores display as `0.83` (not `0.829999`) and null scores don't render as empty strings.
> - Recharts' `ResponsiveContainer` makes charts resize with their parent container. The `Tooltip` is styled to match the dark theme.

---

### Step 3.9: Tasks Page — Filterable Paginated Table

**Why:** Researchers need to browse, filter, and manage tasks. The task table supports server-side pagination (the backend's `GET /api/tasks/` already accepts `page` and `page_size` query params), status and type filters, flag/delete actions, and quality/IAA score columns for at-a-glance data quality assessment.

**Create `frontend/src/pages/Tasks.tsx`:**

```tsx
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';

export default function Tasks() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const pageSize = 20;

  const { data, isLoading } = useQuery({
    queryKey: ['tasks', page, statusFilter, typeFilter],
    queryFn: () => api.getTasks({
      page,
      page_size: pageSize,
      status: statusFilter || undefined,
      annotation_type: typeFilter || undefined,
    }),
  });

  const flagMutation = useMutation({
    mutationFn: api.flagTask,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['tasks'] }),
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteTask,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['tasks'] }),
  });

  const totalPages = data ? Math.ceil(data.total / pageSize) : 0;

  return (
    <div>
      <div className="page-header">
        <h1>Tasks</h1>
        <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          {data?.total ?? 0} total
        </span>
      </div>

      <div className="filter-bar">
        <select value={statusFilter} onChange={e => { setStatusFilter(e.target.value); setPage(1); }}>
          <option value="">All Statuses</option>
          <option value="pending">Pending</option>
          <option value="in_progress">In Progress</option>
          <option value="completed">Completed</option>
          <option value="flagged">Flagged</option>
        </select>
        <select value={typeFilter} onChange={e => { setTypeFilter(e.target.value); setPage(1); }}>
          <option value="">All Types</option>
          <option value="ranking">Ranking</option>
          <option value="scalar">Scalar</option>
          <option value="binary">Binary</option>
          <option value="critique">Critique</option>
        </select>
      </div>

      <div className="card">
        {isLoading ? (
          <div style={{ padding: 40 }}>
            {Array.from({ length: 5 }).map((_, i) => (
              <div className="skeleton" key={i} style={{ height: 20, marginBottom: 12 }} />
            ))}
          </div>
        ) : !data?.tasks.length ? (
          <div className="empty-state">
            <p>No tasks found</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Prompt</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Quality</th>
                  <th>IAA</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.tasks.map(task => (
                  <tr key={task.id}>
                    <td style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {task.prompt}
                    </td>
                    <td><span className="mono">{task.annotation_type}</span></td>
                    <td><span className={`badge ${task.status}`}>{task.status}</span></td>
                    <td><span className="mono">{task.quality_score?.toFixed(2) ?? '--'}</span></td>
                    <td><span className="mono">{task.iaa?.toFixed(2) ?? '--'}</span></td>
                    <td style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                      {task.created_at ? new Date(task.created_at).toLocaleDateString() : '--'}
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: '4px 10px', fontSize: 12 }}
                          onClick={() => flagMutation.mutate(task.id)}
                          disabled={task.status === 'flagged'}
                        >
                          Flag
                        </button>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: '4px 10px', fontSize: 12, color: 'var(--error)' }}
                          onClick={() => { if (confirm('Delete this task?')) deleteMutation.mutate(task.id); }}
                        >
                          Del
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button className="btn btn-secondary" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>
            Prev
          </button>
          <span className="page-info">{page} / {totalPages}</span>
          <button className="btn btn-secondary" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>
            Next
          </button>
        </div>
      )}
    </div>
  );
}
```

> **What's happening here:**
> - The `queryKey` includes `[page, statusFilter, typeFilter]` — when any of these change, React Query knows the previous cached data is for a different query and fetches fresh data. This is how filter and pagination changes trigger new API calls automatically.
> - Filter changes reset the page to 1 (`setPage(1)`) — otherwise you could be on page 5 of an unfiltered list and switch to a filter that only has 2 pages, resulting in an empty view.
> - `invalidateQueries({ queryKey: ['tasks'] })` after flag/delete mutations invalidates *all* task queries (regardless of page/filter combination), so the table refreshes with up-to-date data.
> - The prompt column uses `text-overflow: ellipsis` to truncate long prompts at 300px — this keeps table rows consistent and scannable.
> - The delete button uses `confirm()` as a minimal safety net to prevent accidental deletions.

---

### Step 3.10: Annotate Page — Pairwise Ranking UI

**Why:** This is the core annotation workflow — the reason the platform exists. An annotator selects their identity, pops the next task from the Redis queue, sees model responses side by side, clicks to rank them (1st click = chosen, subsequent clicks = rejected), adjusts criterion score sliders, sets their confidence level, optionally writes a critique, and submits. The backend immediately recomputes IAA and quality scores.

**Create `frontend/src/pages/Annotate.tsx`:**

```tsx
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, type Task } from '../api/client';

export default function Annotate() {
  const queryClient = useQueryClient();
  const [annotatorId, setAnnotatorId] = useState('');
  const [currentTask, setCurrentTask] = useState<Task | null>(null);
  const [ranking, setRanking] = useState<number[]>([]);
  const [criterionScores, setCriterionScores] = useState<Record<string, number>>({});
  const [confidence, setConfidence] = useState(0.8);
  const [critiqueText, setCritiqueText] = useState('');
  const [banner, setBanner] = useState<{ type: 'success' | 'error'; msg: string } | null>(null);

  const { data: annotators } = useQuery({
    queryKey: ['annotators'],
    queryFn: api.getAnnotators,
  });

  const fetchNextTask = useMutation({
    mutationFn: (id: string) => api.getNextTask(id),
    onSuccess: (task) => {
      setCurrentTask(task);
      setRanking([]);
      setCriterionScores({});
      setConfidence(0.8);
      setCritiqueText('');
      setBanner(null);
      // Initialize criterion scores
      if (task.evaluation_criteria) {
        const scores: Record<string, number> = {};
        for (const c of task.evaluation_criteria) scores[c] = 0.5;
        setCriterionScores(scores);
      }
    },
    onError: () => setBanner({ type: 'error', msg: 'No tasks available in queue' }),
  });

  const submitMutation = useMutation({
    mutationFn: api.submitFeedback,
    onSuccess: () => {
      setBanner({ type: 'success', msg: 'Feedback submitted successfully!' });
      setCurrentTask(null);
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      queryClient.invalidateQueries({ queryKey: ['metrics'] });
    },
    onError: (err) => setBanner({ type: 'error', msg: `Submission failed: ${err.message}` }),
  });

  const handleRankClick = (idx: number) => {
    setRanking(prev => {
      if (prev.includes(idx)) return prev.filter(i => i !== idx);
      return [...prev, idx];
    });
  };

  const handleSubmit = () => {
    if (!currentTask || !annotatorId) return;
    submitMutation.mutate({
      task_id: currentTask.id,
      annotator_id: annotatorId,
      ranking: ranking.length ? ranking : undefined,
      criterion_scores: Object.keys(criterionScores).length ? criterionScores : undefined,
      confidence,
      critique_text: critiqueText || undefined,
    });
  };

  return (
    <div>
      <div className="page-header">
        <h1>Annotate</h1>
      </div>

      {banner && <div className={`banner ${banner.type}`}>{banner.msg}</div>}

      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'end' }}>
          <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
            <label>Select Annotator</label>
            <select value={annotatorId} onChange={e => setAnnotatorId(e.target.value)}>
              <option value="">Choose annotator...</option>
              {annotators?.map(a => (
                <option key={a.id} value={a.id}>{a.name} ({a.email})</option>
              ))}
            </select>
          </div>
          <button
            className="btn btn-primary"
            disabled={!annotatorId || fetchNextTask.isPending}
            onClick={() => fetchNextTask.mutate(annotatorId)}
          >
            Get Next Task
          </button>
        </div>
      </div>

      {!currentTask ? (
        <div className="card">
          <div className="empty-state">
            <p>Select an annotator and click "Get Next Task" to start annotating</p>
          </div>
        </div>
      ) : (
        <>
          <div className="card" style={{ marginBottom: 20 }}>
            <div style={{ marginBottom: 12 }}>
              <span className="badge" style={{ marginRight: 8, background: 'var(--accent-dim)', color: 'var(--accent)' }}>
                {currentTask.annotation_type}
              </span>
              {currentTask.tags?.map(tag => (
                <span key={tag} className="badge" style={{ marginRight: 4, background: 'var(--bg-hover)', color: 'var(--text-muted)' }}>
                  {tag}
                </span>
              ))}
            </div>
            <h3 style={{ fontSize: 16, marginBottom: 16, lineHeight: 1.5 }}>{currentTask.prompt}</h3>

            {currentTask.responses && currentTask.responses.length > 0 && (
              <>
                <label style={{ marginBottom: 10 }}>
                  Rank responses (click in preferred order: 1st = best)
                </label>
                <div className="response-cards">
                  {currentTask.responses.map((resp, idx) => {
                    const rankPos = ranking.indexOf(idx);
                    let cls = 'response-card';
                    if (rankPos === 0) cls += ' chosen';
                    else if (rankPos > 0) cls += ' rejected';
                    else if (rankPos >= 0) cls += ' selected';
                    return (
                      <div key={idx} className={cls} onClick={() => handleRankClick(idx)}>
                        <div className="model-label">
                          {resp.model_id}
                          {rankPos >= 0 && (
                            <span style={{ marginLeft: 8, color: 'var(--accent)', fontWeight: 600 }}>
                              #{rankPos + 1}
                            </span>
                          )}
                        </div>
                        <div className="response-text">{resp.text}</div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>

          {currentTask.evaluation_criteria && currentTask.evaluation_criteria.length > 0 && (
            <div className="card" style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: 14, color: 'var(--text-secondary)', marginBottom: 14 }}>
                Criterion Scores
              </h3>
              <div className="criteria-sliders">
                {currentTask.evaluation_criteria.map(criterion => (
                  <div className="criterion-row" key={criterion}>
                    <label>{criterion}</label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={criterionScores[criterion] ?? 0.5}
                      onChange={e => setCriterionScores(prev => ({ ...prev, [criterion]: parseFloat(e.target.value) }))}
                    />
                    <span className="criterion-value">{(criterionScores[criterion] ?? 0.5).toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card" style={{ marginBottom: 20 }}>
            <div className="form-group">
              <label>Confidence</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={confidence}
                  onChange={e => setConfidence(parseFloat(e.target.value))}
                />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 14, color: 'var(--accent)', minWidth: 36 }}>
                  {confidence.toFixed(2)}
                </span>
              </div>
            </div>

            <div className="form-group">
              <label>Critique / Notes (optional)</label>
              <textarea
                rows={3}
                value={critiqueText}
                onChange={e => setCritiqueText(e.target.value)}
                placeholder="Explain your ranking rationale..."
              />
            </div>
          </div>

          <button
            className="btn btn-primary"
            style={{ width: '100%', justifyContent: 'center', padding: '12px 20px', fontSize: 15 }}
            onClick={handleSubmit}
            disabled={submitMutation.isPending}
          >
            {submitMutation.isPending ? 'Submitting...' : 'Submit Feedback'}
          </button>
        </>
      )}
    </div>
  );
}
```

> **What's happening here:**
> - **"Get Next Task"** calls `GET /api/annotators/{id}/next-task`, which pops a task from the Redis queue and creates a `task_assignment` record. This is how tasks are distributed fairly — each annotator gets the next available task from the queue.
> - **Click-to-rank** uses a simple state machine: clicking a response adds its index to the `ranking` array. The first click marks it as "chosen" (green border), subsequent clicks mark responses as "rejected" (red border). Clicking an already-ranked response removes it from the ranking (toggle behavior). This maps directly to the `ranking: list[int]` field the backend expects.
> - **Criterion scores** are only shown when the task has `evaluation_criteria` defined. Each criterion gets a `0.0–1.0` range slider initialized at `0.5`. These scores feed into the quality scoring formula on the backend.
> - **Confidence slider** defaults to `0.8` — most annotators are reasonably confident, so this reduces friction. The value is sent as `confidence: float` in the feedback payload and is used by the backend's reliability-weighted consensus reward calculation.
> - After successful submission, `setCurrentTask(null)` clears the form and both `['tasks']` and `['metrics']` query caches are invalidated, so the Overview and Tasks pages reflect the new feedback immediately.

---

### Step 3.11: Training Page — Metric Line Charts

**Why:** Researchers need to monitor training runs that consume the exported datasets. This page displays reward curves, KL divergence, and loss over training steps — the three metrics that tell you whether a DPO or PPO run is converging properly. Clicking a row in the training run table switches the chart to that run's data.

**Create `frontend/src/pages/Training.tsx`:**

```tsx
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { api, type TrainingRun } from '../api/client';

export default function Training() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const { data: runs, isLoading } = useQuery({
    queryKey: ['training-runs'],
    queryFn: api.getTrainingRuns,
  });

  const selectedRun = runs?.find(r => r.id === selectedRunId) ?? runs?.[0] ?? null;
  const chartData = buildChartData(selectedRun);

  return (
    <div>
      <div className="page-header">
        <h1>Training Runs</h1>
      </div>

      {isLoading ? (
        <div className="card">
          {Array.from({ length: 3 }).map((_, i) => (
            <div className="skeleton" key={i} style={{ height: 20, marginBottom: 12 }} />
          ))}
        </div>
      ) : !runs?.length ? (
        <div className="card">
          <div className="empty-state">
            <p>No training runs yet</p>
            <p style={{ fontSize: 13 }}>Training runs will appear here once datasets are used for fine-tuning</p>
          </div>
        </div>
      ) : (
        <>
          <div className="filter-bar">
            <select
              value={selectedRun?.id ?? ''}
              onChange={e => setSelectedRunId(e.target.value)}
            >
              {runs.map(run => (
                <option key={run.id} value={run.id}>
                  {run.algorithm} — {run.status} ({run.id.slice(0, 8)})
                </option>
              ))}
            </select>
          </div>

          {selectedRun && chartData.length > 0 && (
            <div className="chart-grid">
              <div className="chart-card" style={{ gridColumn: '1 / -1' }}>
                <h3>Training Metrics</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
                    <XAxis dataKey="step" stroke="#6b7280" fontSize={12} label={{ value: 'Step', position: 'insideBottom', offset: -5, fill: '#6b7280' }} />
                    <YAxis stroke="#6b7280" fontSize={12} />
                    <Tooltip contentStyle={{ background: '#181c23', border: '1px solid #2a2f3a', borderRadius: 8, color: '#e8eaed' }} />
                    <Legend />
                    <Line type="monotone" dataKey="reward" stroke="#f59e0b" strokeWidth={2} dot={false} name="Reward" />
                    <Line type="monotone" dataKey="kl" stroke="#3b82f6" strokeWidth={2} dot={false} name="KL Divergence" />
                    <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} name="Loss" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          <div className="card">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Algorithm</th>
                    <th>Dataset</th>
                    <th>Status</th>
                    <th>Steps</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map(run => (
                    <tr
                      key={run.id}
                      onClick={() => setSelectedRunId(run.id)}
                      style={{ cursor: 'pointer', background: run.id === selectedRun?.id ? 'var(--bg-hover)' : undefined }}
                    >
                      <td><span className="mono">{run.id.slice(0, 8)}</span></td>
                      <td>{run.algorithm}</td>
                      <td><span className="mono">{run.dataset_id.slice(0, 8)}</span></td>
                      <td><span className={`badge ${run.status}`}>{run.status}</span></td>
                      <td><span className="mono">{run.reward_history?.length ?? 0}</span></td>
                      <td style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                        {run.created_at ? new Date(run.created_at).toLocaleDateString() : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function buildChartData(run: TrainingRun | null) {
  if (!run) return [];
  const maxLen = Math.max(
    run.reward_history?.length ?? 0,
    run.kl_history?.length ?? 0,
    run.loss_history?.length ?? 0,
  );
  return Array.from({ length: maxLen }, (_, i) => ({
    step: i + 1,
    reward: run.reward_history?.[i] ?? null,
    kl: run.kl_history?.[i] ?? null,
    loss: run.loss_history?.[i] ?? null,
  }));
}
```

> **What's happening here:**
> - `buildChartData` normalizes the three metric arrays (reward, KL, loss) into a single array of `{ step, reward, kl, loss }` objects — the format Recharts needs. Arrays may have different lengths, so it uses the longest one and fills gaps with `null` (Recharts renders nulls as gaps in the line).
> - Three `Line` components share one chart, color-coded: amber for reward (the metric you want to go up), blue for KL divergence (should stay bounded), red for loss (should go down). This is the standard RL training dashboard layout.
> - The `gridColumn: '1 / -1'` on the chart card makes it span the full width, overriding the default 2-column grid. Training metrics need horizontal space to show trends clearly.
> - Clicking a table row updates `selectedRunId`, which changes the chart. The default selection is `runs?.[0]` — the first run in the list.

---

### Step 3.12: Exports Page — Dataset Builder

**Why:** The Exports page is where researchers turn annotated tasks into training data. The form lets them name a dataset, choose a format (JSONL/Parquet/HuggingFace), set quality and IAA thresholds via sliders, and trigger a background export. The dataset table shows export status and provides download links when ready.

**Create `frontend/src/pages/Exports.tsx`:**

```tsx
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';

export default function Exports() {
  const queryClient = useQueryClient();
  const [name, setName] = useState('');
  const [format, setFormat] = useState('jsonl');
  const [minQuality, setMinQuality] = useState(0);
  const [minIaa, setMinIaa] = useState(0);
  const [status, setStatus] = useState('completed');
  const [banner, setBanner] = useState<{ type: 'success' | 'error'; msg: string } | null>(null);

  const { data: datasets, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: api.getDatasets,
    refetchInterval: 10_000,
  });

  const createMutation = useMutation({
    mutationFn: api.createDataset,
    onSuccess: (ds) => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      setBanner({ type: 'success', msg: `Dataset "${ds.name}" created. Export is building in the background.` });
      setName('');
    },
    onError: (err) => setBanner({ type: 'error', msg: `Failed: ${err.message}` }),
  });

  const handleCreate = () => {
    if (!name.trim()) return;
    createMutation.mutate({
      name: name.trim(),
      export_format: format,
      filters: {
        min_quality_score: minQuality,
        min_iaa: minIaa,
        status,
      },
    });
  };

  return (
    <div>
      <div className="page-header">
        <h1>Exports</h1>
      </div>

      {banner && <div className={`banner ${banner.type}`}>{banner.msg}</div>}

      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, color: 'var(--text-secondary)', marginBottom: 16 }}>Build New Dataset</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="form-group">
            <label>Dataset Name</label>
            <input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. coding-dpo-v1" />
          </div>
          <div className="form-group">
            <label>Export Format</label>
            <select value={format} onChange={e => setFormat(e.target.value)}>
              <option value="jsonl">JSONL</option>
              <option value="parquet">Parquet</option>
              <option value="hf">HuggingFace Dataset</option>
            </select>
          </div>
          <div className="form-group">
            <label>Min Quality Score: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>{minQuality.toFixed(2)}</span></label>
            <input type="range" min="0" max="1" step="0.05" value={minQuality} onChange={e => setMinQuality(parseFloat(e.target.value))} />
          </div>
          <div className="form-group">
            <label>Min IAA (kappa): <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>{minIaa.toFixed(2)}</span></label>
            <input type="range" min="0" max="1" step="0.05" value={minIaa} onChange={e => setMinIaa(parseFloat(e.target.value))} />
          </div>
          <div className="form-group">
            <label>Task Status</label>
            <select value={status} onChange={e => setStatus(e.target.value)}>
              <option value="completed">Completed</option>
              <option value="all">All</option>
              <option value="pending">Pending</option>
            </select>
          </div>
        </div>
        <button
          className="btn btn-primary"
          style={{ marginTop: 8 }}
          onClick={handleCreate}
          disabled={!name.trim() || createMutation.isPending}
        >
          {createMutation.isPending ? 'Creating...' : 'Create Dataset & Export'}
        </button>
      </div>

      <div className="card">
        <h3 style={{ fontSize: 14, color: 'var(--text-secondary)', marginBottom: 16 }}>Datasets</h3>
        {isLoading ? (
          <div>
            {Array.from({ length: 3 }).map((_, i) => (
              <div className="skeleton" key={i} style={{ height: 20, marginBottom: 12 }} />
            ))}
          </div>
        ) : !datasets?.length ? (
          <div className="empty-state">
            <p>No datasets exported yet</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Format</th>
                  <th>Tasks</th>
                  <th>Status</th>
                  <th>Exported</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map(ds => (
                  <tr key={ds.id}>
                    <td>{ds.name}</td>
                    <td><span className="mono">{ds.export_format}</span></td>
                    <td><span className="mono">{ds.task_count}</span></td>
                    <td>
                      {ds.exported_at ? (
                        <span className="badge completed">Ready</span>
                      ) : (
                        <span className="badge pending">Building...</span>
                      )}
                    </td>
                    <td style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                      {ds.exported_at ? new Date(ds.exported_at).toLocaleString() : '--'}
                    </td>
                    <td>
                      {ds.exported_at && (
                        <a
                          href={api.downloadDataset(ds.id)}
                          className="btn btn-secondary"
                          style={{ padding: '4px 10px', fontSize: 12, textDecoration: 'none' }}
                        >
                          Download
                        </a>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
```

> **What's happening here:**
> - `refetchInterval: 10_000` polls the dataset list every 10 seconds. This is how the "Building..." badge transitions to "Ready" — the backend sets `exported_at` when the background export completes, and the next poll picks it up.
> - Quality and IAA sliders send their values as `filters` in the create request. The backend's `_build_export()` uses these to filter which tasks make it into the exported dataset — only tasks meeting the minimum thresholds are included. This is critical for training quality: a dataset with IAA < 0.4 is mostly noise.
> - The download link uses `api.downloadDataset(ds.id)` which returns a raw URL string (not a fetch call). The browser navigates to this URL directly, and the backend's `FileResponse` streams the JSONL file with the correct `Content-Disposition: attachment` header.
> - The "Building..." / "Ready" badge maps to the presence of `exported_at` — `null` means the background task is still running, a timestamp means the file is ready for download.

---

### Step 3.13: CreateTaskModal Component

**Why:** A modal is the right pattern for task creation because it can be triggered from any page (via the sidebar button) without navigating away from the current view. After creating a task, the user stays on whatever page they were on, and the cache invalidation ensures the Tasks page and Overview metrics update automatically.

**Create `frontend/src/components/CreateTaskModal.tsx`:**

```tsx
import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';

interface Props {
  onClose: () => void;
}

export default function CreateTaskModal({ onClose }: Props) {
  const queryClient = useQueryClient();
  const [prompt, setPrompt] = useState('');
  const [annotationType, setAnnotationType] = useState('ranking');
  const [minAnnotations, setMinAnnotations] = useState(3);
  const [tags, setTags] = useState('');
  const [criteria, setCriteria] = useState('');
  const [responses, setResponses] = useState([
    { model_id: 'model-a', text: '' },
    { model_id: 'model-b', text: '' },
  ]);
  const [error, setError] = useState('');

  const mutation = useMutation({
    mutationFn: api.createTask,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      queryClient.invalidateQueries({ queryKey: ['metrics'] });
      onClose();
    },
    onError: (err) => setError(err.message),
  });

  const handleSubmit = () => {
    if (!prompt.trim()) { setError('Prompt is required'); return; }
    setError('');
    const filteredResponses = responses.filter(r => r.text.trim());
    mutation.mutate({
      prompt: prompt.trim(),
      annotation_type: annotationType,
      min_annotations: minAnnotations,
      responses: filteredResponses.length ? filteredResponses : undefined,
      tags: tags.trim() ? tags.split(',').map(t => t.trim()) : undefined,
      evaluation_criteria: criteria.trim() ? criteria.split(',').map(c => c.trim()) : undefined,
    });
  };

  const updateResponse = (idx: number, field: 'model_id' | 'text', value: string) => {
    setResponses(prev => prev.map((r, i) => i === idx ? { ...r, [field]: value } : r));
  };

  const addResponse = () => {
    setResponses(prev => [...prev, { model_id: `model-${String.fromCharCode(97 + prev.length)}`, text: '' }]);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <h2>Create New Task</h2>

        {error && <div className="banner error">{error}</div>}

        <div className="form-group">
          <label>Prompt</label>
          <textarea rows={3} value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="Enter the task prompt..." />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div className="form-group">
            <label>Annotation Type</label>
            <select value={annotationType} onChange={e => setAnnotationType(e.target.value)}>
              <option value="ranking">Ranking</option>
              <option value="scalar">Scalar</option>
              <option value="binary">Binary</option>
              <option value="critique">Critique</option>
            </select>
          </div>
          <div className="form-group">
            <label>Min Annotations</label>
            <input type="number" min={1} max={10} value={minAnnotations} onChange={e => setMinAnnotations(Number(e.target.value))} />
          </div>
        </div>

        <div className="form-group">
          <label>Tags (comma-separated)</label>
          <input value={tags} onChange={e => setTags(e.target.value)} placeholder="python, concurrency, ..." />
        </div>

        <div className="form-group">
          <label>Evaluation Criteria (comma-separated)</label>
          <input value={criteria} onChange={e => setCriteria(e.target.value)} placeholder="correctness, code quality, ..." />
        </div>

        <div style={{ marginBottom: 16 }}>
          <label>Model Responses</label>
          {responses.map((resp, idx) => (
            <div key={idx} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
              <input
                style={{ width: 120, flexShrink: 0 }}
                value={resp.model_id}
                onChange={e => updateResponse(idx, 'model_id', e.target.value)}
                placeholder="Model ID"
              />
              <textarea
                rows={2}
                value={resp.text}
                onChange={e => updateResponse(idx, 'text', e.target.value)}
                placeholder="Response text..."
              />
            </div>
          ))}
          <button className="btn btn-secondary" style={{ fontSize: 12 }} onClick={addResponse}>
            + Add Response
          </button>
        </div>

        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={handleSubmit} disabled={mutation.isPending}>
            {mutation.isPending ? 'Creating...' : 'Create Task'}
          </button>
        </div>
      </div>
    </div>
  );
}
```

> **What's happening here:**
> - `onClick={onClose}` on the overlay closes the modal when clicking outside; `e.stopPropagation()` on the modal body prevents the overlay's click handler from firing when clicking inside the form.
> - The form starts with two empty response slots (`model-a` and `model-b`) — the minimum for pairwise ranking. The "+ Add Response" button appends more (auto-naming them `model-c`, `model-d`, etc. using `String.fromCharCode`).
> - Empty responses (no text) are filtered out before sending to the API: `responses.filter(r => r.text.trim())`. If all responses are empty, `responses` is sent as `undefined` (creating a task without responses is valid — responses can be added later).
> - Comma-separated tags and criteria are split into arrays: `tags.split(',').map(t => t.trim())`. Empty strings are excluded by checking `.trim()` before splitting.
> - On successful creation, both `['tasks']` and `['metrics']` caches are invalidated, ensuring the task list and Overview KPI counts update immediately.

---

### Step 3.14: Frontend Dockerfile and nginx

**Why:** The production frontend is a static build served by nginx. Vite compiles TypeScript and bundles everything into optimized JS/CSS in the `dist/` directory. A multi-stage Docker build keeps the final image tiny — only nginx and the static files, no Node.js runtime. The nginx config also includes a reverse proxy for `/api/` routes, so the frontend container can serve both the UI and proxy API requests in production.

**Create `frontend/Dockerfile`:**

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Create `frontend/nginx.conf`:**

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

> **What's happening here:**
> - **Multi-stage build:** Stage 1 (`node:20-alpine`) installs deps and runs `npm run build`. Stage 2 (`nginx:alpine`) copies only the built `dist/` files. The final image is ~25MB instead of ~300MB because Node.js and `node_modules` are discarded.
> - `COPY package.json package-lock.json* ./` — the `*` glob on `package-lock.json` means the build won't fail if the lockfile doesn't exist yet (it's generated on first `npm install`).
> - **`try_files $uri $uri/ /index.html`** is the SPA fallback — when the user navigates to `/tasks` or `/annotate`, nginx doesn't have a real file at that path (it's a client-side route). This directive falls back to `index.html`, which loads the React app, and React Router handles the route.
> - **`/api/` reverse proxy** forwards API requests from the frontend container to the `api` service on port 8000. This means in production, the frontend can use relative `/api/` URLs instead of cross-origin calls. The Docker Compose service name `api` resolves to the correct container IP via Docker's internal DNS.

---

### Step 3.15: Docker Compose — Add Frontend Service

**Why:** Adding the frontend service to `docker-compose.yml` means `docker compose up --build` starts the complete stack — database, cache, API, worker, and frontend — with a single command.

**Update `docker-compose.yml`** — add the `frontend` service block before `worker`:

```yaml
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
```

> **What's happening here:**
> - Port mapping `3000:80` maps host port 3000 to the nginx container's port 80. The dashboard is accessible at `http://localhost:3000`.
> - `depends_on: api: condition: service_healthy` ensures the frontend container only starts after the API's health check passes. This prevents the user from seeing the dashboard before the backend is ready to serve requests.
> - `VITE_API_URL` is set as an environment variable, but note that Vite environment variables are baked in at *build time*, not runtime. The `import.meta.env.VITE_API_URL` in `client.ts` is replaced with the literal string during `npm run build`. For local Docker development, the frontend uses the nginx reverse proxy (`/api/` → `api:8000`), so this variable primarily matters for the dev server.

---

### Verify Phase 3

**Build check (no Docker required):**

```bash
cd frontend && npm run build
```

This should complete with zero TypeScript errors:

```
✓ 886 modules transformed.
dist/index.html                   0.71 kB │ gzip:   0.39 kB
dist/assets/index-*.css           6.77 kB │ gzip:   1.99 kB
dist/assets/index-*.js          626.04 kB │ gzip: 179.03 kB
✓ built in ~900ms
```

**Full stack verification:**

```bash
docker compose up --build
```

Then verify:

1. **Frontend loads:** Open `http://localhost:3000` — you should see the dark-themed dashboard with the sidebar and Overview page.
2. **Overview shows data:** KPI cards display zeros (or real data if you've seeded tasks). Charts render with empty state or real data.
3. **Create a task:** Click "+ New Task" in the sidebar, fill in a prompt, add two model responses, and click "Create Task." The task should appear in the Tasks page.
4. **Task table works:** Navigate to `/tasks` — the table shows the created task. Filters and pagination work.
5. **Annotate flow:** Go to `/annotate`, select an annotator, click "Get Next Task" (requires a registered annotator and a task in the queue). Rank responses, set confidence, and submit.
6. **API health:** `curl http://localhost:8000/health` returns `{"status":"ok","db":"ok","redis":"ok"}`.

**Acceptance criteria checklist:**

- [ ] All 5 pages render with real data from the backend API
- [ ] Overview page shows live feedback count and updates on refresh
- [ ] Creating a task from the modal adds it to the task table and Redis queue
- [ ] Annotate page submits feedback and shows success confirmation
- [ ] `docker compose up` serves frontend at `http://localhost:3000`
- [ ] `npm run build` succeeds with no TypeScript errors

---

## Updated Project Structure

After completing Phase 3, the project directory looks like this:

```
agent-rl-training-data-platform/
├── CLAUDE.md
├── SPEC.md
├── README.md
├── docker-compose.yml                    # Updated: frontend service added
│
├── backend/
│   ├── main.py                           # Updated: CORS middleware added
│   ├── models.py
│   ├── schemas.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── pytest.ini
│   ├── core/
│   │   ├── database.py
│   │   └── redis_client.py
│   ├── routes/
│   │   ├── tasks.py
│   │   ├── feedback.py
│   │   ├── annotators.py
│   │   ├── metrics.py
│   │   └── exports.py
│   ├── workers/
│   │   └── quality_worker.py
│   └── tests/
│       ├── conftest.py
│       ├── test_tasks.py
│       ├── test_feedback.py
│       ├── test_annotators.py
│       ├── test_metrics.py
│       └── test_exports.py
│
├── frontend/                              # NEW: entire directory
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   ├── Dockerfile
│   ├── nginx.conf
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css
│       ├── vite-env.d.ts
│       ├── api/
│       │   └── client.ts
│       ├── pages/
│       │   ├── Overview.tsx
│       │   ├── Tasks.tsx
│       │   ├── Annotate.tsx
│       │   ├── Training.tsx
│       │   └── Exports.tsx
│       └── components/
│           └── CreateTaskModal.tsx
│
├── scripts/
│   └── init.sql
│
└── docs/
    └── tutorial/
        ├── tutorial-phases-1-2.md
        └── tutorial-phase-3.md            # This file
```

---

## Common Issues and Troubleshooting

**CORS errors in the browser console**
If you see `Access-Control-Allow-Origin` errors, make sure the CORS middleware in `backend/main.py` includes the origin your frontend is running on. For Vite dev server, that's `http://localhost:5173` (Vite's default) or `http://localhost:3000` (our config). Both are included in the middleware.

**`npm run build` fails with TypeScript errors**
Run `npx tsc --noEmit` to see the full error list. Common causes: a missing `| null` on an interface field that the backend returns as `null`, or using a React Query method with the wrong type parameter.

**Charts render but show no data**
Charts depend on tasks existing in the database. Create a few tasks via the CreateTaskModal or the API directly (`POST /api/tasks/`). The Overview page's velocity chart groups tasks by `created_at` date, so tasks created today will show as a single point.

**"Get Next Task" returns an error on the Annotate page**
This requires two things: (1) a registered annotator (create one via `POST /api/annotators/`), and (2) tasks in the Redis queue (creating a task via `POST /api/tasks/` automatically enqueues it). If the queue is empty, the backend returns a 404 which the frontend displays as "No tasks available in queue."

**Frontend Dockerfile build fails on `npm run build`**
The Docker build copies all source files and runs the full TypeScript + Vite build inside the container. If it fails, run `npm run build` locally first to see the actual error. Common cause: a `node_modules` or `dist` directory being copied into the Docker context — add a `.dockerignore` file with `node_modules` and `dist` entries.

**nginx returns 502 Bad Gateway for `/api/` routes**
The nginx reverse proxy uses the Docker Compose service name `api` as the hostname. This only resolves inside the Docker network. If you're running nginx outside Docker, change `proxy_pass http://api:8000` to `proxy_pass http://localhost:8000`.

**Fonts don't load (fallback to system fonts)**
The Google Fonts link in `index.html` requires internet access. If running offline or in an air-gapped environment, download the IBM Plex Mono and Inter font files and serve them from the `public/` directory instead.

---

## Next Steps

With Phase 3 complete, the full frontend is functional and connected to the backend API. The logical next phases are:

**Phase 4 — Worker + Exports + Seed Data:** Replace the `quality_worker.py` placeholder with a real Redis consumer that processes feedback events, recomputes quality scores in the background, and updates task statuses. Add a `scripts/seed.py` that populates 50 synthetic tasks with realistic feedback distributions. Add Parquet and HuggingFace export formats.

**Phase 5 — Observability:** Add Prometheus metrics middleware to FastAPI (`prometheus-fastapi-instrumentator`), configure Grafana dashboards for annotator throughput and queue depth, and add structured JSON logging with `structlog`.

**Relevant Documentation:**
- React Query v5: https://tanstack.com/query/latest
- Recharts: https://recharts.org/en-US/api
- React Router v6: https://reactrouter.com/en/main
- Vite: https://vitejs.dev/guide/
- nginx reverse proxy: https://nginx.org/en/docs/http/ngx_http_proxy_module.html
