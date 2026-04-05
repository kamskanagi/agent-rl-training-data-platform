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
