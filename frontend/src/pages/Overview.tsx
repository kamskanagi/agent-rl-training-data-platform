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
