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
