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
