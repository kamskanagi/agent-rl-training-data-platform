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
