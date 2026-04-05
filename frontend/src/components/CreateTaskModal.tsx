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
