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
