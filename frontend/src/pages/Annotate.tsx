import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, hasAuthToken, setAuthToken, type Task } from '../api/client';

/** Click order (response indices, best first) -> rank of each response, 1 = best. */
function toRanks(order: number[], count: number): number[] {
  const ranks = new Array<number>(count);
  order.forEach((responseIdx, position) => { ranks[responseIdx] = position + 1; });
  return ranks;
}

export default function Annotate() {
  const queryClient = useQueryClient();
  const [tokenInput, setTokenInput] = useState('');
  const [currentTask, setCurrentTask] = useState<Task | null>(null);
  const [ranking, setRanking] = useState<number[]>([]);
  const [criterionScores, setCriterionScores] = useState<Record<string, number>>({});
  const [confidence, setConfidence] = useState(0.8);
  const [critiqueText, setCritiqueText] = useState('');
  const [banner, setBanner] = useState<{ type: 'success' | 'error'; msg: string } | null>(null);

  // Who the stored token belongs to. A 401 here means the token is missing or invalid.
  const { data: me, error: meError } = useQuery({
    queryKey: ['me'],
    queryFn: api.getMe,
    enabled: hasAuthToken(),
    retry: false,
  });

  const signIn = () => {
    setAuthToken(tokenInput.trim());
    setTokenInput('');
    queryClient.resetQueries({ queryKey: ['me'] });
  };

  const signOut = () => {
    setAuthToken(null);
    setCurrentTask(null);
    queryClient.removeQueries({ queryKey: ['me'] });
    setBanner(null);
  };

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
    onError: () => setBanner({ type: 'error', msg: 'No tasks available for you right now' }),
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

  const responseCount = currentTask?.responses?.length ?? 0;
  const rankingComplete = responseCount < 2 || ranking.length === responseCount;

  const handleSubmit = () => {
    if (!currentTask || !me || !rankingComplete) return;
    submitMutation.mutate({
      task_id: currentTask.id,
      ranking: responseCount >= 2 ? toRanks(ranking, responseCount) : undefined,
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
          {me ? (
            <>
              <div style={{ flex: 1 }}>
                <label>Signed in as</label>
                <div style={{ fontSize: 15 }}>
                  {me.name} <span style={{ color: 'var(--text-muted)' }}>({me.email})</span>
                  <span style={{ marginLeft: 12, fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-muted)' }}>
                    reliability {me.reliability_score.toFixed(2)}
                  </span>
                </div>
              </div>
              <button className="btn btn-secondary" onClick={signOut}>Sign out</button>
              <button
                className="btn btn-primary"
                disabled={fetchNextTask.isPending}
                onClick={() => fetchNextTask.mutate(me.id)}
              >
                Get Next Task
              </button>
            </>
          ) : (
            <>
              <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
                <label>Annotator API token</label>
                <input
                  type="password"
                  value={tokenInput}
                  onChange={e => setTokenInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && tokenInput.trim()) signIn(); }}
                  placeholder="Paste the token you received when you were registered"
                />
                {meError && hasAuthToken() && (
                  <div style={{ color: 'var(--error)', fontSize: 13, marginTop: 6 }}>
                    That token was not recognised.
                  </div>
                )}
              </div>
              <button className="btn btn-primary" disabled={!tokenInput.trim()} onClick={signIn}>
                Sign in
              </button>
            </>
          )}
        </div>
      </div>

      {!currentTask ? (
        <div className="card">
          <div className="empty-state">
            <p>{me ? 'Click "Get Next Task" to start annotating' : 'Sign in with your annotator token to start annotating'}</p>
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
            disabled={submitMutation.isPending || !rankingComplete}
          >
            {submitMutation.isPending
              ? 'Submitting...'
              : rankingComplete
                ? 'Submit Feedback'
                : `Rank all ${responseCount} responses to submit`}
          </button>
        </>
      )}
    </div>
  );
}
