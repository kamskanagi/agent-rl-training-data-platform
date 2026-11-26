"""Quality scoring shared by the feedback route and the quality worker."""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Annotator, FeedbackItem


def fleiss_kappa(rankings: list[list[int]]) -> float:
    """Fleiss' kappa over the pairwise preferences implied by ranking vectors.

    Each pair of responses (i, j) is a subject; each rater places it in one of two
    categories: "i ranked above j" or "j ranked above i". Expected agreement P_e is
    computed from the observed category proportions, as in standard Fleiss' kappa.
    Rankings whose length differs from the most common length are ignored.
    """
    if len(rankings) < 2:
        return 0.0
    n_items = Counter(len(r) for r in rankings).most_common(1)[0][0]
    rankings = [r for r in rankings if len(r) == n_items]
    n = len(rankings)
    if n < 2 or n_items < 2:
        return 0.0

    pairs = list(combinations(range(n_items), 2))
    p_bar = 0.0
    first_total = 0
    for i, j in pairs:
        first = sum(1 for r in rankings if r[i] < r[j])
        second = n - first
        p_bar += (first * first + second * second - n) / (n * (n - 1))
        first_total += first
    p_bar /= len(pairs)

    p_first = first_total / (n * len(pairs))
    p_e = p_first ** 2 + (1 - p_first) ** 2
    if p_e >= 1.0:
        # Every rater put every pair in the same category: perfect agreement.
        return 1.0
    kappa = (p_bar - p_e) / (1.0 - p_e)
    return max(-1.0, min(1.0, kappa))


def _absolute_reward(fb: FeedbackItem) -> float | None:
    if fb.scalar_reward is not None:
        return fb.scalar_reward
    if fb.binary_label is not None:
        return 1.0 if fb.binary_label else 0.0
    return None


def _top_response_reward(rankings: list[tuple[list[int], float]]) -> float:
    """Weighted mean reward of the response the raters favour most.

    Each ranking maps rank r (1 = best) of n responses to reward 1 - (r-1)/(n-1).
    """
    n_items = Counter(len(r) for r, _ in rankings).most_common(1)[0][0]
    totals = [0.0] * n_items
    weight_sum = 0.0
    for ranking, w in rankings:
        if len(ranking) != n_items:
            continue
        for idx, rank in enumerate(ranking):
            totals[idx] += w * (1.0 - (rank - 1) / (n_items - 1))
        weight_sum += w
    if weight_sum == 0:
        return 0.0
    return max(totals) / weight_sum


def consensus_reward(
    feedbacks: list[FeedbackItem], reliability: dict[str, float] | None = None
) -> float:
    """Reliability-weighted average reward.

    Scalar and binary feedback contribute their value directly. Rankings contribute
    the consensus reward of the top-ranked response. Each annotator is weighted by
    their reliability score (default 1.0).
    """
    reliability = reliability or {}
    weight = lambda fb: max(0.0, reliability.get(fb.annotator_id, 1.0))

    values: list[tuple[float, float]] = []
    rankings: list[tuple[list[int], float]] = []
    for fb in feedbacks:
        reward = _absolute_reward(fb)
        if reward is not None:
            values.append((reward, weight(fb)))
        elif fb.ranking is not None and len(fb.ranking) >= 2:
            rankings.append((fb.ranking, weight(fb)))

    if rankings:
        values.append((_top_response_reward(rankings), sum(w for _, w in rankings)))

    total_weight = sum(w for _, w in values)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in values) / total_weight


def quality_score(
    iaa: float | None,
    feedbacks: list[FeedbackItem],
    min_annotations: int,
) -> float:
    """quality_score = 0.4*IAA + 0.4*(1-reward_stdev) + 0.2*coverage"""
    iaa_val = max(0.0, iaa) if iaa is not None else 0.0

    rewards = [r for fb in feedbacks if (r := _absolute_reward(fb)) is not None]
    if len(rewards) >= 2:
        mean = sum(rewards) / len(rewards)
        stdev = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    else:
        stdev = 0.0

    non_flagged = [fb for fb in feedbacks if not fb.flagged]
    coverage = min(1.0, len(non_flagged) / max(1, min_annotations))

    return round(0.4 * iaa_val + 0.4 * (1.0 - min(1.0, stdev)) + 0.2 * coverage, 4)


async def load_reliability(db: AsyncSession, feedbacks: list[FeedbackItem]) -> dict[str, float]:
    """Map annotator_id -> reliability_score for the annotators in `feedbacks`."""
    ids = {fb.annotator_id for fb in feedbacks}
    if not ids:
        return {}
    result = await db.execute(
        select(Annotator.id, Annotator.reliability_score).where(Annotator.id.in_(ids))
    )
    return {aid: score for aid, score in result.all() if score is not None}


# --- Annotator reliability -----------------------------------------------------

# Pseudo-observations at full reliability: a new annotator is not judged on one task.
RELIABILITY_PRIOR_WEIGHT = 3.0


def pairwise_agreement(a: FeedbackItem, b: FeedbackItem) -> float | None:
    """Agreement in [0, 1] between two feedback items on the same task, or None
    if they are not comparable (different feedback types or ranking lengths)."""
    if a.ranking and b.ranking and len(a.ranking) == len(b.ranking) >= 2:
        pairs = list(combinations(range(len(a.ranking)), 2))
        same = sum(
            1 for i, j in pairs
            if (a.ranking[i] < a.ranking[j]) == (b.ranking[i] < b.ranking[j])
        )
        return same / len(pairs)
    ra, rb = _absolute_reward(a), _absolute_reward(b)
    if ra is not None and rb is not None:
        return max(0.0, 1.0 - abs(ra - rb))
    return None


def annotator_agreement(feedbacks: list[FeedbackItem], annotator_id: str) -> list[float]:
    """One agreement sample per task the annotator worked on.

    A sample is the annotator's mean agreement with the other (unflagged) annotators
    on that task. Feedback that was flagged counts as 0. Tasks with nothing to compare
    against are skipped.
    """
    by_task: dict[str, list[FeedbackItem]] = {}
    for fb in feedbacks:
        by_task.setdefault(fb.task_id, []).append(fb)

    samples: list[float] = []
    for items in by_task.values():
        mine = [fb for fb in items if fb.annotator_id == annotator_id]
        if not mine:
            continue
        if mine[0].flagged:
            samples.append(0.0)
            continue
        others = [fb for fb in items if fb.annotator_id != annotator_id and not fb.flagged]
        scores = [s for o in others if (s := pairwise_agreement(mine[0], o)) is not None]
        if scores:
            samples.append(sum(scores) / len(scores))
    return samples


def reliability_from_samples(samples: list[float]) -> float:
    """Mean agreement shrunk toward 1.0, so reliability moves as evidence builds up."""
    return (RELIABILITY_PRIOR_WEIGHT + sum(samples)) / (RELIABILITY_PRIOR_WEIGHT + len(samples))


async def update_reliability(db: AsyncSession, annotator_ids: set[str]) -> None:
    """Recompute reliability_score and avg_agreement_rate for these annotators
    from every task they have worked on."""
    if not annotator_ids:
        return
    task_ids = select(FeedbackItem.task_id).where(FeedbackItem.annotator_id.in_(annotator_ids))
    result = await db.execute(select(FeedbackItem).where(FeedbackItem.task_id.in_(task_ids)))
    feedbacks = list(result.scalars().all())

    annotators = await db.execute(select(Annotator).where(Annotator.id.in_(annotator_ids)))
    for annotator in annotators.scalars():
        samples = annotator_agreement(feedbacks, annotator.id)
        annotator.reliability_score = round(reliability_from_samples(samples), 4)
        annotator.avg_agreement_rate = round(sum(samples) / len(samples), 4) if samples else None
