from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.scoring import consensus_reward, fleiss_kappa, quality_score


def fb(annotator_id="a", ranking=None, scalar_reward=None, binary_label=None, flagged=False):
    return SimpleNamespace(
        annotator_id=annotator_id, ranking=ranking, scalar_reward=scalar_reward,
        binary_label=binary_label, flagged=flagged,
    )


# --- Fleiss' kappa -----------------------------------------------------------

def test_kappa_needs_two_raters():
    assert fleiss_kappa([[1, 2, 3]]) == 0.0


def test_kappa_perfect_agreement():
    assert fleiss_kappa([[1, 3, 2], [1, 3, 2], [1, 3, 2]]) == 1.0


def test_kappa_uses_observed_marginals():
    # Pairs (0,1): 3/0, (0,2): 2/1, (1,2): 0/3 -> P_bar = 7/9
    # p_first = 5/9 -> P_e = 41/81 -> kappa = (7/9 - 41/81) / (40/81) = 0.55
    assert fleiss_kappa([[1, 3, 2], [1, 3, 2], [2, 3, 1]]) == pytest.approx(0.55)


def test_kappa_total_disagreement_is_negative():
    assert fleiss_kappa([[1, 2], [2, 1]]) == pytest.approx(-1.0)


def test_kappa_ignores_rankings_of_wrong_length():
    assert fleiss_kappa([[1, 2, 3], [1, 2, 3], [1, 2]]) == 1.0


# --- Consensus reward --------------------------------------------------------

def test_ranking_reward_reflects_agreement_on_top_response():
    # Old bug: min(ranking) is always 1, so every ranking scored 1.0.
    agreed = consensus_reward([fb("a", [1, 2, 3]), fb("b", [1, 2, 3])])
    split = consensus_reward([fb("a", [1, 2, 3]), fb("b", [3, 2, 1])])
    assert agreed == pytest.approx(1.0)
    assert split == pytest.approx(0.5)


def test_consensus_is_reliability_weighted():
    feedbacks = [fb("good", scalar_reward=0.9), fb("bad", scalar_reward=0.1)]
    assert consensus_reward(feedbacks) == pytest.approx(0.5)
    weighted = consensus_reward(feedbacks, {"good": 0.9, "bad": 0.1})
    assert weighted == pytest.approx((0.9 * 0.9 + 0.1 * 0.1) / 1.0)


def test_consensus_binary_and_empty():
    assert consensus_reward([fb(binary_label=True), fb(binary_label=False)]) == 0.5
    assert consensus_reward([]) == 0.0
    assert consensus_reward([fb("a", scalar_reward=1.0)], {"a": 0.0}) == 0.0


# --- Quality score -----------------------------------------------------------

def test_quality_score_formula():
    feedbacks = [fb(scalar_reward=0.8), fb(scalar_reward=0.8)]
    # 0.4*1.0 + 0.4*(1-0) + 0.2*min(1, 2/2) = 1.0
    assert quality_score(1.0, feedbacks, 2) == 1.0
    # Negative IAA is clipped to 0; coverage 2/4
    assert quality_score(-0.5, feedbacks, 4) == pytest.approx(0.5)
