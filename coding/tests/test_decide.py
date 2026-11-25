"""Tests for coding.decide."""

import math
import time

import numpy as np
import pytest

from coding.decide import REVIEW, choose_threshold, decide, expected_costs, realised_cost

COST = np.array([[0.0, 1.0, 5.0], [2.0, 0.0, 5.0], [50.0, 30.0, 0.0]])
PROBS = np.array([
    [0.90, 0.08, 0.02],
    [0.10, 0.85, 0.05],
    [0.60, 0.30, 0.10],
    [0.20, 0.10, 0.70],
    [0.70, 0.26, 0.04],
])
LABELS = np.array([0, 1, 2, 2, 0])


def test_public_examples():
    d = decide(PROBS, COST, threshold=3.0, max_review=5)
    np.testing.assert_array_equal(d, np.array([0, 1, REVIEW, 2, 1]))
    assert realised_cost(d, LABELS, COST, review_cost=3.0) == pytest.approx(0.8, abs=1e-12)
    thr = choose_threshold(PROBS, LABELS, COST, review_cost=3.0, max_review_fraction=0.4)
    assert thr == pytest.approx(3.6, abs=1e-12)


def test_expected_costs_and_queue_tie_breaking():
    probs = np.array([[0.5, 0.5]])
    cost = np.array([[1.0, 3.0], [3.0, 1.0]])  # both queues have expected cost 2.0
    ec = expected_costs(probs, cost)
    np.testing.assert_allclose(ec, [[2.0, 2.0]])
    # Lowest queue index (0) wins on ties
    np.testing.assert_array_equal(decide(probs, cost, threshold=10.0, max_review=1), np.array([0]))


def test_review_budget_prioritizes_highest_risk_and_lower_row_on_ties():
    probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.1, 0.9]])  # risks: [2.0, 2.0, 0.4]
    cost = np.array([[0.0, 4.0], [4.0, 0.0]])
    # max_review=1 -> tie between row 0 and row 1 at risk=2.0 -> row 0 (lower index) wins
    d1 = decide(probs, cost, threshold=2.0, max_review=1)
    np.testing.assert_array_equal(d1, np.array([REVIEW, 0, 1]))
    # max_review=0 -> nothing reviewed
    d0 = decide(probs, cost, threshold=0.0, max_review=0)
    np.testing.assert_array_equal(d0, np.array([0, 0, 1]))
    # threshold=inf -> nothing reviewed
    d_inf = decide(probs, cost, threshold=math.inf, max_review=3)
    np.testing.assert_array_equal(d_inf, np.array([0, 0, 1]))


def test_choose_threshold_tie_breaking_picks_largest_and_handles_inf():
    # If review_cost is huge, never reviewing (threshold=inf) is strictly best or tied.
    thr = choose_threshold(PROBS, LABELS, COST, review_cost=100.0, max_review_fraction=1.0)
    assert thr == math.inf

    # When max_review_fraction=0.0, all thresholds give identical cost -> inf wins tie-break.
    thr0 = choose_threshold(PROBS, LABELS, COST, review_cost=1.0, max_review_fraction=0.0)
    assert thr0 == math.inf

    # Brute-force reference check on random problem
    rng = np.random.default_rng(11)
    raw = rng.uniform(size=(40, 3))
    p = raw / raw.sum(axis=1, keepdims=True)
    y = rng.integers(0, 3, size=40)
    thr_fast = choose_threshold(p, y, COST, review_cost=3.0, max_review_fraction=0.25)

    ec = expected_costs(p, COST)
    risks = ec[np.arange(40), np.argmin(ec, axis=1)]
    cands = sorted(set(risks.tolist()) | {math.inf}, reverse=True)
    costs = [realised_cost(decide(p, COST, t, 10), y, COST, 3.0) for t in cands]
    min_c = min(costs)
    expected_thr = max(t for t, c in zip(cands, costs) if c <= min_c + 1e-12)
    assert thr_fast == expected_thr


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        expected_costs(PROBS, np.array([[0.0, 1.0], [1.0, 0.0]]))  # shape mismatch
    with pytest.raises(ValueError):
        expected_costs(PROBS, np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        decide(PROBS, COST, threshold=float("nan"), max_review=1)
    with pytest.raises(ValueError):
        decide(PROBS, COST, threshold=1.0, max_review=-1)
    with pytest.raises(ValueError):
        realised_cost(np.array([0, 1, -2, 2, 1]), LABELS, COST, 3.0)
    with pytest.raises(ValueError):
        realised_cost(np.array([0, 1, 0, 2, 1]), LABELS, COST, -0.5)
    with pytest.raises(ValueError):
        choose_threshold(PROBS, LABELS, COST, review_cost=3.0, max_review_fraction=1.2)
    with pytest.raises(ValueError):
        choose_threshold(PROBS, LABELS, COST, review_cost=3.0, max_review_fraction=-0.1)


def test_performance_20k():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1.0, 1.0, 1.0], size=20_000)
    labels = rng.integers(0, 3, size=20_000)
    t0 = time.perf_counter()
    thr = choose_threshold(probs, labels, COST, review_cost=3.0, max_review_fraction=0.2)
    assert time.perf_counter() - t0 < 2.0
    assert math.isfinite(thr) or thr == math.inf
