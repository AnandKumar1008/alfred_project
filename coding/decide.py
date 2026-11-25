"""Cost-sensitive message routing with human review escalation.

Approach: expected_costs computes E = probs @ cost. decide picks each row's
min-cost queue (lowest index on ties) as its risk and routes up to max_review
rows with risk >= threshold to REVIEW (-1), highest risk first (lowest row on
ties). choose_threshold sweeps inf and sorted unique risks using prefix sums of
(review_cost - queue_cost) in O(N log N).
Why threshold ≈ review_cost: With calibrated probs and no budget cap, reviewing
saves (risk - review_cost) in expectation, so optimal threshold equals
review_cost. Optimal T << review_cost signals overconfidence (risk underestimated);
T >> review_cost signals underconfidence or a binding review budget.
"""

from __future__ import annotations

import math

import numpy as np

REVIEW = -1  # send to a person


def _validate_probs(probs: np.ndarray) -> np.ndarray:
    """Validate (N, C) probability array per C2 rules."""
    arr = np.asarray(probs)
    if arr.ndim != 2 or arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError("probs must be a numeric 2-D array.")
    arr = arr.astype(np.float64, copy=False)
    n, c = arr.shape
    if n < 1 or c < 2:
        raise ValueError(f"probs must have shape (N >= 1, C >= 2); got {arr.shape}.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("probs must be finite and non-negative.")
    if np.any(np.abs(arr.sum(axis=1) - 1.0) > 1e-6):
        raise ValueError("Each row of probs must sum to 1 within 1e-6.")
    return arr


def _validate_cost(cost: np.ndarray, c: int | None = None) -> np.ndarray:
    """Validate square (C, C) finite non-negative cost matrix."""
    arr = np.asarray(cost)
    if arr.ndim != 2 or arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError("cost must be a numeric 2-D array.")
    arr = arr.astype(np.float64, copy=False)
    rows, cols = arr.shape
    if rows < 2 or rows != cols:
        raise ValueError(f"cost must be square (C >= 2, C); got {arr.shape}.")
    if c is not None and rows != c:
        raise ValueError(f"cost shape {arr.shape} does not match C={c} classes.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("cost must be finite and non-negative.")
    return arr


def _validate_int_1d(arr_in: np.ndarray, name: str, n: int | None = None) -> np.ndarray:
    """Validate 1-D non-empty integer array."""
    arr = np.asarray(arr_in)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array.")
    if n is not None and len(arr) != n:
        raise ValueError(f"{name} must have length {n}; got {len(arr)}.")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must have integer dtype; got {arr.dtype}.")
    return arr.astype(np.intp, copy=False)


def expected_costs(probs: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Compute E[n, j] = sum_i probs[n, i] * cost[i, j]."""
    p = _validate_probs(probs)
    c_mat = _validate_cost(cost, p.shape[1])
    return p @ c_mat


def decide(
    probs: np.ndarray,
    cost: np.ndarray,
    threshold: float,
    max_review: int,
) -> np.ndarray:
    """Route messages to best expected-cost queue or REVIEW (-1) up to max_review."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)):
        raise ValueError(f"threshold must be a real number; got {threshold!r}.")
    thresh = float(threshold)
    if math.isnan(thresh):
        raise ValueError("threshold must not be NaN.")
    if isinstance(max_review, bool) or not isinstance(max_review, (int, np.integer)) or max_review < 0:
        raise ValueError(f"max_review must be an integer >= 0; got {max_review!r}.")

    ec = expected_costs(probs, cost)
    best_queue = np.argmin(ec, axis=1)  # lowest column index on ties
    risk = ec[np.arange(len(ec)), best_queue]
    decisions = best_queue.astype(np.intp, copy=True)

    candidates = np.where(risk >= thresh)[0]
    if len(candidates) > 0 and max_review > 0:
        order = np.lexsort((candidates, -risk[candidates]))
        decisions[candidates[order[: int(max_review)]]] = REVIEW

    return decisions


def realised_cost(
    decisions: np.ndarray,
    labels: np.ndarray,
    cost: np.ndarray,
    review_cost: float,
) -> float:
    """Compute mean realised cost over decisions, using review_cost for REVIEW."""
    c_mat = _validate_cost(cost)
    c = c_mat.shape[0]
    dec = _validate_int_1d(decisions, "decisions")
    lbl = _validate_int_1d(labels, "labels", len(dec))

    if np.any((lbl < 0) | (lbl >= c)):
        raise ValueError(f"labels must be in [0, {c}).")
    if np.any((dec < REVIEW) | (dec >= c)):
        raise ValueError(f"decisions must be in [{REVIEW}, {c}).")
    if isinstance(review_cost, bool) or not isinstance(review_cost, (int, float, np.number)):
        raise ValueError(f"review_cost must be a finite non-negative number; got {review_cost!r}.")
    rc = float(review_cost)
    if not math.isfinite(rc) or rc < 0.0:
        raise ValueError(f"review_cost must be finite and >= 0; got {review_cost!r}.")

    is_review = dec == REVIEW
    non_review = ~is_review
    total = float(np.sum(is_review)) * rc
    if np.any(non_review):
        total += float(np.sum(c_mat[lbl[non_review], dec[non_review]]))
    return total / len(dec)


def choose_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    cost: np.ndarray,
    review_cost: float,
    max_review_fraction: float,
) -> float:
    """Pick threshold in {inf} U unique_risks minimizing realised_cost (largest on ties)."""
    p = _validate_probs(probs)
    n, c = p.shape
    c_mat = _validate_cost(cost, c)
    lbl = _validate_int_1d(labels, "labels", n)
    if np.any((lbl < 0) | (lbl >= c)):
        raise ValueError(f"labels must be in [0, {c}).")

    if isinstance(review_cost, bool) or not isinstance(review_cost, (int, float, np.number)):
        raise ValueError(f"review_cost must be a finite non-negative number; got {review_cost!r}.")
    rc = float(review_cost)
    if not math.isfinite(rc) or rc < 0.0:
        raise ValueError(f"review_cost must be finite and >= 0; got {review_cost!r}.")

    if isinstance(max_review_fraction, bool) or not isinstance(max_review_fraction, (int, float, np.number)):
        raise ValueError(f"max_review_fraction must be in [0, 1]; got {max_review_fraction!r}.")
    mrf = float(max_review_fraction)
    if not math.isfinite(mrf) or mrf < 0.0 or mrf > 1.0:
        raise ValueError(f"max_review_fraction must be finite and in [0, 1]; got {max_review_fraction!r}.")

    max_review = math.floor(mrf * n)
    ec = p @ c_mat
    best_queue = np.argmin(ec, axis=1)
    risk = ec[np.arange(n), best_queue]
    queue_cost = c_mat[lbl, best_queue]

    order = np.argsort(-risk, kind="stable")
    sorted_risk = risk[order]
    sorted_qcost = queue_cost[order]

    total_qcost = float(np.sum(queue_cost))
    cum_deltas = np.cumsum(rc - sorted_qcost)
    cum_padded = np.empty(n + 1, dtype=np.float64)
    cum_padded[0] = 0.0
    cum_padded[1:] = cum_deltas

    unique_risks = np.unique(risk)  # ascending
    ks = np.searchsorted(-sorted_risk, -unique_risks, side="right")
    ks = np.clip(ks, 0, max_review)

    ks_all = np.empty(len(unique_risks) + 1, dtype=np.intp)
    ks_all[0] = 0
    ks_all[1:] = ks

    mean_costs = (total_qcost + cum_padded[ks_all]) / n
    min_cost = float(np.min(mean_costs))
    near_best = np.where(mean_costs <= min_cost + 1e-12)[0]

    if 0 in near_best:
        return math.inf
    return float(unique_risks[int(near_best[-1]) - 1])
