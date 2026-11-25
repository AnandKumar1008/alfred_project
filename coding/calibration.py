"""Calibration metrics and temperature scaling for multi-class classifiers.

Approach: reliability_bins computes confidence p=max(row), prediction argmax
(lowest index on ties), and assigns bin b=min(n_bins-1, max(0, ceil(p*n_bins)-1))
in floating point. ECE weights |accuracy - mean_confidence| by count/N.
apply_temperature shifts logits/T by row-max before exp to prevent overflow at
extreme logits/T. fit_temperature checks NLL boundary gradients wrt beta=1/T,
then searches log(T) via golden-section using log-softmax NLL (no underflow).
Accuracy is unchanged because dividing by T>0 preserves logit ordering/argmax.
T must be fit on validation data, never training data, to avoid overfit bias.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float          # b / n_bins
    upper: float          # (b + 1) / n_bins
    count: int
    mean_confidence: float  # 0.0 when count == 0
    accuracy: float         # 0.0 when count == 0


def _validate_labels(labels: np.ndarray, n: int, n_classes: int) -> np.ndarray:
    """Validate 1-D integer label array of length n with values in [0, n_classes)."""
    arr = np.asarray(labels)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"labels must be 1-D of length {n}; got shape {arr.shape}.")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"labels must have integer dtype; got {arr.dtype}.")
    if np.any((arr < 0) | (arr >= n_classes)):
        raise ValueError(f"labels must be in [0, {n_classes}); found out-of-range value.")
    return arr.astype(np.intp, copy=False)


def _validate_probs(probs: np.ndarray) -> np.ndarray:
    """Validate (N, C) probability matrix."""
    arr = np.asarray(probs)
    if arr.ndim != 2 or arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError("probs must be a numeric 2-D array.")
    arr = arr.astype(np.float64, copy=False)
    n, c = arr.shape
    if n < 1 or c < 2:
        raise ValueError(f"probs must have shape (N >= 1, C >= 2); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("probs must contain only finite values.")
    if np.any(arr < 0.0):
        raise ValueError("probs must be non-negative.")
    row_sums = arr.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-6):
        raise ValueError("Each row of probs must sum to 1 within 1e-6.")
    return arr


def _validate_logits(logits: np.ndarray) -> np.ndarray:
    """Validate (N, C) finite logit matrix."""
    arr = np.asarray(logits)
    if arr.ndim != 2 or arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError("logits must be a numeric 2-D array.")
    arr = arr.astype(np.float64, copy=False)
    n, c = arr.shape
    if n < 1 or c < 2:
        raise ValueError(f"logits must have shape (N >= 1, C >= 2); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("logits must contain only finite values.")
    return arr


def _validate_n_bins(n_bins: int) -> int:
    """Validate n_bins is an integer >= 1."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be an integer >= 1; got {n_bins!r}.")
    return int(n_bins)


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Compute equal-width reliability bins over predicted confidences."""
    p_arr = _validate_probs(probs)
    n, c = p_arr.shape
    y_arr = _validate_labels(labels, n, c)
    nb = _validate_n_bins(n_bins)

    confidences = np.max(p_arr, axis=1)
    predictions = np.argmax(p_arr, axis=1)  # lowest column index on ties
    correct = (predictions == y_arr).astype(np.float64)

    # Exact floating-point formula per spec: min(n_bins - 1, max(0, ceil(p * n_bins) - 1))
    bin_indices = np.minimum(
        nb - 1,
        np.maximum(0, np.ceil(confidences * nb).astype(np.intp) - 1),
    )

    bins: list[ReliabilityBin] = []
    for b in range(nb):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            mean_conf = float(np.mean(confidences[mask]))
            acc = float(np.mean(correct[mask]))
        else:
            mean_conf = 0.0
            acc = 0.0
        bins.append(
            ReliabilityBin(
                lower=b / nb,
                upper=(b + 1) / nb,
                count=count,
                mean_confidence=mean_conf,
                accuracy=acc,
            )
        )
    return bins


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) over non-empty reliability bins."""
    bins = reliability_bins(probs, labels, n_bins)
    n = len(probs)
    return float(sum((b.count / n) * abs(b.accuracy - b.mean_confidence) for b in bins if b.count > 0))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return numerically stable row-wise softmax(logits / temperature)."""
    z = _validate_logits(logits)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float, np.number)):
        raise ValueError(f"temperature must be a positive finite number; got {temperature!r}.")
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"temperature must be finite and > 0; got {temperature!r}.")

    # Shift each row by its max before dividing? Careful: if z has max 10000 and min -10000,
    # subtracting row max first keeps differences <= 0 before dividing by small T.
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    scaled = z_shifted / t
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)


def _nll_and_grad_beta(z_shifted: np.ndarray, y_shifted: np.ndarray, beta: float) -> tuple[float, float]:
    """Compute mean NLL and d(NLL)/dbeta where beta = 1 / T, using shifted logits."""
    scaled = z_shifted * beta
    m = np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(scaled - m)
    sum_exp = np.sum(exp_s, axis=1, keepdims=True)
    probs = exp_s / sum_exp
    log_z = m[:, 0] + np.log(sum_exp[:, 0])
    nll = float(np.mean(log_z - y_shifted * beta))
    grad = float(np.mean(np.sum(probs * z_shifted, axis=1) - y_shifted))
    return nll, grad


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 20.0,
) -> float:
    """Find T in [t_min, t_max] minimizing mean negative log-likelihood on (logits, labels)."""
    z = _validate_logits(logits)
    n, c = z.shape
    y = _validate_labels(labels, n, c)

    if isinstance(t_min, bool) or isinstance(t_max, bool) or not (
        isinstance(t_min, (int, float, np.number)) and isinstance(t_max, (int, float, np.number))
    ):
        raise ValueError("t_min and t_max must be finite numbers.")
    lo = float(t_min)
    hi = float(t_max)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or lo >= hi:
        raise ValueError(f"Must have 0 < t_min < t_max; got t_min={t_min!r}, t_max={t_max!r}.")

    # Constant rows carry no relative logit signal; return 1.0 per spec.
    if np.all(z == z[:, :1]):
        return 1.0

    # Shift logits per row for overflow-free NLL & gradient evaluation.
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    y_shifted = z_shifted[np.arange(n), y]

    # NLL is convex in beta = 1 / T. Check boundary derivatives first so
    # boundary optima (including flat separable plateaus) return the exact bound.
    nll_at_hi, grad_at_hi = _nll_and_grad_beta(z_shifted, y_shifted, 1.0 / hi)
    if grad_at_hi >= 0.0:
        return hi

    nll_at_lo, grad_at_lo = _nll_and_grad_beta(z_shifted, y_shifted, 1.0 / lo)
    if grad_at_lo <= 0.0:
        return lo

    # Unimodal golden-section search in log(T) for scale-invariant precision.
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a = math.log(lo)
    b = math.log(hi)
    x1 = b - inv_phi * (b - a)
    x2 = a + inv_phi * (b - a)
    f1, _ = _nll_and_grad_beta(z_shifted, y_shifted, math.exp(-x1))
    f2, _ = _nll_and_grad_beta(z_shifted, y_shifted, math.exp(-x2))

    for _ in range(70):
        if abs(b - a) < 1e-10:
            break
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - inv_phi * (b - a)
            f1, _ = _nll_and_grad_beta(z_shifted, y_shifted, math.exp(-x1))
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + inv_phi * (b - a)
            f2, _ = _nll_and_grad_beta(z_shifted, y_shifted, math.exp(-x2))

    best_t = math.exp(0.5 * (a + b))
    best_t = min(hi, max(lo, best_t))
    nll_best, _ = _nll_and_grad_beta(z_shifted, y_shifted, 1.0 / best_t)

    if nll_at_lo <= nll_best and nll_at_lo <= nll_at_hi:
        return lo
    if nll_at_hi <= nll_best and nll_at_hi <= nll_at_lo:
        return hi
    return best_t
