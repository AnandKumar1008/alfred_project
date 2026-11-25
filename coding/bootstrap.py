"""Bootstrap confidence intervals with exact-test coverage floors.

Approach: metric_ci computes accuracy or per-class recall over m relevant rows
and bootstraps via default_rng(seed). paired_diff_ci bootstraps x_A - x_B on
the same rows. Exact Wilson (single metric) and McNemar-style (paired diff)
floors widen intervals that are narrower than discrete data supports.
Edge cases: Empty/unequal/non-1D inputs, unknown metric, cls on accuracy, missing
cls on recall, empty cls, level outside (0,1), n_resamples < 100, invalid k/n
all raise ValueError. Significant is False whenever the floor contains 0.
Limits: Wilson/McNemar assume i.i.d. test items (not grouped/clustered data).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float           # reported, after the floor
    high: float
    boot_low: float      # raw percentile bootstrap
    boot_high: float
    level: float
    n_items: int
    widened: bool        # the floor moved low or high
    significant: bool    # low > 0 or high < 0 (paired); False in metric_ci


def _inv_norm_cdf(level: float) -> float:
    """Compute z = Phi^-1(1 - (1 - level) / 2) to machine precision using stdlib erf."""
    p = (1.0 + level) / 2.0
    # Initial guess via Abramowitz & Stegun 26.2.23
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)

    # Newton-Raphson refinement on Phi(z) - p = 0
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    for _ in range(5):
        cdf = 0.5 * (1.0 + math.erf(z * inv_sqrt2))
        pdf = inv_sqrt2pi * math.exp(-0.5 * z * z)
        z -= (cdf - p) / pdf
    return z


def _validate_level(level: float) -> float:
    """Validate confidence level strictly in (0, 1)."""
    if isinstance(level, bool) or not isinstance(level, (int, float, np.number)):
        raise ValueError(f"level must be a float in (0, 1); got {level!r}.")
    lv = float(level)
    if not math.isfinite(lv) or lv <= 0.0 or lv >= 1.0:
        raise ValueError(f"level must be in (0, 1); got {level!r}.")
    return lv


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Return exact-score Wilson confidence interval (low, high) for k/n."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer; got {k!r}.")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n must be an integer; got {n!r}.")
    k_i = int(k)
    n_i = int(n)
    if n_i < 1:
        raise ValueError(f"n must be >= 1; got {n_i}.")
    if k_i < 0 or k_i > n_i:
        raise ValueError(f"k must satisfy 0 <= k <= n; got k={k_i}, n={n_i}.")
    lv = _validate_level(level)

    z = _inv_norm_cdf(lv)
    z2 = z * z
    p_hat = k_i / n_i
    denom = 1.0 + z2 / n_i
    centre = (p_hat + z2 / (2.0 * n_i)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n_i + z2 / (4.0 * n_i * n_i))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _validate_eval_args(
    y_true: np.ndarray,
    preds: tuple[np.ndarray, ...],
    metric: str,
    cls: int | None,
    level: float,
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Validate inputs shared by metric_ci and paired_diff_ci."""
    yt = np.asarray(y_true)
    if yt.ndim != 1 or len(yt) == 0:
        raise ValueError("y_true must be a non-empty 1-D array.")
    n = len(yt)

    pred_arrs: list[np.ndarray] = []
    for i, p in enumerate(preds):
        pa = np.asarray(p)
        if pa.ndim != 1 or len(pa) != n:
            raise ValueError(f"Prediction array {i} must be 1-D of length {n}; got shape {pa.shape}.")
        pred_arrs.append(pa)

    if metric not in ("accuracy", "recall"):
        raise ValueError(f"Unknown metric {metric!r}; expected 'accuracy' or 'recall'.")
    if metric == "recall":
        if cls is None or isinstance(cls, bool) or not isinstance(cls, (int, np.integer)):
            raise ValueError(f"recall requires integer cls; got {cls!r}.")
    elif cls is not None:
        raise ValueError("cls must be None when metric='accuracy'.")

    lv = _validate_level(level)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 100:
        raise ValueError(f"n_resamples must be an integer >= 100; got {n_resamples!r}.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer; got {seed!r}.")
    return yt, pred_arrs, lv


def _correctness_vector(
    y_true: np.ndarray, y_pred: np.ndarray, metric: str, cls: int | None
) -> np.ndarray:
    """Return binary 1/0 correctness indicator over relevant rows."""
    correct = (y_pred == y_true).astype(np.float64)
    if metric == "accuracy":
        return correct
    mask = y_true == int(cls)  # type: ignore[arg-type]
    if not np.any(mask):
        raise ValueError(f"No rows with true label cls={cls}.")
    return correct[mask]


def metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "accuracy",
    cls: int | None = None,
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Interval:
    """Compute bootstrap confidence interval with Wilson score floor for one model."""
    yt, [yp], lv = _validate_eval_args(y_true, (y_pred,), metric, cls, level, n_resamples, seed)
    x = _correctness_vector(yt, yp, metric, cls)
    m = len(x)
    estimate = float(np.mean(x))

    alpha = (1.0 - lv) / 2.0
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, m, size=(int(n_resamples), m))
    stats = x[idx].mean(axis=1)
    boot_low, boot_high = (float(v) for v in np.quantile(stats, [alpha, 1.0 - alpha]))

    floor_low, floor_high = wilson_interval(int(np.sum(x)), m, lv)
    low = min(boot_low, floor_low)
    high = max(boot_high, floor_high)
    widened = (low < boot_low) or (high > boot_high)

    return Interval(
        estimate=estimate,
        low=low,
        high=high,
        boot_low=boot_low,
        boot_high=boot_high,
        level=lv,
        n_items=m,
        widened=widened,
        significant=False,
    )


def paired_diff_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str = "accuracy",
    cls: int | None = None,
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Interval:
    """Compute paired bootstrap CI for (A - B) floored by a McNemar-style interval."""
    yt, [pa, pb], lv = _validate_eval_args(
        y_true, (pred_a, pred_b), metric, cls, level, n_resamples, seed
    )
    x_a = _correctness_vector(yt, pa, metric, cls)
    x_b = _correctness_vector(yt, pb, metric, cls)
    x = x_a - x_b
    m = len(x)
    estimate = float(np.mean(x))

    alpha = (1.0 - lv) / 2.0
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, m, size=(int(n_resamples), m))
    stats = x[idx].mean(axis=1)
    boot_low, boot_high = (float(v) for v in np.quantile(stats, [alpha, 1.0 - alpha]))

    only_a = (x_a == 1.0) & (x_b == 0.0)
    only_b = (x_a == 0.0) & (x_b == 1.0)
    b_cnt = int(np.sum(only_a))
    d_cnt = b_cnt + int(np.sum(only_b))

    if d_cnt > 0:
        ql, qh = wilson_interval(b_cnt, d_cnt, lv)
        floor_low = (2.0 * ql - 1.0) * d_cnt / m
        floor_high = (2.0 * qh - 1.0) * d_cnt / m
    else:
        z = _inv_norm_cdf(lv)
        z2 = z * z
        u = z2 / (m + z2)
        floor_low = -u
        floor_high = u

    low = min(boot_low, floor_low)
    high = max(boot_high, floor_high)
    widened = (low < boot_low) or (high > boot_high)
    floor_has_zero = floor_low <= 0.0 <= floor_high
    significant = (not floor_has_zero) and ((low > 0.0) or (high < 0.0))

    return Interval(
        estimate=estimate,
        low=low,
        high=high,
        boot_low=boot_low,
        boot_high=boot_high,
        level=lv,
        n_items=m,
        widened=widened,
        significant=significant,
    )
