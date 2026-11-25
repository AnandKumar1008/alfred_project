"""Tests for coding.bootstrap and coding.evaluate_models."""

import subprocess
import sys
import time

import numpy as np
import pytest

from coding.bootstrap import Interval, metric_ci, paired_diff_ci, wilson_interval
from coding.demo_data import make_demo


def test_public_examples():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    iv = metric_ci(y, p)
    assert isinstance(iv, Interval)
    assert iv.estimate == pytest.approx(0.8)
    assert (iv.boot_low, iv.boot_high) == pytest.approx((0.5, 1.0))
    assert iv.low == pytest.approx(0.490, abs=1e-3)
    assert iv.high == pytest.approx(1.0)
    assert iv.widened is True
    assert iv.significant is False

    iv_diff = paired_diff_ci(
        np.full(5, 2), np.full(5, 2), np.array([0, 0, 0, 2, 2]), metric="recall", cls=2
    )
    assert (iv_diff.boot_low, iv_diff.boot_high) == pytest.approx((0.2, 1.0))
    assert iv_diff.low == pytest.approx(-0.074, abs=1e-3)
    assert iv_diff.high == pytest.approx(1.0)
    assert iv_diff.widened is True
    assert iv_diff.significant is False


def test_20_of_20_floor_and_scipy_wilson_match():
    scipy_stats = pytest.importorskip("scipy.stats")
    lo, hi = wilson_interval(20, 20, level=0.95)
    assert lo == pytest.approx(0.8389, abs=1e-3)
    assert hi == pytest.approx(1.0, abs=1e-12)

    # Compare wilson_interval against exact scipy norm.ppf
    z = float(scipy_stats.norm.ppf(0.975))
    k, n = 13, 25
    denom = 1 + z**2 / n
    exp_lo = (k / n + z**2 / (2 * n) - z * np.sqrt(k / n * (1 - k / n) / n + z**2 / (4 * n**2))) / denom
    exp_hi = (k / n + z**2 / (2 * n) + z * np.sqrt(k / n * (1 - k / n) / n + z**2 / (4 * n**2))) / denom
    got_lo, got_hi = wilson_interval(k, n, 0.95)
    assert got_lo == pytest.approx(exp_lo, abs=1e-12)
    assert got_hi == pytest.approx(exp_hi, abs=1e-12)

    y = np.zeros(20, dtype=int)
    iv = metric_ci(y, y)
    assert (iv.boot_low, iv.boot_high) == (1.0, 1.0)
    assert iv.low < 0.85 and iv.widened is True


def test_paired_diff_d_zero_floor_and_invariants():
    # Identical predictions -> D = 0 -> floor is (-u, u)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    iv = paired_diff_ci(y, y, y, metric="accuracy", level=0.95)
    assert iv.estimate == 0.0
    assert (iv.boot_low, iv.boot_high) == (0.0, 0.0)
    assert iv.low < 0.0 < iv.high
    assert iv.widened is True
    assert iv.significant is False

    # Randomized invariant check: interval never narrower than floor, never significant when floor contains 0
    rng = np.random.default_rng(123)
    for _ in range(30):
        m = int(rng.integers(10, 60))
        yt = rng.integers(0, 3, size=m)
        pa = rng.integers(0, 3, size=m)
        pb = rng.integers(0, 3, size=m)
        res = paired_diff_ci(yt, pa, pb, metric="accuracy", n_resamples=200, seed=0)
        assert res.low <= res.boot_low + 1e-12
        assert res.high >= res.boot_high - 1e-12
        only_a = ((pa == yt) & (pb != yt)).sum()
        only_b = ((pa != yt) & (pb == yt)).sum()
        d = int(only_a + only_b)
        if d > 0:
            ql, qh = wilson_interval(int(only_a), d, 0.95)
            fl, fh = (2 * ql - 1) * d / m, (2 * qh - 1) * d / m
        else:
            fl, fh = -1.0, 1.0
        assert res.low <= fl + 1e-12 and res.high >= fh - 1e-12
        if fl <= 0.0 <= fh:
            assert res.significant is False


def test_invalid_inputs_raise_value_error():
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 1, 1, 1])
    with pytest.raises(ValueError):
        wilson_interval(-1, 5)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(True, 5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        metric_ci(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        metric_ci(y, np.array([0, 1]))
    with pytest.raises(ValueError):
        metric_ci(y, p, metric="f1")
    with pytest.raises(ValueError):
        metric_ci(y, p, metric="recall", cls=None)
    with pytest.raises(ValueError):
        metric_ci(y, p, metric="recall", cls=2)  # no true members
    with pytest.raises(ValueError):
        metric_ci(y, p, metric="accuracy", cls=0)  # cls forbidden on accuracy
    with pytest.raises(ValueError):
        metric_ci(y, p, level=1.0)
    with pytest.raises(ValueError):
        metric_ci(y, p, level=0.0)
    with pytest.raises(ValueError):
        metric_ci(y, p, n_resamples=99)
    with pytest.raises(ValueError):
        paired_diff_ci(y, p, np.array([0, 1]))


def test_performance_10k_2k_and_evaluate_models_cli():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=10_000)
    a = rng.integers(0, 3, size=10_000)
    b = rng.integers(0, 3, size=10_000)
    t0 = time.perf_counter()
    paired_diff_ci(y, a, b, metric="accuracy", n_resamples=2000, seed=0)
    assert time.perf_counter() - t0 < 2.0

    y_d, pa_d, pb_d = make_demo(seed=0)
    assert len(y_d) == 400 and len(pa_d) == 400 and len(pb_d) == 400
    proc = subprocess.run(
        [sys.executable, "-m", "coding.evaluate_models", "--seed", "0"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ACCURACY" in proc.stdout and "URGENT RECALL" in proc.stdout
