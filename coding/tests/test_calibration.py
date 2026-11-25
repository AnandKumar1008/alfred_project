"""Tests for coding.calibration."""

import numpy as np
import pytest

from coding.calibration import (
    ReliabilityBin,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
)


def test_public_examples():
    probs = np.array([
        [0.95, 0.03, 0.02],
        [0.10, 0.85, 0.05],
        [0.55, 0.30, 0.15],
        [0.20, 0.72, 0.08],
        [0.38, 0.33, 0.29],
    ])
    labels = np.array([0, 1, 1, 1, 2])
    bins = reliability_bins(probs, labels, n_bins=5)
    assert [b.count for b in bins] == [0, 1, 1, 1, 2]
    assert expected_calibration_error(probs, labels, n_bins=5) == pytest.approx(0.282, abs=1e-12)

    # Float arithmetic rule: ceil(0.55 * 100) - 1 == 55
    edge_bins = reliability_bins(np.array([[0.55, 0.45]]), np.array([0]), n_bins=100)
    assert edge_bins[55].count == 1
    assert edge_bins[54].count == 0


def test_bin_structure_empty_defaults_and_tie_breaking():
    probs = np.array([[0.5, 0.5], [1.0, 0.0]])
    labels = np.array([0, 0])  # tie goes to lowest column index (0) -> both right
    bins = reliability_bins(probs, labels, n_bins=4)
    assert len(bins) == 4
    assert all(isinstance(b, ReliabilityBin) for b in bins)
    assert bins[1].count == 1 and bins[1].accuracy == 1.0 and bins[1].mean_confidence == 0.5
    assert bins[3].count == 1 and bins[3].accuracy == 1.0
    for empty in (bins[0], bins[2]):
        assert empty.count == 0 and empty.mean_confidence == 0.0 and empty.accuracy == 0.0
    for idx, b in enumerate(bins):
        assert b.lower == pytest.approx(idx / 4)
        assert b.upper == pytest.approx((idx + 1) / 4)


def test_apply_temperature_stability_and_scipy_match():
    scipy_special = pytest.importorskip("scipy.special")
    logits = np.array([[10000.0, 9999.0, -10000.0], [-10000.0, -10000.5, -10002.0]])
    probs = apply_temperature(logits, temperature=0.05)
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)

    rng = np.random.default_rng(7)
    z = rng.normal(size=(20, 5))
    for t in (0.1, 1.0, 4.5):
        np.testing.assert_allclose(apply_temperature(z, t), scipy_special.softmax(z / t, axis=1), atol=1e-12)


def test_fit_temperature_constant_bounds_and_accuracy_invariance():
    # Constant logits rows -> 1.0
    const_logits = np.full((8, 3), 2.5)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    assert fit_temperature(const_logits, labels) == 1.0

    # Separable overconfident logits -> bound t_min when optimal T <= t_min
    sep_logits = np.array([[10.0, 0.0], [0.0, 10.0], [8.0, 0.0], [0.0, 8.0]])
    sep_labels = np.array([0, 1, 0, 1])
    assert fit_temperature(sep_logits, sep_labels, t_min=0.5, t_max=5.0) == pytest.approx(0.5, abs=1e-9)

    # Completely wrong confident logits -> bound t_max
    wrong_labels = np.array([1, 0, 1, 0])
    assert fit_temperature(sep_logits, wrong_labels, t_min=0.5, t_max=5.0) == pytest.approx(5.0, abs=1e-9)

    # Accuracy invariance and <= 1% of fine grid search
    rng = np.random.default_rng(123)
    z = rng.normal(size=(200, 4)) * 2.5
    true_probs = apply_temperature(z, 2.5)
    y = np.array([rng.choice(4, p=p) for p in true_probs])

    t_fit = fit_temperature(z, y, t_min=0.05, t_max=20.0)
    grid = np.geomspace(0.05, 20.0, 5000)
    nlls = []
    for t in grid:
        p = apply_temperature(z, float(t))
        nlls.append(-np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-300))))
    t_grid = float(grid[int(np.argmin(nlls))])
    assert abs(t_fit - t_grid) / t_grid < 0.01

    np.testing.assert_array_equal(np.argmax(z, axis=1), np.argmax(apply_temperature(z, t_fit), axis=1))


def test_invalid_inputs_raise_value_error():
    good_p = np.array([[0.7, 0.3], [0.2, 0.8]])
    good_y = np.array([0, 1])
    good_z = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError):
        reliability_bins(np.array([[1.0]]), np.array([0]))  # C < 2
    with pytest.raises(ValueError):
        reliability_bins(np.empty((0, 2)), np.empty(0, dtype=int))
    with pytest.raises(ValueError):
        reliability_bins(np.array([[1.1, -0.1]]), np.array([0]))
    with pytest.raises(ValueError):
        reliability_bins(np.array([[0.6, 0.3]]), np.array([0]))  # sum != 1
    with pytest.raises(ValueError):
        reliability_bins(good_p, np.array([0, 2]))  # label out of [0, C)
    with pytest.raises(ValueError):
        reliability_bins(good_p, np.array([0.0, 1.0]))  # non-int labels
    with pytest.raises(ValueError):
        reliability_bins(good_p, good_y, n_bins=0)
    with pytest.raises(ValueError):
        apply_temperature(np.array([[np.nan, 0.0]]), 1.0)
    with pytest.raises(ValueError):
        apply_temperature(good_z, 0.0)
    with pytest.raises(ValueError):
        apply_temperature(good_z, -1.0)
    with pytest.raises(ValueError):
        fit_temperature(good_z, good_y, t_min=0.0, t_max=5.0)
    with pytest.raises(ValueError):
        fit_temperature(good_z, good_y, t_min=5.0, t_max=5.0)
