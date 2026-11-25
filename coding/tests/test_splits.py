"""Tests for coding.splits.group_stratified_kfold."""

from collections import Counter
import time

import numpy as np
import pytest

from coding.splits import Fold, group_stratified_kfold

LABELS = [
    "request", "request", "complaint", "urgent", "request", "complaint",
    "request", "urgent", "request", "complaint", "request", "request",
]
GROUPS = ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4", "c5", "c5", "c6", "c6"]


def test_partition_and_train_complement():
    folds = group_stratified_kfold(LABELS, GROUPS, k=2, seed=0, tolerance=0.2)
    assert len(folds) == 2
    all_test = [i for f in folds for i in f.test]
    assert sorted(all_test) == list(range(len(LABELS)))
    for f in folds:
        assert isinstance(f, Fold)
        assert f.train == sorted(f.train)
        assert f.test == sorted(f.test)
        assert set(f.train).isdisjoint(f.test)
        assert sorted(f.train + f.test) == list(range(len(LABELS)))
        assert all(type(i) is int for i in f.train + f.test)


def test_group_integrity_and_mixed_groups():
    folds = group_stratified_kfold(LABELS, GROUPS, k=3, seed=0, tolerance=0.35)
    for f in folds:
        groups_in_fold = {GROUPS[i] for i in f.test}
        for g in groups_in_fold:
            g_rows = {i for i, x in enumerate(GROUPS) if x == g}
            assert g_rows <= set(f.test)


def test_stratification_and_min_fold_size():
    folds = group_stratified_kfold(LABELS, GROUPS, k=2, seed=0, tolerance=0.2)
    overall = Counter(LABELS)
    n = len(LABELS)
    min_size = n // (2 * 2)
    for f in folds:
        assert len(f.test) >= min_size
        counts = Counter(LABELS[i] for i in f.test)
        for cls in overall:
            assert abs(counts[cls] / len(f.test) - overall[cls] / n) <= 0.2 + 1e-12


def test_exact_zero_tolerance_when_feasible():
    labels = ["A", "B"] * 6
    groups = [f"g{i // 2}" for i in range(12)]
    folds = group_stratified_kfold(labels, groups, k=3, seed=5, tolerance=0.0)
    for f in folds:
        c = Counter(labels[i] for i in f.test)
        assert c["A"] == c["B"]


def test_reproducibility_and_seed_diversity():
    f0_a = group_stratified_kfold(LABELS, GROUPS, k=2, seed=42, tolerance=0.2)
    f0_b = group_stratified_kfold(LABELS, GROUPS, k=2, seed=42, tolerance=0.2)
    assert f0_a == f0_b
    results = [group_stratified_kfold(LABELS, GROUPS, k=2, seed=s, tolerance=0.2) for s in range(1, 15)]
    assert any(r != f0_a for r in results)


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        group_stratified_kfold([], [], k=2)
    with pytest.raises(ValueError):
        group_stratified_kfold(["a", "b"], ["g1"], k=2)
    with pytest.raises(ValueError):
        group_stratified_kfold(LABELS, GROUPS, k=1)
    with pytest.raises(ValueError):
        group_stratified_kfold(LABELS, GROUPS, k=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        group_stratified_kfold(LABELS, GROUPS, k=7)
    with pytest.raises(ValueError):
        group_stratified_kfold(LABELS, GROUPS, k=2, tolerance=-0.01)
    with pytest.raises(ValueError):
        group_stratified_kfold(LABELS, GROUPS, k=2, tolerance=1.01)
    with pytest.raises(ValueError):
        group_stratified_kfold(LABELS, GROUPS, k=2, tolerance=float("nan"))


def test_early_impossible_class_raises_without_searching():
    # Class C only appears in 1 group with share 4/24 = 0.167 > tolerance 0.05;
    # A and B each appear in >= 3 groups.
    labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 4
    groups = (
        ["a1"] * 3 + ["a2"] * 3 + ["a3"] * 4
        + ["b1"] * 3 + ["b2"] * 3 + ["b3"] * 4
        + ["c1"] * 4
    )
    with pytest.raises(ValueError, match="C"):
        group_stratified_kfold(labels, groups, k=3, tolerance=0.05)


def test_impossible_split_names_class_and_fold():
    # All classes appear in >= k groups, yet discrete group sizes make tol=0.01 impossible.
    labels = ["A"] * 6 + ["B"] * 6
    groups = ["g1"] * 4 + ["g2"] * 2 + ["g3"] * 4 + ["g4"] * 2
    with pytest.raises(ValueError) as exc:
        group_stratified_kfold(labels, groups, k=3, tolerance=0.01)
    msg = str(exc.value).lower()
    assert "fold" in msg and ("class 'a'" in msg or "class 'b'" in msg)


def test_performance_10k_rows_3k_groups():
    rng = np.random.default_rng(0)
    n, n_groups = 10_000, 3_000
    labels = rng.choice(["request", "complaint", "urgent"], size=n, p=[0.7, 0.25, 0.05]).tolist()
    groups = [f"g{i % n_groups}" for i in range(n)]
    t0 = time.perf_counter()
    folds = group_stratified_kfold(labels, groups, k=5, seed=0, tolerance=0.05)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
    assert len(folds) == 5
    overall = Counter(labels)
    for f in folds:
        assert len(f.test) >= n // 10
        counts = Counter(labels[i] for i in f.test)
        for cls in overall:
            assert abs(counts[cls] / len(f.test) - overall[cls] / n) <= 0.05 + 1e-12
