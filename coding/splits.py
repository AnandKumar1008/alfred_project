from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class Fold:
    train: list[int]
    test: list[int]

def group_stratified_kfold(labels: Sequence[str], groups: Sequence[str], k: int = 5,
                           seed: int = 0, tolerance: float = 0.05) -> list[Fold]:
    y, g = list(labels), list(groups)
    if len(y) != len(g) or not y: raise ValueError("labels and groups must have equal, non-zero length")
    if k < 2: raise ValueError("k must be at least 2")
    if not 0 <= tolerance <= 1: raise ValueError("tolerance must be in [0, 1]")
    unique = list(dict.fromkeys(g))
    if k > len(unique): raise ValueError("more folds than groups")
    classes = sorted(set(y)); totals = Counter(y); n = len(y)
    group_rows = defaultdict(list)
    for i, (label, group) in enumerate(zip(y, g)): group_rows[group].append((i, label))
    # A class represented by fewer than k independent groups cannot appear in every test fold.
    for c in classes:
        represented = sum(any(lbl == c for _, lbl in rows) for rows in group_rows.values())
        if represented < k and totals[c] / n > tolerance:
            raise ValueError(f"class {c!r} occurs in only {represented} groups; cannot satisfy fold {k} tolerance {tolerance}")
    rng = np.random.default_rng(seed)
    vectors = {}
    for group, rows in group_rows.items():
        cnt = Counter(lbl for _, lbl in rows)
        vectors[group] = np.array([cnt[c] for c in classes], dtype=float)
    # Greedy multi-start assignment minimizes class-share error and size imbalance.
    best = None; best_score = float('inf'); diagnostic = None; diagnostic_score = float('inf')
    for restart in range(2):
        rarity = {z: float((vectors[z] / np.maximum(np.array([totals[c] for c in classes]), 1)).sum()) for z in unique}
        order = sorted(unique, key=lambda z: (-rarity[z], -len(group_rows[z]), rng.random()))
        counts = np.zeros((k, len(classes))); sizes = np.zeros(k); assignment = {}
        for position, group in enumerate(order):
            v = vectors[group]; m = len(group_rows[group]); candidates = []
            if position < k:
                f = position; assignment[group] = f; counts[f] += v; sizes[f] += m
                continue
            min_size = sizes.min()
            eligible = np.flatnonzero(sizes == min_size)
            for f in eligible:
                ss = sizes[f] + m
                share = (counts[f] + v) / ss
                target = np.array([totals[c] / n for c in classes])
                score = np.square(share - target).sum()
                candidates.append((score, sizes[f], int(f)))
            f = min(candidates)[2]; assignment[group] = f; counts[f] += v; sizes[f] += m
        # Every class share in each test fold must be within tolerance.
        shares = counts / sizes[:, None]
        target = np.array([totals[c] / n for c in classes])
        deviations = np.abs(shares - target)
        score = float(np.square(deviations).sum() + np.square((sizes - n/k)/(n/k)).sum())
        if score < diagnostic_score:
            diagnostic_score = score; diagnostic = (assignment.copy(), deviations.copy(), sizes.copy())
        if np.all(sizes >= n // (2*k)) and np.all(deviations <= tolerance + 1e-12) and score < best_score:
            best, best_score = assignment, score
    if best is None:
        _, bad, fold_sizes = diagnostic
        fold, ci = np.unravel_index(int(np.argmax(bad)), bad.shape)
        raise ValueError(f"class {classes[ci]!r} in fold {fold} differs from its overall share by {bad[fold, ci]:.4f} (tolerance={tolerance}); no feasible split found")
    folds = []
    for f in range(k):
        test = sorted(i for group, rows in group_rows.items() if best[group] == f for i, _ in rows)
        train = sorted(set(range(n)) - set(test))
        folds.append(Fold(train=train, test=test))
    return folds
