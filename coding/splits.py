"""Group-aware stratified k-fold splitting.

Approach: Check upfront feasibility (classes in <k groups with share > tol fail
immediately), then run seeded restarts of count-deficit greedy assignment plus
local group move/swap repair to satisfy tolerance and min_fold_size = n//(2*k).
Edge cases: Empty/mismatched inputs, k<2, k>n_groups, invalid tolerance, bools,
and unresolvable splits raise ValueError naming the failing class and fold.
Limits: Exact partitioning is NP-hard; tight discrete group sizes may exhaust
restarts/time budget even when a combinatorial split exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class Fold:
    train: list[int]  # row indices, ascending
    test: list[int]


def _validate_inputs(
    labels: Sequence[str],
    groups: Sequence[str],
    k: int,
    seed: int,
    tolerance: float,
) -> None:
    """Validate arguments for group_stratified_kfold."""
    if isinstance(labels, (str, bytes)) or isinstance(groups, (str, bytes)):
        raise ValueError("labels and groups must be sequences of strings, not a single string.")
    n = len(labels)
    if n == 0:
        raise ValueError("labels and groups must not be empty.")
    if len(groups) != n:
        raise ValueError(f"labels and groups must have the same length; got {n} and {len(groups)}.")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError(f"k must be an integer >= 2; got {k!r}.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer; got {seed!r}.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float, np.number)):
        raise ValueError(f"tolerance must be a float in [0, 1]; got {tolerance!r}.")
    tol = float(tolerance)
    if not math.isfinite(tol) or tol < 0.0 or tol > 1.0:
        raise ValueError(f"tolerance must be finite and in [0, 1]; got {tolerance!r}.")


def _check_assignment(
    fold_counts: np.ndarray,
    fold_sizes: np.ndarray,
    overall_prop: np.ndarray,
    min_fold_size: int,
    tolerance: float,
    classes: list[str],
) -> tuple[bool, float, int, str | None]:
    """Check constraints; return (valid, max_dev, failing_fold, failing_class)."""
    k = len(fold_sizes)
    worst_dev = -1.0
    fail_fold = 0
    fail_class: str | None = None

    for f in range(k):
        fs = int(fold_sizes[f])
        if fs < min_fold_size:
            return False, math.inf, f, classes[0] if fs == 0 else None
        devs = np.abs(fold_counts[f] / fs - overall_prop)
        idx = int(np.argmax(devs))
        max_d = float(devs[idx])
        if max_d > worst_dev:
            worst_dev = max_d
            fail_fold = f
            fail_class = classes[idx]

    if worst_dev > tolerance + 1e-12:
        return False, worst_dev, fail_fold, fail_class
    return True, worst_dev, fail_fold, None


def _violation_score(
    fold_counts: np.ndarray,
    fold_sizes: np.ndarray,
    overall_prop: np.ndarray,
    min_fold_size: int,
    tolerance: float,
) -> float:
    """Compute scalar violation penalty across folds (0.0 iff valid)."""
    size_shortfall = np.maximum(0.0, min_fold_size - fold_sizes)
    score = float(np.sum(size_shortfall) * 1000.0)
    nonzero = fold_sizes > 0
    if np.any(nonzero):
        props = fold_counts[nonzero] / fold_sizes[nonzero, None]
        excess = np.maximum(0.0, np.abs(props - overall_prop) - tolerance)
        score += float(np.sum(excess * excess) + 0.05 * np.max(excess))
    return score


def _repair_assignment(
    assignment: np.ndarray,
    fold_counts: np.ndarray,
    fold_sizes: np.ndarray,
    group_profiles: np.ndarray,
    group_sizes: np.ndarray,
    overall_prop: np.ndarray,
    min_fold_size: int,
    tolerance: float,
    rng: np.random.Generator,
    max_steps: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily move or swap groups from violating folds to repair constraints."""
    k = len(fold_sizes)
    score = _violation_score(fold_counts, fold_sizes, overall_prop, min_fold_size, tolerance)
    if score == 0.0:
        return assignment, fold_counts, fold_sizes

    fold_groups: list[list[int]] = [[] for _ in range(k)]
    for gi, f in enumerate(assignment):
        fold_groups[int(f)].append(gi)

    for _ in range(max_steps):
        if score == 0.0:
            break

        # Identify the worst violating fold
        bad_f = 0
        bad_val = -1.0
        for f in range(k):
            fs = fold_sizes[f]
            if fs < min_fold_size:
                val = 1000.0 + (min_fold_size - fs)
            else:
                val = float(np.max(np.abs(fold_counts[f] / fs - overall_prop))) - tolerance
            if val > bad_val:
                bad_val = val
                bad_f = f

        if bad_val <= 1e-12:
            break

        best_move: tuple | None = None
        best_score = score - 1e-14

        # Candidate groups in bad_f and in other folds (subsample if very large)
        bad_list = fold_groups[bad_f]
        if len(bad_list) > 50:
            bad_cands = [bad_list[i] for i in rng.choice(len(bad_list), size=50, replace=False)]
        else:
            bad_cands = list(bad_list)

        other_cands: list[int] = []
        for f in range(k):
            if f == bad_f:
                continue
            flist = fold_groups[f]
            if len(flist) > 25:
                other_cands.extend(flist[i] for i in rng.choice(len(flist), size=25, replace=False))
            else:
                other_cands.extend(flist)

        # Try moving a group out of bad_f into another fold
        for gi in bad_cands:
            prof = group_profiles[gi]
            gs = group_sizes[gi]
            fold_counts[bad_f] -= prof
            fold_sizes[bad_f] -= gs
            for dst in range(k):
                if dst == bad_f:
                    continue
                fold_counts[dst] += prof
                fold_sizes[dst] += gs
                s = _violation_score(fold_counts, fold_sizes, overall_prop, min_fold_size, tolerance)
                if s < best_score:
                    best_score = s
                    best_move = ("move", gi, bad_f, dst)
                fold_counts[dst] -= prof
                fold_sizes[dst] -= gs
            fold_counts[bad_f] += prof
            fold_sizes[bad_f] += gs

        # Try moving a group from another fold into bad_f
        for gi in other_cands:
            src = int(assignment[gi])
            prof = group_profiles[gi]
            gs = group_sizes[gi]
            fold_counts[src] -= prof
            fold_sizes[src] -= gs
            fold_counts[bad_f] += prof
            fold_sizes[bad_f] += gs
            s = _violation_score(fold_counts, fold_sizes, overall_prop, min_fold_size, tolerance)
            if s < best_score:
                best_score = s
                best_move = ("move", gi, src, bad_f)
            fold_counts[src] += prof
            fold_sizes[src] += gs
            fold_counts[bad_f] -= prof
            fold_sizes[bad_f] -= gs

        # Try swapping a group in bad_f with a group in another fold
        if bad_cands and other_cands:
            for g1 in bad_cands[:25]:
                p1, s1 = group_profiles[g1], group_sizes[g1]
                for g2 in other_cands[:40]:
                    f2 = int(assignment[g2])
                    p2, s2 = group_profiles[g2], group_sizes[g2]
                    diff_p = p2 - p1
                    diff_s = s2 - s1
                    fold_counts[bad_f] += diff_p
                    fold_sizes[bad_f] += diff_s
                    fold_counts[f2] -= diff_p
                    fold_sizes[f2] -= diff_s
                    s = _violation_score(fold_counts, fold_sizes, overall_prop, min_fold_size, tolerance)
                    if s < best_score:
                        best_score = s
                        best_move = ("swap", g1, bad_f, g2, f2)
                    fold_counts[bad_f] -= diff_p
                    fold_sizes[bad_f] -= diff_s
                    fold_counts[f2] += diff_p
                    fold_sizes[f2] += diff_s

        if best_move is None:
            break

        if best_move[0] == "move":
            _, gi, src, dst = best_move
            prof, gs = group_profiles[gi], group_sizes[gi]
            fold_counts[src] -= prof
            fold_sizes[src] -= gs
            fold_counts[dst] += prof
            fold_sizes[dst] += gs
            assignment[gi] = dst
            fold_groups[src].remove(gi)
            fold_groups[dst].append(gi)
        else:
            _, g1, f1, g2, f2 = best_move
            diff_p = group_profiles[g2] - group_profiles[g1]
            diff_s = group_sizes[g2] - group_sizes[g1]
            fold_counts[f1] += diff_p
            fold_sizes[f1] += diff_s
            fold_counts[f2] -= diff_p
            fold_sizes[f2] -= diff_s
            assignment[g1] = f2
            assignment[g2] = f1
            fold_groups[f1].remove(g1)
            fold_groups[f1].append(g2)
            fold_groups[f2].remove(g2)
            fold_groups[f2].append(g1)

        score = best_score

    return assignment, fold_counts, fold_sizes


def group_stratified_kfold(
    labels: Sequence[str],
    groups: Sequence[str],
    k: int = 5,
    seed: int = 0,
    tolerance: float = 0.05,
) -> list[Fold]:
    """Return *k* Folds whose test sets are group-aware and stratified."""
    _validate_inputs(labels, groups, k, seed, tolerance)
    n = len(labels)
    k = int(k)
    tol = float(tolerance)

    group_indices: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        group_indices.setdefault(g, []).append(i)

    unique_groups = list(group_indices.keys())
    n_groups = len(unique_groups)
    if k > n_groups:
        raise ValueError(f"k ({k}) must not exceed the number of groups ({n_groups}).")

    class_counts = Counter(labels)
    classes = sorted(class_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    overall_prop = np.array([class_counts[c] / n for c in classes], dtype=np.float64)

    # Early feasibility check: a class in fewer than k groups with share > tolerance
    # will be absent from at least one fold, violating tolerance without searching.
    class_groups: dict[str, set[str]] = {c: set() for c in classes}
    for lbl, grp in zip(labels, groups):
        class_groups[lbl].add(grp)

    failing_early = [
        (c, len(class_groups[c]), overall_prop[ci])
        for ci, c in enumerate(classes)
        if len(class_groups[c]) < k and overall_prop[ci] > tol + 1e-12
    ]
    if failing_early:
        c0, ng0, share0 = failing_early[0]
        names = ", ".join(f"'{c}'" for c, _, _ in failing_early)
        raise ValueError(
            f"Impossible split for fold 0 (class(es) {names}): class '{c0}' appears in only "
            f"{ng0} group(s) (< k={k}), giving share 0.0 vs overall {share0:.4f} > tolerance={tol}."
        )

    group_profiles = np.zeros((n_groups, n_classes), dtype=np.float64)
    group_sizes = np.zeros(n_groups, dtype=np.float64)
    for gi, g in enumerate(unique_groups):
        idxs = group_indices[g]
        group_sizes[gi] = float(len(idxs))
        for i in idxs:
            group_profiles[gi, class_to_idx[labels[i]]] += 1.0

    min_fold_size = n // (2 * k)
    target_size = n / k
    target_counts = overall_prop * target_size
    rng = np.random.default_rng(int(seed))

    best_assignment: np.ndarray | None = None
    best_max_dev = math.inf
    best_fail_fold = 0
    best_fail_class: str | None = classes[0]

    # Rarity score per group for mixed ordering strategies
    total_counts_arr = np.array([class_counts[c] for c in classes], dtype=np.float64)
    group_rarity = np.max(group_profiles / np.maximum(total_counts_arr, 1.0), axis=1)

    t_start = time.perf_counter()
    max_attempts = 24
    for attempt in range(max_attempts):
        if attempt > 0 and time.perf_counter() - t_start > 1.5:
            break

        perm = rng.permutation(n_groups)
        fold_perm = rng.permutation(k)

        if attempt % 3 == 0:
            # Pure random group order for strong seed diversity
            order = perm
        elif attempt % 3 == 1:
            # Largest-first with random tie-break within perm
            order = perm[np.argsort(-group_sizes[perm], kind="stable")]
        else:
            # Rarest-class / highest-impact groups first
            order = perm[np.argsort(-group_rarity[perm], kind="stable")]

        diff_counts = -np.tile(target_counts, (k, 1))
        diff_sizes = np.full(k, -target_size, dtype=np.float64)
        assignment = np.empty(n_groups, dtype=np.int64)

        for gi in order:
            v = group_profiles[gi]
            s = group_sizes[gi]
            # Delta in sum_f (||F_f - target_counts||^2 + (S_f - target_size)^2)
            # Only linear term depends on fold f.
            deltas = diff_counts @ v + s * diff_sizes
            # Break ties using seeded fold permutation
            chosen_pos = int(np.argmin(deltas[fold_perm]))
            f = int(fold_perm[chosen_pos])
            assignment[gi] = f
            diff_counts[f] += v
            diff_sizes[f] += s

        fold_counts = diff_counts + target_counts
        fold_sizes = diff_sizes + target_size

        valid, max_dev, fail_fold, fail_class = _check_assignment(
            fold_counts, fold_sizes, overall_prop, min_fold_size, tol, classes
        )
        if not valid:
            assignment, fold_counts, fold_sizes = _repair_assignment(
                assignment,
                fold_counts,
                fold_sizes,
                group_profiles,
                group_sizes,
                overall_prop,
                min_fold_size,
                tol,
                rng,
            )
            valid, max_dev, fail_fold, fail_class = _check_assignment(
                fold_counts, fold_sizes, overall_prop, min_fold_size, tol, classes
            )

        if valid:
            fold_test_indices: list[list[int]] = [[] for _ in range(k)]
            for gi, f in enumerate(assignment):
                fold_test_indices[int(f)].extend(group_indices[unique_groups[gi]])

            folds: list[Fold] = []
            for f in range(k):
                test = sorted(fold_test_indices[f])
                test_set = set(test)
                train = [i for i in range(n) if i not in test_set]
                folds.append(Fold(train=train, test=test))
            return folds

        if max_dev < best_max_dev or best_assignment is None:
            best_max_dev = max_dev
            best_assignment = assignment.copy()
            best_fail_fold = fail_fold
            best_fail_class = fail_class

    if best_fail_class is not None:
        raise ValueError(
            f"No split satisfies tolerance={tol} and min_fold_size={min_fold_size}: "
            f"class '{best_fail_class}' failed in fold {best_fail_fold} "
            f"(max deviation {best_max_dev:.4f})."
        )
    raise ValueError(
        f"No split satisfies min_fold_size={min_fold_size}: "
        f"fold {best_fail_fold} had too few rows."
    )
