"""Train and calibrate the BANKING77 customer message routing bundle.

Uses Part 1's group_stratified_kfold (over near-duplicate clusters), fit_temperature,
apply_temperature, and choose_threshold so calibration and threshold selection are
performed strictly on out-of-fold validation predictions without test leakage.

Usage:
    python -m project.train_model
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

# Ensure repository root is importable when run as script or module
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from coding.calibration import apply_temperature, expected_calibration_error, fit_temperature
from coding.decide import choose_threshold
from coding.splits import group_stratified_kfold
from project.download_data import download_data
from project.routing_model import (
    RoutingBundle,
    RoutingConfig,
    build_cost_matrix,
    build_model_pipeline,
    cluster_near_duplicates,
    compute_class_centroids,
    extract_logits,
    max_centroid_similarity,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BUNDLE_PATH = os.path.join(MODEL_DIR, "routing_bundle.joblib")


def load_csv(path: str) -> tuple[list[str], list[str]]:
    """Load (texts, categories) from a CSV file with columns text, category."""
    texts: list[str] = []
    labels: list[str] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["category"])
    return texts, labels


def train_and_save(
    model_name: str = "hybrid_logreg",
    c_reg: float = 10.0,
    seed: int = 42,
    save_path: str = BUNDLE_PATH,
) -> RoutingBundle:
    """Train final RoutingBundle with 5-fold group-stratified OOF calibration & routing."""
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        download_data()

    train_texts, train_cats = load_csv(train_path)
    classes = sorted(set(train_cats))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([cls_to_idx[c] for c in train_cats], dtype=np.int64)
    n_classes = len(classes)

    print(f"Loaded {len(train_texts)} training rows across {n_classes} categories.")

    # Cluster near-duplicates in train so validation folds never leak paraphrases
    t0 = time.perf_counter()
    comp_ids = cluster_near_duplicates(train_texts, sim_threshold=0.85)
    groups = [str(g) for g in comp_ids]
    print(
        f"Clustered near-duplicates (cos>=0.85): {len(set(groups))} groups "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    folds = group_stratified_kfold(
        train_cats, groups, k=5, seed=seed, tolerance=0.01
    )

    # Collect out-of-fold logits for honest calibration & threshold selection
    oof_logits = np.zeros((len(train_texts), n_classes), dtype=np.float64)
    oof_sims = np.zeros(len(train_texts), dtype=np.float64)

    t_cv = time.perf_counter()
    for fi, fold in enumerate(folds):
        pipe_f = build_model_pipeline(model_name, c_reg=c_reg)
        tr_texts = [train_texts[i] for i in fold.train]
        va_texts = [train_texts[i] for i in fold.test]
        pipe_f.fit(tr_texts, y_train[fold.train])
        oof_logits[fold.test] = extract_logits(pipe_f, va_texts)
        cents_f = compute_class_centroids(pipe_f, tr_texts, y_train[fold.train], n_classes)
        oof_sims[fold.test] = max_centroid_similarity(pipe_f, cents_f, va_texts)
        fold_acc = float(np.mean(np.argmax(oof_logits[fold.test], axis=1) == y_train[fold.test]))
        print(f"  Fold {fi}: train={len(fold.train)}, val={len(fold.test)}, acc={fold_acc:.4f}")

    oof_acc = float(np.mean(np.argmax(oof_logits, axis=1) == y_train))
    oof_probs_raw = apply_temperature(oof_logits, 1.0)
    ece_raw = expected_calibration_error(oof_probs_raw, y_train, n_bins=10)
    temperature = fit_temperature(oof_logits, y_train, t_min=0.05, t_max=20.0)
    oof_probs_cal = apply_temperature(oof_logits, temperature)
    ece_cal = expected_calibration_error(oof_probs_cal, y_train, n_bins=10)

    print(
        f"5-fold Group-Stratified OOF ({time.perf_counter() - t_cv:.1f}s): "
        f"acc={oof_acc:.4f}, T={temperature:.4f}, ECE {ece_raw:.4f} -> {ece_cal:.4f}"
    )

    # Choose cost-optimal review threshold on calibrated OOF probabilities
    cfg = RoutingConfig(ood_sim_threshold=float(np.quantile(oof_sims, 0.01)))
    cost_matrix = build_cost_matrix(classes, cfg)
    risk_threshold = choose_threshold(
        oof_probs_cal,
        y_train,
        cost_matrix,
        review_cost=cfg.review_cost,
        max_review_fraction=cfg.max_review_fraction,
    )
    print(
        f"OOF routing threshold={risk_threshold:.4f} (review_cost={cfg.review_cost}, "
        f"budget={cfg.max_review_fraction:.0%}), OOD sim 1st-pct={cfg.ood_sim_threshold:.4f}"
    )

    # Fit final model on 100% of training data
    t_final = time.perf_counter()
    final_pipe = build_model_pipeline(model_name, c_reg=c_reg)
    final_pipe.fit(train_texts, y_train)
    centroids = compute_class_centroids(final_pipe, train_texts, y_train, n_classes)
    print(f"Final model fit on all {len(train_texts)} rows in {time.perf_counter() - t_final:.1f}s")

    bundle = RoutingBundle(
        pipeline=final_pipe,
        classes=classes,
        temperature=temperature,
        cost_matrix=cost_matrix,
        risk_threshold=risk_threshold,
        centroids=centroids,
        config=cfg,
        metadata={
            "model_name": model_name,
            "c_reg": c_reg,
            "seed": seed,
            "oof_accuracy": oof_acc,
            "oof_ece_raw": ece_raw,
            "oof_ece_calibrated": ece_cal,
            "n_train": len(train_texts),
            "n_dup_clusters": len(set(groups)),
        },
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bundle.save(save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Saved RoutingBundle to {save_path} ({size_mb:.2f} MB)")
    return bundle


if __name__ == "__main__":
    train_and_save()
