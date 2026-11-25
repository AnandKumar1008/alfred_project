"""Comprehensive, honest evaluation for Part 2 (BANKING77 message routing).

Evaluates 5 candidate models on:
  - Test accuracy & urgent recall with Wilson-floored bootstrap 95% CIs (coding.bootstrap)
  - Paired statistical comparisons vs baseline (coding.bootstrap.paired_diff_ci)
  - Calibration before/after validation temperature scaling (coding.calibration)
  - Asymmetric cost-sensitive routing with human review budget (coding.decide)
  - Near-duplicate leakage vs novel-message generalization across train/test
  - Smallest-category CIs, label-shuffle sanity check, OOD/paraphrase probes, error dump

Usage:
    python -m project.evaluate
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from coding.bootstrap import Interval, metric_ci, paired_diff_ci
from coding.calibration import apply_temperature, expected_calibration_error, fit_temperature
from coding.decide import REVIEW, choose_threshold, decide, expected_costs, realised_cost
from coding.splits import group_stratified_kfold
from project.routing_model import (
    URGENT_SET,
    RoutingBundle,
    RoutingConfig,
    build_cost_matrix,
    build_model_pipeline,
    cluster_near_duplicates,
    extract_logits,
    max_centroid_similarity,
    normalize_text,
)
from project.train_model import BUNDLE_PATH, DATA_DIR, load_csv

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")


def _iv_dict(iv: Interval) -> dict[str, Any]:
    return {
        "estimate": round(iv.estimate, 4),
        "low": round(iv.low, 4),
        "high": round(iv.high, 4),
        "boot_low": round(iv.boot_low, 4),
        "boot_high": round(iv.boot_high, 4),
        "n_items": iv.n_items,
        "widened": iv.widened,
        "significant": iv.significant,
    }


def _find_test_train_near_dups(
    train_texts: list[str], test_texts: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (max_char_cosine_sim, nearest_train_idx) for every test message."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    x_tr = vec.fit_transform([normalize_text(t) for t in train_texts])
    x_te = vec.transform([normalize_text(t) for t in test_texts])
    sims = x_te @ x_tr.T
    best_idx = np.asarray(sims.argmax(axis=1)).ravel()
    max_sim = np.asarray(sims.max(axis=1).toarray()).ravel()
    return max_sim, best_idx


def main() -> None:
    train_texts, train_cats = load_csv(os.path.join(DATA_DIR, "train.csv"))
    test_texts, test_cats = load_csv(os.path.join(DATA_DIR, "test.csv"))

    classes = sorted(set(train_cats))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([cls_to_idx[c] for c in train_cats], dtype=np.int64)
    y_test = np.array([cls_to_idx[c] for c in test_cats], dtype=np.int64)
    is_urg_test = np.array([c in URGENT_SET for c in test_cats], dtype=int)

    # Load final trained bundle
    bundle: RoutingBundle = RoutingBundle.load(BUNDLE_PATH)
    cfg = bundle.config
    cost_matrix = bundle.cost_matrix

    # 1. Group-stratified split on train (fold 0 as shared validation split for fair model comparison)
    comp_ids = cluster_near_duplicates(train_texts, sim_threshold=0.85)
    folds = group_stratified_kfold(
        train_cats, [str(g) for g in comp_ids], k=5, seed=42, tolerance=0.01
    )
    tr_idx = np.array(folds[0].train)
    val_idx = np.array(folds[0].test)
    tr_texts_cv = [train_texts[i] for i in tr_idx]
    va_texts_cv = [train_texts[i] for i in val_idx]

    # 2. Near-duplicate analysis across train and test
    max_sim_te, nearest_tr = _find_test_train_near_dups(train_texts, test_texts)
    tr_norm_set = {normalize_text(t) for t in train_texts}
    exact_norm_matches = sum(1 for t in test_texts if normalize_text(t) in tr_norm_set)
    novel_85_mask = max_sim_te < 0.85
    dup_85_mask = ~novel_85_mask

    dup_summary = {
        "train_rows": len(train_texts),
        "train_dup_clusters_cos85": int(len(set(comp_ids))),
        "test_rows": len(test_texts),
        "test_exact_norm_matches": exact_norm_matches,
        "test_cos_ge_095": int((max_sim_te >= 0.95).sum()),
        "test_cos_ge_095_same_label": int(
            ((max_sim_te >= 0.95) & (y_test == y_train[nearest_tr])).sum()
        ),
        "test_cos_ge_090": int((max_sim_te >= 0.90).sum()),
        "test_cos_ge_090_same_label": int(
            ((max_sim_te >= 0.90) & (y_test == y_train[nearest_tr])).sum()
        ),
        "test_cos_ge_085": int(dup_85_mask.sum()),
        "test_cos_ge_085_same_label": int(
            (dup_85_mask & (y_test == y_train[nearest_tr])).sum()
        ),
        "test_novel_lt_085": int(novel_85_mask.sum()),
    }
    print("=== Near-Duplicate Analysis (Train vs Test) ===")
    print(json.dumps(dup_summary, indent=2))

    # 3. Benchmark 5 models on fold-0 train/val -> test
    candidates = [
        ("multinomial_nb", 1.0),
        ("word_logreg", 5.0),
        ("word_svc", 1.0),
        ("hybrid_svc", 0.8),
        ("hybrid_logreg", 10.0),
    ]
    model_rows: dict[str, dict[str, Any]] = {}
    model_preds_test: dict[str, np.ndarray] = {}
    model_urg_any_preds: dict[str, np.ndarray] = {}

    print("\n=== Model Comparison (80% Group-Stratified Train, Val-Calibrated -> Test) ===")
    for name, c_reg in candidates:
        pipe = build_model_pipeline(name, c_reg=c_reg)
        t0 = time.perf_counter()
        pipe.fit(tr_texts_cv, y_train[tr_idx])
        fit_s = time.perf_counter() - t0

        val_logits = extract_logits(pipe, va_texts_cv)
        te_logits = extract_logits(pipe, test_texts)
        t_opt = fit_temperature(val_logits, y_train[val_idx], t_min=0.05, t_max=20.0)
        val_probs = apply_temperature(val_logits, t_opt)
        te_probs_raw = apply_temperature(te_logits, 1.0)
        te_probs_cal = apply_temperature(te_logits, t_opt)

        preds = np.argmax(te_probs_cal, axis=1)
        model_preds_test[name] = preds
        pred_is_urg = np.array([classes[p] in URGENT_SET for p in preds], dtype=int)
        model_urg_any_preds[name] = pred_is_urg

        acc_ci = metric_ci(y_test, preds, metric="accuracy", seed=0)
        urg_bin_ci = metric_ci(is_urg_test, pred_is_urg, metric="recall", cls=1, seed=0)
        novel_ci = metric_ci(y_test[novel_85_mask], preds[novel_85_mask], metric="accuracy", seed=0)
        dup_ci = metric_ci(y_test[dup_85_mask], preds[dup_85_mask], metric="accuracy", seed=0)

        ece_raw = expected_calibration_error(te_probs_raw, y_test, n_bins=10)
        ece_cal = expected_calibration_error(te_probs_cal, y_test, n_bins=10)

        thr = choose_threshold(
            val_probs, y_train[val_idx], cost_matrix, cfg.review_cost, cfg.max_review_fraction
        )
        max_rev = int(np.floor(cfg.max_review_fraction * len(test_texts)))
        decs = decide(te_probs_cal, cost_matrix, threshold=thr, max_review=max_rev)
        rcost = realised_cost(decs, y_test, cost_matrix, cfg.review_cost)
        rcost_no_rev = realised_cost(
            decide(te_probs_cal, cost_matrix, threshold=float("inf"), max_review=0),
            y_test,
            cost_matrix,
            cfg.review_cost,
        )

        # Latency benchmark (100 single messages)
        t_inf = time.perf_counter()
        for txt in test_texts[:100]:
            extract_logits(pipe, [txt])
        ms_msg = (time.perf_counter() - t_inf) / 100.0 * 1000.0

        # Urgent recall after human escalation (sent to REVIEW OR urgent queue)
        urg_rows = is_urg_test == 1
        urg_caught = sum(
            1 for d in decs[urg_rows] if (d == REVIEW or classes[int(d)] in URGENT_SET)
        )
        urg_missed = int(np.sum(urg_rows)) - urg_caught

        row = {
            "fit_seconds": round(fit_s, 2),
            "ms_per_msg": round(ms_msg, 2),
            "temperature": round(t_opt, 4),
            "ece_raw": round(ece_raw, 4),
            "ece_cal": round(ece_cal, 4),
            "acc_all": _iv_dict(acc_ci),
            "acc_novel": _iv_dict(novel_ci),
            "acc_dup": _iv_dict(dup_ci),
            "urgent_any_recall": _iv_dict(urg_bin_ci),
            "routing_threshold": round(thr, 4),
            "review_fraction": round(float(np.mean(decs == REVIEW)), 4),
            "realised_cost_no_review": round(rcost_no_rev, 4),
            "realised_cost_with_review": round(rcost, 4),
            "urgent_missed_after_review": urg_missed,
            "urgent_caught_after_review": urg_caught,
        }
        model_rows[name] = row
        print(
            f"  {name:15s} | acc={acc_ci.estimate:.4f} [{acc_ci.low:.4f},{acc_ci.high:.4f}] "
            f"| novel={novel_ci.estimate:.4f} dup={dup_ci.estimate:.4f} "
            f"| urg_any={urg_bin_ci.estimate:.4f} -> after_rev={urg_caught}/240 "
            f"| ECE {ece_raw:.3f}->{ece_cal:.3f} | cost {rcost_no_rev:.3f}->{rcost:.3f} | {ms_msg:.2f}ms"
        )

    # Paired bootstrap comparisons vs recommended hybrid_logreg
    paired_vs_word_logreg = paired_diff_ci(
        y_test, model_preds_test["hybrid_logreg"], model_preds_test["word_logreg"], "accuracy", seed=0
    )
    paired_vs_hybrid_svc = paired_diff_ci(
        y_test, model_preds_test["hybrid_logreg"], model_preds_test["hybrid_svc"], "accuracy", seed=0
    )
    paired_urg_vs_word = paired_diff_ci(
        is_urg_test,
        model_urg_any_preds["hybrid_logreg"],
        model_urg_any_preds["word_logreg"],
        "recall",
        cls=1,
        seed=0,
    )
    print("\n=== Paired Comparisons (hybrid_logreg vs baselines) ===")
    print("  acc(hybrid_logreg - word_logreg):", _iv_dict(paired_vs_word_logreg))
    print("  acc(hybrid_logreg - hybrid_svc) :", _iv_dict(paired_vs_hybrid_svc))
    print("  urg_any(hybrid_logreg - word_logreg):", _iv_dict(paired_urg_vs_word))

    # 4. Evaluate FINAL 100%-train RoutingBundle on test.csv
    te_logits_final = extract_logits(bundle.pipeline, test_texts)
    te_probs_final_raw = apply_temperature(te_logits_final, 1.0)
    te_probs_final = apply_temperature(te_logits_final, bundle.temperature)
    preds_final = np.argmax(te_probs_final, axis=1)
    ec_final = expected_costs(te_probs_final, cost_matrix)
    best_q_final = np.argmin(ec_final, axis=1)

    max_rev_test = int(np.floor(cfg.max_review_fraction * len(test_texts)))
    decs_final = decide(
        te_probs_final, cost_matrix, threshold=bundle.risk_threshold, max_review=max_rev_test
    )
    te_sims = max_centroid_similarity(bundle.pipeline, bundle.centroids, test_texts)

    final_acc_ci = metric_ci(y_test, preds_final, "accuracy", seed=0)
    final_novel_ci = metric_ci(y_test[novel_85_mask], preds_final[novel_85_mask], "accuracy", seed=0)
    final_dup_ci = metric_ci(y_test[dup_85_mask], preds_final[dup_85_mask], "accuracy", seed=0)
    final_pred_urg = np.array([classes[p] in URGENT_SET for p in preds_final], dtype=int)
    final_urg_bin_ci = metric_ci(is_urg_test, final_pred_urg, "recall", cls=1, seed=0)
    final_urg_exact = float(np.mean(preds_final[is_urg_test == 1] == y_test[is_urg_test == 1]))
    final_ece_raw = expected_calibration_error(te_probs_final_raw, y_test, n_bins=10)
    final_ece_cal = expected_calibration_error(te_probs_final, y_test, n_bins=10)
    final_cost_no_rev = realised_cost(
        best_q_final, y_test, cost_matrix, cfg.review_cost
    )
    final_cost_rev = realised_cost(decs_final, y_test, cost_matrix, cfg.review_cost)
    final_urg_caught = sum(
        1 for d in decs_final[is_urg_test == 1] if (d == REVIEW or classes[int(d)] in URGENT_SET)
    )

    # Full-explanation latency benchmark
    t_exp = time.perf_counter()
    for txt in test_texts[:50]:
        bundle.predict_one(txt)
    ms_full_explain = (time.perf_counter() - t_exp) / 50.0 * 1000.0

    # 5. Smallest & Urgent Category Breakdown with Wilson-floored CIs
    train_counts = {c: train_cats.count(c) for c in classes}
    smallest_cats = sorted(classes, key=lambda c: train_counts[c])[:5]
    per_cat_stats: dict[str, Any] = {}
    for c in smallest_cats + [c for c in sorted(URGENT_SET) if c not in smallest_cats]:
        ci = metric_ci(y_test, preds_final, metric="recall", cls=cls_to_idx[c], seed=0)
        per_cat_stats[c] = {
            "train_count": train_counts[c],
            "test_count": ci.n_items,
            "is_urgent": c in URGENT_SET,
            "recall_ci": _iv_dict(ci),
        }

    # 6. Label-shuffle sanity check
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y_train)
    pipe_shuf = build_model_pipeline("word_logreg", c_reg=5.0)
    pipe_shuf.fit(tr_texts_cv, y_shuf[tr_idx])
    shuf_acc = float(np.mean(pipe_shuf.predict(test_texts) == y_test))

    # 7. Paraphrase / typo & OOD stress tests
    new_wording_cases = [
        ("Somebody grabbed my handbag with my debit card inside, lock it asap!", "lost_or_stolen_card"),
        ("I see a 200 dollar charge at BestBuy I never made on my statement", "card_payment_not_recognised"),
        ("My mobile got snatched on the train, please freeze access", "lost_or_stolen_phone"),
        ("ATM didnt spit out cash yet my balance went down by 100", "cash_withdrawal_not_recognised"),
        ("Why did my account get billed twice for the same coffee shop?", "transaction_charged_twice"),
        ("Can I change the passcode on my plastic card?", "change_pin"),
        ("How long until my brand new card shows up in the mail?", "card_arrival"),
        ("My direct debit has a charge from a company I don't know", "direct_debit_payment_not_recognised"),
    ]
    new_wording_results = []
    for txt, exp in new_wording_cases:
        res = bundle.predict_one(txt)
        new_wording_results.append({
            "text": txt,
            "expected": exp,
            "predicted": res["top_prob_category"],
            "chosen_queue": res["chosen_queue"],
            "top_prob": round(res["top_probs"][0][1], 4),
            "send_to_person": res["send_to_person"],
            "is_ood": res["is_ood"],
        })

    ood_cases = [
        "The ceiling tiles in the building are broken and leaking water.",
        "What is the weather forecast for Mumbai tomorrow?",
        "asdfghjkl qwerty zxcvbnm",
        "I want to apply for a 30 year fixed home mortgage refinance.",
        "Please recommend a good Italian restaurant near the station.",
    ]
    ood_results = []
    for txt in ood_cases:
        res = bundle.predict_one(txt)
        ood_results.append({
            "text": txt,
            "top_prob_category": res["top_prob_category"],
            "top_prob": round(res["top_probs"][0][1], 4),
            "max_centroid_sim": round(res["max_centroid_sim"], 4),
            "risk": round(res["risk"], 3),
            "is_ood": res["is_ood"],
            "send_to_person": res["send_to_person"],
        })

    # 8. Dump misclassified test examples (urgent and normal) for honest error analysis
    error_indices = np.where(preds_final != y_test)[0]
    urgent_errors = []
    normal_errors = []
    for idx in error_indices:
        true_cat = classes[int(y_test[idx])]
        pred_cat = classes[int(preds_final[idx])]
        prob_vec = te_probs_final[idx]
        top3 = [(classes[int(j)], round(float(prob_vec[int(j)]), 4)) for j in np.argsort(prob_vec)[::-1][:3]]
        tr_match_idx = int(nearest_tr[idx])
        entry = {
            "test_idx": int(idx),
            "text": test_texts[idx],
            "true_category": true_cat,
            "pred_category": pred_cat,
            "top3": top3,
            "routed_decision": "REVIEW" if decs_final[idx] == REVIEW else classes[int(decs_final[idx])],
            "nearest_train_sim": round(float(max_sim_te[idx]), 4),
            "nearest_train_cat": train_cats[tr_match_idx],
            "nearest_train_text": train_texts[tr_match_idx],
        }
        if true_cat in URGENT_SET:
            urgent_errors.append(entry)
        else:
            normal_errors.append(entry)

    # Sort errors so we inspect both high-sim label-conflict errors and lower-sim model errors
    urgent_errors_sorted = sorted(urgent_errors, key=lambda e: -e["nearest_train_sim"])
    normal_errors_sorted = sorted(normal_errors, key=lambda e: -e["nearest_train_sim"])

    results = {
        "duplicate_analysis": dup_summary,
        "model_comparison_80pct": model_rows,
        "paired_comparisons": {
            "acc_hybrid_logreg_vs_word_logreg": _iv_dict(paired_vs_word_logreg),
            "acc_hybrid_logreg_vs_hybrid_svc": _iv_dict(paired_vs_hybrid_svc),
            "urg_any_hybrid_logreg_vs_word_logreg": _iv_dict(paired_urg_vs_word),
        },
        "final_bundle": {
            "oof_accuracy": round(bundle.metadata["oof_accuracy"], 4),
            "temperature": round(bundle.temperature, 4),
            "risk_threshold": round(bundle.risk_threshold, 4),
            "ood_sim_threshold": round(cfg.ood_sim_threshold, 4),
            "test_acc_all": _iv_dict(final_acc_ci),
            "test_acc_novel": _iv_dict(final_novel_ci),
            "test_acc_dup": _iv_dict(final_dup_ci),
            "test_urgent_exact_recall": round(final_urg_exact, 4),
            "test_urgent_any_recall": _iv_dict(final_urg_bin_ci),
            "test_urgent_caught_after_review": f"{final_urg_caught}/{int(is_urg_test.sum())}",
            "test_review_fraction": round(float(np.mean(decs_final == REVIEW)), 4),
            "test_ece_raw": round(final_ece_raw, 4),
            "test_ece_calibrated": round(final_ece_cal, 4),
            "test_realised_cost_no_review": round(final_cost_no_rev, 4),
            "test_realised_cost_with_review": round(final_cost_rev, 4),
            "test_ood_flagged_pct": round(float(np.mean(te_sims < cfg.ood_sim_threshold)), 4),
            "ms_per_msg_with_explanation": round(ms_full_explain, 2),
        },
        "per_category_stats": per_cat_stats,
        "label_shuffle_test_acc": round(shuf_acc, 4),
        "new_wording_probes": new_wording_results,
        "ood_probes": ood_results,
        "total_test_errors": int(len(error_indices)),
        "total_urgent_errors_before_review": len(urgent_errors),
        "sample_urgent_errors": urgent_errors_sorted[:10],
        "sample_normal_errors": normal_errors_sorted[:10],
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Final 100%-Train RoutingBundle Summary ===")
    print(json.dumps(results["final_bundle"], indent=2))
    print(f"\nLabel-shuffle sanity check accuracy: {shuf_acc:.4f}")
    print(f"Saved full evaluation report data to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
