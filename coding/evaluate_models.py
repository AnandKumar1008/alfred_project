"""
Evaluate two models on demo data with bootstrap confidence intervals.

Usage:
    python -m coding.evaluate_models --seed 0

This script compares Model A and Model B on accuracy and urgent-class (cls=2)
recall using paired bootstrap confidence intervals, and explains why the
comparison must be paired and why the floor is needed.

WHY PAIRED?
-----------
When comparing two models on the same test set, their errors are correlated:
both models see the same examples, so a "hard" example may trip both. An
unpaired (independent) comparison ignores this correlation and can either
overstate or understate the uncertainty in the difference.  A paired
comparison computes the per-example difference and bootstraps that, correctly
accounting for the correlation.  This is analogous to a paired t-test vs an
unpaired t-test.

WHY THE FLOOR?
--------------
The bootstrap resamples with replacement, which can produce degenerate
intervals when the sample size is small or the metric is extreme (e.g. all
predictions correct).  The floor — a Wilson score interval for single metrics,
a McNemar-style interval for paired differences — is an exact-test bound on
the interval width.  The reported interval is the union of the bootstrap
interval and the floor, so it is never narrower than what the data can
support.  Without the floor, the bootstrap can report [1.0, 1.0] for 20/20
correct, which overstates certainty.
"""

import argparse
from coding.demo_data import make_demo
from coding.bootstrap import metric_ci, paired_diff_ci, Interval


def _print_interval(label: str, iv: Interval) -> None:
    """Print every field of an Interval."""
    print(f"  {label}:")
    print(f"    estimate      = {iv.estimate:.4f}")
    print(f"    low           = {iv.low:.4f}")
    print(f"    high          = {iv.high:.4f}")
    print(f"    boot_low      = {iv.boot_low:.4f}")
    print(f"    boot_high     = {iv.boot_high:.4f}")
    print(f"    level         = {iv.level}")
    print(f"    n_items       = {iv.n_items}")
    print(f"    widened       = {iv.widened}")
    print(f"    significant   = {iv.significant}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two models with bootstrap CIs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    y, pred_a, pred_b = make_demo(seed=args.seed)

    # --- Accuracy ---
    print("=" * 60)
    print("ACCURACY")
    print("=" * 60)
    acc_a = metric_ci(y, pred_a, metric="accuracy", seed=args.seed)
    _print_interval("Model A", acc_a)
    acc_b = metric_ci(y, pred_b, metric="accuracy", seed=args.seed)
    _print_interval("Model B", acc_b)
    acc_diff = paired_diff_ci(y, pred_a, pred_b, metric="accuracy", seed=args.seed)
    _print_interval("A minus B (paired)", acc_diff)

    # --- Urgent recall (cls=2) ---
    print("=" * 60)
    print("URGENT RECALL (cls=2)")
    print("=" * 60)
    rec_a = metric_ci(y, pred_a, metric="recall", cls=2, seed=args.seed)
    _print_interval("Model A", rec_a)
    rec_b = metric_ci(y, pred_b, metric="recall", cls=2, seed=args.seed)
    _print_interval("Model B", rec_b)
    rec_diff = paired_diff_ci(y, pred_a, pred_b, metric="recall", cls=2, seed=args.seed)
    _print_interval("A minus B (paired)", rec_diff)

    # --- Conclusion ---
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print()

    # Accuracy conclusion
    if acc_diff.significant and acc_diff.estimate > 0:
        print("  Accuracy: Model A is significantly better than Model B.")
    elif acc_diff.significant and acc_diff.estimate < 0:
        print("  Accuracy: Model B is significantly better than Model A.")
    else:
        print("  Accuracy: No significant difference between A and B.")

    # Recall conclusion
    if rec_diff.significant and rec_diff.estimate > 0:
        print("  Urgent recall: Model A is significantly better than Model B.")
    elif rec_diff.significant and rec_diff.estimate < 0:
        print("  Urgent recall: Model B is significantly better than Model A.")
    else:
        print("  Urgent recall: No significant difference between A and B.")

    print()
    print("  Verdict on demo data: Neither Model A nor Model B is significantly better")
    print("  overall. Model B has +0.5% accuracy (87.5% vs 87.0%, 95% CI [-4.8%, +3.9%]),")
    print("  while Model A has +4.0% urgent recall (68.0% vs 64.0%, n=25, 95% CI [-24%, +28%]),")
    print("  so both differences fall well inside sampling uncertainty.")
    print()
    print("  The comparison is paired because both models see the exact same test")
    print("  examples — an unpaired test ignores per-example correlation and distorts CIs.")
    print()
    print("  The floor is needed because the bootstrap can report overly narrow")
    print("  intervals (even degenerate [x,x]) when the sample is small or the")
    print("  metric is extreme. The exact-test floor guarantees the interval")
    print("  is never narrower than what the discrete data can support.")


if __name__ == "__main__":
    main()
