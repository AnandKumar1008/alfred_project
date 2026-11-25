# Part 1 — Written Explanations

## C1. Group-Aware Stratified Folds (`coding/splits.py`)
1. **Approach**: Check upfront feasibility, then run seeded restarts of greedy count-deficit assignment (minimizing squared deviation from per-fold target class counts and fold size) followed by local group moves/swaps to repair any residual tolerance or size violations.
2. **Edge cases**: Empty inputs, length mismatches, `k < 2`, `k > n_groups`, `bool` args, `tolerance` outside `[0, 1]`, and unresolvable splits raise `ValueError` naming the failing class and fold; classes in fewer than `k` groups with share above `tolerance` fail fast before searching.
3. **Known limits**: Exact group-stratified partitioning is NP-hard; tightly constrained discrete group sizes may fail within the restart/time budget even if a combinatorial solution exists, and `min_fold_size` cannot be met if a single group leaves fewer than `(k - 1) * (n // (2 * k))` rows.

## C2. Calibration and Temperature Scaling (`coding/calibration.py`)
1. **Approach**: `reliability_bins` assigns rows via `min(n_bins-1, max(0, ceil(p*n_bins)-1))` in float64 and aggregates confidence/accuracy; `apply_temperature` shifts row-max before dividing/exponentiating; `fit_temperature` checks NLL boundary derivatives wrt `beta = 1/T` and searches `log(T)` via golden-section using log-softmax.
2. **Why accuracy is unchanged**: Dividing logits by `T > 0` is strictly monotone, preserving rank order and tie positions so `argmax(logits / T) == argmax(logits)` for every row.
3. **Which data to fit T on**: Fit `T` on a held-out **validation** set (never training data, where logits are overfitted/overconfident, and never test data, which leaks evaluation).
4. **Edge cases & limits**: Handles extreme logits (`10000` at `T=0.05`) without overflow/NaN, returns `1.0` for constant rows, returns boundary `t_min`/`t_max` when optimal, and rejects invalid shapes/sums/dtypes; single-parameter temperature scaling cannot fix non-monotonic or class-conditional miscalibration.

## C3. Routing with Review (`coding/decide.py`)
1. **Approach**: Compute `E = probs @ cost`, take each row's minimum expected cost (`risk`) and lowest-index argmin queue, route up to `max_review` rows with `risk >= threshold` (highest risk, then lowest row) to `REVIEW`, and sweep thresholds via sorted prefix sums in `O(N log N)`.
2. **Why threshold ≈ review_cost**: With calibrated probabilities and unlimited budget, sending a message to review changes expected cost by `review_cost - risk`, so reviewing helps iff `risk >= review_cost`.
3. **When optimal threshold is far from review_cost**: A much lower threshold means the model is overconfident (underestimating true risk); a much higher threshold means the model is underconfident (overestimating risk) or the `max_review_fraction` budget is binding.
4. **Edge cases & limits**: Handles `max_review = 0`, `threshold = math.inf`, tie-breaking on both queues and risks, and invalid arrays/budgets; assumes human reviewers resolve escalated messages accurately at fixed `review_cost`.

## C4. Bootstrap Intervals (`coding/bootstrap.py`, `coding/evaluate_models.py`)
1. **Approach**: Vectorized `default_rng(seed)` percentile bootstrap over `x` (accuracy/recall correctness) or `x_A - x_B`, combined via union (`min(boot_low, floor_low)`, `max(boot_high, floor_high)`) with Wilson score (`metric_ci`) or McNemar-style (`paired_diff_ci`) floors using stdlib `math.erf` + Newton refinement for `Phi^-1`.
2. **Why paired & why floor**: Both models see identical test items so errors are correlated—pairing removes shared item variance; discrete floors prevent degenerate zero-width intervals (e.g., `[1.0, 1.0]` at 20/20) and prohibit calling a difference `significant` when the floor contains 0.
3. **Demo data verdict (`--seed 0`)**: Neither A nor B is significantly better: accuracy is 87.0% vs 87.5% (`A - B = -0.005`, 95% CI `[-0.048, +0.039]`), and urgent recall (`n=25`) is 68.0% vs 64.0% (`+0.040`, 95% CI `[-0.240, +0.280]`)—both intervals span zero.
4. **Edge cases & limits**: Handles `D = 0`, invalid `k/n/level/metric/cls/n_resamples`, and small-sample widening; Wilson/McNemar floors assume i.i.d. test rows, not clustered groups.
