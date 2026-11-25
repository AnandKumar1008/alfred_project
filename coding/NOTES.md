# Part 1 — approach, edge cases and limits

## C1 — group-aware stratified folds

- Aggregate each group into a class-count vector, then greedily assign whole groups to the currently smallest fold while minimizing class-share deviation.
- Seeded tie-breaking makes runs deterministic; multiple seeded restarts reduce sensitivity to group order.
- Build each training set as the complement of its test set, so every row is tested exactly once and no group crosses folds.
- Reject mismatched/empty inputs, invalid `k`/tolerance, too many folds, and classes represented by too few independent groups when their share makes the requirement impossible.
- If the requested tolerance cannot be met, raise a `ValueError` naming the worst class and fold.
- Limit: greedy assignment is not a proof of global optimality; an infeasible result can mean the constraints are impossible or the heuristic did not find a feasible arrangement.

## C2 — probability calibration

- Reliability bins use the maximum class probability and assign confidence `p` to `ceil(p * n_bins) - 1`, clipping the top endpoint into the last bin.
- ECE is the count-weighted absolute difference between mean confidence and empirical accuracy.
- Temperature scaling divides logits by positive `T`, then uses a max-shifted softmax to avoid overflow.
- Fit `T` by a bounded logarithmic grid followed by a one-dimensional golden-section refinement of validation negative log-likelihood.
- Reject malformed probabilities/logits, invalid labels, invalid bin counts and invalid temperature bounds; constant-logit rows return `T=1`.
- Temperature scaling preserves each row's argmax, so it changes confidence but not classification accuracy.
- Limit: a single scalar temperature cannot fix class-specific or subgroup-specific miscalibration; fit it on a held-out, representative calibration set.

## C3 — cost-sensitive routing and review

- Expected action costs are `probabilities @ cost_matrix`; the lowest-cost queue is selected, with NumPy's first-index tie break.
- Messages whose minimum expected cost meets the threshold are eligible for review; select highest-risk rows first and break ties by original row index.
- Threshold search evaluates only distinct risk boundaries and uses prefix sums, avoiding a quadratic scan on 20,000 rows.
- The threshold is chosen by measured realised cost under the validation review budget; ties choose the largest threshold.
- Reject malformed probabilities/costs, negative costs, invalid labels, thresholds, budgets and review costs.
- Limit: the result is only as good as calibrated probabilities and the supplied cost matrix; review capacity and costs must be agreed with the business.

## C4 — uncertainty intervals

- Percentile bootstrap resamples rows with replacement using the requested seeded NumPy generator.
- Accuracy uses all rows; recall filters to true members of the requested class and rejects empty classes.
- Wilson score intervals form a floor for single-model metrics so a tiny/all-success sample cannot claim impossible certainty.
- Paired differences resample the same rows for both models; a Wilson interval on discordant pairs supplies a conservative floor, including a finite floor when no pairs disagree.
- The reported interval is the union of the bootstrap interval and its floor; paired significance is false whenever the reported interval includes zero.
- Reject empty/unequal arrays, unknown metrics, missing recall class, invalid levels, invalid class counts and fewer than 100 resamples.
- Limit: bootstrap intervals describe sampling uncertainty under the observed sample; they do not correct dataset shift or label bias.
