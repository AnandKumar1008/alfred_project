# Customer message routing — technical report

## Executive summary

The baseline is a CPU-friendly word/character TF-IDF classifier with logistic regression. It predicts one of the 77 BANKING77 queues, estimates calibrated class probabilities, applies a cost-sensitive decision rule, and abstains to a human when expected risk or uncertainty is high. It is a baseline for controlled evaluation, not a production safety guarantee.

## 1. Model choice and evidence

**Recommendation:** use the word + character TF-IDF logistic-regression baseline for a CPU deployment candidate, subject to a shadow-mode safety gate. The training run also reports a most-frequent-class baseline so the lift over a trivial router is measurable.

Word n-grams capture domain phrases; character n-grams help with misspellings and novel surface forms. Logistic regression is fast for short messages and exposes additive token contributions for decision transparency. A held-out, exact-duplicate-grouped calibration split is used to fit a scalar temperature and select the review threshold. The official test split is used only once for final evaluation. Compare accuracy, macro-F1, urgent recall and expected calibration error in `evaluation.json`.

## 2. Leakage and validation

Exact normalized train/test overlaps are removed from the training pool. A high-precision character-ngram cosine audit (threshold 0.96) also removes likely near-duplicate training rows and writes `near_duplicate_audit.csv`; exact normalized duplicate messages are grouped together for the fit/calibration split. Test labels are not used in this text-only audit. The BANKING77 source does not provide a reliable customer identifier, so semantic near-duplicates below the threshold or messages from the same real customer may still cross partitions. For production, split by customer and time, and audit near-duplicates with human review.

## 3. Trust, calibration and urgency

The six safety categories from the assignment are `lost_or_stolen_card`, `lost_or_stolen_phone`, `compromised_card`, `card_payment_not_recognised`, `cash_withdrawal_not_recognised`, and `direct_debit_payment_not_recognised`. Missing an urgent report is assigned a much higher routing cost than a normal queue error. A human-review budget and review cost are explicit policy assumptions in `project/model.py`; they must be agreed with the operations team. Temperature scaling improves probability calibration on held-out data but does not guarantee calibration after drift. Track ECE, reliability diagrams, urgent recall, and review rates over time.

## 4. Errors and novel messages

Inspect the per-class report in `evaluation.json`, especially confusion between overlapping categories and urgent false negatives. Low confidence and a small top-two probability margin are used as conservative novelty/ambiguity signals. These are not a formal out-of-distribution detector. New wording, new products, and new intents should default to human review until labelled examples are collected and the model is revalidated.

## 5. Human-in-the-loop deployment

**Would I allow unattended routing immediately? No.** Start in shadow mode, require human confirmation for safety categories, and only expand automation after a monitored acceptance period.

The app displays the selected queue, top class probabilities, and positive token contributions. Before sending any message automatically, show the message and rationale to an operator during a shadow period. Require a human to confirm urgent cases initially; log corrections, queue outcomes, and model version; monitor false negatives and drift; and define a rollback path. No message is transmitted to a third-party API by this implementation.

## 6. Reproducibility

Install the pinned-minimum dependencies in `requirements.txt`, download the CC BY 4.0 BANKING77 dataset using `python -m project.download_data`, then run `python -m project.train --seed 42`. The command saves the model and a machine-readable report. Run `pytest -q` for the Part 1 tests and `python -m coding.evaluate_models --seed 0` for the bootstrap demonstration.

> **Important:** scores must be populated from a real local run. Do not invent or substitute demo results for BANKING77 test metrics.
