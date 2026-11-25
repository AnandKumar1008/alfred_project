# Alfred Lab — AI Engineer project

This repository implements both parts of the supplied 11-page assignment.

## What is included

### Part 1 — tested algorithms (`coding/`)
- `splits.py`: deterministic group-aware stratified k-fold assignment. Groups never cross folds; class-share and fold-size constraints are checked and infeasible inputs raise `ValueError`.
- `calibration.py`: reliability bins, ECE, numerically stable temperature scaling, and bounded 1-D temperature fitting without SciPy optimizers.
- `decide.py`: expected routing costs, budget-limited human review, realised cost, and threshold selection.
- `bootstrap.py`: percentile bootstrap intervals, Wilson floors, and paired model-difference intervals.
- `demo_data.py`, `evaluate_models.py`: reproducible public synthetic demo specified in the brief.

### Part 2 — customer message routing (`project/`, `app.py`)
- Uses only BANKING77's 77 labelled categories; six urgent categories are explicitly marked.
- Word + character TF-IDF with balanced logistic regression, held-out group-separated temperature calibration, expected-cost routing, review budget, and token contribution explanations.
- Streamlit interface runs locally; no messages are sent to an external service.
- Test evaluation is kept separate from training/calibration data.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Part 1 checks

From the repository root:

```bash
pytest -q
python -m coding.evaluate_models --seed 0
```

The evaluation command prints accuracy and urgent recall intervals for models A/B and the paired A-minus-B comparison.

## Part 2 data and training

The assignment specifies public BANKING77 (PolyAI, CC BY 4.0), with `train.csv` and `test.csv` containing `text` and `category` columns. Download it with:

```bash
python -m project.download_data
```

Then train and evaluate:

```bash
python -m project.train --seed 42
```

This creates `models/router.joblib`, `reports/evaluation.json`, `reports/REPORT.md`, and `reports/error_audit.csv`. Exact normalized overlaps and high-similarity character-ngram near-duplicates (cosine >= 0.96) are removed from the training pool; `reports/near_duplicate_audit.csv` records the near-duplicate matches, while `reports/error_audit.csv` lists misclassified test examples, prioritizing urgent false negatives. The training code holds out normalized duplicate-message groups from fitting for probability calibration and review-threshold selection; the supplied test split is used only for final evaluation.

## Launch the app

After training:

```bash
streamlit run app.py
```

Type a customer message. The app shows the selected queue or human-review decision, top five temperature-calibrated probabilities, and positive token contributions for the top predicted class. The demo app does not persist a shared daily review counter; production deployment needs a central queue/capacity store to enforce a global daily review budget.

## Design decisions and limitations

1. **Model:** a word/character TF-IDF + logistic-regression baseline is fast on CPU, interpretable at token level, and suited to short text. It avoids large pretrained model downloads and meets the laptop constraint.
2. **Urgency:** the six specified categories receive a larger false-negative routing cost. Review is constrained by a 15% validation-set budget and a per-review cost of 3; these are explicit policy assumptions, not facts supplied by the dataset. Tune them with operations stakeholders.
3. **Calibration:** temperature calibration is fitted on a held-out group split. ECE is reported, but calibration can drift and should be monitored on fresh, representative messages.
4. **Novel wording:** low confidence and a small top-two margin trigger human review. This is a pragmatic abstention guard, not a guarantee of out-of-distribution detection.
5. **Data leakage:** exact normalized duplicates are kept in the same partition. Near-duplicate semantic leakage can still occur; a stronger production audit could use customer identifiers or a reviewed similarity-clustering process.
6. **No live data in this archive:** the execution environment had no network access, so BANKING77 could not be downloaded and a genuine 77-class score is not fabricated. Run the downloader and training commands in a network-enabled environment to create the actual model and report. The synthetic demo is only for checking Part 1 utilities.

## Repository layout

```text
coding/       four algorithm implementations + demo evaluator
project/      BANKING77 loader, downloader, model and training pipeline
tests/        focused unit tests
app.py        Streamlit application
models/       trained model output (created after training)
reports/      evaluation output (created after training)
```
