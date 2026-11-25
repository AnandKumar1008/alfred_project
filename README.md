# Alfred AI Engineer Assignment (`cs25mtech11004`)

Repository containing Part 1 (four core ML evaluation/routing modules) and Part 2 (offline CPU customer message routing system on BANKING77).

## Repository Layout
```text
.
├── AI_USAGE.md
├── README.md
├── requirements.txt
├── coding/
│   ├── __init__.py
│   ├── splits.py              # C1: Group-aware stratified k-fold
│   ├── calibration.py         # C2: Reliability bins, ECE, temperature scaling
│   ├── decide.py              # C3: Cost-sensitive routing + REVIEW budget
│   ├── bootstrap.py           # C4: Bootstrap CIs with Wilson/McNemar floors
│   ├── demo_data.py           # C4: Demo generator
│   ├── evaluate_models.py     # C4: Paired model evaluation CLI
│   ├── EXPLANATIONS.md        # Concise <=10-line written answers (C1-C4)
│   └── tests/                 # Pytest suite for C1-C4
└── project/
    ├── __init__.py
    ├── download_data.py       # Downloads BANKING77 train.csv / test.csv
    ├── routing_model.py       # Shared model, calibration, routing, OOD, explanations
    ├── train_model.py         # 5-fold group-CV training + OOF calibration/thresholding
    ├── evaluate.py            # Full evaluation, paired CIs, duplicate/noise/error analysis
    ├── predict.py             # Compatibility wrapper for single-message inference
    ├── app.py                 # Offline Streamlit routing UI
    ├── models/                # Trained RoutingBundle (~17 MB, reproducible via train_model.py)
    ├── evaluation_results.json
    └── REPORT.md              # Technical lead report (<= 2 pages)
```

## Setup & Part 1 Verification (Clean Machine)
```bash
pip install -r requirements.txt
pytest coding/tests
python -m coding.evaluate_models --seed 0
```

## Part 2 Reproducibility (Get Data, Train, Evaluate, Run App)
1. **Download BANKING77 (`train.csv`: 10,003 rows, `test.csv`: 3,080 rows)**
   ```bash
   python -m project.download_data
   ```
   Source: PolyAI official BANKING77 CSVs (`CC BY 4.0`, GitHub `PolyAI-LDN/task-specific-datasets`), with Hugging Face Datasets Server (`PolyAI/banking77`) fallback.

2. **Train & calibrate the recommended model (`RoutingBundle`)**
   ```bash
   python -m project.train_model
   ```
   Deterministic (`seed=42`); takes ~3 minutes on CPU and saves `project/models/routing_bundle.joblib` (~17 MB).

3. **Run the full honest evaluation**
   ```bash
   python -m project.evaluate
   ```
   Generates `project/evaluation_results.json` with model comparisons, Wilson-floored bootstrap CIs, near-duplicate vs. novel breakdowns, and misclassified examples.

4. **Launch the offline Streamlit app (1 command)**
   ```bash
   streamlit run project/app.py
   # Or, if Scripts/ is not on PATH:
   python -m streamlit run project/app.py
   ```
   Runs completely locally on CPU (~29 ms/message including leave-one-word-out explanations) with no external network requests.
