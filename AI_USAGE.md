# AI Usage Disclosure

Per the assignment instructions, here is an honest breakdown of AI assistant usage across this project:

- **Tools used**: Antigravity AI coding assistant (Claude Opus 4.6 & Gemini)
- **What AI was used for**:
  1. **Part 1 (`coding/`)**: Structuring `splits.py` (greedy count-deficit assignment + local move/swap repair), `calibration.py` (log-softmax NLL, analytic boundary derivative checks, golden-section search in `log(T)`), `decide.py` (prefix-sum `O(N log N)` threshold sweep), `bootstrap.py` (Wilson/McNemar floors + stdlib `math.erf` Newton refinement), and writing `pytest` test cases cross-checked against SciPy.
  2. **Part 2 (`project/`)**: Implementing the BANKING77 downloader, near-duplicate cosine clustering, 5-model CPU comparison (`MultinomialNB`, `Word-LogReg`, `Word-LinearSVC`, `Hybrid-LinearSVC`, `Hybrid-LogReg`), out-of-fold calibration and cost-sensitive escalation, leave-one-word-out explanations, Streamlit UI, and evaluation/report structure.
- **Verification & debugging performed**:
  - Verified float-precision bin edge behavior (`ceil(0.55 * 100) - 1 == 55`) in `calibration.py`.
  - Fixed greedy fold collapse in `splits.py`, `log(T)` sign inversion in `fit_temperature`, and PolyAI BANKING77 label-alignment bug in `download_data.py`.
  - Ran `pytest coding/tests`, `pytest project/tests`, `python -m coding.evaluate_models --seed 0`, `python -m project.train_model`, `python -m project.evaluate`, and Streamlit startup health checks locally and inspected all metrics and error cases directly.
