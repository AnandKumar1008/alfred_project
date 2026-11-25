# Technical Report: Customer Message Routing on BANKING77

**To:** Engineering & R&D Technical Lead | **Dataset:** PolyAI BANKING77 (10,003 train / 3,080 test, 77 queues; 6 urgent safety queues)

## 1. Recommended Model & Experimental Findings

**Recommendation:** **Hybrid Word (1–3) + Character (`char_wb` 3–5) TF-IDF + Multinomial Logistic Regression (`C=10`)**, calibrated with out-of-fold **Temperature Scaling ($T = 0.653$)**, wrapped in an **asymmetric expected-cost router** (`cost_urgent_missed = 30.0`, `review_cost = 2.5`, max review budget 15%) plus a **centroid-similarity OOD check**. On CPU it trains in 46.8s and runs full inference + leave-one-word-out explanations in **28.8 ms/message** (raw scoring: 2.15 ms), well inside the 1s latency limit and completely offline.

**80% Group-Stratified Train (Val-Calibrated) $\rightarrow$ Official Test (`n=3,080`, 95% Wilson-floored Bootstrap CIs):**

| Model | Test Acc [95% CI] | Novel Acc (`<0.85`) | Dup Acc (`>=0.85`) | Urgent Any Recall | Urgent Caught (w/ Review) | ECE Raw $\rightarrow$ Cal | Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| `multinomial_nb` | 83.31% [81.95, 84.59] | 79.08% | 91.51% | 94.58% | 238 / 240 | 0.133 $\rightarrow$ 0.012 | 1.87 ms |
| `word_logreg` | 88.15% [86.96, 89.29] | 84.45% | 95.32% | 92.08% | 238 / 240 | 0.248 $\rightarrow$ 0.009 | 0.36 ms |
| `word_svc` | 88.31% [87.13, 89.48] | 84.99% | 94.75% | 94.17% | 239 / 240 | 0.805 $\rightarrow$ 0.022 | 0.38 ms |
| `hybrid_svc` | 90.52% [89.43, 91.50] | 87.84% | 95.71% | 95.42% | 239 / 240 | 0.810 $\rightarrow$ 0.017 | 2.99 ms |
| `hybrid_logreg` | **91.20%** [90.15, 92.18] | **88.68%** | **96.09%** | **96.25%** | 237 / 240 | 0.116 $\rightarrow$ 0.014 | 2.15 ms |
| **Final 100% Bundle** | **91.56%** [90.52, 92.50] | **88.98%** | **96.56%** | **97.08%** [94.10, 99.17] | **239 / 240** (99.6%) | **0.094 $\rightarrow$ 0.009** | **28.8 ms** |

**What each attempt taught us:**
- `multinomial_nb` is fastest to fit (0.13s) but struggles on novel wording (79.1%), showing feature-independence misses multi-word intent phrases.
- `word_logreg` improves by +4.8 pp (88.2%) but fails on unseen inflections/spelling variants (`recognised` vs `recognized`, typos). Adding `char_wb` (3–5) features yields a statistically significant paired gain of **+3.05%** overall (`95% CI [+2.27%, +3.83%]`) and **+4.17%** on urgent recall (`[+1.67%, +6.67%]`).
- `LinearSVC` matches close accuracy (`90.52%`, `hybrid_logreg` beats it by paired `+0.68%`, `[+0.12%, +1.23%]`), but raw SVM margins are severely miscalibrated (**ECE = 0.810**). Temperature scaling fixes ECE (`0.017`), yet `hybrid_logreg` achieves lower calibrated ECE (**0.009** on final bundle), cleaner probability tails for asymmetric risk routing, and direct log-odds word attribution.

## 2. Sanity Checks & Near-Duplicates Across Train and Test

- **Sanity checks:** Shuffling training labels collapses test accuracy to **1.23%** (matching uniform $1/77 = 1.30\%$). Out-of-fold validation accuracy without duplicate leakage is **89.61%**, closely matching novel test accuracy (**88.98%**).
- **Near-duplicate leakage found:** Using normalized text and `char_wb` (3–5) TF-IDF cosine similarity: **11** test rows match train verbatim (normalized), **370/3,080 (12.0%)** have cosine $\ge 0.95$, **701 (22.8%)** have $\ge 0.90$, and **1,048 (34.0%)** have $\ge 0.85$.
- **Impact & mitigation:** Model accuracy is **96.56%** on the 1,048 near-duplicate test rows versus **88.98%** on the 2,032 novel rows—the official 91.56% headline number is inflated by ~2.6 pp due to cross-split overlap. Within `train.csv`, we clustered rows with cosine $\ge 0.85$ into **7,928 groups** (`coding.splits.group_stratified_kfold`) so all validation tuning ($T=0.653$, threshold $=2.132$) occurred strictly across disjoint paraphrase clusters.

## 3. Trustworthiness: Noisy Labels & Small Categories

- **Label noise & overlap:** Among test–train pairs with cosine $\ge 0.95$, **11/370 (3.0%)** carry conflicting ground-truth labels; at $\ge 0.85$, **68/1,048 (6.5%)** disagree (e.g., *"How long does a card delivery take?"* labeled `card_arrival` in test vs *"how long does delivery take?"* labeled `card_delivery_estimate` in train). This creates an accuracy ceiling near ~93–95%.
- **Small categories:** `train.csv` ranges from **35** (`contactless_not_working`) and **41** (`virtual_card_not_working`) to 187 rows, while `test.csv` has only **40 rows per class**. Even 90.0% test recall (36/40) carries a wide Wilson-floored 95% CI of **[76.9%, 97.5%]** (`virtual_card_not_working`: 82.5% **[68.1%, 92.5%]**). Similarly, each urgent class (`n=40`) spans 10–20 percentage points of uncertainty; we report aggregate urgent coverage (**239/240 caught**, 97.1% before routing) alongside widened per-class CIs so no single-class figure is over-trusted.
- **New wording & OOD fragility:** On 8 custom paraphrase probes, the model missed *"ATM didnt spit out cash yet my balance went down by 100"* (routed to `wrong_amount_of_cash_received`), and on 5 non-banking OOD probes, 4 were escalated (`sim < 0.153`) while *"The ceiling tiles in the building are broken and leaking water."* scored `sim = 0.172` and predicted `country_support` at `p = 0.756`.

## 4. Misclassified Examples (Model Errors vs. Label Errors)

| # | Test Text (`test_idx`) | True Label | Predicted (`p`) | Nearest Train Match (`sim`) | Diagnosis |
|---|---|---|---|---|---|
| 1 | *"Someone stole my cards!"* (`#1672`, **Urgent**) | `lost_or_stolen_phone` | `lost_or_stolen_card` (`0.804`) | *"Someone stole my card!"* (`0.920`, `lost_or_stolen_card`) | **Label error** (ground truth says phone; text clearly says cards) |
| 2 | *"How do I cancel my card? There are charges on my account that I didn't make."* (`#2746`, **Urgent**) | `cash_withdrawal_not_recognised` | `card_payment_not_recognised` (`0.955`) | *"My identity has been stolen... charges I didn't make..."* (`0.637`, `cash_withdrawal...`) | **Ambiguous / Label error** (no mention of cash/ATM; still routed urgent) |
| 3 | *"I cannot find my credit card."* (`#476`, **Urgent**) | `lost_or_stolen_card` | `topping_up_by_card` (`0.974`) | *"i cannot find my card PIN"* (`0.742`, `get_physical_card`) | **Model error** (misses *"cannot find"* = lost; 1/240 urgent miss) |
| 4 | *"My card was taken from me"* (`#467`, **Urgent**) | `lost_or_stolen_card` | `cash_withdrawal_not_recognised` (`0.409`) | *"Why was money taken for a transfer?"* (`0.576`, `transfer_fee...`) | **Model error** (bag-of-words overfits *"taken"* to cash/money taken) |
| 5 | *"how many transactions can i make with a disposable card"* (`#2637`) | `get_disposable_virtual_card` | `disposable_card_limits` (`0.9998`) | *"How many transactions can I make on my disposable card?"* (`0.955`, `disposable_card_limits`) | **Label error** (overlapping classes; text explicitly asks about limits) |
| 6 | *"Do I need to verify my identity?"* (`#1193`) | `why_verify_identity` | `verify_my_identity` (`0.736`) | *"Why do I need to verify my identity?"* (`0.974`, `why_verify_identity`) | **Model error** (misses modal phrasing *"do I need"* vs *"how to"*) |

## 5. Production Readiness Verdict

**No — do not let this model auto-route without human review yet.** While asymmetric routing catches **239/240 (99.6%)** test urgent messages, a pure TF-IDF linear classifier still failed silently on *"I cannot find my credit card"* (`p=0.974` non-urgent) and *"ATM didnt spit out cash..."*, and failed to flag 1 of 5 non-banking OOD probes. **Before unattended routing of high-confidence non-urgent queues:** (1) add a semantic dense/ONNX embedding encoder (e.g., MiniLM on CPU) to handle synonyms like *"cannot find"* $\rightarrow$ *"lost"*; (2) enforce a stricter OOD / minimum-similarity gate plus keyword safety net for card/theft/fraud phrasing; (3) audit and merge overlapping queues (`card_arrival` vs `card_delivery_estimate`); and (4) run shadow-mode logging on real internal customer text with periodic calibration re-fitting (`fit_temperature`) and human review of all urgent/borderline predictions.

**Reproducibility:** `pip install -r requirements.txt` $\rightarrow$ `python -m project.download_data` $\rightarrow$ `python -m project.train_model` $\rightarrow$ `python -m project.evaluate` $\rightarrow$ `streamlit run project/app.py`.
