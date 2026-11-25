"""Shared model definition, calibration, routing, OOD checks, and word attribution.

Used by both training/evaluation scripts and the offline Streamlit app.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from coding.calibration import apply_temperature
from coding.decide import REVIEW, decide, expected_costs

URGENT_CATEGORIES: tuple[str, ...] = (
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "compromised_card",
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "direct_debit_payment_not_recognised",
)
URGENT_SET: frozenset[str] = frozenset(URGENT_CATEGORIES)


@dataclass(frozen=True)
class RoutingConfig:
    """Cost and escalation parameters for routing decisions."""

    cost_normal_wrong: float = 1.0
    cost_normal_to_urgent: float = 2.0
    cost_urgent_wrong_urgent: float = 5.0
    cost_urgent_missed: float = 30.0
    review_cost: float = 2.5
    max_review_fraction: float = 0.15
    ood_sim_threshold: float = 0.22


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation/extra spaces for duplicate detection."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(text).lower())).strip()


def cluster_near_duplicates(texts: list[str], sim_threshold: float = 0.85) -> np.ndarray:
    """Cluster messages with char-ngram cosine >= sim_threshold into group IDs."""
    normed = [normalize_text(t) for t in texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    x = vec.fit_transform(normed)
    sim = x @ x.T
    adj = (sim >= sim_threshold).astype(np.int8)
    _, comp_ids = connected_components(adj, directed=False)
    return comp_ids


def build_cost_matrix(classes: list[str], cfg: RoutingConfig = RoutingConfig()) -> np.ndarray:
    """Build asymmetric (C, C) misrouting cost matrix prioritizing urgent recall."""
    c = len(classes)
    cost = np.zeros((c, c), dtype=np.float64)
    is_urg = np.array([cls in URGENT_SET for cls in classes], dtype=bool)
    for i in range(c):
        for j in range(c):
            if i == j:
                continue
            if is_urg[i] and not is_urg[j]:
                cost[i, j] = cfg.cost_urgent_missed
            elif is_urg[i] and is_urg[j]:
                cost[i, j] = cfg.cost_urgent_wrong_urgent
            elif (not is_urg[i]) and is_urg[j]:
                cost[i, j] = cfg.cost_normal_to_urgent
            else:
                cost[i, j] = cfg.cost_normal_wrong
    return cost


def build_model_pipeline(model_name: str, c_reg: float = 5.0) -> Pipeline:
    """Construct one of the benchmarked CPU-friendly sklearn text pipelines."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=30000,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )
    hybrid_vec = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=2,
                strip_accents="unicode",
                token_pattern=r"(?u)\b\w+\b",
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                min_df=3,
            ),
        ),
    ])

    if model_name == "multinomial_nb":
        return Pipeline([("vec", word_vec), ("clf", MultinomialNB(alpha=0.1))])
    if model_name == "word_logreg":
        return Pipeline([("vec", word_vec), ("clf", LogisticRegression(C=c_reg, max_iter=1000))])
    if model_name == "word_svc":
        return Pipeline([("vec", word_vec), ("clf", LinearSVC(C=1.0))])
    if model_name == "hybrid_svc":
        return Pipeline([("vec", hybrid_vec), ("clf", LinearSVC(C=0.8))])
    if model_name == "hybrid_logreg":
        return Pipeline([("vec", hybrid_vec), ("clf", LogisticRegression(C=c_reg, max_iter=1000))])
    raise ValueError(f"Unknown model_name: {model_name!r}")


def extract_logits(pipeline: Pipeline, texts: list[str] | Any) -> np.ndarray:
    """Return raw (N, C) logits or log-probabilities for temperature scaling."""
    vec = pipeline.named_steps["vec"]
    clf = pipeline.named_steps["clf"]
    x = vec.transform(texts)
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(x), dtype=np.float64)
    probs = np.asarray(clf.predict_proba(x), dtype=np.float64)
    return np.log(np.clip(probs, 1e-12, 1.0))


def compute_class_centroids(pipeline: Pipeline, texts: list[str], y: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute L2-normalized TF-IDF class centroids for fast OOD similarity detection."""
    x = pipeline.named_steps["vec"].transform(texts)
    centroids = np.zeros((n_classes, x.shape[1]), dtype=np.float32)
    for cls_idx in range(n_classes):
        mask = y == cls_idx
        if np.any(mask):
            mean_vec = np.asarray(x[mask].mean(axis=0)).ravel().astype(np.float32)
            norm = float(np.linalg.norm(mean_vec))
            centroids[cls_idx] = mean_vec / max(norm, 1e-12)
    return centroids


def max_centroid_similarity(pipeline: Pipeline, centroids: np.ndarray, texts: list[str]) -> np.ndarray:
    """Compute max cosine similarity between each input's normalized feature vector and centroids."""
    x: csr_matrix = pipeline.named_steps["vec"].transform(texts)
    row_norms = np.sqrt(x.multiply(x).sum(axis=1)).A1
    row_norms = np.maximum(row_norms, 1e-12)
    sims = (x @ centroids.T) / row_norms[:, None]
    return np.asarray(sims.max(axis=1), dtype=np.float64)


def explain_words(pipeline: Pipeline, text: str, class_idx: int) -> list[tuple[str, float]]:
    """Compute exact leave-one-word-out logit contributions for each token in text."""
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return []
    variants = [text]
    for i in range(len(tokens)):
        ablated = " ".join(tokens[:i] + tokens[i + 1 :]).strip()
        variants.append(ablated if ablated else "")
    logits = extract_logits(pipeline, variants)
    base_logit = float(logits[0, class_idx])
    results: list[tuple[str, float]] = []
    for i, tok in enumerate(tokens):
        contrib = base_logit - float(logits[i + 1, class_idx])
        results.append((tok, contrib))
    return results


@dataclass
class RoutingBundle:
    """Serialized bundle holding the trained pipeline, calibration T, routing & OOD stats."""

    pipeline: Pipeline
    classes: list[str]
    temperature: float
    cost_matrix: np.ndarray
    risk_threshold: float
    centroids: np.ndarray
    config: RoutingConfig
    metadata: dict[str, Any]

    def predict_one(
        self,
        text: str,
        threshold_override: float | None = None,
        ood_threshold_override: float | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Predict queue, calibrated probs, human escalation flag, OOD check, and word drivers."""
        clean = text.strip()
        if not clean:
            raise ValueError("Input message cannot be empty.")

        logits = extract_logits(self.pipeline, [clean])
        probs = apply_temperature(logits, self.temperature)
        ec = expected_costs(probs, self.cost_matrix)
        best_queue_idx = int(np.argmin(ec[0]))
        risk = float(ec[0, best_queue_idx])

        thresh = self.risk_threshold if threshold_override is None else float(threshold_override)
        ood_thr = self.config.ood_sim_threshold if ood_threshold_override is None else float(ood_threshold_override)

        dec = int(decide(probs, self.cost_matrix, threshold=thresh, max_review=1)[0])
        sim = float(max_centroid_similarity(self.pipeline, self.centroids, [clean])[0])
        is_ood = sim < ood_thr

        # Escalate to a person if expected risk exceeds threshold OR message looks out-of-distribution
        send_to_person = (dec == REVIEW) or is_ood

        prob_vec = probs[0]
        top_idx = np.argsort(prob_vec)[::-1][:top_k]
        top_probs = [(self.classes[int(i)], float(prob_vec[int(i)])) for i in top_idx]
        top_prob_idx = int(top_idx[0])
        top_prob_cat = self.classes[top_prob_idx]
        chosen_queue = self.classes[best_queue_idx]

        # Probability summed across all 6 urgent safety categories
        urgent_prob = float(
            sum(prob_vec[i] for i, c in enumerate(self.classes) if c in URGENT_SET)
        )
        is_urgent = (chosen_queue in URGENT_SET) or (top_prob_cat in URGENT_SET) or (urgent_prob >= 0.15)

        word_scores = explain_words(self.pipeline, clean, best_queue_idx)

        return {
            "chosen_queue": chosen_queue,
            "top_prob_category": top_prob_cat,
            "top_probs": top_probs,
            "urgent_prob": urgent_prob,
            "is_urgent": is_urgent,
            "risk": risk,
            "threshold": thresh,
            "max_centroid_sim": sim,
            "is_ood": is_ood,
            "send_to_person": send_to_person,
            "word_scores": word_scores,
        }

    def save(self, path: str) -> None:
        """Persist bundle to disk with compression."""
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str) -> "RoutingBundle":
        """Load serialized RoutingBundle from disk."""
        return joblib.load(path)
