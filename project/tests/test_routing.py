"""Unit tests for Part 2 shared routing, OOD detection, and word attribution."""

import os

import pytest

from project.app import render_highlighted_words
from project.predict import load_model, predict
from project.routing_model import (
    URGENT_CATEGORIES,
    RoutingConfig,
    build_cost_matrix,
    cluster_near_duplicates,
    normalize_text,
)
from project.train_model import BUNDLE_PATH


def test_cost_matrix_asymmetry() -> None:
    classes = ["card_arrival", "lost_or_stolen_card", "change_pin"]
    cost = build_cost_matrix(classes, RoutingConfig())
    assert cost[0, 0] == 0.0 and cost[1, 1] == 0.0
    assert cost[1, 0] > cost[0, 2]
    assert cost[1, 0] == RoutingConfig().cost_urgent_missed


def test_near_duplicate_clustering() -> None:
    texts = [
        "How long does a card delivery take?",
        "how long does a card delivery take",
        "Where is my cash withdrawal?",
    ]
    groups = cluster_near_duplicates(texts, sim_threshold=0.85)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
    assert normalize_text("Hello, World!!") == "hello world"


def test_trained_bundle_inference_and_explanation() -> None:
    if not os.path.exists(BUNDLE_PATH):
        pytest.skip("Trained bundle not present yet; run python -m project.train_model")
    bundle = load_model(BUNDLE_PATH)
    assert len(bundle.classes) == 77
    assert set(URGENT_CATEGORIES).issubset(set(bundle.classes))

    res = predict(bundle, "Someone stole my debit card and wallet, block it immediately!")
    assert res["chosen_queue"] in bundle.classes
    assert res["is_urgent"] is True
    assert len(res["top_probs"]) == 5
    assert abs(sum(p for _, p in res["top_probs"]) - 1.0) < 1.0 + 1e-6
    assert len(res["word_scores"]) > 0

    html_out = render_highlighted_words(res["word_scores"])
    assert "stole" in html_out and "<span" in html_out

    with pytest.raises(ValueError):
        bundle.predict_one("   ")
