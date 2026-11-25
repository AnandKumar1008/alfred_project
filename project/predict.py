"""Compatibility wrapper exposing load_model and predict using RoutingBundle."""

from __future__ import annotations

import os
from typing import Any

from project.routing_model import RoutingBundle, URGENT_SET

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "routing_bundle.joblib")
URGENT_CATEGORIES = set(URGENT_SET)


def load_model(path: str | None = None) -> RoutingBundle:
    """Load the serialized RoutingBundle from disk."""
    target = path or MODEL_PATH
    if not os.path.exists(target):
        raise FileNotFoundError(f"Model bundle not found at {target}. Run: python -m project.train_model")
    return RoutingBundle.load(target)


def predict(bundle: RoutingBundle, text: str) -> dict[str, Any]:
    """Run calibrated, cost-sensitive prediction and word attribution for one message."""
    return bundle.predict_one(text)
