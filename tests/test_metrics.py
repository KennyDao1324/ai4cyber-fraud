import numpy as np

from ai4cyber.decision import make_review_decision
from ai4cyber.metrics import binary_metrics, threshold_at_fpr


def test_threshold_respects_fpr_budget():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.02, 0.08, 0.12, 0.91, 0.70, 0.95])
    threshold = threshold_at_fpr(labels, scores, target_fpr=0.25)
    metrics = binary_metrics(labels, scores, threshold)
    assert metrics["false_positive_rate"] <= 0.25
    assert metrics["recall"] == 1.0


def test_decision_is_deterministic_and_ml_owned():
    result = make_review_decision(score=0.75, threshold=0.50)
    assert result["label"] == "NEEDS_REVIEW"
    assert result["decision_source"] == "trained ML classifier"


def test_threshold_fallback_never_defaults_to_unsafe_half():
    labels = np.array([0, 1])
    scores = np.array([0.9, 0.2])
    threshold = threshold_at_fpr(labels, scores, target_fpr=0.0)
    assert threshold > scores.max()
    assert binary_metrics(labels, scores, threshold)["false_positive_rate"] == 0.0
