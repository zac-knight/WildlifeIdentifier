import numpy as np
import pytest
from scipy.special import log_softmax, softmax

from wildlens import metrics as m


def test_topk_and_macro_accuracy() -> None:
    logits = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.0, 3.0, 2.0]])
    labels = np.array([0, 1, 2, 2])
    assert m.topk_accuracy(logits, labels, 1) == 0.5
    assert m.topk_accuracy(logits, labels, 2) == 1.0
    # Per class: 1.0, 0.0, 0.5
    assert m.macro_accuracy(logits, labels) == pytest.approx(0.5)


def test_confused_pairs_merge_both_directions() -> None:
    logits = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    pairs = m.confused_pairs(logits, np.array([0, 1, 0]), ["a", "b"])
    assert pairs == [{"a": "a", "b": "b", "errors": 2}]


def test_fit_temperature_recovers_overconfidence() -> None:
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(5000, 5)) * 2
    labels = np.array(
        [rng.choice(5, p=probabilities) for probabilities in softmax(true_logits, axis=1)]
    )
    assert m.fit_temperature(true_logits * 3, labels) == pytest.approx(3, abs=0.3)


def test_temperature_scaling_reduces_ece() -> None:
    rng = np.random.default_rng(1)
    true_logits = rng.normal(size=(5000, 5)) * 2
    labels = np.array(
        [rng.choice(5, p=probabilities) for probabilities in softmax(true_logits, axis=1)]
    )
    overconfident = true_logits * 3
    before = m.expected_calibration_error(softmax(overconfident, axis=1), labels)
    temperature = m.fit_temperature(overconfident, labels)
    after = m.expected_calibration_error(softmax(overconfident / temperature, axis=1), labels)
    assert after < before / 2


def test_perfect_calibration_has_low_ece() -> None:
    probabilities = np.tile([0.8, 0.2], (10000, 1))
    labels = (np.random.default_rng(0).random(10000) > 0.8).astype(int)
    assert m.expected_calibration_error(probabilities, labels) < 0.02


def test_energy_separates_strong_from_weak_logits() -> None:
    strong = np.array([[10.0, 0.0, 0.0]])
    weak = np.array([[0.1, 0.0, 0.0]])
    assert m.energy_score(strong)[0] > m.energy_score(weak)[0]


def test_ood_auroc_extremes() -> None:
    assert m.ood_auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert m.ood_auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0


def test_threshold_keeps_requested_true_positive_rate() -> None:
    scores = np.arange(1000.0)
    threshold = m.threshold_at_tpr(scores, 0.95)
    assert (scores >= threshold).mean() == pytest.approx(0.95, abs=0.002)
    assert m.fpr_at_threshold(np.array([-1.0, 1e9]), threshold) == 0.5


def test_fusion_prior_can_change_ranking() -> None:
    log_image = np.log(np.array([[0.6, 0.4]]))
    log_prior = np.log(np.array([[0.01, 0.9]]))
    assert m.fuse(log_image, log_prior).argmax() == 1


def test_fusion_floor_limits_the_prior_veto() -> None:
    log_image = np.log(np.array([[0.999, 0.001]]))
    log_prior = np.log(np.array([[1e-12, 0.9]]))
    assert m.fuse(log_image, log_prior, floor=1e-3).argmax() == 0


def test_fusion_without_prior_returns_image_scores() -> None:
    log_image = log_softmax(np.array([[1.0, 2.0]]), axis=1)
    assert np.array_equal(m.fuse(log_image), log_image)


GENERA = ["A", "A", "B", "C"]
FAMILIES = ["F", "F", "F", "G"]


def decide(probabilities):
    return m.hierarchical_decision(np.array(probabilities), GENERA, FAMILIES, 0.7, 0.7, 0.7)


def test_confident_species_is_reported() -> None:
    assert decide([0.8, 0.1, 0.05, 0.05])[:2] == ("species", 0)


def test_falls_back_to_genus() -> None:
    assert decide([0.4, 0.4, 0.1, 0.1])[:2] == ("genus", "A")


def test_falls_back_to_family() -> None:
    assert decide([0.3, 0.2, 0.3, 0.2])[:2] == ("family", "F")


def test_admits_when_nothing_is_confident() -> None:
    probabilities = np.array([0.5, 0.5])
    assert m.hierarchical_decision(probabilities, ["A", "B"], ["F", "G"], 0.7, 0.7, 0.7)[0] == (
        "unknown"
    )
