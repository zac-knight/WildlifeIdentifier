from wildlens.promotion import Scores, should_promote, validate_feedback_rows

CURRENT = Scores(top1=0.80, macro=0.78, geographic_macro=0.70)


def test_improvement_is_promoted() -> None:
    assert should_promote(CURRENT, Scores(0.82, 0.80, 0.71)).promote


def test_noise_within_tolerance_is_allowed() -> None:
    assert should_promote(CURRENT, Scores(0.799, 0.779, 0.699)).promote


def test_any_single_regression_blocks_promotion() -> None:
    decision = should_promote(CURRENT, Scores(0.90, 0.75, 0.75))
    assert not decision.promote
    assert "macro" in decision.reasons[0]


def test_geographic_regression_blocks_even_with_better_top1() -> None:
    assert not should_promote(CURRENT, Scores(0.90, 0.90, 0.60)).promote


def test_every_failure_is_reported() -> None:
    assert len(should_promote(CURRENT, Scores(0.1, 0.1, 0.1)).reasons) == 3


ROW = {
    "review_status": "approved",
    "reviewed_species": "A",
    "correct_species": "B",
    "consented_image_path": "x.jpg",
}


def test_approved_consented_row_is_kept() -> None:
    assert validate_feedback_rows([ROW], {"A"}) == [ROW]


def test_unreviewed_feedback_never_reaches_training() -> None:
    assert validate_feedback_rows([{**ROW, "review_status": "pending"}], {"A"}) == []


def test_row_without_a_consented_image_is_dropped() -> None:
    assert validate_feedback_rows([{**ROW, "consented_image_path": None}], {"A"}) == []


def test_unknown_species_is_dropped() -> None:
    assert validate_feedback_rows([ROW], {"Other"}) == []


def test_user_label_is_used_when_reviewer_only_approved() -> None:
    row = {**ROW, "reviewed_species": None}
    assert validate_feedback_rows([row], {"B"}) == [row]
