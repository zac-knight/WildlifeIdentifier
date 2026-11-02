import numpy as np
import pytest

from wildlens.sdm import (
    BIOCLIM_NAMES,
    bioclim,
    boyce_index,
    change_agreement,
    novel_climate_mask,
    rolling_quarters,
    spatial_folds,
)

MONTHS = np.arange(12.0)


def cube(values: np.ndarray) -> np.ndarray:
    return values[:, None, None]


def result() -> np.ndarray:
    tas = cube(MONTHS + 283.15)
    return bioclim(tas, tas + 5, tas - 5, cube(MONTHS * 10))[0, 0]


def variable(name: str) -> float:
    return float(result()[BIOCLIM_NAMES.index(name)])


def test_quarters_wrap_over_the_new_year() -> None:
    sums = rolling_quarters(cube(MONTHS))[:, 0, 0]
    assert sums[0] == 0 + 1 + 2
    assert sums[11] == 11 + 0 + 1


def test_temperatures_convert_from_kelvin() -> None:
    assert variable("bio1_mean_temp") == pytest.approx(15.5)


def test_diurnal_range_and_extremes() -> None:
    assert variable("bio2_diurnal_range") == pytest.approx(10)
    assert variable("bio5_max_temp_warmest") == pytest.approx(26)
    assert variable("bio6_min_temp_coldest") == pytest.approx(5)
    assert variable("bio7_temp_annual_range") == pytest.approx(21)


def test_quarter_temperatures() -> None:
    assert variable("bio10_temp_warmest_quarter") == pytest.approx(20)
    assert variable("bio11_temp_coldest_quarter") == pytest.approx(11)


def test_rainfall_variables() -> None:
    assert variable("bio12_annual_precip") == pytest.approx(660)
    assert variable("bio13_precip_wettest_month") == pytest.approx(110)
    assert variable("bio14_precip_driest_month") == pytest.approx(0)
    assert variable("bio16_precip_wettest_quarter") == pytest.approx(300)
    assert variable("bio17_precip_driest_quarter") == pytest.approx(30)


def test_nan_ocean_cells_stay_nan() -> None:
    values = np.full((12, 1, 1), np.nan)
    assert np.isnan(bioclim(values, values, values, values)).all()


def test_boyce_is_high_when_presences_favour_high_suitability() -> None:
    rng = np.random.default_rng(0)
    background = rng.random(5000)
    presence = rng.random(5000) ** 0.3
    assert boyce_index(presence, background) > 0.9


def test_boyce_is_low_when_presences_are_random() -> None:
    rng = np.random.default_rng(1)
    assert abs(boyce_index(rng.random(20000), rng.random(20000))) < 0.5


def test_boyce_is_negative_for_reversed_model() -> None:
    rng = np.random.default_rng(2)
    assert boyce_index(rng.random(5000) ** 3, rng.random(5000)) < -0.8


def test_spatial_folds_never_split_a_block() -> None:
    rng = np.random.default_rng(0)
    latitude, longitude = rng.uniform(-40, -10, 500), rng.uniform(115, 150, 500)
    blocks = np.floor(latitude / 5) * 1000 + np.floor(longitude / 5)
    for train, test in spatial_folds(latitude, longitude, 5, 5):
        assert not set(blocks[train]) & set(blocks[test])


def test_spatial_folds_need_two_blocks() -> None:
    assert spatial_folds(np.array([-30.1, -30.2]), np.array([150.1, 150.2]), 5, 5) == []


def test_novel_climate_is_flagged() -> None:
    reference = np.array([[0.0, 0.0], [10.0, 10.0]])
    grid = np.array([[[5.0, 5.0], [11.0, 5.0]]])
    assert novel_climate_mask(grid, reference).tolist() == [[False, True]]


def test_change_agreement_requires_all_members_to_agree() -> None:
    baseline = np.array([0.0, 1.0, 0.0])
    median = np.array([1.0, 0.0, 1.0])
    wet = np.array([1.0, 0.0, 0.0])
    assert change_agreement(baseline, [median, wet], 0.5).tolist() == [True, True, False]
