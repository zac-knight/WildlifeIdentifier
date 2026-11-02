"""Bioclimatic variables and species distribution model evaluation.

Why boosted trees rather than Maxent: both are established SDM methods (Elith et al., 2008, for
boosted regression trees). Trees capture interactions and thresholds with no feature engineering
and train in seconds per species. Output is relative suitability, not a probability of presence,
because presence-background data cannot reveal true prevalence.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

BIOCLIM_NAMES = (
    "bio1_mean_temp",
    "bio2_diurnal_range",
    "bio4_temp_seasonality",
    "bio5_max_temp_warmest",
    "bio6_min_temp_coldest",
    "bio7_temp_annual_range",
    "bio10_temp_warmest_quarter",
    "bio11_temp_coldest_quarter",
    "bio12_annual_precip",
    "bio13_precip_wettest_month",
    "bio14_precip_driest_month",
    "bio15_precip_seasonality",
    "bio16_precip_wettest_quarter",
    "bio17_precip_driest_quarter",
)


def rolling_quarters(monthly: np.ndarray) -> np.ndarray:
    """Three-month running sums, wrapping over the new year. Axis 0 must be the 12 months."""
    return sum(np.roll(monthly, -shift, axis=0) for shift in range(3))


def bioclim(tas: np.ndarray, tasmax: np.ndarray, tasmin: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """Standard BIOCLIM variables from monthly climatologies with shape (12, rows, cols).

    Temperatures are in kelvin on input and Celsius on output. Rainfall is in mm per month.
    The result has shape (rows, cols, variables). The same function handles present and future
    climate, so a projection cannot differ merely because of how a variable was calculated.
    """
    tas, tasmax, tasmin = tas - 273.15, tasmax - 273.15, tasmin - 273.15
    temperature_quarters = rolling_quarters(tas) / 3
    rain_quarters = rolling_quarters(pr)
    annual = pr.sum(axis=0)
    mean_monthly_rain = annual / 12
    # Rainfall seasonality is a coefficient of variation. The offset avoids dividing by zero in arid
    # cells, as the BIOCLIM definition does.
    seasonality = 100 * pr.std(axis=0, ddof=1) / (1 + mean_monthly_rain)
    variables = [
        tas.mean(axis=0),
        (tasmax - tasmin).mean(axis=0),
        100 * tas.std(axis=0, ddof=1),
        tasmax.max(axis=0),
        tasmin.min(axis=0),
        tasmax.max(axis=0) - tasmin.min(axis=0),
        temperature_quarters.max(axis=0),
        temperature_quarters.min(axis=0),
        annual,
        pr.max(axis=0),
        pr.min(axis=0),
        seasonality,
        rain_quarters.max(axis=0),
        rain_quarters.min(axis=0),
    ]
    return np.stack(variables, axis=-1).astype(np.float32)


def boyce_index(presence: np.ndarray, background: np.ndarray, bins: int = 10) -> float:
    """Continuous Boyce index, a presence-only measure of whether suitability tracks occurrence.

    In each moving suitability window it compares the presence share to the background share. It
    returns their Spearman correlation with suitability: +1 is ideal and 0 is no better than chance.
    AUC is reported too, but it cannot be read as usual without true absences.
    """
    low, high = float(np.min(background)), float(np.max(background))
    if high <= low:
        return float("nan")
    width = (high - low) / bins
    centres = np.linspace(low + width / 2, high - width / 2, 100)
    ratios, positions = [], []
    for centre in centres:
        start, end = centre - width / 2, centre + width / 2
        p = np.mean((presence >= start) & (presence <= end))
        b = np.mean((background >= start) & (background <= end))
        if b > 0:
            ratios.append(p / b)
            positions.append(centre)
    if len(ratios) < 3 or np.std(ratios) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(np.array(positions)), _rank(np.array(ratios)))[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    # Average tied ranks, as Spearman correlation requires.
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spatial_folds(
    latitude: np.ndarray, longitude: np.ndarray, degrees: float, folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Whole blocks are held out. Nearby records are alike, so random folds flatter a model by
    testing it on places it has effectively already seen."""
    blocks = (np.floor(latitude / degrees) * 100_000 + np.floor(longitude / degrees)).astype(int)
    count = min(folds, len(np.unique(blocks)))
    if count < 2:
        return []
    return list(GroupKFold(n_splits=count).split(latitude, groups=blocks))


def auc(presence: np.ndarray, background: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(presence)), np.zeros(len(background))])
    return float(roc_auc_score(labels, np.concatenate([presence, background])))


def novel_climate_mask(grid: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """True where any variable falls outside the training range.

    Trees cannot extrapolate. They flatten at the edge of the data, so projections for climates
    with no modern analogue are untrustworthy and must be flagged.
    """
    low, high = np.nanmin(reference, axis=0), np.nanmax(reference, axis=0)
    return ((grid < low) | (grid > high)).any(axis=-1)


def change_agreement(
    baseline: np.ndarray, members: list[np.ndarray], threshold: float
) -> np.ndarray:
    """True where every climate percentile agrees on gain, loss or no change in suitable habitat."""
    present = baseline >= threshold
    changes = [(member >= threshold).astype(int) - present.astype(int) for member in members]
    return np.all([change == changes[0] for change in changes], axis=0)


def climate_cell(latitude: np.ndarray, longitude: np.ndarray, degrees: float = 0.25) -> np.ndarray:
    return (np.floor(latitude / degrees) * 100_000 + np.floor(longitude / degrees)).astype(np.int64)
