"""Location and date prior, following Spatial Implicit Neural Representations (Cole et al., 2023).

Only presence records exist: nobody records where a species was absent. So absence is approximated
by the "assume negative" idea. At an observed record, other species are treated as weakly absent,
and at a random location every species is treated as weakly absent. A presence is weighted up by
`positive_weight`, which stops the many pseudo-absences from swamping the rare true presences.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def encode(latitude: torch.Tensor, longitude: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
    """Sine and cosine make the input wrap: 31 December sits beside 1 January, and 180 degrees of
    longitude beside -180. Plain numbers would place them at opposite ends of the input range."""
    lat = torch.deg2rad(latitude)
    lon = torch.deg2rad(longitude)
    season = 2 * torch.pi * day / 365.25
    return torch.stack(
        [
            torch.sin(lat),
            torch.cos(lat),
            torch.sin(lon),
            torch.cos(lon),
            torch.sin(season),
            torch.cos(season),
        ],
        dim=-1,
    )


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class SpeciesPrior(nn.Module):
    """Small by design: under 1 MB, so it adds almost nothing to the browser download."""

    def __init__(self, classes: int, width: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(6, width), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.output = nn.Linear(width, classes)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(encoded)))


class PriorForExport(nn.Module):
    """Takes raw latitude, longitude and day, so the browser cannot get the encoding wrong."""

    def __init__(self, model: SpeciesPrior) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = encode(inputs[:, 0], inputs[:, 1], inputs[:, 2])
        return torch.sigmoid(self.model(encoded))


def assume_negative_loss(
    logits_observed: torch.Tensor,
    labels: torch.Tensor,
    logits_random: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    """The SINR "AN-full" loss: every species is a label at each observed and random location."""
    epsilon = 1e-5
    observed = torch.sigmoid(logits_observed)
    random = torch.sigmoid(logits_random)
    rows = torch.arange(len(labels), device=labels.device)
    positive = -torch.log(observed[rows, labels] + epsilon)
    negative_mask = torch.ones_like(observed, dtype=torch.bool)
    negative_mask[rows, labels] = False
    negative = -torch.log(1 - observed + epsilon) * negative_mask
    background = -torch.log(1 - random + epsilon)
    classes = observed.shape[1]
    return (
        positive_weight * positive.mean()
        + negative.sum(dim=1).mean() / (classes - 1)
        + background.mean()
    )


def day_of_year(months: np.ndarray, days: np.ndarray) -> np.ndarray:
    cumulative = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    return cumulative[months - 1] + days


def coarsen(values: np.ndarray, degrees: float) -> np.ndarray:
    """Snap to the centre of a coarse grid cell."""
    return (np.floor(values / degrees) + 0.5) * degrees
