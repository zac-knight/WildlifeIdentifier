import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wildlens.prior import (  # noqa: E402
    PriorForExport,
    SpeciesPrior,
    assume_negative_loss,
    coarsen,
    day_of_year,
    encode,
)

pytestmark = pytest.mark.train


def test_december_and_january_are_close() -> None:
    zero = torch.tensor([0.0])
    december = encode(zero, zero, torch.tensor([365.0]))
    january = encode(zero, zero, torch.tensor([1.0]))
    july = encode(zero, zero, torch.tensor([182.0]))
    assert torch.dist(december, january) < torch.dist(december, july) / 10


def test_antimeridian_is_continuous() -> None:
    zero = torch.tensor([0.0])
    east = encode(zero, torch.tensor([179.9]), zero)
    west = encode(zero, torch.tensor([-179.9]), zero)
    assert torch.dist(east, west) < 0.01


def test_export_wrapper_matches_manual_encoding() -> None:
    model = SpeciesPrior(5).eval()
    inputs = torch.tensor([[-33.0, 151.0, 100.0]])
    expected = torch.sigmoid(model(encode(inputs[:, 0], inputs[:, 1], inputs[:, 2])))
    assert torch.allclose(PriorForExport(model)(inputs), expected)


def test_loss_rewards_correct_presence() -> None:
    labels = torch.tensor([0])
    good = torch.tensor([[10.0, -10.0, -10.0]])
    bad = torch.tensor([[-10.0, 10.0, 10.0]])
    background = torch.full((1, 3), -10.0)
    assert assume_negative_loss(good, labels, background, 32) < assume_negative_loss(
        bad, labels, background, 32
    )


def test_day_of_year() -> None:
    assert day_of_year(np.array([1, 3]), np.array([1, 1])).tolist() == [1, 60]


def test_coarsening_hides_fine_position() -> None:
    assert coarsen(np.array([-33.87, -33.99]), 0.5).tolist() == [-33.75, -33.75]
