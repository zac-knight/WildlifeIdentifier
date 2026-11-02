"""Export helpers: ONNX graph creation, size reduction and parity checks."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from timm.layers import set_layer_config
from torch import nn


def build_for_export(architecture: str, classes: int) -> nn.Module:
    """Rebuild a timm model with fixed padding, ready for its trained weights to be loaded.

    TensorFlow-style "same" padding is normally computed at run time, which puts Shape, Ceil and
    Cast arithmetic in the graph. That breaks half-precision conversion and slows browser inference.
    Export mode computes it once for the fixed input size. The weights are unchanged.
    """
    with set_layer_config(exportable=True):
        return timm.create_model(architecture, pretrained=False, num_classes=classes)


class ImageForExport(nn.Module):
    """Returns logits and the last spatial feature map, so CAMs can be computed in the browser."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.forward_features(image)
        logits = self.model.forward_head(features)
        return logits, features


def classifier_weights(model: nn.Module) -> np.ndarray:
    return model.get_classifier().weight.detach().cpu().numpy()


def class_activation_map(features: np.ndarray, weights: np.ndarray, class_index: int) -> np.ndarray:
    """Weighted sum of feature channels, for one image with shape (channels, height, width)."""
    return np.tensordot(weights[class_index], features, axes=1)


def export_onnx(
    model: nn.Module,
    example: torch.Tensor,
    path: Path,
    inputs: list[str],
    outputs: list[str],
    dynamic_axes: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.eval(),
        (example,),
        str(path),
        input_names=inputs,
        output_names=outputs,
        dynamic_axes=dynamic_axes or {name: {0: "batch"} for name in [*inputs, *outputs]},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))


class HalfImageForExport(nn.Module):
    """Half-precision weights with float32 inputs and outputs for the JavaScript side.

    Half precision halves the file size. Converting from PyTorch keeps every operation consistently
    typed, whereas onnxconverter-common produced a graph that ONNX Runtime refused to load.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = copy.deepcopy(model).half()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.forward_features(image.half())
        return self.model.forward_head(features).float(), features.float()


class _Reader(CalibrationDataReader):
    def __init__(self, name: str, batches: list[np.ndarray]) -> None:
        self._items = iter({name: batch} for batch in batches)

    def get_next(self) -> dict | None:
        return next(self._items, None)


def to_int8(source: Path, target: Path, input_name: str, batches: list[np.ndarray]) -> None:
    """Static quantisation with calibration images. Dynamic quantisation is not used, as it leaves
    convolutions, the bulk of this model, largely unquantised and uses ConvInteger, which browser
    runtimes support poorly. Per-channel weights limit the accuracy loss."""
    prepared = target.with_suffix(".pre.onnx")
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
    quantize_static(
        str(prepared),
        str(target),
        _Reader(input_name, batches),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Percentile,
    )
    prepared.unlink(missing_ok=True)


def session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def max_absolute_difference(
    path: Path, inputs: dict[str, np.ndarray], expected: np.ndarray
) -> float:
    """The parity measure: largest difference from PyTorch across every output value."""
    output = session(path).run(None, inputs)[0]
    return float(np.abs(output - expected).max())
