import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
pytest.importorskip("onnxruntime")

from wildlens.audio import CLIP_SAMPLES, LogMel  # noqa: E402
from wildlens.onnx_export import (  # noqa: E402
    HalfImageForExport,
    ImageForExport,
    build_for_export,
    class_activation_map,
    classifier_weights,
    export_onnx,
    max_absolute_difference,
    session,
    to_int8,
)

pytestmark = pytest.mark.train
ARCHITECTURE = "tf_efficientnetv2_b2"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_for_export(ARCHITECTURE, 7).eval()


def test_export_build_gives_same_outputs_as_training_build(model) -> None:
    """Trained weights load into the export build, and predictions do not change."""
    training = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=7).eval()
    training.load_state_dict(model.state_dict())
    image = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        assert torch.allclose(training(image), model(image), atol=1e-5)


def test_cam_average_equals_logit_minus_bias(model) -> None:
    """Global-average pooling into one linear layer makes the CAM exact, not an approximation."""
    image = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        logits, features = ImageForExport(model)(image)
    weights = classifier_weights(model)
    bias = model.get_classifier().bias.detach().numpy()
    cam = class_activation_map(features[0].numpy(), weights, 3)
    assert cam.mean() + bias[3] == pytest.approx(float(logits[0, 3]), abs=1e-3)


def test_onnx_outputs_match_pytorch(model, tmp_path) -> None:
    wrapper = ImageForExport(model)
    image = torch.randn(1, 3, 224, 224)
    path = tmp_path / "m.onnx"
    export_onnx(wrapper, image, path, ["image"], ["logits", "features"])
    other = torch.randn(2, 3, 224, 224)  # A new batch size proves the batch axis is dynamic.
    with torch.inference_mode():
        logits, features = wrapper(other)
    outputs = session(path).run(None, {"image": other.numpy()})
    assert np.abs(outputs[0] - logits.numpy()).max() < 1e-3
    assert np.abs(outputs[1] - features.numpy()).max() < 1e-3


def test_fp16_stays_close_and_shrinks(model, tmp_path) -> None:
    wrapper = ImageForExport(model)
    image = torch.randn(1, 3, 224, 224)
    source, target = tmp_path / "m.onnx", tmp_path / "h.onnx"
    export_onnx(wrapper, image, source, ["image"], ["logits", "features"])
    export_onnx(HalfImageForExport(model), image, target, ["image"], ["logits", "features"])
    assert target.stat().st_size < source.stat().st_size * 0.6
    with torch.inference_mode():
        expected = wrapper(image)[0].numpy()
    assert max_absolute_difference(target, {"image": image.numpy()}, expected) < 0.05


def test_int8_runs_and_shrinks(model, tmp_path) -> None:
    wrapper = ImageForExport(model)
    image = torch.randn(1, 3, 224, 224)
    source, target = tmp_path / "m.onnx", tmp_path / "q.onnx"
    export_onnx(wrapper, image, source, ["image"], ["logits", "features"])
    to_int8(source, target, "image", [torch.randn(1, 3, 224, 224).numpy() for _ in range(3)])
    assert target.stat().st_size < source.stat().st_size * 0.4
    assert session(target).run(None, {"image": image.numpy()})[0].shape == (1, 7)


def test_log_mel_exports_inside_the_graph(tmp_path) -> None:
    model = LogMel().eval()
    wave = torch.randn(1, CLIP_SAMPLES)
    path = tmp_path / "mel.onnx"
    export_onnx(model, wave, path, ["waveform"], ["image"])
    with torch.inference_mode():
        expected = model(wave).numpy()
    assert max_absolute_difference(path, {"waveform": wave.numpy()}, expected) < 1e-2
