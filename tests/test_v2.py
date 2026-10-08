import io
import json

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from rotdet.c4net import C4Net, merge_orient, split_orient
from rotdet.predict import Detector, main


@pytest.fixture(params=[256, 384])
def detector(tmp_path, request):
    import hashlib
    torch.set_num_threads(2)
    cfg = dict(in_ch=1, stem_ch=16, widths=[32, 64, 128],
               num_classes=4, head_type="orientation")
    torch.manual_seed(42)
    model = C4Net(**cfg)
    weights = tmp_path / "model.safetensors"
    save_file(model.state_dict(), str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    (tmp_path / "config.json").write_text(json.dumps(dict(
        resolution=request.param, architecture=cfg, sha256=digest)))
    return Detector(tmp_path)


def page():
    pixels = np.arange(32 * 45, dtype=np.uint8).reshape(32, 45)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def test_preprocess_and_rotation(detector):
    data = page()
    tensor = detector.tensor(data)
    assert tensor.shape == (1, 1, detector.resolution, detector.resolution)
    expected = np.asarray(Image.open(io.BytesIO(data)).convert("RGB").convert("L")
                          .resize((detector.resolution,) * 2, Image.Resampling.BICUBIC))
    torch.testing.assert_close(tensor[0, 0], torch.from_numpy(expected.copy()).float() / 255)
    torch.testing.assert_close(detector.tensor(data, 1), torch.rot90(tensor, 1, (-2, -1)))


def test_network_equivariance(detector):
    data = page()
    with torch.inference_mode():
        base = detector.model(detector.tensor(data))
        for k in range(1, 4):
            rotated = detector.model(detector.tensor(data, k))
            torch.testing.assert_close(rotated, torch.roll(base, k, 1), atol=1e-5, rtol=1e-4)
    result = detector.predict(data)
    assert result["correction_cw_degrees"] == result["class"] * 90
    assert sum(result["probabilities"]) == pytest.approx(1)


def test_checksum_and_bad_input(detector, tmp_path):
    with pytest.raises(OSError):
        detector.predict(b"not an image")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="SHA-256"):
        Detector(tmp_path)


def test_channel_layout():
    tensor = torch.arange(1 * 12 * 4 * 4).reshape(1, 12, 4, 4)
    assert torch.equal(merge_orient(split_orient(tensor)), tensor)
    assert split_orient(tensor)[0, 1, 2, 0, 0] == tensor[0, 7, 0, 0]


def test_invalid_resolution(tmp_path):
    (tmp_path / "config.json").write_text('{"resolution": 128}')
    with pytest.raises(ValueError, match="resolutions"):
        Detector(tmp_path)


def test_cli_error_status(detector, tmp_path, capsys):
    image = tmp_path / "bad.png"
    image.write_bytes(b"invalid image")
    assert main(["--model", str(tmp_path), str(image)]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
