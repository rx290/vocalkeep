import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.separation import (
    _get_vram_gb, _segment_seconds_for_vram, extract_audio, remux_audio,
    separate_vocals,
)
from core.device import DeviceProbeResult


def test_segment_shrinks_for_low_vram_card():
    """The exact real case this exists for: a 2GB-class GPU (this project's
    own dev machine's card, when/if it becomes usable) must get a smaller
    segment than Demucs' own desktop-GPU-sized default."""
    assert _segment_seconds_for_vram(2.0) == 3
    assert _segment_seconds_for_vram(2.0) < _segment_seconds_for_vram(8.0)


def test_segment_is_always_an_integer_since_demucs_cli_requires_one():
    """Confirmed directly against demucs.separate's own --help output: the
    --segment flag rejects a float ('invalid int value: 7.8'), caught by
    actually running the CLI, not assumed from documentation."""
    for vram in (None, 2.0, 4.0, 24.0):
        assert isinstance(_segment_seconds_for_vram(vram), int)


def test_segment_uses_demucs_default_for_ample_vram():
    assert _segment_seconds_for_vram(24.0) == 7


def test_segment_never_exceeds_htdemucs_transformer_max_of_7_point_8s():
    """Regression test for a real error hit while building this: rounding
    the 7.8s ceiling UP to 8 (instead of down to 7) causes demucs.separate
    to hard-fail with 'Cannot use a Transformer model with a longer segment
    than it was trained for.' Every VRAM tier must stay at or under 7."""
    for vram in (None, 1.0, 2.0, 4.0, 8.0, 100.0):
        assert _segment_seconds_for_vram(vram) <= 7


def test_segment_for_none_vram_falls_back_to_default_not_a_crash():
    """CPU/MPS/XPU devices don't expose the same VRAM query CUDA does --
    must not raise, must return a sane default."""
    assert _segment_seconds_for_vram(None) == 7


def test_get_vram_gb_returns_none_for_non_cuda_device():
    assert _get_vram_gb(torch.device("cpu")) is None


def test_get_vram_gb_returns_none_gracefully_if_cuda_query_raises():
    """If a CUDA device is reported but querying its properties fails for
    any reason, this must degrade to 'unknown VRAM', not crash the whole
    pipeline over a diagnostic-only query."""
    with patch("torch.cuda.get_device_properties", side_effect=RuntimeError("no such device")):
        assert _get_vram_gb(torch.device("cuda")) is None


def test_extract_audio_raises_runtime_error_with_ffmpeg_stderr_on_failure(tmp_path):
    """A nonexistent input file should surface ffmpeg's real error message,
    not fail silently or with an unrelated exception."""
    nonexistent = tmp_path / "does_not_exist.mp4"
    out_wav = tmp_path / "out.wav"
    with pytest.raises(RuntimeError) as exc_info:
        extract_audio(nonexistent, out_wav)
    assert "ffmpeg" in str(exc_info.value).lower()


def test_remux_audio_raises_runtime_error_on_failure(tmp_path):
    nonexistent_video = tmp_path / "no_video.mp4"
    nonexistent_audio = tmp_path / "no_audio.wav"
    out = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError):
        remux_audio(nonexistent_video, nonexistent_audio, out)


def test_separate_vocals_raises_with_demucs_stderr_on_failure(tmp_path):
    """Mocks subprocess to simulate Demucs itself failing -- must surface
    the real stderr in the raised error, not swallow it."""
    fake_input = tmp_path / "input.wav"
    fake_input.write_bytes(b"not a real wav, just needs to exist")
    fake_output = tmp_path / "vocals.wav"

    fake_device = DeviceProbeResult(torch.device("cpu"), "cpu", True, "CPU (forced)")
    fake_result = MagicMock(returncode=1, stderr="demucs: model download failed", stdout="")

    with patch("core.separation.pick_device", return_value=fake_device), \
         patch("core.separation.subprocess.run", return_value=fake_result):
        with pytest.raises(RuntimeError) as exc_info:
            separate_vocals(fake_input, fake_output)

    assert "demucs" in str(exc_info.value).lower()
    assert "model download failed" in str(exc_info.value)


def test_separate_vocals_raises_if_output_file_missing_despite_success_returncode(tmp_path):
    """Defends against Demucs reporting success (returncode 0) but not
    actually producing the expected output file -- a real failure mode
    worth catching explicitly rather than silently returning an empty/
    missing result."""
    fake_input = tmp_path / "input.wav"
    fake_input.write_bytes(b"placeholder")
    fake_output = tmp_path / "vocals.wav"

    fake_device = DeviceProbeResult(torch.device("cpu"), "cpu", True, "CPU (forced)")
    fake_result = MagicMock(returncode=0, stderr="", stdout="Separated tracks will be stored in ...")

    with patch("core.separation.pick_device", return_value=fake_device), \
         patch("core.separation.subprocess.run", return_value=fake_result):
        with pytest.raises(RuntimeError) as exc_info:
            separate_vocals(fake_input, fake_output)

    assert "missing" in str(exc_info.value).lower()
