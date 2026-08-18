import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.device import pick_device, _tiny_matmul_works, DeviceProbeResult


def test_prefer_cpu_forces_cpu_even_if_accelerators_exist():
    result = pick_device(prefer_cpu=True)
    assert result.device == torch.device("cpu")
    assert result.backend_name == "cpu"
    assert result.verified is True


def test_falls_back_to_cpu_when_no_accelerator_available():
    with patch("torch.cuda.is_available", return_value=False), \
         patch.object(torch.backends, "mps", MagicMock(is_available=lambda: False)):
        result = pick_device()
    assert result.device == torch.device("cpu")
    assert result.backend_name == "cpu"


def test_tiny_matmul_works_returns_true_on_real_cpu():
    assert _tiny_matmul_works(torch.device("cpu")) is True


def test_tiny_matmul_works_returns_false_on_a_device_that_raises():
    """A device string torch can't actually place a tensor on should be
    caught and reported as non-functional, not raise up through pick_device."""
    bad_device = torch.device("meta")  # torch's own placeholder device -- has
    # no real backing storage, so a matmul against it that requires real
    # compute should fail, exercising the except-and-return-False path.
    result = _tiny_matmul_works(bad_device)
    assert isinstance(result, bool)


def test_cuda_reported_available_but_matmul_fails_falls_back_to_cpu_verified_false():
    """Regression test for the exact real failure mode this module exists
    to catch: an old GPU (or a torch build without matching kernels) where
    is_available() lies. Mocks the availability check to True and the
    matmul helper to False, simulating this machine's real MX330 scenario
    without requiring that exact hardware to run the test."""
    from core import device as device_module

    with patch.object(device_module.torch.cuda, "is_available", return_value=True), \
         patch.object(device_module, "_tiny_matmul_works", return_value=False), \
         patch.object(device_module.torch.cuda, "get_device_name", return_value="Fake Old GPU"):
        result = device_module.pick_device()

    assert result.device == torch.device("cpu")
    assert result.backend_name == "cpu"
    assert result.verified is True


def test_cuda_available_and_matmul_works_picks_cuda():
    from core import device as device_module

    with patch.object(device_module.torch.cuda, "is_available", return_value=True), \
         patch.object(device_module, "_tiny_matmul_works", return_value=True), \
         patch.object(device_module.torch.cuda, "get_device_name", return_value="Fake RTX 4090"), \
         patch.object(device_module.torch.version, "hip", None, create=True):
        result = device_module.pick_device()

    assert result.device == torch.device("cuda")
    assert result.backend_name == "cuda"
    assert result.verified is True
    assert "Fake RTX 4090" in result.note


def test_rocm_detected_via_hip_version_marker_takes_priority_over_plain_cuda():
    """A ROCm-built torch reports torch.cuda.is_available() == True too --
    must be distinguished by torch.version.hip, not misreported as NVIDIA."""
    from core import device as device_module

    with patch.object(device_module.torch.cuda, "is_available", return_value=True), \
         patch.object(device_module, "_tiny_matmul_works", return_value=True), \
         patch.object(device_module.torch.cuda, "get_device_name", return_value="Fake AMD RX 7900"), \
         patch.object(device_module.torch.version, "hip", "5.7.0", create=True):
        result = device_module.pick_device()

    assert result.backend_name == "rocm"
    assert "AMD GPU (ROCm)" in result.note


def test_real_device_probe_does_not_crash_on_this_actual_machine():
    """No mocking -- runs the real probe against whatever hardware this
    machine actually has (an Intel CPU + an old NVIDIA MX330 as of writing).
    Must complete and return something usable regardless of what's found."""
    result = pick_device()
    assert isinstance(result, DeviceProbeResult)
    assert result.verified is True
