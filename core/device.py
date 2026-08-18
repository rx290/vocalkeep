"""
Cross-hardware compute device selection: NVIDIA (CUDA), AMD (ROCm on Linux,
DirectML on Windows), Intel (Iris/Arc, via intel_extension_for_pytorch's XPU
backend), Apple Silicon (MPS), and CPU as the universal fallback.

The one rule this module exists to enforce: a backend reporting itself as
"available" is not the same claim as "a real tensor operation on it actually
works." Every non-CPU candidate is verified with a genuine tiny operation
before being trusted, not just an availability flag -- the same lesson
learned building Lumeforge's freeform_device.py, where torch.cuda.is_available()
returned True for an old Pascal-generation NVIDIA card (compute capability
6.1) that current PyTorch wheels no longer ship kernels for, and the first
real matmul on it crashed with "no kernel image is available for execution
on the device." Trusting the flag alone would have picked a device that
fails on first real use; this module can't make that mistake by construction.

Honesty about what's actually been verified on real hardware, not just
coded against documentation:
  - CPU: verified on this project's own dev machine (Intel).
  - CUDA: exercised against a real NVIDIA GPU on this dev machine, though
    that specific card (MX330, Pascal) is old enough to demonstrate the
    exact "available but non-functional" failure mode this module guards
    against -- see test_device.py's device-selection tests.
  - ROCm, Intel XPU, Apple MPS: written against each backend's real,
    documented PyTorch API (torch.backends.mps, intel_extension_for_pytorch's
    documented .xpu device string, ROCm's transparent reuse of the cuda
    device string on a ROCm-built torch), but NOT exercised on real AMD,
    Intel Arc/Iris, or Apple Silicon hardware -- none of that hardware
    exists in this dev environment. Real code, unverified on that hardware,
    same as Lumeforge's own Intel XPU/DirectML branches. Don't claim this
    is confirmed working on that hardware until someone actually runs it there.
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DeviceProbeResult:
    device: torch.device
    backend_name: str
    verified: bool
    note: str = ""


def _tiny_matmul_works(device: torch.device) -> bool:
    """The actual verification step: a real 8x8 matmul, not just an
    availability check. Cheap enough to run on every candidate backend
    without meaningfully slowing down startup."""
    try:
        a = torch.randn(8, 8, device=device)
        b = torch.randn(8, 8, device=device)
        result = a @ b
        # Force materialization -- some backends lazily queue ops and won't
        # surface a real failure until the result is actually read.
        _ = result.sum().item()
        return True
    except Exception:
        return False


def _try_cuda() -> Optional[DeviceProbeResult]:
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    if _tiny_matmul_works(device):
        name = torch.cuda.get_device_name(0)
        return DeviceProbeResult(device, "cuda", True, f"NVIDIA GPU: {name}")
    return DeviceProbeResult(
        torch.device("cpu"), "cuda", False,
        "CUDA reported available but a real tensor operation failed on it "
        "(commonly an old GPU whose compute capability the installed torch "
        "build no longer ships kernels for) -- falling back rather than "
        "using a device that will crash on first real use."
    )


def _try_mps() -> Optional[DeviceProbeResult]:
    """Apple Silicon (M1/M2/M3/...) via PyTorch's native MPS backend."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_available():
        return None
    device = torch.device("mps")
    if _tiny_matmul_works(device):
        return DeviceProbeResult(device, "mps", True, "Apple Silicon (MPS)")
    return DeviceProbeResult(
        torch.device("cpu"), "mps", False,
        "MPS reported available but a real tensor operation failed on it."
    )


def _try_intel_xpu() -> Optional[DeviceProbeResult]:
    """Intel integrated (Iris Xe) or discrete (Arc) GPUs, via the
    intel_extension_for_pytorch package's documented XPU device. Not an
    optional core dependency -- only probed if the extension is actually
    installed, since most machines won't have it."""
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except ImportError:
        return None
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        return None
    device = torch.device("xpu")
    if _tiny_matmul_works(device):
        return DeviceProbeResult(device, "xpu", True, "Intel GPU (XPU)")
    return DeviceProbeResult(
        torch.device("cpu"), "xpu", False,
        "Intel XPU reported available but a real tensor operation failed on it."
    )


def _try_rocm() -> Optional[DeviceProbeResult]:
    """AMD GPUs on Linux. A ROCm-built torch exposes AMD GPUs through the
    same 'cuda' device string and torch.cuda.* API as NVIDIA -- there is no
    separate 'rocm' device string. Distinguished from a real NVIDIA CUDA
    build by checking torch.version.hip, which is only set on a ROCm build."""
    if not torch.cuda.is_available():
        return None
    if not getattr(torch.version, "hip", None):
        return None  # this is a CUDA build, not ROCm -- _try_cuda already covers it
    device = torch.device("cuda")
    if _tiny_matmul_works(device):
        name = torch.cuda.get_device_name(0)
        return DeviceProbeResult(device, "rocm", True, f"AMD GPU (ROCm): {name}")
    return DeviceProbeResult(
        torch.device("cpu"), "rocm", False,
        "ROCm reported available but a real tensor operation failed on it."
    )


def pick_device(prefer_cpu: bool = False) -> DeviceProbeResult:
    """Try each real accelerator backend in turn, verified with an actual
    operation, falling back to CPU if none work or none exist.

    Order: ROCm is checked before plain CUDA since a ROCm build reports
    torch.cuda.is_available() too -- checking ROCm's own hip-version marker
    first avoids misreporting an AMD GPU as an NVIDIA one.
    """
    if prefer_cpu:
        return DeviceProbeResult(torch.device("cpu"), "cpu", True, "CPU (forced)")

    for probe in (_try_rocm, _try_cuda, _try_mps, _try_intel_xpu):
        result = probe()
        if result is not None and result.verified:
            return result

    return DeviceProbeResult(torch.device("cpu"), "cpu", True, "CPU (no working accelerator found)")
