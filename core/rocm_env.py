"""
ROCm gfx override, applied before torch is imported.

Official ROCm builds ship kernels for a few flagship AMD architectures only.
Consumer RDNA2/RDNA3 cards next to them (an RX 6700 XT is gfx1031, an RX 6600
is gfx1032) are the same ISA as the supported gfx1030/gfx1100 but report a
different target, so every HIP kernel launch fails. Without this, torch still
says cuda is available, core.device's real matmul check fails, and the run
silently falls back to CPU. Setting HSA_OVERRIDE_GFX_VERSION makes the runtime
treat the card as its supported sibling -- the standard, documented fix.

It has to be set before the HIP runtime initialises (i.e. before torch is
imported), which is why core/__init__.py calls it. The Demucs subprocess
inherits it through os.environ. A value the user already set always wins.
"""
import glob
import os

# gfx family -> the supported target with the same ISA
_OVERRIDES = {
    "1031": "10.3.0", "1032": "10.3.0", "1033": "10.3.0", "1034": "10.3.0",
    "1035": "10.3.0", "1036": "10.3.0",
    "1101": "11.0.0", "1102": "11.0.0", "1103": "11.0.0",
}


def _gfx_targets() -> list[str]:
    """Reads each GPU's gfx target from the kernel's KFD topology (Linux
    only), e.g. 100301 -> '1031'. Returns [] anywhere that doesn't exist."""
    targets = []
    for path in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties"):
        try:
            with open(path) as f:
                for line in f:
                    key, _, value = line.partition(" ")
                    if key == "gfx_target_version" and int(value) > 0:
                        v = int(value)
                        targets.append(f"{v // 10000}{(v // 100) % 100:x}{v % 100:x}")
        except (OSError, ValueError):
            continue
    return targets


def apply_rocm_override() -> str | None:
    """Sets HSA_OVERRIDE_GFX_VERSION when the first AMD GPU needs one.
    Returns the value now in effect, or None if no override applies."""
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return os.environ["HSA_OVERRIDE_GFX_VERSION"]
    for target in _gfx_targets():
        if target in _OVERRIDES:
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = _OVERRIDES[target]
            return _OVERRIDES[target]
        return None  # first GPU is natively supported (or unknown): leave it alone
    return None
