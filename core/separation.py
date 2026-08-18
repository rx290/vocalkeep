"""
Core separation engine: video/audio in, speech-only video/audio out.

Pipeline: extract the audio track (ffmpeg) -> run Demucs source separation
(vocals stem vs. everything else) -> remux the vocals stem back against the
original video (ffmpeg), replacing the original mixed audio track.

VRAM-aware chunking: Demucs processes audio in overlapping segments, and the
segment length directly trades off against peak memory use. A GPU that
passes core.device's compatibility check can still run out of memory on a
long file if it has limited VRAM (a 2GB card is a real, common case, not a
hypothetical) -- so the segment length is chosen from the actual detected
VRAM, not a single fixed default that assumes a modern desktop GPU's
headroom. This is a genuine reliability feature, not a performance tweak:
without it, a low-VRAM GPU would pass the "does it work at all" check in
core.device and then still crash with an out-of-memory error partway
through a real file.
"""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from .device import pick_device, DeviceProbeResult

DEFAULT_MODEL = "htdemucs"


@dataclass
class SeparationResult:
    output_path: Path
    device_used: DeviceProbeResult
    segment_seconds: float


def _get_vram_gb(device: torch.device) -> float | None:
    """Returns available VRAM in GB for a CUDA/ROCm device, or None if it
    can't be determined (CPU, MPS, or XPU -- none of which expose the same
    total_memory query as CUDA's device_properties API)."""
    if device.type != "cuda":
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        return props.total_memory / 1e9
    except Exception:
        return None


def _segment_seconds_for_vram(vram_gb: float | None) -> int:
    """Demucs' own default segment length (htdemucs) is tuned for a card
    with several GB of headroom. Scale it down on tighter cards rather than
    risk an out-of-memory crash partway through a real file -- these
    thresholds are deliberately conservative, not tuned to the exact edge
    of failure, since a slower-but-working run beats a crash every time.

    Demucs' own CLI (`demucs.separate --segment`) requires an integer number
    of seconds -- confirmed directly against its own argparse definition,
    not assumed. And htdemucs specifically (a Transformer-based model) has a
    hard maximum of 7.8s -- confirmed by actually hitting the real error
    ("Cannot use a Transformer model with a longer segment than it was
    trained for. Maximum segment is: 7.8") when this rounded up to 8 instead
    of down. Rounding UP past a model's trained ceiling is a real failure
    mode, not just a wasted request -- always round toward 7, never 8.
    """
    if vram_gb is None:
        return 7  # CPU/MPS/XPU: rely on system RAM instead, use htdemucs' real max (floor of 7.8)
    if vram_gb < 3.0:
        return 3   # a 2GB-class card (this project's own dev machine's GPU, e.g.)
    if vram_gb < 6.0:
        return 5
    return 7  # htdemucs' real maximum segment (7.8s), floored to satisfy its int-only CLI flag


def extract_audio(video_path: Path, out_wav_path: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn",
         "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(out_wav_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")


def remux_audio(video_path: Path, new_audio_path: Path, out_video_path: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-i", str(new_audio_path),
         "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
         "-shortest", str(out_video_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg remux failed:\n{result.stderr}")


def separate_vocals(input_wav: Path, output_wav: Path, model_name: str = DEFAULT_MODEL,
                     prefer_cpu: bool = False) -> DeviceProbeResult:
    """Runs Demucs on a WAV file and writes the vocals-only stem to
    output_wav. Raises RuntimeError with Demucs' own stderr on failure
    rather than swallowing it -- a silent failure here would be far worse
    than a loud one, since the whole point of this tool is that the output
    is correct."""
    device_result = pick_device(prefer_cpu=prefer_cpu)
    vram_gb = _get_vram_gb(device_result.device)
    segment = _segment_seconds_for_vram(vram_gb)

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", model_name,
            "--two-stems", "vocals",
            "-d", device_result.device.type,
            "--segment", str(segment),
            "-o", tmpdir,
            str(input_wav),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Demucs separation failed (device={device_result.device.type}, "
                f"segment={segment}s):\n{result.stderr}"
            )

        stem_name = input_wav.stem
        vocals_path = Path(tmpdir) / model_name / stem_name / "vocals.wav"
        if not vocals_path.exists():
            raise RuntimeError(
                f"Demucs reported success but the expected output file is missing: "
                f"{vocals_path}. stdout was:\n{result.stdout}"
            )
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        vocals_path.replace(output_wav)

    return device_result


def remove_music_from_video(input_video: Path, output_video: Path,
                             model_name: str = DEFAULT_MODEL,
                             prefer_cpu: bool = False) -> SeparationResult:
    """The full pipeline: video -> extract audio -> separate -> remux.
    Cleans up its own temp files regardless of success or failure."""
    input_video = Path(input_video)
    output_video = Path(output_video)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_audio = Path(tmpdir) / "extracted.wav"
        tmp_vocals = Path(tmpdir) / "vocals.wav"

        extract_audio(input_video, tmp_audio)
        device_result = separate_vocals(tmp_audio, tmp_vocals, model_name=model_name,
                                         prefer_cpu=prefer_cpu)
        remux_audio(input_video, tmp_vocals, output_video)

        vram_gb = _get_vram_gb(device_result.device)
        segment = _segment_seconds_for_vram(vram_gb)

    return SeparationResult(output_path=output_video, device_used=device_result,
                             segment_seconds=segment)
