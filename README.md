# VocalKeep

Strips background music from a video while keeping the speech — for reels, shorts, and clips where the music makes it hard to hear what's actually being said.

## Why this isn't one project pretending to be simple

"Remove music from any video, live, on any social media app" is actually three or four different engineering problems bundled into one sentence, and one of them runs into a hard platform wall. Built in phases, each one proving the next is worth attempting before starting it:

- **Phase 1 (done): core engine, offline/batch.** A local CLI: video in, speech-only video out. No cloud, no browser, no live processing — just the actual audio separation working correctly and reliably across as much real hardware as possible.
- **Phase 2 (done): local web UI.** Drag a video in, get the processed one back. Still fully local, still no browser injection — proves the pipeline is usable by something other than a terminal.
- **Phase 3: Chrome extension, offline/batch mode.** Right-click a video or upload one, same processing, talking to a small local backend service (Demucs is too heavy to run inside a browser tab via WASM).
- **Phase 4: live processing.** Two genuinely different sub-problems: live mode for video the browser can actually access the raw audio of, and a separate Android app using `AudioPlaybackCapture` for live audio from other apps. iOS has no equivalent API for third-party apps under normal App Store rules, so that path doesn't exist here.

## Web UI (Phase 2)

```bash
.venv/bin/python web/app.py          # then open http://127.0.0.1:8610
```

Drag a video onto the page, pick the model, watch the preview, download the clean file. It is a small standard-library server (no extra packages), bound to localhost, running one job at a time so the GPU is never shared between two videos. Uploads stream to disk, finished files are deleted an hour later or when you stop the server.

## How it works

[Demucs](https://github.com/facebookresearch/demucs) (Meta's open-source music source-separation model) splits the audio into a vocals stem and everything else. `ffmpeg` extracts the original audio, Demucs runs the separation, and the vocals stem gets remuxed back against the original video, replacing the original mixed track.

## Built to run on real, varied hardware — not just the one machine it was written on

The compute backend is chosen automatically, in this order: NVIDIA (CUDA), AMD (ROCm on Linux), Apple Silicon (MPS), Intel Iris/Arc (via `intel_extension_for_pytorch`'s XPU backend), falling back to CPU if none of those work. The one rule enforced throughout: **a backend reporting itself as "available" is not verified as actually working.** Every candidate is tested with a real tiny tensor operation before being trusted — this project's own dev machine has a real, live example of why that matters: an old NVIDIA MX330 (Pascal architecture) that `torch.cuda.is_available()` reports as present, but that current PyTorch builds no longer ship compiled kernels for at all. Trusting the availability flag alone would pick a device that crashes on the very first real operation; `core/device.py` catches that and falls back to CPU cleanly instead.

**VRAM-aware processing.** A GPU that passes the compatibility check can still run out of memory on a long video if it has limited VRAM. Demucs' processing segment length (which trades off directly against peak memory use) is chosen from the actual detected VRAM rather than assuming every GPU has a modern desktop card's headroom — a 2GB-class card gets a smaller segment automatically, not a crash partway through a real file.

**Honesty about what's actually been verified, not just coded against documentation:**
- CPU and NVIDIA/CUDA: exercised for real on this project's own dev machine.
- AMD ROCm: verified on an RX 6700 XT (gfx1031, ROCm 7.0, torch 2.10). A 52 s 480p trailer took 11.5 s end to end on the GPU, about 4.5x real time including model load; music-only stretches dropped from about 70 dB to near silence while the dialogue stayed. Cards like this one need `HSA_OVERRIDE_GFX_VERSION`, which `core/rocm_env.py` now sets automatically from the kernel's GPU topology (before torch loads) for RDNA2/RDNA3 consumer cards; a value you set yourself always wins.
- Intel XPU, Apple MPS: written against each backend's real, documented PyTorch API, but not yet run on that actual hardware. Real code, honestly marked as unverified until someone runs it there.

## A note on old/incompatible GPUs

If `torch.cuda.is_available()` reports `True` but the device fails the real verification check, that's usually not a speed problem — it's the installed PyTorch build having no compiled kernel for that specific GPU architecture at all, a hard incompatibility rather than a "runs slower" tradeoff. The legitimate way to salvage a genuinely old, unsupported GPU is building PyTorch from source with that GPU's exact compute capability included in `TORCH_CUDA_ARCH_LIST` — a real, heavier undertaking, not something this project attempts automatically. Worth doing if you have hardware like this and want to actually use it; not something to fake.

## Install

```bash
git clone https://github.com/rx290/vocalkeep.git
cd vocalkeep
python3 -m venv .venv

# CPU only:
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA (CUDA), the default install:
.venv/bin/pip install torch

# AMD (ROCm, Linux only) -- check pytorch.org for the current ROCm version tag:
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm6.1

.venv/bin/pip install demucs
sudo apt install ffmpeg   # or brew install ffmpeg, or ffmpeg.org
```

## Usage

```bash
.venv/bin/python cli/main.py input.mp4 -o output.mp4
.venv/bin/python cli/main.py input.mp4 -o output.mp4 --cpu    # force CPU even if a GPU is usable
.venv/bin/python cli/main.py input.mp4 -o output.mp4 --model htdemucs_ft   # slower, higher quality
```

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -v
```

## Status

**Phase 1 complete (2026-08-18):** device selection, VRAM-aware segment sizing, and the full separation pipeline built, unit-tested (19 tests), and verified end-to-end with a real run — a synthetic test video (ffmpeg test pattern + two sine tones, no copyrighted content) went through extraction → Demucs separation → remux and produced a valid output video with both video and audio streams intact, on the CPU path.

Two real bugs were caught only by that real run, not by the unit tests alone: the separation subprocess was calling the system `python3` instead of the virtualenv's own interpreter (so it couldn't find the installed `demucs` package), and the VRAM-based segment-length calculation rounded up to a whole-second value that exceeded `htdemucs`'s actual hard maximum (a Transformer-based model capped at 7.8s) instead of rounding down — both fixed, both now covered by regression tests.

Not yet verified: the CUDA/GPU path on this machine (this dev environment's own GPU, an old NVIDIA MX330, is a known non-functional case per `core/device.py`'s own design), and the AMD/ROCm path (no AMD hardware in this dev environment — will be verified on a real AMD RX 6700 XT/12GB workstation once available). Phases 2-4 (local web UI, Chrome extension, live processing) not yet started.

## Author

**Muhammad Asad Waseem**, [github.com/rx290](https://github.com/rx290)
