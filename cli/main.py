#!/usr/bin/env python3
"""
VocalKeep CLI: strip background music from a video, keep the speech.

    vocalkeep input.mp4 -o output.mp4
    vocalkeep input.mp4 -o output.mp4 --cpu          # force CPU even if a GPU is usable
    vocalkeep input.mp4 -o output.mp4 --model htdemucs_ft   # higher-quality, slower model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.separation import remove_music_from_video, DEFAULT_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=str, help="Input video file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output video file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                         help=f"Demucs model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a working GPU is found")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Processing {input_path} ...")
    try:
        result = remove_music_from_video(
            input_path, Path(args.output), model_name=args.model, prefer_cpu=args.cpu,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Done: {result.output_path}")
    print(f"  Device: {result.device_used.backend_name} ({result.device_used.note})")
    print(f"  Segment length used: {result.segment_seconds}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
