"""
plate_recognition_patch.py - YouTube/Local video plate recognition (CLI)

Usage:
    python plate_recognition_patch.py -s "https://www.youtube.com/watch?v=XXXX"
    python plate_recognition_patch.py -s "video.mp4"
    python plate_recognition_patch.py -s "video.mp4" -o ./results
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def is_youtube_url(source: str) -> bool:
    return any(d in source for d in ["youtube.com", "youtu.be", "youtube.co", "yt.be"])


def download_youtube(url: str, output_dir: str = "./temp_youtube") -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(title).80s.%(ext)s")

    format_str = (
        "bestvideo[height>=2160]+bestaudio/best[height>=2160]/"
        "bestvideo[height>=1440]+bestaudio/best[height>=1440]/"
        "bestvideo+bestaudio/best"
    )

    cmd = [
        "yt-dlp",
        "-f", format_str,
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        url,
    ]

    print(f"  [YouTube] Downloading (best quality)...")

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
    except FileNotFoundError:
        print("[ERROR] yt-dlp not installed: pip install yt-dlp")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[ERROR] Download timed out (10 min)")
        sys.exit(1)

    stderr_text = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        print(f"[ERROR] Download failed:\n{stderr_text}")
        sys.exit(1)

    filepath = None
    for f in os.listdir(output_dir):
        if f.endswith(".mp4"):
            candidate = os.path.join(output_dir, f)
            if filepath is None or os.path.getmtime(candidate) > os.path.getmtime(filepath):
                filepath = candidate

    if filepath is None or not os.path.exists(filepath):
        print(f"[ERROR] Downloaded file not found in {output_dir}")
        sys.exit(1)

    mb = os.path.getsize(filepath) / (1024 * 1024)
    safe_name = os.path.basename(filepath).encode("ascii", errors="replace").decode("ascii")
    print(f"  [YouTube] Done: {safe_name} ({mb:.1f}MB)")
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(description="Plate recognition from YouTube/local video")
    parser.add_argument("-s", "--source", required=True, help="Video source (file path or YouTube URL)")
    parser.add_argument("-o", "--output", default="./plate_results", help="Output directory")
    parser.add_argument("--model", default=None, help="Custom YOLO model .pt path")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--no-sahi", action="store_true", help="Disable SAHI tiling")
    parser.add_argument("--frame-skip", type=int, default=5, help="Frame skip interval")
    parser.add_argument("--keep-video", action="store_true", help="Keep downloaded video after processing")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Plate Recognition Patch")
    print("=" * 60)
    print(f"  Source: {args.source}")
    print(f"  Output: {args.output}")
    print("=" * 60)

    start_time = time.time()
    temp_dir = None

    # Resolve video source
    if is_youtube_url(args.source):
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_youtube")
        video_path = download_youtube(args.source, temp_dir)
    else:
        if not os.path.isfile(args.source):
            print(f"[ERROR] File not found: {args.source}")
            sys.exit(1)
        video_path = args.source

    # Run plate recognition
    print(f"\n  [Recognition] Loading model...")
    from plate_recognition_4k import PlateRecognizer

    recognizer = PlateRecognizer(
        model_path=args.model,
        confidence_threshold=args.confidence,
        use_sahi=not args.no_sahi,
        frame_skip=args.frame_skip,
    )

    print(f"  [Recognition] Processing video...")
    results = recognizer.process_video(video_path, args.output)

    # Print results
    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    if results:
        valid_count = sum(1 for r in results if r.get("is_valid_plate"))
        print(f"  Total: {len(results)} plates (valid: {valid_count})\n")
        for i, r in enumerate(results):
            mark = " [VALID]" if r.get("is_valid_plate") else ""
            print(
                f"  #{i}: \"{r['text']}\"{mark} "
                f"| OCR: {r['ocr_confidence']:.3f} "
                f"| Det: {r['detection_confidence']:.3f} "
                f"| Method: {r.get('detection_method', '?')}"
            )
    else:
        print("  No plates found.")

    # Confirmed plates from tracker
    confirmed = recognizer.get_confirmed_plates()
    if confirmed:
        print(f"\n  Confirmed plates ({len(confirmed)}):")
        for i, cp in enumerate(confirmed):
            print(f"    {i+1}. {cp.get('text', '?')} (score={cp.get('pattern_score', 0):.2f})")

    # Cleanup
    if temp_dir and not args.keep_video:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\n  [Cleanup] Removed {temp_dir}")
        except Exception:
            pass

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
