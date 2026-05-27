# -*- coding: utf-8 -*-
"""bench_fps — hiway.mp4 N-frame FPS 벤치마크 (headless).

플레이트 엔진 풀 파이프라인을 N프레임 돌려 평균 FPS / ms/frame 만 리포트.
가장 가벼운 회귀 가드용.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict

from .bench_common import (
    PROGRESS_INTERVAL,
    RESULT_BAR,
    build_arg_parser,
    configure_utf8_stdout,
    fix_cwd_and_path,
    open_video,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser("hiway.mp4 N-frame FPS 벤치마크 (headless)")
    return parser.parse_args(argv)


def run_benchmark(
    video_path: str, max_frames: int, camera_id: str
) -> Dict[str, Any]:
    """비디오 max_frames 만큼 process_frame 호출, FPS 계산."""
    # 헤비 의존성은 patch/세팅 끝난 뒤에 import
    from yolo11_plate.plate_engine_pro import PlateEnginePro

    engine = PlateEnginePro()
    print(f"=== FPS Benchmark: {max_frames} frames ===", flush=True)

    frame_count = 0
    started = time.perf_counter()

    with open_video(video_path) as cap:
        while frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            engine.process_frame(frame, camera_id)
            frame_count += 1
            if frame_count % PROGRESS_INTERVAL == 0:
                elapsed = time.perf_counter() - started
                fps = frame_count / elapsed
                print(
                    f"  [{frame_count}/{max_frames}] "
                    f"{fps:.1f} FPS ({elapsed:.1f}s)",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    return {
        "frames": frame_count,
        "elapsed_s": elapsed,
        "fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "ms_per_frame": (elapsed / frame_count * 1000) if frame_count else 0.0,
    }


def print_result(result: Dict[str, Any]) -> None:
    print(f"\n{RESULT_BAR}")
    print(" Result")
    print(RESULT_BAR)
    print(f"  Frames:   {result['frames']}")
    print(f"  Time:     {result['elapsed_s']:.1f}s")
    print(f"  FPS:      {result['fps']:.1f}")
    print(f"  ms/frame: {result['ms_per_frame']:.0f}")
    print(RESULT_BAR)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdout()
    fix_cwd_and_path()
    args = parse_args(argv)

    try:
        result = run_benchmark(args.video, args.max_frames, args.camera_id)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
