# -*- coding: utf-8 -*-
"""bench_profile — hiway.mp4 N-frame: OCR + YOLO 시간 분해 프로파일링.

bench_full 보다 한 단계 더 깊은 가시성:
  - YOLO vehicle(416) / plate(640) 모델별 평균 추론시간
  - per-frame 시간 분해 (YOLO / Fallback / OCR+기타)

소스 코드는 건드리지 않고 monkey-patch 로만 계측한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict

from bench_common import (
    RESULT_BAR,
    OcrStats,
    YoloStats,
    build_arg_parser,
    configure_utf8_stdout,
    fix_cwd_and_path,
    format_summary_line,
    open_video,
    patch_run_ocr,
    patch_yolo_call,
    percent_of_wall,
    silence_engine_logs,
)

PROGRESS_INTERVAL: int = 100


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser(
        "hiway.mp4 N-frame OCR + YOLO 시간 분해 프로파일링"
    )
    return parser.parse_args(argv)


def run_benchmark(
    video_path: str, max_frames: int, camera_id: str
) -> Dict[str, Any]:
    """엔진 가동 후 (frames, elapsed_s) 반환. 계측은 stats 객체에 누적."""
    from plate_engine_pro import PlateEnginePro

    engine = PlateEnginePro()
    count = 0

    with silence_engine_logs() as raw_print, open_video(video_path) as cap:
        raw_print(f"=== Profiling: {max_frames} frames ===", flush=True)
        started = time.perf_counter()
        while count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            engine.process_frame(frame, camera_id)
            count += 1
            if count % PROGRESS_INTERVAL == 0:
                raw_print(f"  [{count}/{max_frames}] ...", flush=True)
        elapsed = time.perf_counter() - started

    return {"frames": count, "elapsed_s": elapsed}


def build_report(
    run_result: Dict[str, Any], ocr: OcrStats, yolo: YoloStats
) -> Dict[str, Any]:
    frames: int = run_result["frames"]
    elapsed: float = run_result["elapsed_s"]
    fps = frames / elapsed if elapsed > 0 else 0.0

    fb_ms_total = sum(ocr.fallback_ms)
    yolo_vehicle_total = sum(yolo.vehicle_ms)
    yolo_plate_total = sum(yolo.plate_ms)
    yolo_total = yolo_vehicle_total + yolo_plate_total
    other_total_ms = max(0.0, elapsed * 1000 - yolo_total - fb_ms_total)

    return {
        "frames": frames,
        "elapsed_s": elapsed,
        "fps": fps,
        "ocr": {
            "calls": ocr.ocr_calls,
            "fast_path": ocr.fast_path,
            "fallback": ocr.fallback,
            "early_return": ocr.early_return,
        },
        "fallback_ms": {
            "calls": len(ocr.fallback_ms),
            "avg": (sum(ocr.fallback_ms) / len(ocr.fallback_ms))
                    if ocr.fallback_ms else 0.0,
            "total": fb_ms_total,
            "wall_percent": percent_of_wall(fb_ms_total, elapsed),
        },
        "yolo": {
            "vehicle_calls": len(yolo.vehicle_ms),
            "vehicle_avg_ms": (yolo_vehicle_total / len(yolo.vehicle_ms))
                              if yolo.vehicle_ms else 0.0,
            "plate_calls": len(yolo.plate_ms),
            "plate_avg_ms": (yolo_plate_total / len(yolo.plate_ms))
                            if yolo.plate_ms else 0.0,
            "total_ms": yolo_total,
            "per_frame_ms": (yolo_total / frames) if frames else 0.0,
            "wall_percent": percent_of_wall(yolo_total, elapsed),
        },
        "decomposition_ms_per_frame": {
            "yolo": (yolo_total / frames) if frames else 0.0,
            "fallback": (fb_ms_total / frames) if frames else 0.0,
            "other": (other_total_ms / frames) if frames else 0.0,
        },
    }


def print_report(report: Dict[str, Any]) -> None:
    frames = report["frames"]
    elapsed = report["elapsed_s"]
    ocr = report["ocr"]
    fb = report["fallback_ms"]
    yolo = report["yolo"]
    decomp = report["decomposition_ms_per_frame"]
    total_calls = ocr["calls"]

    print(f"\n{RESULT_BAR}")
    print(f" Profiling Results  ({format_summary_line(frames, elapsed)})")
    print(RESULT_BAR)

    print(f"\n 1. _run_ocr 호출 합계:    {total_calls}")
    if total_calls:
        fast = ocr["fast_path"]
        fbk = ocr["fallback"]
        early = ocr["early_return"]
        print(f"    Fast path (rec-only):  {fast:4d}  ({fast / total_calls * 100:5.1f}%)")
        print(f"    Fallback (predict):    {fbk:4d}  ({fbk / total_calls * 100:5.1f}%)")
        print(f"    Early return (empty):  {early:4d}  ({early / total_calls * 100:5.1f}%)")

    print("\n 2. Fallback 소요시간:")
    if fb["calls"]:
        print(f"    평균: {fb['avg']:.0f}ms")
        print(f"    총합: {fb['total']:.0f}ms  ({fb['wall_percent']:.1f}% of wall time)")
    else:
        print("    (0 calls)")

    print("\n 3. YOLO 소요시간:")
    if yolo["vehicle_calls"]:
        print(
            f"    yolov8n (vehicle, 416):  "
            f"{yolo['vehicle_avg_ms']:.1f}ms avg × {yolo['vehicle_calls']} calls"
        )
    if yolo["plate_calls"]:
        print(
            f"    best.pt (plate, 640):    "
            f"{yolo['plate_avg_ms']:.1f}ms avg × {yolo['plate_calls']} calls"
        )
    if yolo["vehicle_calls"] and yolo["plate_calls"]:
        print(f"    YOLO 합계/프레임:        {yolo['per_frame_ms']:.1f}ms avg")
        print(f"    YOLO 총합:              {yolo['total_ms']:.0f}ms  ({yolo['wall_percent']:.1f}% of wall time)")

    print("\n 4. 시간 분해 (per frame):")
    print(f"    YOLO:       {decomp['yolo']:6.1f}ms  ({percent_of_wall(decomp['yolo'] * frames, elapsed):5.1f}%)")
    print(f"    Fallback:   {decomp['fallback']:6.1f}ms  ({percent_of_wall(decomp['fallback'] * frames, elapsed):5.1f}%)")
    print(f"    OCR+기타:   {decomp['other']:6.1f}ms  ({percent_of_wall(decomp['other'] * frames, elapsed):5.1f}%)")
    print(RESULT_BAR)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdout()
    fix_cwd_and_path()
    args = parse_args(argv)

    ocr_stats = OcrStats()
    yolo_stats = YoloStats()
    patch_run_ocr(ocr_stats)
    patch_yolo_call(yolo_stats)

    try:
        run_result = run_benchmark(args.video, args.max_frames, args.camera_id)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    report = build_report(run_result, ocr_stats, yolo_stats)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
