# -*- coding: utf-8 -*-
"""bench_full — hiway.mp4 N-frame: FPS + OCR 경로 분포 + 인식 번호판 목록.

bench_fps 보다 한 단계 깊은 가시성:
  - OCR fast-path / fallback / early-return 비율
  - fallback 평균 지연시간 + wall-clock 점유율
  - 인식된 번호판 unique 목록 + 최고 confidence + 등장 프레임 범위
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

from .bench_common import (
    HANGUL_PATTERN,
    RESULT_BAR,
    OcrStats,
    build_arg_parser,
    configure_utf8_stdout,
    fix_cwd_and_path,
    format_summary_line,
    open_video,
    patch_run_ocr,
    percent_of_wall,
    silence_engine_logs,
)

# ── 파일별 상수 ──
PROGRESS_INTERVAL: int = 100
PLATE_COLUMN_WIDTH: int = 20


# ── Plate aggregation ────────────────────────────────────────────────────
def _empty_plate_record() -> Dict[str, Any]:
    return {"count": 0, "max_conf": 0.0, "frames": []}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser(
        "hiway.mp4 N-frame FPS + OCR 경로 분포 + 인식 번호판 목록"
    )
    return parser.parse_args(argv)


def run_benchmark(
    video_path: str,
    max_frames: int,
    camera_id: str,
    stats: OcrStats,
) -> Dict[str, Any]:
    """엔진을 실행하고 (frames, elapsed, plates_seen) 을 반환."""
    from yolo11_plate.plate_engine_pro import PlateEnginePro

    engine = PlateEnginePro()
    plates_seen: Dict[str, Dict[str, Any]] = defaultdict(_empty_plate_record)
    count = 0

    with silence_engine_logs() as raw_print, open_video(video_path) as cap:
        raw_print(f"=== Benchmark: {max_frames} frames ===", flush=True)
        started = time.perf_counter()
        while count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            results = engine.process_frame(frame, camera_id)
            count += 1
            for detection in results:
                plate = detection["plate"]
                conf = detection["confidence"]
                record = plates_seen[plate]
                record["count"] += 1
                record["max_conf"] = max(record["max_conf"], conf)
                record["frames"].append(count)
            if count % PROGRESS_INTERVAL == 0:
                raw_print(f"  [{count}/{max_frames}] ...", flush=True)
        elapsed = time.perf_counter() - started

    return {
        "frames": count,
        "elapsed_s": elapsed,
        "plates_seen": dict(plates_seen),
    }


def build_report(run_result: Dict[str, Any], stats: OcrStats) -> Dict[str, Any]:
    """수치로 떨어지는 결과 딕셔너리 — print/JSON 양쪽에서 재사용."""
    frames: int = run_result["frames"]
    elapsed: float = run_result["elapsed_s"]
    plates_seen: Dict[str, Dict[str, Any]] = run_result["plates_seen"]
    fps = frames / elapsed if elapsed > 0 else 0.0

    total = stats.ocr_calls
    fb_ms = stats.fallback_ms
    fb_avg_ms = (sum(fb_ms) / len(fb_ms)) if fb_ms else 0.0
    fb_total_ms = sum(fb_ms)

    plates_report: List[Dict[str, Any]] = []
    for plate in sorted(plates_seen, key=lambda p: plates_seen[p]["frames"][0]):
        info = plates_seen[plate]
        plates_report.append({
            "plate": plate,
            "count": info["count"],
            "max_conf": info["max_conf"],
            "has_hangul": bool(HANGUL_PATTERN.search(plate)),
            "first_frame": info["frames"][0],
            "last_frame": info["frames"][-1],
        })

    return {
        "frames": frames,
        "elapsed_s": elapsed,
        "fps": fps,
        "ocr": {
            "total": total,
            "fast_path": stats.fast_path,
            "fallback": stats.fallback,
            "early_return": stats.early_return,
        },
        "fallback_ms": {
            "calls": len(fb_ms),
            "avg": fb_avg_ms,
            "total": fb_total_ms,
            "wall_percent": percent_of_wall(fb_total_ms, elapsed),
        },
        "plates": plates_report,
    }


def print_report(report: Dict[str, Any]) -> None:
    frames = report["frames"]
    elapsed = report["elapsed_s"]
    ocr = report["ocr"]
    total = ocr["total"]
    fb = report["fallback_ms"]

    print(f"\n{RESULT_BAR}")
    print(f" Results ({format_summary_line(frames, elapsed)})")
    print(RESULT_BAR)

    print(f"\n [1] OCR Path Distribution ({total} calls)")
    if total:
        fast = ocr["fast_path"]
        fbk = ocr["fallback"]
        early = ocr["early_return"]
        print(f"     Fast path (rec-only):  {fast:4d}  ({fast / total * 100:5.1f}%)")
        print(f"     Fallback (predict):    {fbk:4d}  ({fbk / total * 100:5.1f}%)")
        print(f"     Early return:          {early:4d}  ({early / total * 100:5.1f}%)")

    print("\n [2] Fallback timing:")
    if fb["calls"]:
        print(f"     Avg: {fb['avg']:.0f}ms × {fb['calls']} calls")
        print(f"     Total: {fb['total']:.0f}ms ({fb['wall_percent']:.1f}% of wall)")
    else:
        print("     (0 calls)")

    plates = report["plates"]
    print(f"\n [3] Recognized Plates ({len(plates)} unique)")
    print(
        f"     {'번호판':<{PLATE_COLUMN_WIDTH}s} {'횟수':>4s} "
        f"{'최고conf':>8s} {'한글':>4s}  프레임 범위"
    )
    print(f"     {'-' * len(RESULT_BAR)}")
    for row in plates:
        has_kr = "O" if row["has_hangul"] else "X"
        frange = (
            f"{row['first_frame']}-{row['last_frame']}"
            if row["first_frame"] != row["last_frame"]
            else str(row["first_frame"])
        )
        print(
            f"     {row['plate']:<{PLATE_COLUMN_WIDTH}s} "
            f"{row['count']:>4d} {row['max_conf']:>8.3f} "
            f"{has_kr:>4s}  {frange}"
        )

    print(RESULT_BAR)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdout()
    fix_cwd_and_path()
    args = parse_args(argv)

    stats = OcrStats()
    patch_run_ocr(stats)

    try:
        run_result = run_benchmark(args.video, args.max_frames, args.camera_id, stats)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    report = build_report(run_result, stats)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
