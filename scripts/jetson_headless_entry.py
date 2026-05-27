# -*- coding: utf-8 -*-
"""
Jetson 실차 headless 엔트리: 영상/RTSP 입력, 로그·evidence 경로 지정, 최대 시간/프레임 제한.

사용법:
  python scripts/jetson_headless_entry.py --video /data/dashcam/seg.mp4 --out-dir /data/evidence --logs-dir logs
  python scripts/jetson_headless_entry.py --rtsp "rtsp://192.168.1.100:554/stream1" --max-duration 1800
"""
import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

if "PLATE_CONSECUTIVE_FRAMES" not in os.environ:
    os.environ["PLATE_CONSECUTIVE_FRAMES"] = "1"


def main():
    p = argparse.ArgumentParser(description="Jetson headless run with log/evidence dirs")
    p.add_argument("--video", type=str, help="영상 파일 경로")
    p.add_argument("--rtsp", type=str, help="RTSP URL (예: rtsp://ip:554/stream1)")
    p.add_argument("--out-dir", type=str, default=os.environ.get("EVIDENCE_OUTPUT_DIR", str(ROOT / "evidence_output")))
    p.add_argument("--logs-dir", type=str, default=str(ROOT / "logs"))
    p.add_argument("--max-duration", type=float, default=0, help="최대 실행 시간(초), 0=무제한")
    p.add_argument("--max-frames", type=int, default=0, help="최대 프레임 수, 0=무제한")
    args = p.parse_args()

    source = args.video or args.rtsp
    if not source:
        print("--video 또는 --rtsp 필요")
        sys.exit(1)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    import cv2
    from yolo11_plate.plate_engine_pro import PlateEnginePro

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG if args.rtsp else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"영상/스트림 열기 실패: {source}")
        sys.exit(1)
    if args.rtsp:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    max_frames = int(args.max_duration * fps) if args.max_duration > 0 else (args.max_frames or 0)
    if max_frames == 0:
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 999999

    engine = PlateEnginePro()
    n = 0
    plates_seen = []
    t0 = time.perf_counter()
    while n < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        results = engine.process_frame(frame, "HEADLESS", use_multiframe=False)
        for r in results:
            p = r.get("plate", "")
            if p and p not in plates_seen:
                plates_seen.append(p)
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{max_frames} 프레임, 인식 {len(plates_seen)}건...", flush=True)
    cap.release()
    elapsed = time.perf_counter() - t0

    # 메트릭 요약 (CSV 한 줄)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    fps_actual = n / elapsed if elapsed > 0 else 0
    metrics_line = f"{stamp},sess_{stamp},{n},{len(plates_seen)},{elapsed:.1f},{fps_actual:.1f}\n"
    metrics_csv = Path(args.logs_dir) / f"jetson_metrics_{stamp}.csv"
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write("timestamp_utc,session_id,total_frames,unique_plates,elapsed_sec,fps_avg\n")
        f.write(metrics_line)
    print(f"Metrics: {metrics_csv}")
    print(f"처리 프레임: {n}, 고유 번호판: {len(plates_seen)}, 소요: {elapsed:.1f}s, FPS: {fps_actual:.1f}")


if __name__ == "__main__":
    main()
