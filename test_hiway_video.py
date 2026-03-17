#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hiway.mp4 실시간 영상 테스트 (headless)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PlateEnginePro로 hiway.mp4 영상을 프레임 단위로 처리하여
번호판 인식이 정상 작동하는지 확인.

측정 항목:
1. 인식된 번호판 목록 + 빈도
2. 프레임당 처리 속도
3. Detection 통계
"""

import os
import sys
import io
import cv2
import time
import re
from pathlib import Path
from collections import Counter

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from plate_engine_pro import PlateEnginePro


def normalize_plate(text: str) -> str:
    """비교용 정규화"""
    if not text:
        return ""
    text = re.sub(r"[\s\-\.\(\)（）]", "", text)
    text = text.replace("(F)", "")
    return text.strip()


def main():
    video_path = ROOT / "movie" / "hiway.mp4"
    if not video_path.is_file():
        print(f"[에러] 영상 없음: {video_path}")
        return 1

    # 설정
    SAMPLE_EVERY = 5       # 5프레임마다 처리 (속도 최적화)
    MAX_PROCESS = 200      # 200프레임 FPS 측정

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 65)
    print("  hiway.mp4 실시간 영상 테스트")
    print("=" * 65)
    print(f"  영상: {video_path.name}")
    print(f"  해상도: {width}x{height} @ {fps:.1f}fps, 총 {total_frames}프레임")
    print(f"  샘플링: {SAMPLE_EVERY}프레임 간격, 최대 {MAX_PROCESS}프레임 처리")

    # 엔진 초기화 (영상 모드: 연속 2프레임 확인으로 1회성 변이 제거)
    os.environ["PLATE_CONSECUTIVE_FRAMES"] = "2"
    print("\n[초기화] PlateEnginePro 로딩... (consecutive=2)")
    t0 = time.time()
    engine = PlateEnginePro()
    print(f"[초기화] 완료 ({time.time() - t0:.1f}초)")

    # 측정 변수
    processed = 0
    frame_idx = 0
    total_detections = 0
    frames_with_plates = 0
    all_plates = Counter()      # {plate_text: 프레임 수}
    plate_confidences = {}      # {plate_text: [conf, ...]}
    frame_times = []

    print(f"\n  처리 시작...")
    print("-" * 65)

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 샘플링
        if frame_idx % SAMPLE_EVERY != 0:
            continue
        processed += 1
        if processed > MAX_PROCESS:
            break

        # 엔진 처리
        t1 = time.perf_counter()
        detections = engine.process_frame(frame)
        dt = time.perf_counter() - t1
        frame_times.append(dt)

        if detections:
            frames_with_plates += 1
            total_detections += len(detections)

            for det in detections:
                plate = normalize_plate(det.get("plate", ""))
                conf = det.get("confidence", 0)
                if plate:
                    all_plates[plate] += 1
                    if plate not in plate_confidences:
                        plate_confidences[plate] = []
                    plate_confidences[plate].append(conf)

        # 진행 표시 (50프레임마다)
        if processed % 50 == 0:
            elapsed = time.time() - t_start
            fps_proc = processed / elapsed if elapsed > 0 else 0
            top_plates = all_plates.most_common(3)
            top_str = ", ".join(f"{p}({c})" for p, c in top_plates)
            print(f"  [{processed:4d}/{MAX_PROCESS}] F:{frame_idx} "
                  f"Det:{total_detections} HasPlate:{frames_with_plates} "
                  f"({fps_proc:.1f} proc/s) Top: {top_str}", flush=True)

    cap.release()
    elapsed_total = time.time() - t_start

    # ── 결과 출력 ──
    print(f"\n{'='*65}")
    print(f"  결과 요약")
    print(f"{'='*65}")
    print(f"  처리 프레임: {processed} / 영상 {frame_idx}프레임")
    print(f"  처리 시간: {elapsed_total:.1f}초 ({processed/elapsed_total:.1f} frames/sec)")
    if frame_times:
        avg_ms = sum(frame_times) / len(frame_times) * 1000
        p50 = sorted(frame_times)[len(frame_times)//2] * 1000
        p95 = sorted(frame_times)[int(len(frame_times)*0.95)] * 1000
        print(f"  프레임 처리: 평균 {avg_ms:.0f}ms, 중앙값 {p50:.0f}ms, P95 {p95:.0f}ms")
    print()

    print(f"  ── 감지 통계 ──")
    print(f"  총 detection: {total_detections}")
    print(f"  번호판 감지 프레임: {frames_with_plates}/{processed} ({frames_with_plates/max(processed,1)*100:.1f}%)")
    print(f"  고유 번호판 수: {len(all_plates)}")
    print()

    print(f"  ── 인식된 번호판 (빈도순) ──")
    for plate, cnt in all_plates.most_common(20):
        confs = plate_confidences.get(plate, [])
        avg_conf = sum(confs)/len(confs) if confs else 0
        print(f"    {plate:20s}  {cnt:3d}회  (평균 conf: {avg_conf:.0%})")

    # ── 판정 ──
    print(f"\n{'='*65}")
    has_plates = len(all_plates) > 0
    detect_rate = frames_with_plates / max(processed, 1)

    if has_plates and detect_rate > 0.05:
        print(f"  판정: PASS — 번호판 인식 정상 작동")
        print(f"    고유 번호판: {len(all_plates)}개")
        print(f"    감지율: {detect_rate:.1%}")
    elif has_plates:
        print(f"  판정: WARN — 감지율 낮음 ({detect_rate:.1%})")
    else:
        print(f"  판정: FAIL — 번호판 인식 없음")

    print(f"{'='*65}")
    return 0 if has_plates else 1


if __name__ == "__main__":
    sys.exit(main())
