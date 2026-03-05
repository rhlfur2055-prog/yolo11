"""
test_goldentime_realrun.py - 골든타임 2.0 실전 YOLO 연동 채증 테스트

hiway.mp4 번호판 밀집 구간에서 자동 사이렌 + 채증 시나리오를 검증한다.
SimulationFramework의 이벤트 포맷에 맞춰 detection_result를 발행.

핵심 포맷:
  detection_result = {
      "detections": [{
          "plate": "36다7117",
          "confidence": 0.93,
          "bbox": [x1, y1, x2, y2],
          ...
      }],
      "frame_idx": 930,
      "timestamp": time.time(),
  }
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
import time
import numpy as np


def main():
    print("=" * 60)
    print("  골든타임 2.0 -- 실전 YOLO 채증 시나리오")
    print("  (hiway.mp4 F60~F1200, threshold=3초)")
    print("=" * 60)
    print()

    # ── 엔진 로딩 ──
    from plate_engine_pro import PlateEnginePro
    engine = PlateEnginePro()
    print("[OK] PlateEnginePro 로드 완료")

    # ── EventBus + 모듈 초기화 ──
    from simulation.simulation_framework import EventBus
    from simulation.goldentime.siren_trigger import SirenTrigger
    from simulation.goldentime.plate_evidence import PlateEvidence
    from simulation.goldentime.distance_checker import DistanceChecker
    from simulation.goldentime.evidence_export import EvidenceExport

    bus = EventBus()
    event_log = []

    def log_event(name):
        def handler(data):
            event_log.append((name, time.time(), data))
            plate = ""
            if isinstance(data, dict):
                plate = data.get("plate", "")
            extra = f" ({plate})" if plate else ""
            print(f"  >>> [{name}]{extra}")
        return handler

    for evt in ["SIREN_DETECTED", "SIREN_ENDED", "EVIDENCE_STARTED",
                "EVIDENCE_COMPLETE", "DISTANCE_VIOLATION"]:
        bus.subscribe(evt, log_event(evt))

    siren = SirenTrigger(bus)

    # PlateEvidence: threshold=3초, 빠른 테스트
    evidence = PlateEvidence(bus, fps=30.0, config={
        "continuous_threshold_sec": 3.0,
        "pre_buffer_sec": 3.0,
        "post_buffer_sec": 5.0,
        "gap_tolerance_sec": 2.0,
        "min_confidence": 0.3,
    })

    cap = cv2.VideoCapture("movie/hiway.mp4")
    if not cap.isOpened():
        print("[FAIL] hiway.mp4 열기 실패")
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # DistanceChecker: 기본 DISTANCE_CONFIG에서 일부만 오버라이드
    from simulation.goldentime.distance_checker import DISTANCE_CONFIG
    dist_cfg = DISTANCE_CONFIG.copy()
    dist_cfg["close_ratio_threshold"] = 0.0003
    dist_cfg["violation_duration_sec"] = 3.0
    distance = DistanceChecker(bus, frame_width=w, frame_height=h, config=dist_cfg)

    # EvidenceExport는 테스트에서 실제 파일 저장은 안 함 (더미)
    export = EvidenceExport(bus, output_dir="./evidence_output",
                           fps=fps, frame_width=w, frame_height=h)

    print(f"[OK] 영상: {w}x{h}, {fps:.0f}fps")
    print()

    # ── 시뮬레이션 파라미터 ──
    START_FRAME = 60
    END_FRAME = 1200     # 40초 구간
    SIREN_ON_FRAME = 70
    OCR_INTERVAL = 15    # 15프레임마다 OCR (CPU 부하 고려)

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    total_detections = 0
    detected_plates = {}
    frames_processed = 0
    ocr_calls = 0

    print(f"[시작] F{START_FRAME}~F{END_FRAME} ({(END_FRAME-START_FRAME)/fps:.1f}초)")
    print(f"  OCR 주기: {OCR_INTERVAL}프레임마다")
    print(f"  채증 기준: 3초 연속 감지")
    print("-" * 60)

    sim_start_time = time.time()

    for fidx in range(START_FRAME, END_FRAME):
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1

        # ★ 핵심: 영상 타임스탬프 사용 (CPU OCR 지연과 무관하게 정확한 시간)
        video_time = fidx / fps

        # ── frame_read 이벤트 (프레임 버퍼용) ──
        bus.publish("frame_read", {
            "frame": frame.copy(),
            "frame_idx": fidx,
        })

        # ── 사이렌 자동 ON ──
        if fidx == SIREN_ON_FRAME:
            bus.publish("key_pressed", {"char": "s", "key": 115, "frame_idx": fidx})
            print(f"[F{fidx:04d}] === 사이렌 ON (video_time={video_time:.1f}s) ===")

        # ── YOLO + OCR (사이렌 활성시, OCR_INTERVAL 간격) ──
        if fidx % OCR_INTERVAL == 0 and siren.is_active:
            ocr_calls += 1
            results = engine.process_frame(frame)

            if results and isinstance(results, list):
                # SimulationFramework 포맷으로 변환
                # ★ timestamp를 video_time으로 설정 (time.time() 대신)
                det_data = {
                    "detections": results,
                    "frame_idx": fidx,
                    "timestamp": video_time,  # 영상 기준 시간
                }

                bus.publish("detection_result", det_data)

                for r in results:
                    plate = r.get("plate", "")
                    conf = r.get("confidence", 0)
                    bbox = r.get("bbox", [])
                    if plate:
                        total_detections += 1
                        detected_plates[plate] = detected_plates.get(plate, 0) + 1
                        print(f"[F{fidx:04d}] {plate} ({conf:.0%}) t={video_time:.1f}s")

        # ── 진행 표시 ──
        if fidx > START_FRAME and (fidx - START_FRAME) % 200 == 0:
            elapsed = time.time() - sim_start_time
            print(f"[F{fidx:04d}] ... {fidx-START_FRAME}/{END_FRAME-START_FRAME} "
                  f"({elapsed:.0f}초, OCR {ocr_calls}회)")

    # ── 사이렌 종료 ──
    bus.publish("key_pressed", {"char": "e", "key": 101, "frame_idx": END_FRAME})
    print(f"[F{END_FRAME}] === 사이렌 OFF ===")

    cap.release()
    total_time = time.time() - sim_start_time

    # ── 최종 결과 ──
    print()
    print("=" * 60)
    print("  실전 채증 시나리오 결과")
    print("=" * 60)
    print(f"  처리 프레임: {frames_processed}")
    print(f"  OCR 호출: {ocr_calls}회")
    print(f"  총 소요: {total_time:.1f}초")
    print(f"  총 감지: {total_detections}건")
    print(f"  고유 번호판: {len(detected_plates)}개")

    if detected_plates:
        print()
        print("  [감지된 번호판]")
        for plate, cnt in sorted(detected_plates.items(), key=lambda x: -x[1]):
            print(f"    {plate}: {cnt}회")

    print()
    print("  [이벤트 로그]")
    for name, ts, data in event_log:
        plate = data.get("plate", "") if isinstance(data, dict) else ""
        dur = data.get("continuous_duration", "") if isinstance(data, dict) else ""
        extra = ""
        if plate:
            extra += f" plate={plate}"
        if dur:
            extra += f" dur={dur:.1f}s"
        print(f"    {name}{extra}")
    print(f"  총 이벤트: {len(event_log)}건")

    # PlateEvidence 추적 상태
    if hasattr(evidence, "tracked_plates") and evidence.tracked_plates:
        print()
        print("  [PlateEvidence 추적 기록]")
        for plate, record in evidence.tracked_plates.items():
            dur = record.continuous_duration
            cnt = record.detection_count
            is_ev = record.is_evidence_target
            status = "[EVIDENCE]" if is_ev else f"[{dur:.1f}s]"
            print(f"    {plate}: {status} x{cnt}회 감지")

    # 채증 결과
    evidence_count = sum(1 for e in event_log if e[0] == "EVIDENCE_STARTED")
    violation_count = sum(1 for e in event_log if e[0] == "DISTANCE_VIOLATION")

    print()
    print("-" * 60)
    if evidence_count > 0:
        print(f"  채증 대상: {evidence_count}건, 거리 위반: {violation_count}건")
        print("  >>> 골든타임 채증 시나리오 성공!")
    else:
        print(f"  채증 대상: 0건 (threshold 미도달)")
        print(f"  참고: 같은 번호판이 {3*fps/OCR_INTERVAL:.0f}회 이상 연속 감지되어야 3초 충족")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
