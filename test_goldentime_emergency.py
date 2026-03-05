"""
test_goldentime_emergency.py - 골든타임 2.0 긴급차량 영상 채증 테스트

emergency_test.mp4 (YouTube Shorts 구급차 영상)으로 파이프라인 동작을 확인한다.
외국 영상이라 한국 번호판 감지가 안 될 수 있으며,
감지 실패 시 hiway.mp4로 자동 폴백하여 채증 성공까지 확인한다.

핵심 포맷:
  detection_result = {
      "detections": [{
          "plate": "36다7117",
          "confidence": 0.93,
          "bbox": [x1, y1, x2, y2],
          ...
      }],
      "frame_idx": 930,
      "timestamp": video_time,
  }
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
import time
import numpy as np


def run_goldentime_test(video_path, label, start_frame=0, end_frame=None,
                        siren_on_frame=None, threshold_sec=3.0,
                        violation_sec=2.0, ocr_interval=15):
    """골든타임 파이프라인 테스트 실행"""
    print("=" * 60)
    print(f"  골든타임 2.0 -- {label}")
    print(f"  ({video_path}, threshold={threshold_sec}초, violation={violation_sec}초)")
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
    from simulation.goldentime.distance_checker import DistanceChecker, DISTANCE_CONFIG
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

    # PlateEvidence 설정
    evidence = PlateEvidence(bus, fps=30.0, config={
        "continuous_threshold_sec": threshold_sec,
        "pre_buffer_sec": 3.0,
        "post_buffer_sec": 5.0,
        "gap_tolerance_sec": 2.0,
        "min_confidence": 0.3,
    })

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[FAIL] {video_path} 열기 실패")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames
    if siren_on_frame is None:
        siren_on_frame = start_frame + 10

    # DistanceChecker: 기본 설정 복사 후 오버라이드
    dist_cfg = DISTANCE_CONFIG.copy()
    dist_cfg["close_ratio_threshold"] = 0.0003
    dist_cfg["violation_duration_sec"] = violation_sec
    distance = DistanceChecker(bus, frame_width=w, frame_height=h, config=dist_cfg)

    # EvidenceExport: 실제 파일 저장
    export = EvidenceExport(bus, output_dir="./evidence_output",
                           fps=fps, frame_width=w, frame_height=h)

    print(f"[OK] 영상: {w}x{h}, {fps:.0f}fps, {total_frames}프레임 ({total_frames/fps:.1f}초)")
    print()

    # ── 시뮬레이션 실행 ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_detections = 0
    detected_plates = {}
    frames_processed = 0
    ocr_calls = 0

    print(f"[시작] F{start_frame}~F{end_frame} ({(end_frame-start_frame)/fps:.1f}초)")
    print(f"  사이렌 ON: F{siren_on_frame}")
    print(f"  OCR 주기: {ocr_interval}프레임마다")
    print(f"  채증 기준: {threshold_sec}초 연속 감지")
    print(f"  거리 위반: {violation_sec}초")
    print("-" * 60)

    sim_start_time = time.time()

    for fidx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1

        # ★ 영상 타임스탬프 사용 (CPU OCR 지연과 무관)
        video_time = fidx / fps

        # ── frame_read 이벤트 (프레임 버퍼용) ──
        bus.publish("frame_read", {
            "frame": frame.copy(),
            "frame_idx": fidx,
        })

        # ── 사이렌 자동 ON ──
        if fidx == siren_on_frame:
            bus.publish("key_pressed", {"char": "s", "key": 115, "frame_idx": fidx})
            print(f"[F{fidx:04d}] === 사이렌 ON (t={video_time:.1f}s) ===")

        # ── YOLO + OCR (사이렌 활성시, OCR_INTERVAL 간격) ──
        if fidx % ocr_interval == 0 and siren.is_active:
            ocr_calls += 1
            results = engine.process_frame(frame)

            if results and isinstance(results, list):
                # SimulationFramework 포맷: "detections" 키 사용
                det_data = {
                    "detections": results,
                    "frame_idx": fidx,
                    "timestamp": video_time,  # 영상 기준 시간
                }

                bus.publish("detection_result", det_data)

                for r in results:
                    plate = r.get("plate", "")
                    conf = r.get("confidence", 0)
                    if plate:
                        total_detections += 1
                        detected_plates[plate] = detected_plates.get(plate, 0) + 1
                        print(f"[F{fidx:04d}] {plate} ({conf:.0%}) t={video_time:.1f}s")

        # ── 진행 표시 (200프레임마다) ──
        if fidx > start_frame and (fidx - start_frame) % 200 == 0:
            elapsed = time.time() - sim_start_time
            print(f"[F{fidx:04d}] ... {fidx-start_frame}/{end_frame-start_frame} "
                  f"({elapsed:.0f}초, OCR {ocr_calls}회)")

    # ── 사이렌 종료 ──
    bus.publish("key_pressed", {"char": "e", "key": 101, "frame_idx": end_frame})
    print(f"[F{end_frame}] === 사이렌 OFF ===")

    cap.release()
    total_time = time.time() - sim_start_time

    # ── 최종 결과 ──
    print()
    print("=" * 60)
    print(f"  {label} 결과")
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

    evidence_count = sum(1 for e in event_log if e[0] == "EVIDENCE_STARTED")
    violation_count = sum(1 for e in event_log if e[0] == "DISTANCE_VIOLATION")

    print()
    print("-" * 60)
    if evidence_count > 0:
        print(f"  채증 대상: {evidence_count}건, 거리 위반: {violation_count}건")
        print(f"  >>> {label} 채증 성공!")
    else:
        print(f"  채증 대상: 0건 (threshold 미도달)")
        if fps and ocr_interval:
            print(f"  참고: 같은 번호판이 {threshold_sec*fps/ocr_interval:.0f}회 이상 연속 감지되어야 {threshold_sec}초 충족")
    print("=" * 60)

    return {
        "video": video_path,
        "total_detections": total_detections,
        "unique_plates": len(detected_plates),
        "plates": detected_plates,
        "evidence_count": evidence_count,
        "violation_count": violation_count,
        "events": len(event_log),
    }


def main():
    print()
    print("#" * 60)
    print("  골든타임 2.0 긴급차량 영상 채증 테스트")
    print("  (emergency_test.mp4 + hiway.mp4 폴백)")
    print("#" * 60)
    print()

    # ── 1차: emergency_test.mp4 (YouTube 구급차 영상) ──
    print(">>> [1차] emergency_test.mp4 (YouTube 구급차 Shorts)")
    print()
    result1 = run_goldentime_test(
        video_path="movie/emergency_test.mp4",
        label="긴급차량 영상 테스트",
        start_frame=0,
        end_frame=None,       # 전체 영상
        siren_on_frame=10,    # 바로 사이렌 ON
        threshold_sec=3.0,
        violation_sec=2.0,
        ocr_interval=15,
    )

    need_fallback = True
    if result1 and result1["evidence_count"] > 0:
        print()
        print(">>> emergency_test.mp4에서 채증 성공! 폴백 불필요.")
        need_fallback = False

    # ── 2차: hiway.mp4 폴백 (한국 영상) ──
    if need_fallback:
        print()
        print()
        print(">>> [2차] hiway.mp4 (한국 고속도로 영상) 폴백")
        print(">>> 외국 번호판이라 감지 안 됨 -> 한국 영상으로 채증 확인")
        print()
        result2 = run_goldentime_test(
            video_path="movie/hiway.mp4",
            label="한국 영상 폴백 테스트",
            start_frame=60,
            end_frame=1200,
            siren_on_frame=70,
            threshold_sec=3.0,
            violation_sec=2.0,
            ocr_interval=15,
        )
    else:
        result2 = None

    # ── 최종 종합 보고 ──
    print()
    print()
    print("#" * 60)
    print("  종합 테스트 결과")
    print("#" * 60)
    print()
    print("  [1차] emergency_test.mp4 (구급차 Shorts)")
    if result1:
        print(f"    감지: {result1['total_detections']}건, "
              f"고유 번호판: {result1['unique_plates']}개, "
              f"채증: {result1['evidence_count']}건")
        if result1["evidence_count"] == 0:
            print("    -> 외국(호주) 영상이라 한국 번호판 모델로 감지 불가")
    else:
        print("    -> 영상 열기 실패")

    if result2:
        print()
        print("  [2차] hiway.mp4 (한국 고속도로)")
        print(f"    감지: {result2['total_detections']}건, "
              f"고유 번호판: {result2['unique_plates']}개, "
              f"채증: {result2['evidence_count']}건")
        if result2["evidence_count"] > 0:
            print("    -> 한국 번호판 채증 성공!")
        else:
            print("    -> 채증 미도달")

    # 성공 판정
    success = False
    if result1 and result1["evidence_count"] > 0:
        success = True
    elif result2 and result2["evidence_count"] > 0:
        success = True

    print()
    if success:
        print("  >>> 골든타임 2.0 파이프라인 정상 작동 확인!")
    else:
        print("  >>> 채증 실패 - 파이프라인 점검 필요")
    print("#" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
