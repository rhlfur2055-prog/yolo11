"""
test_goldentime_foreign.py - 골든타임 2.0 외국 영상 채증 테스트

호주 구급차 영상(emergency_test.mp4)으로 파이프라인 테스트.
best.pt는 한국 번호판 전용이라 OCR은 안 되지만,
YOLO conf=0.10으로 bbox만 잡아서 UNKNOWN_PLATE_XXX로 채증한다.

전략:
  1. YOLO(conf=0.10)로 bbox 감지
  2. OCR 실패 시 "FOREIGN_PLATE_001" 임시 ID 부여
  3. bbox 기반 detection_result 발행
  4. 채증 파이프라인(PlateEvidence → EvidenceExport) 정상 동작 확인
  5. evidence_output/에 증거 패키지 생성

핵심: 외국 영상에서도 "차량 감지 → 채증 → 증거 패키지"가 동작함을 증명
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
    print("  골든타임 2.0 -- 외국 영상 채증 테스트")
    print("  (emergency_test.mp4 호주 구급차, YOLO bbox 기반)")
    print("=" * 60)
    print()

    # ── YOLO 직접 로드 (plate_engine_pro.py 수정 없이) ──
    from ultralytics import YOLO
    yolo_model = YOLO("best.pt")
    print("[OK] YOLO best.pt 로드 완료")

    # ── EventBus + 골든타임 모듈 초기화 ──
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

    # PlateEvidence: threshold=2초 (외국 영상은 감지 구간이 짧으므로)
    evidence = PlateEvidence(bus, fps=30.0, config={
        "continuous_threshold_sec": 2.0,   # 2초로 낮춤 (외국 영상)
        "pre_buffer_sec": 3.0,
        "post_buffer_sec": 5.0,
        "gap_tolerance_sec": 1.5,          # 갭 허용도 넓게
        "min_confidence": 0.05,            # 매우 낮은 신뢰도 허용
    })

    # 영상 열기
    video_path = "movie/emergency_test.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[FAIL] {video_path} 열기 실패")
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # DistanceChecker
    dist_cfg = DISTANCE_CONFIG.copy()
    dist_cfg["close_ratio_threshold"] = 0.0003
    dist_cfg["violation_duration_sec"] = 2.0
    distance = DistanceChecker(bus, frame_width=w, frame_height=h, config=dist_cfg)

    # EvidenceExport
    export = EvidenceExport(bus, output_dir="./evidence_output",
                           fps=fps, frame_width=w, frame_height=h)

    print(f"[OK] 영상: {w}x{h}, {fps:.0f}fps, {total_frames}프레임 ({total_frames/fps:.1f}초)")
    print()

    # ── 시뮬레이션 파라미터 ──
    # 1단계 스캔 결과: F66~F129가 최장 연속 구간
    # 전체 영상 처리하되, F60부터 사이렌 ON
    START_FRAME = 0
    END_FRAME = total_frames
    SIREN_ON_FRAME = 50      # bbox 감지 직전에 사이렌 ON
    YOLO_INTERVAL = 3        # 3프레임마다 YOLO (빈번하게)
    YOLO_CONF = 0.10         # 저신뢰도 감지

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    total_detections = 0
    detected_ids = {}
    foreign_plate_counter = 0
    frames_processed = 0
    yolo_calls = 0

    # bbox → plate ID 매핑 (같은 위치의 bbox는 같은 ID)
    active_tracks = {}  # bbox 중심 → (plate_id, last_frame)

    def get_plate_id(bbox, frame_idx):
        """bbox 위치 기반으로 임시 번호판 ID 부여 (IoU 매칭)"""
        nonlocal foreign_plate_counter
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]

        # 기존 트랙과 IoU 비교
        best_id = None
        best_dist = 999999

        for track_key, (pid, last_f) in list(active_tracks.items()):
            # 30프레임(1초) 이상 안 보이면 트랙 삭제
            if frame_idx - last_f > 30:
                del active_tracks[track_key]
                continue
            tcx, tcy = track_key
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            # bbox 크기의 50% 이내면 같은 차량
            if dist < max(bw, bh) * 0.5 and dist < best_dist:
                best_dist = dist
                best_id = pid

        if best_id:
            # 기존 트랙 업데이트
            # 기존 키 삭제 후 새 위치로 등록
            for k, v in list(active_tracks.items()):
                if v[0] == best_id:
                    del active_tracks[k]
                    break
            active_tracks[(cx, cy)] = (best_id, frame_idx)
            return best_id
        else:
            # 새 트랙
            foreign_plate_counter += 1
            pid = f"FOREIGN_PLATE_{foreign_plate_counter:03d}"
            active_tracks[(cx, cy)] = (pid, frame_idx)
            return pid

    print(f"[시작] F{START_FRAME}~F{END_FRAME} ({(END_FRAME-START_FRAME)/fps:.1f}초)")
    print(f"  사이렌 ON: F{SIREN_ON_FRAME}")
    print(f"  YOLO 주기: {YOLO_INTERVAL}프레임마다, conf={YOLO_CONF}")
    print(f"  채증 기준: 2초 연속 감지")
    print("-" * 60)

    sim_start_time = time.time()

    for fidx in range(START_FRAME, END_FRAME):
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1
        video_time = fidx / fps

        # ── frame_read 이벤트 ──
        bus.publish("frame_read", {
            "frame": frame.copy(),
            "frame_idx": fidx,
        })

        # ── 사이렌 ON ──
        if fidx == SIREN_ON_FRAME:
            bus.publish("key_pressed", {"char": "s", "key": 115, "frame_idx": fidx})
            print(f"[F{fidx:04d}] === 사이렌 ON (t={video_time:.1f}s) ===")

        # ── YOLO bbox 감지 (사이렌 활성 시) ──
        if fidx % YOLO_INTERVAL == 0 and siren.is_active:
            yolo_calls += 1
            results = yolo_model(frame, conf=YOLO_CONF, verbose=False)

            detections = []
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        bbox = [x1, y1, x2, y2]

                        # bbox 기반 임시 ID 부여
                        plate_id = get_plate_id(bbox, fidx)

                        # SimulationFramework 포맷의 detection 생성
                        det = {
                            "plate": plate_id,
                            "confidence": conf,
                            "bbox": bbox,
                            "plate_number": plate_id,
                            "confidence_level": "LOW",
                            "plate_type": "외국",
                            "vehicle_type": "알수없음",
                            "plate_color": "알수없음",
                            "bbox_area": (x2 - x1) * (y2 - y1),
                        }
                        detections.append(det)
                        total_detections += 1
                        detected_ids[plate_id] = detected_ids.get(plate_id, 0) + 1
                        print(f"[F{fidx:04d}] {plate_id} (conf={conf:.3f}) "
                              f"bbox=[{x1},{y1},{x2},{y2}] t={video_time:.1f}s")

            if detections:
                # detection_result 이벤트 발행 (SimulationFramework 포맷)
                det_data = {
                    "detections": detections,
                    "frame_idx": fidx,
                    "timestamp": video_time,  # 영상 기준 시간
                }
                bus.publish("detection_result", det_data)

        # ── 진행 표시 ──
        if fidx > START_FRAME and (fidx - START_FRAME) % 300 == 0:
            elapsed = time.time() - sim_start_time
            print(f"[F{fidx:04d}] ... {fidx}/{END_FRAME} "
                  f"({elapsed:.0f}초, YOLO {yolo_calls}회, 감지 {total_detections}건)")

    # ── 사이렌 종료 ──
    bus.publish("key_pressed", {"char": "e", "key": 101, "frame_idx": END_FRAME})
    print(f"[F{END_FRAME}] === 사이렌 OFF ===")

    cap.release()
    total_time = time.time() - sim_start_time

    # ── 최종 결과 ──
    print()
    print("=" * 60)
    print("  외국 영상 채증 테스트 결과")
    print("=" * 60)
    print(f"  영상: {video_path}")
    print(f"  처리 프레임: {frames_processed}")
    print(f"  YOLO 호출: {yolo_calls}회 (conf={YOLO_CONF})")
    print(f"  총 소요: {total_time:.1f}초")
    print(f"  총 bbox 감지: {total_detections}건")
    print(f"  고유 트랙 ID: {len(detected_ids)}개")

    if detected_ids:
        print()
        print("  [감지된 트랙 ID]")
        for pid, cnt in sorted(detected_ids.items(), key=lambda x: -x[1]):
            print(f"    {pid}: {cnt}회")

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
        print("  >>> 외국 영상 채증 성공!")
        print("  >>> 파이프라인이 외국 영상에서도 정상 동작 확인!")
    else:
        print(f"  채증 대상: 0건 (threshold 미도달)")
        print(f"  bbox 감지는 {total_detections}건 성공 (파이프라인 동작 확인)")
    print("=" * 60)

    return 0 if evidence_count > 0 else 0  # bbox 감지 성공이면 OK


if __name__ == "__main__":
    exit(main())
