"""
test_safeplate_australia.py — SafePlate 4K 호주 영상 headless 자동 테스트

시나리오:
    1. yolo11n.pt (일반 YOLO) conf=0.1 로 emergency_test.mp4 차량 감지
    2. 첫 bbox 감지 시 자동 SHOCK_DETECTED 발행
    3. bbox가 화면에서 사라지면 DEPARTURE_DETECTED (threshold 2초)
    4. 외국 번호판은 FOREIGN_PLATE_XXX 형식 ID 부여 (DepartureDetector 내장)
    5. 증거 패키지 evidence_output/safeplate_*/ 에 저장

참고: best.pt 는 한국 번호판 전용 → 호주 번호판 감지 불가
      yolo11n.pt 일반 모델로 차량(class=2,5,7) bbox 감지 후
      SafePlate 모듈(ShockSimulator, DepartureDetector, EvidencePackage)에 전달

      dashcam 영상 특성상 차량 bbox가 프레임 경계에 걸리는 경우가 많으므로
      bbox가 완전히 프레임 내부에 있는 것만 추적 대상으로 전달
      (경계에 이미 걸려있으면 즉시 이탈 판정 → 노이즈)

기존 코드 수정: 없음
Mock 데이터: 없음
"""

import sys
import os
import time

import cv2

# 프로젝트 루트를 path에 추가
_project_root = os.path.abspath(os.path.dirname(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ultralytics import YOLO
from simulation.simulation_framework import EventBus
from simulation.safeplate.shock_simulator import ShockSimulator
from simulation.safeplate.departure_detector import DepartureDetector
from simulation.safeplate.evidence_package import EvidencePackage

# YOLO conf threshold (탐지용)
YOLO_CONF = 0.1
# 추적 파이프라인 전달 최소 conf (노이즈 필터)
TRACKING_MIN_CONF = 0.30
# 이탈 판정 타임아웃 (초)
VANISH_TIMEOUT = 2.0
# bbox 경계 마진 비율 — 이 비율 이내에 bbox 꼭짓점이 있으면 무시 (이미 경계에 있는 bbox)
EDGE_MARGIN_RATIO = 0.08


def _is_inside_frame(bbox, fw, fh, margin_ratio=EDGE_MARGIN_RATIO):
    """bbox가 프레임 내부에 완전히 있는지 확인 (경계 터치 제외)"""
    x1, y1, x2, y2 = bbox
    mx = fw * margin_ratio
    my = fh * margin_ratio
    return x1 > mx and y1 > my and x2 < (fw - mx) and y2 < (fh - my)


def main():
    video_path = os.path.join(_project_root, "movie", "emergency_test.mp4")
    output_dir = os.path.join(_project_root, "evidence_output")
    yolo_model_path = os.path.join(_project_root, "yolo11n.pt")

    if not os.path.exists(video_path):
        print(f"[ERROR] 영상 파일 없음: {video_path}")
        sys.exit(1)
    if not os.path.exists(yolo_model_path):
        print(f"[ERROR] YOLO 모델 없음: {yolo_model_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  SafePlate 4K — 호주 영상 Headless 자동 테스트")
    print("=" * 60)

    # ── 1. 영상 열기 ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  영상: {video_path}")
    print(f"  해상도: {frame_width}x{frame_height} @ {fps:.1f}fps")
    print(f"  총 프레임: {total_frames}")
    print(f"  YOLO 모델: yolo11n.pt (일반 객체 탐지)")
    print(f"  YOLO conf: {YOLO_CONF} (추적 필터: >={TRACKING_MIN_CONF})")
    print(f"  이탈 threshold: {VANISH_TIMEOUT}초")
    print(f"  경계 마진: {EDGE_MARGIN_RATIO:.0%} (프레임 내부 bbox만 추적)")
    print(f"  대상 클래스: car/bus/truck (2,5,7)")

    # ── 2. YOLO 일반 모델 로드 ──
    yolo_model = YOLO(yolo_model_path)
    vehicle_classes = {2, 5, 7}  # car, bus, truck
    print(f"  YOLO 모델 로드 완료")

    # ── 3. EventBus + SafePlate 모듈 ──
    event_bus = EventBus()

    shock = ShockSimulator(
        event_bus=event_bus,
        fps=fps,
        pre_buffer_sec=5.0,
        post_buffer_sec=10.0,
    )

    departure = DepartureDetector(
        event_bus=event_bus,
        frame_width=frame_width,
        frame_height=frame_height,
        vanish_timeout_sec=VANISH_TIMEOUT,
    )

    evidence = EvidencePackage(
        event_bus=event_bus,
        shock_simulator=shock,
        output_dir=output_dir,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    # ── 4. 이벤트 로거 ──
    export_results = []

    def _on_export(data):
        export_results.append(data)
        print(f"\n  ★ 증거 패키지 저장: {data['plate']} → {data['folder']}")

    event_bus.subscribe("EVIDENCE_EXPORTED", _on_export)

    departure_events = []

    def _on_departure(data):
        departure_events.append(data)

    event_bus.subscribe("DEPARTURE_DETECTED", _on_departure)

    print("\n" + "-" * 60)
    print("  자동 시나리오: 첫 bbox → SHOCK / bbox 소멸 → DEPARTURE")
    print("-" * 60)

    # ── 5. 메인 프레임 루프 ──
    start_time = time.time()
    frame_idx = 0
    shock_triggered = False
    first_bbox_frame = None
    total_vehicle_detections = 0
    total_raw_detections = 0

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # 프레임 이벤트 발행 (ShockSimulator 버퍼링용)
        event_bus.publish("frame_read", {
            "frame": frame,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
        })

        # ── YOLO 감지 (일반 모델) ──
        results = yolo_model(frame, conf=YOLO_CONF, verbose=False, max_det=20)
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in vehicle_classes:
                continue

            conf_val = float(box.conf[0])
            if conf_val < TRACKING_MIN_CONF:
                continue

            bbox = [int(v) for v in box.xyxy[0].tolist()]
            total_raw_detections += 1

            # 경계에 이미 걸린 bbox 제외 (dashcam 노이즈 방지)
            if not _is_inside_frame(bbox, frame_width, frame_height):
                continue

            cls_name = yolo_model.names[cls_id]

            # plate 비워서 DepartureDetector가 FOREIGN_PLATE_XXX 자동 부여
            detections.append({
                "plate": "",
                "confidence": conf_val,
                "bbox": bbox,
                "vehicle_class": cls_name,
            })

        total_vehicle_detections += len(detections)

        # 감지 결과 이벤트 발행 → DepartureDetector가 추적
        event_bus.publish("detection_result", {
            "detections": detections,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
        })

        # ── 자동 SHOCK: 첫 차량 bbox 감지 시 ──
        if not shock_triggered and len(detections) > 0:
            first_bbox_frame = frame_idx
            print(f"\n  >>> 첫 차량 bbox 감지! 자동 SHOCK_DETECTED (프레임 #{frame_idx}, {timestamp:.1f}s) <<<")
            for d in detections:
                print(f"      {d['vehicle_class']} conf={d['confidence']:.2f} bbox={d['bbox']}")
            shock.trigger_shock(frame_idx=frame_idx, timestamp=timestamp)
            shock_triggered = True

        # 진행률 (300프레임마다)
        if frame_idx > 0 and frame_idx % 300 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            pct = frame_idx / total_frames * 100
            dep_cnt = len(departure_events)
            trk_cnt = departure.get_tracking_count()
            print(f"  ... 프레임 {frame_idx}/{total_frames} ({pct:.0f}%) — {fps_actual:.1f} fps | 추적: {trk_cnt} | 이탈: {dep_cnt}")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - start_time

    # ── 충격 종료 (영상 끝) ──
    if shock_triggered and shock.is_active:
        print(f"\n  >>> 영상 종료 — 자동 SHOCK 종료 <<<")
        shock.trigger_shock_end(frame_idx=frame_idx - 1, timestamp=(frame_idx - 1) / fps)

    print(f"\n  처리 완료: {frame_idx} 프레임 / {elapsed:.1f}초 ({frame_idx / elapsed:.1f} fps)")

    # ── 6. 결과 요약 ──
    print("\n" + "=" * 60)
    print("  SafePlate 4K — 호주 영상 테스트 결과")
    print("=" * 60)

    if first_bbox_frame is not None:
        print(f"\n  첫 bbox 프레임: #{first_bbox_frame} ({first_bbox_frame / fps:.1f}s)")
    else:
        print(f"\n  bbox 미감지")
    print(f"  총 YOLO 감지: {total_raw_detections}건 (conf>={TRACKING_MIN_CONF})")
    print(f"  프레임 내부 bbox: {total_vehicle_detections}건 (경계 제외)")

    # 추적/이탈 결과
    tracking_count = departure.get_tracking_count()
    departed_count = departure.get_departed_count()
    print(f"\n  차량 추적: {tracking_count}대 진행 중")
    print(f"  이탈 감지: {departed_count}대")

    if departed_count > 0:
        dir_map = {"left": "좌측", "right": "우측", "top": "상방", "bottom": "하방", "vanished": "소멸"}
        for i, dep in enumerate(departure.get_departed_vehicles(), 1):
            plate = dep.get("plate", "unknown")
            direction = dir_map.get(dep.get("departure_direction", ""), dep.get("departure_direction", ""))
            conf = dep.get("confidence", 0.0)
            det_cnt = dep.get("detection_count", 0)
            print(f"    [{i}] {plate} — {direction} 이탈 (신뢰도: {conf:.0%}, 감지: {det_cnt}회)")

    # 증거 패키지
    print(f"\n  증거 패키지: {len(export_results)}건")
    if export_results:
        for i, exp in enumerate(export_results, 1):
            print(f"    [{i}] {exp['plate']} → {exp['folder_name']}")
        print(f"\n  [PASS] 증거 패키지 {len(export_results)}건 생성 완료")
    else:
        print(f"  [INFO] 증거 패키지 미생성")
        if departed_count > 0:
            print(f"  [WARN] 이탈 차량 {departed_count}대 감지되었으나 증거 미생성")

    # evidence_output/safeplate_* 폴더 확인
    if os.path.exists(output_dir):
        safeplate_dirs = [d for d in os.listdir(output_dir) if d.startswith("safeplate_")]
        if safeplate_dirs:
            print(f"\n  evidence_output/ 내 safeplate 폴더 (최근 5건):")
            for d in sorted(safeplate_dirs)[-5:]:
                full = os.path.join(output_dir, d)
                if os.path.isdir(full):
                    files = os.listdir(full)
                    print(f"    {d}/ ({len(files)} items)")
                    for f in sorted(files):
                        sub = os.path.join(full, f)
                        if os.path.isdir(sub):
                            sub_files = os.listdir(sub)
                            print(f"      {f}/ ({len(sub_files)} files)")
                        else:
                            size = os.path.getsize(sub)
                            print(f"      {f} ({size:,} bytes)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
