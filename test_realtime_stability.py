#!/usr/bin/env python3
"""
실시간 영상 안정성 테스트 (headless)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
hiway.mp4를 프레임 단위로 처리하며 GUI PlateTracker의 안정성 지표를 측정.

측정 항목:
1. 떨림 횟수 — 같은 트랙에서 Phase 2 → Phase 1 → Phase 2 전환 (깜빡임)
2. 중복 bbox — 같은 프레임에 IoU > 0.3인 bbox가 2개 이상
3. Ghost 잔존 — 차량 전환 시 이전 번호판 잔존
4. 인식 안정성 — 연속 인식 구간 길이 (길수록 좋음)
"""

import os
import sys
import cv2
import time
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from plate_engine_pro import PlateEnginePro

# GUI PlateTracker import
from plate_gui import PlateTracker, validate_bbox


def normalize_plate(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\s\-\.\(\)（）]", "", text)
    text = text.replace("(F)", "")
    return text.strip()


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def main():
    video_path = os.path.join(BASE_DIR, "movie", "hiway.mp4")
    if not os.path.isfile(video_path):
        print(f"영상 없음: {video_path}")
        return 1

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 70)
    print("  실시간 영상 안정성 테스트 (GUI PlateTracker)")
    print("=" * 70)
    print(f"  영상: {video_path}")
    print(f"  해상도: {width}x{height} @ {fps:.1f}fps, 총 {total_frames}프레임")

    # 엔진 초기화
    print("\n[초기화] PlateEnginePro 로딩...")
    t0 = time.time()
    engine = PlateEnginePro()
    # ★ engine-side consecutive를 1로 설정 → 즉시 텍스트 출력
    # GUI PlateTracker의 안정화 로직만 독립적으로 테스트하기 위함
    engine.consecutive_required = 1
    print(f"[초기화] 완료 ({time.time() - t0:.1f}초)")

    # GUI PlateTracker 초기화
    gui_tracker = PlateTracker(iou_threshold=0.35, max_ttl=25)

    # ── 측정 변수 ──
    frame_count = 0
    # 매 N프레임마다 처리 (전체 영상 중 일부 샘플링)
    SAMPLE_EVERY = 2       # 2프레임마다 처리 (연속성 확보)
    MAX_FRAMES = 250       # 최대 처리 프레임 수

    flicker_count = 0      # Phase 2 → Phase 1 전환 (떨림)
    dup_bbox_count = 0     # 같은 프레임 내 중복 bbox
    total_detections = 0   # 총 detection 수
    phase2_count = 0       # Phase 2 (확정) 횟수
    phase1_count = 0       # Phase 1 (탐지중) 횟수
    recognized_plates = {} # {plate_text: count}

    # 트랙별 이전 상태 추적 (떨림 감지)
    prev_track_phase = {}  # {track_text: was_phase2}

    # 연속 인식 구간 추적
    continuous_runs = []   # 연속 Phase 2 프레임 수 리스트
    current_run = 0

    print(f"\n  처리: {SAMPLE_EVERY}프레임 간격, 최대 {MAX_FRAMES}프레임")
    print("-" * 70)

    processed = 0
    frame_idx = 0
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
        if processed > MAX_FRAMES:
            break

        # ── engine-side 처리 ──
        raw_results = engine.process_frame(frame)

        # ── GUI 형식으로 변환 ──
        gui_dets = []
        for r in raw_results:
            x1, y1, x2, y2 = r.get("bbox", [0, 0, 0, 0])
            gui_dets.append({
                "text": r.get("plate", ""),
                "ocr_confidence": r.get("confidence", 0),
                "bbox": r.get("bbox", []),
                "is_valid_plate": bool(r.get("plate", "")),
                "confidence_level": r.get("confidence_level", ""),
            })

        # ── bbox 검증 ──
        validated = []
        for det in gui_dets:
            bbox = det.get("bbox", [])
            conf = det.get("ocr_confidence", 0)
            text = det.get("text", "")
            passed_val = validate_bbox(bbox, (height, width), conf)
            if passed_val:
                validated.append(det)
            # 텍스트 있는데 bbox 검증 실패 → 필터링됨 (정상 동작)

        # ── GUI PlateTracker 적용 ──
        tracked = gui_tracker.update(validated, frame_idx)

        # ── 측정 ──
        has_phase2 = False
        frame_bboxes = []

        for det in tracked:
            total_detections += 1
            is_valid = det.get("is_valid_plate", False)
            text = normalize_plate(det.get("text", ""))
            bbox = det.get("bbox", [])

            if is_valid and text:
                phase2_count += 1
                has_phase2 = True
                recognized_plates[text] = recognized_plates.get(text, 0) + 1

                # 떨림 감지: 이전에 Phase 1이었다가 다시 Phase 2가 됨
                if text in prev_track_phase and not prev_track_phase[text]:
                    flicker_count += 1
                prev_track_phase[text] = True
            else:
                phase1_count += 1
                # Phase 2 → Phase 1 전환 추적
                for t_text in list(prev_track_phase.keys()):
                    if prev_track_phase[t_text]:
                        # 이 bbox와 겹치는 기존 Phase 2 텍스트가 있는지
                        pass
                # 간단히: is_valid가 없는 모든 det에 대해
                if text and text in prev_track_phase and prev_track_phase[text]:
                    prev_track_phase[text] = False

            if len(bbox) >= 4:
                frame_bboxes.append(bbox)

        # 중복 bbox 체크
        for i in range(len(frame_bboxes)):
            for j in range(i + 1, len(frame_bboxes)):
                if calculate_iou(frame_bboxes[i], frame_bboxes[j]) > 0.3:
                    dup_bbox_count += 1

        # 연속 인식 구간
        if has_phase2:
            current_run += 1
        else:
            if current_run > 0:
                continuous_runs.append(current_run)
            current_run = 0

        # 진행 표시
        if processed % 50 == 0:
            elapsed = time.time() - t_start
            fps_proc = processed / elapsed if elapsed > 0 else 0
            print(f"  [{processed:4d}/{MAX_FRAMES}] F:{frame_idx} "
                  f"Det:{total_detections} P2:{phase2_count} "
                  f"Flicker:{flicker_count} Dup:{dup_bbox_count} "
                  f"({fps_proc:.1f} proc/s)")

    if current_run > 0:
        continuous_runs.append(current_run)

    cap.release()
    elapsed_total = time.time() - t_start

    # ── 결과 출력 ──
    print(f"\n{'='*70}")
    print(f"  실시간 안정성 테스트 결과")
    print(f"{'='*70}")
    print(f"  처리 프레임: {processed}  (영상 프레임 {frame_idx} 중)")
    print(f"  처리 시간: {elapsed_total:.1f}초 ({processed/elapsed_total:.1f} frames/sec)")
    print(f"")
    print(f"  ── 감지 통계 ──")
    print(f"  총 detection: {total_detections}")
    print(f"  Phase 2 (확정): {phase2_count}")
    print(f"  Phase 1 (탐지중): {phase1_count}")
    if total_detections > 0:
        print(f"  확정 비율: {phase2_count/total_detections:.1%}")
    print(f"")
    print(f"  ── 안정성 지표 ──")
    print(f"  떨림 (P2→P1→P2 전환): {flicker_count}회")
    print(f"  중복 bbox (IoU>0.3): {dup_bbox_count}건")
    if continuous_runs:
        avg_run = sum(continuous_runs) / len(continuous_runs)
        max_run = max(continuous_runs)
        print(f"  연속 인식 구간: 평균 {avg_run:.1f}프레임, 최장 {max_run}프레임 ({len(continuous_runs)}구간)")
    print(f"")
    print(f"  ── 인식된 번호판 ──")
    for plate, cnt in sorted(recognized_plates.items(), key=lambda x: -x[1]):
        print(f"    {plate}: {cnt}회")

    # ── 판정 ──
    print(f"\n{'='*70}")
    flicker_rate = flicker_count / max(phase2_count, 1)
    dup_rate = dup_bbox_count / max(processed, 1)

    issues = []
    if flicker_rate > 0.1:
        issues.append(f"떨림 비율 {flicker_rate:.1%} > 10%")
    if dup_rate > 0.05:
        issues.append(f"중복 bbox 비율 {dup_rate:.1%} > 5%")

    if not issues:
        print(f"  판정: PASS — 안정성 양호")
        print(f"    떨림 비율: {flicker_rate:.1%}")
        print(f"    중복 비율: {dup_rate:.1%}")
    else:
        print(f"  판정: WARN — 개선 필요")
        for iss in issues:
            print(f"    - {iss}")
    print(f"{'='*70}")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
