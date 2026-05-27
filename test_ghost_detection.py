#!/usr/bin/env python3
"""
Ghost Detection 테스트 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ghost = 이전 차량의 번호판이 다음 차량에 잔존하는 현상

시나리오: 연속 차량이 지나가는 상황을 시뮬레이션
- 이미지 A를 N프레임 반복 → 이미지 B를 N프레임 반복
- B 프레임에서 A의 번호판이 나오면 Ghost 발생 → FAIL

테스트 조건:
1. 같은 위치에 다른 차량이 나타나는 경우 (worst case)
2. TTL 프레임 경과 후 다른 차량이 나타나는 경우
3. 연속 4대 차량 시퀀스
"""

import os
import sys
import cv2
import time
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from yolo11_plate.plate_engine_pro import PlateEnginePro

IMG_DIR = os.path.join(BASE_DIR, "22")

# ── 테스트 차량 정의 ──
# (파일명, 정답 번호판)
VEHICLES = [
    ("서울바9203.png",          "서울70바9203"),
    ("트럭 경기91바6286.png",   "경기91바6286"),
    ("01나8060.png",            "01나8060"),
    ("58두9599.png",            "58두9599"),
]


def normalize_plate(text: str) -> str:
    """비교용 정규화"""
    if not text:
        return ""
    text = re.sub(r"[\s\-\.\(\)（）]", "", text)
    text = text.replace("(F)", "")
    return text.strip().upper()


def load_image(filename):
    """22/ 폴더에서 이미지 로드"""
    filepath = os.path.join(IMG_DIR, filename)
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {filepath}")
    return img


def extract_plates(detections):
    """검출 결과에서 번호판 텍스트 목록 추출"""
    if not detections:
        return []
    return [normalize_plate(d.get("plate", "")) for d in detections]


def test_ghost_sequential(engine, frames_per_vehicle=5, gap_frames=0):
    """
    테스트 1: 연속 차량 시퀀스 (reset 없이)
    - 각 차량을 frames_per_vehicle 프레임 반복
    - gap_frames 만큼 빈 프레임 삽입 (TTL 소진 시뮬레이션)
    - 다음 차량 프레임에서 이전 번호판 잔존 여부 확인
    """
    print(f"\n{'='*60}")
    print(f"  테스트: 연속 차량 시퀀스 (gap={gap_frames} frames)")
    print(f"{'='*60}")

    engine.reset_state()
    engine.consecutive_required = 1  # 1프레임으로 즉시 결과

    ghost_count = 0
    total_transitions = 0

    for v_idx in range(len(VEHICLES)):
        filename, gt = VEHICLES[v_idx]
        img = load_image(filename)

        print(f"\n  [{v_idx+1}] 차량: {gt} ({filename})")

        # 해당 차량 프레임 반복
        last_plates = []
        for f in range(frames_per_vehicle):
            detections = engine.process_frame(img)
            plates = extract_plates(detections)
            last_plates = plates
            if plates:
                print(f"      프레임 {f+1}: {', '.join(plates)}")
            else:
                print(f"      프레임 {f+1}: (미인식)")

        # 빈 프레임 삽입 (검은 화면으로 TTL 소진)
        if gap_frames > 0 and v_idx < len(VEHICLES) - 1:
            black = img.copy() * 0  # 검은 화면
            for g in range(gap_frames):
                engine.process_frame(black)
            print(f"      ... (빈 프레임 {gap_frames}개 삽입)")

        # 다음 차량으로 전환 시 Ghost 검사
        if v_idx < len(VEHICLES) - 1:
            next_filename, next_gt = VEHICLES[v_idx + 1]
            next_img = load_image(next_filename)
            total_transitions += 1

            # 전환 직후 첫 프레임
            detections = engine.process_frame(next_img)
            transition_plates = extract_plates(detections)

            # Ghost 검사: 이전 차량 번호판이 나타나면 Ghost
            prev_gt_norm = normalize_plate(gt)
            has_ghost = False
            for p in transition_plates:
                if p == prev_gt_norm and p != normalize_plate(next_gt):
                    has_ghost = True
                    ghost_count += 1
                    print(f"      >>> GHOST 발견! 전환 시 이전 번호판 '{gt}' 잔존")
                    break

            if not has_ghost and transition_plates:
                print(f"      >>> 전환 정상: {', '.join(transition_plates)}")

    # 결과
    passed = ghost_count == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  결과: {status} — Ghost {ghost_count}건 / 전환 {total_transitions}회")
    return passed, ghost_count


def test_ghost_same_position(engine):
    """
    테스트 2: 동일 위치에 다른 차량 (worst case)
    - 같은 크기로 리사이즈하여 bbox가 동일 위치에 오도록 시뮬레이션
    - 트래커가 IoU 매칭으로 같은 트랙에 할당할 가능성이 높음
    - 이전 번호판 텍스트가 우선되면 Ghost
    """
    print(f"\n{'='*60}")
    print(f"  테스트: 동일 위치 차량 교체 (IoU 매칭 worst case)")
    print(f"{'='*60}")

    engine.reset_state()
    engine.consecutive_required = 1

    # 차량 A: 3프레임
    fname_a, gt_a = VEHICLES[0]
    img_a = load_image(fname_a)
    print(f"\n  차량 A: {gt_a}")
    for f in range(3):
        dets = engine.process_frame(img_a)
        plates = extract_plates(dets)
        print(f"    프레임 {f+1}: {', '.join(plates) if plates else '(미인식)'}")

    # 차량 B: 동일 크기로 리사이즈 → 같은 프레임에 넣기
    fname_b, gt_b = VEHICLES[3]
    img_b = load_image(fname_b)
    # 동일 해상도로 맞춤
    img_b_resized = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    print(f"\n  차량 B: {gt_b} (동일 해상도로 교체)")
    ghost_found = False
    for f in range(5):
        dets = engine.process_frame(img_b_resized)
        plates = extract_plates(dets)
        plate_str = ', '.join(plates) if plates else '(미인식)'
        print(f"    프레임 {f+1}: {plate_str}")

        # Ghost 검사: A의 번호판이 B 프레임에서 나타남
        gt_a_norm = normalize_plate(gt_a)
        for p in plates:
            if p == gt_a_norm:
                ghost_found = True
                print(f"    >>> GHOST! 차량 B 프레임에서 A의 번호판 '{gt_a}' 감지")

    passed = not ghost_found
    status = "PASS" if passed else "FAIL"
    print(f"\n  결과: {status}")
    return passed


def test_ghost_ttl_expiry(engine):
    """
    테스트 3: TTL 만료 후 트랙 제거 확인
    - 차량 A를 N프레임 → TTL+1 프레임 빈 화면 → 트랙이 사라졌는지 확인
    """
    print(f"\n{'='*60}")
    print(f"  테스트: TTL 만료 후 트랙 제거")
    print(f"{'='*60}")

    engine.reset_state()
    engine.consecutive_required = 1
    ttl = engine._tracker.ttl_frames

    # 차량 A: 3프레임
    fname_a, gt_a = VEHICLES[0]
    img_a = load_image(fname_a)
    print(f"\n  차량 A: {gt_a} (3프레임)")
    for f in range(3):
        engine.process_frame(img_a)

    tracks_before = len(engine._tracker.tracks)
    print(f"  TTL 전 트랙 수: {tracks_before}")

    # 빈 프레임으로 TTL+1 소진
    black = img_a.copy() * 0
    print(f"  빈 프레임 {ttl + 1}개 삽입 (TTL={ttl})...")
    for g in range(ttl + 1):
        engine.process_frame(black)

    tracks_after = len(engine._tracker.tracks)
    print(f"  TTL 후 트랙 수: {tracks_after}")

    passed = tracks_after == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  결과: {status} — 트랙 {'전부 만료됨' if passed else f'{tracks_after}개 잔존!'}")
    return passed


def test_four_vehicle_sequence(engine):
    """
    테스트 4: 연속 4대 차량 전체 시나리오
    시나리오:
        1번 차량: 서울70바9203 (버스)
        2번 차량: 경기91바6286 (트럭)
        3번 차량: 01나8060 (SUV)
        4번 차량: 58두9599 (KIA)

    검증:
        - 4번 차량 감지 시 이전 번호판이 표시되면 → FAIL (Ghost!)
        - 각 차량 전환 시 이전 결과 잔존 여부 확인
    """
    print(f"\n{'='*60}")
    print(f"  테스트: 4대 차량 연속 시나리오")
    print(f"{'='*60}")

    engine.reset_state()
    engine.consecutive_required = 1

    all_prev_plates = set()  # 이전까지 나온 모든 번호판
    ghost_events = []

    for v_idx, (filename, gt) in enumerate(VEHICLES):
        img = load_image(filename)
        gt_norm = normalize_plate(gt)

        print(f"\n  [{v_idx+1}] 차량: {gt}")

        # 5프레임 반복
        for f in range(5):
            dets = engine.process_frame(img)
            plates = extract_plates(dets)

            # Ghost 검사: 현재 차량이 아닌 이전 차량 번호판이 나타나는지
            for p in plates:
                if p in all_prev_plates and p != gt_norm:
                    ghost_events.append({
                        "frame": f + 1,
                        "vehicle": gt,
                        "ghost_plate": p,
                    })
                    print(f"    프레임 {f+1}: {', '.join(plates)} >>> GHOST: '{p}' 잔존!")
                    break
            else:
                if plates:
                    print(f"    프레임 {f+1}: {', '.join(plates)}")
                else:
                    print(f"    프레임 {f+1}: (미인식)")

        all_prev_plates.add(gt_norm)

    passed = len(ghost_events) == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  결과: {status} — Ghost 이벤트 {len(ghost_events)}건")
    if ghost_events:
        for ev in ghost_events:
            print(f"    - 차량 '{ev['vehicle']}' 프레임 {ev['frame']}: Ghost '{ev['ghost_plate']}'")
    return passed


def main():
    print("=" * 60)
    print("  Ghost Detection 테스트")
    print("=" * 60)

    # 엔진 초기화
    print("\n[초기화] PlateEnginePro 로딩...")
    t0 = time.time()
    engine = PlateEnginePro()
    print(f"[초기화] 완료 ({time.time() - t0:.1f}초)")

    results = {}

    # 테스트 1: 연속 시퀀스 (gap 없음)
    t1 = time.time()
    passed1, ghosts1 = test_ghost_sequential(engine, frames_per_vehicle=5, gap_frames=0)
    results["sequential_no_gap"] = passed1
    print(f"  (소요: {time.time()-t1:.1f}초)")

    # 테스트 2: 연속 시퀀스 (gap=TTL+1)
    t2 = time.time()
    ttl = engine._tracker.ttl_frames
    passed2, ghosts2 = test_ghost_sequential(engine, frames_per_vehicle=5, gap_frames=ttl + 1)
    results["sequential_with_gap"] = passed2
    print(f"  (소요: {time.time()-t2:.1f}초)")

    # 테스트 3: 동일 위치 교체
    t3 = time.time()
    passed3 = test_ghost_same_position(engine)
    results["same_position"] = passed3
    print(f"  (소요: {time.time()-t3:.1f}초)")

    # 테스트 4: TTL 만료
    t4 = time.time()
    passed4 = test_ghost_ttl_expiry(engine)
    results["ttl_expiry"] = passed4
    print(f"  (소요: {time.time()-t4:.1f}초)")

    # 테스트 5: 4대 차량 전체 시나리오
    t5 = time.time()
    passed5 = test_four_vehicle_sequence(engine)
    results["four_vehicle"] = passed5
    print(f"  (소요: {time.time()-t5:.1f}초)")

    # ── 최종 결과 ──
    print(f"\n{'='*60}")
    print(f"  Ghost Detection 최종 결과")
    print(f"{'='*60}")
    total_pass = sum(1 for v in results.values() if v)
    total = len(results)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "+" if passed else "x"
        print(f"  [{icon}] {name}: {status}")
    print(f"\n  총 {total}건 중 {total_pass}건 통과 ({total_pass}/{total})")

    all_passed = all(results.values())
    print(f"\n  최종 판정: {'PASS — Ghost 0건' if all_passed else 'FAIL — Ghost 발생!'}")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
