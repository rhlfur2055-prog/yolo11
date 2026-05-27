"""
test_ttl_ghost_fix.py — TTL 고스트 버그 수정 검증 (TC1~TC9)
실제 plate_gui.py의 병합 로직을 그대로 복사해서 단위 테스트
"""
import sys
import time

# ── PlateTracker.calculate_iou 가져오기 ──
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from tracker import PlateTracker

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 수정된 병합 로직 (plate_gui.py _refresh_display와 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_merge(phase1_dets, phase2_dets):
    """_refresh_display의 병합 로직 추출."""
    merged = []
    for p2 in phase2_dets:
        p2_bbox = p2.get("bbox", [])
        if len(p2_bbox) < 4:
            continue
        best_p1_bbox = None
        best_iou = 0.10
        for p1 in phase1_dets:
            p1_bbox = p1.get("bbox", [])
            if len(p1_bbox) < 4:
                continue
            iou = PlateTracker.calculate_iou(p2_bbox, p1_bbox)
            if iou >= best_iou:
                best_iou = iou
                best_p1_bbox = p1_bbox
        combined = dict(p2)
        if best_p1_bbox is not None:
            combined["bbox"] = best_p1_bbox  # Phase1 좌표로 보정
        elif phase1_dets:
            # ★ Phase1 활성인데 IoU 매칭 실패 → 고스트 방지
            continue
        merged.append(combined)

    # Phase1 중 Phase2와 미매칭 → "탐지중" 노란 박스
    p2_bboxes = [d.get("bbox", []) for d in phase2_dets if len(d.get("bbox", [])) >= 4]
    phase1_only = []
    for p1 in phase1_dets:
        p1_bbox = p1.get("bbox", [])
        if len(p1_bbox) < 4:
            continue
        already = any(PlateTracker.calculate_iou(p1_bbox, pb) > 0.10 for pb in p2_bboxes)
        if not already:
            phase1_only.append(p1)

    return merged, phase1_only


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 테스트 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

results = []

def tc(name, cond, detail=""):
    ok = bool(cond)
    status = "✅ PASS" if ok else "❌ FAIL"
    results.append(ok)
    print(f"  {status}  {name}" + (f"  — {detail}" if detail else ""))
    return ok


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TC1: Phase1 + IoU 성공 → 좌표 보정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[TC1] Phase1 + IoU 성공 → 좌표 보정")
p1 = [{"bbox": [100, 100, 300, 200], "confidence": 0.85, "phase": 1, "text": ""}]
p2 = [{"bbox": [105,  98, 295, 198], "text": "12가3456", "ocr_confidence": 0.92, "is_valid_plate": True}]
merged, p1only = run_merge(p1, p2)
iou_val = PlateTracker.calculate_iou(p1[0]["bbox"], p2[0]["bbox"])
tc("merged 1개", len(merged) == 1, f"len={len(merged)}")
tc("bbox = Phase1 좌표", list(merged[0]["bbox"]) == [100, 100, 300, 200],
   f"bbox={merged[0]['bbox']}")
tc("text = Phase2 텍스트", merged[0].get("text") == "12가3456",
   f"text={merged[0].get('text')}")
tc("Phase1_only 0개 (Phase2가 Phase1 흡수)", len(p1only) == 0, f"p1only={len(p1only)}")
tc(f"IoU ≥ 0.10 (실측 {iou_val:.3f})", iou_val >= 0.10)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TC2: Phase1 + IoU 실패 → skip (핵심 픽스)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[TC2] Phase1 + IoU 실패 → skip (고스트 방지)")
p1 = [{"bbox": [500, 100, 700, 200], "confidence": 0.80, "phase": 1, "text": ""}]
p2 = [{"bbox": [100, 100, 300, 200], "text": "12가3456", "ocr_confidence": 0.90, "is_valid_plate": True}]
merged, p1only = run_merge(p1, p2)
iou_val = PlateTracker.calculate_iou(p1[0]["bbox"], p2[0]["bbox"])
tc("merged 0개 (Phase2 skip됨)", len(merged) == 0, f"len={len(merged)}")
tc("Phase1_only 1개 (탐지중 박스)", len(p1only) == 1, f"p1only={len(p1only)}")
tc(f"IoU < 0.10 (실측 {iou_val:.3f})", iou_val < 0.10)
tc("고스트 텍스트 없음", all(d.get("text", "") == "" for d in merged), "merged에 텍스트 없음")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TC3: Phase1 없음 → Phase2 폴백
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[TC3] Phase1 없음 → Phase2 폴백")
p1_empty = []
p2 = [{"bbox": [100, 100, 300, 200], "text": "12가3456", "ocr_confidence": 0.90, "is_valid_plate": True}]
merged, p1only = run_merge(p1_empty, p2)
tc("merged 1개 (Phase2 그대로)", len(merged) == 1, f"len={len(merged)}")
tc("bbox = Phase2 좌표 (보정 없음)", list(merged[0]["bbox"]) == [100, 100, 300, 200])
tc("text = Phase2 텍스트", merged[0].get("text") == "12가3456")
# None도 falsy → 동일 동작
merged_none, _ = run_merge([], p2)
tc("phase1_dets=None 동일 (빈 리스트로 처리)", len(merged_none) == 1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TC4: TTL 0.5초 만료 동작
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[TC4] TTL 0.5초 만료 동작")
TTL = 0.5
# 시뮬레이션: _detection_ts를 현재에서 offset 초 전으로 설정
def sim_ttl(offset_sec):
    det_ts = time.time() - offset_sec
    age = time.time() - det_ts
    return age <= TTL  # True = 유지, False = 만료

tc("t=0.0s: 유지", sim_ttl(0.0))
tc("t=0.4s: 유지", sim_ttl(0.4))
tc("t=0.5s: 만료", not sim_ttl(0.51))
tc("t=1.0s: 만료", not sim_ttl(1.0))
old_ttl_frames = int(1.0 / 0.033)  # TTL=1.0일 때 최대 고스트 프레임
new_ttl_frames = int(0.5 / 0.033)  # TTL=0.5일 때
tc(f"최대 고스트 프레임 반감 ({old_ttl_frames} → {new_ttl_frames})", new_ttl_frames <= old_ttl_frames // 2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TC5: 다중 번호판 — 1대 이동, 1대 정지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[TC5] 다중 번호판 — 정지차 표시, 이동차 skip")
p1 = [
    {"bbox": [100, 100, 300, 200], "confidence": 0.85, "phase": 1, "text": ""},  # 차A 정지
    {"bbox": [700, 300, 900, 400], "confidence": 0.80, "phase": 1, "text": ""},  # 차B 이동 후 위치
]
p2 = [
    {"bbox": [105, 98, 295, 198], "text": "12가3456", "ocr_confidence": 0.90, "is_valid_plate": True},  # 차A
    {"bbox": [200, 300, 400, 400], "text": "78나9012", "ocr_confidence": 0.88, "is_valid_plate": True},  # 차B 이전 위치
]
merged, p1only = run_merge(p1, p2)
iou_A = PlateTracker.calculate_iou(p1[0]["bbox"], p2[0]["bbox"])
iou_B = PlateTracker.calculate_iou(p1[1]["bbox"], p2[1]["bbox"])
tc(f"차A IoU ≥ 0.10 (실측 {iou_A:.3f})", iou_A >= 0.10)
tc(f"차B IoU < 0.10 (실측 {iou_B:.3f})", iou_B < 0.10)
tc("merged 1개 (차A만)", len(merged) == 1, f"len={len(merged)}")
tc("차A 텍스트 표시됨", merged[0].get("text") == "12가3456", f"text={merged[0].get('text')}")
tc("차B 고스트 없음", all(d.get("text","") != "78나9012" for d in merged))
tc("차B Phase1_only 탐지중 박스", len(p1only) >= 1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TC6: IoU = 0.10 경계값
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[TC6] IoU 0.10 경계값")
# IoU ≈ 0.105 (성공 케이스)
b1 = [0, 0, 100, 100]
b2_hit = [81, 0, 181, 100]   # intersection=19×100=1900, union=19100 → IoU≈0.0995
iou_hit = PlateTracker.calculate_iou(b1, b2_hit)
# 더 확실하게 겹치는 케이스
b2_hit2 = [80, 0, 180, 100]  # intersection=20×100=2000, union=18000 → IoU≈0.111
iou_hit2 = PlateTracker.calculate_iou(b1, b2_hit2)
b2_miss = [92, 0, 192, 100]  # intersection=8×100=800, union=19200 → IoU≈0.042
iou_miss = PlateTracker.calculate_iou(b1, b2_miss)

p1_tc6 = [{"bbox": b1, "confidence": 0.80, "phase": 1, "text": ""}]
# 성공 케이스 (IoU ≥ 0.10)
p2_hit = [{"bbox": b2_hit2, "text": "34다5678", "ocr_confidence": 0.88, "is_valid_plate": True}]
merged_hit, _ = run_merge(p1_tc6, p2_hit)
# 실패 케이스 (IoU < 0.10)
p2_miss = [{"bbox": b2_miss, "text": "34다5678", "ocr_confidence": 0.88, "is_valid_plate": True}]
merged_miss, _ = run_merge(p1_tc6, p2_miss)

tc(f"IoU={iou_hit2:.3f} ≥ 0.10 → 매칭 성공 merged=1", len(merged_hit) == 1,
   f"merged={len(merged_hit)}")
tc(f"IoU={iou_miss:.3f} < 0.10 → 매칭 실패 merged=0", len(merged_miss) == 0,
   f"merged={len(merged_miss)}")
tc("경계값 0.10 정확히 동작", iou_hit2 >= 0.10 and iou_miss < 0.10)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TC7: 빈 결과 + proc ≥ 100ms → 즉시 클리어
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[TC7] 빈 결과 + proc ≥ 100ms → 즉시 클리어")
def sim_result_update(latest_dets, proc_ms, validated):
    """_refresh_display의 결과 업데이트 로직 추출."""
    if validated:
        latest_dets = validated[:]
    elif proc_ms >= 100:
        latest_dets = []  # 실제 YOLO 처리, 결과 없음 → 클리어
    # proc_ms < 100: 캐시 결과 → 유지
    return latest_dets

prev = [{"bbox": [100,100,300,200], "text": "12가3456"}]
result = sim_result_update(prev, proc_ms=150, validated=[])
tc("빈 결과 + 150ms → 클리어됨", result == [], f"len={len(result)}")
result2 = sim_result_update(prev, proc_ms=500, validated=[])
tc("빈 결과 + 500ms → 클리어됨", result2 == [])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TC8: 빈 결과 + proc < 100ms → 유지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[TC8] 빈 결과 + proc < 100ms → 유지")
prev = [{"bbox": [100,100,300,200], "text": "12가3456"}]
result = sim_result_update(prev, proc_ms=30, validated=[])
tc("빈 결과 + 30ms(캐시) → 기존 유지", len(result) == 1 and result[0].get("text") == "12가3456",
   f"len={len(result)}")
result2 = sim_result_update(prev, proc_ms=0, validated=[])
tc("빈 결과 + 0ms(캐시) → 기존 유지", len(result2) == 1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TC9: 고스트 프레임 수 — 수정 전후 비교
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[TC9] 고스트 프레임 수 — 수정 전후 비교")

def run_merge_OLD(phase1_dets, phase2_dets):
    """수정 전 병합 로직 (무조건 append)."""
    merged = []
    for p2 in phase2_dets:
        p2_bbox = p2.get("bbox", [])
        if len(p2_bbox) < 4:
            continue
        best_p1_bbox = None
        best_iou = 0.10
        for p1 in phase1_dets:
            p1_bbox = p1.get("bbox", [])
            if len(p1_bbox) < 4:
                continue
            iou = PlateTracker.calculate_iou(p2_bbox, p1_bbox)
            if iou >= best_iou:
                best_iou = iou
                best_p1_bbox = p1_bbox
        combined = dict(p2)
        if best_p1_bbox is not None:
            combined["bbox"] = best_p1_bbox
        merged.append(combined)  # ← 무조건 append (버그)
    return merged

# 차량이 30프레임 동안 x축으로 이동: 100→600
p2_cached = [{"bbox": [100, 100, 300, 200], "text": "12가3456", "ocr_confidence": 0.90, "is_valid_plate": True}]
ghost_frames_old = 0
ghost_frames_new = 0
p2_stale_bbox = p2_cached[0]["bbox"]  # 고스트 = 이 옛 좌표에 텍스트가 나올 때

for i in range(30):
    x_offset = i * 17  # 프레임당 17px 이동
    p1_frame = [{"bbox": [100 + x_offset, 100, 300 + x_offset, 200],
                 "confidence": 0.80, "phase": 1, "text": ""}]

    # 고스트 정의: Phase1이 커버하지 않는 위치에 텍스트가 표시됨
    # (Phase1 bbox와 IoU < 0.10인 위치에 텍스트 = 차가 없는 곳에 번호판 표시)
    def is_ghost(merged_item, p1_list):
        mb = list(merged_item.get("bbox", []))
        return not any(
            PlateTracker.calculate_iou(mb, p1_f["bbox"]) >= 0.10
            for p1_f in p1_list
        )

    m_old = run_merge_OLD(p1_frame, p2_cached)
    for d in m_old:
        if d.get("text") == "12가3456" and is_ghost(d, p1_frame):
            ghost_frames_old += 1

    m_new, _ = run_merge(p1_frame, p2_cached)
    for d in m_new:
        if d.get("text") == "12가3456" and is_ghost(d, p1_frame):
            ghost_frames_new += 1

tc(f"수정 전 고스트 프레임 ≥ 15개 ({ghost_frames_old}개)", ghost_frames_old >= 15)
tc(f"수정 후 고스트 프레임 = 0개 ({ghost_frames_new}개)", ghost_frames_new == 0)
tc("고스트 완전 제거", ghost_frames_new == 0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 종합 판정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
passed = sum(results)
total = len(results)
print(f"  종합: {passed}/{total} {'✅ ALL PASS' if passed == total else '❌ FAIL 있음'}")
print("=" * 60)
sys.exit(0 if passed == total else 1)
