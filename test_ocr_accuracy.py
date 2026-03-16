#!/usr/bin/env python3
"""
OCR 정확도 테스트 스크립트
- 22/ 폴더의 12장 번호판 이미지에 대해 PlateEnginePro.process_frame() 실행
- 정답(Ground Truth) vs OCR 결과 비교
- 정확도 통계 출력
"""

import os
import sys
import cv2
import time
import re

# ── 경로 설정 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from plate_engine_pro import PlateEnginePro

# ── 테스트 이미지 + 정답 (Ground Truth) ──
# 파일명에서 정답 추출 (파일명 = 번호판 번호)
TEST_CASES = [
    ("경기76바7789.png",        "경기76바7789"),
    ("서울70바9203.png",        "서울70바9203"),
    ("트럭 경기91바6286.png",   "경기91바6286"),
    ("01나8060.png",            "01나8060"),
    ("02누2754.png",            "02누2754"),
    ("14나3234.png",            "14나3234"),
    ("36다7117.png",            "36다7117"),
    ("48보7062.png",            "48보7062"),
    ("55저9392.png",            "55저9392"),
    ("58두9599.png",            "58두9599"),
    ("70버6393.png",            "70버6393"),
    ("80부5915.png",            "80부5915"),
]

IMG_DIR = os.path.join(BASE_DIR, "22")


def normalize_plate(text: str) -> str:
    """비교용 정규화: 공백/특수문자 제거, 대문자 변환"""
    if not text:
        return ""
    text = re.sub(r"[\s\-\.\(\)（）]", "", text)
    text = text.replace("(F)", "")  # COCO fallback 마크 제거
    return text.strip().upper()


def plates_match(ocr_text: str, gt_text: str) -> bool:
    """정답과 OCR 결과 비교 (정규화 후)"""
    return normalize_plate(ocr_text) == normalize_plate(gt_text)


def main():
    print("=" * 70)
    print("  번호판 OCR 정확도 테스트")
    print("=" * 70)
    print()

    # ── 엔진 초기화 ──
    print("[초기화] PlateEnginePro 로딩...")
    t0 = time.time()
    engine = PlateEnginePro()
    # ★ 이미지 단독 테스트: 연속 프레임 제약 해제 (1프레임으로 즉시 결과)
    engine.consecutive_required = 1
    print(f"[초기화] 완료 ({time.time() - t0:.1f}초)")
    print()

    # ── 테스트 실행 ──
    total = len(TEST_CASES)
    correct = 0
    partial = 0
    failed = 0
    results_table = []

    for idx, (filename, gt) in enumerate(TEST_CASES, 1):
        filepath = os.path.join(IMG_DIR, filename)
        if not os.path.exists(filepath):
            results_table.append((idx, filename, gt, "파일없음", "-", "❌ 파일없음"))
            failed += 1
            continue

        img = cv2.imread(filepath)
        if img is None:
            results_table.append((idx, filename, gt, "로드실패", "-", "❌ 로드실패"))
            failed += 1
            continue

        # ★ 매 이미지마다 내부 캐시 초기화 (이전 결과 오염 방지)
        engine.reset_state()

        # process_frame 호출
        t1 = time.time()
        detections = engine.process_frame(img)
        elapsed = time.time() - t1

        if not detections:
            results_table.append((idx, filename, gt, "(미인식)", f"{elapsed:.2f}s", "❌"))
            failed += 1
            continue

        # 가장 높은 confidence 결과 사용
        best = max(detections, key=lambda d: d.get("confidence", 0))
        ocr_text = best.get("plate", "")
        ocr_conf = best.get("confidence", 0)

        if plates_match(ocr_text, gt):
            judge = "✅ 정확"
            correct += 1
        elif normalize_plate(gt) in normalize_plate(ocr_text) or normalize_plate(ocr_text) in normalize_plate(gt):
            judge = "⚠️ 부분일치"
            partial += 1
        else:
            judge = "❌ 오인식"
            failed += 1

        results_table.append((idx, filename, gt, f"{ocr_text} ({ocr_conf:.0%})", f"{elapsed:.2f}s", judge))

    # ── 결과 출력 ──
    print("-" * 70)
    print(f"{'#':>2}  {'파일명':<28} {'정답':<14} {'OCR결과':<20} {'시간':>6}  {'판정'}")
    print("-" * 70)

    for row in results_table:
        idx, fname, gt, ocr, elapsed, judge = row
        # 파일명 축약 (너무 길면)
        display_fname = fname if len(fname) <= 26 else fname[:23] + "..."
        print(f"{idx:>2}  {display_fname:<28} {gt:<14} {ocr:<20} {elapsed:>6}  {judge}")

    print("-" * 70)
    print()
    print(f"  총 {total}장  |  ✅ 정확: {correct}  |  ⚠️ 부분: {partial}  |  ❌ 실패: {failed}")
    print(f"  정확도: {correct}/{total} = {correct/total*100:.1f}%")
    if partial:
        print(f"  부분일치 포함: {(correct+partial)}/{total} = {(correct+partial)/total*100:.1f}%")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
