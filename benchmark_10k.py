"""
AI Hub 10,000건 번호판 라벨 벤치마크
- validate_plate_format() 교정 정확도 측정
- correct_hangul_similarity() / correct_ocr_hangul() 정확도 측정
- 실전 CCTV 예상 정확도 추정
"""
import sys
import os
import json
import glob
import re

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

from plate_recognition_4k import (
    validate_plate_format,
    correct_ocr_hangul,
    correct_hangul_similarity,
    _VALID_PLATE_HANGUL_ALL,
    _HANGUL_PLATE_CORRECTION,
    _find_nearest_valid_hangul,
    _RE_NEW_PLATE,
    _RE_OLD_PLATE,
)


def main():
    val_dir = os.path.join(os.path.dirname(__file__), "aihub_data", "plate_ocr_val")
    files = sorted(glob.glob(os.path.join(val_dir, "*.json")))
    total = len(files)
    print(f"=== AI Hub 10K 벤치마크 ===")
    print(f"  검증 데이터: {total}건\n")

    stats = {
        "total": total,
        "format_match_new": 0,     # 신형 패턴 매칭
        "format_match_old": 0,     # 구형 패턴 매칭
        "format_no_match": 0,      # 패턴 불일치
        "hangul_valid": 0,         # 한글이 유효 52자에 포함
        "hangul_in_correction": 0, # 교정 테이블에 있음
        "hangul_jamo_fallback": 0, # 자모 유사도 폴백 사용
        "hangul_no_fix": 0,        # 교정 불가
        "validate_perfect": 0,     # score=1.0 (교정 불필요)
        "validate_corrected": 0,   # score=0.8~0.9 (교정됨)
        "validate_wrong_correction": 0,  # 교정이 원본과 다름 (오교정)
        "validate_no_change": 0,   # 교정 적용됐지만 원본과 동일
        "similarity_changed": 0,   # correct_hangul_similarity가 변경함
        "similarity_wrong": 0,     # 변경이 원본과 불일치
    }

    wrong_corrections = []
    similarity_issues = []
    no_match_samples = []

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        original = data.get("value", "").strip()
        if not original:
            continue

        # 1. 패턴 매칭 테스트
        m_new = _RE_NEW_PLATE.match(original)
        m_old = _RE_OLD_PLATE.match(original)
        if m_new:
            stats["format_match_new"] += 1
            hangul = m_new.group(2)
        elif m_old:
            stats["format_match_old"] += 1
            hangul = m_old.group(3)
        else:
            stats["format_no_match"] += 1
            if len(no_match_samples) < 20:
                no_match_samples.append(original)
            continue

        # 2. 한글 유효성 테스트
        if hangul in _VALID_PLATE_HANGUL_ALL:
            stats["hangul_valid"] += 1
        elif hangul in _HANGUL_PLATE_CORRECTION:
            stats["hangul_in_correction"] += 1
        elif _find_nearest_valid_hangul(hangul):
            stats["hangul_jamo_fallback"] += 1
        else:
            stats["hangul_no_fix"] += 1

        # 3. validate_plate_format 테스트
        corrected_text, score = validate_plate_format(original)
        if score >= 1.0:
            stats["validate_perfect"] += 1
        elif score >= 0.8:
            stats["validate_corrected"] += 1
            if corrected_text != original:
                stats["validate_wrong_correction"] += 1
                if len(wrong_corrections) < 30:
                    wrong_corrections.append((original, corrected_text, score))
            else:
                stats["validate_no_change"] += 1

        # 4. correct_hangul_similarity 테스트
        sim_result = correct_hangul_similarity(original)
        if sim_result != original:
            stats["similarity_changed"] += 1
            if len(similarity_issues) < 20:
                similarity_issues.append((original, sim_result))

    # === 결과 출력 ===
    matched = stats["format_match_new"] + stats["format_match_old"]
    print("=" * 60)
    print("  [1] 번호판 형식 매칭")
    print("=" * 60)
    print(f"  신형 (XXX가XXXX): {stats['format_match_new']:,}건 ({stats['format_match_new']/total*100:.1f}%)")
    print(f"  구형 (서울XX가XXXX): {stats['format_match_old']:,}건 ({stats['format_match_old']/total*100:.1f}%)")
    print(f"  불일치: {stats['format_no_match']:,}건 ({stats['format_no_match']/total*100:.1f}%)")
    print(f"  총 매칭률: {matched/total*100:.1f}%")

    print(f"\n{'=' * 60}")
    print("  [2] 한글 유효성 (매칭된 {0:,}건 기준)".format(matched))
    print("=" * 60)
    print(f"  유효 52자 포함: {stats['hangul_valid']:,}건 ({stats['hangul_valid']/matched*100:.2f}%)")
    print(f"  교정 테이블 매칭: {stats['hangul_in_correction']:,}건 ({stats['hangul_in_correction']/matched*100:.2f}%)")
    print(f"  자모 폴백 교정: {stats['hangul_jamo_fallback']:,}건 ({stats['hangul_jamo_fallback']/matched*100:.2f}%)")
    print(f"  교정 불가: {stats['hangul_no_fix']:,}건 ({stats['hangul_no_fix']/matched*100:.2f}%)")
    coverage = (stats['hangul_valid'] + stats['hangul_in_correction'] + stats['hangul_jamo_fallback']) / matched * 100
    print(f"  총 커버리지: {coverage:.2f}%")

    print(f"\n{'=' * 60}")
    print("  [3] validate_plate_format() 정확도")
    print("=" * 60)
    print(f"  완벽 (score=1.0, 교정 불필요): {stats['validate_perfect']:,}건 ({stats['validate_perfect']/matched*100:.2f}%)")
    print(f"  교정됨 (score=0.8~0.9): {stats['validate_corrected']:,}건")
    print(f"    └ 오교정 (원본≠교정): {stats['validate_wrong_correction']:,}건")
    if wrong_corrections:
        print(f"\n  [오교정 샘플 (최대 30건)]")
        for orig, corr, sc in wrong_corrections:
            print(f"    {orig} → {corr}  (score={sc:.2f})")

    print(f"\n{'=' * 60}")
    print("  [4] correct_hangul_similarity() 영향")
    print("=" * 60)
    print(f"  변경된 건수: {stats['similarity_changed']:,}건")
    if similarity_issues:
        print(f"  [변경 샘플]")
        for orig, changed in similarity_issues:
            print(f"    {orig} → {changed}")

    if no_match_samples:
        print(f"\n{'=' * 60}")
        print("  [5] 패턴 불일치 샘플 (최대 20건)")
        print("=" * 60)
        for s in no_match_samples:
            print(f"    {s}")

    # === 실전 CCTV 예상 정확도 계산 ===
    print(f"\n{'=' * 60}")
    print("  [6] 실전 CCTV 예상 정확도")
    print("=" * 60)

    # 교정 정확도: 유효 한글을 잘못 교정하지 않는 비율
    false_correction_rate = stats['validate_wrong_correction'] / matched * 100 if matched > 0 else 0
    # 한글 커버리지
    hangul_coverage = coverage
    # 형식 매칭률
    format_rate = matched / total * 100

    # OCR 자체 정확도 (test_short.mp4 기준: 평균 OCR conf 0.937)
    ocr_base = 0.937
    # YOLO 탐지율 (mAP@50: 0.9813)
    yolo_recall = 0.9508
    # 교정으로 인한 정확도 보정
    correction_boost = hangul_coverage / 100.0
    # 오교정으로 인한 정확도 손실
    correction_penalty = 1.0 - (false_correction_rate / 100.0)

    # 실전 예상 정확도 = YOLO 탐지율 × OCR 기본 정확도 × 교정 커버리지 × (1 - 오교정률)
    estimated_accuracy = yolo_recall * ocr_base * correction_boost * correction_penalty * 100

    print(f"  YOLO 탐지율 (Recall): {yolo_recall*100:.1f}%")
    print(f"  OCR 기본 정확도: {ocr_base*100:.1f}% (test_short.mp4 기준)")
    print(f"  한글 교정 커버리지: {hangul_coverage:.2f}%")
    print(f"  오교정률: {false_correction_rate:.2f}%")
    print(f"  형식 매칭률: {format_rate:.1f}%")
    print(f"  ────────────────────────")
    print(f"  ★ 실전 CCTV 예상 정확도: {estimated_accuracy:.1f}%")
    print(f"  (= 탐지율 × OCR정확도 × 교정커버리지 × (1-오교정률))")

    print(f"\n{'=' * 60}")
    print("  벤치마크 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
