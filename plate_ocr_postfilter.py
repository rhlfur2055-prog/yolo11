# -*- coding: utf-8 -*-
"""
plate_ocr_postfilter.py — 한국 번호판 OCR 후처리 엔진
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOLO+OCR 1차 결과에 대해:
  오탐 제거 → 패턴 검증 → 허용 한글 검증 → 숫자 구조 검증 → 중복 병합 → 유사 번호 교차검증
으로 최종 정제된 번호판 목록만 출력합니다.

입력 형식 (plate_recognition_4k 호환):
  [{"text": str, "ocr_confidence": float, "pattern_score": float,
    "detection_method": str, "sharpness": float, ...}, ...]
출력: 동일 키 유지 + detection_count, confidence_bonus, final_confidence 추가
"""

import re
from collections import defaultdict

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한국 번호판 허용 한글 (실제 번호판에 사용되는 문자만)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID_HANGUL = set(
    "가나다라마거너더러머버서어저"
    "고노도로모보소오조"
    "구누두루무부수우주"
    "아바사자배허하호"
)

# 구형 지역명 (지역+숫자2+한글+숫자4)
REGION_NAMES = (
    "서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|"
    "충북|충남|전북|전남|경북|경남|제주"
)

# 한국 번호판 정규식 (신형 2자리/3자리 + 구형 지역명)
PLATE_PATTERNS = [
    re.compile(r"^\d{2}[가-힣]\d{4}$"),   # 00가0000
    re.compile(r"^\d{3}[가-힣]\d{4}$"),   # 000가0000
    re.compile(r"^(" + REGION_NAMES + r")\d{2}[가-힣]\d{4}$"),  # 서울01가1234
]


def _levenshtein(a: str, b: str) -> int:
    """편집 거리 (Levenshtein distance)."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(
                min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (0 if ca == cb else 1),
                )
            )
        prev = curr
    return prev[-1]


def is_valid_korean_plate(text: str) -> bool:
    """
    한국 번호판 형식 검증.
    - 정규식 매칭
    - 한글 정확히 1자 + VALID_HANGUL 내 문자
    - 앞 숫자 2~3자리, 뒤 숫자 4자리
    """
    if not text or len(text) < 6:
        return False
    cleaned = re.sub(r"[^가-힣0-9]", "", text)
    if not cleaned:
        return False

    # 정규식 매칭
    if not any(p.match(cleaned) for p in PLATE_PATTERNS):
        return False

    # 한글 문자 검증 (정확히 1자, 허용 목록 내)
    hangul = [c for c in cleaned if "\uac00" <= c <= "\ud7a3"]
    if len(hangul) != 1:
        return False
    if hangul[0] not in VALID_HANGUL:
        return False

    # 숫자 구조: 한글 기준 앞 2~3자리, 뒤 4자리
    idx = next(i for i, c in enumerate(cleaned) if "\uac00" <= c <= "\ud7a3")
    before = cleaned[:idx]
    after = cleaned[idx + 1:]
    if len(before) not in (2, 3) or not before.isdigit():
        return False
    if len(after) != 4 or not after.isdigit():
        return False

    return True


def _immediate_reject(text: str) -> bool:
    """즉시 거부 조건: 영문만, 숫자만 4자 이하, 한글 2자 이상 연속 등."""
    cleaned = re.sub(r"[^가-힣0-9A-Za-z]", "", text)
    if not cleaned:
        return True
    # 영문+숫자만 (한글 없음) → 영문 전용이면 거부
    if re.match(r"^[A-Za-z0-9]+$", cleaned):
        # 숫자만 4자 이하는 거부
        if re.match(r"^\d+$", cleaned) and len(cleaned) <= 4:
            return True
        # 영문만으로 구성 (한국 번호판 아님)
        if re.match(r"^[A-Za-z]+$", cleaned) or (
            re.match(r"^[A-Za-z0-9]+$", cleaned) and not any("\uac00" <= c <= "\ud7a3" for c in cleaned)
        ):
            return True
    # 한글이 2자 이상 연속
    hangul = [c for c in cleaned if "\uac00" <= c <= "\ud7a3"]
    if len(hangul) >= 2:
        return True
    # 한글 1자인데 숫자 구조 불일치
    if len(hangul) == 1:
        idx = next((i for i, c in enumerate(cleaned) if "\uac00" <= c <= "\ud7a3"), None)
        if idx is not None:
            before = cleaned[:idx]
            after = cleaned[idx + 1:]
            if len(before) not in (2, 3) or len(after) != 4:
                return True
    return False


def filter_plates(
    results: list[dict],
    *,
    ocr_threshold: float = 0.70,
    coco_pattern_min: float = 0.90,
    plate_pattern_min: float = 0.85,
) -> list[dict]:
    """
    번호판 결과 필터링 파이프라인.

    입력 results: plate_recognition_4k 형식
      - text (또는 plate), ocr_confidence (또는 ocr), pattern_score (또는 pattern),
      - detection_method (또는 method), sharpness, 기타 키 유지

    Returns:
      정제된 결과 리스트 (동일 키 + detection_count, confidence_bonus, final_confidence)
    """
    if not results:
        return []

    # 키 정규화 (plate/ocr/pattern/method 호환)
    def _norm(r: dict) -> dict:
        out = dict(r)
        out["_plate"] = (r.get("text") or r.get("plate") or "").strip()
        out["_ocr"] = float(r.get("ocr_confidence") or r.get("ocr") or 0)
        out["_pattern"] = float(r.get("pattern_score") or r.get("pattern") or 0)
        out["_method"] = (r.get("detection_method") or r.get("method") or "").strip()
        out["_sharpness"] = float(r.get("sharpness") or 0)
        return out

    normalized = [_norm(r) for r in results]
    filtered = []

    for r in normalized:
        plate = r["_plate"]
        ocr = r["_ocr"]
        pattern = r["_pattern"]
        method = r["_method"]

        # STEP 1: OCR 신뢰도 하한선
        if ocr < ocr_threshold:
            continue

        # STEP 2: 즉시 거부 (영문 전용, 숫자만 4자 이하, 한글 2자 연속 등)
        if _immediate_reject(plate):
            continue

        # STEP 3: 한국 번호판 패턴 + 한글 검증
        if not is_valid_korean_plate(plate):
            continue

        # STEP 4: 방식별 패턴 하한선
        if method == "coco_fallback" and pattern < coco_pattern_min:
            continue
        if method == "plate_model" and pattern < plate_pattern_min:
            continue

        filtered.append(r)

    # STEP 5: 중복 병합 (같은 번호판 → OCR·선명도 우선 1건)
    best: dict[str, dict] = {}
    count: dict[str, int] = defaultdict(int)
    for r in filtered:
        plate = r["_plate"]
        count[plate] += 1
        if plate not in best:
            best[plate] = r
        else:
            b = best[plate]
            # OCR 우선, 동일하면 선명도 우선
            if r["_ocr"] > b["_ocr"] or (
                r["_ocr"] == b["_ocr"] and r["_sharpness"] > b["_sharpness"]
            ):
                best[plate] = r

    # STEP 6: 유사 번호 교차검증 (Levenshtein ≤ 2 → 동일 차량, 높은 신뢰도만 유지)
    groups: list[list[str]] = []
    used = set()

    for plate in best:
        if plate in used:
            continue
        group = [plate]
        used.add(plate)
        for other in best:
            if other in used:
                continue
            if _levenshtein(plate, other) <= 2:
                group.append(other)
                used.add(other)
        groups.append(group)

    # 각 그룹에서 최고 OCR 1건만 최종 선택
    final_plates: set[str] = set()
    for group in groups:
        best_in_group = max(
            group,
            key=lambda p: (best[p]["_ocr"], best[p]["_sharpness"]),
        )
        final_plates.add(best_in_group)

    # 최종 결과 구성 (원본 dict 복원 + 보너스)
    final = []
    for plate in final_plates:
        r = best[plate]
        c = count[plate]
        # 원본 결과로 복원 (text, ocr_confidence 등 원 키 유지)
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        out["detection_count"] = c
        out["confidence_bonus"] = 0.05 if c >= 2 else 0.0
        out["final_confidence"] = min(1.0, r["_ocr"] + out["confidence_bonus"])
        final.append(out)

    final.sort(key=lambda x: x.get("final_confidence", 0), reverse=True)
    return final


def apply_postfilter(results: list[dict], **kwargs) -> list[dict]:
    """
    plate_recognition_4k 결과에 그대로 적용.
    results 내 각 항목에 text, ocr_confidence, pattern_score, detection_method, sharpness 가 있으면 됨.
    """
    return filter_plates(results, **kwargs)
