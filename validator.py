# -*- coding: utf-8 -*-
"""Korean license plate text validation, cleanup, and pattern reconstruction.

Extracted from plate_engine_pro.py (refactor — SRP 분리).
공개 API: PlateValidator 클래스 + 그 안의 클래스 상수
(_COMMERCIAL_CHARS, _REGION_PREFIXES, _GOV_PREFIXES_2CHAR)는
plate_engine_pro.py 본체에서 ``PlateValidator.XXX`` 형태로 외부 참조됨.
이름·시그니처 변경 금지.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from config import OCRConfig, ThresholdConfig
from plate_recognition_4k import (
    correct_ocr_hangul,
    correct_hangul_similarity,
    _HANGUL_PLATE_CORRECTION,
    _VALID_PLATE_HANGUL_ALL,
    validate_plate_format,
    _correct_region,
    _REGION_SET,
)


class PlateValidator:
    """번호판 유효성 검증기 (한글+숫자 조합, 5~10자만 허용 — config 기반).

    검증 우선순위:
      1) 정방향 패턴 매칭
      2) 구형 지역번호판 복원(앞 1~2자리 숫자 → 지역명)
      3) 순수 숫자 → 한글 누락 복원 (PaddleOCR 한글 누락 대응)
      4) 역방향/혼동 교정
      5) validate_plate_format 폴백 (자모 유사도)
    """

    # 클래스 상수 (외부 참조 — 이름 유지 필수)
    _KR_CONFUSION: dict = _HANGUL_PLATE_CORRECTION
    _COMMERCIAL_CHARS: set = set("비바사아자배하")
    _GOV_PREFIXES_2CHAR: List[str] = [
        "전기",
        "이나", "오수", "아자", "이아", "이마", "오아",
        "하나", "하다", "하라", "하마",
    ]
    _REGION_PREFIXES: List[str] = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
    ]

    # OCR 혼동 문자 폴백 (config에 없을 때 사용)
    _DEFAULT_CONFUSION = {
        "O": "0", "I": "1", "Z": "2", "S": "5",
        "B": "8", "D": "0", "Q": "0", "G": "6",
        "ㅇ": "0", "ㅣ": "1",
    }

    def __init__(self) -> None:
        self.patterns = [re.compile(p) for p in OCRConfig.KR_PATTERNS]
        self.min_len: int = ThresholdConfig.PLATE_MIN_LEN
        self.max_len: int = ThresholdConfig.PLATE_MAX_LEN

    # 내부 유틸
    def _normalize_for_validation(self, text: str) -> str:
        """공백/특수문자 제거, OCR 글자 잘림 보정용 정규화."""
        s = re.sub(r"[\s\-\.\,\;\:\'\"]", "", text)
        allowed = re.compile(r"[0-9가-힣바사아자외교]")
        return "".join(c for c in s if allowed.match(c))

    def _should_be_digit(self, text: str, pos: int) -> bool:
        if pos > 0 and text[pos - 1].isdigit():
            return True
        if pos < len(text) - 1 and text[pos + 1].isdigit():
            return True
        return False

    def _try_patterns(self, text: str) -> Tuple[bool, str]:
        """패턴 매칭 시도 (정방향 + 역방향 + 한글교정)."""
        candidates = [text, text[::-1]]
        corrected = "".join(self._KR_CONFUSION.get(c, c) for c in text)
        if corrected != text:
            candidates.append(corrected)
            candidates.append(corrected[::-1])

        for candidate in candidates:
            norm = self._normalize_for_validation(candidate)
            if not (self.min_len <= len(norm) <= self.max_len):
                continue
            for pattern in self.patterns:
                if pattern.match(norm):
                    return True, norm
        return False, text

    # 공개 API
    def is_valid_length(self, text: str) -> bool:
        clean = self._normalize_for_validation(text)
        return self.min_len <= len(clean) <= self.max_len

    def validate(self, text: str) -> Tuple[bool, str]:
        clean = self._normalize_for_validation(text)
        if not (self.min_len <= len(clean) <= self.max_len):
            rev = self._normalize_for_validation(text[::-1])
            if self.min_len <= len(rev) <= self.max_len:
                ok, result = self._try_patterns(rev)
                if ok:
                    return True, result
            return False, clean

        # 구형 지역번호판 우선 교정: 앞 1~2자리 숫자가 지역명 오인식
        # 예) 176바7789 → 경기76바7789 (앞 '1' = 지역명 OCR 잔여)
        m_reg = re.match(r'^[0-9]{1,2}([0-9]{2}([가-힣])[0-9]{4})$', clean)
        if m_reg and m_reg.group(2) in PlateValidator._COMMERCIAL_CHARS:
            suffix = m_reg.group(1)
            # "00" 연식 코드는 실제 번호판에 없음 → 허위감지 차단
            if suffix[:2] != "00":
                for region in PlateValidator._REGION_PREFIXES:
                    candidate = region + suffix
                    nc = self._normalize_for_validation(candidate)
                    for pattern in self.patterns:
                        if pattern.match(nc):
                            return True, nc

        # 정방향 패턴 매칭
        for pattern in self.patterns:
            if pattern.match(clean):
                # 한글 유효성 추가 검증: 번호판에 쓰이지 않는 한글이면 교정
                fmt_corrected, fmt_score = validate_plate_format(clean)
                if fmt_score > 0 and fmt_corrected != clean:
                    return True, fmt_corrected
                return True, clean

        # 순수 숫자 7~9자리 → 한글 누락 복원
        digits_only = re.match(r'^[0-9]{7,9}$', clean)
        if digits_only:
            corrected = correct_ocr_hangul(clean)
            if corrected != clean:
                for pattern in self.patterns:
                    if pattern.match(corrected):
                        return True, corrected

            # 스마트 한글 삽입: 뒤 4자리 고정 → 한글 위치 탐색
            suffix = clean[-4:]
            if suffix.isdigit():
                _n = len(clean)
                _split_order = [3, 2, 4] if _n >= 8 else [2, 3, 4]
                for split_pos in _split_order:
                    if len(clean) >= split_pos + 5:
                        prefix = clean[:split_pos]
                        mid_digit = clean[split_pos]
                        from plate_recognition_4k import _HANGUL_CONFUSE_MAP
                        mapped = _HANGUL_CONFUSE_MAP.get(mid_digit)
                        if mapped:
                            candidate = prefix + mapped + suffix
                            for pattern in self.patterns:
                                if pattern.match(candidate):
                                    return True, candidate
                        for h in _VALID_PLATE_HANGUL_ALL:
                            candidate = prefix + h + suffix
                            for pattern in self.patterns:
                                if pattern.match(candidate):
                                    return True, candidate

        # 역방향 / 혼동 교정
        ok, result = self._try_patterns(clean)
        if ok:
            return True, result

        # 최종 폴백: validate_plate_format (한글 교정 테이블 + 자모 유사도)
        fmt_corrected, fmt_score = validate_plate_format(clean)
        if fmt_score > 0:
            for pattern in self.patterns:
                if pattern.match(fmt_corrected):
                    return True, fmt_corrected

        return False, clean

    def clean_ocr_text(self, text: str) -> str:
        """OCR 후처리: 특수문자 완전 제거 + 혼동문자 보정 + 지역명 교정."""
        clean = text.strip()
        # 한글 자모(ㅣㅡ등) + 특수문자 명시적 제거
        clean = re.sub(r'[|/\\.\-ㅣㅡㅏㅓㅗㅜㅐㅔㅑㅕ]', '', clean)
        clean = re.sub(r"[^\w가-힣]", "", clean, flags=re.ASCII)
        clean = re.sub(r"\s+", "", clean)

        replacements = OCRConfig.CONFUSION_MAP or self._DEFAULT_CONFUSION
        result = []
        for i, ch in enumerate(clean):
            if ch in replacements and self._should_be_digit(clean, i):
                result.append(replacements[ch])
            else:
                result.append(ch)
        cleaned = "".join(result)

        # 한글 보정 (plate_recognition_4k 테이블)
        cleaned = correct_ocr_hangul(cleaned)
        cleaned = correct_hangul_similarity(cleaned)

        # 지역명 오인식 보정 (시울→서울 등) — 7자 이상
        if len(cleaned) >= 7:
            prefix2 = cleaned[:2]
            corrected_region = _correct_region(prefix2)
            if corrected_region != prefix2 and corrected_region in _REGION_SET:
                cleaned = corrected_region + cleaned[2:]

        return cleaned
