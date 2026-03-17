"""
plate_recognition_4k.py - 4K 영상 번호판 인식 핵심 모듈 v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[핵심 변경사항 v1 → v2]
- 번호판 전용 YOLO 모델 (HuggingFace: morsetechlab/yolov11-license-plate-detection)
  → COCO 범용 모델의 2단계 탐지(차량→번호판) 제거, 직접 번호판 탐지
  → mAP@50: 0.9813 (Precision: 0.9893, Recall: 0.9508)
- 듀얼 탐지 전략: 번호판 직접 탐지 + COCO 차량 크롭 폴백
- 한국 번호판 OCR 최적화: allowlist + 패턴 검증
- 소형 크롭 3x 업스케일 + 최소 차량 크기 필터

사용법:
    python plate_recognition_4k.py video.mp4 -o ./results
    python plate_recognition_4k.py video.mp4 --no-sahi
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse
from enum import Enum, auto

# Windows 콘솔 한글 깨짐 방지
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from typing import Optional

import cv2
import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON 직렬화 유틸리티 (ndarray 안전 변환)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# config.py 중앙 설정 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os as _os
from config import PathConfig, ThresholdConfig, OCRConfig, DisplayConfig

def _load_best_model():
    """우선순위에 따라 가장 좋은 모델 자동 로드"""
    from ultralytics import YOLO
    best = PathConfig.find_best_model()
    print(f"[YOLO26] 모델 로드: {best}")
    return YOLO(best)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 기본 설정값
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# -- 모델 설정 (config.py에서 가져옴) --
HF_PLATE_REPO: str = PathConfig.HF_PLATE_REPO
HF_PLATE_FILE: str = PathConfig.HF_PLATE_FILE
LOCAL_ENGINE_MODEL: str = PathConfig.YOLO_ENGINE
LOCAL_ONNX_MODEL: str = PathConfig.YOLO_ONNX
LOCAL_PLATE_MODEL: str = PathConfig.YOLO_FALLBACK
DEFAULT_PLATE_MODEL_SIZE: str = PathConfig.DEFAULT_MODEL_SIZE
COCO_MODEL: str = PathConfig.YOLO_COCO_FALLBACK
VEHICLE_CLASS_IDS: set[int] = ThresholdConfig.VEHICLE_CLASS_IDS

# -- 탐지 설정 (config.py 통일값) --
DEFAULT_CONFIDENCE: float = ThresholdConfig.DETECT_CONF
MIN_VEHICLE_WIDTH: int = ThresholdConfig.MIN_VEHICLE_WIDTH
MIN_VEHICLE_HEIGHT: int = ThresholdConfig.MIN_VEHICLE_HEIGHT
MIN_PLATE_WIDTH: int = ThresholdConfig.MIN_PLATE_WIDTH
MIN_PLATE_HEIGHT: int = ThresholdConfig.MIN_PLATE_HEIGHT
PLATE_MIN_ASPECT: float = ThresholdConfig.PLATE_MIN_ASPECT
PLATE_MAX_ASPECT: float = ThresholdConfig.PLATE_MAX_ASPECT
PLATE_MAX_AREA_RATIO: float = ThresholdConfig.PLATE_MAX_AREA_RATIO
MAX_PLATE_TEXT_LEN: int = ThresholdConfig.MAX_PLATE_TEXT_LEN
MIN_OCR_CONFIDENCE: float = ThresholdConfig.OCR_CONF
MIN_DET_CONFIDENCE: float = ThresholdConfig.MIN_DET_CONFIDENCE
CONFIRM_FRAME_COUNT: int = ThresholdConfig.CONFIRM_FRAME_COUNT
UPSCALE_THRESHOLD: int = ThresholdConfig.UPSCALE_THRESHOLD
UPSCALE_FACTOR: int = ThresholdConfig.UPSCALE_FACTOR

# -- 프레임 스킵 & 버스트 캡처 (config.py) --
DEFAULT_FRAME_SKIP: int = ThresholdConfig.FRAME_SKIP
BURST_FRAME_COUNT: int = ThresholdConfig.BURST_FRAME_COUNT
NO_DETECT_TOLERANCE: int = ThresholdConfig.NO_DETECT_TOLERANCE

# -- Detection Log OCR --
LOG_OCR_INTERVAL: int = ThresholdConfig.LOG_OCR_INTERVAL

# -- SAHI 타일링 (config.py) --
SAHI_SLICE_SIZE: int = ThresholdConfig.SAHI_SLICE_SIZE
SAHI_OVERLAP_RATIO: float = ThresholdConfig.SAHI_OVERLAP_RATIO

# -- 크롭 & 선명도 (config.py) --
PLATE_PADDING_RATIO: float = ThresholdConfig.PLATE_PADDING_RATIO
PLATE_MODEL_PADDING_H: float = ThresholdConfig.PLATE_MODEL_PADDING_H
PLATE_MODEL_PADDING_V: float = ThresholdConfig.PLATE_MODEL_PADDING_V
SHARPNESS_THRESHOLD: float = ThresholdConfig.SHARPNESS_THRESHOLD

# -- 시간축 앙상블 (config.py) --
TEMPORAL_WINDOW: int = ThresholdConfig.TEMPORAL_WINDOW
TEMPORAL_LEVENSHTEIN_MAX: int = ThresholdConfig.TEMPORAL_LEVENSHTEIN_MAX

# -- 한국 번호판 OCR 설정 (config.py에서 가져옴) --
KOREAN_PLATE_HANGUL = OCRConfig.KOREAN_PLATE_HANGUL
KOREAN_PLATE_ALLOWLIST = OCRConfig.KOREAN_PLATE_ALLOWLIST
KOREAN_PLATE_PATTERNS = OCRConfig.KR_COMPILED_PATTERNS
INTERNATIONAL_PLATE_PATTERNS = OCRConfig.INTL_COMPILED_PATTERNS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상태 머신
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CaptureState(Enum):
    """
    프레임 캡처 상태 머신

    SCANNING  → 프레임 스킵 적용 (속도 우선)
    TRACKING  → 번호판 최초 탐지 (버스트 준비)
    CAPTURING → 모든 프레임 분석 (화질 우선, 최고 선명도 확보)
    """
    SCANNING = auto()
    TRACKING = auto()
    CAPTURING = auto()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 다운로드 유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_plate_model(
    size: str = DEFAULT_PLATE_MODEL_SIZE,
    cache_dir: Optional[str] = None,
) -> str:
    """
    HuggingFace에서 번호판 전용 YOLO 모델 다운로드 (캐시 지원)

    Args:
        size: 모델 크기 (n/s/m/l/x)
        cache_dir: 캐시 디렉토리 (None이면 기본 HF 캐시)

    Returns:
        다운로드된 .pt 파일의 로컬 경로
    """
    # 내부 상수 정리: 기존 전역과 호환
    HF_REPO_ID = HF_PLATE_REPO
    HF_MODEL_VARIANTS = {k: HF_PLATE_FILE for k in ("n", "s", "m", "l", "x")}
    if size not in HF_MODEL_VARIANTS:
        raise ValueError(f"지원하지 않는 모델 크기: {size} (가능: {list(HF_MODEL_VARIANTS.keys())})")

    filename = HF_MODEL_VARIANTS[size]

    # 로컬에 이미 있으면 그대로 사용
    if os.path.isfile(filename):
        print(f"  [모델] 로컬 파일 사용: {filename}")
        return filename

    try:
        from huggingface_hub import hf_hub_download

        print(f"  [모델] HuggingFace에서 다운로드 중: {HF_REPO_ID}/{filename}")
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            cache_dir=cache_dir,
        )
        print(f"  [모델] 다운로드 완료: {local_path}")
        return local_path

    except ImportError:
        print("  [경고] huggingface_hub 미설치 → pip install huggingface_hub")
        print("  [폴백] COCO 범용 모델로 전환합니다.")
        return ""
    except Exception as e:
        print(f"  [경고] 모델 다운로드 실패: {e}")
        print("  [폴백] COCO 범용 모델로 전환합니다.")
        return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한국 번호판 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_korean_plate(text: str) -> tuple[bool, str, float]:
    """
    번호판 패턴 검증 및 정규화 (한국 + 국제 형식 지원)

    Args:
        text: OCR 인식 텍스트

    Returns:
        (is_valid, normalized_text, pattern_score)
        - is_valid: 패턴 매칭 여부
        - normalized_text: 공백/특수문자 제거된 정규화 텍스트
        - pattern_score: 패턴 신뢰도 (0.0~1.0)
    """
    # 공백, 특수문자 제거
    cleaned = re.sub(r"[^가-힣0-9A-Za-z]", "", text)
    cleaned_upper = cleaned.upper()

    if not cleaned:
        return False, "", 0.0

    # 1순위: 한국 번호판 패턴 (한글 포함)
    pattern_scores = [1.0, 0.95, 0.90, 0.85, 0.88, 0.45, 0.40, 0.35]
    for i, pattern in enumerate(KOREAN_PLATE_PATTERNS):
        match = pattern.search(cleaned)
        if match:
            matched_text = match.group()
            score = pattern_scores[i] if i < len(pattern_scores) else 0.3
            return True, matched_text, score

    # 2순위: 국제 번호판 패턴 (UK/EU/US)
    intl_scores = [0.85, 0.80, 0.75, 0.55, 0.60]
    for i, pattern in enumerate(INTERNATIONAL_PLATE_PATTERNS):
        match = pattern.search(cleaned_upper)
        if match:
            matched_text = match.group().replace(" ", "")
            if len(matched_text) >= 5:  # 최소 5자 이상
                score = intl_scores[i] if i < len(intl_scores) else 0.5
                return True, matched_text, score

    # 패턴 불일치라도 글자 수 기반 기본 score 부여 (후보 유지)
    # 5자 이상이면 부분 인식 가능성 → 버리지 않음
    fallback_score = min(len(re.sub(r"[^가-힣0-9A-Z]", "", cleaned_upper)) / 10.0, 0.30)
    return False, cleaned, fallback_score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한국 번호판 OCR 오인식 보정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 신형 번호판: 숫자2-3 + 한글1 + 숫자4  (예: 39가9665)
# EasyOCR이 한글 자리를 숫자/영문으로 오인식하는 패턴 → 한글 후보 매핑
_HANGUL_CONFUSE_MAP: dict[str, str] = {
    # 숫자 오인식 (자주 발생)
    "2": "가", "7": "나", "4": "라", "0": "오",
    "3": "가", "1": "이", "5": "마", "6": "바",
    "8": "바", "9": "자",
    # 영문 오인식
    "A": "아", "B": "바", "C": "소", "D": "다",
    "E": "어", "F": "하", "G": "거", "H": "하",
    "I": "이", "J": "자", "K": "카", "L": "나",
    "M": "마", "N": "나", "O": "오", "P": "파",
    "Q": "고", "R": "라", "S": "서", "T": "다",
    "U": "우", "V": "나", "W": "마", "X": "사",
    "Y": "아", "Z": "자",
}

# 한글↔한글 혼동 보정 (OCR이 유사한 한글을 잘못 읽는 경우)
# 앞뒤 숫자 패턴 확인 후에만 적용
_HANGUL_SIMILAR_MAP: dict[str, str] = {
    "시": "저",  # 시(2건)→저(2001건), ㅅ↔ㅈ + ㅣ↔ㅓ 혼동
    "차": "저",  # 차(305건)↔저(2001건), ㅈ↔ㅊ 혼동 빈발 (빈도비 6.5:1)
    "지": "자",  # 지(1건)→자(2085건)
    "히": "하",  # 히(1건)→하(2058건)
    "에": "아",  # 에(0건)→아(5600건)
    "배": "바",  # 배(153건)→바(5600건), 영업용 오인식 빈발
    # ※ 너(1966건)는 유효 한글 - 교정 대상에서 제외
}

# 신형 번호판 정규식: 숫자 2-3자리 + 오인식문자1개 + 숫자 4자리
_PLATE_CORRECTION_RE = re.compile(
    r"^(\d{2,3})"            # 앞 숫자
    r"([가-힣0-9A-Z])"       # 한글 자리 (오인식 포함)
    r"(\d{4})$"              # 뒤 숫자 4자리
)


def correct_ocr_hangul(text: str) -> str:
    """
    신형 번호판 패턴(숫자+한글+숫자)에서 한글 자리 오인식 보정.

    예: "3929665" → "39가9665" (가 자리에서 2 오인식 보정)
    """
    cleaned = re.sub(r"[^가-힣0-9A-Z]", "", text.upper())

    m = _PLATE_CORRECTION_RE.match(cleaned)
    if not m:
        return text  # 패턴 불일치 시 원본 반환

    prefix, mid, suffix = m.group(1), m.group(2), m.group(3)

    # 한글인 경우: 한글↔한글 유사 보정 (시→사, 지→자, 히→하 등)
    if "\uac00" <= mid <= "\ud7a3":
        corrected = _HANGUL_SIMILAR_MAP.get(mid, mid)
        return prefix + corrected + suffix

    # 숫자/영문이 한글 자리에 온 경우: 한글로 변환
    hangul = _HANGUL_CONFUSE_MAP.get(mid.upper())
    if hangul:
        return prefix + hangul + suffix

    return text


def correct_hangul_similarity(text: str) -> str:
    """
    번호판 텍스트 내 한글↔한글 유사 문자 보정.

    correct_ocr_hangul과 달리 모든 패턴에 적용 (앞뒤 숫자 확인).
    예: "54시555" → "54저555", "54차555" → "54저555", "에447" → "아447"
    """
    if not text:
        return text
    result = list(text)
    for i, ch in enumerate(result):
        if ch in _HANGUL_SIMILAR_MAP:
            # 앞 또는 뒤에 숫자가 있으면 번호판 한글로 판단
            before_digit = i > 0 and result[i - 1].isdigit()
            after_digit = i < len(result) - 1 and result[i + 1].isdigit()
            if before_digit or after_digit:
                result[i] = _HANGUL_SIMILAR_MAP[ch]
    return "".join(result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한국 번호판 형식 교정 테이블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 번호판에 사용 가능한 한글 (용도별 분류)
_VALID_PLATE_HANGUL_PRIVATE = set("가나다라마바사아자차카타파하")  # 자가용 (Row 1)
_VALID_PLATE_HANGUL_RENTAL = set("거너더러머버서어저처커터퍼허")  # 렌터카 (Row 2)
_VALID_PLATE_HANGUL_ROW3 = set("고노도로모보소오조호")  # 자가용 확장 (AI Hub 검증: 각 1600~2100건)
_VALID_PLATE_HANGUL_ROW4 = set("구누두루무부수우주")  # 자가용 확장 (AI Hub 검증: 각 999~2074건)
_VALID_PLATE_HANGUL_BUSINESS = set("배육")  # 영업용/특수 (AI Hub: 배=153, 육=16)
_VALID_PLATE_HANGUL_REGION = set("서울부산대구인천광주대전울산세종경기강원충북충남전북전남경북경남제주")  # 지역명
_VALID_PLATE_HANGUL_ALL = (
    _VALID_PLATE_HANGUL_PRIVATE | _VALID_PLATE_HANGUL_RENTAL
    | _VALID_PLATE_HANGUL_ROW3 | _VALID_PLATE_HANGUL_ROW4
    | _VALID_PLATE_HANGUL_BUSINESS | _VALID_PLATE_HANGUL_REGION
)

# 번호판에 절대 쓰이지 않는 한글 → 가장 유사한 유효 한글로 보정
# ※ AI Hub 90,000건 실데이터 기반 교정 (2026-02 갱신)
# ※ 고노도로모보소오조호 / 구누두루무부수우주 는 실제 유효 한글 (각 1000~2100건 확인)
_HANGUL_PLATE_CORRECTION: dict[str, str] = {
    # OCR이 빈번하게 혼동하는 자가용 한글 (Row 1 유사자)
    "기": "가", "개": "가", "깨": "가", "까": "가", "간": "가",
    "내": "나", "네": "나", "니": "나", "닝": "나", "녀": "너",
    "디": "다", "데": "다", "대": "다",
    "리": "라", "래": "라", "레": "라",
    "미": "마", "매": "마", "메": "마",
    "비": "바", "베": "바", "뱌": "바", "빠": "바",
    "세": "사", "새": "사",
    "이": "아", "에": "아", "애": "아", "여": "아",
    "제": "자", "재": "자",
    "체": "차", "채": "차",
    "키": "카", "케": "카",
    "티": "타", "테": "타",
    "피": "파", "페": "파",
    "혜": "하", "해": "하",
    # 렌터카 한글 혼동 (Row 2 유사자 → Row 2로 교정)
    "그": "거",
    # Row 2 전용 교정 (거너더러머버서어저 유사자)
    "초": "처", "추": "처",
    "코": "커", "쿠": "커",
    "토": "터", "투": "터",
    "포": "퍼", "푸": "퍼",
    "후": "허",
    # Row 3/4 유사자 → 가장 빈도 높은 유효 한글로 교정
    "곧": "고", "괴": "고",
    "뇌": "노", "놈": "노",
    "됨": "도", "돼": "도",
    "뢰": "로", "룰": "루",
    "묘": "모", "뮤": "무",
    "뵈": "보", "볼": "보",
    "쇼": "소", "숲": "수",
    "왜": "오", "워": "우",
    "죄": "조", "줄": "주",
    "혹": "호",
}

# ── OCR 오인식 한글 교정 확장 (plate_ocr_pipeline.py 통합, 개선4) ──
# EasyOCR/PaddleOCR가 빈번하게 오인식하는 패턴 → 실제 번호판 한글로 매핑
_HANGUL_OCR_EXTENDED: dict[str, str] = {
    # EasyOCR 오인식 (받침 포함 문자 → 용도 한글)
    '륙': '바', '릎': '바', '휴': '바', '푹': '바', '선': '바',
    '춤': '바', '식': '바', '겸': '바', '겨': '바', '겪': '바',
    '릅': '바', '륜': '바', '륨': '바', '륩': '바',
    '늑': '나', '닉': '나', '냑': '나',
    '딕': '다', '딘': '다', '덕': '다',
    '럭': '라', '럽': '라', '렉': '라',
    '먹': '마', '멕': '마', '먕': '마',
    '벽': '버', '볍': '버', '벡': '버',
    '석': '서', '섭': '서', '섞': '서',
    '억': '어', '엌': '어',
    '젝': '저', '젖': '저', '젊': '저',
    '곡': '고', '곤': '고', '곧': '고',
    '녹': '노', '논': '노', '놉': '노',
    '독': '도', '돈': '도', '돋': '도',
    '록': '로', '론': '로', '롯': '로',
    '목': '모', '몬': '모', '몫': '모',
    '복': '보', '본': '보', '볼': '보',
    '속': '소', '손': '소', '솔': '소',
    '옥': '오', '온': '오', '올': '오',
    '족': '조', '존': '조', '졸': '조',
    '국': '구', '군': '구', '굿': '구',
    '눈': '누', '눌': '누', '눔': '누',
    '둔': '두', '둘': '두', '둠': '두',
    '룬': '루', '룰': '루', '룸': '루',
    '문': '무', '물': '무', '뭄': '무',
    '분': '부', '불': '부', '붐': '부',
    '순': '수', '술': '수', '숨': '수',
    '운': '우', '울': '우', '움': '우',
    '준': '주', '줄': '주', '줌': '주',
    '헌': '허', '헐': '허', '험': '허',
    '한': '하', '할': '하', '함': '하',
    '혼': '호', '홀': '호', '홈': '호',
    '백': '배', '밸': '배', '뱅': '배',
}
# _HANGUL_PLATE_CORRECTION 에 없는 항목만 병합
for _k, _v in _HANGUL_OCR_EXTENDED.items():
    if _k not in _HANGUL_PLATE_CORRECTION:
        _HANGUL_PLATE_CORRECTION[_k] = _v


def _jamo_decompose(ch: str) -> tuple[int, int, int]:
    """한글 1글자를 초성/중성/종성 인덱스로 분해."""
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return (-1, -1, -1)
    cho = code // 588
    jung = (code % 588) // 28
    jong = code % 28
    return (cho, jung, jong)


# 유효 한글 52자의 자모 분해 캐시
_VALID_HANGUL_JAMO: list[tuple[str, tuple[int, int, int]]] = [
    (ch, _jamo_decompose(ch)) for ch in _VALID_PLATE_HANGUL_ALL
    if ch not in _VALID_PLATE_HANGUL_REGION  # 지역명 한글은 제외 (단자 교정용)
]


def _find_nearest_valid_hangul(ch: str) -> str | None:
    """자모 유사도 기반으로 가장 가까운 유효 번호판 한글 반환."""
    cho, jung, jong = _jamo_decompose(ch)
    if cho < 0:
        return None
    best_ch = None
    best_dist = 999
    for valid_ch, (v_cho, v_jung, v_jong) in _VALID_HANGUL_JAMO:
        # 초성 일치 가중치 0, 불일치 3 / 중성 불일치 2 / 종성 불일치 1
        dist = (0 if cho == v_cho else 3) + (0 if jung == v_jung else 2) + (0 if jong == v_jong else 1)
        if dist < best_dist:
            best_dist = dist
            best_ch = valid_ch
    # 거리 5 이하만 교정 (너무 다른 글자는 교정하지 않음)
    return best_ch if best_dist <= 5 else None


# ── 지역명 교정 테이블 (개선4) ──
_REGION_LIST = [
    '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
    '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주',
]
_REGION_SET = set(_REGION_LIST)

_REGION_CORRECTION: dict[str, str] = {
    # 경기
    '걍기': '경기', '겅기': '경기', '견기': '경기',
    '경끼': '경기', '경키': '경기', '껭기': '경기', '격기': '경기',
    '전기': '경기', '점기': '경기', '정기': '경기',  # ★ 전기→경기 오인식
    # 서울
    '서을': '서울', '서운': '서울', '셔울': '서울',
    '서욿': '서울', '석울': '서울', '시울': '서울', '시을': '서울',
    # 인천
    '인쳔': '인천', '인촌': '인천', '인첨': '인천',
    # 부산
    '부선': '부산', '부샨': '부산',
    # 대구
    '대귀': '대구', '대굴': '대구', '대국': '대구',
    # 대전
    '대잔': '대전', '대젼': '대전', '대졘': '대전',
    # 광주
    '괄주': '광주', '광쥬': '광주', '괌주': '광주',
    # 울산
    '울선': '울산',
    # 강원
    '깡원': '강원', '강월': '강원', '강완': '강원',
    # 충북
    '충붂': '충북', '총북': '충북',
    # 충남
    '총남': '충남', '충납': '충남', '충나': '충남',
    # 전북
    '전붂': '전북', '젼북': '전북',
    # 전남
    '젼남': '전남', '전납': '전남',
    # 경북
    '겅북': '경북', '경붂': '경북',
    # 경남
    '겅남': '경남', '경납': '경남',
    # 제주
    '재주': '제주', '제쥬': '제주', '졔주': '제주',
    # 세종
    '셰종': '세종', '세좀': '세종',
}


def _correct_region(text: str) -> str:
    """지역명 2자 교정"""
    if text in _REGION_SET:
        return text
    return _REGION_CORRECTION.get(text, text)


def _find_region_in_text(text: str) -> str | None:
    """텍스트에서 지역명(2자) 후보 찾기"""
    hangul_only = re.findall(r'[가-힣]', text)
    if len(hangul_only) >= 2:
        for i in range(len(hangul_only) - 1):
            pair = hangul_only[i] + hangul_only[i + 1]
            corrected = _correct_region(pair)
            if corrected in _REGION_SET:
                return corrected
    return None


# 숫자 자리에 나타나는 영문 → 숫자 교정
_DIGIT_CORRECTION: dict[str, str] = {
    'O': '0', 'o': '0', 'Q': '0', 'D': '0',
    'I': '1', 'l': '1', '|': '1', 'i': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'B': '8', 'b': '6',
    'G': '6', 'g': '9',
    'T': '7', 'A': '4',
}


def _correct_single_hangul(ch: str) -> str:
    """한글 1자 교정: 유효 번호판 한글이면 그대로, 아니면 교정 테이블 참조"""
    if ch in _VALID_PLATE_HANGUL_ALL:
        return ch
    return _HANGUL_PLATE_CORRECTION.get(ch, ch)


# 신형 번호판 구조 검증 정규식
_RE_NEW_PLATE = re.compile(r"^(\d{2,3})([가-힣])(\d{4})$")     # 123가4567
_RE_OLD_PLATE = re.compile(r"^([가-힣]{2})(\d{1,2})([가-힣])(\d{4})$")  # 서울12가1234


def validate_plate_format(text: str) -> tuple[str, float]:
    """
    한국 번호판 형식 엄격 교정.

    검증 규칙:
    1. 신형 (2019~): 숫자2-3 + 한글1 + 숫자4  (예: 123가4567)
       - 한글 위치가 유효 번호판 한글인지 확인
       - 유효하지 않으면 교정 테이블로 보정
    2. 구형: 지역명2 + 숫자1-2 + 한글1 + 숫자4  (예: 서울12가1234)

    Returns:
        (corrected_text, format_score)
        - corrected_text: 교정된 번호판 텍스트
        - format_score: 형식 신뢰도 (0.0~1.0, 1.0=완벽한 형식)
    """
    if not text:
        return text, 0.0

    # 신형 번호판 검증
    m = _RE_NEW_PLATE.match(text)
    if m:
        prefix, hangul, suffix = m.group(1), m.group(2), m.group(3)
        if hangul in _VALID_PLATE_HANGUL_ALL:
            return text, 1.0  # 완벽한 형식
        # 유효하지 않은 한글 → 교정 시도 (테이블 → 자모 유사도 폴백)
        corrected = _HANGUL_PLATE_CORRECTION.get(hangul)
        if corrected:
            return prefix + corrected + suffix, 0.90
        corrected = _find_nearest_valid_hangul(hangul)
        if corrected:
            return prefix + corrected + suffix, 0.80  # 자모 유사도 교정

    # 구형 번호판 검증
    m = _RE_OLD_PLATE.match(text)
    if m:
        region, num, hangul, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        # 지역명 검증
        if all(c in _VALID_PLATE_HANGUL_REGION for c in region):
            if hangul in _VALID_PLATE_HANGUL_ALL:
                return text, 1.0
            corrected = _HANGUL_PLATE_CORRECTION.get(hangul)
            if corrected:
                return region + num + corrected + suffix, 0.90
            corrected = _find_nearest_valid_hangul(hangul)
            if corrected:
                return region + num + corrected + suffix, 0.80
        return text, 0.50

    return text, 0.0  # 형식 불일치


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UK/국제 번호판 OCR 오인식 보정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# OCR에서 자주 혼동되는 문자 쌍 (영문/숫자 구분)
# 번호판 앞 2자리는 반드시 영문 → 숫자처럼 생긴 영문 복원
# 번호판 3-4번째는 반드시 숫자 → 영문처럼 생긴 숫자 복원
# 3,4번 자리(숫자): 영문→숫자 변환
_OCR_LETTER_TO_NUM: dict[str, str] = {
    "O": "0", "I": "1", "B": "8", "S": "5", "G": "6",
    "E": "6", "Z": "2", "L": "1", "A": "4",
}
# 1,2번 / 5,6,7번 자리(알파벳): 숫자→영문 변환
_OCR_NUM_TO_LETTER: dict[str, str] = {
    "0": "O", "1": "I", "8": "B", "5": "S", "6": "G", "4": "A",
    "2": "Z", "7": "T",
}

# UK 신형 번호판: AB12 CDE (2문자 + 2숫자 + 3문자)
_UK_PLATE_RE = re.compile(r"^([A-Z0-9]{2})([A-Z0-9]{2})([A-Z0-9]{3})$")


def correct_ocr_uk(text: str) -> str:
    """
    UK/국제 영문 번호판 OCR 오인식 보정.

    영문/숫자 혼동 패턴을 UK 번호판 형식에 맞게 정규화:
    - 위치 1-2 (영문 자리): 숫자→영문 변환 (예: 0→O, 1→I)
    - 위치 3-4 (숫자 자리): 영문→숫자 변환 (예: O→0, I→1)
    - 위치 5-7 (영문 자리): 숫자→영문 변환
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    m = _UK_PLATE_RE.match(cleaned)
    if not m:
        return text

    part1, part2, part3 = m.group(1), m.group(2), m.group(3)

    # part1/part3: 영문 자리 - 숫자로 오인식된 것 복원
    corrected_p1 = "".join(_OCR_NUM_TO_LETTER.get(c, c) for c in part1)
    corrected_p3 = "".join(_OCR_NUM_TO_LETTER.get(c, c) for c in part3)
    # part2: 숫자 자리 - 영문으로 오인식된 것 복원
    corrected_p2 = "".join(_OCR_LETTER_TO_NUM.get(c, c) for c in part2)

    return corrected_p1 + corrected_p2 + corrected_p3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PlateRecognizer 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PlateRecognizer:
    """
    4K 영상 번호판 인식기 v2.0

    [듀얼 탐지 전략]
    1차 탐지: 번호판 전용 YOLO 모델 (직접 탐지, 높은 정확도)
    2차 폴백: COCO 모델로 차량 탐지 → OpenCV 번호판 추출

    [핵심 개선사항 v1 → v2]
    1. 번호판 전용 모델: mAP@50 0.9813 → 직접 번호판 바운딩박스 출력
    2. 한국 번호판 OCR: allowlist로 인식 문자 제한 + 패턴 검증
    3. 소형 크롭 업스케일: 120px 이하 → 3x 확대 후 OCR
    4. SAHI 타일링: 4K 원본 640x640 분할 → 소형 번호판 탐지
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = DEFAULT_PLATE_MODEL_SIZE,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        use_sahi: bool = True,
        sahi_slice_size: int = SAHI_SLICE_SIZE,
        sahi_overlap: float = SAHI_OVERLAP_RATIO,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        burst_frames: int = BURST_FRAME_COUNT,
        ocr_languages: Optional[list[str]] = None,
    ) -> None:
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.use_sahi = use_sahi
        self.sahi_slice_size = sahi_slice_size
        self.sahi_overlap = sahi_overlap
        self.frame_skip = frame_skip
        self.burst_frames = burst_frames
        self.ocr_languages = ocr_languages or ["ko", "en"]

        # 상태 머신
        self.state = CaptureState.SCANNING
        self.burst_counter: int = 0
        self.no_detect_count: int = 0

        # 추론 전략 (process_video에서 결정)
        self._imgsz: Optional[int] = None
        self._use_sahi_for_video: bool = False

        # 모델 타입 플래그
        self._is_plate_model: bool = False

        # 번호판 추적 상태 (confirmed 승격용)
        self._plate_tracker: dict[str, dict] = {}  # text → {count, first_frame, last_frame, best_result}
        self._confirmed_plates: list[dict] = []  # 3프레임 이상 연속 인식된 확정 번호판

        # Detection Log OCR 캐시 (화면 내 텍스트 번호판)
        self._log_ocr_cache: dict[str, int] = {}  # plate_text → first_frame
        self._last_log_ocr_frame: int = -LOG_OCR_INTERVAL  # 마지막 log OCR 프레임

        # 부분 인식 → 전체 인식 캐시
        self._partial_cache: dict[str, tuple] = {}

        # bbox 기반 OCR 스킵 캐시 (동일 위치 반복 OCR 방지)
        self._bbox_cache: list[dict] = []  # [{xyxy, text, frame_idx, ...}]
        self._bbox_cache_max: int = 20

        # ROI 다각형 (정규화 좌표, 예: [(0.3,0.2),(0.7,0.2),(0.8,0.9),(0.2,0.9)])
        self.roi_polygon_norm: Optional[list[tuple[float, float]]] = None

        # 모델 로드
        self._load_models(model_path)
        self._init_ocr()

    # ── ROI 설정/검사 ─────────────────────────────────────
    def set_roi_polygon(self, pts_norm: list[tuple[float, float]] | None) -> None:
        """
        ROI 다각형을 [0,1] 정규화 좌표로 설정. None이면 ROI 비활성화.
        """
        self.roi_polygon_norm = pts_norm if pts_norm else None

    @staticmethod
    def _point_in_polygon(nx: float, ny: float, poly: list[tuple[float, float]]) -> bool:
        """
        정규화 좌표계에서 포인트가 다각형 내부인지 검사 (ray casting).
        """
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersect = ((yi > ny) != (yj > ny)) and (nx < (xj - xi) * (ny - yi) / (yj - yi + 1e-9) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    def _bbox_center_in_roi(self, bbox: list[float], frame_w: int, frame_h: int) -> bool:
        """
        bbox 중심점이 ROI 다각형 내부인지 검사. ROI 미설정 시 True.
        """
        if not self.roi_polygon_norm:
            return True
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0 / max(1, frame_w)
        cy = (y1 + y2) / 2.0 / max(1, frame_h)
        return self._point_in_polygon(cx, cy, self.roi_polygon_norm)

    # ── 모델 초기화 ──────────────────────────────────────

    def _load_models(self, model_path: Optional[str] = None) -> None:
        """
        모델 로드 (5단계 우선순위):
        0. 사용자 지정 모델 (--model 인자)
        1. 로컬 yolo26.engine (TensorRT FP16)
        2. 로컬 yolo26.onnx (ONNX Runtime GPU)
        3. HuggingFace 번호판 전용 (nickmuchi/yolov5-base-plates-detection)
        4. 로컬 yolo26.pt (번호판 전용)
        5. yolo11n.pt 차량 탐지 폴백
        """
        from ultralytics import YOLO

        # 0) 사용자 지정 모델 경로가 있으면 최우선 사용
        if model_path and os.path.isfile(model_path):
            self.model = YOLO(model_path)
            basename = os.path.basename(model_path).lower()
            self._is_plate_model = ("plate" in basename or "license" in basename
                                    or "yolo26" in basename)
            model_type = "번호판 전용" if self._is_plate_model else "사용자 지정"
            print(f"  [모델] {model_type} 모델 로드: {model_path}")
            self.model_path = model_path
            self._coco_model: Optional[object] = None
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 1순위: 로컬 yolo26.engine (TensorRT FP16 - 최고 속도)
        engine_path = os.path.join(script_dir, LOCAL_ENGINE_MODEL)
        if os.path.isfile(engine_path):
            try:
                self.model = YOLO(engine_path, task="detect")
                self._is_plate_model = True
                self.model_path = engine_path
                print(f"  [모델] 1순위 TensorRT 엔진: {LOCAL_ENGINE_MODEL}")
                self._coco_model: Optional[object] = None
                return
            except Exception as e:
                print(f"  [경고] TensorRT 엔진 로드 실패: {e}")

        # 2순위: 로컬 yolo26.onnx (ONNX Runtime GPU - 2~3배 빠름)
        onnx_path = os.path.join(script_dir, LOCAL_ONNX_MODEL)
        if os.path.isfile(onnx_path):
            try:
                self.model = YOLO(onnx_path, task="detect")
                self._is_plate_model = True
                self.model_path = onnx_path
                print(f"  [모델] 2순위 ONNX Runtime: {LOCAL_ONNX_MODEL}")
                self._coco_model: Optional[object] = None
                return
            except Exception as e:
                print(f"  [경고] ONNX 모델 로드 실패: {e}")

        # 3순위: HuggingFace 번호판 전용 모델 (hf_hub_download)
        try:
            from huggingface_hub import hf_hub_download
            hf_path = hf_hub_download(repo_id=HF_PLATE_REPO, filename=HF_PLATE_FILE)
            self.model = YOLO(hf_path)
            self._is_plate_model = True
            self.model_path = hf_path
            print(f"  [모델] 3순위 HuggingFace 번호판 전용: {HF_PLATE_REPO}")
            self._coco_model: Optional[object] = None
            return
        except Exception as e:
            print(f"  [경고] HuggingFace 모델 로드 실패: {e}")

        # 4순위: 로컬 yolo26.pt (자동 판별: 번호판 전용 or COCO)
        local_plate = os.path.join(script_dir, LOCAL_PLATE_MODEL)
        if os.path.isfile(local_plate):
            self.model = YOLO(local_plate)
            # 클래스 이름으로 번호판 전용 모델인지 자동 판별
            names = self.model.names or {}
            name_values = [str(v).lower() for v in names.values()]
            is_plate = any(
                kw in n for n in name_values
                for kw in ("plate", "license", "번호판")
            )
            self._is_plate_model = is_plate
            self.model_path = local_plate
            model_type = "번호판 전용" if is_plate else "COCO 범용 (차량 탐지)"
            print(f"  [모델] 4순위 로컬 모델: {LOCAL_PLATE_MODEL} ({model_type})")
            self._coco_model: Optional[object] = None
            return

        # 5순위: yolo11n.pt 차량 탐지 폴백
        coco_path = os.path.join(script_dir, COCO_MODEL)
        if not os.path.isfile(coco_path):
            coco_path = COCO_MODEL  # ultralytics 자동 다운로드
        print(f"  [모델] 5순위 COCO 차량 탐지 폴백: {COCO_MODEL}")
        self.model = YOLO(coco_path)
        self._is_plate_model = False
        self.model_path = coco_path
        self._coco_model: Optional[object] = None

    def _load_coco_fallback(self) -> None:
        """COCO 폴백 모델 지연 로드 (필요할 때만)"""
        if self._coco_model is not None:
            return
        try:
            from ultralytics import YOLO
            script_dir = os.path.dirname(os.path.abspath(__file__))
            coco_path = os.path.join(script_dir, COCO_MODEL)
            if not os.path.isfile(coco_path):
                coco_path = COCO_MODEL
            self._coco_model = YOLO(coco_path)
            print(f"  [폴백] COCO 모델 추가 로드: {COCO_MODEL}")
        except Exception as e:
            print(f"  [경고] COCO 폴백 모델 로드 실패: {e}")

    def _init_ocr(self) -> None:
        """OCR 리더 초기화: PaddleOCR 한국어(1순위) → EasyOCR 한국어(2순위) → EasyOCR 영문(3순위) → Tesseract"""
        self._ocr_engine = "none"
        self._en_reader = None        # EasyOCR 영문 전용
        self._ko_reader = None        # EasyOCR 한국어+영문
        self._tess_cmd = None
        self.paddle_reader = None

        # 1순위: PaddleOCR 한국어 (한글 인식 최강, 최우선)
        _PADDLE_MODEL_ROOT = str(PathConfig.paddle_model_dir())
        _DET_DIR = f"{_PADDLE_MODEL_ROOT}/det/ml/Multilingual_PP-OCRv3_det_infer"
        _REC_DIR = f"{_PADDLE_MODEL_ROOT}/rec/korean/korean_PP-OCRv4_rec_infer"
        _CLS_DIR = f"{_PADDLE_MODEL_ROOT}/cls/ch_ppocr_mobile_v2.0_cls_infer"
        _has_models = (
            os.path.isfile(f"{_DET_DIR}/inference.pdmodel")
            and os.path.isfile(f"{_REC_DIR}/inference.pdmodel")
        )
        try:
            from paddleocr import PaddleOCR
            paddle_kwargs: dict = {
                "use_angle_cls": True,
                "lang": "korean",
                "use_gpu": False,
                "enable_mkldnn": True,
                "cpu_threads": 4,
            }
            if _has_models:
                paddle_kwargs.update({
                    "det_model_dir": _DET_DIR,
                    "rec_model_dir": _REC_DIR,
                    "cls_model_dir": _CLS_DIR,
                })
            self.paddle_reader = PaddleOCR(**paddle_kwargs)
            self._ocr_engine = "paddle"
            print(f"  [OCR] 1순위 PaddleOCR 한국어 초기화 완료 (한글 인식 최강)")
        except Exception as e:
            print(f"  [OCR] PaddleOCR 초기화 실패: {e}")

        # 2순위: EasyOCR 한국어+영문 (PaddleOCR 보조, 앙상블)
        try:
            import easyocr
            gpu = self._is_gpu_available()
            self._ko_reader = easyocr.Reader(["ko", "en"], gpu=gpu)
            if self._ocr_engine == "none":
                self._ocr_engine = "easyocr_ko"
            print(f"  [OCR] 2순위 EasyOCR (ko+en) 한국어 초기화 완료 [GPU={gpu}]")
        except Exception as e:
            print(f"  [OCR] EasyOCR ko+en 초기화 실패: {e}")

        # 3순위: EasyOCR 영문 전용 (국제 번호판 폴백)
        try:
            import easyocr
            gpu = self._is_gpu_available()
            self._en_reader = easyocr.Reader(["en"], gpu=gpu)
            if self._ocr_engine == "none":
                self._ocr_engine = "easyocr_en"
            print(f"  [OCR] 3순위 EasyOCR (en 전용) 초기화 완료")
        except Exception as e:
            print(f"  [OCR] EasyOCR en 초기화 실패: {e}")

        # 4순위: Tesseract (영문+숫자 whitelist, 최후 수단)
        try:
            import pytesseract
            _TESS_PATH = PathConfig.tesseract_cmd()
            if os.path.isfile(_TESS_PATH):
                pytesseract.pytesseract.tesseract_cmd = _TESS_PATH
            pytesseract.get_tesseract_version()
            self._tess_cmd = True
            if self._ocr_engine == "none":
                self._ocr_engine = "tesseract"
            print(f"  [OCR] 4순위 Tesseract 초기화 완료 (PSM7, 영문+숫자)")
        except Exception as e:
            print(f"  [OCR] Tesseract 초기화 실패: {e}")

    # ── 해상도 전략 ─────────────────────────────────────

    def _determine_strategy(self, w: int, h: int) -> tuple[int, bool]:
        """프레임 해상도 기반 추론 전략 결정"""
        max_dim = max(w, h)

        if max_dim <= 1920:
            return 1280, False
        elif max_dim <= 2560:
            return 1920, False
        else:
            if self.use_sahi:
                return self.sahi_slice_size, True
            else:
                return 1920, False

    # ── 탐지 엔진 ────────────────────────────────────────

    def _detect_with_sahi(self, frame: np.ndarray) -> list[dict]:
        """SAHI 슬라이스 추론 (4K 프레임)"""
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError:
            print("  [경고] SAHI 미설치 → 직접 추론 폴백")
            return self._detect_direct(frame)

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device="cuda:0" if self._is_gpu_available() else "cpu",
        )

        result = get_sliced_prediction(
            image=frame,
            detection_model=detection_model,
            slice_height=self.sahi_slice_size,
            slice_width=self.sahi_slice_size,
            overlap_height_ratio=self.sahi_overlap,
            overlap_width_ratio=self.sahi_overlap,
        )

        fh, fw = frame.shape[:2]
        frame_area = fw * fh

        detections: list[dict] = []
        for pred in result.object_prediction_list:
            # 번호판 전용 모델: 모든 탐지가 번호판
            # COCO 모델: 차량 클래스만 필터
            if not self._is_plate_model and pred.category.id not in VEHICLE_CLASS_IDS:
                continue
            bbox = pred.bbox
            bw = bbox.maxx - bbox.minx
            bh = bbox.maxy - bbox.miny
            if self._is_plate_model and bh > 0:
                aspect = bw / bh
                if aspect < PLATE_MIN_ASPECT or aspect > PLATE_MAX_ASPECT:
                    continue
                if (bw * bh) / frame_area > PLATE_MAX_AREA_RATIO:
                    continue
            detections.append({
                "xyxy": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
                "confidence": pred.score.value,
                "class_id": pred.category.id,
                "is_plate": self._is_plate_model,
            })

        return detections

    def _detect_direct(self, frame: np.ndarray) -> list[dict]:
        """직접 YOLO 추론 (TTA + NMS IoU 최적화)"""
        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=self.confidence_threshold,
            iou=0.45,
            verbose=False,
        )

        fh, fw = frame.shape[:2]
        frame_area = fw * fh

        detections: list[dict] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())

                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]

                if self._is_plate_model:
                    # 번호판 전용 모델: 크기 + 가로세로비 + 면적비 필터
                    if bw < MIN_PLATE_WIDTH or bh < MIN_PLATE_HEIGHT:
                        continue
                    aspect = bw / bh if bh > 0 else 0
                    if aspect < PLATE_MIN_ASPECT or aspect > PLATE_MAX_ASPECT:
                        continue
                    area_ratio = (bw * bh) / frame_area if frame_area > 0 else 0
                    if area_ratio > PLATE_MAX_AREA_RATIO:
                        continue
                else:
                    # COCO 모델: car 클래스(class_id=2)만 필터
                    if cls_id != 2:
                        continue
                    if bw < MIN_VEHICLE_WIDTH or bh < MIN_VEHICLE_HEIGHT:
                        continue

                detections.append({
                    "xyxy": xyxy,
                    "confidence": conf,
                    "class_id": cls_id,
                    "is_plate": self._is_plate_model,
                })

        return detections

    def _detect_coco_fallback(self, frame: np.ndarray) -> list[dict]:
        """COCO 모델로 차량 탐지 (폴백)"""
        self._load_coco_fallback()
        if self._coco_model is None:
            return []

        results = self._coco_model.predict(
            source=frame,
            imgsz=640,
            conf=max(self.confidence_threshold - 0.1, 0.25),
            iou=0.45,
            verbose=False,
        )

        detections: list[dict] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())

                if cls_id not in VEHICLE_CLASS_IDS:
                    continue

                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                if bw < MIN_VEHICLE_WIDTH or bh < MIN_VEHICLE_HEIGHT:
                    continue

                detections.append({
                    "xyxy": xyxy,
                    "confidence": conf,
                    "class_id": cls_id,
                    "is_plate": False,
                })

        return detections

    def detect_plates(self, frame: np.ndarray) -> list[dict]:
        """번호판 탐지 (해상도 기반 전략 자동 선택)"""
        h, w = frame.shape[:2]

        if self._imgsz is None:
            self._imgsz, self._use_sahi_for_video = self._determine_strategy(w, h)
            strategy = (
                f"SAHI 타일링 ({self.sahi_slice_size}x{self.sahi_slice_size})"
                if self._use_sahi_for_video
                else f"직접 추론 (imgsz={self._imgsz})"
            )
            model_type = "번호판 전용" if self._is_plate_model else "COCO 범용"
            print(f"  [전략] {w}x{h} → {strategy} | 모델: {model_type}")

        if self._use_sahi_for_video:
            return self._detect_with_sahi(frame)
        else:
            return self._detect_direct(frame)

    # ── 크롭 & 전처리 ───────────────────────────────────

    def _crop_region(
        self,
        frame: np.ndarray,
        box_xyxy: list[float],
        padding_ratio: float = PLATE_PADDING_RATIO,
        padding_left: float | None = None,
        padding_right: float | None = None,
        padding_top: float | None = None,
        padding_bottom: float | None = None,
    ) -> np.ndarray:
        """원본 프레임에서 영역 크롭 (패딩 포함, 4방향 비대칭 지원)"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box_xyxy

        box_w = x2 - x1
        box_h = y2 - y1
        pad_left = box_w * (padding_left if padding_left is not None else padding_ratio)
        pad_right = box_w * (padding_right if padding_right is not None else padding_ratio)
        pad_top = box_h * (padding_top if padding_top is not None else padding_ratio)
        pad_bottom = box_h * (padding_bottom if padding_bottom is not None else padding_ratio)

        x1 = max(0, int(x1 - pad_left))
        y1 = max(0, int(y1 - pad_top))
        x2 = min(w, int(x2 + pad_right))
        y2 = min(h, int(y2 + pad_bottom))

        return frame[y1:y2, x1:x2].copy()

    def _upscale_if_small(self, img: np.ndarray) -> np.ndarray:
        """소형 크롭 업스케일 (개선2: LANCZOS4 + 언샤프 마스킹)"""
        h, w = img.shape[:2]
        short_side = min(h, w)
        if short_side < 100:
            # 단변 100px 미만이면 LANCZOS4로 업스케일
            scale = max(UPSCALE_FACTOR, 100 / short_side)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            # 업스케일 후 경량 샤프닝
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
            img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
            img = np.clip(img, 0, 255).astype(np.uint8)
        elif w < UPSCALE_THRESHOLD:
            img = cv2.resize(
                img,
                (w * UPSCALE_FACTOR, h * UPSCALE_FACTOR),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return img

    def _extract_plate_from_vehicle(self, vehicle_img: np.ndarray) -> list[np.ndarray]:
        """
        차량 크롭에서 번호판 영역 추출 (COCO 폴백 전용)

        차량 하단 50%에서 윤곽선 기반 번호판 검출
        """
        vh, vw = vehicle_img.shape[:2]
        if vh < 30 or vw < 30:
            return []

        bottom_half = vehicle_img[vh // 2:, :]
        bh, bw = bottom_half.shape[:2]

        gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 30, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plates: list[np.ndarray] = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (bh * bw * 0.005):
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
            if rh == 0:
                continue
            aspect = rw / rh
            if not (1.5 <= aspect <= 6.0):
                continue

            x, y, w_r, h_r = cv2.boundingRect(contour)
            plate_y1 = vh // 2 + y
            plate_y2 = vh // 2 + y + h_r
            plate_x1 = x
            plate_x2 = x + w_r

            plate_crop = vehicle_img[plate_y1:plate_y2, plate_x1:plate_x2]
            if plate_crop.size > 0 and plate_crop.shape[0] > 5 and plate_crop.shape[1] > 10:
                plates.append(plate_crop)
            if len(plates) >= 3:
                break

        if not plates:
            bottom_third = vehicle_img[vh * 2 // 3:, vw // 4: vw * 3 // 4]
            if bottom_third.size > 0:
                plates.append(bottom_third)

        return plates

    def _deskew_plate(self, img: np.ndarray) -> np.ndarray:
        """
        기울기 보정 (개선2: HoughLines 우선 → minAreaRect 폴백)
        HoughLines가 직선 각도를 더 정확하게 추출하여 OCR 정확도 향상.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        angle = 0.0

        # 방법1: HoughLines 기반 각도 추출 (개선2)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min(img.shape[1] // 3, 80))
        if lines is not None and len(lines) > 0:
            angles = []
            for rho, theta in lines[:20, 0]:
                deg = np.degrees(theta) - 90  # 수평선 기준 각도
                if abs(deg) < 20:  # 수평에 가까운 선만
                    angles.append(deg)
            if angles:
                angle = float(np.median(angles))

        # 방법2: minAreaRect 폴백
        if abs(angle) < 0.3:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            coords = cv2.findNonZero(thresh)
            if coords is not None and len(coords) > 10:
                rect = cv2.minAreaRect(coords)
                (_, _), (w_r, h_r), rect_angle = rect
                if w_r < h_r:
                    rect_angle = rect_angle + 90
                if abs(rect_angle) > 45:
                    rect_angle = rect_angle - 90 if rect_angle > 0 else rect_angle + 90
                angle = rect_angle

        # 작은 기울기만 보정 (±15도 이내)
        if abs(angle) < 0.5 or abs(angle) > 15:
            return img
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _normalize_plate_size(self, img: np.ndarray, min_height: int = 64) -> np.ndarray:
        """
        크기 정규화 + 고품질 업스케일 + 경량 샤프닝.

        소형 번호판(20px)을 LANCZOS4로 확대 후 언샤프 마스킹으로
        블러 보상. 과도한 확대는 오히려 노이즈를 증폭하므로 64px 최소.
        """
        h, w = img.shape[:2]
        if h >= min_height:
            return img
        scale = min_height / h
        new_w = int(w * scale)
        # LANCZOS4: 업스케일 시 가장 선명한 보간법
        upscaled = cv2.resize(img, (new_w, min_height), interpolation=cv2.INTER_LANCZOS4)
        # 경량 언샤프 마스킹: 업스케일 블러 보상
        blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        번호판 OCR 전처리 파이프라인 (개선2: CLAHE 3.0/(4,4) + 샤프닝)
        """
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # CLAHE (개선2: tileGridSize 8→4, 로컬 대비 더 강화)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # 언샤프 마스킹 (개선2: 샤프닝 추가)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        binary = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11, C=2,
        )

        return binary

    def _preprocess_plate_enhanced(self, plate_img: np.ndarray) -> np.ndarray:
        """
        강화된 전처리 (개선3: 앙상블용 추가 전처리 변형)
        컬러 유지 + CLAHE + 샤프닝 (한글 인식에 유리)
        """
        if len(plate_img.shape) == 2:
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)

        # LAB 컬러 공간에서 L 채널만 CLAHE
        lab = cv2.cvtColor(plate_img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 샤프닝
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _preprocess_plate_soft(self, plate_img: np.ndarray) -> np.ndarray:
        """
        소프트 전처리 (한글 자획 보존용)
        이진화 없이 CLAHE + 샤프닝만 적용하여 한글 자획 파괴를 방지.
        """
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # 샤프닝 커널
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    def _preprocess_plate_deblur(self, plate_img: np.ndarray) -> np.ndarray:
        """
        흐릿한 번호판 디블러링 전처리 (CCTV 원거리 번호판용)

        언샤프 마스킹(Unsharp Masking) + 강한 CLAHE 적용:
        - 모션 블러 / 초점 흐림 번호판에 효과적
        - 흐릿한 번호판 분석 기술의 핵심 원리 적용
        """
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # 1단계: 가우시안 블러로 배경 추정
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)

        # 2단계: 언샤프 마스킹 (원본 - 블러 = 엣지 강화)
        # amount=1.5: 엣지를 1.5배 강화
        unsharp = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)

        # 3단계: CLAHE로 대비 극대화
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(unsharp)

        # 4단계: 미디안 블러로 노이즈 제거 (엣지는 유지)
        denoised = cv2.medianBlur(enhanced, 3)

        return denoised

    def _preprocess_plate_deblur_strong(self, plate_img: np.ndarray) -> np.ndarray:
        """
        강력한 디블러링 + Otsu 이진화 (심하게 흐릿한 UK/CCTV 번호판용)

        흐릿한 번호판 분석 기술 적용:
        - 언샤프 마스킹 최대 강도 (amount=3.5)
        - CLAHE 고강도 (clipLimit=6.0)
        - Otsu's 자동 이진화 (흰색/노란색 배경 UK 번호판 최적)
        - 모폴로지 Close 연산으로 글자 획 보완
        """
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # 1단계: 강력한 언샤프 마스킹
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
        unsharp = cv2.addWeighted(gray, 3.5, blurred, -2.5, 0)
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)

        # 2단계: CLAHE 고강도 대비 극대화
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(unsharp)

        # 3단계: Otsu's 이진화 (자동 임계값, 흰배경+검정글자 최적)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4단계: 모폴로지 Close - 글자 획 단절 보완
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def _preprocess_plate_stretch(self, plate_img: np.ndarray) -> np.ndarray:
        """
        히스토그램 스트레칭 (저조도/과노출 CCTV 번호판 복원)

        픽셀 밝기 범위를 0-255로 확장:
        - 야간/황혼 CCTV 번호판 (어두운 이미지) 개선
        - 역광/과노출 번호판 개선
        - 대비가 낮아 OCR이 실패하는 경우 해결
        """
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # 히스토그램 스트레칭: min~max → 0~255
        min_val = int(gray.min())
        max_val = int(gray.max())
        if max_val > min_val:
            stretched = ((gray.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            stretched = gray

        # 스트레칭 후 CLAHE로 미세 대비 조정
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(stretched)

    def _preprocess_plate_gamma(self, plate_img: np.ndarray, gamma: float = 1.8) -> np.ndarray:
        """
        감마 보정 (황혼/야간 CCTV 어두운 번호판 밝게)

        gamma > 1.0: 이미지 밝게 (어두운 번호판용)
        gamma < 1.0: 이미지 어둡게 (과노출용)
        LUT(Look-Up Table) 방식으로 빠르게 처리
        """
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        inv_gamma = 1.0 / gamma
        table = np.array([
            min(255, int((i / 255.0) ** inv_gamma * 255))
            for i in range(256)
        ], dtype=np.uint8)
        brightened = cv2.LUT(gray, table)

        # 감마 보정 후 CLAHE로 대비 유지
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        return clahe.apply(brightened)

    def _ocr_plate_tesseract(self, img: np.ndarray) -> tuple[str, float]:
        """
        Tesseract OCR - 번호판 특화 설정 (PSM 7: 단일 텍스트 라인)

        UK/국제 번호판에서 EasyOCR보다 높은 정확도 제공:
        - PSM 7: 전체 이미지를 하나의 텍스트 줄로 처리 (번호판 최적)
        - whitelist: 영문+숫자만 허용 (노이즈 제거)
        - OEM 3: LSTM + 레거시 엔진 (정확도 극대화)
        """
        try:
            import pytesseract
            _TESS_PATH = PathConfig.tesseract_cmd()
            if os.path.isfile(_TESS_PATH):
                pytesseract.pytesseract.tesseract_cmd = _TESS_PATH

            if len(img.shape) == 2:
                img_input = img
            else:
                img_input = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # PSM 7: 단일 텍스트 라인, OEM 3: LSTM
            config = (
                "--psm 7 --oem 3 "
                "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            text = pytesseract.image_to_string(img_input, config=config).strip()
            text = re.sub(r"[^A-Z0-9]", "", text.upper())

            if not text:
                return "", 0.0

            # Tesseract confidence
            data = pytesseract.image_to_data(
                img_input, config=config,
                output_type=pytesseract.Output.DICT,
            )
            confs = [c for c in data["conf"] if isinstance(c, (int, float)) and c > 0]
            avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.5

            return text, avg_conf
        except Exception:
            return "", 0.0

    # ── OCR ──────────────────────────────────────────────

    def _ocr_plate(self, img: np.ndarray, use_allowlist: bool = False) -> tuple[str, float]:
        """
        번호판 텍스트 추출 (EasyOCR 영문 우선)

        Args:
            img: 번호판 이미지
            use_allowlist: 미사용 (호환성 유지)
        """
        # EasyOCR 영문 전용으로 시도
        if self._en_reader is not None:
            try:
                if len(img.shape) == 2:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = img
                result = self._en_reader.readtext(
                    img_bgr, detail=1, paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )
                if result:
                    result.sort(key=lambda r: r[0][0][0])
                    texts = [r[1].strip() for r in result if r[1].strip()]
                    confs = [r[2] for r in result if r[1].strip()]
                    combined = "".join(texts)
                    avg_conf = sum(confs) / len(confs) if confs else 0.0
                    if combined:
                        return combined, avg_conf
            except Exception:
                pass
        # PaddleOCR 폴백
        if self.paddle_reader is not None:
            return self._ocr_plate_paddle(img)
        return "", 0.0

    def _ocr_plate_paddle(self, img: np.ndarray) -> tuple[str, float]:
        """PaddleOCR로 번호판 텍스트 추출"""
        # PaddleOCR는 BGR 또는 RGB numpy array 모두 지원
        # 그레이스케일이면 BGR로 변환
        if len(img.shape) == 2:
            img_input = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_input = img

        try:
            result = self.paddle_reader.ocr(img_input, cls=True)
        except Exception:
            return "", 0.0

        if not result or not result[0]:
            return "", 0.0

        # result[0]은 라인별 리스트: [[bbox, (text, conf)], ...]
        lines = result[0]
        if not lines:
            return "", 0.0

        # x좌표 기준 좌→우 정렬
        def get_x(item):
            try:
                return item[0][0][0]  # 좌상단 x
            except (IndexError, TypeError):
                return 0

        lines_sorted = sorted(lines, key=get_x)

        texts: list[str] = []
        total_conf = 0.0
        count = 0
        for item in lines_sorted:
            try:
                text = item[1][0].strip()
                conf = float(item[1][1])
                if text:
                    texts.append(text)
                    total_conf += conf
                    count += 1
            except (IndexError, TypeError):
                continue

        combined = "".join(texts)
        avg_conf = total_conf / count if count > 0 else 0.0
        return combined, avg_conf

    def _ocr_plate_easyocr(self, img: np.ndarray, use_allowlist: bool = False) -> tuple[str, float]:
        """EasyOCR로 번호판 텍스트 추출"""
        kwargs = {
            "detail": 1,
            "paragraph": False,
            "text_threshold": 0.3,    # 한글 인식 민감도 높임 (기본 0.7)
            "low_text": 0.3,          # 저대비 한글 감지
            "link_threshold": 0.3,    # 문자 연결 임계값
        }
        if use_allowlist:
            kwargs["allowlist"] = KOREAN_PLATE_ALLOWLIST

        results = self.reader.readtext(img, **kwargs)

        if not results:
            return "", 0.0

        # 위치 순서(좌→우)로 정렬하여 번호판 텍스트 순서 보존
        results.sort(key=lambda r: r[0][0][0])  # x좌표 기준 정렬

        texts: list[str] = []
        total_conf = 0.0
        count = 0
        for _bbox, text, conf in results:
            text = text.strip()
            if text:
                texts.append(text)
                total_conf += conf
                count += 1

        combined = "".join(texts)  # 공백 없이 연결 (번호판은 연속)
        avg_conf = total_conf / count if count > 0 else 0.0

        return combined, avg_conf

    @staticmethod
    def _has_korean(text: str) -> bool:
        """텍스트에 한글이 포함되어 있는지 검사."""
        return any('\uac00' <= c <= '\ud7a3' or '\u3131' <= c <= '\u3163' for c in text)

    @staticmethod
    def _clean_en_text(text: str) -> str:
        """영문+숫자만 남기고 나머지 제거."""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    # ── 국가별 번호판 패턴 ──
    _RE_CN_PLATE = re.compile(
        r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}$"  # 京A12345, 沪B9C888
    )
    _RE_KR_PLATE = re.compile(
        r"^[0-9]{2,3}[\uac00-\ud7a3][0-9]{4}$"  # 12가3456, 123가4567
    )
    _RE_UK_PLATE = re.compile(
        r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$"  # BG65USJ
    )

    @staticmethod
    def _detect_plate_country(text: str) -> str:
        """번호판 텍스트에서 국가 자동 판별."""
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return "CN"
        if any('\uac00' <= c <= '\ud7a3' or '\u3131' <= c <= '\u3163' for c in text):
            return "KR"
        return "INT"  # 국제/영문 (UK, EU 등)

    def _reassemble_plate(self, ocr_entries: list[tuple[str, list, float]]) -> tuple[str, float]:
        """
        OCR 결과(텍스트+bbox+confidence)를 번호판 패턴으로 재조합 (개선4)

        bbox y중심 기준 정렬 → 패턴 매칭 → 한글/숫자 교정
        """
        if not ocr_entries:
            return '', 0.0

        # bbox y중심 기준 정렬
        entries_with_y = []
        for text, bbox, conf in ocr_entries:
            y_center = 0.0
            if bbox is not None:
                try:
                    if isinstance(bbox[0], (list, tuple)):
                        y_center = sum(p[1] for p in bbox) / len(bbox)
                    elif len(bbox) >= 4:
                        y_center = (bbox[1] + bbox[3]) / 2
                except (IndexError, TypeError):
                    pass
            entries_with_y.append((text, bbox, conf, y_center))

        entries_with_y.sort(key=lambda x: x[3])

        all_text = ''.join(e[0] for e in entries_with_y)
        avg_conf = sum(e[2] for e in entries_with_y) / len(entries_with_y) if entries_with_y else 0

        # 특수문자 제거 + 영문→숫자 교정
        cleaned = re.sub(r'[^가-힣0-9a-zA-Z]', '', all_text)
        cleaned = ''.join(_DIGIT_CORRECTION.get(c, c) for c in cleaned)

        # 패턴A: 지역명+숫자2~3+한글1+숫자4 (구형: 경기37바5577)
        m = re.search(r'([가-힣]{2})(\d{2,3})([가-힣])(\d{4})', cleaned)
        if m:
            region, nf, usage, nb = m.groups()
            region = _correct_region(region)
            usage = _correct_single_hangul(usage)
            return region + nf + usage + nb, avg_conf

        # 패턴B: 숫자2~3+한글1+숫자4 (신형 or 지역명 누락)
        m = re.search(r'(\d{2,3})([가-힣])(\d{4})', cleaned)
        if m:
            nf, usage, nb = m.groups()
            usage = _correct_single_hangul(usage)
            prefix = cleaned[:m.start()]
            region = _find_region_in_text(prefix)
            if region:
                return region + nf + usage + nb, avg_conf
            return nf + usage + nb, avg_conf

        # 패턴C: 2줄 번호판 상하 분리
        if len(entries_with_y) >= 2:
            y_values = [e[3] for e in entries_with_y]
            y_mid = (min(y_values) + max(y_values)) / 2
            upper = ''.join(re.sub(r'[^가-힣0-9a-zA-Z]', '', e[0]) for e in entries_with_y if e[3] < y_mid)
            lower = ''.join(re.sub(r'[^가-힣0-9a-zA-Z]', '', e[0]) for e in entries_with_y if e[3] >= y_mid)
            upper = ''.join(_DIGIT_CORRECTION.get(c, c) for c in upper)
            lower = ''.join(_DIGIT_CORRECTION.get(c, c) for c in lower)

            lower_nums = re.findall(r'\d{4}', lower)
            upper_nums = re.findall(r'\d{2,3}', upper)
            upper_hangul = re.findall(r'[가-힣]', upper)

            if lower_nums and upper_nums and upper_hangul:
                usage = _correct_single_hangul(upper_hangul[-1])
                region = _find_region_in_text(upper)
                base = upper_nums[0] + usage + lower_nums[0]
                if region:
                    base = region + base
                return base, avg_conf

        # 패턴D: 조각 모아 재조합
        all_nums = re.findall(r'\d+', cleaned)
        all_hangul = re.findall(r'[가-힣]', cleaned)
        if all_nums and all_hangul:
            d4 = [n for n in all_nums if len(n) == 4]
            d23 = [n for n in all_nums if len(n) in (2, 3)]
            if d4 and d23:
                usage = _correct_single_hangul(all_hangul[-1] if len(all_hangul) == 1 else all_hangul[0])
                return d23[0] + usage + d4[0], avg_conf

        return cleaned, avg_conf

    def _run_paddle_ocr(self, img: np.ndarray) -> list[tuple[str, list, float]]:
        """PaddleOCR 실행 → [(text, bbox, confidence), ...]"""
        if self.paddle_reader is None:
            return []
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            result = self.paddle_reader.ocr(img, cls=True)
            entries = []
            if result and result[0]:
                for item in result[0]:
                    try:
                        text = item[1][0].strip()
                        conf = float(item[1][1])
                        bbox = item[0]
                        if text:
                            entries.append((text, bbox, conf))
                    except (IndexError, TypeError):
                        continue
            return entries
        except Exception:
            return []

    def _run_easy_ocr(self, img: np.ndarray) -> list[tuple[str, list, float]]:
        """EasyOCR 실행 → [(text, bbox, confidence), ...]"""
        if self._ko_reader is None:
            return []
        try:
            result = self._ko_reader.readtext(
                img, detail=1, paragraph=False,
                text_threshold=0.3, low_text=0.3,
            )
            entries = []
            if result:
                for bbox, text, conf in result:
                    text = text.strip()
                    if text:
                        entries.append((text, bbox, conf))
            return entries
        except Exception:
            return []

    def _postprocess_ocr_text(self, text: str, conf: float) -> tuple[str, float, bool, float]:
        """OCR 텍스트 후처리: 한글 교정 + 패턴 검증 + 형식 교정"""
        if not text or len(text) < 4:
            return text, conf, False, 0.0
        cleaned = re.sub(r"[^가-힣0-9A-Za-z]", "", text)
        corrected = correct_ocr_hangul(cleaned)
        corrected = correct_hangul_similarity(corrected)
        is_valid, normalized, score = validate_korean_plate(corrected)
        if normalized and len(normalized) >= 4:
            fmt_corrected, fmt_score = validate_plate_format(normalized)
            if fmt_score > 0:
                normalized = fmt_corrected
                score = max(score, fmt_score)
                is_valid = True
            has_hangul = any('\uac00' <= c <= '\ud7a3' for c in normalized)
            if has_hangul:
                score = max(score, 0.90)
            return normalized, conf, is_valid or has_hangul, score
        return text, conf, False, 0.0

    def _ocr_korean_plate(self, plate_img: np.ndarray) -> tuple[str, float, bool, float]:
        """
        한국 번호판 OCR (개선3+4: 다중 전처리 앙상블 + bbox 재조합)

        전략:
        1. PaddleOCR (원본) → bbox 기반 재조합 → 후처리
        2. PaddleOCR (강화 전처리) → bbox 기반 재조합 → 후처리
        3. EasyOCR (원본) → bbox 기반 재조합 → 후처리
        4. EasyOCR (강화 전처리) → bbox 기반 재조합 → 후처리
        5. 유효 패턴 우선 + 신뢰도 가중 투표

        Returns:
            (text, ocr_confidence, is_valid_plate, pattern_score)
        """
        candidates: list[tuple[str, float, bool, float]] = []

        # 원본 컬러 이미지 준비
        if len(plate_img.shape) == 2:
            ocr_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)
        else:
            ocr_img = plate_img

        # ── Tier1: PaddleOCR 원본만 먼저 (conf≥0.8이면 clahe/sharpen 스킵) ──
        _tier1_high_conf = False
        entries = self._run_paddle_ocr(ocr_img)
        if entries:
            text, conf = self._reassemble_plate(entries)
            result = self._postprocess_ocr_text(text, conf)
            if result[0] and len(result[0]) >= 4:
                has_hangul = any('\uac00' <= c <= '\ud7a3' for c in result[0])
                score = result[3]
                if has_hangul and conf >= 0.7:
                    score = max(score, 1.05)
                elif has_hangul:
                    score = max(score, 0.95)
                candidates.append((result[0], result[1], result[2], score))
                # conf ≥ 0.8 → 강화 전처리(clahe/sharpen) 및 EasyOCR 스킵
                if conf >= 0.8:
                    _tier1_high_conf = True

        if not _tier1_high_conf:
            # 강화 전처리 이미지 (개선3: 다중 전처리) — Tier1 통과 못한 경우만
            enhanced_img = self._preprocess_plate_enhanced(plate_img)

            # ── 2. PaddleOCR 강화 전처리 ──
            entries = self._run_paddle_ocr(enhanced_img)
            if entries:
                text, conf = self._reassemble_plate(entries)
                result = self._postprocess_ocr_text(text, conf)
                if result[0] and len(result[0]) >= 4:
                    candidates.append((result[0], result[1] * 0.95, result[2], result[3] * 0.98))

            # ── 3. EasyOCR 원본 (보조) ──
            entries = self._run_easy_ocr(ocr_img)
            if entries:
                text, conf = self._reassemble_plate(entries)
                result = self._postprocess_ocr_text(text, conf)
                if result[0] and len(result[0]) >= 4:
                    candidates.append((result[0], result[1], result[2], result[3]))

            # ── 4. EasyOCR 강화 전처리 ──
            entries = self._run_easy_ocr(enhanced_img)
            if entries:
                text, conf = self._reassemble_plate(entries)
                result = self._postprocess_ocr_text(text, conf)
                if result[0] and len(result[0]) >= 4:
                    candidates.append((result[0], result[1] * 0.95, result[2], result[3] * 0.98))

        # ── 5. 소프트 전처리 PaddleOCR 재시도 (모두 실패 시) ──
        if not candidates and self.paddle_reader is not None:
            try:
                soft = self._preprocess_plate_soft(plate_img)
                soft_bgr = cv2.cvtColor(soft, cv2.COLOR_GRAY2BGR)
                entries = self._run_paddle_ocr(soft_bgr)
                if entries:
                    text, conf = self._reassemble_plate(entries)
                    result = self._postprocess_ocr_text(text, conf)
                    if result[0] and len(result[0]) >= 4:
                        candidates.append((result[0], result[1] * 0.90, result[2], result[3] * 0.90))
            except Exception:
                pass

        if not candidates:
            return "", 0.0, False, 0.0

        # 앙상블: 최고 패턴 점수 선택 (다수결 포함)
        if len(candidates) >= 2:
            return self._ensemble_vote(candidates)
        return candidates[0]

    def _ocr_english_plate(self, plate_img: np.ndarray) -> tuple[str, float, bool, float]:
        """
        영문/국제 번호판 OCR (EasyOCR English, UK 패턴 보정)
        """
        if self._en_reader is None:
            return "", 0.0, False, 0.0

        try:
            if len(plate_img.shape) == 2:
                ocr_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)
            else:
                ocr_img = plate_img

            result = self._en_reader.readtext(
                ocr_img, detail=1, paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            )
            if not result:
                return "", 0.0, False, 0.0

            result.sort(key=lambda r: r[0][0][0])
            texts = [r[1].strip() for r in result if r[1].strip()]
            confs = [r[2] for r in result if r[1].strip()]
            combined = "".join(texts)
            avg_conf = sum(confs) / len(confs) if confs else 0.0

            if len(combined) < 2:
                return "", 0.0, False, 0.0

            cleaned = self._clean_en_text(combined)
            if len(cleaned) < 2:
                return "", 0.0, False, 0.0

            corrected = correct_ocr_uk(cleaned)
            is_valid, normalized, pattern_score = validate_korean_plate(corrected)
            if not is_valid:
                is_valid = bool(self._RE_UK_PLATE.match(corrected))
                pattern_score = 0.90 if is_valid else 0.40

            return corrected, avg_conf, is_valid, pattern_score

        except Exception:
            return "", 0.0, False, 0.0

    def _ocr_with_validation(self, plate_img: np.ndarray) -> tuple[str, float, bool, float]:
        """
        다국어 번호판 OCR + 검증 (한국 우선, 영문 폴백)

        전략:
        1. 한국 번호판 OCR (PaddleOCR + EasyOCR Korean 앙상블)
        2. 영문/국제 번호판 OCR (EasyOCR English)
        3. 최고 패턴 점수 결과 반환

        Returns:
            (text, ocr_confidence, is_valid_plate, pattern_score)
        """
        candidates: list[tuple[str, float, bool, float]] = []

        # 1. 한국 번호판 OCR (최우선)
        kr_result = self._ocr_korean_plate(plate_img)
        if kr_result[0] and len(kr_result[0]) >= 4:
            candidates.append(kr_result)

        # 2. 영문/국제 번호판 OCR (폴백)
        en_result = self._ocr_english_plate(plate_img)
        if en_result[0] and len(en_result[0]) >= 4:
            candidates.append(en_result)

        if not candidates:
            # 짧은 텍스트라도 반환 (부분 인식)
            kr_short = self._ocr_korean_plate(plate_img)
            en_short = self._ocr_english_plate(plate_img)
            for r in [kr_short, en_short]:
                if r[0]:
                    candidates.append(r)

        if not candidates:
            return "", 0.0, False, 0.0

        # 한국 패턴 매칭 결과 우선
        kr_valid = [c for c in candidates if c[2] and self._has_korean(c[0])]
        if kr_valid:
            return max(kr_valid, key=lambda c: c[3])

        # 그 외 최고 점수 반환
        return max(candidates, key=lambda c: c[3])

    def _ensemble_vote(
        self, candidates: list[tuple[str, float, bool, float]]
    ) -> tuple[str, float, bool, float]:
        """
        OCR 앙상블 투표 로직.
        - 정규화된 텍스트 기준으로 투표 집계
        - 2개 이상 동일 결과 → 다수결 채택 + 신뢰도 보너스
        - 모두 다르면 최고 score 채택
        """
        from collections import Counter

        # 텍스트 정규화 (공백/특수문자 제거, 대문자 통일)
        def _norm(t: str) -> str:
            return re.sub(r"[^가-힣0-9A-Z]", "", t.upper())

        # 투표 집계
        vote_map: dict[str, list[tuple[str, float, bool, float]]] = {}
        for cand in candidates:
            key = _norm(cand[0])
            if not key:
                continue
            if key not in vote_map:
                vote_map[key] = []
            vote_map[key].append(cand)

        if not vote_map:
            return max(candidates, key=lambda c: c[3])

        # 다수결: 2표 이상이면 채택 + 신뢰도 보너스
        multi_vote = [(k, v) for k, v in vote_map.items() if len(v) >= 2]
        if multi_vote:
            # 가장 많은 표를 받은 것, 같으면 최고 score
            multi_vote.sort(key=lambda kv: (len(kv[1]), max(c[3] for c in kv[1])), reverse=True)
            best_group = multi_vote[0][1]
            # 그룹 내 최고 score 후보 선택
            best = max(best_group, key=lambda c: c[3])
            # 다수결 보너스: 투표 수에 비례 (2표=+0.05, 3표=+0.08, 4표+=0.10)
            vote_bonus = min(len(best_group) * 0.03, 0.10)
            # 가중 평균 신뢰도
            avg_conf = sum(c[1] for c in best_group) / len(best_group)
            boosted_score = best[3] + vote_bonus
            return best[0], avg_conf, best[2], boosted_score

        # 모두 다른 경우: 최고 score 채택
        return max(candidates, key=lambda c: c[3])

    # ── 선명도 측정 ──────────────────────────────────────

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Levenshtein 편집 거리 계산 (개선5)"""
        if len(s1) < len(s2):
            return PlateRecognizer._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                curr_row.append(min(
                    curr_row[j] + 1,        # insert
                    prev_row[j + 1] + 1,    # delete
                    prev_row[j] + cost,      # replace
                ))
            prev_row = curr_row
        return prev_row[-1]

    def _track_plate(self, text: str, frame_idx: int, result: dict) -> None:
        """
        번호판 추적 + 시간축 앙상블 (개선5)

        - 정확히 같은 키 → 기존 로직 (카운트 증가)
        - Levenshtein ≤ 2인 유사 키 → 같은 번호판으로 그룹핑
        - 5프레임 슬라이딩 윈도우 내 최고 confidence 결과 채택
        """
        key = re.sub(r"[^가-힣0-9A-Z]", "", text.upper())
        if not key or len(key) < 2:
            return

        safe_result = {k: v for k, v in result.items() if k not in ("plate_img", "preprocessed")}

        # 시간축 앙상블: Levenshtein ≤ 2인 기존 키 찾기 (개선5)
        matched_key = None
        if key in self._plate_tracker:
            matched_key = key
        else:
            for existing_key in self._plate_tracker:
                if abs(len(existing_key) - len(key)) <= TEMPORAL_LEVENSHTEIN_MAX:
                    if self._levenshtein(key, existing_key) <= TEMPORAL_LEVENSHTEIN_MAX:
                        matched_key = existing_key
                        break

        if matched_key is not None:
            tracker = self._plate_tracker[matched_key]
            tracker["count"] += 1
            tracker["last_frame"] = frame_idx

            # 슬라이딩 윈도우에 현재 결과 추가 (개선5)
            if "window" not in tracker:
                tracker["window"] = []
            tracker["window"].append({
                "text": key,
                "score": result.get("pattern_score", 0),
                "conf": result.get("ocr_confidence", 0),
                "frame": frame_idx,
                "result": safe_result,
            })
            # 윈도우 크기 제한
            if len(tracker["window"]) > TEMPORAL_WINDOW:
                tracker["window"] = tracker["window"][-TEMPORAL_WINDOW:]

            # 윈도우 내 최고 score 결과로 best_result 갱신 (개선5)
            best_in_window = max(tracker["window"], key=lambda w: w["score"] * 0.6 + w["conf"] * 0.4)
            if best_in_window["score"] >= tracker["best_result"].get("pattern_score", 0):
                tracker["best_result"] = best_in_window["result"]
                # 윈도우에서 최빈 텍스트로 교정 (다수결)
                from collections import Counter
                text_counts = Counter(w["text"] for w in tracker["window"])
                majority_text, majority_count = text_counts.most_common(1)[0]
                if majority_count >= 2 and majority_text != key:
                    # 다수결 텍스트가 더 신뢰할 만하면 채택
                    tracker["best_result"]["text"] = majority_text

            # CONFIRM_FRAME_COUNT 도달 시 confirmed 승격
            if tracker["count"] >= CONFIRM_FRAME_COUNT and not tracker.get("confirmed"):
                tracker["confirmed"] = True
                confirmed_entry = dict(tracker["best_result"])
                confirmed_entry["first_frame"] = tracker["first_frame"]
                confirmed_entry["last_frame"] = frame_idx
                confirmed_entry["detection_count"] = tracker["count"]
                self._confirmed_plates.append(confirmed_entry)
        else:
            self._plate_tracker[key] = {
                "count": 1,
                "first_frame": frame_idx,
                "last_frame": frame_idx,
                "confirmed": False,
                "best_result": safe_result,
                "window": [{
                    "text": key,
                    "score": result.get("pattern_score", 0),
                    "conf": result.get("ocr_confidence", 0),
                    "frame": frame_idx,
                    "result": safe_result,
                }],
            }

    def get_confirmed_plates(self) -> list[dict]:
        """확정된 번호판 목록 반환 (편집거리 1 이하 중복 제거)."""
        deduped: list[dict] = []
        for plate in self._confirmed_plates:
            text = plate.get("text", "")
            merged = False
            for existing in deduped:
                ex_text = existing.get("text", "")
                if abs(len(text) - len(ex_text)) <= 1 and self._levenshtein(text, ex_text) <= 1:
                    # 더 높은 score 유지
                    if plate.get("pattern_score", 0) > existing.get("pattern_score", 0):
                        existing.update(plate)
                    existing["detection_count"] = existing.get("detection_count", 0) + plate.get("detection_count", 0)
                    merged = True
                    break
            if not merged:
                deduped.append(dict(plate))
        return deduped

    @staticmethod
    def _calculate_sharpness(image: np.ndarray) -> float:
        """Laplacian 기반 이미지 선명도 측정"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _is_gpu_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ── bbox 기반 OCR 스킵 캐시 ─────────────────────────

    @staticmethod
    def _bbox_iou(a: list[float], b: list[float]) -> float:
        """두 bbox의 IoU(Intersection over Union) 계산."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _find_bbox_cache(self, xyxy: list[float], frame_idx: int) -> dict | None:
        """최근 캐시에서 동일 위치(IoU>0.5) 번호판을 찾아 반환. 5프레임 이내만 (OCR 캐시)."""
        for entry in reversed(self._bbox_cache):
            if frame_idx - entry["frame_idx"] > 5:
                continue
            if self._bbox_iou(xyxy, entry["xyxy"]) > 0.5:
                return entry
        return None

    def _update_bbox_cache(self, xyxy: list[float], frame_idx: int, result: dict) -> None:
        """bbox 캐시에 결과 추가. 오래된 항목 자동 제거."""
        self._bbox_cache.append({
            "xyxy": xyxy,
            "frame_idx": frame_idx,
            "text": result.get("text", ""),
            "ocr_confidence": result.get("ocr_confidence", 0),
            "is_valid_plate": result.get("is_valid_plate", False),
            "pattern_score": result.get("pattern_score", 0),
        })
        # 오래된 캐시 제거
        if len(self._bbox_cache) > self._bbox_cache_max:
            self._bbox_cache = self._bbox_cache[-self._bbox_cache_max:]

    # ── Detection Log OCR (화면 내 텍스트 번호판) ──────────

    _LOG_PLATE_RE = re.compile(r'\d{2,3}[가-힣]\d{3,4}')

    def _ocr_log_region(self, frame: np.ndarray, frame_idx: int) -> list[dict]:
        """
        화면의 Detection Log 영역에서 번호판 텍스트를 OCR로 추출.
        N프레임마다 실행하여 속도 영향 최소화.

        주의: 이 기능은 타 LPR 시스템 화면 녹화 영상에서
        Detection Log 텍스트를 읽는 보조 기능.
        순수 CCTV 영상에서는 불필요하므로 비활성화.
        """
        # log_ocr 비활성화: 외부 프로그램 화면 인식 방지
        return []
        if frame_idx - self._last_log_ocr_frame < LOG_OCR_INTERVAL:
            return []
        if self.paddle_reader is None:
            return []

        # 이미 충분한 log 번호판을 찾았으면 주기를 늘림
        if len(self._log_ocr_cache) >= 5:
            if frame_idx - self._last_log_ocr_frame < LOG_OCR_INTERVAL * 3:
                return []

        self._last_log_ocr_frame = frame_idx
        h, w = frame.shape[:2]

        # Detection Log 영역 추정 (화면 우측 중하단)
        # 다양한 해상도에 대응: 우측 50%, 상단 25%~하단 60%
        log_x1 = int(w * 0.45)
        log_y1 = int(h * 0.25)
        log_x2 = w
        log_y2 = int(h * 0.62)
        log_region = frame[log_y1:log_y2, log_x1:log_x2]

        if log_region.size == 0:
            return []

        results = []
        try:
            ocr_result = self.paddle_reader.ocr(log_region, cls=True)
            if ocr_result and ocr_result[0]:
                for item in ocr_result[0]:
                    try:
                        text = item[1][0].strip()
                        conf = float(item[1][1])
                        if conf < 0.8:
                            continue
                        # 한글 유사 보정 적용
                        text = correct_hangul_similarity(text)
                        # 번호판 패턴 추출
                        for m in self._LOG_PLATE_RE.finditer(text):
                            plate = m.group()
                            plate = correct_ocr_hangul(plate)
                            plate = correct_hangul_similarity(plate)
                            if plate not in self._log_ocr_cache:
                                self._log_ocr_cache[plate] = frame_idx
                            results.append({
                                "frame_idx": frame_idx,
                                "text": plate,
                                "ocr_confidence": conf,
                                "detection_confidence": conf,
                                "bbox": [log_x1, log_y1, log_x2, log_y2],
                                "is_valid_plate": True,
                                "pattern_score": 0.90,
                                "detection_method": "log_ocr",
                                "sharpness": 999.0,
                                "preprocessed": log_region,
                                "plate_img": log_region,
                            })
                    except (IndexError, TypeError):
                        continue
        except Exception:
            pass

        return results

    # ── 프레임 처리 (핵심) ────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> list[dict]:
        """
        단일 프레임 처리 (듀얼 탐지 전략)

        1차: 번호판 전용 모델 → 직접 크롭 → OCR
        2차: 1차 실패 시 COCO 차량 탐지 → 번호판 추출 → OCR
        """
        try:
            return self._process_frame_inner(frame, frame_idx)
        except Exception as e:
            print(f"  [오류] 프레임 {frame_idx} 처리 실패: {e}")
            return []

    def _process_frame_inner(self, frame: np.ndarray, frame_idx: int) -> list[dict]:
        """process_frame 내부 구현."""
        # 1차 탐지: 메인 모델
        detections = self.detect_plates(frame)
        results: list[dict] = []
        fh, fw = frame.shape[:2]

        if self._is_plate_model:
            # === 번호판 전용 모델: 직접 크롭 후 OCR ===
            for det in detections:
                plate_img = self._crop_region(
                    frame, det["xyxy"],
                    padding_left=PLATE_MODEL_PADDING_H, padding_right=PLATE_MODEL_PADDING_H,
                    padding_top=PLATE_MODEL_PADDING_V, padding_bottom=PLATE_MODEL_PADDING_V,
                )
                if plate_img.size == 0 or plate_img.shape[0] < 5 or plate_img.shape[1] < 10:
                    continue

                # 개선1: 크롭 품질 필터 (aspect ratio + min size)
                ph, pw = plate_img.shape[:2]
                if pw < MIN_PLATE_WIDTH or ph < MIN_PLATE_HEIGHT:
                    continue
                aspect = pw / ph if ph > 0 else 0
                if aspect < PLATE_MIN_ASPECT or aspect > PLATE_MAX_ASPECT:
                    continue

                # 개선2: 업스케일 + 기울기 보정
                plate_img = self._upscale_if_small(plate_img)
                plate_img = self._deskew_plate(plate_img)

                # 속도 최적화: 동일 위치 bbox에 이미 확정된 번호판이 있으면 OCR 스킵
                cached = self._find_bbox_cache(det["xyxy"], frame_idx)
                if cached is not None:
                    text, ocr_conf = cached["text"], cached["ocr_confidence"]
                    is_valid, pattern_score = cached["is_valid_plate"], cached["pattern_score"]
                else:
                    text, ocr_conf, is_valid, pattern_score = self._ocr_with_validation(plate_img)

                # 부분 인식 재시도: [한글]\d{3} 패턴 → 확대 크롭으로 접두 숫자 복구
                # 예: "아447" → 확대 크롭 → "85아447"
                if text and re.match(r'^[가-힣]\d{3,4}$', text):
                    partial_key = re.sub(r"[^가-힣0-9A-Z]", "", text.upper())
                    # 캐시 확인: 이전에 해결된 부분→전체 매핑 사용 (OCR 재시도 없이)
                    if not hasattr(self, '_partial_cache'):
                        self._partial_cache: dict[str, tuple] = {}
                    if partial_key in self._partial_cache:
                        text, ocr_conf, is_valid, pattern_score = self._partial_cache[partial_key]
                    else:
                        expanded = self._crop_region(
                            frame, det["xyxy"],
                            padding_ratio=0.10, padding_left=0.20, padding_top=0.25,
                        )
                        if expanded.size > 0:
                            t2, c2, v2, s2 = self._ocr_with_validation(expanded)
                            if len(t2) > len(text) and any('\uac00' <= c <= '\ud7a3' for c in t2):
                                text, ocr_conf, is_valid, pattern_score = t2, c2, v2, s2
                                self._partial_cache[partial_key] = (t2, c2, v2, s2)

                # 텍스트 길이 필터: 번호판은 최대 MAX_PLATE_TEXT_LEN자
                if len(text) > MAX_PLATE_TEXT_LEN:
                    continue

                result_entry = {
                    "frame_idx": frame_idx,
                    "plate_img": plate_img,
                    "text": text,
                    "ocr_confidence": ocr_conf,
                    "detection_confidence": det["confidence"],
                    "bbox": det["xyxy"],
                    "is_valid_plate": is_valid,
                    "pattern_score": pattern_score,
                    "detection_method": "plate_model",
                    "sharpness": self._calculate_sharpness(plate_img),
                    "preprocessed": plate_img,
                }
                results.append(result_entry)

                # bbox 캐시 업데이트 (다음 프레임에서 OCR 스킵용)
                if text and len(text) >= 4 and is_valid:
                    self._update_bbox_cache(det["xyxy"], frame_idx, result_entry)

            # 2차 폴백: 번호판 모델 탐지 0건 → COCO 차량 탐지
            if not results and self._is_plate_model:
                coco_detections = self._detect_coco_fallback(frame)
                for det in coco_detections:
                    vehicle_img = self._crop_region(frame, det["xyxy"], padding_ratio=0.05)
                    if vehicle_img.size == 0:
                        continue

                    plate_candidates = self._extract_plate_from_vehicle(vehicle_img)
                    for plate_img in plate_candidates:
                        if plate_img.size == 0:
                            continue

                        text, ocr_conf, is_valid, pattern_score = self._ocr_with_validation(plate_img)

                        results.append({
                            "frame_idx": frame_idx,
                            "plate_img": plate_img,
                            "text": text,
                            "ocr_confidence": ocr_conf,
                            "detection_confidence": det["confidence"],
                            "bbox": det["xyxy"],
                            "is_valid_plate": is_valid,
                            "pattern_score": pattern_score,
                            "detection_method": "coco_fallback",
                            "sharpness": self._calculate_sharpness(plate_img),
                            "preprocessed": plate_img,
                        })
        else:
            # === yolo11n.pt 차량 탐지 → 하단 35% 크롭 → OCR (2단계 전략) ===
            for det in detections:
                vehicle_img = self._crop_region(frame, det["xyxy"], padding_ratio=0.05)
                if vehicle_img.size == 0 or vehicle_img.shape[0] < 30:
                    continue

                # 차량 하단 35% 크롭 (번호판 위치)
                vh, vw = vehicle_img.shape[:2]
                plate_img = vehicle_img[int(vh * 0.65):, :]
                if plate_img.size == 0:
                    continue

                text, ocr_conf, is_valid, pattern_score = self._ocr_with_validation(plate_img)

                results.append({
                    "frame_idx": frame_idx,
                    "plate_img": plate_img,
                    "text": text,
                    "ocr_confidence": ocr_conf,
                    "detection_confidence": det["confidence"],
                    "bbox": det["xyxy"],
                    "is_valid_plate": is_valid,
                    "pattern_score": pattern_score,
                    "detection_method": "yolo11n_bottom35",
                    "sharpness": self._calculate_sharpness(plate_img),
                    "preprocessed": plate_img,
                })

        # Detection Log OCR: 화면 내 텍스트 번호판 보조 인식
        log_plates = self._ocr_log_region(frame, frame_idx)
        if log_plates:
            results.extend(log_plates)

        # 신뢰도 필터: OCR ≥ MIN_OCR_CONFIDENCE, 탐지 ≥ MIN_DET_CONFIDENCE
        filtered = []
        for r in results:
            text = r.get("text", "")
            if not text or len(text) < 2:
                continue
            # log_ocr 결과는 신뢰도 필터 면제 (이미 0.8+ 필터링됨)
            if r.get("detection_method") != "log_ocr":
                if r["ocr_confidence"] < MIN_OCR_CONFIDENCE:
                    continue
                if r["detection_confidence"] < MIN_DET_CONFIDENCE:
                    continue
            # ROI 필터: bbox 중심이 ROI 내부인지 확인
            bbox = r.get("bbox", None)
            if bbox is not None and not self._bbox_center_in_roi(bbox, fw, fh):
                continue
            filtered.append(r)

        # 추적: CONFIRM_FRAME_COUNT 연속 인식 시 confirmed 승격
        for r in filtered:
            self._track_plate(r.get("text", ""), frame_idx, r)

        return filtered

    # ── 상태 머신 ─────────────────────────────────────────

    def _reset_state_machine(self) -> None:
        self.state = CaptureState.SCANNING
        self.burst_counter = 0
        self.no_detect_count = 0
        self._imgsz = None
        self._use_sahi_for_video = False

    # ── 비디오 전체 처리 ──────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        progress_callback=None,
    ) -> list[dict]:
        """
        비디오 전체 처리 (상태 머신 기반)

        SCANNING → TRACKING → CAPTURING → 최고 선명도 선택
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"비디오를 열 수 없습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n{'='*60}")
        print(f"  [비디오 정보]")
        print(f"  해상도: {width}x{height}")
        print(f"  FPS: {fps:.1f}")
        print(f"  총 프레임: {total_frames}")
        if fps > 0:
            print(f"  총 길이: {total_frames / fps:.1f}초")
        print(f"{'='*60}\n")

        self._reset_state_machine()

        best_results: list[dict] = []
        burst_results: list[dict] = []

        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            should_process = False
            if self.state == CaptureState.SCANNING:
                should_process = (frame_idx % self.frame_skip == 0)
            elif self.state in (CaptureState.TRACKING, CaptureState.CAPTURING):
                should_process = True

            if should_process:
                frame_results = self.process_frame(frame, frame_idx)

                if frame_results:
                    self.no_detect_count = 0
                    burst_results.extend(frame_results)

                    if self.state == CaptureState.SCANNING:
                        self.state = CaptureState.TRACKING
                        self.burst_counter = 0
                        print(f"  프레임 {frame_idx}: 번호판 탐지! → TRACKING")

                    elif self.state == CaptureState.TRACKING:
                        self.state = CaptureState.CAPTURING
                        print(f"  프레임 {frame_idx}: → CAPTURING (버스트 시작)")

                    if self.state == CaptureState.CAPTURING:
                        self.burst_counter += 1
                else:
                    self.no_detect_count += 1

                if self.state == CaptureState.CAPTURING:
                    burst_done = self.burst_counter >= self.burst_frames
                    lost = self.no_detect_count >= NO_DETECT_TOLERANCE

                    if burst_done or lost:
                        if burst_results:
                            # 종합 점수 기반 최선 선택 (선명도 + OCR + 패턴)
                            best = max(
                                burst_results,
                                key=lambda r: (
                                    r["sharpness"] * 0.3 +
                                    r["ocr_confidence"] * 100 * 0.3 +
                                    r["pattern_score"] * 100 * 0.4
                                ),
                            )
                            best_results.append(best)
                            reason = "버스트 소진" if burst_done else "타겟 소실"
                            method = best.get("detection_method", "unknown")
                            valid_mark = " [유효]" if best.get("is_valid_plate") else ""
                            print(
                                f"  버스트 완료 ({reason}): "
                                f"{len(burst_results)}건 → "
                                f"\"{best['text']}\"{valid_mark} "
                                f"(선명도={best['sharpness']:.1f}, "
                                f"OCR={best['ocr_confidence']:.2f}, "
                                f"방식={method})"
                            )

                        burst_results = []
                        self.state = CaptureState.SCANNING
                        self.burst_counter = 0
                        self.no_detect_count = 0

            frame_idx += 1

            if frame_idx % 200 == 0:
                pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                elapsed = time.time() - start_time
                remaining = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0
                print(
                    f"  진행: {pct:.1f}% ({frame_idx}/{total_frames}) "
                    f"| {elapsed:.0f}초 경과 | 잔여: {remaining:.0f}초 "
                    f"| {self.state.name} | 인식: {len(best_results)}건"
                )
                if progress_callback:
                    progress_callback(frame_idx, len(best_results))

        cap.release()

        # 마지막 버스트 처리
        if burst_results:
            best = max(
                burst_results,
                key=lambda r: r["sharpness"] * 0.3 + r["ocr_confidence"] * 100 * 0.3 + r["pattern_score"] * 100 * 0.4,
            )
            best_results.append(best)

        total_elapsed = time.time() - start_time
        print(f"\n  [완료] {frame_idx}프레임 처리, {len(best_results)}개 번호판 인식")
        print(f"  [완료] 처리 시간: {total_elapsed:.1f}초")

        # 한국 번호판 OCR 후처리 (오탐 제거, 패턴·한글 검증, 중복 병합, 유사 번호 교차검증)
        try:
            from plate_ocr_postfilter import apply_postfilter
            n_before = len(best_results)
            best_results = apply_postfilter(best_results)
            if n_before != len(best_results):
                print(f"  [후처리] {n_before}건 → {len(best_results)}건 정제 (plate_ocr_postfilter)")
        except Exception as e:
            print(f"  [후처리] 스킵: {e}")

        if output_dir and best_results:
            self._save_results(best_results, output_dir)

        return best_results

    # ── 결과 저장 ────────────────────────────────────────

    def _save_results(self, results: list[dict], output_dir: str) -> None:
        """탐지 결과를 이미지 + JSON으로 저장"""
        os.makedirs(output_dir, exist_ok=True)

        summary: list[dict] = []
        for i, result in enumerate(results):
            img_path = os.path.join(output_dir, f"plate_{i:04d}.png")
            cv2.imwrite(img_path, result["plate_img"])

            pre_path = os.path.join(output_dir, f"plate_{i:04d}_preprocessed.png")
            cv2.imwrite(pre_path, result["preprocessed"])

            summary.append({
                "index": i,
                "frame_idx": result["frame_idx"],
                "text": result["text"],
                "ocr_confidence": round(result["ocr_confidence"], 4),
                "detection_confidence": round(result["detection_confidence"], 4),
                "sharpness": round(result["sharpness"], 2),
                "bbox": [round(v, 1) for v in result["bbox"]],
                "is_valid_plate": result.get("is_valid_plate", False),
                "pattern_score": round(result.get("pattern_score", 0.0), 4),
                "detection_method": result.get("detection_method", "unknown"),
                "image_path": img_path,
                "preprocessed_path": pre_path,
                "detection_count": result.get("detection_count", 1),
                "final_confidence": round(result.get("final_confidence", result["ocr_confidence"]), 4),
            })

        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        print(f"\n  [저장] {len(results)}개 번호판 → {output_dir}")
        print(f"  [저장] 요약 JSON: {json_path}")
        for s in summary:
            valid_mark = " [유효]" if s["is_valid_plate"] else ""
            print(
                f"    #{s['index']}: \"{s['text']}\"{valid_mark} "
                f"(OCR={s['ocr_confidence']:.2f}, "
                f"탐지={s['detection_confidence']:.2f}, "
                f"패턴={s['pattern_score']:.2f}, "
                f"방식={s['detection_method']})"
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    parser = argparse.ArgumentParser(
        description="4K 영상 번호판 인식 v2.0 (번호판 전용 YOLO + EasyOCR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python plate_recognition_4k.py dashcam.mp4
  python plate_recognition_4k.py dashcam.mp4 -o ./results --model-size s
  python plate_recognition_4k.py dashcam.mp4 --no-sahi --frame-skip 3
  python plate_recognition_4k.py dashcam.mp4 --model my_custom.pt
        """,
    )
    parser.add_argument("video", help="입력 비디오 파일 경로")
    parser.add_argument("--output", "-o", default="./plate_results", help="결과 출력 디렉토리")
    parser.add_argument(
        "--model", default=None,
        help="사용자 지정 YOLO 모델 .pt 경로 (미지정 시 HuggingFace 자동 다운로드)",
    )
    parser.add_argument(
        "--model-size", default=DEFAULT_PLATE_MODEL_SIZE,
        choices=list(HF_MODEL_VARIANTS.keys()),
        help=f"번호판 모델 크기 (기본: {DEFAULT_PLATE_MODEL_SIZE})",
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="최소 탐지 신뢰도")
    parser.add_argument("--no-sahi", action="store_true", help="SAHI 타일링 비활성화")
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP, help="SCANNING 프레임 스킵 간격")
    parser.add_argument("--burst-frames", type=int, default=BURST_FRAME_COUNT, help="CAPTURING 버스트 프레임 수")

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"[오류] 비디오 파일을 찾을 수 없습니다: {args.video}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  4K 번호판 인식 v2.0")
    print("  번호판 전용 YOLO + SAHI + EasyOCR + 한국 번호판 검증")
    print("=" * 60)
    print(f"  입력: {args.video}")
    print(f"  출력: {args.output}")
    print(f"  모델: {args.model or f'HuggingFace 자동 ({args.model_size})'}")
    print(f"  신뢰도: {args.confidence}")
    print(f"  SAHI: {'ON' if not args.no_sahi else 'OFF'}")
    print(f"  프레임 스킵: {args.frame_skip}")
    print(f"  버스트 프레임: {args.burst_frames}")
    print("=" * 60)

    print("\n[1/2] 모델 초기화...")
    recognizer = PlateRecognizer(
        model_path=args.model,
        model_size=args.model_size,
        confidence_threshold=args.confidence,
        use_sahi=not args.no_sahi,
        frame_skip=args.frame_skip,
        burst_frames=args.burst_frames,
    )

    print("\n[2/2] 비디오 처리 시작...")
    results = recognizer.process_video(args.video, args.output)

    # 최종 결과
    print()
    print("=" * 60)
    print("  인식 결과 요약")
    print("=" * 60)
    if results:
        valid_count = sum(1 for r in results if r.get("is_valid_plate"))
        print(f"  총 {len(results)}개 번호판 (유효 패턴: {valid_count}개)\n")
        for i, r in enumerate(results):
            valid_mark = " [유효]" if r.get("is_valid_plate") else ""
            print(
                f"  #{i}: \"{r['text']}\"{valid_mark} "
                f"| OCR: {r['ocr_confidence']:.3f} "
                f"| 탐지: {r['detection_confidence']:.3f} "
                f"| 선명도: {r['sharpness']:.1f} "
                f"| 방식: {r.get('detection_method', '?')}"
            )
    else:
        print("  번호판을 찾지 못했습니다.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
