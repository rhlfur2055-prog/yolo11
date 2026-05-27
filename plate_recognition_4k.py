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

# stdlib
import argparse
import json
import os
import re
import sys
import time
from enum import Enum, auto
from typing import Any, Optional

# third-party
import cv2
import numpy as np

# local
from config import DisplayConfig, OCRConfig, PathConfig, ThresholdConfig

# Windows 콘솔 한글 깨짐 방지
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# JSON 직렬화 유틸리티 (ndarray 안전 변환)
class NumpyEncoder(json.JSONEncoder):
    """numpy ndarray/scalar 를 안전하게 직렬화하는 JSON 인코더."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# 기본 설정값

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


# 상태 머신

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


# 모델 다운로드 유틸리티

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


# 한국 번호판 검증

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


# 한국 번호판 OCR 오인식 보정

# 신형 번호판: 숫자2-3 + 한글1 + 숫자4  (예: 39가9665)
# EasyOCR이 한글 자리를 숫자/영문으로 오인식하는 패턴 → 한글 후보 매핑
_HANGUL_CONFUSE_MAP: dict[str, str] = {
    # 숫자 오인식 (자주 발생)
    # "2"→"리": PaddleOCR이 리(ㄹ+ㅣ) 자획을 숫자 2로 오독하는 패턴
    "2": "리", "7": "나", "4": "라", "0": "오",
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


# 한국 번호판 형식 교정 테이블

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

# OCR 오인식 한글 교정 확장 (EasyOCR/PaddleOCR 빈번 오인식)
# 받침 포함 문자 → 용도 한글. dict 병합으로 흡수 (런타임 merge 루프 제거).
# ※ 기존 키와 중복되는 항목(곧/볼/룰/줄)은 동일 값이라 생략 — 동작 보존.
_HANGUL_PLATE_CORRECTION.update({
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
    '곡': '고', '곤': '고',
    '녹': '노', '논': '노', '놉': '노',
    '독': '도', '돈': '도', '돋': '도',
    '록': '로', '론': '로', '롯': '로',
    '목': '모', '몬': '모', '몫': '모',
    '복': '보', '본': '보',
    '속': '소', '손': '소', '솔': '소',
    '옥': '오', '온': '오', '올': '오',
    '족': '조', '존': '조', '졸': '조',
    '국': '구', '군': '구', '굿': '구',
    '눈': '누', '눌': '누', '눔': '누',
    '둔': '두', '둘': '두', '둠': '두',
    '룬': '루', '룸': '루',
    '문': '무', '물': '무', '뭄': '무',
    '분': '부', '불': '부', '붐': '부',
    '순': '수', '술': '수', '숨': '수',
    '운': '우', '울': '우', '움': '우',
    '준': '주', '줌': '주',
    '헌': '허', '헐': '허', '험': '허',
    '한': '하', '할': '하', '함': '하',
    '혼': '호', '홀': '호', '홈': '호',
    '백': '배', '밸': '배', '뱅': '배',
})


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


# 지역명 교정 테이블 (개선4)
_REGION_LIST = [
    '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
    '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주',
]
_REGION_SET = set(_REGION_LIST)

_REGION_CORRECTION: dict[str, str] = {
    # 경기
    '걍기': '경기', '겅기': '경기', '견기': '경기',
    '경끼': '경기', '경키': '경기', '껭기': '경기', '격기': '경기',
    '전기': '경기', '점기': '경기', '정기': '경기',  # 전기→경기 오인식
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
    """텍스트 내 인접한 한글 2자가 지역명으로 교정 가능한지 탐색."""
    hangul_only = re.findall(r'[가-힣]', text)
    if len(hangul_only) < 2:
        return None
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
    if not m:
        return text, 0.0  # 형식 불일치
    region, num, hangul, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
    if not all(c in _VALID_PLATE_HANGUL_REGION for c in region):
        return text, 0.50  # 지역명 무효
    if hangul in _VALID_PLATE_HANGUL_ALL:
        return text, 1.0
    corrected = _HANGUL_PLATE_CORRECTION.get(hangul)
    if corrected:
        return region + num + corrected + suffix, 0.90
    corrected = _find_nearest_valid_hangul(hangul)
    if corrected:
        return region + num + corrected + suffix, 0.80
    return text, 0.50  # 한글 자리 교정 실패


# UK/국제 번호판 OCR 오인식 보정

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


# PlateRecognizer 폴백 엔진 분리: plate_recognizer.py
# 2,026줄 거대 클래스를 별도 모듈로 캡슐화 (SRP). 본 파일은 한글/번호판 교정
# 헬퍼 라이브러리로 축소되었다. 기존 import 경로 호환을 위해 PlateRecognizer를
# PEP 562 모듈-레벨 __getattr__로 lazy 재공개한다 —
# `from plate_recognition_4k import PlateRecognizer` 코드가 detection_worker /
# plate_gui / CLI 진입점에서 모두 그대로 동작하면서, plate_recognizer가 본 모듈의
# 헬퍼를 import하는 순환 의존을 피한다 (lazy 평가 시점에는 본 모듈 초기화 완료).
_REEXPORT_NAMES = {"PlateRecognizer", "main"}


def __getattr__(name: str):  # PEP 562
    if name in _REEXPORT_NAMES:
        from plate_recognizer import PlateRecognizer, main
        globals()["PlateRecognizer"] = PlateRecognizer
        globals()["main"] = main
        return globals()[name]
    raise AttributeError(f"module 'plate_recognition_4k' has no attribute {name!r}")


if __name__ == "__main__":
    # CLI 진입점은 plate_recognizer.main()으로 위임
    from plate_recognizer import main as _main
    _main()
