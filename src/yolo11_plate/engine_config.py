# -*- coding: utf-8 -*-
"""
plate_engine_pro 의 런타임 설정 (PlateEngineConfig) 격리 모듈.

설계 의도
---------
- 엔진 코드(추론 파이프라인)와 설정 객체(임계값/경로/패턴)를 분리해
  PlateEnginePro / PlateEngineFast 가 외부 설정에 대해 명시적으로 의존하도록 한다.
- frozen dataclass 로 만들어 런타임 상호 수정으로 인한 사이드이펙트를 차단한다.
- 기본값은 모두 config.py(PathConfig / ThresholdConfig / OCRConfig) 에서 위임받으며,
  테스트 / 실험 시에는 키워드 인자로 즉시 오버라이드 가능하다:

      cfg = PlateEngineConfig(OCR_CONF=0.5, MULTIFRAME_SIZE=3)
      engine = PlateEnginePro(config=cfg)
"""

from dataclasses import dataclass, field
from typing import Mapping

from .config import OCRConfig, PathConfig, ThresholdConfig


@dataclass(frozen=True)
class PlateEngineConfig:
    """번호판 인식 엔진 설정 (불변).

    필드는 모두 config.py 중앙 설정의 위임값이거나,
    엔진 내부에서만 의미를 가지는 ROI 기본 박스이다.
    """

    # ── 모델 경로 (PathConfig 위임) ──
    YOLO_MODEL: str = PathConfig.YOLO_PRIMARY
    YOLO_FALLBACK: str = PathConfig.YOLO_FALLBACK

    # ── 탐지 / OCR 임계값 (ThresholdConfig 위임) ──
    DETECT_CONF: float = ThresholdConfig.DETECT_CONF
    OCR_CONF: float = ThresholdConfig.OCR_CONF

    # ── ROI (기본은 사실상 전체 프레임) ──
    ROI_X1: int = 0
    ROI_X2: int = 9999
    ROI_Y1: int = 0
    ROI_Y2: int = 9999

    # ── 한국 번호판 패턴 / 길이 제약 ──
    KR_PATTERNS: tuple[str, ...] = OCRConfig.KR_PATTERNS
    PLATE_MIN_LEN: int = ThresholdConfig.PLATE_MIN_LEN
    PLATE_MAX_LEN: int = ThresholdConfig.PLATE_MAX_LEN
    CONSECUTIVE_FRAMES_REQUIRED: int = ThresholdConfig.CONFIRM_FRAME_COUNT

    # ── 자주 혼동되는 OCR 문자 교정 매핑 (MappingProxyType) ──
    OCR_CONFUSION_MAP: Mapping[str, str] = field(
        default_factory=lambda: OCRConfig.CONFUSION_MAP
    )

    # ── 멀티프레임 누적 OCR 설정 ──
    MULTIFRAME_SIZE: int = ThresholdConfig.MULTIFRAME_SIZE
    MULTIFRAME_PLATE_WIDTH_THRESHOLD: int = ThresholdConfig.MULTIFRAME_PLATE_WIDTH_THRESHOLD

    # ── 기록용 SQLite DB 경로 (Path → str 변환) ──
    DB_PATH: str = str(PathConfig.DB_PATH)

    # ── 활성 전처리 메서드 목록 ──
    PREPROCESS_METHODS: tuple[str, ...] = OCRConfig.PREPROCESS_METHODS
