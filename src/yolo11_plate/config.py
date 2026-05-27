"""config.py — 시스템 단일 진실 공급원 (Single Source of Truth).

경로, 임계값, 서버/디스플레이 설정을 한곳에서 관리한다.
다른 파일에서는 반드시 ``from config import ...`` 형태로 사용한다.

설계 메모:
    * 외부 모듈(plate_engine_pro / plate_recognition_4k / plate_gui / ...)이
      ``PathConfig.MODEL_DIR`` 같은 **클래스 속성 접근** API를 직접 사용하므로,
      dataclass 변환은 API 호환을 위해 의도적으로 보류했다.
    * 각 클래스는 인스턴스화하지 않는 네임스페이스(상수 컨테이너)다.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

logger: Final[logging.Logger] = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 환경 자동 감지 (모듈 레벨 상수)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IS_WINDOWS: Final[bool] = sys.platform.startswith("win")
IS_LINUX: Final[bool] = sys.platform.startswith("linux")
IS_MAC: Final[bool] = sys.platform == "darwin"

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
USER_HOME: Final[Path] = Path.home()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 경로 설정 (PathConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PathConfig:
    """모델/데이터/리소스 파일 경로 모음."""

    # -- 모델 디렉토리 --
    MODEL_DIR: Path = BASE_DIR / "models"

    # -- YOLO 모델 우선순위 (번호판 전용, COCO 제외) --
    YOLO_MODEL_PRIORITY: tuple[str, ...] = (
        "runs/detect/plate_korean_3k_v2/weights/best.pt",  # 1순위: v2 학습 결과
        "runs/detect/plate_korean_3k/weights/best.pt",     # 2순위: v1 학습 결과
        "best.pt",                                         # 3순위: 루트 복사본
    )
    YOLO_PRIMARY: str = "best.pt"
    YOLO_PATH: Path = MODEL_DIR / YOLO_PRIMARY
    YOLO_FALLBACK: str = "runs/detect/plate_korean_3k/weights/best.pt"
    YOLO_COCO_FALLBACK: str = "yolo11n.pt"

    # -- TensorRT / ONNX 가속 모델 --
    YOLO_ENGINE: str = "YOLO11.engine"
    YOLO_ONNX: str = "YOLO11.onnx"

    # -- HuggingFace 번호판 모델 (plate_recognition_4k 3순위 폴백) --
    HF_PLATE_REPO: str = "morsetechlab/yolov11-license-plate-detection"
    HF_PLATE_FILE: str = "license-plate-finetune-v1m.pt"
    DEFAULT_MODEL_SIZE: str = "n"

    # -- 데이터/출력 경로 --
    DB_PATH: Path = BASE_DIR / "plate_records.db"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    RESULTS_DIR: Path = BASE_DIR / "plate_results_final"
    DATASET_DIR: Path = BASE_DIR / "dataset"

    # -- 동적 경로 탐색 ----------------------------------------------------

    @staticmethod
    def paddle_model_dir() -> Path:
        """PaddleOCR 모델 디렉토리. 환경변수 > Windows 시스템 > 프로젝트 > 홈."""
        env = os.environ.get("PADDLE_MODEL_DIR")
        if env and Path(env).exists():
            return Path(env)

        candidates: list[Path] = [
            BASE_DIR / "models" / "paddleocr",
            USER_HOME / ".paddleocr",
        ]
        if IS_WINDOWS:
            candidates.insert(0, Path("C:/tools/paddleocr_models"))

        for path in candidates:
            if path.exists():
                return path
        return BASE_DIR / "models" / "paddleocr"  # 기본값(자동 생성)

    @staticmethod
    def tesseract_cmd() -> str:
        """Tesseract 실행 파일 경로. 환경변수 > Windows 표준 경로 > PATH."""
        env = os.environ.get("TESSERACT_CMD")
        if env:
            return env
        if IS_WINDOWS:
            win_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if win_path.exists():
                return str(win_path)
        return "tesseract"

    @staticmethod
    def font_path(bold: bool = True) -> str:
        """OS별 한글 폰트 경로. 환경변수 > OS 표준 > 프로젝트 폴백 > 빈 문자열."""
        env = os.environ.get("FONT_PATH")
        if env and Path(env).exists():
            return env

        if IS_WINDOWS:
            candidates: list[str] = [
                "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
                "C:/Windows/Fonts/malgun.ttf",
                "C:/Windows/Fonts/gulim.ttc",
            ]
        elif IS_MAC:
            candidates = [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/Library/Fonts/NanumGothicBold.ttf",
            ]
        else:
            candidates = [
                "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            ]
        candidates.append(str(BASE_DIR / "fonts" / "NanumGothicBold.ttf"))

        for fp in candidates:
            if Path(fp).exists():
                return fp
        return ""  # 폰트 없음 → PIL default 사용

    # -- YOLO 모델 자동 탐색 ----------------------------------------------

    @classmethod
    def find_best_model(cls) -> str:
        """번호판 모델 자동 선택. 우선순위 순회 → 첫 유효 경로 반환.

        탐색 순서:
            1. ``./best.pt`` (작업 디렉토리)
            2. ``YOLO_MODEL_PRIORITY`` 각 항목 (상대경로 → ``MODEL_DIR`` 기준)
            3. 모두 실패 시 ``priority[0]`` (Ultralytics 자동 다운로드에 위임)
        """
        candidates: list[Path] = [Path("best.pt")]
        for name in cls.YOLO_MODEL_PRIORITY:
            candidates.append(Path(name))
            candidates.append(cls.MODEL_DIR / name)

        for path in candidates:
            if path.exists() and cls._validate_plate_model(str(path)):
                return str(path)
        return cls.YOLO_MODEL_PRIORITY[0]

    @staticmethod
    def _validate_plate_model(model_path: str) -> bool:
        """번호판 전용 모델 검증 — COCO/범용 모델(80클래스 등) 혼입 방지.

        Ultralytics 미설치, 모델 로드 실패, 메타데이터 부재 시 모두 통과(보수적).
        """
        try:
            from ultralytics import YOLO  # 지연 import
            names: Mapping[int, str] = getattr(YOLO(model_path, verbose=False), "names", {}) or {}
        except Exception:
            return True  # 검증 불가 → 보수적으로 통과

        if not names:
            return True
        if len(names) <= 5:  # 번호판 전용 모델은 클래스가 매우 적음
            return True

        has_plate = any("plate" in str(v).lower() for v in names.values())
        if has_plate or len(names) <= 10:
            return True

        logger.warning(
            "[config] COCO/범용 모델 감지, 건너뜀: %s (classes=%d)",
            model_path, len(names),
        )
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 감지/인식 임계값 (ThresholdConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ThresholdConfig:
    """YOLO 탐지, OCR 인식, 후처리 단계의 모든 임계값."""

    # -- 감지/인식 신뢰도 --
    DETECT_CONF: float = 0.30          # 앞유리/그릴/화물칸 오탐 방지
    OCR_CONF: float = 0.15             # 번호판 전용 모델은 낮춰 재현율 확보
    MIN_DET_CONFIDENCE: float = 0.30   # 노이즈 박스 컷오프

    # -- 프레임 처리 --
    FRAME_SKIP: int = 2
    BURST_FRAME_COUNT: int = 10
    NO_DETECT_TOLERANCE: int = 3
    CONFIRM_FRAME_COUNT: int = int(os.environ.get("PLATE_CONSECUTIVE_FRAMES", "1"))

    # -- 번호판 박스 필터 --
    MIN_PLATE_WIDTH: int = 35
    MIN_PLATE_HEIGHT: int = 16
    PLATE_MIN_ASPECT: float = 2.0
    PLATE_MAX_ASPECT: float = 6.0
    PLATE_MAX_AREA_RATIO: float = 0.08
    MAX_PLATE_TEXT_LEN: int = 12

    # -- COCO 폴백 차량 크기 --
    MIN_VEHICLE_WIDTH: int = 160
    MIN_VEHICLE_HEIGHT: int = 120
    VEHICLE_CLASS_IDS: frozenset[int] = frozenset({2, 3, 5, 7})  # car, motorcycle, bus, truck

    # -- 업스케일 / 멀티프레임 합성 --
    UPSCALE_THRESHOLD: int = 300
    UPSCALE_FACTOR: int = 6
    MULTIFRAME_SIZE: int = 5
    MULTIFRAME_PLATE_WIDTH_THRESHOLD: int = 80

    # -- 크롭 패딩 / 선명도 --
    PLATE_PADDING_RATIO: float = 0.35
    PLATE_MODEL_PADDING_H: float = 0.30
    PLATE_MODEL_PADDING_V: float = 0.20
    SHARPNESS_THRESHOLD: float = 100.0

    # -- SAHI 타일링 (4K 입력용) --
    SAHI_SLICE_SIZE: int = 640
    SAHI_OVERLAP_RATIO: float = 0.2

    # -- 시간축 앙상블 --
    TEMPORAL_WINDOW: int = 5
    TEMPORAL_LEVENSHTEIN_MAX: int = 2

    # -- 번호판 텍스트 길이 --
    PLATE_MIN_LEN: int = 5
    PLATE_MAX_LEN: int = 10

    # -- Detection Log OCR 주기 --
    LOG_OCR_INTERVAL: int = 90

    # -- ROI 필터 (1920x1080 기준 정규화) --
    ROI_ENABLED: bool = False
    ROI_X1: int = 0
    ROI_Y1: int = 0
    ROI_X2: int = 1920
    ROI_Y2: int = 1080


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. OCR 설정 (OCRConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OCRConfig:
    """OCR 후처리(혼동 문자 보정, 한국 번호판 형식 검증)에 사용하는 상수."""

    # -- 혼동 문자 보정 맵 (영문/한글 → 숫자) --
    CONFUSION_MAP: Mapping[str, str] = MappingProxyType({
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1", "l": "1",
        "B": "8", "S": "5", "Z": "2", "G": "6",
        "ㅇ": "0", "ㅣ": "1",
    })

    # -- 한글 번호판 문자 + 지역명 --
    KOREAN_PLATE_HANGUL: Final[str] = (
        "가나다라마바사아자차카타파하"
        "거너더러머버서어저처커터퍼허"
        "고노도로모보소오조초코토포호"
        "구누두루무부수우주추쿠투푸후"
        "배"
        "서울부산대구인천광주대전울산세종"
        "경기강원충북충남전북전남경북경남제주"
    )
    KOREAN_PLATE_ALLOWLIST: Final[str] = (
        "0123456789" + KOREAN_PLATE_HANGUL + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    # -- 한국 번호판 정규식 (앵커 포함, full-match 용) --
    KR_PATTERNS: tuple[str, ...] = (
        r"^[가-힣]{2}[0-9]{2}[가-힣][0-9]{4}$",         # 구형: 서울12가3456
        r"^[0-9]{2,4}[가-힣][0-9]{4}$",                  # 신형: 12가4567, 8519우6374
        r"^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$",        # 구형 지역포함
        r"^[가-힣]{2}[0-9]{2}[바사아자배비하][0-9]{4}$",  # 영업/버스
        r"^[가-힣]{2,3}[0-9]{4}[가-힣]{1}$",             # 영업용 변형
        r"^외교[0-9]{3}-?[0-9]{3}$",                      # 외교
        r"^[가-힣]{2}[0-9]{3}[가-힣]$",                   # 이륜차
        r"^[가-힣]{2}[0-9]{1,2}[가-힣]{1,2}[0-9]{4}$",   # 혼합형
        r"^전기[0-9]{4}$",                                # 전기차 구형
        r"^[가-힣]{2}전기[0-9]{4}$",                      # 지역+전기차
        r"^[0-9]{2}[가-힣][0-9]{4}$",                     # 신형 전기차
        r"^[가-힣][0-9]{2}[가-힣][0-9]{4}$",              # 영업 1줄: 충86다6118
        r"^[가-힣][0-9]{4}$",                             # 2줄판 하단: 바6286
        r"^[가-힣]{2}[0-9]{4}$",                          # 관용 2자+4숫자: 이나8060
    )

    # -- 컴파일된 한국 번호판 패턴 (부분 매칭, 텍스트 스캔용) --
    KR_COMPILED_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\d{2,3}[가-힣]\d{4}"),
        re.compile(r"[가-힣]{2}\d{1,2}[가-힣]\d{4}"),
        re.compile(r"[가-힣]{2,3}\d{1,2}[가-힣]\d{4}"),
        re.compile(r"[하허호]\d{4}"),
        re.compile(r"\d{2,3}[가-힣]\d{3}"),
        re.compile(r"\d{4,}"),
        re.compile(r"[가-힣]+\d{2,}"),
        re.compile(r"\d{2,}[가-힣]+\d{2,}"),
    )

    # -- 국제 번호판 패턴 (영문 위주) --
    INTL_COMPILED_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"[A-Z]{2}\d{2}\s?[A-Z]{3}"),
        re.compile(r"[A-Z]\d{3}[A-Z]{3}"),
        re.compile(r"[A-Z]{1,3}\s?\d{1,4}\s?[A-Z]{1,3}"),
        re.compile(r"[A-Z0-9]{5,8}"),
        re.compile(r"[A-Z]{1,3}\d{2,4}[A-Z]{0,3}"),
    )

    # -- 전처리 방법 목록 (야간/역광 대응 포함 10종) --
    PREPROCESS_METHODS: tuple[str, ...] = (
        "original", "clahe", "sharpen",
        "invert_color", "green_plate", "yellow_plate", "color_plate_clahe",
        "night_clahe", "backlight_adaptive", "brightness_normalize",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 서버 설정 (ServerConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ServerConfig:
    """API/WebSocket 서버 바인딩 설정."""

    HOST: str = "0.0.0.0"
    API_PORT: int = int(os.environ.get("API_PORT", "8765"))
    WEB_PORT: int = int(os.environ.get("WEB_PORT", "5000"))
    STREAM_COOLDOWN: float = 30.0
    MAX_STREAM_LOG: int = 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 디스플레이 설정 (DisplayConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DisplayConfig:
    """GUI / 영상 표시 관련 해상도·타이밍 상수."""

    # -- video_plate_recognizer --
    VIDEO_DISPLAY_W: int = 1280
    VIDEO_DISPLAY_H: int = 720
    TTL_SECONDS: float = 2.5
    LOG_MAX: int = 20

    # -- plate_gui --
    GUI_DISPLAY_W: int = 960
    GUI_DISPLAY_H: int = 540
    SIDE_PANEL_W: int = 300
    REFRESH_MS: int = 33  # ~30 FPS

    # -- 4K 다운스케일 임계값 --
    DOWNSCALE_THRESHOLD: int = 2560
    DOWNSCALE_TARGET: int = 1920


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 디렉토리 자동 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_BOOTSTRAP_DIRS: Final[tuple[Path, ...]] = (
    PathConfig.MODEL_DIR,
    PathConfig.UPLOAD_DIR,
    PathConfig.RESULTS_DIR,
)
for _dir in _BOOTSTRAP_DIRS:
    _dir.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 모델 워밍업
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def warmup_model(model: Any, imgsz: int = 640) -> None:
    """YOLO 모델 워밍업 — 더미 프레임 1회 추론으로 첫 프레임 지연 제거."""
    import numpy as np  # 지연 import: config import 시 numpy 의존 회피
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy, conf=0.5, verbose=False)
