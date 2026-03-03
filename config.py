# config.py — 시스템의 유일한 진실 공급원 (Single Source of Truth)
# 모든 경로, 임계값, 서버 설정을 이 파일 한 곳에서 관리합니다.
# 다른 파일에서는 반드시 from config import ... 로 사용하세요.

import os
import sys
import re
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 환경 자동 감지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"

BASE_DIR = Path(__file__).parent.absolute()
USER_HOME = Path(os.path.expanduser("~"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 경로 설정 (PathConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PathConfig:
    # -- 모델 디렉토리 --
    MODEL_DIR = BASE_DIR / "models"

    # -- YOLO 모델 우선순위 (위에서부터 탐색, 첫 번째 존재하는 파일 사용) --
    YOLO_MODEL_PRIORITY = [
        "yolo26n.pt",
        "yolo26s.pt",
        "yolo11x_plate.pt",
        "yolo11n_plate.pt",
        "yolo26.pt",
        "yolo11n.pt",
        "yolov8n.pt",
    ]
    YOLO_PRIMARY = "yolo11x_plate.pt"
    YOLO_PATH = MODEL_DIR / YOLO_PRIMARY    # models/yolo11x_plate.pt
    YOLO_FALLBACK = "yolo26.pt"
    YOLO_COCO_FALLBACK = "yolo11n.pt"

    # -- TensorRT / ONNX 가속 모델 --
    YOLO_ENGINE = "yolo26.engine"
    YOLO_ONNX = "yolo26.onnx"

    # -- HuggingFace 번호판 모델 --
    HF_PLATE_REPO = "morsetechlab/yolov11-license-plate-detection"
    HF_PLATE_FILE = "license-plate-finetune-v1m.pt"
    DEFAULT_MODEL_SIZE = "n"

    # -- PaddleOCR 모델 (동적 경로) --
    @staticmethod
    def paddle_model_dir() -> Path:
        """PaddleOCR 모델 디렉토리. 환경 변수 > 프로젝트 내부 > 홈 디렉토리 순으로 탐색."""
        env = os.environ.get("PADDLE_MODEL_DIR")
        if env and Path(env).exists():
            return Path(env)
        candidates = [
            BASE_DIR / "models" / "paddleocr",
            USER_HOME / ".paddleocr",
        ]
        if IS_WINDOWS:
            candidates.insert(0, Path("C:/tools/paddleocr_models"))
        for p in candidates:
            if p.exists():
                return p
        # 기본값: 프로젝트 내부 (자동 생성됨)
        return BASE_DIR / "models" / "paddleocr"

    # -- Tesseract 경로 --
    @staticmethod
    def tesseract_cmd() -> str:
        """Tesseract 실행 파일 경로."""
        env = os.environ.get("TESSERACT_CMD")
        if env:
            return env
        if IS_WINDOWS:
            win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if Path(win_path).exists():
                return win_path
        return "tesseract"  # Linux/Mac: PATH에 있는 것 사용

    # -- 폰트 경로 --
    @staticmethod
    def font_path(bold: bool = True) -> str:
        """OS에 맞는 한글 폰트 경로 반환."""
        env = os.environ.get("FONT_PATH")
        if env and Path(env).exists():
            return env

        if IS_WINDOWS:
            candidates = [
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
        # 프로젝트 내부 폰트 폴백
        candidates.append(str(BASE_DIR / "fonts" / "NanumGothicBold.ttf"))

        for fp in candidates:
            if Path(fp).exists():
                return fp
        return ""  # 폰트 없음 — PIL default font 사용

    # -- 데이터/출력 경로 --
    DB_PATH = BASE_DIR / "plate_records.db"
    UPLOAD_DIR = BASE_DIR / "uploads"
    RESULTS_DIR = BASE_DIR / "plate_results_final"
    DATASET_DIR = BASE_DIR / "dataset"

    # -- YOLO 모델 자동 탐색 --
    @classmethod
    def find_best_model(cls) -> str:
        """우선순위에 따라 가장 적합한 YOLO 모델 파일을 찾아 반환."""
        for m in cls.YOLO_MODEL_PRIORITY:
            if Path(m).exists():
                return m
            model_dir_path = cls.MODEL_DIR / m
            if model_dir_path.exists():
                return str(model_dir_path)
        return cls.YOLO_MODEL_PRIORITY[0]  # 없으면 첫 번째(자동 다운로드)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 감지/인식 임계값 (ThresholdConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ThresholdConfig:
    # -- 감지 신뢰도 (두 엔진 통일: 0.45 vs 0.50 → 0.50) --
    DETECT_CONF: float = 0.50

    # -- OCR 인식 신뢰도 (두 엔진 통일: 0.60 vs 0.70 → 0.60) --
    OCR_CONF: float = 0.60

    # -- 탐지 최소 신뢰도 (노이즈 제거) --
    MIN_DET_CONFIDENCE: float = 0.50

    # -- 프레임 스킵 (영상 처리 시) --
    FRAME_SKIP: int = 5
    BURST_FRAME_COUNT: int = 10
    NO_DETECT_TOLERANCE: int = 3

    # -- 연속 프레임 확인 횟수 (환경 변수 오버라이드 가능) --
    CONFIRM_FRAME_COUNT: int = int(
        os.environ.get("PLATE_CONSECUTIVE_FRAMES", "5")
    )

    # -- 번호판 크기 필터 --
    MIN_PLATE_WIDTH: int = 60
    MIN_PLATE_HEIGHT: int = 20
    PLATE_MIN_ASPECT: float = 2.0
    PLATE_MAX_ASPECT: float = 5.0
    PLATE_MAX_AREA_RATIO: float = 0.05
    MAX_PLATE_TEXT_LEN: int = 12

    # -- COCO 폴백 차량 크기 --
    MIN_VEHICLE_WIDTH: int = 200
    MIN_VEHICLE_HEIGHT: int = 150
    VEHICLE_CLASS_IDS: set = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    # -- 업스케일 --
    UPSCALE_THRESHOLD: int = 300
    UPSCALE_FACTOR: int = 6

    # -- 멀티프레임 합성 --
    MULTIFRAME_SIZE: int = 5
    MULTIFRAME_PLATE_WIDTH_THRESHOLD: int = 80

    # -- 크롭 & 패딩 --
    PLATE_PADDING_RATIO: float = 0.35
    PLATE_MODEL_PADDING_H: float = 0.30
    PLATE_MODEL_PADDING_V: float = 0.20
    SHARPNESS_THRESHOLD: float = 100.0

    # -- SAHI 타일링 --
    SAHI_SLICE_SIZE: int = 640
    SAHI_OVERLAP_RATIO: float = 0.2

    # -- 시간축 앙상블 --
    TEMPORAL_WINDOW: int = 5
    TEMPORAL_LEVENSHTEIN_MAX: int = 2

    # -- 번호판 텍스트 길이 --
    PLATE_MIN_LEN: int = 6
    PLATE_MAX_LEN: int = 10

    # -- Detection Log OCR 주기 --
    LOG_OCR_INTERVAL: int = 90


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. OCR 설정 (OCRConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OCRConfig:
    # -- 혼동 문자 보정 맵 --
    CONFUSION_MAP = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1", "l": "1",
        "B": "8", "S": "5", "Z": "2", "G": "6",
        "ㅇ": "0", "ㅣ": "1",
    }

    # -- 한글 번호판 문자 목록 --
    KOREAN_PLATE_HANGUL = (
        "가나다라마바사아자차카타파하"
        "거너더러머버서어저처커터퍼허"
        "고노도로모보소오조초코토포호"
        "구누두루무부수우주추쿠투푸후"
        "배"
        "서울부산대구인천광주대전울산세종"
        "경기강원충북충남전북전남경북경남제주"
    )
    KOREAN_PLATE_ALLOWLIST = "0123456789" + KOREAN_PLATE_HANGUL + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # -- 한국 번호판 정규식 (통합) --
    KR_PATTERNS = [
        r"^[가-힣]{2}[0-9]{2}[가-힣][0-9]{4}$",         # 구형: 서울12가3456
        r"^[0-9]{2,3}[가-힣][0-9]{4}$",                  # 신형: 123가4567
        r"^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$",        # 구형지역포함
        r"^[가-힣]{2}[0-9]{2}[바사아자배비하][0-9]{4}$",  # 영업/버스
        r"^[가-힣]{2,3}[0-9]{4}[가-힣]{1}$",             # 영업용 변형
        r"^외교[0-9]{3}-?[0-9]{3}$",                      # 외교
        r"^[가-힣]{2}[0-9]{3}[가-힣]$",                   # 이륜차
        r"^[가-힣]{2}[0-9]{1,2}[가-힣]{1,2}[0-9]{4}$",   # 혼합형
        r"^전기[0-9]{4}$",                                # 전기차 구형
        r"^[가-힣]{2}전기[0-9]{4}$",                      # 지역+전기차
        r"^[0-9]{2}[가-힣][0-9]{4}$",                     # 신형 전기차
    ]

    # -- 컴파일된 한국 번호판 패턴 --
    KR_COMPILED_PATTERNS = [
        re.compile(r"\d{2,3}[가-힣]\d{4}"),
        re.compile(r"[가-힣]{2}\d{1,2}[가-힣]\d{4}"),
        re.compile(r"[가-힣]{2,3}\d{1,2}[가-힣]\d{4}"),
        re.compile(r"[하허호]\d{4}"),
        re.compile(r"\d{2,3}[가-힣]\d{3}"),
        re.compile(r"\d{4,}"),
        re.compile(r"[가-힣]+\d{2,}"),
        re.compile(r"\d{2,}[가-힣]+\d{2,}"),
    ]

    # -- 국제 번호판 패턴 --
    INTL_COMPILED_PATTERNS = [
        re.compile(r"[A-Z]{2}\d{2}\s?[A-Z]{3}"),
        re.compile(r"[A-Z]\d{3}[A-Z]{3}"),
        re.compile(r"[A-Z]{1,3}\s?\d{1,4}\s?[A-Z]{1,3}"),
        re.compile(r"[A-Z0-9]{5,8}"),
        re.compile(r"[A-Z]{1,3}\d{2,4}[A-Z]{0,3}"),
    ]

    # -- 전처리 방법 목록 --
    PREPROCESS_METHODS = [
        "original", "denoise", "clahe", "gray_threshold",
        "adaptive_threshold", "deblur", "gamma_bright", "gamma_dark",
        "bilateral", "morphology", "deskew", "sharpen", "median_blur",
        "otsu_inv", "upscale_2x", "brightness_boost", "hist_equalize",
        "adaptive_mean", "deskew_otsu",
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 서버 설정 (ServerConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ServerConfig:
    HOST = "0.0.0.0"
    API_PORT: int = int(os.environ.get("API_PORT", "8765"))
    WEB_PORT: int = int(os.environ.get("WEB_PORT", "5000"))
    STREAM_COOLDOWN: float = 30.0
    MAX_STREAM_LOG: int = 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 디스플레이 설정 (DisplayConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DisplayConfig:
    # video_plate_recognizer.py
    VIDEO_DISPLAY_W: int = 1280
    VIDEO_DISPLAY_H: int = 720
    TTL_SECONDS: float = 2.5
    LOG_MAX: int = 20

    # plate_gui.py
    GUI_DISPLAY_W: int = 960
    GUI_DISPLAY_H: int = 540
    SIDE_PANEL_W: int = 300
    REFRESH_MS: int = 33  # ~30 FPS

    # 4K 다운스케일 임계값
    DOWNSCALE_THRESHOLD: int = 2560
    DOWNSCALE_TARGET: int = 1920


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 디렉토리 자동 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for _dir in [PathConfig.MODEL_DIR, PathConfig.UPLOAD_DIR, PathConfig.RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
