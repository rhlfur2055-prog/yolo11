
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO26 통합 모델 로더 (Ultralytics 최신 모델)
# YOLO26 특징: NMS-free 엔드투엔드 / YOLO11 대비 +5% 정확도
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os as _os

# 모델 우선순위 (위에서부터 먼저 찾으면 사용)
_MODEL_PRIORITY = [
    "yolo26n.pt",          # ★ YOLO26n - 최신 경량 (번호판 전용 fine-tune 필요)
    "yolo26s.pt",          # ★ YOLO26s - 소형
    "yolo11x_plate.pt",    # YOLOv11x fine-tuned (mAP@50=98.4%)
    "yolo11n_plate.pt",    # YOLOv11n 경량
    "yolo26.pt",           # 기존 프로젝트 모델
    "yolo11n.pt",          # COCO fallback
    "yolov8n.pt",         # 최후 fallback
]

def _load_best_model():
    """우선순위에 따라 가장 좋은 모델 자동 로드"""
    from ultralytics import YOLO
    for m in _MODEL_PRIORITY:
        if _os.path.exists(m):
            print(f"[YOLO26] 모델 로드: {m}")
            return YOLO(m)
    # 없으면 YOLO26n 자동 다운로드 (ultralytics에서)
    print("[YOLO26] yolo26n.pt 자동 다운로드 중...")
    return YOLO("yolo26n.pt")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# -*- coding: utf-8 -*-
# ============================================
# plate_engine_pro.py
# 상용급 번호판 인식 엔진 (비젼인급 품질)
# ============================================

import os
import re
import time
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ── plate_recognition_4k 한글 교정 함수 임포트 ──
try:
    from plate_recognition_4k import (
        correct_ocr_hangul,
        correct_hangul_similarity,
        validate_plate_format,
        validate_korean_plate,
    )
    HAS_PLATE_4K_CORRECTION = True
except ImportError:
    HAS_PLATE_4K_CORRECTION = False

# ── OCR 엔진 임포트 ──
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import fast_alpr  # pip install fast-alpr[onnx-gpu]
    HAS_FAST_ALPR = True
except ImportError:
    fast_alpr = None
    HAS_FAST_ALPR = False


class PlateEngineConfig:
    """엔진 설정"""
    # ── 모델 경로 ──
    YOLO_MODEL = "yolo11x_plate.pt"      # 번호판 전용 파인튜닝 (mAP@50=98.4%)
    YOLO_FALLBACK = "yolo26.pt"

    # ── 인식 임계값 (0.7 이상만 표시) ──
    DETECT_CONF = 0.45
    OCR_CONF = 0.70

    # ── MareArts/한국 번호판 정규식 완전판 ──
    KR_PATTERNS = [
        r'^[가-힣]{2}[0-9]{2}[가-힣][0-9]{4}$',         # 구형: 서울12가3456 (8자)
        r'^[0-9]{2,3}[가-힣][0-9]{4}$',                  # 신형: 123가4567
        r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$',        # 구형지역포함: 서울70비9203 (9자)
        r'^[가-힣]{2}[0-9]{2}[바사아자배비하][0-9]{4}$',  # 영업/버스
        r'^[가-힣]{2,3}[0-9]{4}[가-힣]{1}$',             # 영업용 변형
        r'^외교[0-9]{3}-?[0-9]{3}$',
        r'^[가-힣]{2}[0-9]{3}[가-힣]$',                  # 이륜차
        r'^[가-힣]{2}[0-9]{1,2}[가-힣]{1,2}[0-9]{4}$',  # 혼합형
        # ── 전기차/친환경 번호판 ──
        r'^전기[0-9]{4}$',                               # 구형 전기차: 전기1234 (6자)
        r'^[가-힣]{2}전기[0-9]{4}$',                     # 지역+전기차: 서울전기1234
        r'^[0-9]{2}[가-힣][0-9]{4}$',                    # 신형 전기차 (2자리): 12가3456 (7자)
    ]
    PLATE_MIN_LEN = 6    # 전기차 구형(전기1234=6자) 허용
    PLATE_MAX_LEN = 10   # 구형 지역명 포함(서울70비9203=9자) 허용
    CONSECUTIVE_FRAMES_REQUIRED = 3
    # 환경변수로 오버라이드: set PLATE_CONSECUTIVE_FRAMES=1
    _cf = os.environ.get('PLATE_CONSECUTIVE_FRAMES', '')
    if _cf.strip().isdigit():
        CONSECUTIVE_FRAMES_REQUIRED = int(_cf)

    # 자주 혼동되는 문자 보정 (MareArts) 0↔O, 1↔I, 8↔B, 6↔G
    OCR_CONFUSION_MAP = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1", "l": "1",
        "B": "8", "S": "5", "Z": "2", "G": "6",
        "ㅇ": "0", "ㅣ": "1",
    }
    # 멀티프레임 (MultiFrame-LPR): 번호판 픽셀 너비 < 80 이면 멀티프레임 ON
    MULTIFRAME_SIZE = 5
    MULTIFRAME_PLATE_WIDTH_THRESHOLD = 80

    DB_PATH = "plate_records.db"

    PREPROCESS_METHODS = [
        "original",
        "denoise",
        "clahe",
        "gray_threshold",
        "adaptive_threshold",
        "deblur",
        "gamma_bright",
        "gamma_dark",
        "bilateral",
        "morphology",
        "deskew",
        # ⑧~⑮ 추가 전처리
        "sharpen",
        "median_blur",
        "otsu_inv",
        "upscale_2x",
        "brightness_boost",
        "hist_equalize",
        "adaptive_mean",
        "deskew_otsu",
    ]


def _deskew_and_otsu(gray):
    """기울기 보정 후 Otsu 이진화"""
    try:
        coords = np.column_stack(np.where(gray > 128))
        if len(coords) < 50:
            raise ValueError
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = -(angle - 90)
        if abs(angle) > 15:
            raise ValueError
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h),
                                 borderMode=cv2.BORDER_REPLICATE)
        _, result = cv2.threshold(rotated, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result
    except Exception:
        _, result = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result


def normalize(text: str) -> str:
    """OCR 혼동 문자 교정 (숫자 자리에서 O→0, I→1 등)"""
    text = text.strip().replace(' ', '').upper()
    table = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '8'}
    result = []
    for i, ch in enumerate(text):
        if i in (2, 3) and '\uAC00' <= ch <= '\uD7A3':
            result.append(ch)
        elif ch in table:
            result.append(table[ch])
        else:
            result.append(ch)
    return ''.join(result)


class ImagePreprocessor:
    """18종 이미지 전처리 파이프라인"""

    @staticmethod
    def gray_threshold(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adaptive_threshold(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def clahe(img):
        """CLAHE 대비 향상 (clipLimit 2.5→4.0, tile 8x8)"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(img):
        """노이즈 제거 (hqdn3d 스타일 가우시안 + bilateral)"""
        blurred = cv2.bilateralFilter(img, 9, 75, 75)
        return cv2.GaussianBlur(blurred, (3, 3), 0.5)

    @staticmethod
    def deblur(img):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def gamma_bright(img, gamma=0.5):
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    @staticmethod
    def gamma_dark(img, gamma=1.5):
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    @staticmethod
    def bilateral(img):
        return cv2.bilateralFilter(img, 11, 75, 75)

    @staticmethod
    def morphology(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def deskew(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 30:
                    angles.append(angle)
            if angles:
                median_angle = np.median(angles)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
                return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return img

    # ── ⑧~⑮ 추가 전처리 ──

    @staticmethod
    def sharpen(img):
        """⑧ 샤프닝"""
        kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel_sharp)

    @staticmethod
    def median_blur(img):
        """⑨ 중앙값 필터 (점잡음 제거)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.medianBlur(gray, 3)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def otsu_inv(img):
        """⑩ Otsu 반전 (흰 배경 번호판)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(otsu)
        return cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def upscale_2x(img):
        """⑪ 2배 업스케일 (작은 번호판)"""
        return cv2.resize(img, None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def brightness_boost(img):
        """⑫ 밝기 보정 (alpha=1.5, beta=+30)"""
        return cv2.convertScaleAbs(img, alpha=1.5, beta=30)

    @staticmethod
    def hist_equalize(img):
        """⑬ 히스토그램 평활화"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adaptive_mean(img):
        """⑭ Adaptive Mean (blockSize=15)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 15, 8)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def deskew_otsu(img):
        """⑮ 기울기 보정 후 Otsu"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = _deskew_and_otsu(gray)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


class PlateValidator:
    """번호판 유효성 검증기 (한글+숫자 조합, 7~8자만 허용)"""

    def __init__(self):
        self.patterns = [re.compile(p) for p in PlateEngineConfig.KR_PATTERNS]
        self.min_len = PlateEngineConfig.PLATE_MIN_LEN
        self.max_len = PlateEngineConfig.PLATE_MAX_LEN

    # 자주 혼동되는 한글 문자 쌍 (번호판 기준)
    _KR_CONFUSION = {
        "스": "소", "수": "소", "서": "소",  # 소 혼동
        "오": "0",  "아": "아", "어": "어",
        "르": "르", "프": "프",
    }

    def _try_patterns(self, text):
        """패턴 매칭 시도 (정방향 + 역방향 + 한글교정)"""
        candidates = [text, text[::-1]]  # 정방향, 역방향
        # 한글 혼동 교정 버전
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

    # 구형 지역번호판에서만 나오는 상용차 계열 문자 (일반 신형 가나다 제외)
    _COMMERCIAL_CHARS = set("비바사아자배하")

    def validate(self, text):
        # ★ 4자리 숫자만 읽힌 경우 → 전기차 하단 잘림 처리
        # 예: 8060 → 전기8060
        _pure4 = re.match(r'^[0-9]{4}$', text.strip())
        if _pure4:
            candidate = "전기" + text.strip()
            for pattern in self.patterns:
                if pattern.match(candidate):
                    return True, candidate
        clean = self._normalize_for_validation(text)
        if not (self.min_len <= len(clean) <= self.max_len):
            rev = self._normalize_for_validation(text[::-1])
            if self.min_len <= len(rev) <= self.max_len:
                ok, result = self._try_patterns(rev)
                if ok:
                    return True, result
            return False, clean

        # ★ 구형 지역번호판 우선 교정: 앞 1~2자리 숫자가 지역명 오인식
        # 예) 376비7789 → 경기76비7789  (비/바/사/아/자 등 상용차 문자가 있을 때만)
        # 일반 신형 123가4567에는 적용 안 함 (가/나/다 등은 _COMMERCIAL_CHARS 제외)
        m_reg = re.match(r'^[0-9]{1,2}([0-9]{2}([가-힣])[0-9]{4})$', clean)
        if m_reg and m_reg.group(2) in PlateValidator._COMMERCIAL_CHARS:
            suffix = m_reg.group(1)
            for region in PlateValidator._REGION_PREFIXES:
                candidate = region + suffix
                nc = self._normalize_for_validation(candidate)
                for pattern in self.patterns:
                    if pattern.match(nc):
                        return True, nc

        # 정방향 패턴 매칭
        for pattern in self.patterns:
            if pattern.match(clean):
                return True, clean

        # 역방향 / 혼동 교정 시도
        ok, result = self._try_patterns(clean)
        if ok:
            return True, result

        return False, clean

    def _normalize_for_validation(self, text):
        """공백/특수문자 제거, OCR 글자 잘림 보정용 정규화"""
        s = re.sub(r"[\s\-\.\,\;\:\'\"]", "", text)
        # 번호판 문자만 유지: 한글 1자 + 숫자 (앞뒤 잡문자 제거)
        allowed = re.compile(r"[0-9가-힣바사아자외교]")
        return "".join(c for c in s if allowed.match(c))

    # 한국 지역명 접두사 (구형 번호판: 서울, 경기 등)
    _REGION_PREFIXES = [
        "서울","부산","대구","인천","광주","대전","울산","세종",
        "경기","강원","충북","충남","전북","전남","경북","경남","제주",
    ]

    def clean_ocr_text(self, text):
        """OCR 후처리: 특수문자 완전 제거 + 혼동문자 보정 + 두 줄 번호판 교정"""
        clean = text.strip()
        # ★ 핵심: 중간 특수문자도 모두 제거 (번호판에는 숫자·한글·영문만)
        clean = re.sub(r"[^\w가-힣]", "", clean, flags=re.ASCII)
        clean = re.sub(r"\s+", "", clean)

        replacements = getattr(PlateEngineConfig, "OCR_CONFUSION_MAP", {}) or {
            "O": "0", "I": "1", "Z": "2", "S": "5",
            "B": "8", "D": "0", "Q": "0", "G": "6",
            "ㅇ": "0", "ㅣ": "1",
        }
        result = []
        for i, ch in enumerate(clean):
            if ch in replacements and self._should_be_digit(clean, i):
                result.append(replacements[ch])
            else:
                result.append(ch)
        cleaned = "".join(result)

        # 두 줄 번호판 교정: 앞에 숫자가 오면 지역명이 잘렸을 가능성 → 패턴 매칭 시도
        # 예: 376비7789 → 경기76비7789 (앞 숫자가 지역명+숫자로 잘린 것)
        # 이 보정은 validate()에서 패턴 매칭이 실패할 때 별도로 처리
        return cleaned

    def _should_be_digit(self, text, pos):
        if pos > 0 and text[pos - 1].isdigit():
            return True
        if pos < len(text) - 1 and text[pos + 1].isdigit():
            return True
        return False

    def is_valid_length(self, text):
        clean = self._normalize_for_validation(text)
        return self.min_len <= len(clean) <= self.max_len


class PlateDatabase:
    """번호판 기록 데이터베이스"""

    def __init__(self, db_path=None):
        db_path = db_path or PlateEngineConfig.DB_PATH
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS plate_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                camera_id TEXT,
                image_path TEXT,
                vehicle_type TEXT,
                vehicle_color TEXT,
                speed_estimate REAL,
                direction TEXT,
                is_alert INTEGER DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_list (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT UNIQUE NOT NULL,
                alert_type TEXT,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_number ON plate_records(plate_number)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON plate_records(timestamp)")
        self.conn.commit()

    def record_plate(self, plate_number, confidence, camera_id="CAM01",
                     image_path=None, vehicle_type=None, vehicle_color=None):
        alert = self.conn.execute(
            "SELECT * FROM alert_list WHERE plate_number=?",
            (plate_number,)
        ).fetchone()
        is_alert = 1 if alert else 0
        self.conn.execute("""
            INSERT INTO plate_records
            (plate_number, confidence, camera_id, image_path,
             vehicle_type, vehicle_color, is_alert)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (plate_number, confidence, camera_id, image_path,
              vehicle_type, vehicle_color, is_alert))
        self.conn.commit()
        return is_alert, alert

    def add_alert(self, plate_number, alert_type="수배", description=""):
        self.conn.execute("""
            INSERT OR REPLACE INTO alert_list
            (plate_number, alert_type, description)
            VALUES (?, ?, ?)
        """, (plate_number, alert_type, description))
        self.conn.commit()

    def search_plates(self, query, limit=100):
        return self.conn.execute("""
            SELECT * FROM plate_records
            WHERE plate_number LIKE ?
            ORDER BY timestamp DESC LIMIT ?
        """, (f"%{query}%", limit)).fetchall()


class PlateEnginePro:
    """
    상용급 번호판 인식 엔진
    [영상입력] → [YOLO탐지] → [ROI추출] → [10종전처리]
    → [멀티OCR] → [검증/보정] → [DB기록] → [경고알림]
    """

    def __init__(self, config=None):
        self.config = config or PlateEngineConfig()
        self.preprocessor = ImagePreprocessor()
        self.validator = PlateValidator()
        self.db = PlateDatabase()

        model_path = Path(self.config.YOLO_MODEL)
        if not model_path.exists():
            model_path = Path(self.config.YOLO_FALLBACK)
        if not model_path.exists():
            model_path = Path("yolo11n.pt")  # ultralytics 기본
        self.model = YOLO(str(model_path))
        print(f"[엔진] YOLO 모델 로드: {model_path}")

        self.ocr_engines = {}
        if HAS_PADDLEOCR:
            paddle_kwargs = dict(lang="korean", use_angle_cls=True, show_log=False)
            # Windows 한글 경로 우회: 영문 경로에 모델이 있으면 직접 지정
            _paddle_model_root = Path("C:/tools/paddleocr_models")
            if _paddle_model_root.exists():
                paddle_kwargs["det_model_dir"] = str(_paddle_model_root / "det/ml/Multilingual_PP-OCRv3_det_infer")
                paddle_kwargs["rec_model_dir"] = str(_paddle_model_root / "rec/korean/korean_PP-OCRv4_rec_infer")
                paddle_kwargs["cls_model_dir"] = str(_paddle_model_root / "cls/ch_ppocr_mobile_v2.0_cls_infer")
            try:
                self.ocr_engines["paddleocr"] = PaddleOCR(**paddle_kwargs)
            except Exception as e:
                print(f"[엔진] PaddleOCR 초기화 실패: {e}")
        if HAS_EASYOCR:
            self.ocr_engines["easyocr"] = easyocr.Reader(["ko", "en"], gpu=True)
        print(f"[엔진] OCR 엔진: {list(self.ocr_engines.keys())}")

        self.recent_plates = defaultdict(lambda: {"count": 0, "last_seen": 0, "consecutive": 0})
        self.DUPLICATE_THRESHOLD = 3.0
        # 연속 N프레임 감지 시 표시 (이미지 슬라이드 영상은 PLATE_CONSECUTIVE_FRAMES=1 로 설정)
        _env = os.environ.get("PLATE_CONSECUTIVE_FRAMES")
        default_consecutive = int(_env) if (_env and _env.isdigit()) else getattr(
            self.config, "CONSECUTIVE_FRAMES_REQUIRED", 3
        )
        self.consecutive_required = default_consecutive
        # 멀티프레임: 최근 5프레임 크롭 저장 (번호판 너비 < 80px 시 사용)
        self._multiframe_buffer = deque(maxlen=PlateEngineConfig.MULTIFRAME_SIZE)
        # 리테스트/벤치마크용 통계
        self.stats = {
            "frames_processed": 0,
            "plates_shown": 0,
            "filtered_by_length": 0,
            "filtered_by_pattern": 0,
            "filtered_by_confidence": 0,
            "confidences": [],
            "multiframe_used": 0,
            "singleframe_used": 0,
        }

    def _composite_multiframe(self, crops):
        """5프레임 크롭을 하나로 합성 (median → 노이즈 감소)."""
        if not crops:
            return None
        target_h, target_w = crops[0].shape[:2]
        resized = [cv2.resize(c, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for c in crops]
        stack = np.stack(resized, axis=0)
        return np.median(stack, axis=0).astype(np.uint8)

    def process_frame(self, frame, camera_id="CAM01", use_multiframe=False, full_frame=None):
        """
        frame: YOLO 추론용 (640px 등 축소 가능)
        full_frame: OCR 크롭용 원본 고해상도 프레임 (None이면 frame에서 크롭)
        """
        results = []
        self.stats["frames_processed"] += 1
        detections = self.model(frame, conf=self.config.DETECT_CONF, verbose=False)

        crop_src = full_frame if full_frame is not None else frame
        ch_full, cw_full = crop_src.shape[:2]
        ch_det, cw_det = frame.shape[:2]
        sx = cw_full / cw_det
        sy = ch_full / ch_det

        seen_this_frame = set()

        for det in detections[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0])

            ox1, oy1 = int(x1 * sx), int(y1 * sy)
            ox2, oy2 = int(x2 * sx), int(y2 * sy)

            margin_x = int((ox2 - ox1) * 0.1)
            margin_y = int((oy2 - oy1) * 0.15)
            rx1 = max(0, ox1 - margin_x)
            ry1 = max(0, oy1 - margin_y)
            rx2 = min(cw_full, ox2 + margin_x)
            ry2 = min(ch_full, oy2 + margin_y)
            roi = crop_src[ry1:ry2, rx1:rx2]

            if roi.size == 0:
                continue

            roi_h, roi_w = roi.shape[:2]
            # 멀티프레임: 번호판 픽셀 너비 < 80 이면 5프레임 합성 후 OCR
            roi_for_ocr = roi
            if use_multiframe and roi_w < PlateEngineConfig.MULTIFRAME_PLATE_WIDTH_THRESHOLD:
                self._multiframe_buffer.append((roi.copy(), (x1, y1, x2, y2)))
                if len(self._multiframe_buffer) >= PlateEngineConfig.MULTIFRAME_SIZE:
                    crops = [c[0] for c in self._multiframe_buffer]
                    roi_for_ocr = self._composite_multiframe(crops)
                    self._multiframe_buffer.clear()
                    self.stats["multiframe_used"] = self.stats.get("multiframe_used", 0) + 1
                else:
                    continue  # 버퍼 채울 때까지 이 ROI는 스킵
            else:
                if use_multiframe:
                    self.stats["singleframe_used"] = self.stats.get("singleframe_used", 0) + 1
                target_w = 300
                if roi_w < target_w:
                    scale = target_w / roi_w
                else:
                    scale = 1.0
                if scale > 1.0:
                    roi_for_ocr = cv2.resize(
                        roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                    )
                    # 업스케일 후 선명화 적용
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                    roi_for_ocr = cv2.filter2D(roi_for_ocr, -1, kernel)
                else:
                    roi_for_ocr = roi

            # ── 구형 두 줄 번호판 감지: 세로 비율이 높으면 상단/하단 분리 추가 ──
            # (구형 번호판은 가로:세로 ≈ 2:1, 신형은 4:1)
            extra_crops = []
            if roi_h > roi_w * 0.45:   # 세로가 어느정도 있는 번호판
                top_crop = roi_for_ocr[:int(roi_for_ocr.shape[0] * 0.5), :]
                bot_crop = roi_for_ocr[int(roi_for_ocr.shape[0] * 0.4):, :]
                extra_crops = [("top", top_crop), ("bot", bot_crop)]

            # ── 앙상블 투표: 전처리 × N개 OCR → Counter 투표 ──
            from collections import Counter
            all_candidates = []  # [(normalized_text, confidence), ...]

            for method in self.config.PREPROCESS_METHODS:
                try:
                    if method == "original":
                        processed = roi_for_ocr.copy()
                    else:
                        proc_func = getattr(self.preprocessor, method, None)
                        if proc_func is None:
                            continue
                        processed = proc_func(roi_for_ocr.copy())

                    for engine_name, engine in self.ocr_engines.items():
                        text, ocr_conf = self._run_ocr(engine_name, engine, processed)
                        if not text or ocr_conf < 0.25:
                            continue
                        cleaned = self.validator.clean_ocr_text(text)
                        if not self.validator.is_valid_length(cleaned):
                            self.stats["filtered_by_length"] += 1
                            continue
                        is_valid, final_text = self.validator.validate(cleaned)
                        if not is_valid:
                            self.stats["filtered_by_pattern"] += 1
                            continue
                        all_candidates.append((final_text, ocr_conf))
                except Exception:
                    continue

            # 구형 번호판 상단+하단 결합 시도
            if extra_crops:
                top_texts, bot_texts = [], []
                top_confs, bot_confs = [], []
                for crop_name, crop_img in extra_crops:
                    for eng_name, eng in self.ocr_engines.items():
                        t, c = self._run_ocr(eng_name, eng, crop_img)
                        if t and c > 0.2:
                            cleaned_t = self.validator.clean_ocr_text(t)
                            if crop_name == "top":
                                top_texts.append(cleaned_t); top_confs.append(c)
                            else:
                                bot_texts.append(cleaned_t); bot_confs.append(c)
                # 상단 + 하단 조합해서 유효 패턴 탐색
                for tt in (top_texts or [""]):
                    for bt in (bot_texts or [""]):
                        combined = (tt + bt).strip()
                        norm = self.validator._normalize_for_validation(combined)
                        if self.validator.is_valid_length(norm):
                            is_v, final = self.validator.validate(norm)
                            if is_v:
                                avg_c = float(np.mean((top_confs or [0.3]) + (bot_confs or [0.3])))
                                # 지역명 포함 구형 번호판은 투표 가중치 6배 (full-crop 오인식 이김)
                                weight = 6 if re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', final) else 2
                                for _ in range(weight):
                                    all_candidates.append((final, avg_c))

            # 투표: 최다 득표 번호판 선택, 평균 confidence 계산
            best_text = ""
            best_conf = 0.0
            if all_candidates:
                counter = Counter(t for t, _ in all_candidates)
                best_text = counter.most_common(1)[0][0]
                confs = [c for t, c in all_candidates if t == best_text]
                best_conf = sum(confs) / len(confs)

            if best_text and best_conf >= self.config.OCR_CONF:
                seen_this_frame.add(best_text)
                plate_info = self.recent_plates[best_text]
                plate_info["consecutive"] = plate_info.get("consecutive", 0) + 1
                plate_info["last_seen"] = time.time()
                plate_info["count"] += 1

                # 연속 3프레임 이상 감지 시에만 표시 (DB 기록은 3프레임 도달 시 1회만)
                if plate_info["consecutive"] >= self.consecutive_required:
                    is_alert, alert_info = (0, None)
                    if plate_info["consecutive"] == self.consecutive_required:
                        try:
                            is_alert, alert_info = self.db.record_plate(
                                best_text, best_conf, camera_id
                            )
                            if is_alert and alert_info:
                                self._trigger_alert(best_text, alert_info)
                        except Exception:
                            pass
                    self.stats["plates_shown"] += 1
                    self.stats["confidences"].append(best_conf)
                    results.append({
                        "plate": best_text,
                        "confidence": best_conf,
                        "bbox": [ox1, oy1, ox2, oy2],
                        "is_alert": bool(is_alert),
                        "alert_info": alert_info,
                    })

        # 이번 프레임에 없던 번호판은 연속 카운트 리셋
        for key in list(self.recent_plates.keys()):
            if key not in seen_this_frame:
                self.recent_plates[key]["consecutive"] = 0

        return results

    def _run_ocr(self, engine_name, engine, image):
        """OCR 실행. 구형 두 줄 번호판 대응: y좌표 정렬 + 분할 읽기."""
        try:
            if engine_name == "paddleocr":
                result = engine.ocr(image, cls=True)
                if result and result[0]:
                    # y좌표 기준 정렬 (상→하)
                    lines = sorted(result[0], key=lambda l: l[0][0][1])
                    texts = [l[1][0] for l in lines]
                    confs = [l[1][1] for l in lines]
                    if texts:
                        return "".join(texts), float(np.mean(confs))
            elif engine_name == "easyocr":
                result = engine.readtext(image, detail=1, paragraph=False)
                if result:
                    # y좌표 기준 정렬 (상→하: 지역명 먼저, 번호 나중)
                    result_sorted = sorted(result, key=lambda r: r[0][0][1])
                    texts = [r[1] for r in result_sorted]
                    confs = [r[2] for r in result_sorted]
                    combined = "".join(texts)
                    avg_conf = float(np.mean(confs))

                    # 결과가 너무 짧으면 (하단만 읽힌 경우) 상단 별도 시도
                    h, w = image.shape[:2]
                    if len(combined.replace(" ", "")) < 7 and h > w * 0.5:
                        top_half = image[:int(h * 0.55), :]
                        top_res = engine.readtext(top_half, detail=1, paragraph=False)
                        if top_res:
                            top_texts = [r[1] for r in sorted(top_res, key=lambda r: r[0][0][1])]
                            top_confs = [r[2] for r in top_res]
                            combined = "".join(top_texts) + combined
                            avg_conf = float(np.mean(confs + top_confs))

                    return combined, avg_conf
        except Exception:
            pass
        return "", 0.0

    def _trigger_alert(self, plate_number, alert_info):
        print("\n" + "=" * 50)
        print("🚨 [경고] 수배 차량 감지!")
        print(f"   번호판: {plate_number}")
        print(f"   유형: {alert_info[2] if alert_info else '미상'}")
        print(f"   시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50 + "\n")

    def process_video(self, source, camera_id="CAM01", show=True, save=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[에러] 영상 열기 실패: {source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save:
            out_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_count = 0
        total_plates = 0
        start_time = time.time()
        print(f"[시작] 영상 처리: {source} ({w}x{h} @ {fps}fps)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            results = self.process_frame(frame, camera_id)
            total_plates += len(results)

            for r in results:
                x1, y1, x2, y2 = r["bbox"]
                color = (0, 0, 255) if r["is_alert"] else (0, 255, 0)
                thickness = 3 if r["is_alert"] else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f"{r['plate']} ({r['confidence']:.0%})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if r["is_alert"]:
                    cv2.putText(frame, "!! ALERT !!", (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            info = f"FPS: {current_fps:.1f} | Plates: {total_plates}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer:
                writer.write(frame)
            if show:
                cv2.imshow("ANPR Pro", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"\n[완료] {frame_count}프레임, {total_plates}대 인식, 평균 {frame_count/elapsed:.1f} FPS")


# ============================================
# [통합1] FastALPR → ONNX 고속 엔진 (출처: github.com/ankandrew/fast-alpr)
# ============================================
class PlateEngineFast:
    """FastALPR ONNX 고속 엔진. pip install fast-alpr[onnx-gpu] 필요."""

    def __init__(self):
        self._engine = None
        self._validator = PlateValidator()
        if HAS_FAST_ALPR and fast_alpr is not None:
            try:
                if hasattr(fast_alpr, "ALPR"):
                    self._engine = fast_alpr.ALPR()
                elif hasattr(fast_alpr, "Pipeline"):
                    self._engine = fast_alpr.Pipeline()
                elif callable(getattr(fast_alpr, "run", None)):
                    self._engine = fast_alpr
                else:
                    self._engine = None
            except Exception as e:
                print(f"[FastALPR] 초기화 실패: {e}")
                self._engine = None
        if self._engine is None and HAS_FAST_ALPR:
            print("[FastALPR] API 불일치. pip install fast-alpr[onnx-gpu] 후 문서 참조")
        elif not HAS_FAST_ALPR:
            print("[FastALPR] 미설치. pip install fast-alpr[onnx-gpu] 권장")

    @property
    def available(self):
        return self._engine is not None

    def process_frame(self, frame, camera_id="CAM01"):
        """프레임 처리 → [{plate, confidence, bbox}, ...] (Pro와 동일 형식)."""
        results = []
        if not self._engine:
            return results
        try:
            t0 = time.time()
            # fast_alpr 일반적 사용: run(frame) 또는 detect(frame)
            if hasattr(self._engine, "run"):
                raw = self._engine.run(frame)
            elif hasattr(self._engine, "detect"):
                raw = self._engine.detect(frame)
            else:
                raw = []
            elapsed_ms = (time.time() - t0) * 1000
            if not raw:
                return results
            # raw가 리스트 of (text, conf, box) 또는 dict 리스트 등으로 올 수 있음
            for item in (raw if isinstance(raw, list) else [raw]):
                if isinstance(item, dict):
                    text = item.get("plate", item.get("text", ""))
                    conf = float(item.get("confidence", item.get("conf", 0)))
                    bbox = item.get("bbox", item.get("box", [0, 0, 0, 0]))
                else:
                    text = str(item[0]) if len(item) > 0 else ""
                    conf = float(item[1]) if len(item) > 1 else 0
                    bbox = list(item[2]) if len(item) > 2 else [0, 0, 0, 0]
                clean = self._validator.clean_ocr_text(text)
                if not self._validator.is_valid_length(clean):
                    continue
                valid, final = self._validator.validate(clean)
                if valid and conf >= PlateEngineConfig.OCR_CONF:
                    results.append({
                        "plate": final,
                        "confidence": conf,
                        "bbox": bbox,
                        "is_alert": False,
                        "alert_info": None,
                        "engine": "Fast",
                    })
        except Exception as e:
            pass
        return results


def process_frame_unified(
    frame,
    camera_id="CAM01",
    engine_pro=None,
    engine_fast=None,
    engine_mode="pro",
    use_multiframe=False,
):
    """
    Pro / Fast 병렬 실행 후 engine_mode에 따라 결과 반환.
    engine_mode: "pro" | "fast" | "auto"(높은 confidence 채택)
    반환: (results, process_ms_pro, process_ms_fast)
    """
    results = []
    ms_pro, ms_fast = 0.0, 0.0
    pro_results, fast_results = [], []

    if engine_mode in ("pro", "auto") and engine_pro is not None:
        t0 = time.time()
        pro_results = engine_pro.process_frame(frame, camera_id, use_multiframe=use_multiframe)
        ms_pro = (time.time() - t0) * 1000

    if engine_mode in ("fast", "auto") and engine_fast is not None and getattr(engine_fast, "available", True):
        t0 = time.time()
        fast_results = engine_fast.process_frame(frame, camera_id)
        ms_fast = (time.time() - t0) * 1000

    if engine_mode == "pro":
        results = pro_results
    elif engine_mode == "fast":
        results = fast_results
    else:
        # auto: 동일 번호판이면 confidence 높은 것 채택
        by_plate = {}
        for r in pro_results:
            by_plate[r["plate"]] = {**r, "engine": "Pro", "ms": ms_pro}
        for r in fast_results:
            p = r["plate"]
            if p not in by_plate or r["confidence"] > by_plate[p]["confidence"]:
                by_plate[p] = {**r, "engine": "Fast", "ms": ms_fast}
        results = [by_plate[p] for p in by_plate]

    return results, ms_pro, ms_fast


# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ANPR Pro Engine")
    parser.add_argument("--input", default="0", help="영상 소스 (0=웹캠, 파일경로, rtsp)")
    parser.add_argument("--camera", default="CAM01", help="카메라 ID")
    parser.add_argument("--no-show", action="store_true", help="화면 표시 안 함")
    parser.add_argument("--no-save", action="store_true", help="결과 영상 저장 안 함")
    parser.add_argument("--alert-add", help="경고 목록에 번호판 추가")
    args = parser.parse_args()

    engine = PlateEnginePro()

    if args.alert_add:
        engine.db.add_alert(args.alert_add)
        print(f"[경고등록] {args.alert_add}")
    else:
        source = int(args.input) if args.input.isdigit() else args.input
        engine.process_video(source, args.camera, show=not args.no_show, save=not args.no_save)
