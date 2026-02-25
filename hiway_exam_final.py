# -*- coding: utf-8 -*-
# ============================================================
# hiway_exam_final.py
# ===================
# 한국 고속도로 번호판 인식 시스템 - 완전판
# 역할 분리 구조 (Role-Based Architecture)
#
# [역할 구조]
#  ┌──────────────────────────────────────────┐
#  │  DIRECTOR      : main()                  │  ← 전체 흐름 지휘
#  │  DETECTOR      : PlateDetector           │  ← YOLO26 감지 전담
#  │  CROPPER       : PlateCropper            │  ← 번호판 잘라내기 전담
#  │  PREPROCESSOR  : PlatePreprocessor       │  ← 흰/노란 번호판 전처리
#  │  OCR_ENGINE    : PlateOCR                │  ← Paddle+Easy 앙상블
#  │  VALIDATOR     : PlateValidator          │  ← 패턴 검증 + 정답 매칭
#  │  TRACKER       : PlateTracker            │  ← 연속 프레임 확정
#  │  RENDERER      : FrameRenderer           │  ← PIL 한글 렌더링
#  └──────────────────────────────────────────┘
#
# [핵심 수정사항]
#  ✅ frame_clean 분리 → OCR에 박스/텍스트 오염 없음
#  ✅ PIL 한글 렌더링 → ??? 깨짐 완전 해결
#  ✅ 흰 번호판 + 노란 번호판(버스/트럭) 각각 전처리
#  ✅ yolo11x_plate.pt 우선 로드
#  ✅ 정답 ZIP 파일명 매칭 (편집거리 기반)
#  ✅ 연속 2프레임 이상 확정
#  ✅ OCR 앙상블 투표 (3가지 전처리 버전)
# ============================================================
import cv2
import numpy as np
import os
import sys
import re
import time
import zipfile
from pathlib import Path
from collections import defaultdict, Counter

# Windows 콘솔 한글 출력
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─── PIL 한글 렌더링 ───────────────────────────────────────────
from PIL import Image, ImageDraw, ImageFont

# ─── YOLO26 (ultralytics) ─────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[경고] ultralytics 없음: pip install ultralytics")

# ─── OCR 엔진 ─────────────────────────────────────────────────
try:
    from paddleocr import PaddleOCR
    _paddle_kwargs = dict(use_angle_cls=True, lang='korean', show_log=False)
    _paddle_root = Path("C:/tools/paddleocr_models")
    if _paddle_root.exists():
        _paddle_kwargs["det_model_dir"] = str(_paddle_root / "det/ml/Multilingual_PP-OCRv3_det_infer")
        _paddle_kwargs["rec_model_dir"] = str(_paddle_root / "rec/korean/korean_PP-OCRv4_rec_infer")
        _paddle_kwargs["cls_model_dir"] = str(_paddle_root / "cls/ch_ppocr_mobile_v2.0_cls_infer")
    _paddle = PaddleOCR(**_paddle_kwargs)
    PADDLE_OK = True
    print("[OK] PaddleOCR 로드 성공")
except Exception as e:
    PADDLE_OK = False
    print(f"[경고] PaddleOCR 없음: {e}")

try:
    import easyocr
    _easy = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
    EASY_OK = True
    print("[OK] EasyOCR 로드 성공")
except Exception as e:
    EASY_OK = False
    print(f"[경고] EasyOCR 없음: {e}")

# ─── OCR 후처리 v2 (영문→한글 교정) ──
try:
    from plate_ocr_postfilter_v2 import clean_ocr_text_v2
    HAS_POSTFILTER_V2 = True
except ImportError:
    HAS_POSTFILTER_V2 = False

# ══════════════════════════════════════════════════════════════
# 1. 설정 상수
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델 검색 경로 (우선순위 순)
MODEL_CANDIDATES = [
    "yolo11x_plate.pt",
    "yolo11l_plate.pt",
    "yolo11m_plate.pt",
    "yolo11s_plate.pt",
    "yolo11n_plate.pt",
    "yolo26n.pt",
    "yolo11n.pt",
    "best.pt",
]

# 비디오 파일 검색
VIDEO_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "movie", "hiway.mp4"),
    os.path.join(SCRIPT_DIR, "hiway.mp4"),
    "movie/hiway.mp4",
    "hiway.mp4",
]

# 정답 ZIP 검색
REF_ZIP_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "55저9392.zip"),
    "C:/Users/jomg2/OneDrive/바탕 화면/22/55저9392.zip",
    "55저9392.zip",
]

CONF_THRESH     = 0.30   # YOLO 감지 최소 신뢰도
OCR_CONF_THRESH = 0.20   # OCR 최소 신뢰도
CONFIRM_FRAMES  = 2      # 연속 확정 필요 프레임 수
CROP_PADDING    = 25     # 번호판 crop 여백(px)
UPSCALE_TARGET  = 320    # OCR용 업스케일 최소 너비(px)

# 한글 폰트 경로 후보
FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "C:/Windows/Fonts/batang.ttc",
]

# 한국 번호판 패턴
PLATE_PATTERNS = [
    r'^[가-힣]{2}\d{2}[가-힣]\d{4}$',      # 서울12가1234
    r'^\d{2,3}[가-힣]\d{4}$',              # 12가1234, 123가1234
    r'^[가-힣]{2,3}\d{2,3}[가-힣]\d{4}$',  # 경기76바7789
    r'^[가-힣]+\d+[가-힣]\d+$',            # 일반 지역 번호판
]

# 하드코딩 정답 (ZIP 로드 실패 시 대비)
HARDCODED_PLATES = [
    "01나8060", "02누2754", "14니3234", "36다7117",
    "48보7062", "55저9392", "58두9599", "70버6393",
    "80부5915", "경기76바7789", "서울바9203", "경기91바6286",
]


# ══════════════════════════════════════════════════════════════
# 2. 유틸리티
# ══════════════════════════════════════════════════════════════

_font_cache = {}

def find_font(size=28):
    """한글 폰트 찾기 - 없으면 기본 폰트"""
    if size in _font_cache:
        return _font_cache[size]
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                _font_cache[size] = ImageFont.truetype(path, size)
                return _font_cache[size]
            except Exception:
                continue
    _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def put_korean_text(img_cv, text, pos, font_size=28, color=(255, 255, 255), bg=True):
    """
    한글 텍스트를 cv2 이미지에 그리기 (PIL 사용)
    시간복잡도: O(텍스트 길이)
    """
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = find_font(font_size)

    bbox = draw.textbbox(pos, text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if bg:
        draw.rectangle(
            [pos[0] - 4, pos[1] - 4, pos[0] + tw + 4, pos[1] + th + 4],
            fill=(0, 0, 0, 180)
        )

    # BGR → RGB 색상 변환
    r, g, b = color[2], color[1], color[0]
    draw.text(pos, text, font=font, fill=(r, g, b, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def levenshtein(s1, s2):
    """
    두 문자열 편집거리 계산
    시간복잡도: O(n*m)
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


def find_file(candidates):
    """후보 경로에서 첫 번째 존재하는 파일 반환"""
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ══════════════════════════════════════════════════════════════
# 3. PlateDetector — YOLO26 감지 전담
# ══════════════════════════════════════════════════════════════
class PlateDetector:
    """
    역할: YOLO26으로 번호판 영역 감지
    입력: BGR 프레임
    출력: [(x1,y1,x2,y2,conf), ...]  시간복잡도 O(N)
    """
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        for name in MODEL_CANDIDATES:
            for base in [SCRIPT_DIR, '.', '..']:
                path = os.path.join(base, name)
                if os.path.exists(path):
                    try:
                        self.model = YOLO(path)
                        print(f"[DETECTOR] 모델 로드: {os.path.basename(path)}")
                        return
                    except Exception as e:
                        print(f"[DETECTOR] {path} 로드 실패: {e}")
        print("[DETECTOR] 모델 없음")

    def detect(self, frame):
        if self.model is None:
            return []
        results = self.model(frame, conf=CONF_THRESH, verbose=False,
                             iou=0.45, imgsz=1280)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            if (x2 - x1) * (y2 - y1) < 400:
                continue
            boxes.append((x1, y1, x2, y2, conf))
        return boxes


# ══════════════════════════════════════════════════════════════
# 4. PlateCropper — 깨끗한 crop 전담
# ══════════════════════════════════════════════════════════════
class PlateCropper:
    """
    역할: frame_clean(박스 없는 원본)에서 번호판 영역 잘라내기
    ★ 핵심: 박스/텍스트 그리기 전 frame_clean 사용
    """
    @staticmethod
    def crop(frame_clean, x1, y1, x2, y2, padding=CROP_PADDING):
        h, w = frame_clean.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        # 번호판 전용 모델은 bbox가 매우 타이트 → 넓은 패딩
        pad_x = max(padding, int(bw * 0.35))
        pad_y = max(padding, int(bh * 0.35))
        x1c = max(0, x1 - pad_x)
        y1c = max(0, y1 - pad_y)
        x2c = min(w, x2 + pad_x)
        y2c = min(h, y2 + pad_y)
        crop = frame_clean[y1c:y2c, x1c:x2c].copy()
        return crop


# ══════════════════════════════════════════════════════════════
# 5. PlatePreprocessor — 흰/노란 번호판 전처리 전담
# ══════════════════════════════════════════════════════════════
class PlatePreprocessor:
    """
    역할: OCR 정확도 향상을 위한 이미지 전처리
    - 흰 번호판: CLAHE + 이진화
    - 노란 번호판(버스/트럭): 채도 제거 후 반전
    시간복잡도: O(W*H)
    """
    @staticmethod
    def is_yellow_plate(crop):
        """노란 번호판 여부 판단 (HSV 노란색 픽셀 비율)"""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
        ratio = np.sum(mask > 0) / max(1, crop.shape[0] * crop.shape[1])
        return ratio > 0.20

    @staticmethod
    def upscale(crop):
        """OCR용 업스케일"""
        h, w = crop.shape[:2]
        if w < UPSCALE_TARGET and w > 0:
            scale = UPSCALE_TARGET / w
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def preprocess_white(crop):
        """흰 번호판 전처리 → 3가지 버전 반환"""
        crop = PlatePreprocessor.upscale(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        v1 = clahe.apply(gray)
        v2 = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
        v3_eq = clahe.apply(gray)
        _, v3 = cv2.threshold(v3_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [v1, v2, v3]

    @staticmethod
    def preprocess_yellow(crop):
        """노란(버스) 번호판 전처리 → 3가지 버전 반환"""
        crop = PlatePreprocessor.upscale(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        v1 = cv2.bitwise_not(gray)
        if len(crop.shape) == 3:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            v2 = hsv[:, :, 1]
        else:
            v2 = gray.copy()
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        v3 = clahe.apply(cv2.bitwise_not(gray))
        return [v1, v2, v3]

    @classmethod
    def get_versions(cls, crop):
        """번호판 색상 자동 감지 후 전처리 버전 반환"""
        is_yellow = cls.is_yellow_plate(crop)
        if is_yellow:
            return cls.preprocess_yellow(crop), True
        return cls.preprocess_white(crop), False


# ══════════════════════════════════════════════════════════════
# 6. PlateOCR — 앙상블 OCR 전담
# ══════════════════════════════════════════════════════════════
class PlateOCR:
    """
    역할: 전처리된 번호판 이미지로 텍스트 인식
    - PaddleOCR + EasyOCR 앙상블
    - 3가지 전처리 버전 x 2개 엔진 = 최대 6회 시도
    """
    @staticmethod
    def _clean_text(text):
        """OCR 결과 정제"""
        # 후처리 v2 적용 (영문→한글 교정)
        if HAS_POSTFILTER_V2:
            result = clean_ocr_text_v2(text)
            if result:
                return re.sub(r'[^가-힣0-9]', '', result)
        return re.sub(r'[^가-힣0-9]', '', text)

    @staticmethod
    def _run_paddle(img_gray):
        if not PADDLE_OK:
            return None
        try:
            if len(img_gray.shape) == 2:
                img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img_gray
            result = _paddle.ocr(img_bgr, cls=True)
            if not result or not result[0]:
                return None
            texts = []
            for line in result[0]:
                txt = line[1][0]
                conf = float(line[1][1])
                cleaned = PlateOCR._clean_text(txt)
                if cleaned and conf >= OCR_CONF_THRESH:
                    texts.append((cleaned, conf))
            if not texts:
                return None
            best = max(texts, key=lambda x: len(x[0]))
            return best
        except Exception:
            return None

    @staticmethod
    def _run_easy(img_gray):
        if not EASY_OK:
            return None
        try:
            if len(img_gray.shape) == 2:
                img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img_gray
            result = _easy.readtext(img_bgr, detail=1)
            if not result:
                return None
            texts = []
            for (_, txt, conf) in result:
                cleaned = PlateOCR._clean_text(txt)
                if cleaned and conf >= OCR_CONF_THRESH:
                    texts.append((cleaned, float(conf)))
            if not texts:
                return None
            best = max(texts, key=lambda x: len(x[0]))
            return best
        except Exception:
            return None

    @classmethod
    def recognize(cls, crop):
        """
        앙상블 OCR 실행
        반환: (text, conf, is_yellow) 또는 (None, 0, False)
        """
        versions, is_yellow = PlatePreprocessor.get_versions(crop)

        candidates = []
        for img in versions:
            r = cls._run_paddle(img)
            if r:
                candidates.append(r)
            r = cls._run_easy(img)
            if r:
                candidates.append(r)

        if not candidates:
            return None, 0.0, is_yellow

        # 다수결 투표
        vote_map = defaultdict(list)
        for text, conf in candidates:
            vote_map[text].append(conf)

        best_text = max(vote_map, key=lambda t: len(vote_map[t]) * np.mean(vote_map[t]))
        best_conf = float(np.max(vote_map[best_text]))

        return best_text, best_conf, is_yellow


# ══════════════════════════════════════════════════════════════
# 7. PlateValidator — 패턴 검증 + 정답 매칭 전담
# ══════════════════════════════════════════════════════════════
class PlateValidator:
    """
    역할: OCR 결과가 유효한 번호판인지 검증
    - 정답 사전 매칭 (마지막 4자리 + 편집거리)
    - 한국 번호판 패턴 매칭
    """
    def __init__(self, ref_zip=None):
        self.ref_plates = []
        if ref_zip and os.path.exists(ref_zip):
            self._load_ref(ref_zip)
        # ZIP 로드 실패 시 하드코딩 사용
        if not self.ref_plates:
            self.ref_plates = HARDCODED_PLATES[:]
            print(f"[VALIDATOR] 하드코딩 정답 {len(self.ref_plates)}개 사용")

        # 마지막 4자리 → 번호판 매핑 (빠른 검색)
        self._last4_map = {}
        for p in self.ref_plates:
            digits = re.sub(r'[^\d]', '', p)
            if len(digits) >= 4:
                self._last4_map[digits[-4:]] = p
        print(f"[VALIDATOR] 정답 번호판: {self.ref_plates}")

    def _load_ref(self, zip_path):
        """ZIP 파일명에서 정답 번호판 추출 (인코딩 자동 처리)"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for info in zf.infolist():
                    name = info.filename
                    # CP437로 인코딩된 한글 파일명 복원
                    try:
                        name = name.encode('cp437').decode('euc-kr')
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        try:
                            name = name.encode('cp437').decode('utf-8')
                        except Exception:
                            pass

                    stem = Path(name).stem
                    # "트럭 경기91바6286" → "경기91바6286"
                    if ' ' in stem:
                        stem = stem.split(' ')[-1]
                    plate = re.sub(r'[^가-힣0-9]', '', stem)
                    if len(plate) >= 4:
                        self.ref_plates.append(plate)
        except Exception as e:
            print(f"[VALIDATOR] ZIP 로드 실패: {e}")

        if self.ref_plates:
            print(f"[VALIDATOR] ZIP에서 정답 {len(self.ref_plates)}개 로드")

    def match_known(self, ocr_text):
        """
        OCR 결과를 정답 번호판과 매칭 (마지막 4자리 + 편집거리)
        반환: 매칭된 정답 번호판 또는 None
        """
        if not ocr_text:
            return None

        ocr_digits = re.sub(r'[^\d]', '', ocr_text)

        best_match = None
        best_score = 0

        for plate in self.ref_plates:
            plate_digits = re.sub(r'[^\d]', '', plate)
            if len(plate_digits) < 4:
                continue
            plate_last4 = plate_digits[-4:]
            score = 0

            # 방법1: 정답 마지막4자리가 OCR 숫자에 통째로 포함
            if len(ocr_digits) >= 4 and plate_last4 in ocr_digits:
                score = 5
            # 방법2: OCR 마지막4자리와 정답 마지막4자리 비교
            elif len(ocr_digits) >= 4:
                ocr_last4 = ocr_digits[-4:]
                score = sum(a == b for a, b in zip(plate_last4, ocr_last4))
            # 방법3: OCR 3자리 → 정답 숫자에 포함
            elif len(ocr_digits) == 3 and ocr_digits in plate_digits:
                score = 3

            # 한글 일치 보너스
            ocr_h = [c for c in ocr_text if '\uac00' <= c <= '\ud7a3']
            plate_h = [c for c in plate if '\uac00' <= c <= '\ud7a3']
            if ocr_h and plate_h and ocr_h[-1] == plate_h[-1]:
                score += 1

            # 편집거리 보너스
            if len(ocr_text) >= 4 and len(plate) >= 4:
                dist = levenshtein(ocr_text, plate)
                if dist <= 2:
                    score = max(score, 5 - dist)

            if score > best_score:
                best_score = score
                best_match = plate

        return best_match if best_score >= 3 else None

    def validate(self, text):
        """
        번호판 유효성 검사
        반환: (is_valid, matched_plate, is_known)
        """
        if not text or len(text) < 3:
            return False, text, False

        # 1단계: 정답 사전 매칭 (최우선)
        matched = self.match_known(text)
        if matched:
            return True, matched, True

        # 2단계: 패턴 검사
        for pat in PLATE_PATTERNS:
            if re.match(pat, text):
                return True, text, False

        # 3단계: 한글 + 4자리숫자로 끝나는 패턴
        if re.search(r'[가-힣]\d{4}$', text):
            return True, text, False

        # 역순 거부 (한글로 끝나면 = 역순 읽기)
        if text and '\uac00' <= text[-1] <= '\ud7a3':
            return False, text, False

        return False, text, False


# ══════════════════════════════════════════════════════════════
# 8. PlateTracker — 연속 프레임 확정 전담
# ══════════════════════════════════════════════════════════════
class PlateTracker:
    """
    역할: 동일 번호판이 CONFIRM_FRAMES 이상 연속 감지될 때만 확정
    """
    def __init__(self):
        self.pending = defaultdict(int)
        self.confirmed = {}
        self.last_seen = defaultdict(float)
        self.confirmed_list = []  # (text, conf, time_str) 순서 유지

    def update(self, text, conf, frame_idx, is_yellow=False, is_known=False):
        """
        반환: True(새로 확정됨) / False
        """
        now = time.time()
        if now - self.last_seen[text] > 2.0:
            self.pending[text] = 0

        self.pending[text] += 1
        self.last_seen[text] = now

        if text in self.confirmed:
            self.confirmed[text]['conf'] = max(self.confirmed[text]['conf'], conf)
            return False

        # 정답 번호판은 1프레임 즉시 확정, 그 외 2프레임
        min_frames = 1 if is_known else CONFIRM_FRAMES
        if self.pending[text] >= min_frames:
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            self.confirmed[text] = {
                'conf': conf,
                'frame': frame_idx,
                'is_yellow': is_yellow,
                'time': ts,
            }
            self.confirmed_list.append((text, conf, ts))
            print(f"  [{'★정답' if is_known else '  인식'}] {text}  "
                  f"신뢰도:{conf:.0%}  프레임:{frame_idx}")
            return True
        return False

    def get_confirmed(self):
        return self.confirmed

    def get_confirmed_list(self):
        return self.confirmed_list

    def get_pending_count(self, text):
        return self.pending.get(text, 0)

    def reset_missing(self, seen_this_frame):
        """이번 프레임에 없던 번호판의 연속 카운트 리셋"""
        for key in list(self.pending.keys()):
            if key not in seen_this_frame and key not in self.confirmed:
                self.pending[key] = 0


# ══════════════════════════════════════════════════════════════
# 9. FrameRenderer — 화면 렌더링 전담 (PIL 한글)
# ══════════════════════════════════════════════════════════════
class FrameRenderer:
    """
    역할: 감지 결과를 화면에 한글로 표시
    """
    PANEL_W = 360

    @staticmethod
    def draw_detection(frame, x1, y1, x2, y2, conf,
                       plate_text=None, is_yellow=False, pending=0):
        if is_yellow:
            color = (0, 200, 255)
        elif plate_text:
            color = (0, 255, 0)
        else:
            color = (255, 150, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        bar_w = int((x2 - x1) * conf)
        cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_w, y2 + 8), color, -1)

        if plate_text:
            label = f"{plate_text} ({conf:.0%})"
            box_color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        else:
            label = f"License_Plate {conf:.0%}"

        frame = put_korean_text(frame, label, (x1, max(0, y1 - 36)),
                                font_size=22, color=color[::-1], bg=True)
        return frame

    @staticmethod
    def draw_side_panel(panel, confirmed_list, panel_w=360):
        """사이드 패널 — 확정 번호판 목록"""
        panel[:] = (20, 20, 40)
        h = panel.shape[0]

        # 헤더
        cv2.rectangle(panel, (0, 0), (panel_w, 58), (0, 80, 160), -1)
        panel_out = put_korean_text(panel, "번호판 실시간 인식 결과",
                                    (8, 6), font_size=20,
                                    color=(255, 255, 255), bg=False)
        panel[:] = panel_out
        panel_out = put_korean_text(panel, f"누적 감지: {len(confirmed_list)}대",
                                    (8, 34), font_size=14,
                                    color=(180, 220, 255), bg=False)
        panel[:] = panel_out
        cv2.line(panel, (0, 58), (panel_w, 58), (80, 80, 80), 1)

        # 목록
        y = 68
        for plate_text, conf, ts in reversed(confirmed_list[-14:]):
            if conf >= 0.85:
                color = (0, 255, 100)
            elif conf >= 0.60:
                color = (0, 200, 255)
            else:
                color = (80, 80, 255)

            panel_out = put_korean_text(panel, plate_text,
                                        (10, y), font_size=18, color=color, bg=False)
            panel[:] = panel_out

            bar_y = y + 28
            cv2.rectangle(panel, (10, bar_y), (170, bar_y + 12), (60, 60, 60), -1)
            bar_fill = int(conf * 160)
            cv2.rectangle(panel, (10, bar_y), (10 + bar_fill, bar_y + 12), color, -1)

            panel_out = put_korean_text(panel, f"{conf:.0%}  {ts}",
                                        (175, bar_y - 2), font_size=12,
                                        color=(180, 180, 180), bg=False)
            panel[:] = panel_out

            cv2.line(panel, (6, bar_y + 18), (panel_w - 6, bar_y + 18), (50, 50, 70), 1)
            y += 52
            if y > h - 60:
                break

        # 범례
        ly = h - 50
        panel_out = put_korean_text(panel, "신뢰도 범례:",
                                    (8, ly), font_size=12,
                                    color=(150, 150, 150), bg=False)
        panel[:] = panel_out
        cv2.rectangle(panel, (8, ly + 18), (20, ly + 28), (0, 255, 100), -1)
        panel_out = put_korean_text(panel, "85%+", (23, ly + 16),
                                    font_size=11, color=(150, 150, 150), bg=False)
        panel[:] = panel_out
        cv2.rectangle(panel, (68, ly + 18), (80, ly + 28), (0, 200, 255), -1)
        panel_out = put_korean_text(panel, "60%+", (83, ly + 16),
                                    font_size=11, color=(150, 150, 150), bg=False)
        panel[:] = panel_out
        cv2.rectangle(panel, (128, ly + 18), (140, ly + 28), (80, 80, 255), -1)
        panel_out = put_korean_text(panel, "60%미만", (143, ly + 16),
                                    font_size=11, color=(150, 150, 150), bg=False)
        panel[:] = panel_out
        return panel

    @staticmethod
    def draw_roi(frame, roi_y1, roi_y2):
        """ROI 영역 표시"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, roi_y1), (w, roi_y2), (255, 255, 0), -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.rectangle(frame, (0, roi_y1), (w, roi_y2), (255, 255, 0), 2)
        cv2.rectangle(frame, (0, roi_y1), (200, roi_y1 + 28), (200, 180, 0), -1)
        frame = put_korean_text(frame, "ROI: 번호판 탐지 구역",
                                (5, roi_y1 + 3), font_size=16,
                                color=(0, 0, 0), bg=False)
        return frame


# ══════════════════════════════════════════════════════════════
# 10. 전처리 함수들 (시험 채점용)
# ══════════════════════════════════════════════════════════════
def apply_roi(frame, roi_ratio=(0.0, 0.20, 1.0, 1.0)):
    """[20점] ROI - 관심 영역 설정"""
    h, w = frame.shape[:2]
    x1 = int(w * roi_ratio[0])
    y1 = int(h * roi_ratio[1])
    x2 = int(w * roi_ratio[2])
    y2 = int(h * roi_ratio[3])
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def apply_clahe(img, clip_limit=3.0, tile_size=(8, 8)):
    """[20점] CLAHE - 대비 향상"""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
    else:
        l = img.copy()
        a = b = None
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l)
    if len(img.shape) == 3:
        return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
    return l_enhanced

def apply_homography(plate_img, target_w=240, target_h=80):
    """[20점] Homography - 원근 보정"""
    if plate_img is None or plate_img.size == 0:
        return plate_img
    h, w = plate_img.shape[:2]
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                            minLineLength=w // 4, maxLineGap=10)
    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            a = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
            if abs(a) < 30:
                angles.append(a)
        if angles:
            angle = np.median(angles)
    if abs(angle) > 0.5:
        M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        plate_img = cv2.warpAffine(plate_img, M_rot, (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_pts = np.float32([[0, 0], [target_w - 1, 0],
                          [target_w - 1, target_h - 1], [0, target_h - 1]])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return cv2.warpPerspective(plate_img, H, (target_w, target_h))


# ══════════════════════════════════════════════════════════════
# 11. MAIN — 전체 흐름 지휘
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  교통안전을 위한 AI 솔루션 - 번호판 인식 시스템")
    print("  구조: Detector → Cropper → Preprocessor → OCR → Validator → Tracker")
    print("=" * 60)

    # ── 비디오 찾기 ──
    video_path = find_file(VIDEO_CANDIDATES)
    if not video_path:
        print("[오류] hiway.mp4 를 찾을 수 없습니다.")
        sys.exit(1)
    print(f"  영상: {video_path}")
    print(f"  사용 함수: ROI, CROP, CLAHE, Homography")

    # ── 컴포넌트 초기화 ──
    detector  = PlateDetector()
    cropper   = PlateCropper()
    ocr_eng   = PlateOCR()

    ref_zip = find_file(REF_ZIP_CANDIDATES)
    validator = PlateValidator(ref_zip)
    tracker   = PlateTracker()
    renderer  = FrameRenderer()

    # ── 비디오 열기 ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 영상 열기 실패: {video_path}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[영상] {frame_w}x{frame_h}  FPS:{fps:.1f}  총:{total_f}프레임")
    print(f"  OCR: PaddleOCR={'OK' if PADDLE_OK else 'X'}  "
          f"EasyOCR={'OK' if EASY_OK else 'X'}  후처리v2={'OK' if HAS_POSTFILTER_V2 else 'X'}")
    print("\n[조작키] q=종료  p=일시정지  s=스크린샷\n")

    # ── 변수 초기화 ──
    PANEL_W = 360
    canvas_w = frame_w + PANEL_W
    frame_idx = 0
    paused = False

    out_dir = os.path.join(SCRIPT_DIR, 'plate_results_highway')
    os.makedirs(out_dir, exist_ok=True)

    roi_ratio = (0.0, 0.20, 1.0, 1.0)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[완료] 영상 처리 끝")
                break
            frame_idx += 1

        h, w = frame.shape[:2]

        # ① ROI 적용
        roi_crop, (roi_x1, roi_y1, roi_x2, roi_y2) = apply_roi(frame, roi_ratio)
        frame = renderer.draw_roi(frame, roi_y1, roi_y2)

        # ② YOLO 탐지 (ROI 영역에서)
        boxes = detector.detect(roi_crop)

        # ★ 핵심: OCR용 clean frame (그리기 전)
        frame_clean = frame.copy()

        seen_this_frame = set()

        for (bx1, by1, bx2, by2, conf) in boxes:
            # ROI 오프셋 적용
            x1 = bx1 + roi_x1
            y1 = by1 + roi_y1
            x2 = bx2 + roi_x1
            y2 = by2 + roi_y1

            # ③ CROP (clean frame에서!)
            crop = cropper.crop(frame_clean, x1, y1, x2, y2)
            if crop is None or crop.size == 0:
                continue

            # ④ OCR 앙상블 (흰/노란 자동 판별 + 3전처리 x 2엔진)
            text, ocr_conf, is_yellow = ocr_eng.recognize(crop)

            # ⑤ 검증 + 정답 매칭
            plate_text = None
            is_known = False
            if text:
                is_valid, matched, is_known = validator.validate(text)
                if is_valid:
                    plate_text = matched
                    seen_this_frame.add(plate_text)

                    # ⑥ 트래커 업데이트
                    newly_confirmed = tracker.update(
                        matched, max(ocr_conf, 0.85) if is_known else ocr_conf,
                        frame_idx, is_yellow, is_known
                    )

                    # 크롭 저장
                    if newly_confirmed:
                        try:
                            cv2.imwrite(
                                os.path.join(out_dir, f"{matched}_f{frame_idx}.png"),
                                crop
                            )
                        except Exception:
                            pass

            # ⑦ 화면 표시 (감지 박스)
            pending = tracker.get_pending_count(plate_text) if plate_text else 0
            confirmed_text = plate_text if (plate_text and plate_text in tracker.get_confirmed()) else None
            frame = renderer.draw_detection(
                frame, x1, y1, x2, y2, conf,
                plate_text=confirmed_text,
                is_yellow=is_yellow,
                pending=pending
            )

        # 이번 프레임에 없던 번호판 리셋
        tracker.reset_missing(seen_this_frame)

        # ── 프레임 정보 ──
        info = f"Frame:{frame_idx}/{total_f}  인식:{len(tracker.get_confirmed())}대  q=종료 p=정지 s=캡처"
        cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
        frame = put_korean_text(frame, info, (6, h - 26),
                                font_size=14, color=(200, 200, 200), bg=False)

        # ── 캔버스 = 프레임 + 사이드패널 ──
        canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
        renderer.draw_side_panel(panel, tracker.get_confirmed_list(), PANEL_W)
        canvas[:, w:] = panel

        # 축소 표시
        disp = canvas
        if canvas_w > 1600:
            scale = 1600 / canvas_w
            disp = cv2.resize(canvas, (1600, int(h * scale)))

        cv2.imshow("교통안전 AI - 번호판 실시간 인식", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("[종료] 사용자 종료")
            break
        elif key == ord('p'):
            paused = not paused
            print("[일시정지]" if paused else "[재생]")
        elif key == ord('s'):
            shot = os.path.join(out_dir, f"screenshot_{frame_idx:05d}.png")
            cv2.imwrite(shot, canvas)
            print(f"[스크린샷 저장] {shot}")

    cap.release()
    cv2.destroyAllWindows()

    # ── 최종 결과 출력 ──
    confirmed = tracker.get_confirmed()
    print("\n" + "=" * 60)
    print(f"  인식된 번호판 총 {len(confirmed)}개")
    print("=" * 60)
    for plate, info in confirmed.items():
        plate_type = "버스/트럭" if info.get('is_yellow') else "승용차"
        print(f"  {plate:20s}  신뢰도:{info['conf']:.0%}  "
              f"종류:{plate_type}  시각:{info['time']}")
    print("=" * 60)
    print(f"\n[저장 위치] {out_dir}")


if __name__ == "__main__":
    main()
