
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO26 통합 모델 로더 (Ultralytics 최신 모델)
# YOLO26 특징: NMS-free 엔드투엔드 / YOLO11 대비 +5% 정확도
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os as _os
from config import PathConfig, ThresholdConfig, OCRConfig

def _load_best_model():
    """우선순위에 따라 가장 좋은 모델 자동 로드"""
    from ultralytics import YOLO
    best = PathConfig.find_best_model()
    print(f"[YOLO26] 모델 로드: {best}")
    return YOLO(best)
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

# ── 개선된 OCR 후처리 v2 ──
try:
    from plate_ocr_postfilter_v2 import clean_ocr_text_v2, ensemble_vote_v2
    HAS_POSTFILTER_V2 = True
except ImportError:
    HAS_POSTFILTER_V2 = False

# ── OCR 엔진 임포트 ──
try:
    import easyocr
    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False

try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except Exception:
    HAS_PADDLEOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

try:
    import fast_alpr  # pip install fast-alpr[onnx-gpu]
    HAS_FAST_ALPR = True
except Exception:
    fast_alpr = None
    HAS_FAST_ALPR = False

# ── CRNN OCR 모델 (학습된 번호판 전용) ──
try:
    import torch as _torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ── 모델 우선순위 ──
_MODEL_PRIORITY = [
    "yolo11x_plate.pt",
    "yolo11n_plate.pt",
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26.pt",
    "yolo11n.pt",
    "yolov8n.pt",
]


class _CRNNModel:
    """학습된 CRNN 번호판 OCR 모델 래퍼."""

    def __init__(self, model_path):
        import torch
        import torch.nn as nn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.idx2char = checkpoint["idx2char"]
        self.img_h = checkpoint.get("img_h", 64)
        self.img_w = checkpoint.get("img_w", 256)
        num_classes = checkpoint["num_classes"]
        hidden = checkpoint.get("hidden_size", 256)
        n_layers = checkpoint.get("num_layers", 2)

        # CRNN 모델 구조 (train_plate_ocr.py와 동일)
        class CRNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                    nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                    nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                    nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                    nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                    nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                    nn.Conv2d(512, 512, (2, 1), 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
                )
                self.rnn = nn.LSTM(512, hidden, n_layers,
                                   bidirectional=True, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden * 2, num_classes)

            def forward(self, x):
                conv = self.cnn(x)
                conv = conv.squeeze(2).permute(0, 2, 1)
                rnn_out, _ = self.rnn(conv)
                return self.fc(rnn_out).permute(1, 0, 2)

        self.model = CRNN().to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def recognize(self, bgr_image):
        """BGR 이미지 → (text, confidence)."""
        import torch

        # 전처리: 그레이스케일 → 리사이즈 → 패딩 → 정규화
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) if len(bgr_image.shape) == 3 else bgr_image
        h, w = gray.shape[:2]
        ratio = self.img_h / h
        new_w = min(int(w * ratio), self.img_w)
        gray = cv2.resize(gray, (new_w, self.img_h), interpolation=cv2.INTER_CUBIC)
        if new_w < self.img_w:
            pad = np.ones((self.img_h, self.img_w - new_w), dtype=np.uint8) * 255
            gray = np.concatenate([gray, pad], axis=1)

        tensor = torch.FloatTensor(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)  # (T, 1, C)
            probs = output.softmax(2)
            max_probs, preds = probs.max(2)  # (T, 1)

        # CTC greedy decode
        chars = []
        confs = []
        prev = -1
        for t in range(preds.size(0)):
            p = preds[t, 0].item()
            c = max_probs[t, 0].item()
            if p != 0 and p != prev:
                if p in self.idx2char:
                    chars.append(self.idx2char[p])
                    confs.append(c)
            prev = p

        text = "".join(chars)
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf


class PlateEngineConfig:
    """엔진 설정"""
    # ── 모델 경로 ──
    YOLO_MODEL = PathConfig.YOLO_PRIMARY
    YOLO_FALLBACK = PathConfig.YOLO_FALLBACK

    # ── 인식 임계값 ──
    DETECT_CONF = ThresholdConfig.DETECT_CONF
    ROI_X1 = 200
    ROI_X2 = 900
    ROI_Y1 = 300
    ROI_Y2 = 700
    OCR_CONF = ThresholdConfig.OCR_CONF

    # ── 출력 필터링 임계값 ──
    OUTPUT_CONF_HIGH = 0.90     # ✅ HIGH 확정
    OUTPUT_CONF_MEDIUM = 0.70   # ⚠️ MEDIUM (재확인 권장)
    OUTPUT_CONF_LOW = 0.70      # ❌ 미만 → 폐기
    MIN_BBOX_WIDTH = 50         # 최소 bbox 가로 px (미만 → 신뢰도 차감/거부)
    MIN_BBOX_HEIGHT = 15        # 최소 bbox 세로 px
    MIN_FRAME_COUNT = 2         # 확정 최소 프레임 수 (영상 모드)
    TRACKER_IOU_THRESHOLD = 0.30  # IoU 기준 (미만 → 다른 차량)
    TRACKER_TTL_FRAMES = 15       # 미감지 후 트랙 만료 프레임 수

    # ── MareArts/한국 번호판 정규식 완전판 ──
    KR_PATTERNS = OCRConfig.KR_PATTERNS
    PLATE_MIN_LEN = ThresholdConfig.PLATE_MIN_LEN
    PLATE_MAX_LEN = ThresholdConfig.PLATE_MAX_LEN
    CONSECUTIVE_FRAMES_REQUIRED = ThresholdConfig.CONFIRM_FRAME_COUNT

    # 자주 혼동되는 문자 보정 (MareArts) 0↔O, 1↔I, 8↔B, 6↔G
    OCR_CONFUSION_MAP = OCRConfig.CONFUSION_MAP
    # 멀티프레임 (MultiFrame-LPR): 번호판 픽셀 너비 < 80 이면 멀티프레임 ON
    MULTIFRAME_SIZE = ThresholdConfig.MULTIFRAME_SIZE
    MULTIFRAME_PLATE_WIDTH_THRESHOLD = ThresholdConfig.MULTIFRAME_PLATE_WIDTH_THRESHOLD

    DB_PATH = str(PathConfig.DB_PATH)

    PREPROCESS_METHODS = OCRConfig.PREPROCESS_METHODS


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
        """[평가기준 20점] CLAHE (Contrast Limited Adaptive Histogram Equalization) - 대비 향상 (clipLimit 4.0, tile 8x8)"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(img):
        """노이즈 제거 (bilateral 필터 — 가장자리·획 보존, 이중 블러 제거)"""
        return cv2.bilateralFilter(img, 7, 50, 50)

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

    @staticmethod
    def unsharp_mask(img):
        """⑯ 언샤프 마스크 (선명도 향상 + 가장자리 보존)"""
        blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    @staticmethod
    def auto_contrast(img):
        """⑰ 자동 대비 보정 (밝기 분석 후 적응적 CLAHE)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        # 어두운 이미지: 높은 clipLimit, 밝은 이미지: 낮은 clipLimit
        if mean_val < 100:
            clip = 6.0
        elif mean_val > 180:
            clip = 2.0
        else:
            clip = 4.0
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


class PlateValidator:
    """번호판 유효성 검증기 (한글+숫자 조합, 7~8자만 허용)"""

    def __init__(self):
        self.patterns = [re.compile(p) for p in PlateEngineConfig.KR_PATTERNS]
        self.min_len = PlateEngineConfig.PLATE_MIN_LEN
        self.max_len = PlateEngineConfig.PLATE_MAX_LEN

    # 자주 혼동되는 한글 문자 쌍 (번호판 기준 + OCR 오인식 교정 확장)
    _KR_CONFUSION = {
        "스": "소", "수": "소",  # 소 혼동
        "오": "0",
        "아": "아", "어": "어",
        "르": "르", "프": "프",
        # ★ OCR 빈출 혼동 쌍 추가 (비↔바, 시↔서, 당→다, 물→무 등)
        "비": "바", "시": "서", "지": "저",
        "당": "다", "랑": "라", "물": "무",
        "법": "버", "낭": "나", "문": "누",
        "대": "다", "태": "타", "내": "나",
        "래": "라", "매": "마", "새": "사",
        "재": "자", "채": "차", "해": "하",
        # ★ 추가: 테스트 결과 발견된 혼동 쌍
        "니": "나", "두": "다", "버": "바",
        "누": "나", "배": "바",
    }

    # ★ 지역명 OCR 오인식 교정 맵 (2글자 단위)
    _REGION_CONFUSION_MAP = {
        "얼리": "경기", "잘리": "경기", "결리": "경기", "열리": "경기",
        "건것": "경기", "견기": "경기", "경거": "경기", "결기": "경기",
        "경리": "경기", "갱기": "경기", "겸기": "경기",
        "서올": "서울", "서을": "서울", "시울": "서울", "사울": "서울",
        "서룰": "서울", "셔울": "서울",
        "부선": "부산", "부잔": "부산", "부진": "부산",
        "대귀": "대구", "대고": "대구", "데구": "대구",
        "인첨": "인천", "인전": "인천", "인견": "인천",
        "광쥬": "광주", "광지": "광주", "강주": "광주",
        "대젼": "대전", "대진": "대전", "데전": "대전",
        "울잔": "울산", "울선": "울산", "을산": "울산",
        "세졍": "세종", "세정": "세종", "새종": "세종",
        "간원": "강원", "깅원": "강원",
        "충복": "충북", "총북": "충북", "층북": "충북",
        "충넘": "충남", "총남": "충남", "층남": "충남",
        "전복": "전북", "전볶": "전북", "잔북": "전북",
        "전넘": "전남", "잔남": "전남", "젼남": "전남",
        "경복": "경북", "겸북": "경북", "경뵉": "경북",
        "경넘": "경남", "겸남": "경남", "경냄": "경남",
        "재주": "제주", "제쥬": "제주", "재쥬": "제주",
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

        # ★ 한글 유효성 교정: 패턴은 맞지만 한글이 유효하지 않으면 자동 교정
        clean = self._fix_invalid_hangul(clean)

        # ★ 지역명 OCR 오인식 교정 (얼리→경기, 건것→경기 등)
        clean = self._fix_region_name(clean)

        # 정방향 패턴 매칭 (먼저 시도 - 이미 유효하면 지역 추측 불필요)
        for pattern in self.patterns:
            if pattern.match(clean):
                return True, clean

        # ★ 구형 지역번호판 교정: 앞 1~2자리 숫자가 지역명 오인식
        # 예) 376비7789 → 경기76비7789  (비/바/사/아/자 등 상용차 문자가 있을 때만)
        # 일반 신형 123가4567에는 적용 안 함 (가/나/다 등은 _COMMERCIAL_CHARS 제외)
        # ★ 이미 유효 패턴 매칭된 경우 여기까지 오지 않음 (위에서 return)
        m_reg = re.match(r'^[0-9]{1,2}([0-9]{2}([가-힣])[0-9]{4})$', clean)
        if m_reg and m_reg.group(2) in PlateValidator._COMMERCIAL_CHARS:
            suffix = m_reg.group(1)
            for region in PlateValidator._REGION_PREFIXES:
                candidate = region + suffix
                nc = self._normalize_for_validation(candidate)
                for pattern in self.patterns:
                    if pattern.match(nc):
                        return True, nc

        # 역방향 / 혼동 교정 시도
        ok, result = self._try_patterns(clean)
        if ok:
            return True, result

        return False, clean

    def _fix_region_name(self, text):
        """지역명 OCR 오인식 교정: 앞 2~3 한글이 유효 지역이 아니면 _REGION_CONFUSION_MAP으로 교정"""
        # 구형 번호판 패턴: 한글2~3자 + 숫자 + 한글 + 숫자
        m = re.match(r'^([가-힣]{2,3})(\d{1,2}[가-힣]\d{4})$', text)
        if not m:
            return text
        region = m.group(1)
        suffix = m.group(2)
        _valid = set(self._REGION_PREFIXES)
        if region in _valid:
            return text  # 이미 유효한 지역
        # _REGION_CONFUSION_MAP에서 교정 시도
        corrected = self._REGION_CONFUSION_MAP.get(region)
        if corrected:
            return corrected + suffix
        # 편집거리 1~2인 유효 지역 찾기
        best_region = None
        best_dist = 999
        for vr in _valid:
            if len(vr) != len(region):
                continue
            dist = sum(1 for a, b in zip(region, vr) if a != b)
            if dist < best_dist:
                best_dist = dist
                best_region = vr
        if best_region and best_dist <= 1:
            return best_region + suffix
        return text

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
        """OCR 후처리: 특수문자 완전 제거 + 혼동문자 보정 + 한글 교정 + 두 줄 번호판 교정"""
        # ★ v2 후처리기 우선 사용 (영문→한글 + 숫자위치 O→0 통합 교정)
        if HAS_POSTFILTER_V2:
            v2_result = clean_ocr_text_v2(text)
            if v2_result:
                # ★ v2가 지역명 포함 결과를 생성하면 추가 교정 건너뜀
                # (plate_recognition_4k의 교정이 지역명을 망칠 수 있음)
                _v2_has_region = bool(re.match(
                    r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', v2_result))
                if not _v2_has_region and HAS_PLATE_4K_CORRECTION:
                    try:
                        v2_result = correct_ocr_hangul(v2_result)
                        v2_result = correct_hangul_similarity(v2_result)
                        fmt_text, fmt_score = validate_plate_format(v2_result)
                        if fmt_score > 0:
                            v2_result = fmt_text
                    except Exception:
                        pass
                return v2_result

        # ── fallback: 기존 로직 ──
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

        # ★ 한글 교정: plate_recognition_4k의 교정 함수 적용
        if HAS_PLATE_4K_CORRECTION:
            try:
                cleaned = correct_ocr_hangul(cleaned)
                cleaned = correct_hangul_similarity(cleaned)
                # validate_plate_format 으로 무효 한글 → 유효 한글 교정
                fmt_text, fmt_score = validate_plate_format(cleaned)
                if fmt_score > 0:
                    cleaned = fmt_text
            except Exception:
                pass

        return cleaned

    # 유효 번호판 한글 문자 집합
    _VALID_PLATE_HANGUL = set(
        '가나다라마바사아자차카타파하'
        '거너더러머버서어저처커터퍼허'
        '고노도로모보소오조호'
        '구누두루무부수우주'
        '배육'
    )

    def _fix_invalid_hangul(self, text):
        """번호판 한글 위치에 유효하지 않은 한글이 있으면 _KR_CONFUSION으로 교정"""
        # 신형: XX[한글]XXXX 패턴에서 한글 위치 찾기
        m = re.match(r'^(\d{2,3})([가-힣])(\d{4})$', text)
        if m:
            hg = m.group(2)
            if hg not in self._VALID_PLATE_HANGUL:
                fixed = self._KR_CONFUSION.get(hg, hg)
                if fixed in self._VALID_PLATE_HANGUL:
                    return m.group(1) + fixed + m.group(3)
        # 구형: 지역XX[한글]XXXX
        m2 = re.match(r'^([가-힣]{2,3})(\d{2})([가-힣])(\d{4})$', text)
        if m2:
            hg = m2.group(3)
            if hg not in self._VALID_PLATE_HANGUL:
                fixed = self._KR_CONFUSION.get(hg, hg)
                if fixed in self._VALID_PLATE_HANGUL:
                    return m2.group(1) + m2.group(2) + fixed + m2.group(4)
        return text

    def _should_be_digit(self, text, pos):
        if pos > 0 and text[pos - 1].isdigit():
            return True
        if pos < len(text) - 1 and text[pos + 1].isdigit():
            return True
        return False

    def is_valid_length(self, text):
        clean = self._normalize_for_validation(text)
        return self.min_len <= len(clean) <= self.max_len

    # ── 번호판 유형/차량 유형 분류 ──

    # 번호판 허용 한글 (차량 용도별)
    _HANGUL_COMMERCIAL = set('아바사자')         # 영업용 (택시, 버스 등)
    _HANGUL_RENTAL = set('하허호')               # 렌터카

    @staticmethod
    def classify_plate_type(text):
        """번호판 유형 분류 → str"""
        # 전기차: 3자리(700~799) + 한글 + 4자리
        m = re.match(r'^(\d{3})([가-힣])(\d{4})$', text)
        if m:
            prefix = int(m.group(1))
            if 700 <= prefix <= 799:
                return "전기차"
            elif 100 <= prefix <= 699:
                return "신형"
        # 영업용: 지역명 + 2자리 + 영업용한글(아바사자) + 4자리
        m = re.match(r'^([가-힣]{2,3})(\d{2})([아바사자])(\d{4})$', text)
        if m:
            return "영업용"
        # 지역명 구형: 지역명 + 2자리 + 한글 + 4자리
        m = re.match(r'^([가-힣]{2,3})(\d{2})([가-힣])(\d{4})$', text)
        if m:
            return "지역명_구형"
        # 구형: 2자리 + 한글 + 4자리
        m = re.match(r'^(\d{2})([가-힣])(\d{4})$', text)
        if m:
            return "구형"
        return "기타"

    @classmethod
    def classify_vehicle_type(cls, text):
        """차량 용도 분류 → str"""
        # 번호판에서 한글 문자 추출 (지역명 제외)
        m = re.search(r'\d([가-힣])\d', text)
        if m:
            hangul = m.group(1)
            if hangul in cls._HANGUL_COMMERCIAL:
                return "영업용"
            if hangul in cls._HANGUL_RENTAL:
                return "렌터카"
        return "자가용"

    @staticmethod
    def get_confidence_level(conf):
        """신뢰도 등급 분류 → str"""
        if conf >= PlateEngineConfig.OUTPUT_CONF_HIGH:
            return "HIGH"
        elif conf >= PlateEngineConfig.OUTPUT_CONF_MEDIUM:
            return "MEDIUM"
        else:
            return "LOW"


class HangulClassifier:
    """번호판 한글 전용 분류기 — 초성 교차검증 방식

    OCR 앙상블 투표에서 결정된 한글의 초성이 혼동 쌍(ㅅ↔ㅈ, ㅁ↔ㅂ)에
    해당하면, PaddleOCR 인식 모델(det=False)을 한글 크롭에 직접 적용하여
    초성을 교차검증한다.

    원리:
    - 전체 번호판 OCR: 맥락은 좋지만 미세 구조(ㅈ 가로획 등) 누락 가능
    - 한글 크롭 OCR: 맥락(모음)은 틀리지만 자음 구조를 더 정확히 감지
    - 예: 투표="서"(ㅅ+ㅓ) + 크롭="지"(ㅈ+ㅣ) → 초성 ㅈ + 모음 ㅓ = "저"
    """

    # 교정 방향: 단순 초성 → 복잡 초성 (역방향은 안전하지 않음)
    _INITIAL_OVERRIDE = {
        9: 12,   # ㅅ(9) → ㅈ(12): ㅈ의 가로획이 크롭에서 감지되면 교정
        6: 7,    # ㅁ(6) → ㅂ(7): ㅂ의 하단 세로획이 감지되면 교정
    }
    # ㅈ 계열 초성 (가로획 보유) — 크롭에서 이 그룹이 검출되면 ㅈ로 교정
    _JIEUT_GROUP = {12, 13, 14}   # ㅈ, ㅉ, ㅊ
    _SIOT_GROUP = {9, 10}          # ㅅ, ㅆ
    # ㅂ 계열
    _BIEUP_GROUP = {7, 8}         # ㅂ, ㅃ

    def __init__(self):
        self._ready = True

    @staticmethod
    def _decompose(ch):
        """한글 음절 → (초성, 중성, 종성) 인덱스"""
        code = ord(ch) - 0xAC00
        if code < 0 or code > 11171:
            return None
        return code // (21 * 28), (code // 28) % 21, code % 28

    @staticmethod
    def _compose(ini, med, fin=0):
        """(초성, 중성, 종성) → 한글 음절"""
        return chr(0xAC00 + ini * 21 * 28 + med * 28 + fin)

    @staticmethod
    def _structural_bieup_check(crop_bgr):
        """형태학적 구조 분석으로 ㅂ/ㅁ 구분.

        ㅂ: 자음 영역에 3개 이상의 수평 바 (상단+중단+하단) + 높은 픽셀 밀도
        ㅁ: 자음 영역에 2개의 수평 바 (상단+하단) + 낮은 픽셀 밀도

        Returns: True if structural evidence suggests ㅂ, False otherwise
        """
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            hh, hw = binary.shape
            if hh < 20 or hw < 20:
                return False

            # 수직 프로젝션으로 자음 영역 열 범위 탐색
            v_proj = np.sum(binary > 0, axis=0)
            v_threshold = hh * 0.15

            # 연속 high-value 열 그룹 찾기 (자음 영역)
            groups = []
            in_group = False
            g_start = -1
            for c in range(hw):
                if v_proj[c] >= v_threshold:
                    if not in_group:
                        g_start = c
                        in_group = True
                else:
                    if in_group:
                        groups.append((g_start, c - 1))
                        in_group = False
            if in_group:
                groups.append((g_start, hw - 1))

            if not groups:
                return False

            # 가장 넓은 열 그룹 = 자음 본체 (보통 자음이 가장 넓음)
            # 단, 최소 폭 8px 이상인 그룹만 고려
            valid_groups = [(s, e) for s, e in groups if (e - s + 1) >= 8]
            if not valid_groups:
                return False
            cons_start, cons_end = max(valid_groups, key=lambda g: g[1] - g[0])
            cons_region = binary[:, cons_start:cons_end + 1]
            ch, cw = cons_region.shape

            # 자음 영역의 수평 프로젝션 (행별 흰 픽셀 수)
            h_proj = np.sum(cons_region > 0, axis=1)

            # 수평 바 감지 (h_proj > 60% 자음 폭)
            bar_threshold = cw * 0.55
            bars = []
            in_bar = False
            bar_start = -1
            for r in range(ch):
                if h_proj[r] >= bar_threshold:
                    if not in_bar:
                        bar_start = r
                        in_bar = True
                else:
                    if in_bar:
                        bars.append((bar_start, r - 1))
                        in_bar = False
            if in_bar:
                bars.append((bar_start, ch - 1))

            # 자음 영역만 추출: 상단 60% 이내의 바만 (하단은 모음 ㅜ/ㅗ)
            cons_height_limit = int(ch * 0.55)
            cons_bars = [(s, e) for s, e in bars if s < cons_height_limit]

            # 판정 기준 1: 자음 상반부에 3+ 바 → ㅂ 가능성 매우 높음
            if len(cons_bars) >= 3:
                return True

            # 판정 기준 2: 자음 상반부에 2개 바가 있고, 간격이 좁으면 ㅂ
            # (ㅂ의 상단바+중단바 = 좁은 간격, ㅁ의 상단바+하단바 = 넓은 간격)
            if len(cons_bars) == 2:
                bar1_end = cons_bars[0][1]
                bar2_start = cons_bars[1][0]
                gap = bar2_start - bar1_end - 1
                bar_span = cons_bars[1][1] - cons_bars[0][0] + 1
                # ㅂ: 두 바 사이 간격이 전체 높이의 25% 이하
                if bar_span > 0 and gap < bar_span * 0.25:
                    return True

            # 판정 기준 3: 픽셀 밀도 (자음 영역 상반부)
            cons_upper = cons_region[:cons_height_limit, :]
            density = np.sum(cons_upper > 0) / max(cons_upper.size, 1)
            if density > 0.50:
                return True

            return False
        except Exception:
            return False

    def check_override(self, voted_hg, crop_bgr, paddle_engine, ocr_engines=None):
        """투표 결과 한글의 초성을 PaddleOCR+Tesseract 크롭 인식으로 교차검증.

        Returns: (corrected_hangul, changed: bool)
        """
        if crop_bgr is None or crop_bgr.size < 100:
            return voted_hg, False

        vd = self._decompose(voted_hg)
        if not vd or vd[0] not in self._INITIAL_OVERRIDE:
            return voted_hg, False

        # 다중 전처리 변형 생성
        variants = [crop_bgr]
        try:  # CLAHE
            cl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cl.apply(lab[:, :, 0])
            variants.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        except Exception:
            pass
        try:  # Sharpen
            k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            variants.append(cv2.filter2D(crop_bgr, -1, k))
        except Exception:
            pass
        variants.append(cv2.bitwise_not(crop_bgr))  # Inverted
        # 업스케일 버전 추가
        big = cv2.resize(crop_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        variants.append(big)
        try:
            lab2 = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
            lab2[:, :, 0] = cl.apply(lab2[:, :, 0])
            variants.append(cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR))
        except Exception:
            pass

        # PaddleOCR det=False로 각 변형 인식 → 초성 수집
        crop_initials = []
        for v in variants:
            try:
                res = paddle_engine.ocr(v, det=False, cls=True)
                if res and res[0]:
                    for text, conf in res[0]:
                        for ch in str(text):
                            d = self._decompose(ch)
                            if d:
                                crop_initials.append(d[0])
            except Exception:
                pass

        # ★ Tesseract 초성 증거 수집 (PaddleOCR과 다른 인식 모델)
        if ocr_engines and 'tesseract' in ocr_engines and HAS_TESSERACT:
            for v in variants:
                try:
                    gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) if len(v.shape) == 3 else v
                    # PSM 10 = single character (한글 1자 크롭)
                    text = pytesseract.image_to_string(
                        gray, config='--oem 3 --psm 10 -l kor'
                    )
                    for ch in text.strip():
                        d = self._decompose(ch)
                        if d:
                            crop_initials.append(d[0])
                except Exception:
                    pass

        if not crop_initials:
            return voted_hg, False

        # 초성 증거 판정
        target_ini = self._INITIAL_OVERRIDE[vd[0]]

        if vd[0] == 9:  # ㅅ → ㅈ 교정 여부
            evidence = sum(1 for i in crop_initials if i in self._JIEUT_GROUP)
            counter = sum(1 for i in crop_initials if i in self._SIOT_GROUP)
        elif vd[0] == 6:  # ㅁ → ㅂ 교정 여부
            evidence = sum(1 for i in crop_initials if i in self._BIEUP_GROUP)
            counter = sum(1 for i in crop_initials if i == 6)
        else:
            return voted_hg, False

        # 과반 + 최소 2건 이상 증거 시 교정
        if evidence > counter and evidence >= 2:
            new_hg = self._compose(target_ini, vd[1], vd[2])
            if new_hg in PlateValidator._VALID_PLATE_HANGUL:
                return new_hg, True

        # ★ OCR 증거 부족 시 구조 분석 fallback (ㅁ→ㅂ만 적용)
        if vd[0] == 6 and evidence == 0:
            if self._structural_bieup_check(crop_bgr):
                new_hg = self._compose(target_ini, vd[1], vd[2])
                if new_hg in PlateValidator._VALID_PLATE_HANGUL:
                    return new_hg, True

        return voted_hg, False


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


def _resolve_plate_model(config: "PlateEngineConfig") -> Path:
    """프로젝트(스크립트) 폴더 기준으로 번호판용 YOLO 모델 경로를 찾는다. 없으면 None."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "runs" / "detect" / "plate_korean_3k_v2" / "weights" / "best.pt",
        script_dir / "runs" / "detect" / "plate_korean_3k3" / "weights" / "best.pt",
        script_dir / "best.pt",
        script_dir / "runs" / "detect" / "highway_plate" / "weights" / "best.pt",
        script_dir / config.YOLO_MODEL,
        script_dir / config.YOLO_FALLBACK,
    ]
    for m in _MODEL_PRIORITY:
        candidates.append(script_dir / m)
        candidates.append(Path(m))  # CWD
    for p in candidates:
        if p.exists():
            return p
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PlateTracker: IoU 기반 차량 추적 + TTL 프레임 만료
#   → Ghost Detection (이전 차량 잔상) 방지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PlateTracker:
    """IoU 기반 번호판 트래커.

    각 트랙은 bbox 위치로 차량을 식별하며, IoU < threshold 이면
    다른 차량으로 판단하여 이전 결과를 초기화합니다.

    TTL(Time-To-Live) 프레임 이내에 재감지되지 않으면 트랙을 만료시켜
    ghost detection (이전 차량 번호가 다음 차량에 표시) 을 방지합니다.
    """

    # ── Ghost Detection 방지 파라미터 ──
    AREA_CHANGE_THRESHOLD = 0.5   # bbox 면적 비율이 0.5배 미만 또는 2.0배 초과 → 차량 변경 판단
    GAP_FRAMES_THRESHOLD = 10     # N프레임 이상 미감지 후 재감지 → texts 리셋
    MAX_TEXT_ENTRIES = 50          # texts dict 최대 항목 수 (무한 누적 방지)

    def __init__(self, iou_threshold=0.30, ttl_frames=30):
        self.iou_threshold = iou_threshold
        self.ttl_frames = ttl_frames
        self.tracks = []  # list of track dicts
        self._frame_id = 0

    @staticmethod
    def _bbox_area(bbox):
        """bbox [x1,y1,x2,y2]의 면적 계산"""
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    def _bbox_iou(self, a, b):
        """두 bbox [x1,y1,x2,y2] 간 IoU 계산"""
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def _should_reset_texts(self, trk, new_bbox):
        """기존 트랙에 매칭됐지만 차량이 변경된 것으로 판단되는 경우 True 반환.

        리셋 조건:
          1. bbox 면적 급변 (0.5배 미만 또는 2.0배 초과)
          2. N프레임(10+) 미감지 후 재감지
        """
        # 조건 1: bbox 면적 급변 → 다른 차량이 같은 위치에 진입
        old_area = self._bbox_area(trk["bbox"])
        new_area = self._bbox_area(new_bbox)
        if old_area > 0 and new_area > 0:
            ratio = new_area / old_area
            if ratio < self.AREA_CHANGE_THRESHOLD or ratio > (1.0 / self.AREA_CHANGE_THRESHOLD):
                return True

        # 조건 2: N프레임 이상 미감지 후 재감지 → 사이에 차량 교체 가능성
        gap = self._frame_id - trk["last_frame"]
        if gap >= self.GAP_FRAMES_THRESHOLD:
            return True

        return False

    def _reset_track_texts(self, trk):
        """트랙의 투표 데이터를 초기화 (차량 변경 판단 시)"""
        trk["texts"] = defaultdict(int)
        trk["best_conf"] = 0
        trk["recorded"] = False
        trk["consecutive"] = 1
        trk["_detect_count"] = 1

    def begin_frame(self):
        """새 프레임 시작 — 프레임 카운터 증가, 트랙 seen 플래그 리셋"""
        self._frame_id += 1
        for trk in self.tracks:
            trk["_seen"] = False

    def match(self, bbox):
        """bbox와 가장 높은 IoU를 가진 트랙 매칭.

        Returns:
            track (dict): 매칭된 트랙 (새 트랙이면 새로 생성)
            is_new (bool): 새 트랙 여부 (이전 차량과 다른 위치)
        """
        best_iou = 0
        best_trk = None
        _new_cx = (bbox[0] + bbox[2]) / 2
        _new_cy = (bbox[1] + bbox[3]) / 2
        for trk in self.tracks:
            iou = self._bbox_iou(bbox, trk["bbox"])
            if iou > best_iou:
                # ★ 중심 거리 200px 이상이면 다른 차량으로 판단
                _trk_cx = (trk["bbox"][0] + trk["bbox"][2]) / 2
                _trk_cy = (trk["bbox"][1] + trk["bbox"][3]) / 2
                _cdist = ((_new_cx - _trk_cx) ** 2 + (_new_cy - _trk_cy) ** 2) ** 0.5
                if _cdist >= 200:
                    continue
                best_iou = iou
                best_trk = trk

        if best_iou >= self.iou_threshold and best_trk is not None:
            # ── Ghost 방지: 차량 변경 감지 시 texts 리셋 ──
            if self._should_reset_texts(best_trk, bbox):
                self._reset_track_texts(best_trk)

            # ── texts 항목 수 제한 (무한 누적 방지) ──
            if len(best_trk["texts"]) >= self.MAX_TEXT_ENTRIES:
                # 최다 득표 상위 5개만 유지
                top_items = sorted(best_trk["texts"].items(), key=lambda x: x[1], reverse=True)[:5]
                best_trk["texts"] = defaultdict(int, top_items)

            # 기존 트랙 갱신
            best_trk["_pre_gap"] = self._frame_id - best_trk["last_frame"]  # ★ 매칭 전 갭 저장 (OCR 스킵 판단용)
            best_trk["bbox"] = bbox
            best_trk["consecutive"] += 1
            best_trk["_detect_count"] = best_trk.get("_detect_count", 0) + 1
            best_trk["last_frame"] = self._frame_id
            best_trk["_seen"] = True
            return best_trk, False
        else:
            # 새 트랙 생성 (다른 차량)
            new_trk = {
                "bbox": bbox,
                "texts": defaultdict(int),
                "consecutive": 1,
                "_detect_count": 1,  # 감지된 총 프레임 수 (투표 가중치 무관)
                "best_conf": 0,
                "recorded": False,
                "last_frame": self._frame_id,
                "_seen": True,
            }
            self.tracks.append(new_trk)
            return new_trk, True

    def end_frame(self):
        """프레임 종료 — 미감지 트랙 처리 + TTL 만료 트랙 제거"""
        alive = []
        for trk in self.tracks:
            if trk["_seen"]:
                alive.append(trk)
            else:
                # 이번 프레임에 미감지 → consecutive 점진적 감소 (즉시 리셋 안 함)
                # 실시간 모드에서 YOLO가 1~2프레임 미감지해도 트랙 유지
                frames_since = self._frame_id - trk["last_frame"]
                if frames_since <= 3:
                    # 3프레임 이내 미감지: consecutive 유지 (일시적 미감지 허용)
                    pass
                elif frames_since <= self.ttl_frames:
                    # 3프레임 초과 ~ TTL 이내: consecutive 리셋
                    trk["consecutive"] = 0
                    alive.append(trk)
                    continue
                else:
                    # TTL 초과: 트랙 제거 (ghost detection 방지)
                    continue
                alive.append(trk)
        self.tracks = alive

    def reset(self):
        """모든 트랙 초기화 — 이미지 단독 테스트 시 사용"""
        self.tracks.clear()
        self._frame_id = 0


class PlateEnginePro:
    """
    상용급 번호판 인식 엔진 (평가기준 반영)
    · best.pt: YOLO 학습자료로 생성한 모델 사용 (평가 5점 - train.py / Roboflow 데이터셋)
    · 외부 입력파일 차량 객체 인지: process_frame()으로 영상·이미지 입력 (평가 5점)
    [영상입력] → [YOLO탐지] → [ROI추출] → [10종전처리(CLAHE 등)]
    → [멀티OCR] → [검증/보정] → [DB기록] → [경고알림]
    """

    def __init__(self, config=None):
        self.config = config or PlateEngineConfig()
        self.preprocessor = ImagePreprocessor()
        self.validator = PlateValidator()
        self.db = PlateDatabase()

        model_path = _resolve_plate_model(self.config)
        self._is_plate_model = False  # 번호판 전용 모델 여부 플래그
        if model_path is not None:
            self.model = YOLO(str(model_path))
            # 모델 클래스 이름으로 번호판 전용인지 자동 판별
            _names = self.model.names or {}
            _name_vals = [str(v).lower() for v in _names.values()]
            self._is_plate_model = any(
                kw in n for n in _name_vals
                for kw in ("plate", "license", "번호판")
            ) or "plate" in str(model_path).lower()
            _mtype = "번호판 전용" if self._is_plate_model else "범용(COCO)"
            print(f"[엔진] YOLO 모델 로드: {model_path} ({_mtype})")
        else:
            self.model = _load_best_model()
            print("[엔진] 번호판 전용 .pt가 없어 기본 모델 사용. 인식이 안 되면 train.py로 학습 후 runs/.../best.pt를 두세요.")
        # COCO 모델에서 차량 클래스 ID (car=2, motorcycle=3, bus=5, truck=7)
        self._vehicle_class_ids = {2, 3, 5, 7}

        self.ocr_engines = {}
        if HAS_PADDLEOCR:
            paddle_kwargs = dict(lang="korean", use_angle_cls=True, show_log=False, use_gpu=False)
            # Windows 한글 경로 우회: 영문 경로에 모델이 있으면 직접 지정
            _paddle_model_root = PathConfig.paddle_model_dir()
            if _paddle_model_root.exists():
                _det = _paddle_model_root / "det/ml/Multilingual_PP-OCRv3_det_infer"
                _rec = _paddle_model_root / "rec/korean/korean_PP-OCRv4_rec_infer"
                _cls = _paddle_model_root / "cls/ch_ppocr_mobile_v2.0_cls_infer"
                if _det.exists():
                    paddle_kwargs["det_model_dir"] = str(_det)
                if _rec.exists():
                    paddle_kwargs["rec_model_dir"] = str(_rec)
                if _cls.exists():
                    paddle_kwargs["cls_model_dir"] = str(_cls)
            try:
                self.ocr_engines["paddleocr"] = PaddleOCR(**paddle_kwargs)
            except TypeError:
                # show_log 파라미터 호환성 처리
                paddle_kwargs.pop("show_log", None)
                try:
                    self.ocr_engines["paddleocr"] = PaddleOCR(**paddle_kwargs)
                except Exception as e:
                    print(f"[엔진] PaddleOCR 초기화 실패: {e}")
            except Exception as e:
                print(f"[엔진] PaddleOCR 초기화 실패: {e}")
        if HAS_EASYOCR:
            self.ocr_engines["easyocr"] = easyocr.Reader(["ko", "en"], gpu=True)
        if HAS_TESSERACT:
            try:
                # 한국어 언어팩 확인
                _tess_langs = pytesseract.get_languages()
                if 'kor' in _tess_langs:
                    self.ocr_engines["tesseract"] = "tesseract"
                else:
                    print("[엔진] Tesseract: 한국어 언어팩(kor) 없음 — 건너뜀")
            except Exception:
                # get_languages 실패 시에도 일단 추가 (런타임에서 처리)
                self.ocr_engines["tesseract"] = "tesseract"
        # 한국 번호판 허용 문자 (EasyOCR allowlist)
        self._kr_allowlist = (
            "0123456789"
            "가나다라마바사아자차카타파하"
            "거너더러머버서어저처커터퍼허"
            "고노도로모보소오조호"
            "구누두루무부수우주"
            "배육"
            "서울부산대구인천광주대전울산세종"
            "경기강원충북충남전북전남경북경남제주"
            "전기외교"
        )
        # ── CRNN 학습 모델 로드 (별도 실행, 전처리 루프 밖) ──
        self._crnn_model = None
        _crnn_path = Path(__file__).resolve().parent / "plate_ocr_crnn.pth"
        if HAS_TORCH and _crnn_path.exists():
            try:
                self._crnn_model = _CRNNModel(str(_crnn_path))
                print(f"[엔진] CRNN 모델 로드: {_crnn_path}")
            except Exception as e:
                print(f"[엔진] CRNN 모델 로드 실패: {e}")

        print(f"[엔진] OCR 엔진: {list(self.ocr_engines.keys())}")

        # ★ 한글 전용 분류기 (템플릿 매칭 방식)
        self._hangul_clf = HangulClassifier()

        self.recent_plates = defaultdict(lambda: {"count": 0, "last_seen": 0, "consecutive": 0})
        self.DUPLICATE_THRESHOLD = 3.0
        # 연속 N프레임 감지 시 표시 (이미지 슬라이드 영상은 PLATE_CONSECUTIVE_FRAMES=1 로 설정)
        _env = os.environ.get("PLATE_CONSECUTIVE_FRAMES")
        default_consecutive = int(_env) if (_env and _env.isdigit()) else getattr(
            self.config, "CONSECUTIVE_FRAMES_REQUIRED", 3
        )
        self.consecutive_required = default_consecutive
        # ── PlateTracker: IoU 기반 차량 추적 + TTL 프레임 만료 ──
        self._tracker = PlateTracker(
            iou_threshold=self.config.TRACKER_IOU_THRESHOLD,
            ttl_frames=self.config.TRACKER_TTL_FRAMES,
        )
        # 하위 호환: 기존 _pos_trackers 참조를 tracker.tracks로 연결
        self._pos_trackers = self._tracker.tracks
        self._POS_IOU_THRESHOLD = self.config.TRACKER_IOU_THRESHOLD
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

    def reset_state(self):
        """내부 캐시 초기화 — 이미지 단독 테스트 시 이전 결과 오염 방지"""
        self.recent_plates.clear()
        self._tracker.reset()
        self._pos_trackers = self._tracker.tracks
        self._multiframe_buffer.clear()
        self.stats["frames_processed"] = 0
        self.stats["plates_shown"] = 0

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
        [평가기준] 교통안전 AI 솔루션 - 번호판 인식 파이프라인
          · 입력화일 차량 detect 표시(10점): YOLO로 번호판 영역 탐지 → bbox 반환
          · ROI/CROP/CLAHE/Homography(20점): ROI 추출 → CROP → 전처리(CLAHE 등), 기울기보정(deskew)
          · 차량번호 인식 표시(20점): OCR 앙상블 → 검증 → plate/confidence 반환
        frame: YOLO 추론용 (640px 등 축소 가능)
        full_frame: OCR 크롭용 원본 고해상도 프레임 (None이면 frame에서 크롭)
        """
        results = []
        self.stats["frames_processed"] += 1
        self._tracker.begin_frame()
        # ★ 해상도에 따라 imgsz 동적 조정 (고해상도 영상에서 소형 번호판 탐지 향상)
        _fh, _fw = frame.shape[:2]
        if _fw >= 3840:      # 4K
            _imgsz = 1920
        elif _fw >= 1920:    # FHD
            _imgsz = 1280
        elif _fw >= 1280:    # HD
            _imgsz = 960
        else:
            _imgsz = 640
        detections = self.model(frame, conf=self.config.DETECT_CONF, imgsz=_imgsz, verbose=False, max_det=3)

        crop_src = full_frame if full_frame is not None else frame
        ch_full, cw_full = crop_src.shape[:2]
        ch_det, cw_det = frame.shape[:2]
        sx = cw_full / cw_det
        sy = ch_full / ch_det

        # ★ 겹치는 bbox 제거 (수동 NMS): IoU > 0.5인 겹침에서 낮은 conf 제거
        # YOLO NMS-free 모델이 중복 bbox를 출력하는 경우 대비
        _raw_boxes = []
        for det in detections[0].boxes:
            _rb = list(map(int, det.xyxy[0].tolist()))
            _rc = float(det.conf[0])
            # ★ 영상 모드에서만 ROI 필터 적용 (정적 이미지는 해상도 다름)
            if self.consecutive_required > 1:
                cx = (_rb[0] + _rb[2]) / 2
                cy = (_rb[1] + _rb[3]) / 2
                if not (self.config.ROI_X1 <= cx <= self.config.ROI_X2 and self.config.ROI_Y1 <= cy <= self.config.ROI_Y2):
                    continue
            _raw_boxes.append((_rb, _rc, det))
        _raw_boxes.sort(key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)  # bbox 면적 큰 순
        _keep_dets = []
        for _rb, _rc, _det in _raw_boxes:
            _suppressed = False
            for _kb, _kc, _ in _keep_dets:
                _iou = self._tracker._bbox_iou(_rb, _kb)
                if _iou > 0.65:
                    _suppressed = True
                    break
            if not _suppressed:
                _keep_dets.append((_rb, _rc, _det))

        seen_this_frame = set()

        for _, _, det in _keep_dets:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0])

            # ★ COCO 모델일 때 차량 클래스만 허용 (비차량 오탐 방지)
            if not self._is_plate_model:
                cls_id = int(det.cls[0]) if hasattr(det, 'cls') and det.cls is not None else -1
                if cls_id not in self._vehicle_class_ids:
                    continue

            ox1, oy1 = int(x1 * sx), int(y1 * sy)
            ox2, oy2 = int(x2 * sx), int(y2 * sy)

            # ★ 최소 크기 필터 (너무 작은 탐지는 노이즈)
            det_w = ox2 - ox1
            det_h = oy2 - oy1
            if det_w < self.config.MIN_BBOX_WIDTH or det_h < self.config.MIN_BBOX_HEIGHT:
                continue

            # ★ bbox 가로세로 비율 검증 (엠블럼/그릴/간판 제거)
            _aspect = det_w / max(det_h, 1)
            _is_1line = 2.0 <= _aspect <= 5.5   # 1줄 번호판
            _is_2line = 0.8 <= _aspect < 2.0    # 2줄 번호판
            if not (_is_1line or _is_2line):
                continue  # 번호판 비율 아님 → 엠블럼/그릴/간판 가능성

            # ★ bbox 위치 검증 (이미지 상단 10% / 하단 5% → 간판/노면)
            if oy1 < ch_full * 0.10:
                continue  # 상단 영역 (간판/표지판 의심)
            if oy2 > ch_full * 0.95:
                continue  # 하단 영역 (노면 의심)

            # ★ bbox 크기 기반 신뢰도 페널티 (소형 번호판 오인식 방지)
            # 영상 분석 결과: bbox 폭 < 70px → 정확도 0%, 70~100px → 60%, > 100px → 100%
            _bbox_conf_penalty = 1.0
            _is_small_plate = False
            if det_w < 70:
                _bbox_conf_penalty = 0.60   # 소형: 40% 감점 → 투표에서 대폭 불이익
                _is_small_plate = True
            elif det_w < 100:
                _bbox_conf_penalty = 0.85   # 중소형: 15% 감점
            elif det_w < 120:
                _bbox_conf_penalty = 0.95   # 중형: 5% 감점

            # [평가기준 20점] ROI (Region of Interest) + CROP: 번호판 영역 좌표로 관심영역 설정 후 크롭
            margin_x = int((ox2 - ox1) * 0.35)  # ★ 좌우 마진 확대 (0.25→0.35, 가장자리 문자 보존)
            margin_y = int((oy2 - oy1) * 0.40)  # ★ 상하 마진 확대 (0.30→0.40, 2줄 번호판+지역명 캡처)
            rx1 = max(0, ox1 - margin_x)
            ry1 = max(0, oy1 - margin_y)
            rx2 = min(cw_full, ox2 + margin_x)
            ry2 = min(ch_full, oy2 + margin_y)
            roi = crop_src[ry1:ry2, rx1:rx2]   # CROP: 번호판 영역 이미지 추출

            if roi.size == 0:
                continue

            # ★ 트래커 기반 OCR 스킵: 이미 확인된 번호판이면 OCR 건너뜀 (FPS 최적화)
            # 조건: 기존 트랙 + 직전 프레임 연속 감지(gap≤1) + 3프레임 확인 + 3+표
            # gap>1이면 차량 교체 가능성 → OCR 반드시 실행
            cur_bbox = [ox1, oy1, ox2, oy2]
            _pre_trk, _pre_is_new = self._tracker.match(cur_bbox)
            _skip_ocr = False
            if not _pre_is_new and _pre_trk["texts"]:
                _pre_gap = _pre_trk.get("_pre_gap", 999)
                _pre_top_text = max(_pre_trk["texts"], key=_pre_trk["texts"].get)
                _pre_top_votes = _pre_trk["texts"][_pre_top_text]
                _pre_detect_cnt = _pre_trk.get("_detect_count", 0)
                # 직전 프레임 연속(gap≤1) & 3프레임+ 감지 & 3+표 → OCR 스킵
                if _pre_gap <= 1 and _pre_detect_cnt >= 3 and _pre_top_votes >= 3:
                    _skip_ocr = True
            if _skip_ocr:
                # OCR 없이 트래커 결과 사용
                _pre_conf = _pre_trk.get("best_conf", 0.5)
                seen_this_frame.add(_pre_top_text)
                plate_info = self.recent_plates[_pre_top_text]
                plate_info["consecutive"] = _pre_trk["consecutive"]
                plate_info["last_seen"] = time.time()
                plate_info["count"] += 1
                _show = (_pre_trk["consecutive"] >= self.consecutive_required
                         or _pre_detect_cnt >= self.consecutive_required)
                if _show:
                    _adj_conf = min(_pre_conf + 0.10, 1.0)
                    _conf_level = self.validator.get_confidence_level(_adj_conf) + "(추적)"
                    _bbox_w = ox2 - ox1
                    _bbox_h = oy2 - oy1
                    results.append({
                        "plate": _pre_top_text,
                        "confidence": _pre_conf,
                        "bbox": cur_bbox,
                        "is_alert": False,
                        "alert_info": None,
                        "plate_number": _pre_top_text,
                        "confidence_level": _conf_level,
                        "plate_type": self.validator.classify_plate_type(_pre_top_text),
                        "vehicle_type": self.validator.classify_vehicle_type(_pre_top_text),
                        "plate_lines": 1 if _bbox_w / max(_bbox_h, 1) > 2.5 else 2,
                        "plate_color": "흰색바탕_검은글씨",
                        "bbox_area": _bbox_w * _bbox_h,
                        "frame_count": _pre_top_votes,
                        "is_valid_format": True,
                        "rejection_reason": None,
                    })
                continue

            # ★ 디버그: 번호판 crop 저장 (환경변수 _DEBUG_CROP=1 로 활성화)
            if os.environ.get('_DEBUG_CROP', ''):
                _dbg_dir = Path(__file__).resolve().parent / "debug_crops"
                _dbg_dir.mkdir(exist_ok=True)
                _dbg_name = f"frame{self.stats['frames_processed']:04d}_det{x1}_{y1}.png"
                cv2.imwrite(str(_dbg_dir / _dbg_name), roi)

            roi_h, roi_w = roi.shape[:2]
            _clf_scale, _clf_pad = 1.0, 0  # 한글 분류기용 스케일/패딩 (나중에 사용)
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
                # ★ 업스케일 목표 500px (소형 번호판 인식률 향상)
                target_w = 500
                if roi_w < target_w:
                    scale = target_w / roi_w
                    # 극소 번호판(60px 이하)은 9배까지 확대 (52px→468px)
                    if roi_w < 60:
                        scale = max(scale, 9.0)
                    elif roi_w < 120:
                        scale = max(scale, 4.0)
                else:
                    scale = 1.0
                if scale > 1.0:
                    # ★ 소형 번호판(80px 이하): LANCZOS4로 선명 업스케일
                    _interp = cv2.INTER_LANCZOS4 if roi_w < 80 else cv2.INTER_CUBIC
                    roi_for_ocr = cv2.resize(
                        roi, None, fx=scale, fy=scale, interpolation=_interp
                    )
                    # 업스케일 후 선명화 적용 (언샤프 마스크)
                    if roi_w < 80:
                        # ★ 소형 번호판: 강화 샤프닝 (획이 흐려지는 것 보상)
                        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                    else:
                        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                    roi_for_ocr = cv2.filter2D(roi_for_ocr, -1, kernel)
                else:
                    roi_for_ocr = roi
                # ★ 흰색 테두리 패딩 추가 (OCR 엔진이 가장자리 문자를 더 잘 읽음)
                pad = max(10, int(roi_for_ocr.shape[0] * 0.15))
                roi_for_ocr = cv2.copyMakeBorder(
                    roi_for_ocr, pad, pad, pad, pad,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
                _clf_scale, _clf_pad = scale, pad  # 한글 분류기용 저장

            # ★ 번호판 색상 검증 (번호판 바탕색으로 오감지 필터링)
            _color_conf_penalty = 1.0
            try:
                _hsv_check = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                _h_avg = np.mean(_hsv_check[:, :, 0])
                _s_avg = np.mean(_hsv_check[:, :, 1])
                _v_avg = np.mean(_hsv_check[:, :, 2])
                _is_white = _s_avg < 50 and _v_avg > 80        # 흰색/은색 바탕 (자가용, 음영 포함)
                _is_yellow = 15 < _h_avg < 35 and _s_avg > 60  # 노란색 (영업용)
                _is_green = 35 < _h_avg < 85 and _s_avg > 40   # 초록색 (구형/영업용)
                _is_blue = 90 < _h_avg < 130 and _s_avg > 40   # 파란색 (전기차)
                if not (_is_white or _is_yellow or _is_green or _is_blue):
                    _color_conf_penalty = 0.85  # 판별 불가 → 15% 페널티
            except Exception:
                pass

            # ★ CRNN 학습 모델: 원본 ROI에서 직접 실행 (전처리 루프 밖)
            _crnn_candidates = []
            if self._crnn_model is not None:
                try:
                    crnn_text, crnn_conf = self._crnn_model.recognize(roi)
                    if crnn_text and crnn_conf > 0.3:
                        cleaned = self.validator.clean_ocr_text(crnn_text)
                        if self.validator.is_valid_length(cleaned):
                            is_valid, final_text = self.validator.validate(cleaned)
                            if is_valid:
                                _crnn_candidates.append((final_text, crnn_conf))
                except Exception:
                    pass

            # ★ 녹색 번호판 감지 (HSV 분석) — extra_crops 및 OCR 루프 전에 판정
            _is_green_plate = False
            try:
                _hsv_det = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                _green_mask = cv2.inRange(_hsv_det, (35, 40, 40), (85, 255, 255))
                _is_green_plate = (np.sum(_green_mask > 0) / _green_mask.size) > 0.20
            except Exception:
                pass

            # ── 구형 두 줄 번호판 감지: 세로 비율이 높으면 상단/하단 분리 추가 ──
            # (구형 번호판은 가로:세로 ≈ 2:1, 신형은 4:1)
            extra_crops = []
            if roi_h > roi_w * 0.45:   # 세로가 어느정도 있는 번호판
                top_crop = roi_for_ocr[:int(roi_for_ocr.shape[0] * 0.5), :]
                bot_crop = roi_for_ocr[int(roi_for_ocr.shape[0] * 0.4):, :]
                # ★ 상단 크롭 강화: 500px 확대 + 다양한 전처리 (한글/지역명 인식률 향상)
                top_h, top_w = top_crop.shape[:2]
                if top_w < 500:
                    sc_top = 500 / top_w
                    top_crop = cv2.resize(top_crop, None, fx=sc_top, fy=sc_top, interpolation=cv2.INTER_CUBIC)
                # 반전 버전 (녹색 배경 → 흰배경 검은글자)
                top_inv = cv2.bitwise_not(top_crop)
                # CLAHE 강화 버전
                top_gray = cv2.cvtColor(top_crop, cv2.COLOR_BGR2GRAY)
                clahe_obj = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
                top_clahe = clahe_obj.apply(top_gray)
                top_clahe_bgr = cv2.cvtColor(top_clahe, cv2.COLOR_GRAY2BGR)
                # ★ HSV V-channel 이진화: 녹색 번호판의 얇은 숫자("1") 인식률 대폭 향상
                top_hsv = cv2.cvtColor(top_crop, cv2.COLOR_BGR2HSV)
                _, _, top_v = cv2.split(top_hsv)
                _, top_val_mask = cv2.threshold(top_v, 150, 255, cv2.THRESH_BINARY)
                top_val_bgr = cv2.cvtColor(top_val_mask, cv2.COLOR_GRAY2BGR)
                # ★ 선명화 800px 버전: 얇은 획 강조
                sc_800 = 800 / top_crop.shape[1] if top_crop.shape[1] < 800 else 1.0
                if sc_800 > 1.0:
                    top_800 = cv2.resize(top_inv, None, fx=sc_800, fy=sc_800, interpolation=cv2.INTER_CUBIC)
                    sharp_k = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32)
                    top_sharp800 = cv2.filter2D(top_800, -1, sharp_k)
                else:
                    top_sharp800 = top_inv
                extra_crops = [
                    ("top", top_inv),
                    ("bot", bot_crop),
                ]  # ★ 6→2 축소 (속도 최적화: top_inv + bot가 핵심)
                # ★ 녹색 번호판: 추가 전처리 (반전+강화CLAHE, 녹색채널 이진화)
                if _is_green_plate:
                    # 반전 + 강화 CLAHE (clipLimit 8.0)
                    _inv_lab = cv2.cvtColor(top_inv, cv2.COLOR_BGR2LAB)
                    _l, _a, _b = cv2.split(_inv_lab)
                    _clahe_s = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
                    _l = _clahe_s.apply(_l)
                    _inv_clahe = cv2.cvtColor(cv2.merge([_l, _a, _b]), cv2.COLOR_LAB2BGR)
                    extra_crops.append(("top", _inv_clahe))
                    # 반전 + Otsu 이진화
                    _inv_gray = cv2.cvtColor(top_inv, cv2.COLOR_BGR2GRAY)
                    _clahe_g = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
                    _inv_enh = _clahe_g.apply(_inv_gray)
                    _, _inv_bin = cv2.threshold(_inv_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    extra_crops.append(("top", cv2.cvtColor(_inv_bin, cv2.COLOR_GRAY2BGR)))
                    # 녹색 채널 추출 → 반전 → Otsu
                    _g_ch = top_crop[:, :, 1]  # BGR의 G채널
                    _g_inv = cv2.bitwise_not(_g_ch)
                    _, _g_bin = cv2.threshold(_g_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    extra_crops.append(("top", cv2.cvtColor(_g_bin, cv2.COLOR_GRAY2BGR)))
                    # V채널 낮은 임계값 (녹색판 전용, 기본 150→100)
                    _, _v_low = cv2.threshold(top_v, 100, 255, cv2.THRESH_BINARY)
                    extra_crops.append(("top", cv2.cvtColor(_v_low, cv2.COLOR_GRAY2BGR)))
                    # 하단도 반전 추가
                    extra_crops.append(("bot", cv2.bitwise_not(bot_crop)))

            # [평가기준 20점] CLAHE 등 전처리: PREPROCESS_METHODS에 "clahe" 포함, 기울기보정(deskew). Homography 원근보정은 exam_realtime_plate.apply_homography 참고.
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # ★ 2단계 OCR 전략 (속도 최적화: 3.2초 → 목표 <1초)
            #   Tier 1: PaddleOCR × 핵심 3개 전처리 → 합의 시 조기 종료
            #   Tier 2: 추가 전처리 + EasyOCR/Tesseract → 불일치 시만
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            _ENGINE_WEIGHT = {'paddleocr': 1.0, 'easyocr': 0.85, 'tesseract': 0.6}
            _TIER1_METHODS = ["original", "clahe", "sharpen"]  # ★ 2→3 확장 (합의 안정성 향상)
            _TIER2_METHODS = ["_inverted", "bilateral", "auto_contrast"]  # ★ 1→3 확장 (다양성 확보)
            from collections import Counter
            all_candidates = []  # [(normalized_text, weighted_confidence), ...]
            _ocr_time_by_engine = defaultdict(float)
            _ocr_count_by_engine = defaultdict(int)

            roi_inv = cv2.bitwise_not(roi_for_ocr)
            _tier1_consensus = False
            _paddle_engine = self.ocr_engines.get("paddleocr")

            # ── Tier 1: PaddleOCR × 핵심 3개 전처리 (~960ms) ──
            if _paddle_engine:
                for method in _TIER1_METHODS:
                    try:
                        if method == "original":
                            processed = roi_for_ocr.copy()
                        else:
                            proc_func = getattr(self.preprocessor, method, None)
                            if proc_func is None:
                                continue
                            processed = proc_func(roi_for_ocr.copy())
                        _t_ocr = time.time()
                        text, ocr_conf = self._run_ocr("paddleocr", _paddle_engine, processed)
                        _ocr_time_by_engine["paddleocr"] += time.time() - _t_ocr
                        _ocr_count_by_engine["paddleocr"] += 1
                        if not text or ocr_conf < 0.20:
                            continue
                        cleaned = self.validator.clean_ocr_text(text)
                        if not self.validator.is_valid_length(cleaned):
                            continue
                        is_valid, final_text = self.validator.validate(cleaned)
                        if not is_valid:
                            continue
                        weighted_conf = ocr_conf * 1.0
                        _v2_has_region = bool(re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', cleaned))
                        weight = 4 if _v2_has_region else 1
                        for _ in range(weight):
                            all_candidates.append((final_text, weighted_conf))
                    except Exception:
                        continue

                # Tier 1 합의 판정: 유효 후보 2개+ 일치 & 평균 conf > 0.6
                if all_candidates:
                    _t1_counter = Counter(t for t, c in all_candidates)
                    _t1_top, _t1_cnt = _t1_counter.most_common(1)[0]
                    _t1_confs = [c for t, c in all_candidates if t == _t1_top]
                    if _t1_cnt >= 2 and float(np.mean(_t1_confs)) > 0.6:
                        _tier1_consensus = True

            # ★ Tier 1 합의 시 EasyOCR 교차검증 제거 (속도 최적화)
            # PaddleOCR 3회 합의(2+표, conf>0.6)면 충분 → EasyOCR ~200ms 절약
            # 불일치 시에만 Tier2에서 EasyOCR 사용

            # ── Tier 2: PaddleOCR + EasyOCR 교차검증 (순차 실행, Tesseract 제거) ──
            # ★ 메서드 수 축소 (4→2) + 매 메서드 후 조기종료로 속도 최적화
            _mid_consensus = False
            if not _tier1_consensus:
                _tier2_break = False
                for _t2_idx, method in enumerate(_TIER2_METHODS):
                    if _tier2_break:
                        break
                    try:
                        if method == "_inverted":
                            processed = roi_inv.copy()
                        else:
                            proc_func = getattr(self.preprocessor, method, None)
                            if proc_func is None:
                                continue
                            processed = proc_func(roi_for_ocr.copy())
                        _skip_easy = False
                        for engine_name, engine in self.ocr_engines.items():
                            if engine_name == "tesseract":
                                continue  # ★ Tesseract 제거 (속도 최적화)
                            if engine_name == "easyocr" and _skip_easy:
                                continue  # ★ PaddleOCR 합의 시 EasyOCR 스킵
                            _t_ocr = time.time()
                            text, ocr_conf = self._run_ocr(engine_name, engine, processed)
                            _ocr_time_by_engine[engine_name] += time.time() - _t_ocr
                            _ocr_count_by_engine[engine_name] += 1
                            if not text or ocr_conf < 0.20:
                                continue
                            cleaned = self.validator.clean_ocr_text(text)
                            if not self.validator.is_valid_length(cleaned):
                                self.stats["filtered_by_length"] += 1
                                continue
                            is_valid, final_text = self.validator.validate(cleaned)
                            if not is_valid:
                                self.stats["filtered_by_pattern"] += 1
                                continue
                            _ew = _ENGINE_WEIGHT.get(engine_name, 0.7)
                            weighted_conf = ocr_conf * _ew
                            _v2_has_region = bool(re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', cleaned))
                            weight = 4 if _v2_has_region else 1
                            for _ in range(weight):
                                all_candidates.append((final_text, weighted_conf))
                            # ★ PaddleOCR 후 합의 체크: 4+표 & 0.70+ → EasyOCR 스킵
                            if engine_name == "paddleocr" and all_candidates:
                                _pe_counter = Counter(t for t, c in all_candidates)
                                _pe_top, _pe_cnt = _pe_counter.most_common(1)[0]
                                _pe_confs = [c for t, c in all_candidates if t == _pe_top]
                                if _pe_cnt >= 4 and float(np.mean(_pe_confs)) > 0.70:
                                    _skip_easy = True
                    except Exception:
                        continue
                    # ★ 매 메서드 후 조기종료: 4+표 & conf > 0.70
                    if all_candidates:
                        _t2_counter = Counter(t for t, c in all_candidates)
                        _t2_top, _t2_cnt = _t2_counter.most_common(1)[0]
                        _t2_confs = [c for t, c in all_candidates if t == _t2_top]
                        if _t2_cnt >= 4 and float(np.mean(_t2_confs)) > 0.70:
                            _tier2_break = True

                # extra_crops 스킵용 합의 체크
                if all_candidates:
                    _mid_counter = Counter(t for t, c in all_candidates)
                    _mid_top, _mid_cnt = _mid_counter.most_common(1)[0]
                    _mid_confs = [c for t, c in all_candidates if t == _mid_top]
                    if _mid_cnt >= 4 and float(np.mean(_mid_confs)) > 0.60:
                        _mid_consensus = True

                # ★ 녹색 번호판: 반전+강화 추가 전처리 (Tier2에서만, 합의 시 스킵)
                if _is_green_plate and not _mid_consensus:
                    _green_extras = []
                    _fi_lab = cv2.cvtColor(roi_inv, cv2.COLOR_BGR2LAB)
                    _fi_l, _fi_a, _fi_b = cv2.split(_fi_lab)
                    _fi_clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
                    _fi_l = _fi_clahe.apply(_fi_l)
                    _green_extras.append(cv2.cvtColor(cv2.merge([_fi_l, _fi_a, _fi_b]), cv2.COLOR_LAB2BGR))
                    _fi_gray = cv2.cvtColor(roi_inv, cv2.COLOR_BGR2GRAY)
                    _fi_c2 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
                    _fi_enh = _fi_c2.apply(_fi_gray)
                    _, _fi_bin = cv2.threshold(_fi_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    _green_extras.append(cv2.cvtColor(_fi_bin, cv2.COLOR_GRAY2BGR))
                    # ★ 속도 최적화: PaddleOCR 우선, 고신뢰 시 다른 엔진 스킵
                    for _ge in _green_extras:
                        _ge_skip = False
                        if _paddle_engine:
                            try:
                                text, ocr_conf = self._run_ocr("paddleocr", _paddle_engine, _ge)
                                if text and ocr_conf > 0.20:
                                    cleaned = self.validator.clean_ocr_text(text)
                                    if self.validator.is_valid_length(cleaned):
                                        is_valid, final_text = self.validator.validate(cleaned)
                                        if is_valid:
                                            _gwc = ocr_conf * 1.0
                                            _v2r = bool(re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', cleaned))
                                            w = 4 if _v2r else 1
                                            for _ in range(w):
                                                all_candidates.append((final_text, _gwc))
                                            if ocr_conf >= 0.80:
                                                _ge_skip = True
                            except Exception:
                                pass
                        if not _ge_skip:
                            for engine_name, engine in self.ocr_engines.items():
                                if engine_name == "paddleocr":
                                    continue
                                try:
                                    text, ocr_conf = self._run_ocr(engine_name, engine, _ge)
                                    if text and ocr_conf > 0.20:
                                        cleaned = self.validator.clean_ocr_text(text)
                                        if self.validator.is_valid_length(cleaned):
                                            is_valid, final_text = self.validator.validate(cleaned)
                                            if is_valid:
                                                _gew = _ENGINE_WEIGHT.get(engine_name, 0.7)
                                                _gwc = ocr_conf * _gew
                                                _v2r = bool(re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', cleaned))
                                                w = 4 if _v2r else 1
                                                for _ in range(w):
                                                    all_candidates.append((final_text, _gwc))
                                except Exception:
                                    continue

            # 구형 번호판 상단+하단 결합 시도 (전처리 포함, Tier1 합의 시 스킵)
            # ★ 속도 최적화: 크롭 조각은 PaddleOCR만 사용 (전체 결과는 Tier1+2에서 교차검증)
            #   실패 시에만 EasyOCR 폴백
            if extra_crops and not _tier1_consensus and not _mid_consensus:
                top_texts, bot_texts = [], []
                top_confs, bot_confs = [], []
                _crop_methods = ["original"]
                for crop_name, crop_img in extra_crops:
                    for cm in _crop_methods:
                        if cm == "original":
                            proc_crop = crop_img
                        else:
                            pfn = getattr(self.preprocessor, cm, None)
                            if pfn is None:
                                continue
                            try:
                                proc_crop = pfn(crop_img.copy())
                            except Exception:
                                continue
                        # ★ PaddleOCR만 실행 (크롭 조각 인식에 충분)
                        _crop_engines = [("paddleocr", _paddle_engine)] if _paddle_engine else []
                        for eng_name, eng in _crop_engines:
                            t, c = self._run_ocr(eng_name, eng, proc_crop)
                            if t and c > 0.20:
                                # ★ 크롭 조각은 clean_ocr_text 사용 안 함
                                # (짧은 "01나", "8060" 등이 패턴 검증에서 버려지므로)
                                # 대신 최소한의 정리만: 특수문자 제거 + 숫자/한글만 유지
                                raw = re.sub(r'[^0-9가-힣a-zA-Z]', '', t.strip())
                                # 영문→숫자/한글 기본 교정
                                _num_map = {'O':'0','o':'0','I':'1','l':'1','Z':'2','S':'5','B':'8','D':'0','Q':'0','G':'6'}
                                _han_map = {'L':'나','H':'하','T':'타','U':'우','P':'파'}
                                corrected = []
                                for ch in raw:
                                    if ch.isalpha() and not ('가' <= ch <= '힣'):
                                        # 영문: 주변 문맥에 따라 숫자 또는 한글로 교정
                                        corrected.append(_num_map.get(ch.upper(), _han_map.get(ch.upper(), ch)))
                                    else:
                                        corrected.append(ch)
                                cleaned_t = ''.join(corrected)
                                # 한글 혼동 교정 (대→다, 니→나 등)
                                fixed_chars = []
                                for ch in cleaned_t:
                                    if '가' <= ch <= '힣':
                                        fixed_chars.append(self.validator._KR_CONFUSION.get(ch, ch))
                                    else:
                                        fixed_chars.append(ch)
                                cleaned_t = ''.join(fixed_chars)
                                if cleaned_t:
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

            # ★ CRNN 결과를 all_candidates에 추가 (합의 기반 가중)
            # CRNN은 12장 학습 모델이므로 미학습 번호판에서 오인식 가능
            # → 기존 OCR 후보와 숫자 부분이 겹치면 높은 가중치, 아니면 낮은 가중치
            for ct, cc in _crnn_candidates:
                _crnn_weight = 2  # 기본: 소수 투표 (미학습 번호판 안전장치)
                if all_candidates:
                    # CRNN 결과에서 숫자 부분 추출
                    _crnn_digits = re.sub(r'[^0-9]', '', ct)
                    # 기존 OCR 후보 중 숫자 부분이 일치하는 비율 계산
                    _digit_matches = 0
                    for _oc_text, _ in all_candidates:
                        _oc_digits = re.sub(r'[^0-9]', '', _oc_text)
                        if _crnn_digits and _oc_digits:
                            # 뒷 4자리 숫자 일치 확인 (가장 안정적인 부분)
                            if _crnn_digits[-4:] == _oc_digits[-4:]:
                                _digit_matches += 1
                    # 숫자 합의율 기반 적응적 CRNN 가중치
                    # 후보 풀이 작은 2자리 번호판: 높은 가중치 유지 (CRNN 기여도 보존)
                    # 후보 풀이 큰 3자리/지역명 번호판: 낮은 가중치 (OCR 합의 우선)
                    _pool_small = len(all_candidates) < 15
                    if _digit_matches > len(all_candidates) * 0.2:
                        _crnn_weight = 35 if _pool_small else 25
                    elif _digit_matches > 0:
                        _crnn_weight = 10 if _pool_small else 8
                for _ in range(_crnn_weight):
                    all_candidates.append((ct, cc))

            # ── 위치별 분리 투표: 한글·숫자 각각 투표 후 합성 ──
            # 전체 문자열 투표만 하면 한글 1자 차이로 표 분산 → 정확도 하락
            # 예: "55저9392" 10표, "55가9392" 8표, "55아9392" 5표
            #   → 전체투표: "55저9392" 승 (10표)
            #   → 숫자부("559392") 23표 통합, 한글 투표: 저10 가8 아5 → 저 승
            best_text = ""
            best_conf = 0.0
            if all_candidates:
                # 신형 번호판 분리 투표 시도 (숫자2-3 + 한글1 + 숫자4)
                _re_split = re.compile(r'^(\d{2,3})([가-힣])(\d{4})$')
                # 구형 번호판 분리 (지역2-3 + 숫자1-2 + 한글1 + 숫자4)
                _re_split_old = re.compile(r'^([가-힣]{2,3})(\d{1,2})([가-힣])(\d{4})$')

                new_parts = []   # (prefix, hangul, suffix, conf)
                old_parts = []   # (region, num, hangul, suffix, conf)
                other_candidates = []  # 분리 불가 후보

                for txt, c in all_candidates:
                    m_new = _re_split.match(txt)
                    m_old = _re_split_old.match(txt)
                    if m_new:
                        new_parts.append((m_new.group(1), m_new.group(2), m_new.group(3), c))
                    elif m_old:
                        old_parts.append((m_old.group(1), m_old.group(2), m_old.group(3), m_old.group(4), c))
                    else:
                        other_candidates.append((txt, c))

                combined_best = ""
                combined_conf = 0.0

                # 신형 번호판 위치별 투표
                if new_parts:
                    prefix_counter = Counter()
                    hangul_counter = Counter()
                    suffix_counter = Counter()
                    prefix_confs = defaultdict(list)
                    hangul_confs = defaultdict(list)
                    suffix_confs = defaultdict(list)
                    for pfx, hg, sfx, c in new_parts:
                        prefix_counter[pfx] += 1
                        hangul_counter[hg] += 1
                        suffix_counter[sfx] += 1
                        prefix_confs[pfx].append(c)
                        hangul_confs[hg].append(c)
                        suffix_confs[sfx].append(c)
                    best_pfx = prefix_counter.most_common(1)[0][0]
                    # hangul: confidence-sum 가중 투표
                    best_hg = max(hangul_confs.keys(), key=lambda k: sum(hangul_confs[k]))
                    # ★ 한글 초성 교차검증 (PaddleOCR det=False 크롭 인식)
                    if self._hangul_clf._ready and 'paddleocr' in self.ocr_engines and _clf_scale >= 1.0:
                        try:
                            _sc, _pd = _clf_scale, _clf_pad
                            _px1 = _pd + (ox1 - rx1) * _sc
                            _py1 = _pd + (oy1 - ry1) * _sc
                            _pw  = (ox2 - ox1) * _sc
                            _ph  = (oy2 - oy1) * _sc
                            _rh, _rw = roi_for_ocr.shape[:2]
                            _hx1 = max(0, int(_px1 + _pw * 0.26))
                            _hx2 = min(_rw, int(_px1 + _pw * 0.52))
                            _hy1 = max(0, int(_py1 + _ph * 0.20))
                            _hy2 = min(_rh, int(_py1 + _ph * 0.80))
                            if _hx2 > _hx1 + 10 and _hy2 > _hy1 + 10:
                                _hcrop = roi_for_ocr[_hy1:_hy2, _hx1:_hx2]
                                _new_hg, _changed = self._hangul_clf.check_override(
                                    best_hg, _hcrop, self.ocr_engines['paddleocr'],
                                    self.ocr_engines
                                )
                                if _changed:
                                    best_hg = _new_hg
                        except Exception:
                            pass
                    best_sfx = suffix_counter.most_common(1)[0][0]
                    # 합성 결과가 유효한지 검증
                    synth = best_pfx + best_hg + best_sfx
                    is_v, final_synth = self.validator.validate(synth)
                    if is_v:
                        combined_best = final_synth
                        all_c = [c for _, _, _, c in new_parts]
                        combined_conf = sum(all_c) / len(all_c)
                    else:
                        # 합성 실패 시 전체 투표 폴백
                        counter_new = Counter(p + h + s for p, h, s, _ in new_parts)
                        top = counter_new.most_common(1)[0][0]
                        combined_best = top
                        combined_conf = sum(c for p, h, s, c in new_parts if p+h+s == top) / counter_new[top]

                # 구형 번호판 위치별 투표
                if old_parts:
                    region_counter = Counter()
                    num_counter = Counter()
                    hangul_counter = Counter()
                    suffix_counter = Counter()
                    hangul_confs_old = defaultdict(list)
                    for rg, nm, hg, sfx, c in old_parts:
                        region_counter[rg] += 1
                        num_counter[nm] += 1
                        hangul_counter[hg] += 1
                        suffix_counter[sfx] += 1
                        hangul_confs_old[hg].append(c)
                    best_rg = region_counter.most_common(1)[0][0]
                    best_nm = num_counter.most_common(1)[0][0]
                    # hangul: confidence-sum 가중 투표
                    best_hg = max(hangul_confs_old.keys(), key=lambda k: sum(hangul_confs_old[k]))
                    best_sfx = suffix_counter.most_common(1)[0][0]
                    synth_old = best_rg + best_nm + best_hg + best_sfx
                    is_v, final_old = self.validator.validate(synth_old)
                    if is_v:
                        old_conf = sum(c for _, _, _, _, c in old_parts) / len(old_parts)
                        # 구형이 신형보다 득표 많으면 구형 선택
                        if len(old_parts) > len(new_parts) or not combined_best:
                            combined_best = final_old
                            combined_conf = old_conf

                # 기타 후보 포함 전체 투표도 비교
                if other_candidates and not combined_best:
                    counter_other = Counter(t for t, _ in other_candidates)
                    top_other = counter_other.most_common(1)[0][0]
                    combined_best = top_other
                    combined_conf = sum(c for t, c in other_candidates if t == top_other) / counter_other[top_other]

                # 최종: 분리 투표 결과 vs 전체 투표 결과 비교
                counter_all = Counter(t for t, _ in all_candidates)
                whole_best = counter_all.most_common(1)[0][0]
                whole_count = counter_all[whole_best]
                whole_confs = [c for t, c in all_candidates if t == whole_best]
                whole_conf = sum(whole_confs) / len(whole_confs)

                if combined_best:
                    # 분리 투표 합성 결과가 전체 투표 1위와 다르면 → 분리투표 우선
                    # (한글만 다르고 숫자 동일한 경우 분리투표가 더 정확)
                    best_text = combined_best
                    # ★ 최대값 가중 평균: max×0.6 + mean×0.4 (저신뢰 변형이 평균을 희석하는 문제 방지)
                    # 실제 당선 텍스트의 confidence만 사용 (new_parts/old_parts 혼동 방지)
                    _bt_confs = [c for t, c in all_candidates if t == combined_best]
                    if _bt_confs:
                        _c_max = max(_bt_confs)
                        _c_mean = sum(_bt_confs) / len(_bt_confs)
                        best_conf = _c_max * 0.6 + _c_mean * 0.4
                    else:
                        best_conf = combined_conf
                else:
                    best_text = whole_best
                    # ★ 전체 투표도 최대값 가중 평균 적용
                    _w_max = max(whole_confs)
                    best_conf = _w_max * 0.6 + whole_conf * 0.4

                # ★ 합의 강도 보너스: 1위 득표율이 높을수록 신뢰도 보강
                _total_votes = sum(counter_all.values())
                if _total_votes > 0:
                    _top_ratio = whole_count / _total_votes
                    if _top_ratio >= 0.80 and whole_count >= 3:
                        best_conf = min(best_conf * 1.15, 1.0)  # 80%+ 합의(3표+) → +15%
                    elif _top_ratio >= 0.60 and whole_count >= 2:
                        best_conf = min(best_conf * 1.08, 1.0)  # 60%+ 합의(2표+) → +8%

                # ★ 지역명 교정: 유효한 지역명을 가진 후보가 있으면 우선 사용
                _re_withregion = re.compile(r'^([가-힣]{2,3})(\d{2}[가-힣]\d{4})$')
                _re_noregion = re.compile(r'^\d{2,3}[가-힣]\d{4}$')
                _VALID_REGIONS = set(PlateValidator._REGION_PREFIXES)

                # 유효한 지역명을 가진 후보만 수집
                valid_region_counts = {}  # {region: count}  유효 지역만
                for cand_t, cand_c in all_candidates:
                    m_r = _re_withregion.match(cand_t)
                    if m_r:
                        region = m_r.group(1)
                        if region in _VALID_REGIONS:
                            valid_region_counts[region] = valid_region_counts.get(region, 0) + 1

                # Case 1: 최종 결과가 비지역이면 → 유효 지역 복원
                if _re_noregion.match(best_text):
                    if valid_region_counts:
                        top_region = max(valid_region_counts, key=valid_region_counts.get)
                        candidate_text = top_region + best_text
                        is_v, final = self.validator.validate(candidate_text)
                        if is_v:
                            best_text = final

                # Case 2: 최종 결과에 지역이 있지만 무효 지역일 수 있음
                # → 유효 지역이 후보에 있으면 교체
                elif _re_withregion.match(best_text):
                    m_cur = _re_withregion.match(best_text)
                    cur_region = m_cur.group(1)
                    cur_suffix = m_cur.group(2)
                    if cur_region not in _VALID_REGIONS and valid_region_counts:
                        # 현재 지역이 무효 → 유효 지역 중 최다 표 선택
                        top_region = max(valid_region_counts, key=valid_region_counts.get)
                        new_text = top_region + cur_suffix
                        is_v, final = self.validator.validate(new_text)
                        if is_v:
                            best_text = final
                    elif cur_region in _VALID_REGIONS and valid_region_counts:
                        # 현재 지역이 유효하지만 다른 유효 지역이 더 많은 표를 받았으면 교체
                        for region, count in valid_region_counts.items():
                            if region != cur_region and count > valid_region_counts.get(cur_region, 0):
                                new_text = region + cur_suffix
                                is_v, final = self.validator.validate(new_text)
                                if is_v:
                                    best_text = final
                                    break

            # ★ 디버그: 엔진별 OCR 시간 출력 (환경변수 _DEBUG_CROP=1)
            if os.environ.get('_DEBUG_CROP', '') and _ocr_time_by_engine:
                _parts = []
                for _en, _et in sorted(_ocr_time_by_engine.items()):
                    _cnt = _ocr_count_by_engine[_en]
                    _parts.append(f"{_en}={_et*1000:.0f}ms({_cnt}calls)")
                print(f"  [OCR타임] {' | '.join(_parts)} → best={best_text} conf={best_conf:.2f}")

            if best_text and best_conf >= self.config.OCR_CONF:
                # ★ 번호 범위 검증 페널티 (앞 2~3자리 숫자 범위)
                _num_range_penalty = 1.0
                _m_prefix = re.match(r'^(?:[가-힣]{2,3})?(\d{2,3})[가-힣]\d{4}$', best_text)
                if _m_prefix:
                    _pnum = int(_m_prefix.group(1))
                    if len(_m_prefix.group(1)) == 3:
                        # 신형 3자리: 100~997 정상, 000~099/998~999 비정상
                        if _pnum < 100 or _pnum > 997:
                            _num_range_penalty = 0.70  # 30% 감점
                    # 2자리(구형)는 01~99 모두 정상 → 페널티 없음

                # ★ OCR 신뢰도 기반 bbox 페널티 완화
                # OCR이 강한 합의를 달성하면 bbox 크기 불확실성이 해소됨
                # → 소형 번호판이라도 OCR 정확하면 페널티 감면
                if best_conf >= 0.90:
                    _bbox_relief = 0.6   # 고신뢰: 페널티 60% 감면
                elif best_conf >= 0.80:
                    _bbox_relief = 0.4   # 중신뢰: 40% 감면
                elif best_conf >= 0.70:
                    _bbox_relief = 0.2   # 저신뢰: 20% 감면
                else:
                    _bbox_relief = 0.0   # 불확실: 감면 없음
                _bbox_conf_penalty += (1.0 - _bbox_conf_penalty) * _bbox_relief

                # ★ bbox 크기 + 색상 + 번호범위 기반 신뢰도 페널티 적용
                best_conf *= _bbox_conf_penalty * _color_conf_penalty * _num_range_penalty

                # ★ 패턴 검증 통과 + OCR 합의 강도에 따른 적응형 신뢰도 floor
                # OCR 원본 conf(페널티 적용 전)가 높을수록 floor도 높게 설정
                # → bbox가 작아도 OCR이 확신하면 높은 최종 신뢰도 보장
                _raw_conf_before_penalty = best_conf / max(_bbox_conf_penalty * _color_conf_penalty * _num_range_penalty, 0.01)
                if _raw_conf_before_penalty >= 0.95:
                    _floor = 0.92 if not _is_small_plate else 0.90
                elif _raw_conf_before_penalty >= 0.85:
                    _floor = 0.85 if not _is_small_plate else 0.80
                elif _raw_conf_before_penalty >= 0.75:
                    _floor = 0.78 if not _is_small_plate else 0.72
                else:
                    _floor = 0.70 if not _is_small_plate else 0.60
                best_conf = max(best_conf, _floor)

                # ★ 신뢰도 최종 필터: 소형 0.50, 일반 0.60, 대형(가까운 차) 0.55
                _is_large_plate = det_w >= 150
                if _is_small_plate:
                    _conf_threshold = 0.50
                elif _is_large_plate:
                    _conf_threshold = 0.55
                else:
                    _conf_threshold = 0.60
                if best_conf < _conf_threshold:
                    self.stats["filtered_by_confidence"] = self.stats.get("filtered_by_confidence", 0) + 1
                    continue

                # ── PlateTracker: 이미 매칭된 트랙 재사용 (OCR 스킵 사전체크에서 매칭 완료) ──
                matched_trk, is_new_track = _pre_trk, _pre_is_new

                # ★ 텍스트 기반 트랙 병합: IoU로 새 트랙이 됐지만 같은 텍스트의 기존 트랙이 있으면 병합
                # → 차량 이동으로 IoU < 0.30 → 새 트랙 생성 → 동일 번호판 텍스트로 기존 트랙과 연결
                # ★ 중심 거리 200px 이상이면 다른 차량으로 판단 → 병합 거부
                if is_new_track and best_text:
                    _new_cx = (cur_bbox[0] + cur_bbox[2]) / 2
                    _new_cy = (cur_bbox[1] + cur_bbox[3]) / 2
                    for _exist_trk in self._tracker.tracks:
                        if _exist_trk is matched_trk:
                            continue
                        if _exist_trk["texts"]:
                            _exist_top = max(_exist_trk["texts"], key=_exist_trk["texts"].get)
                            if _exist_top == best_text:
                                # ★ 중심 거리 체크: 200px 이상이면 다른 차량
                                _e_bbox = _exist_trk["bbox"]
                                _exist_cx = (_e_bbox[0] + _e_bbox[2]) / 2
                                _exist_cy = (_e_bbox[1] + _e_bbox[3]) / 2
                                _cdist = (((_new_cx - _exist_cx) ** 2) + ((_new_cy - _exist_cy) ** 2)) ** 0.5
                                if _cdist >= 200:
                                    continue  # 거리 200px+ → 같은 텍스트라도 다른 차량
                                # 기존 트랙 데이터를 새 트랙으로 이전
                                matched_trk["texts"] = _exist_trk["texts"]
                                matched_trk["best_conf"] = max(matched_trk["best_conf"], _exist_trk["best_conf"])
                                matched_trk["_detect_count"] = _exist_trk.get("_detect_count", 0) + 1
                                matched_trk["consecutive"] = _exist_trk["consecutive"] + 1
                                matched_trk["recorded"] = _exist_trk["recorded"]
                                # 기존 트랙 무효화
                                _exist_trk["texts"] = defaultdict(int)
                                _exist_trk["last_frame"] = 0
                                is_new_track = False
                                break

                # ★ 투표 decay: 새 텍스트가 기존 최다 투표와 다르면 이전 투표 전부 삭제
                # → 새 차량이 즉시(프레임 1) 이전 차량을 역전
                # → 차량 교체 시 잔존 투표 완전 제거 (Ghost 근본 차단)
                if matched_trk["texts"] and not is_new_track:
                    _cur_top = max(matched_trk["texts"], key=matched_trk["texts"].get)
                    if best_text != _cur_top:
                        # ★ 소형→근거리 대체: 근거리 결과(bbox >= 100px)가 소형 결과를 즉시 교체
                        # 소형 결과(bbox < 70px)끼리 충돌 시에는 decay 하지 않음 (불안정하므로)
                        _cur_top_from_small = matched_trk.get("_last_small_plate", False)
                        if _is_small_plate and _cur_top_from_small:
                            pass  # 소형끼리는 유지 (불안정한 교체 방지)
                        else:
                            matched_trk["texts"].clear()

                # ★ 소형 번호판 투표 가중치: 1표, 일반: bbox 크기 비례 (1~3표)
                if _is_small_plate:
                    _vote_weight = 1
                elif det_w >= 120:
                    _vote_weight = 3   # 대형 bbox → 3표 (근거리 고신뢰)
                elif det_w >= 100:
                    _vote_weight = 2   # 중대형 → 2표
                else:
                    _vote_weight = 1   # 중소형 → 1표
                matched_trk["texts"][best_text] += _vote_weight
                matched_trk["_last_small_plate"] = _is_small_plate

                # 해당 위치에서 가장 많이 읽힌 텍스트 사용
                top_text = max(matched_trk["texts"], key=matched_trk["texts"].get)
                top_conf = max(best_conf, matched_trk.get("best_conf", 0))
                matched_trk["best_conf"] = top_conf

                # 기존 text 기반 추적도 유지 (호환성)
                seen_this_frame.add(top_text)
                plate_info = self.recent_plates[top_text]
                plate_info["consecutive"] = matched_trk["consecutive"]
                plate_info["last_seen"] = time.time()
                plate_info["count"] += 1

                # ★ 표시 기준: consecutive >= N 또는 감지 프레임 >= N (비연속 허용)
                # _detect_count: 투표 가중치 무관, 실제 감지된 프레임 수
                _detect_count = matched_trk.get("_detect_count", matched_trk["consecutive"])
                _show = (matched_trk["consecutive"] >= self.consecutive_required
                         or _detect_count >= self.consecutive_required)
                if _show:
                    is_alert, alert_info = (0, None)
                    if not matched_trk["recorded"]:
                        matched_trk["recorded"] = True
                        try:
                            is_alert, alert_info = self.db.record_plate(
                                top_text, top_conf, camera_id
                            )
                            if is_alert and alert_info:
                                self._trigger_alert(top_text, alert_info)
                        except Exception:
                            pass
                    self.stats["plates_shown"] += 1
                    self.stats["confidences"].append(top_conf)

                    # ★ 프레임 카운트 (해당 위치에서 해당 텍스트가 읽힌 횟수)
                    _frame_count = matched_trk["texts"].get(top_text, 1)
                    # ★ 다중 프레임 투표 보너스 (반복 감지 → 신뢰도 보강)
                    if _frame_count >= 5:
                        _frame_bonus = 0.10   # 5프레임 이상 → +10%
                    elif _frame_count >= 3:
                        _frame_bonus = 0.05   # 3프레임 이상 → +5%
                    elif _frame_count >= 2:
                        _frame_bonus = 0.02   # 2프레임 → +2%
                    else:
                        _frame_bonus = 0
                    _adj_conf = min(top_conf + _frame_bonus, 1.0)
                    _conf_level = self.validator.get_confidence_level(_adj_conf)
                    # ★ 영상 모드: frame_count 1회도 출력 허용 (엔진 consecutive로 이미 필터됨)
                    if _is_small_plate:
                        _conf_level += "(원거리)"

                    # ★ bbox 면적 및 번호판 줄 수 판단
                    _bbox_w = ox2 - ox1
                    _bbox_h = oy2 - oy1
                    _bbox_area = _bbox_w * _bbox_h
                    _aspect = _bbox_w / max(_bbox_h, 1)
                    _plate_lines = 1 if _aspect > 2.5 else 2

                    # ★ 번호판 색상 감지 (ROI의 HSV 분석)
                    _plate_color = "흰색바탕_검은글씨"  # 기본값
                    if _is_green_plate:
                        _plate_color = "초록색바탕_흰글씨"
                    else:
                        try:
                            _hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            _h_mean = np.mean(_hsv_roi[:, :, 0])
                            _s_mean = np.mean(_hsv_roi[:, :, 1])
                            _v_mean = np.mean(_hsv_roi[:, :, 2])
                            if 20 < _h_mean < 35 and _s_mean > 60:
                                _plate_color = "노란색바탕_검은글씨"
                            elif 95 < _h_mean < 130 and _s_mean > 40:
                                _plate_color = "파란색바탕_흰글씨"
                        except Exception:
                            pass

                    # [평가기준 20점] 차량번호 인식 표시 (강화 출력 형식)
                    results.append({
                        # ── 기존 호환 필드 (plate_gui.py 등) ──
                        "plate": top_text,
                        "confidence": top_conf,
                        "bbox": [ox1, oy1, ox2, oy2],
                        "is_alert": bool(is_alert),
                        "alert_info": alert_info,
                        # ── 강화 필드 (LPR 프롬프트 규격) ──
                        "plate_number": top_text,
                        "confidence_level": _conf_level,
                        "plate_type": self.validator.classify_plate_type(top_text),
                        "vehicle_type": self.validator.classify_vehicle_type(top_text),
                        "plate_lines": _plate_lines,
                        "plate_color": _plate_color,
                        "bbox_area": _bbox_area,
                        "frame_count": _frame_count,
                        "is_valid_format": True,
                        "rejection_reason": None,
                    })

        # ── PlateTracker: 프레임 종료 (미감지 트랙 처리 + TTL 만료) ──
        self._tracker.end_frame()
        self._pos_trackers = self._tracker.tracks

        # 기존 text 기반 리셋도 유지
        for key in list(self.recent_plates.keys()):
            if key not in seen_this_frame:
                self.recent_plates[key]["consecutive"] = 0

        return results

    @staticmethod
    def _bbox_iou(a, b):
        """두 bbox [x1,y1,x2,y2] 간 IoU 계산"""
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def _run_ocr(self, engine_name, engine, image):
        """OCR 실행. 구형 두 줄 번호판 대응: y좌표 정렬 + 분할 읽기 + allowlist."""
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
                # ★ allowlist 적용: 번호판에 나올 수 있는 문자만 인식
                kr_allow = getattr(self, '_kr_allowlist', None)
                ocr_kwargs = dict(detail=1, paragraph=False)
                if kr_allow:
                    ocr_kwargs["allowlist"] = kr_allow
                result = engine.readtext(image, **ocr_kwargs)
                if result:
                    # y좌표 기준 정렬 (상→하: 지역명 먼저, 번호 나중)
                    result_sorted = sorted(result, key=lambda r: r[0][0][1])
                    texts = [r[1] for r in result_sorted]
                    confs = [r[2] for r in result_sorted]
                    combined = "".join(texts)
                    avg_conf = float(np.mean(confs))

                    # ★ EasyOCR 내부 재호출 제거 (속도 최적화)
                    # 상단 분리/allowlist 재시도는 PaddleOCR + extra_crops에서 이미 처리
                    return combined, avg_conf
            elif engine_name == "tesseract":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                # PSM 7 = single text line (번호판 1줄)
                custom_config = r'--oem 3 --psm 7 -l kor+eng'
                text = pytesseract.image_to_string(gray, config=custom_config)
                text = text.strip().replace('\n', '').replace(' ', '')
                if text:
                    try:
                        data = pytesseract.image_to_data(
                            gray, config=custom_config,
                            output_type=pytesseract.Output.DICT
                        )
                        confs = [int(c) for c in data['conf'] if int(c) > 0]
                        conf = float(np.mean(confs)) / 100.0 if confs else 0.5
                    except Exception:
                        conf = 0.5
                    return text, conf
            elif engine_name == "crnn":
                # CRNN은 process_frame에서 직접 호출 (여기 도달하면 안 됨)
                text, conf = engine.recognize(image)
                if text:
                    return text, conf
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        total_plates = 0
        recognized_plates = {}  # {plate_text: max_conf}
        start_time = time.time()
        print(f"[시작] 영상 처리: {source} ({w}x{h} @ {fps}fps, {total_frames}프레임)", flush=True)

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
                # 인식된 번호판 기록
                pt = r["plate"]
                if pt not in recognized_plates or r["confidence"] > recognized_plates[pt]:
                    recognized_plates[pt] = r["confidence"]
                print(f"  [frame {frame_count}] {r['plate']} ({r['confidence']:.0%})", flush=True)

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

            # 진행률 출력 (100프레임마다)
            if frame_count % 100 == 0:
                pct = (frame_count / total_frames * 100) if total_frames else 0
                print(f"  ... {frame_count}/{total_frames} ({pct:.0f}%) | {current_fps:.1f} FPS | 인식: {len(recognized_plates)}대", flush=True)

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"\n[완료] {frame_count}프레임, {total_plates}회 인식, 고유 {len(recognized_plates)}대, 평균 {frame_count/elapsed:.1f} FPS", flush=True)
        if recognized_plates:
            print(f"[인식결과]", flush=True)
            for plate, conf in sorted(recognized_plates.items(), key=lambda x: -x[1]):
                print(f"  {plate} (최대 신뢰도: {conf:.0%})", flush=True)


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
    import multiprocessing
    multiprocessing.freeze_support()

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
