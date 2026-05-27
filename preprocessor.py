# -*- coding: utf-8 -*-
"""Image preprocessing pipeline for plate OCR.

22가지 전처리 메서드를 정적 메서드 모음으로 제공한다.
``ImagePreprocessor``는 상태를 갖지 않고, 호출자가 메서드 이름 문자열로
``getattr(preprocessor, name)``식 디스패치를 한다(``OCRConfig.PREPROCESS_METHODS``).

Extracted from plate_engine_pro.py (refactor — SRP 분리).
"""
from __future__ import annotations

import cv2
import numpy as np


# ── 공용 커널/유틸 ─────────────────────────────────────────────
# sharpen ↔ deblur가 동일한 라플라시안 커널을 쓰던 중복을 단일 상수로 통합.
_SHARPEN_KERNEL: np.ndarray = np.array(
    [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
)


def _gamma_lut(gamma: float) -> np.ndarray:
    """gamma 값에 대응되는 256-entry uint8 LUT 생성."""
    return np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)]
    ).astype("uint8")


def _deskew_and_otsu(gray: np.ndarray) -> np.ndarray:
    """기울기 보정 후 Otsu 이진화. 실패 시 Otsu만 적용."""
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
        rotated = cv2.warpAffine(
            gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE
        )
        _, result = cv2.threshold(
            rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return result
    except Exception:
        _, result = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return result


class ImagePreprocessor:
    """18~22종 이미지 전처리 파이프라인.

    모든 메서드는 ``np.ndarray (BGR)`` → ``np.ndarray (BGR)``의 시그니처.
    상태가 없으므로 정적 메서드로 구현하지만, 호출자가 인스턴스 + getattr로
    디스패치하는 기존 패턴을 유지하기 위해 클래스 형태를 보존한다.
    """

    # ── ①~⑦ 기본 전처리 ────────────────────────────────────────
    @staticmethod
    def gray_threshold(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adaptive_threshold(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def clahe(img: np.ndarray) -> np.ndarray:
        """CLAHE 대비 향상 (clipLimit 5.0, tile 8x8 — 그림자/음영 강화)"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(img: np.ndarray) -> np.ndarray:
        """노이즈 제거 (hqdn3d 스타일 가우시안 + bilateral)"""
        blurred = cv2.bilateralFilter(img, 9, 75, 75)
        return cv2.GaussianBlur(blurred, (3, 3), 0.5)

    @staticmethod
    def sharpen(img: np.ndarray) -> np.ndarray:
        """라플라시안 샤프닝 (기존 deblur와 동일 커널)."""
        return cv2.filter2D(img, -1, _SHARPEN_KERNEL)

    @staticmethod
    def deblur(img: np.ndarray) -> np.ndarray:
        """sharpen()의 alias — 외부 호출 이름 호환 유지."""
        return ImagePreprocessor.sharpen(img)

    @staticmethod
    def gamma_bright(img: np.ndarray, gamma: float = 0.5) -> np.ndarray:
        return cv2.LUT(img, _gamma_lut(gamma))

    @staticmethod
    def gamma_dark(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        return cv2.LUT(img, _gamma_lut(gamma))

    @staticmethod
    def bilateral(img: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(img, 11, 75, 75)

    @staticmethod
    def morphology(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10
        )
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 30:
                    angles.append(angle)
            if angles:
                median_angle = float(np.median(angles))
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
                return cv2.warpAffine(
                    img, M, (w, h), borderMode=cv2.BORDER_REPLICATE
                )
        return img

    # ── ⑧~⑮ 추가 전처리 ────────────────────────────────────────

    @staticmethod
    def median_blur(img: np.ndarray) -> np.ndarray:
        """⑨ 중앙값 필터 (점잡음 제거)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.medianBlur(gray, 3)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def otsu_inv(img: np.ndarray) -> np.ndarray:
        """⑩ Otsu 반전 (흰 배경 번호판)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        inv = cv2.bitwise_not(otsu)
        return cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def upscale_2x(img: np.ndarray) -> np.ndarray:
        """⑪ 2배 업스케일 (작은 번호판)"""
        return cv2.resize(
            img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )

    @staticmethod
    def brightness_boost(img: np.ndarray) -> np.ndarray:
        """⑫ 밝기 보정 (alpha=1.5, beta=+30)"""
        return cv2.convertScaleAbs(img, alpha=1.5, beta=30)

    @staticmethod
    def hist_equalize(img: np.ndarray) -> np.ndarray:
        """⑬ 히스토그램 평활화"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adaptive_mean(img: np.ndarray) -> np.ndarray:
        """⑭ Adaptive Mean (blockSize=15)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 15, 8,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def deskew_otsu(img: np.ndarray) -> np.ndarray:
        """⑮ 기울기 보정 후 Otsu"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = _deskew_and_otsu(gray)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def invert_color(img: np.ndarray) -> np.ndarray:
        """⑯ 색상 반전 — 초록/노란 번호판 (밝은 글씨 + 컬러 배경)"""
        return cv2.bitwise_not(img)

    @staticmethod
    def green_plate(img: np.ndarray) -> np.ndarray:
        """⑰ 초록 번호판 전용 — HSV 초록 제거 + 반전"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        result = img.copy()
        result[mask > 0] = [0, 0, 0]
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def yellow_plate(img: np.ndarray) -> np.ndarray:
        """⑱ 노란 번호판 전용 — HSV 노란 제거 + 반전"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 40, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        result = img.copy()
        result[mask > 0] = [0, 0, 0]
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def color_plate_clahe(img: np.ndarray) -> np.ndarray:
        """⑲ 컬러 번호판 CLAHE + 반전 (초록/노란 공통)"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.bitwise_not(enhanced)

    # ── ⑳~㉒ 야간/역광 전처리 ─────────────────────────────────

    @staticmethod
    def night_clahe(img: np.ndarray) -> np.ndarray:
        """⑳ 야간 강화 CLAHE (clipLimit=8.0, 저조도 번호판 대비 극대화)"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def backlight_adaptive(img: np.ndarray) -> np.ndarray:
        """㉑ 역광 대응 — 밝기 정규화 + adaptive threshold 조합."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=30)
        normalized = cv2.divide(gray, blur, scale=255)
        binary = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 15,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def brightness_normalize(img: np.ndarray) -> np.ndarray:
        """㉒ 밝기 정규화 — 야간/그림자 환경 대응. 평균 밝기 127로 맞춤 + CLAHE."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        mean_l = float(np.mean(l))
        if mean_l > 0:
            scale = 127.0 / mean_l
            l = np.clip(l.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
