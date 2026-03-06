"""
night_enhancer.py — SafePlate 4K 야간 모드 영상 보정 (계단 B-5)

역할: 야간/저조도 영상에서 번호판 인식률을 높이기 위한 프레임 전처리
비유: 야간 투시경. 어두운 영상을 밝게 보정하여 YOLO + OCR이 번호판을 더 잘 읽도록 도움

동작 흐름:
    1. 프레임 밝기 자동 측정 (평균 V 채널)
    2. 어두운 프레임 감지 시 자동 보정 활성화
    3. CLAHE + 감마 보정 + 디노이징 적용
    4. 보정된 프레임을 YOLO에 전달
    5. 화면에 야간 모드 상태 오버레이 표시

기존 코드 수정: 없음
Mock 데이터: 없음

이벤트 버스 연동:
    - 수신: "frame_read"
    - 발행: "NIGHT_MODE_ACTIVATED", "NIGHT_MODE_DEACTIVATED"
"""

import time
from typing import Optional, Dict, Tuple

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


# 기본 설정
DEFAULT_CONFIG: Dict[str, float] = {
    "brightness_threshold": 80.0,     # V 채널 평균 이 값 이하 → 야간 판정
    "hysteresis_high": 95.0,          # 야간→주간 전환 임계값 (히스테리시스)
    "gamma": 1.8,                     # 감마 보정 값 (>1 = 밝게)
    "clahe_clip_limit": 3.0,          # CLAHE 클리핑 한계
    "clahe_grid_size": 8,             # CLAHE 타일 크기
    "denoise_strength": 5,            # 디노이징 강도 (h 파라미터)
    "denoise_color_strength": 5,      # 색상 디노이징 강도
    "headlight_suppress": True,       # 전조등 글레어 억제
    "headlight_threshold": 240,       # 전조등 판정 밝기 임계값
    "auto_detect": True,              # 자동 야간 감지 (False면 수동)
    "smoothing_frames": 10,           # 밝기 측정 평활화 프레임 수
}


class NightEnhancer:
    """야간 모드 영상 보정기 — 저조도 환경 번호판 인식률 향상

    어두운 영상을 자동 감지하고 CLAHE + 감마 보정 + 디노이징을 적용하여
    YOLO 감지 및 OCR 인식률을 높인다.

    Args:
        event_bus: SimulationFramework의 EventBus 인스턴스
        config: 보정 설정 (None이면 기본값 사용)

    시간복잡도:
        enhance_frame: O(w*h) — 프레임 크기에 비례
        measure_brightness: O(w*h) — V 채널 평균 계산
    """

    # 한글 폰트 경로 후보
    FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothicBold.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]

    def __init__(self, event_bus, config: Optional[Dict] = None) -> None:
        """초기화.

        Args:
            event_bus: 이벤트 버스
            config: 보정 설정 딕셔너리 (None이면 기본값)
        """
        self.event_bus = event_bus
        # 기본값과 병합 (부분 config 전달 시에도 누락 키 방지)
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # ── 상태 ──
        self.is_night_mode = False       # 현재 야간 모드 활성화 여부
        self._force_mode: Optional[bool] = None  # 수동 강제 모드 (None=자동)
        self._brightness_history: list[float] = []  # 밝기 이력 (평활화)
        self._current_brightness = 0.0   # 현재 프레임 밝기
        self._avg_brightness = 0.0       # 평활화된 평균 밝기

        # ── 감마 보정 LUT (미리 계산) ──
        self._gamma_lut = self._build_gamma_lut(self.config["gamma"])

        # ── CLAHE 객체 ──
        grid = int(self.config["clahe_grid_size"])
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config["clahe_clip_limit"],
            tileGridSize=(grid, grid),
        )

        # ── 통계 ──
        self._total_frames = 0
        self._enhanced_frames = 0
        self._activate_time: Optional[float] = None

        # ── 한글 폰트 ──
        self._font = self._load_font(size=20)
        self._font_small = self._load_font(size=14)

        # ── 이벤트 구독 ──
        self.event_bus.subscribe("frame_read", self._on_frame_read)
        self.event_bus.subscribe("key_pressed", self._on_key_pressed)

        mode_str = "자동" if self.config["auto_detect"] else "수동(N키)"
        print(f"[NightEnhancer] 초기화 완료 — 모드: {mode_str}")
        print(f"  밝기 임계값: {self.config['brightness_threshold']}")
        print(f"  감마: {self.config['gamma']}, CLAHE clip: {self.config['clahe_clip_limit']}")

    def _build_gamma_lut(self, gamma: float) -> np.ndarray:
        """감마 보정 룩업 테이블 생성.

        Args:
            gamma: 감마 값 (>1이면 밝게, <1이면 어둡게)

        Returns:
            256개 값의 uint8 LUT

        시간복잡도: O(1) — 256개 고정
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")
        return table

    def _load_font(self, size: int = 20):
        """한글 폰트 로딩"""
        for font_path in self.FONT_CANDIDATES:
            try:
                return ImageFont.truetype(font_path, size)
            except (IOError, OSError):
                continue
        return None

    # ═══════════════════════════════════════════════
    # 핵심 기능: 프레임 보정
    # ═══════════════════════════════════════════════

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """야간 모드 프레임 보정 적용.

        보정 파이프라인:
            1. 전조등 글레어 억제 (선택)
            2. CLAHE 명암 대비 향상 (V 채널)
            3. 감마 보정 (전체 밝기 증가)
            4. 디노이징 (야간 노이즈 제거)

        Args:
            frame: BGR 형식 원본 프레임

        Returns:
            보정된 BGR 프레임

        시간복잡도: O(w*h)
        """
        if frame is None:
            return frame

        enhanced = frame.copy()

        # 1단계: 전조등 글레어 억제
        if self.config["headlight_suppress"]:
            enhanced = self._suppress_headlight_glare(enhanced)

        # 2단계: CLAHE — V 채널 명암 대비 향상
        enhanced = self._apply_clahe(enhanced)

        # 3단계: 감마 보정 — 전체 밝기 증가
        enhanced = cv2.LUT(enhanced, self._gamma_lut)

        # 4단계: 디노이징 — 야간 노이즈 제거
        h_param = self.config["denoise_strength"]
        hc_param = self.config["denoise_color_strength"]
        if h_param > 0:
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced, None, h_param, hc_param, 7, 21
            )

        self._enhanced_frames += 1
        return enhanced

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE 적용 — HSV V 채널 명암 대비 향상.

        Args:
            frame: BGR 프레임

        Returns:
            CLAHE 적용된 BGR 프레임

        시간복잡도: O(w*h)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # V 채널에 CLAHE 적용
        v_enhanced = self._clahe.apply(v)

        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    def _suppress_headlight_glare(self, frame: np.ndarray) -> np.ndarray:
        """전조등 글레어 억제 — 과노출 영역 밝기 감소.

        야간 영상에서 전조등이 과도하게 밝으면 주변 번호판이 날아감.
        밝은 영역만 선택적으로 어둡게 하여 번호판 가독성 확보.

        Args:
            frame: BGR 프레임

        Returns:
            글레어 억제된 BGR 프레임

        시간복잡도: O(w*h)
        """
        threshold = self.config["headlight_threshold"]

        # 그레이스케일에서 과노출 영역 마스크
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # 마스크 확장 (글레어 주변도 포함)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)

        # 블러로 경계 부드럽게
        bright_mask = cv2.GaussianBlur(bright_mask, (21, 21), 0)

        # 과노출 영역 밝기 감소 (70%로)
        mask_float = bright_mask.astype(np.float32) / 255.0
        suppression = 1.0 - (mask_float * 0.3)  # 최대 30% 감소

        result = frame.copy()
        for c in range(3):
            result[:, :, c] = np.clip(
                frame[:, :, c] * suppression, 0, 255
            ).astype(np.uint8)

        return result

    # ═══════════════════════════════════════════════
    # 밝기 측정 + 자동 감지
    # ═══════════════════════════════════════════════

    def measure_brightness(self, frame: np.ndarray) -> float:
        """프레임 밝기 측정 — HSV V 채널 평균.

        Args:
            frame: BGR 프레임

        Returns:
            평균 밝기 (0~255)

        시간복잡도: O(w*h)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))

    def _update_brightness(self, frame: np.ndarray) -> None:
        """밝기 이력 업데이트 + 평활화.

        Args:
            frame: BGR 프레임

        시간복잡도: O(w*h + s) — s=smoothing_frames
        """
        brightness = self.measure_brightness(frame)
        self._current_brightness = brightness

        # 이력에 추가
        max_history = int(self.config["smoothing_frames"])
        self._brightness_history.append(brightness)
        if len(self._brightness_history) > max_history:
            self._brightness_history.pop(0)

        # 평균 계산
        self._avg_brightness = sum(self._brightness_history) / len(self._brightness_history)

    def _auto_detect_night(self) -> None:
        """자동 야간 모드 전환 — 히스테리시스 적용.

        밝기가 threshold 이하 → 야간 모드 ON
        밝기가 hysteresis_high 이상 → 야간 모드 OFF
        그 사이 → 현재 상태 유지 (깜빡임 방지)

        시간복잡도: O(1)
        """
        if self._force_mode is not None:
            # 수동 강제 모드 — 자동 감지 무시
            return

        threshold = self.config["brightness_threshold"]
        hysteresis = self.config["hysteresis_high"]

        was_night = self.is_night_mode

        if self._avg_brightness <= threshold:
            self.is_night_mode = True
        elif self._avg_brightness >= hysteresis:
            self.is_night_mode = False
        # else: 현재 상태 유지 (히스테리시스 구간)

        # 상태 전환 시 이벤트 발행
        if self.is_night_mode and not was_night:
            self._activate_time = time.time()
            self.event_bus.publish("NIGHT_MODE_ACTIVATED", {
                "brightness": self._avg_brightness,
                "threshold": threshold,
                "timestamp": time.time(),
            })
            print(f"[NightEnhancer] 야간 모드 활성화 (밝기: {self._avg_brightness:.1f})")

        elif not self.is_night_mode and was_night:
            self.event_bus.publish("NIGHT_MODE_DEACTIVATED", {
                "brightness": self._avg_brightness,
                "timestamp": time.time(),
            })
            self._activate_time = None
            print(f"[NightEnhancer] 야간 모드 해제 (밝기: {self._avg_brightness:.1f})")

    # ═══════════════════════════════════════════════
    # 이벤트 핸들러
    # ═══════════════════════════════════════════════

    def _on_frame_read(self, data: dict) -> None:
        """프레임 읽기 이벤트 — 밝기 측정 + 자동 야간 감지.

        시간복잡도: O(w*h)
        """
        frame = data.get("frame")
        if frame is None:
            return

        self._total_frames += 1
        self._update_brightness(frame)

        if self.config["auto_detect"]:
            self._auto_detect_night()

    def _on_key_pressed(self, data: dict) -> None:
        """키 입력 — N키로 야간 모드 수동 토글.

        시간복잡도: O(1)
        """
        char = data.get("char", "").lower()

        if char == "n":
            if self._force_mode is None:
                # 자동 → 수동 ON
                self._force_mode = True
                self.is_night_mode = True
                self._activate_time = time.time()
                print(f"[NightEnhancer] 야간 모드 수동 ON")
            elif self._force_mode:
                # 수동 ON → 수동 OFF
                self._force_mode = False
                self.is_night_mode = False
                self._activate_time = None
                print(f"[NightEnhancer] 야간 모드 수동 OFF")
            else:
                # 수동 OFF → 자동
                self._force_mode = None
                print(f"[NightEnhancer] 야간 모드 자동 복귀")

    # ═══════════════════════════════════════════════
    # 오버레이
    # ═══════════════════════════════════════════════

    def draw_night_overlay(self, frame: np.ndarray) -> np.ndarray:
        """야간 모드 상태 오버레이 표시.

        야간 모드 ON:
            - 우상단 달 아이콘 + 밝기 게이지
            - 프레임 테두리 파란색

        야간 모드 OFF:
            - 우상단 작은 밝기 표시만

        Args:
            frame: BGR 프레임

        Returns:
            오버레이가 그려진 프레임

        시간복잡도: O(1) — 고정 크기 그리기
        """
        h, w = frame.shape[:2]

        if self.is_night_mode:
            return self._draw_active_overlay(frame, h, w)
        else:
            return self._draw_inactive_indicator(frame, h, w)

    def _draw_active_overlay(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """야간 모드 활성 상태 오버레이"""
        overlay = frame.copy()

        # ── 1. 우상단 야간 모드 패널 ──
        panel_w, panel_h = 220, 60
        px = w - panel_w - 10
        py = 10

        # 반투명 배경
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                       (40, 30, 20), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        # 모드 표시
        mode_str = "수동" if self._force_mode is not None else "자동"
        night_text = f"NIGHT MODE [{mode_str}]"

        frame = self._put_text(
            frame, night_text,
            (px + 8, py + 22),
            color=(255, 200, 100),
            font=self._font_small,
        )

        # 밝기 게이지 바
        gauge_x = px + 8
        gauge_y = py + 32
        gauge_w = panel_w - 16
        gauge_h = 12

        # 게이지 배경 (어두운 회색)
        cv2.rectangle(frame, (gauge_x, gauge_y),
                       (gauge_x + gauge_w, gauge_y + gauge_h),
                       (60, 60, 60), -1)

        # 게이지 채움 (밝기에 비례)
        fill_ratio = min(self._avg_brightness / 255.0, 1.0)
        fill_w = int(gauge_w * fill_ratio)

        # 색상: 어두우면 파랑, 밝으면 노랑
        if self._avg_brightness < self.config["brightness_threshold"]:
            gauge_color = (200, 150, 50)   # 파란 계열 (BGR)
        else:
            gauge_color = (50, 200, 200)   # 노란 계열

        cv2.rectangle(frame, (gauge_x, gauge_y),
                       (gauge_x + fill_w, gauge_y + gauge_h),
                       gauge_color, -1)

        # 밝기 수치
        bright_text = f"V:{self._avg_brightness:.0f}"
        frame = self._put_text(
            frame, bright_text,
            (px + panel_w - 55, py + panel_h - 6),
            color=(180, 180, 180),
            font=self._font_small,
        )

        # ── 2. 프레임 테두리 (어두운 파란색) ──
        border_color = (180, 120, 40)  # 어두운 파란색 (BGR)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 2)

        return frame

    def _draw_inactive_indicator(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """야간 모드 비활성 상태 — 작은 밝기 표시"""
        bright_text = f"[N] V:{self._avg_brightness:.0f}"
        return self._put_text(
            frame, bright_text,
            (w - 120, 28),
            color=(120, 120, 120),
            font=self._font_small,
        )

    def _put_text(self, frame: np.ndarray, text: str, position: tuple,
                  color: tuple = (255, 255, 255), font=None) -> np.ndarray:
        """한글 텍스트를 프레임에 그리기 — PIL 사용"""
        use_font = font or self._font

        if use_font is not None:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            rgb_color = (color[2], color[1], color[0])

            try:
                text_bbox = draw.textbbox((0, 0), text, font=use_font)
                text_height = text_bbox[3] - text_bbox[1]
            except Exception:
                text_height = 16

            x, y = position
            y_top = y - text_height

            draw.text((x, y_top), text, font=use_font, fill=rgb_color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(frame, text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            return frame

    # ═══════════════════════════════════════════════
    # headless 모드용 API
    # ═══════════════════════════════════════════════

    def force_night_mode(self, enabled: bool) -> None:
        """프로그래밍 방식으로 야간 모드 강제 설정 (headless 테스트용).

        Args:
            enabled: True=야간 ON, False=야간 OFF

        시간복잡도: O(1)
        """
        self._force_mode = enabled
        self.is_night_mode = enabled
        if enabled:
            self._activate_time = time.time()
        else:
            self._activate_time = None
        print(f"[NightEnhancer] 야간 모드 강제: {'ON' if enabled else 'OFF'}")

    def force_auto_mode(self) -> None:
        """자동 감지 모드로 복귀 (headless 테스트용).

        시간복잡도: O(1)
        """
        self._force_mode = None
        print(f"[NightEnhancer] 자동 감지 모드 복귀")

    # ═══════════════════════════════════════════════
    # 상태 조회
    # ═══════════════════════════════════════════════

    def get_stats(self) -> dict:
        """통계 반환.

        Returns:
            {"total_frames": int, "enhanced_frames": int,
             "current_brightness": float, "avg_brightness": float,
             "is_night_mode": bool}

        시간복잡도: O(1)
        """
        return {
            "total_frames": self._total_frames,
            "enhanced_frames": self._enhanced_frames,
            "current_brightness": self._current_brightness,
            "avg_brightness": self._avg_brightness,
            "is_night_mode": self.is_night_mode,
            "enhancement_ratio": (
                self._enhanced_frames / self._total_frames
                if self._total_frames > 0 else 0.0
            ),
        }
