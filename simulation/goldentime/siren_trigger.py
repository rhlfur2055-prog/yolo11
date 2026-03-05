"""
siren_trigger.py — 골든타임 2.0 사이렌 트리거

역할: 사이렌 활성화/비활성화 상태 관리 + 화면 오버레이 표시
비유: 구급차 사이렌 스위치 — ON이면 빨간 깜빡임, OFF이면 꺼짐

동작 흐름:
    1. EventBus "key_pressed" 구독
    2. 'S' 키 → 사이렌 활성화 (is_active = True)
    3. 'E' 키 → 사이렌 비활성화 (is_active = False)
    4. draw_siren_overlay() → SimulationFramework.overlay_callbacks 에 등록
       → 매 프레임마다 호출되어 사이렌 상태를 화면에 표시

기존 코드 수정: 없음
Mock 데이터: 없음

실행: --goldentime 옵션으로 자동 연결
    python -m simulation.simulation_framework movie/hiway.mp4 --goldentime
"""

import time
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


class SirenTrigger:
    """사이렌 트리거 — 긴급차량 경로 확보 시뮬레이션의 핵심 스위치

    EventBus에서 key_pressed 이벤트를 받아 사이렌 ON/OFF 전환.
    draw_siren_overlay()를 SimulationFramework.overlay_callbacks에 등록하여
    매 프레임에 사이렌 상태를 시각적으로 표시.

    Args:
        event_bus: SimulationFramework의 EventBus 인스턴스

    상태:
        is_active: True면 사이렌 ON → 빨간 깜빡임 오버레이 표시
        activate_time: 사이렌 활성화된 시각 (경과 시간 계산용)
    """

    # 한글 폰트 경로 후보 (SimulationFramework와 동일)
    FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothicBold.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]

    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.is_active = False          # 사이렌 ON/OFF 상태
        self.activate_time = 0.0        # 사이렌 활성화 시각
        self._blink_interval = 0.3      # 깜빡임 주기 (초)

        # 한글 폰트 로딩
        self._font = self._load_font(size=24)
        self._font_large = self._load_font(size=36)

        # EventBus에 키 입력 이벤트 구독
        self.event_bus.subscribe("key_pressed", self._on_key_pressed)

        print("[SirenTrigger] 초기화 완료 — S:사이렌 시작, E:사이렌 종료")

    def _load_font(self, size: int = 20) -> ImageFont.FreeTypeFont | None:
        """한글 폰트 로딩 — 시스템에서 사용 가능한 폰트 자동 탐색

        시간복잡도: O(m), m = 폰트 후보 수 (상수, 최대 5개)
        """
        for font_path in self.FONT_CANDIDATES:
            try:
                return ImageFont.truetype(font_path, size)
            except (IOError, OSError):
                continue
        return None

    def _on_key_pressed(self, data: dict) -> None:
        """키 입력 이벤트 핸들러

        Args:
            data: {"key": int, "char": str, "frame_idx": int}

        S키(115) 또는 s키(83) → 사이렌 활성화
        E키(101) 또는 e키(69) → 사이렌 비활성화

        시간복잡도: O(1)
        """
        char = data.get("char", "").lower()

        if char == "s" and not self.is_active:
            # 사이렌 활성화
            self.is_active = True
            self.activate_time = time.time()
            print(f"[SirenTrigger] 🚨 사이렌 활성화! (프레임 #{data.get('frame_idx', '?')})")
            # 사이렌 감지 이벤트 발행 → PlateEvidence가 추적 모드 시작
            self.event_bus.publish("SIREN_DETECTED", {
                "frame_idx": data.get("frame_idx", 0),
                "timestamp": time.time(),
            })

        elif char == "e" and self.is_active:
            # 사이렌 비활성화
            elapsed = time.time() - self.activate_time
            self.is_active = False
            print(f"[SirenTrigger] 사이렌 종료 (경과: {elapsed:.1f}초)")
            # 사이렌 종료 이벤트 발행 → PlateEvidence가 추적 모드 해제
            self.event_bus.publish("SIREN_ENDED", {
                "frame_idx": data.get("frame_idx", 0),
                "duration": elapsed,
                "timestamp": time.time(),
            })

    def draw_siren_overlay(self, frame: np.ndarray) -> np.ndarray:
        """사이렌 상태 오버레이 그리기 — SimulationFramework.overlay_callbacks 에 등록

        사이렌 활성화 상태:
            - 화면 상단에 빨간/파란 교대 깜빡임 바
            - "🚨 긴급차량 경로 확보" 텍스트
            - 경과 시간 표시
            - 화면 테두리 빨간색

        사이렌 비활성화 상태:
            - 좌상단에 작은 "[S] 사이렌 대기" 표시만

        Args:
            frame: 현재 프레임 (BGR)

        Returns:
            오버레이가 그려진 프레임

        시간복잡도: O(1)
        """
        if not self.is_active:
            # 비활성 상태 — 대기 표시만
            return self._draw_standby_indicator(frame)

        # ── 활성 상태 — 사이렌 오버레이 ──
        h, w = frame.shape[:2]
        overlay = frame.copy()
        elapsed = time.time() - self.activate_time

        # ── 1. 상단 경고 바 (빨강/파랑 교대 깜빡임) ──
        bar_height = 48
        blink_phase = int(time.time() / self._blink_interval) % 2

        if blink_phase == 0:
            bar_color = (0, 0, 220)     # 빨간색 (BGR)
        else:
            bar_color = (220, 0, 0)     # 파란색 (BGR)

        cv2.rectangle(overlay, (0, 0), (w, bar_height), bar_color, -1)
        # 반투명 블렌딩 (70%)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # ── 2. 경고 텍스트 ──
        alert_text = "긴급차량 경로 확보"
        time_text = f"경과: {elapsed:.1f}초"

        frame = self._put_text(
            frame, alert_text,
            (w // 2 - 140, bar_height - 8),
            color=(255, 255, 255),
            font=self._font_large
        )

        frame = self._put_text(
            frame, time_text,
            (w - 180, bar_height - 12),
            color=(255, 255, 200),
            font=self._font
        )

        # ── 3. 화면 테두리 빨간색 (경고 프레임) ──
        border_thickness = 4
        border_color = (0, 0, 255)  # 빨간색 (BGR)
        if blink_phase == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, border_thickness)

        # ── 4. 좌상단 사이렌 아이콘 영역 ──
        icon_text = "[E] 사이렌 종료"
        frame = self._put_text(
            frame, icon_text,
            (10, bar_height + 30),
            color=(100, 100, 255),
            font=self._font
        )

        return frame

    def _draw_standby_indicator(self, frame: np.ndarray) -> np.ndarray:
        """사이렌 대기 상태 표시 — 좌상단 작은 안내 텍스트

        시간복잡도: O(1)
        """
        standby_text = "[S] 사이렌 대기"
        return self._put_text(
            frame, standby_text,
            (10, 30),
            color=(100, 200, 100),
            font=self._font
        )

    def _put_text(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        color: tuple = (255, 255, 255),
        font: ImageFont.FreeTypeFont | None = None,
    ) -> np.ndarray:
        """한글 텍스트를 프레임에 그리기 — PIL 사용

        cv2.putText()는 한글 미지원 → PIL로 변환 후 그리기.
        한글 폰트 없으면 cv2.putText() 영문 fallback.

        Args:
            frame: 대상 프레임 (BGR)
            text: 표시할 텍스트 (한글 가능)
            position: (x, y) 좌표 — 텍스트 좌하단 기준
            color: BGR 색상 튜플
            font: PIL 폰트 (None이면 기본 폰트 사용)

        Returns:
            텍스트가 그려진 프레임

        시간복잡도: O(1)
        """
        use_font = font or self._font

        if use_font is not None:
            # PIL 방식 (한글 지원)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # BGR → RGB 색상 변환
            rgb_color = (color[2], color[1], color[0])

            # position은 좌하단 기준 → PIL은 좌상단 기준이므로 y 보정
            try:
                text_bbox = draw.textbbox((0, 0), text, font=use_font)
                text_height = text_bbox[3] - text_bbox[1]
            except Exception:
                text_height = 20

            x, y = position
            y_top = y - text_height

            # 텍스트 배경 (가독성)
            try:
                text_width = text_bbox[2] - text_bbox[0]
                draw.rectangle(
                    [x - 2, y_top - 2, x + text_width + 4, y + 4],
                    fill=(0, 0, 0)
                )
            except Exception:
                pass

            draw.text((x, y_top), text, font=use_font, fill=rgb_color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # fallback: cv2.putText (영문만)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return frame
