"""
shock_simulator.py — SafePlate 4K 충격 감지 시뮬레이터 (계단 B-1)

역할: 충격 감지(S키) / 종료(E키) 상태 관리 + 전후 프레임 버퍼링 + 화면 오버레이
비유: 블랙박스 충격 센서. 충격이 감지되면 전후 영상을 자동 저장하고 이탈 차량 추적 시작

동작 흐름:
    1. EventBus "key_pressed" 구독
    2. 'S' 키 → SHOCK_DETECTED 이벤트 발행 (충격 감지)
    3. 'E' 키 → SHOCK_ENDED 이벤트 발행
    4. 충격 감지 시점 전후 프레임 버퍼링 (collections.deque)
    5. 화면 상단에 주황색 경고 바 오버레이

기존 코드 수정: 없음
Mock 데이터: 없음

이벤트 버스 연동:
    - 수신: "key_pressed", "frame_read"
    - 발행: "SHOCK_DETECTED", "SHOCK_ENDED"
"""

import time
from collections import deque

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


class ShockSimulator:
    """충격 감지 시뮬레이터 — 물피도주 시나리오의 트리거 스위치

    EventBus에서 key_pressed 이벤트를 받아 충격 ON/OFF 전환.
    draw_shock_overlay()를 SimulationFramework.overlay_callbacks에 등록하여
    매 프레임에 충격 감지 상태를 시각적으로 표시.

    Args:
        event_bus: SimulationFramework의 EventBus 인스턴스
        fps: 영상 FPS (프레임 버퍼 크기 계산용)
        pre_buffer_sec: 충격 전 버퍼 시간 (초)
        post_buffer_sec: 충격 후 버퍼 시간 (초)

    상태:
        is_active: True면 충격 감지 중 → 주황 경고 바 오버레이 표시
        activate_time: 충격 감지된 시각 (경과 시간 계산용)
    """

    # 한글 폰트 경로 후보
    FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothicBold.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]

    def __init__(self, event_bus, fps: float = 30.0,
                 pre_buffer_sec: float = 10.0, post_buffer_sec: float = 15.0):
        self.event_bus = event_bus
        self.fps = fps
        self.is_active = False
        self.activate_time = 0.0
        self._warn_interval = 0.5       # 경고 바 깜빡임 주기 (초)

        # ── 충격 전후 프레임 버퍼 ──
        # 비유: 블랙박스 순환 녹화. 충격 전 10초 + 충격 후 15초 = 25초 영상
        self.pre_buffer_sec = pre_buffer_sec
        self.post_buffer_sec = post_buffer_sec
        pre_buffer_frames = int(fps * pre_buffer_sec)
        self.pre_buffer: deque = deque(maxlen=pre_buffer_frames)

        # 충격 감지 시 pre_buffer 스냅샷 저장
        self.shock_pre_frames: list = []

        # 충격 후 프레임 수집
        self.shock_post_frames: list = []
        self._post_remaining = 0   # 남은 post 프레임 수

        # ── 충격 시점 메타데이터 ──
        self.shock_frame_idx = 0
        self.shock_timestamp = 0.0

        # 현재 시각 추적
        self._current_timestamp = 0.0
        self._current_frame_idx = 0

        # 한글 폰트 로딩
        self._font = self._load_font(size=24)
        self._font_large = self._load_font(size=36)

        # EventBus에 키 입력 + 프레임 이벤트 구독
        self.event_bus.subscribe("key_pressed", self._on_key_pressed)
        self.event_bus.subscribe("frame_read", self._on_frame_read)

        print("[ShockSimulator] 초기화 완료 — S:충격 감지, E:충격 종료")
        print(f"  전 버퍼: {pre_buffer_sec}초, 후 버퍼: {post_buffer_sec}초")

    def _load_font(self, size: int = 20):
        """한글 폰트 로딩 — 시스템에서 사용 가능한 폰트 자동 탐색"""
        for font_path in self.FONT_CANDIDATES:
            try:
                return ImageFont.truetype(font_path, size)
            except (IOError, OSError):
                continue
        return None

    def _on_key_pressed(self, data: dict) -> None:
        """키 입력 이벤트 핸들러

        S키 → 충격 감지 활성화
        E키 → 충격 감지 비활성화
        """
        char = data.get("char", "").lower()

        if char == "s" and not self.is_active:
            # 충격 감지 활성화
            self.is_active = True
            self.activate_time = time.time()
            self.shock_frame_idx = data.get("frame_idx", 0)
            self.shock_timestamp = self._current_timestamp

            # pre-buffer 스냅샷 — 충격 직전 프레임들 복사
            self.shock_pre_frames = list(self.pre_buffer)

            # post-buffer 수집 시작
            self.shock_post_frames = []
            self._post_remaining = int(self.fps * self.post_buffer_sec)

            print(f"[ShockSimulator] 충격 감지! (프레임 #{self.shock_frame_idx})")
            print(f"  전 버퍼: {len(self.shock_pre_frames)} 프레임 확보")

            # 충격 감지 이벤트 발행 → DepartureDetector가 이탈 추적 시작
            self.event_bus.publish("SHOCK_DETECTED", {
                "frame_idx": self.shock_frame_idx,
                "timestamp": self.shock_timestamp,
                "activate_time": self.activate_time,
            })

        elif char == "e" and self.is_active:
            # 충격 감지 종료
            elapsed = time.time() - self.activate_time
            self.is_active = False
            print(f"[ShockSimulator] 충격 감지 종료 (경과: {elapsed:.1f}초)")
            print(f"  후 버퍼: {len(self.shock_post_frames)} 프레임 수집")

            self.event_bus.publish("SHOCK_ENDED", {
                "frame_idx": data.get("frame_idx", 0),
                "duration": elapsed,
                "timestamp": self._current_timestamp,
            })

    def _on_frame_read(self, data: dict) -> None:
        """프레임 읽기 → 순환 버퍼에 저장 + post 버퍼 수집"""
        self._current_timestamp = data.get("timestamp", 0.0)
        self._current_frame_idx = data.get("frame_idx", 0)
        frame = data.get("frame")

        if frame is not None:
            # 항상 pre-buffer에 저장 (순환)
            self.pre_buffer.append({
                "frame": frame.copy(),
                "timestamp": self._current_timestamp,
                "frame_idx": self._current_frame_idx,
            })

            # 충격 감지 중이면 post-buffer 수집
            if self.is_active and self._post_remaining > 0:
                self.shock_post_frames.append({
                    "frame": frame.copy(),
                    "timestamp": self._current_timestamp,
                    "frame_idx": self._current_frame_idx,
                })
                self._post_remaining -= 1

    def get_shock_frames(self) -> dict:
        """충격 전후 프레임 반환 — EvidencePackage에서 사용

        Returns:
            {"pre_frames": [...], "post_frames": [...],
             "shock_frame_idx": int, "shock_timestamp": float}
        """
        return {
            "pre_frames": self.shock_pre_frames,
            "post_frames": self.shock_post_frames,
            "shock_frame_idx": self.shock_frame_idx,
            "shock_timestamp": self.shock_timestamp,
        }

    def draw_shock_overlay(self, frame: np.ndarray) -> np.ndarray:
        """충격 감지 상태 오버레이 그리기

        충격 감지 활성화 상태:
            - 화면 상단에 주황색 경고 바
            - "충격 감지! 이탈 차량 추적 시작" 텍스트
            - 경과 시간 표시

        비활성화 상태:
            - 좌상단에 작은 "[S] 충격 대기" 표시
        """
        if not self.is_active:
            return self._draw_standby_indicator(frame)

        h, w = frame.shape[:2]
        overlay = frame.copy()
        elapsed = time.time() - self.activate_time

        # ── 1. 상단 경고 바 (주황색) ──
        bar_height = 48
        warn_phase = int(time.time() / self._warn_interval) % 2

        # 주황색 (밝기 변화로 깜빡임 효과)
        if warn_phase == 0:
            bar_color = (0, 140, 255)     # 주황색 (BGR)
        else:
            bar_color = (0, 100, 220)     # 약간 어두운 주황

        cv2.rectangle(overlay, (0, 0), (w, bar_height), bar_color, -1)
        # 반투명 블렌딩 (70%)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # ── 2. 경고 텍스트 ──
        alert_text = "충격 감지! 이탈 차량 추적 시작"
        time_text = f"경과: {elapsed:.1f}초"

        frame = self._put_text(
            frame, alert_text,
            (w // 2 - 200, bar_height - 8),
            color=(255, 255, 255),
            font=self._font_large
        )

        frame = self._put_text(
            frame, time_text,
            (w - 180, bar_height - 12),
            color=(255, 255, 200),
            font=self._font
        )

        # ── 3. 화면 테두리 주황색 (경고 프레임) ──
        border_thickness = 4
        border_color = (0, 165, 255)  # 주황색 (BGR)
        if warn_phase == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, border_thickness)

        # ── 4. 좌상단 종료 안내 ──
        icon_text = "[E] 충격 종료"
        frame = self._put_text(
            frame, icon_text,
            (10, bar_height + 30),
            color=(100, 180, 255),
            font=self._font
        )

        # ── 5. post 버퍼 수집 상태 ──
        if self._post_remaining > 0:
            remaining_sec = self._post_remaining / self.fps if self.fps > 0 else 0
            buf_text = f"버퍼링: {len(self.shock_post_frames)}/{int(self.fps * self.post_buffer_sec)} ({remaining_sec:.0f}s)"
            frame = self._put_text(
                frame, buf_text,
                (10, bar_height + 55),
                color=(200, 200, 100),
                font=self._font
            )

        return frame

    def _draw_standby_indicator(self, frame: np.ndarray) -> np.ndarray:
        """충격 대기 상태 표시 — 좌상단 작은 안내 텍스트"""
        standby_text = "[S] 충격 대기"
        return self._put_text(
            frame, standby_text,
            (10, 30),
            color=(100, 200, 200),
            font=self._font
        )

    def _put_text(self, frame: np.ndarray, text: str, position: tuple,
                  color: tuple = (255, 255, 255), font=None) -> np.ndarray:
        """한글 텍스트를 프레임에 그리기 — PIL 사용"""
        use_font = font or self._font

        if use_font is not None:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # BGR → RGB 색상 변환
            rgb_color = (color[2], color[1], color[0])

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
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return frame

    # ── headless 모드용 프로그래밍 트리거 ──

    def trigger_shock(self, frame_idx: int = 0, timestamp: float = 0.0) -> None:
        """프로그래밍 방식으로 충격 감지 트리거 (headless 테스트용)"""
        if self.is_active:
            return

        self.is_active = True
        self.activate_time = time.time()
        self.shock_frame_idx = frame_idx
        self.shock_timestamp = timestamp

        self.shock_pre_frames = list(self.pre_buffer)
        self.shock_post_frames = []
        self._post_remaining = int(self.fps * self.post_buffer_sec)

        print(f"[ShockSimulator] 충격 감지! (프로그래밍 트리거, 프레임 #{frame_idx})")

        self.event_bus.publish("SHOCK_DETECTED", {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "activate_time": self.activate_time,
        })

    def trigger_shock_end(self, frame_idx: int = 0, timestamp: float = 0.0) -> None:
        """프로그래밍 방식으로 충격 종료 트리거 (headless 테스트용)"""
        if not self.is_active:
            return

        elapsed = time.time() - self.activate_time
        self.is_active = False

        print(f"[ShockSimulator] 충격 종료 (프로그래밍 트리거, 경과: {elapsed:.1f}초)")

        self.event_bus.publish("SHOCK_ENDED", {
            "frame_idx": frame_idx,
            "duration": elapsed,
            "timestamp": timestamp,
        })
