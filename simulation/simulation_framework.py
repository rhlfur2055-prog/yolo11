"""
simulation_framework.py — 시뮬레이션 프레임워크 (계단 1)

역할: 영상 읽기 → YOLO 감지 → 이벤트 발행 → 오버레이 표시
비유: "눈(YOLO)"은 이미 있고, 이 파일은 "신경망(이벤트 버스)"을 연결하는 척추

기존 코드 수정: 없음
Mock 데이터: 없음 — 실제 PlateEnginePro 연결

실행: python -m simulation.simulation_framework movie/hiway.mp4
검증: python test_ocr_accuracy.py  # 12/12 통과 필수
"""

import sys
import time
from collections import defaultdict
from typing import Callable, Any

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# ──────────────────────────────────────────────
# 기존 엔진 import (수정 없이 그대로 사용)
# ──────────────────────────────────────────────
from plate_engine_pro import PlateEnginePro


# ═══════════════════════════════════════════════
# EventBus — 이벤트 기반 통신 허브
# ═══════════════════════════════════════════════
# 비유: 택배 분류 센터. 이벤트(택배)가 오면 등록된 콜백(배달 주소)으로 전달
# 시간복잡도: subscribe O(1), publish O(k) (k = 해당 이벤트 구독자 수)
class EventBus:
    """이벤트 버스 — 콜백 함수 등록/발행

    지원 이벤트:
        - "frame_read"         : 프레임 읽힐 때마다 발행 → data: {"frame": np.ndarray, "frame_idx": int, "timestamp": float}
        - "detection_result"   : YOLO 감지 완료 시 발행 → data: {"detections": list, "frame_idx": int}
        - "plate_recognized"   : 번호판 텍스트 인식 시 발행 → data: {"plate": str, "confidence": float, "bbox": list, "frame_idx": int}
        - "key_pressed"        : 키 입력 시 발행 → data: {"key": int, "char": str}
    """

    def __init__(self):
        # defaultdict(list) — 이벤트 이름별 콜백 리스트
        # {"detection_result": [callback1, callback2, ...], ...}
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_name: str, callback: Callable) -> None:
        """이벤트 리스너 등록

        Args:
            event_name: 구독할 이벤트 이름 (예: "detection_result")
            callback: 이벤트 발생 시 호출될 함수. data dict를 인자로 받음

        예시:
            bus.subscribe("plate_recognized", lambda data: print(data["plate"]))
        """
        self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable) -> None:
        """이벤트 리스너 제거

        Args:
            event_name: 구독 해제할 이벤트 이름
            callback: 제거할 콜백 함수
        """
        if event_name in self._subscribers:
            self._subscribers[event_name] = [
                cb for cb in self._subscribers[event_name] if cb is not callback
            ]

    def publish(self, event_name: str, data: Any = None) -> None:
        """이벤트 발행 — 등록된 모든 리스너에 데이터 전달

        Args:
            event_name: 발행할 이벤트 이름
            data: 전달할 데이터 (보통 dict)

        시간복잡도: O(k), k = 해당 이벤트 구독자 수
        """
        for callback in self._subscribers.get(event_name, []):
            try:
                callback(data)
            except Exception as e:
                print(f"[EventBus] '{event_name}' 콜백 에러: {e}")


# ═══════════════════════════════════════════════
# SimulationFramework — 메인 시뮬레이션 루프
# ═══════════════════════════════════════════════
# 비유: 영화관의 영사기. 프레임을 한 장씩 읽고, YOLO에게 보여주고, 결과를 화면에 그림
class SimulationFramework:
    """시뮬레이션 프레임워크 — 영상 읽기 + YOLO 탐지 + 오버레이 표시

    기존 PlateEnginePro를 그대로 사용.
    이벤트 버스를 통해 외부 모듈(사이렌 감지, 채증 등)이 결과를 받아감.

    Args:
        video_path: 입력 영상 파일 경로
        window_name: OpenCV 윈도우 이름
        display_scale: 화면 표시 배율 (1.0 = 원본 크기)
    """

    # 한글 폰트 경로 후보 (시스템에 있는 것을 자동 탐색)
    FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",  # Ubuntu
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",       # Ubuntu
        "C:/Windows/Fonts/malgun.ttf",                            # Windows 맑은 고딕
        "C:/Windows/Fonts/NanumGothicBold.ttf",                   # Windows 나눔고딕
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",             # macOS
    ]

    def __init__(self, video_path: str, window_name: str = "YOLO26 Simulation", display_scale: float = 1.0):
        # ── 영상 소스 ──
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── YOLO 엔진 (기존 코드 그대로 사용) ──
        self.engine = PlateEnginePro()

        # ── 이벤트 버스 ──
        self.event_bus = EventBus()

        # ── 표시 설정 ──
        self.window_name = window_name
        self.display_scale = display_scale

        # ── 외부 오버레이 콜백 ──
        # 사이렌 오버레이, 채증 오버레이 등 외부 모듈이 프레임에 그리기 위한 콜백 리스트
        # 각 콜백: (frame: np.ndarray) → np.ndarray
        self.overlay_callbacks: list[callable] = []

        # ── 상태 추적 ──
        self.frame_idx = 0          # 현재 프레임 인덱스
        self.is_running = False     # 루프 실행 중 여부
        self.is_paused = False      # 일시정지 여부

        # ── 한글 폰트 로딩 ──
        self._font = self._load_korean_font(size=20)
        self._font_small = self._load_korean_font(size=14)

        print(f"[SimulationFramework] 초기화 완료")
        print(f"  영상: {video_path}")
        print(f"  해상도: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps:.1f}, 총 프레임: {self.total_frames}")
        print(f"  조작: 'q'=종료, SPACE=일시정지/재개")

    def _load_korean_font(self, size: int = 20) -> ImageFont.FreeTypeFont | None:
        """한글 폰트 로딩 — 시스템에서 사용 가능한 폰트 자동 탐색

        cv2.putText()는 한글을 지원하지 않으므로 PIL로 대체.
        폰트를 못 찾으면 None 반환 → 영문 fallback 사용.

        시간복잡도: O(m), m = 폰트 후보 수 (상수, 최대 5개)
        """
        for font_path in self.FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(font_path, size)
                print(f"  폰트: {font_path}")
                return font
            except (IOError, OSError):
                continue

        print("  [경고] 한글 폰트를 찾을 수 없습니다. 영문으로 표시합니다.")
        return None

    def run(self) -> None:
        """메인 시뮬레이션 루프

        프레임 루프 흐름:
            1. cap.read() → 프레임 읽기
            2. event_bus.publish("frame_read", ...) → 프레임 구독자에게 전달
            3. engine.process_frame(frame) → YOLO 감지 + OCR
            4. event_bus.publish("detection_result", ...) → 감지 결과 전달
            5. 각 인식된 번호판마다 event_bus.publish("plate_recognized", ...)
            6. _draw_overlay() → bbox + 텍스트 오버레이
            7. cv2.imshow() → 화면 표시
            8. 키 입력 처리 ('q' 종료, SPACE 일시정지)

        시간복잡도: O(N * D), N = 총 프레임 수, D = 프레임당 감지 처리 시간
        """
        self.is_running = True
        frame_delay = int(1000 / self.fps)  # 프레임 간 대기 시간 (ms)

        print(f"\n[SimulationFramework] 시뮬레이션 시작")

        while self.is_running:
            # ── 일시정지 처리 ──
            if self.is_paused:
                key = cv2.waitKey(100) & 0xFF
                self._handle_key(key)
                continue

            # ── 1. 프레임 읽기 ──
            ret, frame = self.cap.read()
            if not ret:
                print(f"\n[SimulationFramework] 영상 끝 (총 {self.frame_idx} 프레임 처리)")
                break

            timestamp = self.frame_idx / self.fps  # 현재 시간 (초)

            # ── 2. 프레임 읽기 이벤트 발행 ──
            self.event_bus.publish("frame_read", {
                "frame": frame,
                "frame_idx": self.frame_idx,
                "timestamp": timestamp,
            })

            # ── 3. YOLO 감지 실행 (기존 엔진 그대로) ──
            detections = self.engine.process_frame(frame)
            # detections 형태: [{"plate": "경기76바7789", "confidence": 0.98, "bbox": [x1,y1,x2,y2], ...}, ...]

            # ── 4. 감지 결과 이벤트 발행 ──
            self.event_bus.publish("detection_result", {
                "detections": detections,
                "frame_idx": self.frame_idx,
                "timestamp": timestamp,
            })

            # ── 5. 개별 번호판 인식 이벤트 발행 ──
            for det in detections:
                plate_text = det.get("plate", "")
                if plate_text:  # OCR 결과가 있는 경우만
                    self.event_bus.publish("plate_recognized", {
                        "plate": plate_text,
                        "confidence": det.get("confidence", 0.0),
                        "bbox": det.get("bbox", []),
                        "frame_idx": self.frame_idx,
                        "timestamp": timestamp,
                    })

            # ── 6. 오버레이 그리기 ──
            display_frame = self._draw_overlay(frame, detections)

            # ── 6.5 외부 오버레이 콜백 실행 ──
            # 사이렌 오버레이, 채증 오버레이 등 외부 모듈이 등록한 콜백 순차 실행
            for overlay_cb in self.overlay_callbacks:
                try:
                    display_frame = overlay_cb(display_frame)
                except Exception as e:
                    print(f"[SimulationFramework] 오버레이 콜백 에러: {e}")

            # ── 7. 상태 표시줄 그리기 ──
            display_frame = self._draw_status_bar(display_frame, timestamp)

            # ── 8. 화면 배율 조정 ──
            if self.display_scale != 1.0:
                h, w = display_frame.shape[:2]
                new_w = int(w * self.display_scale)
                new_h = int(h * self.display_scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h))

            # ── 9. 화면 표시 ──
            cv2.imshow(self.window_name, display_frame)

            # ── 10. 키 입력 처리 ──
            key = cv2.waitKey(frame_delay) & 0xFF
            self._handle_key(key)

            self.frame_idx += 1

        self._cleanup()

    def _draw_overlay(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """프레임에 감지 결과 오버레이

        Args:
            frame: 원본 프레임 (BGR)
            detections: YOLO 감지 결과 리스트

        Returns:
            오버레이가 그려진 프레임 (원본 수정 방지를 위해 복사본)

        색상 규칙:
            - 초록(0,255,0): 유효 번호판 (OCR 결과 있음)
            - 노랑(0,255,255): 미확인 (bbox만 있고 OCR 미완료)

        시간복잡도: O(d), d = 감지된 객체 수
        """
        overlay = frame.copy()

        for det in detections:
            bbox = det.get("bbox", [])
            plate_text = det.get("plate", "")
            confidence = det.get("confidence", 0.0)

            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # ── 색상 결정 ──
            if plate_text:
                # 유효 번호판 — 초록색
                color = (0, 255, 0)
                label = f"{plate_text} ({confidence:.0%})"
            else:
                # 미확인 — 노란색
                color = (0, 255, 255)
                label = f"감지중... ({confidence:.0%})"

            # ── bbox 그리기 ──
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # ── 텍스트 배경 ──
            label_y = max(y1 - 8, 20)
            overlay = self._put_text_korean(overlay, label, (x1, label_y), color)

        return overlay

    def _draw_status_bar(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """화면 하단 상태 표시줄

        표시 내용:
            - 현재 시간 / 총 시간
            - 프레임 번호
            - 상태 (재생중 / 일시정지)
            - 조작 안내

        시간복잡도: O(1)
        """
        h, w = frame.shape[:2]
        bar_height = 36

        # 상태 표시줄 배경 (반투명 검정)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # 시간 정보
        total_time = self.total_frames / self.fps
        time_str = f"{timestamp:.1f}s / {total_time:.1f}s"
        frame_str = f"F:{self.frame_idx}/{self.total_frames}"

        # 상태
        if self.is_paused:
            status = "[일시정지]"
        else:
            status = "[재생중]"

        # 조작 안내
        controls = "q:종료 | SPACE:일시정지 | S:사이렌(계단2)"

        status_text = f" {status} {time_str} | {frame_str} | {controls}"
        frame = self._put_text_korean(
            frame, status_text, (4, h - 10),
            color=(200, 200, 200), font=self._font_small
        )

        return frame

    def _put_text_korean(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        color: tuple = (0, 255, 0),
        font: ImageFont.FreeTypeFont | None = None,
    ) -> np.ndarray:
        """한글 텍스트를 프레임에 그리기

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

        시간복잡도: O(1) (텍스트 길이에 비례하지만 실질적으로 상수)
        """
        use_font = font or self._font

        if use_font is not None:
            # PIL 방식 (한글 지원)
            # BGR → RGB 변환
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
            y_top = y - text_height  # 좌하단 → 좌상단 변환

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

            # RGB → BGR 변환 후 반환
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # fallback: cv2.putText (영문만)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return frame

    def _handle_key(self, key: int) -> None:
        """키 입력 처리

        지원 키:
            - 'q' 또는 ESC(27): 종료
            - SPACE(32): 일시정지/재개

        이벤트 버스로 모든 키 입력을 발행 → 외부 모듈(사이렌 트리거 등)이 수신 가능

        시간복잡도: O(1)
        """
        if key == 255:  # 키 입력 없음
            return

        # 키 이벤트 발행 (외부 모듈용)
        char = chr(key) if 32 <= key < 127 else ""
        self.event_bus.publish("key_pressed", {
            "key": key,
            "char": char,
            "frame_idx": self.frame_idx,
        })

        # 종료
        if key == ord('q') or key == 27:  # 'q' 또는 ESC
            print(f"\n[SimulationFramework] 사용자 종료 요청")
            self.is_running = False

        # 일시정지/재개
        elif key == 32:  # SPACE
            self.is_paused = not self.is_paused
            state = "일시정지" if self.is_paused else "재개"
            print(f"[SimulationFramework] {state}")

    def _cleanup(self) -> None:
        """리소스 정리 — 영상 캡처 해제, 윈도우 닫기

        시간복잡도: O(1)
        """
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[SimulationFramework] 정리 완료")


# ═══════════════════════════════════════════════
# CLI 진입점
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python -m simulation.simulation_framework <영상경로> [옵션]")
        print("예시:   python -m simulation.simulation_framework movie/hiway.mp4")
        print("        python -m simulation.simulation_framework movie/hiway.mp4 --goldentime")
        print("옵션:")
        print("  --scale <배율>    화면 배율 (기본 1.0)")
        print("  --goldentime      골든타임 모드 (S키=사이렌, E키=종료)")
        sys.exit(1)

    video_path = sys.argv[1]

    # 옵션: --scale 0.5 (화면 배율)
    scale = 1.0
    if "--scale" in sys.argv:
        idx = sys.argv.index("--scale")
        if idx + 1 < len(sys.argv):
            try:
                scale = float(sys.argv[idx + 1])
            except ValueError:
                print(f"[경고] --scale 값이 유효하지 않습니다. 기본값 1.0 사용")

    # 옵션: --goldentime (골든타임 2.0 사이렌 모드)
    goldentime_mode = "--goldentime" in sys.argv

    sim = SimulationFramework(video_path, display_scale=scale)

    # 기본 콜백 등록 (디버그용 — 콘솔 출력)
    sim.event_bus.subscribe(
        "plate_recognized",
        lambda data: print(f"  [번호판] {data['plate']} (신뢰도: {data['confidence']:.0%}) @ {data['timestamp']:.1f}s")
    )

    # ── 골든타임 모드: 사이렌 + 채증 + 거리 + 증거저장 연결 ──
    if goldentime_mode:
        from simulation.goldentime.siren_trigger import SirenTrigger
        from simulation.goldentime.plate_evidence import PlateEvidence
        from simulation.goldentime.distance_checker import DistanceChecker
        from simulation.goldentime.evidence_export import EvidenceExport

        siren = SirenTrigger(sim.event_bus)
        evidence = PlateEvidence(sim.event_bus, fps=sim.fps)
        distance = DistanceChecker(
            sim.event_bus,
            frame_width=sim.frame_width,
            frame_height=sim.frame_height,
        )
        export = EvidenceExport(
            sim.event_bus,
            output_dir="./evidence_output",
            fps=sim.fps,
            frame_width=sim.frame_width,
            frame_height=sim.frame_height,
        )

        # 오버레이 등록 (순서 중요: 사이렌 → 채증 → 거리 → 내보내기)
        sim.overlay_callbacks.append(siren.draw_siren_overlay)
        sim.overlay_callbacks.append(evidence.draw_evidence_overlay)
        sim.overlay_callbacks.append(distance.draw_distance_overlay)
        sim.overlay_callbacks.append(export.draw_export_overlay)

        print(f"\n[골든타임 모드] 활성화됨")
        print(f"  S키: 사이렌 감지 시작 → 번호판 추적 + 거리 판정")
        print(f"  E키: 사이렌 종료")
        print(f"  15초 이상 연속 감지 → 자동 채증")
        print(f"  가까움 {distance.config['violation_duration_sec']}초 이상 지속 → 거리 미확보 위반")
        print(f"  채증 완료 시 → 증거 패키지 자동 저장 (evidence_output/)")

    sim.run()
