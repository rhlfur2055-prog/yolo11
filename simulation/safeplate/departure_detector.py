"""
departure_detector.py — 이탈 차량 감지기 (계단 B-2)

역할: 충격 감지 후 화면에서 사라지는 차량 = 이탈 의심
비유: CCTV 관제사가 사고 후 "어떤 차가 도망갔는지" 추적하는 것

동작 흐름:
    1. SHOCK_DETECTED 수신 → 현재 감지된 모든 차량 bbox 스냅샷 (사고 시점 기록)
    2. 이후 매 프레임마다 차량 추적 (IoU 기반 bbox 매칭)
    3. bbox가 화면 경계에 도달하거나 사라지면 "이탈" 판정
    4. 이탈 차량의 번호판 + 마지막 위치 + 이탈 방향 기록
    5. DEPARTURE_DETECTED 이벤트 발행

기존 코드 수정: 없음
Mock 데이터: 없음

이벤트 버스 연동:
    - 수신: "SHOCK_DETECTED", "SHOCK_ENDED", "detection_result", "frame_read"
    - 발행: "DEPARTURE_DETECTED"
"""

import time
from typing import Optional

import cv2
import numpy as np


class VehicleTrack:
    """개별 차량 추적 기록

    비유: 용의자 프로필 카드. 충격 시점에 화면에 있던 모든 차량을 기록하고,
         이후 행동(이동, 이탈)을 추적.

    Attributes:
        plate_text: 번호판 텍스트
        first_bbox: 최초 감지 위치 [x1, y1, x2, y2]
        last_bbox: 마지막 감지 위치
        bbox_history: bbox 이력 (이동 경로 추적)
        first_seen_time: 최초 감지 시각
        last_seen_time: 마지막 감지 시각
        last_seen_frame: 마지막 감지 프레임
        is_departed: 이탈 판정 여부
        departure_direction: 이탈 방향 ("left", "right", "top", "bottom", "vanished")
        departure_time: 이탈 판정 시각
        confidence: 최고 신뢰도
    """

    def __init__(self, plate_text: str, bbox: list, timestamp: float,
                 frame_idx: int, confidence: float):
        self.plate_text = plate_text
        self.first_bbox = bbox[:]
        self.last_bbox = bbox[:]
        self.bbox_history: list[list] = [bbox[:]]
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        self.last_seen_frame = frame_idx
        self.is_departed = False
        self.departure_direction: str = ""
        self.departure_time: Optional[float] = None
        self.departure_frame: Optional[int] = None
        self.confidence = confidence
        self.detection_count = 1

    def update(self, bbox: list, timestamp: float, frame_idx: int,
               confidence: float) -> None:
        """위치 갱신"""
        self.last_bbox = bbox[:]
        self.bbox_history.append(bbox[:])
        # 최근 50개만 유지
        if len(self.bbox_history) > 50:
            self.bbox_history = self.bbox_history[-50:]
        self.last_seen_time = timestamp
        self.last_seen_frame = frame_idx
        self.detection_count += 1
        if confidence > self.confidence:
            self.confidence = confidence

    def check_boundary_departure(self, frame_w: int, frame_h: int,
                                 margin_ratio: float = 0.05) -> Optional[str]:
        """bbox가 화면 경계에 도달했는지 확인

        Args:
            frame_w: 프레임 가로 픽셀
            frame_h: 프레임 세로 픽셀
            margin_ratio: 경계 판정 마진 비율 (5%)

        Returns:
            이탈 방향 문자열 또는 None
        """
        if len(self.last_bbox) < 4:
            return None

        x1, y1, x2, y2 = self.last_bbox
        margin_x = frame_w * margin_ratio
        margin_y = frame_h * margin_ratio

        # bbox 중심점
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if x1 <= margin_x:
            return "left"
        if x2 >= frame_w - margin_x:
            return "right"
        if y1 <= margin_y:
            return "top"
        if y2 >= frame_h - margin_y:
            return "bottom"

        return None

    def get_movement_vector(self) -> tuple:
        """이동 벡터 계산 (첫 bbox → 마지막 bbox 중심점 차이)

        Returns:
            (dx, dy) 이동 방향
        """
        if len(self.bbox_history) < 2:
            return (0, 0)

        first = self.bbox_history[0]
        last = self.bbox_history[-1]

        cx1 = (first[0] + first[2]) / 2
        cy1 = (first[1] + first[3]) / 2
        cx2 = (last[0] + last[2]) / 2
        cy2 = (last[1] + last[3]) / 2

        return (cx2 - cx1, cy2 - cy1)


def _compute_iou(bbox_a: list, bbox_b: list) -> float:
    """두 bbox의 IoU (Intersection over Union) 계산"""
    if len(bbox_a) < 4 or len(bbox_b) < 4:
        return 0.0

    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    area_a = abs((bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    area_b = abs((bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class DepartureDetector:
    """이탈 차량 감지기 — 충격 후 화면에서 사라지는 차량 추적

    동작 흐름:
        1. SHOCK_DETECTED 수신 → 감지 모드 활성화
        2. 이후 detection_result로 차량 추적
        3. 차량이 화면 경계 도달 또는 사라지면 이탈 판정
        4. DEPARTURE_DETECTED 발행

    Args:
        event_bus: SimulationFramework의 이벤트 버스
        frame_width: 영상 가로 픽셀
        frame_height: 영상 세로 픽셀
        vanish_timeout_sec: 화면에서 사라진 후 이 시간 경과 시 이탈 판정 (초)
        foreign_plate_counter_start: 외국 번호판 카운터 시작값
    """

    def __init__(self, event_bus, frame_width: int = 1920, frame_height: int = 1080,
                 vanish_timeout_sec: float = 3.0, foreign_plate_counter_start: int = 1):
        self.event_bus = event_bus
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.vanish_timeout_sec = vanish_timeout_sec

        # ── 상태 ──
        self.is_active = False
        self.shock_timestamp = 0.0

        # ── 차량 추적 목록 ──
        # {"번호판텍스트": VehicleTrack, ...}
        self.tracked_vehicles: dict[str, VehicleTrack] = {}

        # ── 이탈 판정 결과 ──
        self.departed_vehicles: list[dict] = []

        # ── 외국 번호판 카운터 ──
        self._foreign_counter = foreign_plate_counter_start

        # ── 현재 시각 추적 ──
        self._current_timestamp = 0.0
        self._current_frame_idx = 0

        # ── 이벤트 구독 ──
        self.event_bus.subscribe("SHOCK_DETECTED", self._on_shock_detected)
        self.event_bus.subscribe("SHOCK_ENDED", self._on_shock_ended)
        self.event_bus.subscribe("frame_read", self._on_frame_read)
        self.event_bus.subscribe("detection_result", self._on_detection_result)

        print(f"[DepartureDetector] 초기화 완료")
        print(f"  프레임: {frame_width}x{frame_height}")
        print(f"  이탈 판정: 미감지 {vanish_timeout_sec}초 후")

    def _on_shock_detected(self, data: dict) -> None:
        """충격 감지 → 이탈 추적 모드 시작"""
        if self.is_active:
            return

        self.is_active = True
        self.shock_timestamp = data.get("timestamp", 0.0)
        self.tracked_vehicles.clear()
        self.departed_vehicles.clear()

        print(f"\n  [DepartureDetector] 이탈 추적 모드 시작")

    def _on_shock_ended(self, data: dict) -> None:
        """충격 종료 — 추적은 계속하되 새 차량 등록은 중지"""
        # 이탈 추적은 계속 유지 (이미 추적 중인 차량의 이탈 감지)
        departed_count = len(self.departed_vehicles)
        tracking_count = len(self.tracked_vehicles)
        print(f"  [DepartureDetector] 충격 종료 — 추적 계속 ({tracking_count}대 감시, {departed_count}대 이탈)")

    def _on_frame_read(self, data: dict) -> None:
        """프레임 읽기 → 시각 갱신 + 프레임 크기 자동 감지"""
        self._current_timestamp = data.get("timestamp", 0.0)
        self._current_frame_idx = data.get("frame_idx", 0)

        # 프레임 크기 자동 감지 (최초 1회)
        frame = data.get("frame")
        if frame is not None and self._current_frame_idx == 0:
            h, w = frame.shape[:2]
            if w != self.frame_width or h != self.frame_height:
                self.frame_width = w
                self.frame_height = h

        # 추적 중인 차량의 이탈 체크 (매 프레임)
        if self.is_active:
            self._check_departures()

    def _on_detection_result(self, data: dict) -> None:
        """감지 결과 수신 → 차량 추적 업데이트"""
        if not self.is_active:
            return

        detections = data.get("detections", [])
        timestamp = data.get("timestamp", self._current_timestamp)
        frame_idx = data.get("frame_idx", self._current_frame_idx)

        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue

            plate_text = det.get("plate", "")
            confidence = det.get("confidence", 0.0)

            # OCR 결과가 없으면 외국 번호판으로 ID 부여
            if not plate_text:
                plate_text = f"FOREIGN_PLATE_{self._foreign_counter:03d}"
                self._foreign_counter += 1

            # ── 기존 차량 매칭 (IoU 기반) ──
            matched = False
            for track_id, track in self.tracked_vehicles.items():
                if track.is_departed:
                    continue
                iou = _compute_iou(track.last_bbox, bbox)
                if iou > 0.20:
                    # 같은 차량 — 위치 갱신
                    # 번호판 텍스트가 더 유효하면 업데이트
                    if plate_text and not track.plate_text.startswith("FOREIGN_PLATE"):
                        pass  # 이미 한국 번호판이 있으면 유지
                    elif plate_text and not plate_text.startswith("FOREIGN_PLATE"):
                        # 외국→한국 번호판으로 업데이트
                        old_id = track_id
                        track.plate_text = plate_text
                    track.update(bbox, timestamp, frame_idx, confidence)
                    matched = True
                    break

            if not matched:
                # 새 차량 등록
                track = VehicleTrack(plate_text, bbox, timestamp, frame_idx, confidence)
                self.tracked_vehicles[plate_text] = track

    def _check_departures(self) -> None:
        """추적 중인 차량의 이탈 여부 확인

        이탈 조건:
            1. bbox가 화면 경계에 도달 (5% 마진)
            2. vanish_timeout_sec 이상 미감지
        """
        current_time = self._current_timestamp
        newly_departed = []

        for plate_text, track in self.tracked_vehicles.items():
            if track.is_departed:
                continue

            # 조건 1: 화면 경계 도달
            direction = track.check_boundary_departure(
                self.frame_width, self.frame_height, margin_ratio=0.05
            )
            if direction:
                track.is_departed = True
                track.departure_direction = direction
                track.departure_time = current_time
                track.departure_frame = self._current_frame_idx
                newly_departed.append(track)
                continue

            # 조건 2: 일정 시간 미감지 → 사라짐
            gap = current_time - track.last_seen_time
            if gap > self.vanish_timeout_sec:
                track.is_departed = True
                track.departure_direction = "vanished"
                track.departure_time = current_time
                track.departure_frame = self._current_frame_idx
                newly_departed.append(track)

        # 이탈 판정된 차량 이벤트 발행
        for track in newly_departed:
            dx, dy = track.get_movement_vector()
            departure_info = {
                "plate": track.plate_text,
                "confidence": track.confidence,
                "first_bbox": track.first_bbox,
                "last_bbox": track.last_bbox,
                "departure_direction": track.departure_direction,
                "departure_time": track.departure_time,
                "departure_frame": track.departure_frame,
                "first_seen_time": track.first_seen_time,
                "last_seen_time": track.last_seen_time,
                "detection_count": track.detection_count,
                "movement_vector": (dx, dy),
                "timestamp": self._current_timestamp,
                "frame_idx": self._current_frame_idx,
            }
            self.departed_vehicles.append(departure_info)

            self.event_bus.publish("DEPARTURE_DETECTED", departure_info)

            dir_label = track.departure_direction
            print(f"\n  [DepartureDetector] 이탈 감지: {track.plate_text}")
            print(f"     방향: {dir_label} | 감지 횟수: {track.detection_count}")
            print(f"     이동 벡터: dx={dx:.0f}, dy={dy:.0f}")

    def draw_departure_overlay(self, frame: np.ndarray) -> np.ndarray:
        """이탈 차량 추적 상태 오버레이

        추적 중인 차량: 파란 bbox
        이탈 판정된 차량: 빨간 "DEPARTED" 라벨
        """
        if not self.is_active and not self.departed_vehicles:
            return frame

        overlay = frame.copy()
        h, w = frame.shape[:2]

        for plate_text, track in self.tracked_vehicles.items():
            bbox = track.last_bbox
            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            if track.is_departed:
                # 이탈 차량 — 빨간색
                color = (0, 0, 255)
                label = f"DEPARTED [{track.departure_direction}] {plate_text}"
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            else:
                # 추적 중 — 파란색
                color = (255, 150, 50)
                gap = self._current_timestamp - track.last_seen_time
                label = f"TRACKING {plate_text} ({gap:.1f}s ago)"
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # 라벨
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            label_y = max(y1 - 8, label_size[1] + 4)
            cv2.rectangle(overlay, (x1, label_y - label_size[1] - 4),
                          (x1 + label_size[0] + 4, label_y + 4), (0, 0, 0), -1)
            cv2.putText(overlay, label, (x1 + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        # ── 우하단 이탈 카운트 패널 ──
        if self.departed_vehicles:
            panel_text = f"DEPARTED: {len(self.departed_vehicles)} vehicles"
            text_size = cv2.getTextSize(panel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            px = w - text_size[0] - 16
            py = h - 80
            cv2.rectangle(frame, (px - 4, py - text_size[1] - 4),
                          (px + text_size[0] + 4, py + 4), (0, 0, 100), -1)
            cv2.putText(frame, panel_text, (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        return frame

    # ── 외부 API ──

    def get_departed_vehicles(self) -> list[dict]:
        """이탈 판정된 차량 목록"""
        return self.departed_vehicles

    def get_tracking_count(self) -> int:
        """현재 추적 중인 차량 수"""
        return sum(1 for t in self.tracked_vehicles.values() if not t.is_departed)

    def get_departed_count(self) -> int:
        """이탈 판정된 차량 수"""
        return len(self.departed_vehicles)
