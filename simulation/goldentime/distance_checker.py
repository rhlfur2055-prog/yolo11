"""
distance_checker.py — 거리 유지 판정 (계단 4)

역할: 채증 대상 번호판의 bbox 크기를 기반으로 긴급차량과의 거리를 근사 판정
      일정 크기 이상 bbox가 유지되면 "거리 미확보" 위반으로 판정

비유: 레이더 측정기. 실제 레이더 대신 "카메라에 얼마나 크게 찍히는지"로 거리를 추정.
     번호판이 크게 보인다 = 가까이 있다 = 거리 미확보

최민수님 스펙:
    "긴급차량과의 거리가 일정 수준(예: 5m) 이하로 유지되고 있는지?"
    → CCTV 시뮬레이션이므로 bbox 크기로 근사

근사 원리:
    - 카메라(= 긴급차량 시점)에서 촬영
    - 번호판 bbox 면적이 클수록 = 카메라에 가까움 = 긴급차량에 가까움
    - bbox 면적을 프레임 전체 면적 대비 비율로 정규화 (해상도 독립)
    - 비율이 임계값 이상 → "가까움" 판정
    - "가까움" 상태가 지속되면 → "거리 미확보" 위반

기존 코드 수정: 없음
Mock 데이터: 없음

이벤트 버스 연동:
    - 수신: "SIREN_DETECTED", "SIREN_ENDED", "detection_result", "frame_read"
    - 발행: "DISTANCE_VIOLATION", "DISTANCE_STATUS_UPDATE", "YIELD_DETECTED"

검증: python test_ocr_accuracy.py  # 12/12 통과 필수
"""

from collections import defaultdict
from typing import Optional

import cv2
import numpy as np


# ═══════════════════════════════════════════════
# 설정값 — config dict로 외부 조정 가능
# ═══════════════════════════════════════════════

DISTANCE_CONFIG = {
    # ── bbox 크기 기반 거리 임계값 ──
    # bbox 면적 / 프레임 면적 비율
    # 이 값 이상이면 "가까움" 판정
    #
    # 참고 추정:
    #   한국 번호판 실제 크기: 520mm x 110mm
    #   5m 거리에서 일반 CCTV(focal length ~6mm, 1080p): 번호판 ~100x21 px
    #   → 면적 비율 ≈ 100*21 / (1920*1080) ≈ 0.001 (0.1%)
    #   더 가까우면 비율이 커짐
    #
    # 시뮬레이션이므로 보수적으로 0.0008 (0.08%)부터 시작
    # 실제 테스트 후 조정 필요
    "close_ratio_threshold": 0.0008,

    # ── 거리 미확보 판정 기준 ──
    # "가까움" 상태가 이 시간 이상 지속되면 위반
    "violation_duration_sec": 5.0,

    # ── 판정 허용 간격 ──
    # 잠깐 멀어졌다 다시 가까워지면 연속으로 간주
    # 비유: 차가 잠깐 차선 변경했다가 다시 앞으로 온 경우
    "gap_tolerance_sec": 2.0,

    # ── 전방 ROI (전방 회피 의무 차량만 대상) ──
    # 이미지 하단 = 전방(카메라 앞). bbox 중심 y가 이 비율 이상이어야 거리 판정 대상.
    # 0.0 = 비활성(전체 프레임). 0.3 = 화면 하단 30% 영역(y >= height*0.7)만 전방으로 간주
    "front_roi_y_min_ratio": 0.0,

    # ── Trapezoid ROI (지평선 기준 원근 다각형) ──
    # True면 bbox 중심이 사다리꼴(도로 영역) 안에 있을 때만 거리 판정
    "use_trapezoid_roi": False,
    "trapezoid_horizon_ratio": 0.35,   # 지평선 y = H * ratio (상단 변)
    "trapezoid_top_margin": 0.10,      # 상단 좌우 마진 (폭의 10%씩 안쪽)

    # ── 핀홀 카메라 거리 d = (f * W) / w ──
    # use_pinhole_distance=True 이고 focal_length_px > 0 이면 bbox_ratio 대신 거리(m)로 판정
    "use_pinhole_distance": False,
    "focal_length_px": 0.0,       # 픽셀 단위 초점거리 f (1080p: 1000~2000, 캘리브레이션 권장)
    "plate_width_m": 0.52,       # W: 번호판 실제 폭 (m). 한국 520mm
    "close_distance_m": 5.0,     # 이 거리(m) 이하면 "가까움" (5m 미확보 위반)

    # ── 표시용 거리 구간 ──
    # bbox 비율에 따른 대략적 거리 표시 (참고용, 정확하지 않음)
    "distance_levels": [
        # (비율 이상, 라벨, 색상 BGR)
        (0.005,  "~1m",  (0, 0, 255)),     # 매우 가까움 — 빨강
        (0.002,  "~3m",  (0, 100, 255)),    # 가까움 — 주황
        (0.0008, "~5m",  (0, 200, 255)),    # 임계 — 노랑
        (0.0004, "~10m", (0, 255, 200)),    # 보통 — 연두
        (0.0,    ">10m", (0, 255, 0)),      # 안전 — 초록
    ],
}


# ═══════════════════════════════════════════════
# PlateDistanceRecord — 개별 번호판 거리 추적
# ═══════════════════════════════════════════════

class PlateDistanceRecord:
    """개별 번호판의 거리 상태 추적

    비유: 과속 카메라의 속도 기록.
         "이 차가 얼마나 오래 가까이 붙어있었나?"를 기록.

    Attributes:
        plate_text: 번호판 텍스트
        bbox_ratio_history: 최근 bbox 비율 이력 (이동평균용)
        is_close: 현재 "가까움" 상태인지
        close_start_time: "가까움" 시작 시각
        close_duration: "가까움" 지속 시간 (초)
        is_violation: 위반 판정 여부
        violation_time: 위반 판정 시각
        last_bbox_ratio: 마지막 bbox 면적 비율
        last_seen_time: 마지막 감지 시각
    """

    # 이동평균 윈도우 크기 (프레임 수)
    SMOOTHING_WINDOW = 10
    # 후진(양보) 판정용 bbox 이력 최대 길이
    BBOX_HISTORY_LEN = 15
    # bbox 면적 증가율: 이 비율(1초당) 이상이면 "후진으로 카메라 쪽 접근" = 양보로 간주
    YIELD_AREA_RATE_MIN = 0.02  # 2% per second
    # 화면 하단 이동: bbox 중심 y가 이 픽셀 이상 증가하면 "후진(카메라 쪽 접근)" 보조 판정
    YIELD_CENTER_Y_MIN_PX = 5.0

    # 고의적 길막 시맨틱: 가까움 + 이동 거의 없음(velocity) 지속 시 suspected_blocking
    BLOCKING_VELOCITY_THRESHOLD = 2.0   # px/frame 이하면 "정지에 가까움"
    BLOCKING_DURATION_SEC = 2.0         # 이 시간 이상 지속 시 길막 의심

    def __init__(self, plate_text: str, timestamp: float):
        self.plate_text = plate_text
        self.bbox_ratio_history: list[float] = []
        self.bbox_history: list[tuple[list, float]] = []  # (bbox, timestamp)
        self.is_close = False
        self.close_start_time: Optional[float] = None
        self.close_duration = 0.0
        self.is_violation = False
        self.violation_time: Optional[float] = None
        self.last_bbox_ratio = 0.0
        self.last_seen_time = timestamp
        self.last_distance_label = ""
        self.last_distance_color = (0, 255, 0)
        self.is_yielding = False  # 후진으로 양보 중이면 True, 이 구간은 위반 누적 안 함
        self._yield_emitted = False  # YIELD_DETECTED 이미 발행했는지
        # 고의적 길막 시맨틱 (velocity/area_rate 기반)
        self.velocity = (0.0, 0.0)  # (vx, vy) px/frame
        self.area_rate = 0.0
        self.suspected_blocking = False
        self.blocking_start_time: Optional[float] = None
        self._blocking_emitted = False  # BLOCKING_SUSPECTED 1회만 발행

    def update_bbox(self, bbox_ratio: float, timestamp: float,
                    close_threshold: float, gap_tolerance: float,
                    bbox: Optional[list] = None) -> None:
        """bbox 비율 업데이트 — 거리 상태 갱신.

        Args:
            bbox_ratio: bbox 면적 / 프레임 면적 비율
            timestamp: 현재 영상 시각 (초)
            close_threshold: "가까움" 판정 임계값
            gap_tolerance: 미감지 허용 간격 (초)
            bbox: [x1,y1,x2,y2] (후진 양보 판정용, 선택)

        로직:
            1. bbox 이력 저장 → 면적 증가율(Δarea/Δt)로 후진(양보) 판정
            2. 이동평균 계산 (노이즈 제거)
            3. 평균값 >= 임계값 → "가까움" (단, 양보 중이면 위반 누적 안 함)

        시간복잡도: O(W), W = SMOOTHING_WINDOW (상수, 10)
        """
        # 시간 간격 체크 — 너무 오래 안 보였으면 리셋
        gap = timestamp - self.last_seen_time
        if gap > gap_tolerance:
            self.bbox_ratio_history.clear()
            self.bbox_history.clear()
            self.is_close = False
            self.close_start_time = None
            self.close_duration = 0.0
            self.is_yielding = False
            self._yield_emitted = False
            self.blocking_start_time = None
            self.suspected_blocking = False
            self._blocking_emitted = False

        self.last_seen_time = timestamp
        self.last_bbox_ratio = bbox_ratio

        # ── bbox 이력 저장 → 후진(양보) 판정: 면적이 증가하면 카메라 쪽 접근 = 후진 ──
        if bbox is not None and len(bbox) >= 4:
            self.bbox_history.append((list(bbox), timestamp))
            if len(self.bbox_history) > self.BBOX_HISTORY_LEN:
                self.bbox_history = self.bbox_history[-self.BBOX_HISTORY_LEN:]
            self._update_yielding()

        # ── 이동평균 ──
        self.bbox_ratio_history.append(bbox_ratio)
        if len(self.bbox_ratio_history) > self.SMOOTHING_WINDOW:
            self.bbox_ratio_history = self.bbox_ratio_history[-self.SMOOTHING_WINDOW:]

        avg_ratio = sum(self.bbox_ratio_history) / len(self.bbox_ratio_history)

        # ── "가까움" 판정 ──
        was_close = self.is_close
        self.is_close = avg_ratio >= close_threshold

        if self.is_yielding:
            # 후진 양보 중에는 위반 누적 안 함
            self.close_start_time = None
            self.close_duration = 0.0
            self.is_close = False
        elif self.is_close and not was_close:
            self.close_start_time = timestamp
            self.close_duration = 0.0
        elif self.is_close and was_close:
            if self.close_start_time is not None:
                self.close_duration = timestamp - self.close_start_time
        elif not self.is_close and was_close:
            self.close_start_time = None
            self.close_duration = 0.0

    def check_violation(self, violation_duration: float, timestamp: float) -> bool:
        """위반 판정 체크

        "가까움" 지속 시간이 violation_duration 이상이면 위반.
        한번 위반 판정되면 유지 (is_violation = True).

        Args:
            violation_duration: 위반 기준 시간 (초)
            timestamp: 현재 시각

        Returns:
            True면 이번 호출에서 새로 위반 판정됨 (최초 1회만 True)

        시간복잡도: O(1)
        """
        if self.is_violation:
            return False  # 이미 판정됨

        if self.is_close and self.close_duration >= violation_duration:
            self.is_violation = True
            self.violation_time = timestamp
            return True  # 새로 판정됨

        return False

    def _update_yielding(self) -> None:
        """bbox 면적 증가 + 화면 하단 이동으로 후진(양보) 여부 갱신.

        조건: (1) Δarea/Δt 비율 >= YIELD_AREA_RATE_MIN
              (2) bbox 중심 y가 증가 (화면에서 아래로 이동 = 카메라 쪽 접근)
        → 후진으로 경로 확보 = 양보로 간주, False Alarm 제거.

        시간복잡도: O(H), H = BBOX_HISTORY_LEN (상수)
        """
        if len(self.bbox_history) < 3:
            self.is_yielding = False
            return
        first_bbox, t0 = self.bbox_history[0]
        last_bbox, t1 = self.bbox_history[-1]
        delta_t = t1 - t0
        if delta_t < 0.3:
            self.is_yielding = False
            return
        a0 = max(1.0, abs((first_bbox[2] - first_bbox[0]) * (first_bbox[3] - first_bbox[1])))
        a1 = abs((last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1]))
        rate = (a1 - a0) / delta_t / a0  # 1초당 면적 증가 비율
        cy_first = (first_bbox[1] + first_bbox[3]) / 2
        cy_last = (last_bbox[1] + last_bbox[3]) / 2
        moving_down = (cy_last - cy_first) >= self.YIELD_CENTER_Y_MIN_PX
        self.is_yielding = rate >= self.YIELD_AREA_RATE_MIN and moving_down

    def update_velocity_blocking(self, velocity: tuple, area_rate: float,
                                 is_close: bool, timestamp: float) -> None:
        """velocity/area_rate 반영 후 고의적 길막 의심 여부 갱신.

        가까운데 이동이 거의 없으면(velocity < threshold) 일정 시간 지속 시 suspected_blocking.
        """
        self.velocity = velocity
        self.area_rate = area_rate
        vx, vy = velocity[0], velocity[1]
        speed = abs(vx) + abs(vy)
        if is_close and speed <= self.BLOCKING_VELOCITY_THRESHOLD:
            if self.blocking_start_time is None:
                self.blocking_start_time = timestamp
            duration = timestamp - self.blocking_start_time
            self.suspected_blocking = duration >= self.BLOCKING_DURATION_SEC
        else:
            self.blocking_start_time = None
            self.suspected_blocking = False

    def is_stale(self, current_time: float, gap_tolerance: float) -> bool:
        """더 이상 감지되지 않는지 확인

        시간복잡도: O(1)
        """
        return (current_time - self.last_seen_time) > gap_tolerance


# ═══════════════════════════════════════════════
# DistanceChecker — 거리 판정 관리자
# ═══════════════════════════════════════════════

class DistanceChecker:
    """bbox 크기 기반 긴급차량-전방차량 거리 판정

    동작 흐름:
        1. SIREN_DETECTED 수신 → 거리 추적 시작
        2. 매 프레임 감지 결과 → 번호판별 bbox 비율 계산 → 거리 상태 갱신
        3. "가까움" 상태가 violation_duration 이상 지속 → DISTANCE_VIOLATION 발행
        4. SIREN_ENDED 수신 → 추적 중단 (위반 기록은 보존)

    PlateEvidence와의 관계:
        - PlateEvidence: "이 번호판이 15초 이상 보이니 채증하자" (시간 기준)
        - DistanceChecker: "이 번호판이 가까이 붙어있으니 위반이다" (거리 기준)
        - 두 모듈은 독립적으로 동작. 결과는 evidence_export (계단5)에서 합침.

    Args:
        event_bus: SimulationFramework의 이벤트 버스
        frame_width: 영상 가로 픽셀 수
        frame_height: 영상 세로 픽셀 수
        config: 설정 dict (DISTANCE_CONFIG 참조)
    """

    def __init__(self, event_bus, frame_width: int = 1920, frame_height: int = 1080,
                 config: dict = None):
        self.event_bus = event_bus
        self.config = config or DISTANCE_CONFIG.copy()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height

        # ── 상태 ──
        self.is_active = False  # 사이렌 이후 활성 상태

        # ── 번호판별 거리 기록 ──
        # {"경기76바7789": PlateDistanceRecord, ...}
        self.distance_records: dict[str, PlateDistanceRecord] = {}

        # ── 위반 판정 목록 ──
        # {"경기76바7789": PlateDistanceRecord, ...} (is_violation == True)
        self.violations: dict[str, PlateDistanceRecord] = {}

        # ── 현재 시각 ──
        self._current_timestamp = 0.0
        self._current_frame_idx = 0

        # ── 이벤트 구독 ──
        self.event_bus.subscribe("SIREN_DETECTED", self._on_siren_detected)
        self.event_bus.subscribe("SIREN_ENDED", self._on_siren_ended)
        self.event_bus.subscribe("frame_read", self._on_frame_read)
        self.event_bus.subscribe("detection_result", self._on_detection_result)

        threshold_pct = self.config["close_ratio_threshold"] * 100
        print(f"[DistanceChecker] 초기화 완료")
        print(f"  프레임: {frame_width}x{frame_height} ({self.frame_area:,} px)")
        print(f"  가까움 임계값: bbox 비율 >= {threshold_pct:.2f}%")
        print(f"  위반 기준: 가까움 {self.config['violation_duration_sec']}초 이상 지속")
        if self.config.get("front_roi_y_min_ratio", 0.0) > 0:
            print(f"  전방 ROI: 하단 {self.config['front_roi_y_min_ratio']*100:.0f}% 영역만 판정 대상")
        if self.config.get("use_pinhole_distance") and self.config.get("focal_length_px"):
            print(f"  핀홀 거리: f={self.config['focal_length_px']}px, W={self.config.get('plate_width_m', 0.52)}m, 가까움 <={self.config.get('close_distance_m', 5.0)}m")
        if self.config.get("use_trapezoid_roi"):
            print(f"  Trapezoid ROI: 지평선 {self.config.get('trapezoid_horizon_ratio', 0.35)*100:.0f}%, 상단마진 {self.config.get('trapezoid_top_margin', 0.1)*100:.0f}%")

    # ───────────────────────────────────────────
    # 이벤트 핸들러
    # ───────────────────────────────────────────

    def _on_siren_detected(self, data: dict) -> None:
        """사이렌 감지 → 거리 추적 시작

        시간복잡도: O(1)
        """
        if self.is_active:
            return

        self.is_active = True
        self.distance_records.clear()
        print(f"  [DistanceChecker] 거리 추적 시작")

    def _on_siren_ended(self, data: dict) -> None:
        """사이렌 종료 → 거리 추적 중단

        위반 기록은 보존 (evidence_export에서 사용).

        시간복잡도: O(1)
        """
        self.is_active = False
        violation_count = len(self.violations)
        print(f"  [DistanceChecker] 거리 추적 종료 (위반: {violation_count}건)")

    def _on_frame_read(self, data: dict) -> None:
        """프레임 읽기 → 시각 갱신 + 프레임 크기 자동 감지

        최초 프레임에서 frame_area를 갱신 (초기화 시 모를 수 있으므로).

        시간복잡도: O(1)
        """
        self._current_timestamp = data.get("timestamp", 0.0)
        self._current_frame_idx = data.get("frame_idx", 0)

        # 프레임 크기 자동 감지 (최초 1회)
        frame = data.get("frame")
        if frame is not None and self._current_frame_idx == 0:
            h, w = frame.shape[:2]
            if w != self.frame_width or h != self.frame_height:
                self.frame_width = w
                self.frame_height = h
                self.frame_area = w * h

    def _on_detection_result(self, data: dict) -> None:
        """감지 결과 수신 → bbox 크기 기반 거리 판정

        각 감지된 번호판에 대해:
            1. bbox 면적 비율 계산
            2. PlateDistanceRecord 갱신
            3. 위반 체크

        시간복잡도: O(d), d = 감지된 객체 수
        """
        if not self.is_active:
            return

        detections = data.get("detections", [])
        timestamp = data.get("timestamp", self._current_timestamp)

        close_threshold = self.config["close_ratio_threshold"]
        gap_tolerance = self.config["gap_tolerance_sec"]
        violation_duration = self.config["violation_duration_sec"]

        for det in detections:
            plate_text = det.get("plate", "")
            bbox = det.get("bbox", [])

            if not plate_text or len(bbox) < 4:
                continue

            # ── 전방 ROI 필터: 화면 하단(전방)에 있는 bbox만 거리 판정 대상 ──
            if self.config.get("front_roi_y_min_ratio", 0.0) > 0:
                if not self._is_in_front_roi(bbox):
                    continue

            # ── Trapezoid ROI: 지평선 기준 원근 다각형 안의 bbox만 거리 판정 ──
            if self.config.get("use_trapezoid_roi", False):
                if not self._is_inside_trapezoid_roi(bbox):
                    continue

            # ── bbox 면적 비율 계산 ──
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            bbox_area = abs((x2 - x1) * (y2 - y1))
            bbox_ratio = bbox_area / self.frame_area if self.frame_area > 0 else 0.0

            # ── 거리 레코드 갱신 ──
            if plate_text not in self.distance_records:
                self.distance_records[plate_text] = PlateDistanceRecord(plate_text, timestamp)

            record = self.distance_records[plate_text]
            record.update_bbox(bbox_ratio, timestamp, close_threshold, gap_tolerance, bbox=bbox)

            # ── 핀홀 거리 d = f*W/w (선택): 설정 시 비율 대신 거리(m)로 가까움 판정 ──
            if self.config.get("use_pinhole_distance") and self.config.get("focal_length_px"):
                d_m = self._pinhole_distance_m(bbox)
                close_distance_m = self.config.get("close_distance_m", 5.0)
                setattr(record, "last_distance_m", d_m)
                record.is_close = d_m <= close_distance_m
                if record.is_close and record.close_start_time is None:
                    record.close_start_time = timestamp
                    record.close_duration = 0.0
                elif record.is_close and record.close_start_time is not None:
                    record.close_duration = timestamp - record.close_start_time
                else:
                    record.close_start_time = None
                    record.close_duration = 0.0
                record.last_distance_label, record.last_distance_color = self._get_distance_level_pinhole(d_m)
                if record.is_yielding:
                    record.is_close = False
                    record.close_start_time = None
                    record.close_duration = 0.0
            else:
                # 표시용 거리 라벨 갱신 (비율 기반)
                record.last_distance_label, record.last_distance_color = self._get_distance_level(bbox_ratio)

            # ── velocity/area_rate 반영 (고의적 길막 시맨틱, 최종 is_close 기준) ──
            velocity = det.get("velocity", (0.0, 0.0))
            if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
                v_tuple = (float(velocity[0]), float(velocity[1]))
            else:
                v_tuple = (0.0, 0.0)
            area_rate = float(det.get("area_rate", 0.0))
            record.update_velocity_blocking(
                v_tuple, area_rate,
                record.is_close,
                timestamp,
            )

            # ── 후진 양보 감지 시 YIELD_DETECTED 발행 (최초 1회) ──
            if record.is_yielding and not getattr(record, "_yield_emitted", False):
                record._yield_emitted = True
                self._publish_yield_detected(record)
            elif not record.is_yielding:
                record._yield_emitted = False

            # ── 위반 체크 (양보 중이면 record.is_close가 False라 위반 안 됨) ──
            newly_violated = record.check_violation(violation_duration, timestamp)
            if newly_violated:
                self.violations[plate_text] = record
                self._publish_violation(record, bbox_ratio)

            # ── 고의적 길막 의심 시 BLOCKING_SUSPECTED 발행 (최초 1회) ──
            if record.suspected_blocking and not getattr(record, "_blocking_emitted", False):
                record._blocking_emitted = True
                self._publish_blocking_suspected(record)
            elif not record.suspected_blocking:
                record._blocking_emitted = False

        # ── stale 기록 정리 ──
        self._cleanup_stale_records(timestamp)

        # ── 상태 이벤트 발행 ──
        self.event_bus.publish("DISTANCE_STATUS_UPDATE", {
            "records": {
                text: {
                    "bbox_ratio": rec.last_bbox_ratio,
                    "is_close": rec.is_close,
                    "close_duration": rec.close_duration,
                    "is_violation": rec.is_violation,
                    "distance_label": rec.last_distance_label,
                    "is_yielding": getattr(rec, "is_yielding", False),
                    "suspected_blocking": getattr(rec, "suspected_blocking", False),
                }
                for text, rec in self.distance_records.items()
            },
            "violation_count": len(self.violations),
            "timestamp": timestamp,
        })

    # ───────────────────────────────────────────
    # 내부 로직
    # ───────────────────────────────────────────

    def _get_trapezoid_polygon(self) -> np.ndarray:
        """지평선 기준 Trapezoid(사다리꼴) 꼭짓점 [하단좌, 하단우, 상단우, 상단좌] (OpenCV 순)."""
        W, H = self.frame_width, self.frame_height
        hr = self.config.get("trapezoid_horizon_ratio", 0.35)
        margin = self.config.get("trapezoid_top_margin", 0.10)
        y_top = H * hr
        x_left = W * margin
        x_right = W * (1.0 - margin)
        # (x, y) 순. 하단 전체 폭, 상단은 마진만큼 안쪽
        return np.array([
            [0, H], [W, H], [x_right, y_top], [x_left, y_top]
        ], dtype=np.float32)

    def _is_inside_trapezoid_roi(self, bbox: list) -> bool:
        """bbox 중심이 지평선 기준 Trapezoid(도로 원근 영역) 안에 있는지."""
        if len(bbox) < 4:
            return False
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        poly = self._get_trapezoid_polygon()
        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0

    def _is_in_front_roi(self, bbox: list) -> bool:
        """전방 ROI 필터: bbox 중심이 화면 하단(전방) 영역에 있는지 확인.

        이미지 좌표계에서 y가 크면 화면 하단 = 카메라 전방.
        front_roi_y_min_ratio=0.3 이면 y >= height*0.7 인 영역만 전방.

        시간복잡도: O(1)
        """
        if len(bbox) < 4:
            return False
        _, y1, _, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cy = (y1 + y2) / 2
        ratio = self.config.get("front_roi_y_min_ratio", 0.0)
        if ratio <= 0:
            return True
        return cy >= self.frame_height * (1.0 - ratio)

    def _pinhole_distance_m(self, bbox: list) -> float:
        """핀홀 카메라 모델 거리: d = f*W/w (m).

        f: 초점거리(px), W: 번호판 실제 폭(m), w: 이미지 내 번호판 폭(px).
        w=0이면 매우 가까움으로 간주하여 0.0 반환.

        시간복잡도: O(1)
        """
        if len(bbox) < 4:
            return 0.0
        x1, x2 = bbox[0], bbox[2]
        w_px = max(1.0, abs(x2 - x1))
        f = self.config.get("focal_length_px", 0.0)
        W = self.config.get("plate_width_m", 0.52)
        if f <= 0:
            return 999.0
        return (f * W) / w_px

    def _get_distance_level_pinhole(self, d_m: float) -> tuple[str, tuple]:
        """핀홀 거리(m)에 따른 표시용 라벨 + 색상.

        시간복잡도: O(1)
        """
        if d_m <= 1.0:
            return "~1m", (0, 0, 255)
        if d_m <= 3.0:
            return "~3m", (0, 100, 255)
        if d_m <= 5.0:
            return "~5m", (0, 200, 255)
        if d_m <= 10.0:
            return "~10m", (0, 255, 200)
        return ">10m", (0, 255, 0)

    def _get_distance_level(self, bbox_ratio: float) -> tuple[str, tuple]:
        """bbox 비율에 따른 대략적 거리 라벨 + 색상

        distance_levels 설정에서 매칭.
        비유: 체온계의 색깔 표시 (파랑=정상, 빨강=고열)

        Args:
            bbox_ratio: bbox 면적 / 프레임 면적 비율

        Returns:
            (라벨 문자열, BGR 색상 튜플)

        시간복잡도: O(L), L = distance_levels 수 (상수, 5개)
        """
        for threshold, label, color in self.config["distance_levels"]:
            if bbox_ratio >= threshold:
                return label, color
        return ">10m", (0, 255, 0)

    def _publish_violation(self, record: PlateDistanceRecord, bbox_ratio: float) -> None:
        """거리 미확보 위반 이벤트 발행

        시간복잡도: O(k), k = DISTANCE_VIOLATION 구독자 수
        """
        self.event_bus.publish("DISTANCE_VIOLATION", {
            "plate": record.plate_text,
            "bbox_ratio": bbox_ratio,
            "close_duration": record.close_duration,
            "distance_label": record.last_distance_label,
            "timestamp": self._current_timestamp,
            "frame_idx": self._current_frame_idx,
        })

        ratio_pct = bbox_ratio * 100
        print(f"\n  ⚠️ [DistanceChecker] 거리 미확보 위반: {record.plate_text}")
        print(f"     거리 추정: {record.last_distance_label} (bbox 비율: {ratio_pct:.3f}%)")
        print(f"     가까움 지속: {record.close_duration:.1f}초 (기준: {self.config['violation_duration_sec']}초)")
        print(f"     → DISTANCE_VIOLATION 이벤트 발행")

    def _publish_yield_detected(self, record: PlateDistanceRecord) -> None:
        """후진 양보 감지 이벤트 발행.

        bbox 면적 증가율로 '후진으로 경로 확보'를 추정한 경우 발행.
        해당 구간에는 DISTANCE_VIOLATION을 누적하지 않음.

        시간복잡도: O(k), k = YIELD_DETECTED 구독자 수
        """
        self.event_bus.publish("YIELD_DETECTED", {
            "plate": record.plate_text,
            "timestamp": self._current_timestamp,
            "frame_idx": self._current_frame_idx,
        })
        print(f"\n  🟢 [DistanceChecker] 후진 양보 감지: {record.plate_text} (YIELD_DETECTED)")

    def _publish_blocking_suspected(self, record: PlateDistanceRecord) -> None:
        """고의적 길막 의심 이벤트 발행.

        가까움 + velocity 거의 0인 상태가 BLOCKING_DURATION_SEC 이상 지속 시 1회 발행.
        """
        self.event_bus.publish("BLOCKING_SUSPECTED", {
            "plate": record.plate_text,
            "velocity": record.velocity,
            "close_duration": record.close_duration,
            "timestamp": self._current_timestamp,
            "frame_idx": self._current_frame_idx,
        })
        print(f"\n  🚧 [DistanceChecker] 고의적 길막 의심: {record.plate_text} (BLOCKING_SUSPECTED)")

    def _cleanup_stale_records(self, current_time: float) -> None:
        """오래된 거리 기록 정리

        위반 판정된 기록은 보존.

        시간복잡도: O(R), R = 추적 중인 기록 수
        """
        gap_tolerance = self.config["gap_tolerance_sec"]
        stale_keys = [
            text for text, rec in self.distance_records.items()
            if rec.is_stale(current_time, gap_tolerance) and not rec.is_violation
        ]
        for key in stale_keys:
            del self.distance_records[key]

    # ───────────────────────────────────────────
    # 오버레이
    # ───────────────────────────────────────────

    def draw_distance_overlay(self, frame: np.ndarray) -> np.ndarray:
        """거리 판정 오버레이 — 각 번호판에 거리 정보 표시

        표시 내용:
            - 비활성 → 아무것도 안 그림
            - 각 번호판 bbox 옆에 거리 라벨 + 색상 인디케이터
            - 위반 번호판 → 빨간 "VIOLATION" 경고
            - 우상단 거리 판정 요약 패널

        Args:
            frame: 대상 프레임 (BGR)

        Returns:
            오버레이가 추가된 프레임

        시간복잡도: O(R), R = 추적 중인 기록 수
        """
        if not self.is_active and not self.violations:
            return frame

        overlay = frame.copy()
        h, w = frame.shape[:2]

        for plate_text, record in self.distance_records.items():
            # bbox 위치를 PlateEvidence의 tracked_plates에서 가져와야 하지만,
            # detection_result에서 직접 받지 않으므로,
            # PLATE_TRACKING_UPDATE 이벤트의 bbox를 활용하거나
            # 여기서는 가장 최근 감지 결과의 위치를 표시
            #
            # → 실제로는 draw_evidence_overlay가 bbox를 그리므로,
            #   여기서는 우상단 패널만 그림
            pass

        # ── 우상단 거리 판정 패널 ──
        frame = self._draw_distance_panel(frame)

        return frame

    def _draw_distance_panel(self, frame: np.ndarray) -> np.ndarray:
        """우상단 거리 판정 요약 패널

        표시 내용:
            - 각 추적 번호판의 거리 라벨 + 가까움 지속 시간
            - 위반 번호판은 빨간 강조

        시간복잡도: O(R), R = 추적 중인 기록 수
        """
        h, w = frame.shape[:2]

        if not self.distance_records:
            return frame

        # 사이렌 오버레이(44px) 아래, 우측에 배치
        panel_y_start = 50
        panel_x_end = w - 8
        line_height = 18
        lines = []

        # 헤더
        viol_count = len(self.violations)
        lines.append(("DISTANCE CHECK", (200, 200, 200), False))

        # 각 번호판 (연속 시간 기준 정렬)
        sorted_records = sorted(
            self.distance_records.values(),
            key=lambda r: r.close_duration,
            reverse=True,
        )[:5]

        for rec in sorted_records:
            ratio_pct = rec.last_bbox_ratio * 100
            label = rec.last_distance_label
            color = rec.last_distance_color

            if rec.is_violation:
                text = f"  {rec.plate_text}: {label} VIOLATION!"
                is_violation = True
            elif getattr(rec, "is_yielding", False):
                text = f"  {rec.plate_text}: {label} [YIELD]"
                is_violation = False
            elif rec.is_close:
                text = f"  {rec.plate_text}: {label} [{rec.close_duration:.1f}s]"
                is_violation = False
            else:
                text = f"  {rec.plate_text}: {label}"
                is_violation = False

            lines.append((text, color, is_violation))

        if viol_count > 0:
            lines.append((f"  Violations: {viol_count}", (0, 0, 255), True))

        # 패널 폭 계산
        panel_width = 360
        panel_height = len(lines) * line_height + 12
        panel_x_start = panel_x_end - panel_width

        # 패널 배경
        panel_overlay = frame.copy()
        cv2.rectangle(
            panel_overlay,
            (panel_x_start, panel_y_start - 4),
            (panel_x_end, panel_y_start + panel_height),
            (0, 0, 0), -1,
        )
        frame = cv2.addWeighted(panel_overlay, 0.6, frame, 0.4, 0)

        # 텍스트
        for i, (text, color, is_viol) in enumerate(lines):
            y = panel_y_start + (i + 1) * line_height
            thickness = 2 if is_viol else 1
            cv2.putText(frame, text, (panel_x_start + 4, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, thickness)

        return frame

    # ───────────────────────────────────────────
    # 외부 API
    # ───────────────────────────────────────────

    def get_violations(self) -> dict[str, PlateDistanceRecord]:
        """위반 판정된 번호판 목록

        Returns:
            {"번호판텍스트": PlateDistanceRecord, ...}

        시간복잡도: O(1)
        """
        return self.violations

    def get_distance_info(self, plate_text: str) -> dict | None:
        """특정 번호판의 거리 정보

        Returns:
            {"bbox_ratio", "is_close", "close_duration", "is_violation",
             "distance_label", "distance_color"} 또는 None

        시간복잡도: O(1)
        """
        record = self.distance_records.get(plate_text)
        if record is None:
            return None

        return {
            "bbox_ratio": record.last_bbox_ratio,
            "is_close": record.is_close,
            "close_duration": record.close_duration,
            "is_violation": record.is_violation,
            "distance_label": record.last_distance_label,
            "distance_color": record.last_distance_color,
        }

    def get_violation_report(self) -> list[dict]:
        """전체 위반 리포트 (evidence_export에서 사용)

        Returns:
            [{"plate", "violation_time", "close_duration", "distance_label"}, ...]

        시간복잡도: O(V), V = 위반 수
        """
        report = []
        for plate_text, record in self.violations.items():
            report.append({
                "plate": plate_text,
                "violation_time": record.violation_time,
                "close_duration": record.close_duration,
                "distance_label": record.last_distance_label,
                "bbox_ratio": record.last_bbox_ratio,
            })
        return report
