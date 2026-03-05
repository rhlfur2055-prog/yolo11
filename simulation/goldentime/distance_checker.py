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
    - 발행: "DISTANCE_VIOLATION", "DISTANCE_STATUS_UPDATE"

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
    # 비유: 체온계처럼 순간값이 아니라 평균으로 판단
    SMOOTHING_WINDOW = 10

    def __init__(self, plate_text: str, timestamp: float):
        self.plate_text = plate_text
        self.bbox_ratio_history: list[float] = []
        self.is_close = False
        self.close_start_time: Optional[float] = None
        self.close_duration = 0.0
        self.is_violation = False
        self.violation_time: Optional[float] = None
        self.last_bbox_ratio = 0.0
        self.last_seen_time = timestamp
        self.last_distance_label = ""
        self.last_distance_color = (0, 255, 0)

    def update_bbox(self, bbox_ratio: float, timestamp: float,
                    close_threshold: float, gap_tolerance: float) -> None:
        """bbox 비율 업데이트 — 거리 상태 갱신

        Args:
            bbox_ratio: bbox 면적 / 프레임 면적 비율
            timestamp: 현재 영상 시각 (초)
            close_threshold: "가까움" 판정 임계값
            gap_tolerance: 미감지 허용 간격 (초)

        로직:
            1. 이동평균 계산 (노이즈 제거)
            2. 평균값 >= 임계값 → "가까움"
            3. "가까움" 지속 시간 추적

        시간복잡도: O(W), W = SMOOTHING_WINDOW (상수, 10)
        """
        # 시간 간격 체크 — 너무 오래 안 보였으면 리셋
        gap = timestamp - self.last_seen_time
        if gap > gap_tolerance:
            self.bbox_ratio_history.clear()
            self.is_close = False
            self.close_start_time = None
            self.close_duration = 0.0

        self.last_seen_time = timestamp
        self.last_bbox_ratio = bbox_ratio

        # ── 이동평균 ──
        self.bbox_ratio_history.append(bbox_ratio)
        if len(self.bbox_ratio_history) > self.SMOOTHING_WINDOW:
            self.bbox_ratio_history = self.bbox_ratio_history[-self.SMOOTHING_WINDOW:]

        avg_ratio = sum(self.bbox_ratio_history) / len(self.bbox_ratio_history)

        # ── "가까움" 판정 ──
        was_close = self.is_close
        self.is_close = avg_ratio >= close_threshold

        if self.is_close and not was_close:
            # 가까움 상태 진입
            self.close_start_time = timestamp
            self.close_duration = 0.0
        elif self.is_close and was_close:
            # 가까움 상태 유지 — 지속 시간 갱신
            if self.close_start_time is not None:
                self.close_duration = timestamp - self.close_start_time
        elif not self.is_close and was_close:
            # 가까움 상태 해제
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

            # ── bbox 면적 비율 계산 ──
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            bbox_area = abs((x2 - x1) * (y2 - y1))
            bbox_ratio = bbox_area / self.frame_area if self.frame_area > 0 else 0.0

            # ── 거리 레코드 갱신 ──
            if plate_text not in self.distance_records:
                self.distance_records[plate_text] = PlateDistanceRecord(plate_text, timestamp)

            record = self.distance_records[plate_text]
            record.update_bbox(bbox_ratio, timestamp, close_threshold, gap_tolerance)

            # 표시용 거리 라벨 갱신
            record.last_distance_label, record.last_distance_color = self._get_distance_level(bbox_ratio)

            # ── 위반 체크 ──
            newly_violated = record.check_violation(violation_duration, timestamp)
            if newly_violated:
                self.violations[plate_text] = record
                self._publish_violation(record, bbox_ratio)

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
                }
                for text, rec in self.distance_records.items()
            },
            "violation_count": len(self.violations),
            "timestamp": timestamp,
        })

    # ───────────────────────────────────────────
    # 내부 로직
    # ───────────────────────────────────────────

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
