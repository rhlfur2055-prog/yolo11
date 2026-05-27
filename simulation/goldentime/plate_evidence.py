"""
plate_evidence.py — 15초 지속 번호판 채증 로직 (계단 3)

역할: 사이렌 감지 후, 15초 이상 연속 탐지되는 번호판을 채증 대상으로 등록하고
      해당 시점 전/후 30초 영상을 버퍼에 저장

비유: CCTV 관제사. 사이렌(비상벨)이 울리면 모니터를 주시하다가,
     같은 차가 15초 넘게 계속 잡히면 "이 차 수상하다!" → 녹화 시작.
     녹화는 이미 30초 전부터 돌고 있었음 (순환 버퍼 = 블랙박스 원리)

최민수님 스펙:
    "사이렌 후 15초 이상 지속적으로 탐지되는 번호판에 대해 채증 시작
     (15초 이상 인식된 시점으로부터 전/후 +30초간 영상 저장)"

기존 코드 수정: 없음
Mock 데이터: 없음 — 실제 YOLO 감지 결과 기반

이벤트 버스 연동:
    - 수신: "SIREN_DETECTED", "SIREN_ENDED", "detection_result", "frame_read"
    - 발행: "EVIDENCE_STARTED", "EVIDENCE_COMPLETE", "PLATE_TRACKING_UPDATE"

검증: python test_ocr_accuracy.py  # 12/12 통과 필수
"""

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


# ═══════════════════════════════════════════════
# 설정값 — 나중에 config 파일로 분리 가능
# ═══════════════════════════════════════════════
DEFAULT_CONFIG = {
    "continuous_threshold_sec": 15.0,   # 채증 대상 판정 기준 (초)
    "pre_buffer_sec": 30.0,             # 채증 시점 이전 버퍼 (초)
    "post_buffer_sec": 30.0,            # 채증 시점 이후 녹화 (초)
    "gap_tolerance_sec": 2.0,            # 번호판 미감지 허용 간격 (초)
                                        # 2초 이내 재감지 → 연속으로 간주
                                        # 비유: 가로등 아래 잠깐 안 보여도 같은 차
    "min_confidence": 0.5,              # 최소 신뢰도 (이하는 무시)
    "compute_blur_score": False,         # True면 프레임 버퍼에 Laplacian variance(블러 점수) 저장
                                        # evidence_export에서 evidence_quality/blur_affected_frames 반영용
}


# ═══════════════════════════════════════════════
# PlateTrackRecord — 개별 번호판 추적 기록
# ═══════════════════════════════════════════════
class PlateTrackRecord:
    """개별 번호판의 연속 감지 기록

    비유: 출석부. 매 프레임마다 "이 번호판 있었나?" 체크.
         15초 연속 출석하면 채증 대상.

    Attributes:
        plate_text: 번호판 텍스트 (예: "경기76바7789")
        first_seen_time: 최초 감지 시각 (영상 기준 초)
        last_seen_time: 마지막 감지 시각
        first_seen_frame: 최초 감지 프레임 번호
        last_seen_frame: 마지막 감지 프레임 번호
        continuous_duration: 연속 감지 시간 (초)
        is_evidence_target: 채증 대상 여부
        last_bbox: 마지막 감지 위치 [x1, y1, x2, y2]
        last_confidence: 마지막 감지 신뢰도
        detection_count: 총 감지 횟수
    """

    def __init__(self, plate_text: str, timestamp: float, frame_idx: int, bbox: list, confidence: float):
        self.plate_text = plate_text
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        self.first_seen_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.continuous_duration = 0.0
        self.is_evidence_target = False
        self.evidence_started_at: Optional[float] = None  # 채증 시작 시각
        self.last_bbox = bbox
        self.last_confidence = confidence
        self.detection_count = 1

    def update(self, timestamp: float, frame_idx: int, bbox: list, confidence: float, gap_tolerance: float) -> None:
        """감지 업데이트 — 연속성 판단 포함

        Args:
            timestamp: 현재 영상 시각 (초)
            frame_idx: 현재 프레임 번호
            bbox: 감지 위치 [x1, y1, x2, y2]
            confidence: 감지 신뢰도
            gap_tolerance: 미감지 허용 간격 (초)

        로직:
            - 마지막 감지로부터 gap_tolerance 이내 → 연속 유지
            - gap_tolerance 초과 → 연속 끊김 → first_seen 리셋

        시간복잡도: O(1)
        """
        gap = timestamp - self.last_seen_time

        if gap > gap_tolerance:
            # 연속 끊김 → 리셋
            self.first_seen_time = timestamp
            self.continuous_duration = 0.0
        else:
            # 연속 유지
            self.continuous_duration = timestamp - self.first_seen_time

        self.last_seen_time = timestamp
        self.last_seen_frame = frame_idx
        self.last_bbox = bbox
        self.last_confidence = confidence
        self.detection_count += 1

    def is_stale(self, current_time: float, gap_tolerance: float) -> bool:
        """이 번호판이 더 이상 감지되지 않는지 확인

        Args:
            current_time: 현재 영상 시각 (초)
            gap_tolerance: 미감지 허용 간격 (초)

        Returns:
            True면 이 번호판은 화면에서 사라진 것으로 판단

        시간복잡도: O(1)
        """
        return (current_time - self.last_seen_time) > gap_tolerance


# ═══════════════════════════════════════════════
# PlateEvidence — 채증 관리자
# ═══════════════════════════════════════════════
class PlateEvidence:
    """15초 지속 번호판 채증 관리자

    동작 흐름:
        1. SIREN_DETECTED 수신 → 추적 모드 활성화
        2. 매 프레임 감지 결과 수신 → 번호판별 연속 감지 시간 추적
        3. 15초 이상 연속 → 채증 대상 등록 + EVIDENCE_STARTED 발행
        4. 채증 시작 후 30초 → EVIDENCE_COMPLETE 발행
        5. SIREN_ENDED 수신 → 추적 모드 비활성화 (진행 중인 채증은 완료까지 유지)

    프레임 버퍼:
        - collections.deque로 최근 30초 프레임을 순환 저장
        - 비유: 블랙박스의 순환 녹화. 항상 최근 30초를 들고 있음
        - 채증 시작 시 → pre_buffer(30초) + 이후 post_buffer(30초) = 총 1분

    Args:
        event_bus: SimulationFramework의 이벤트 버스
        config: 설정 dict (DEFAULT_CONFIG 참조)
        fps: 영상 FPS (프레임 버퍼 크기 계산용)
    """

    def __init__(self, event_bus, fps: float = 30.0, config: dict = None):
        self.event_bus = event_bus
        self.config = config or DEFAULT_CONFIG.copy()
        self.fps = fps

        # ── 상태 ──
        self.is_tracking = False              # 사이렌 이후 추적 모드
        self.siren_timestamp: Optional[float] = None  # 사이렌 감지 시각

        # ── 번호판별 추적 기록 ──
        # {"경기76바7789": PlateTrackRecord, ...}
        self.tracked_plates: dict[str, PlateTrackRecord] = {}

        # ── 채증 대상 목록 ──
        # {"경기76바7789": PlateTrackRecord, ...}  (is_evidence_target == True)
        self.evidence_targets: dict[str, PlateTrackRecord] = {}

        # ── 프레임 순환 버퍼 (최근 30초) ──
        # 비유: 블랙박스 순환 녹화. 항상 돌고 있고, 이벤트 시 "여기서부터 저장"
        pre_buffer_frames = int(self.fps * self.config["pre_buffer_sec"])
        self.frame_buffer: deque = deque(maxlen=pre_buffer_frames)

        # ── 채증 영상 버퍼 (채증 시작 후 저장용) ──
        # 채증 대상별 post-buffer 프레임 저장
        # {"경기76바7789": {"frames": [...], "start_time": float, "remaining_frames": int}, ...}
        self.evidence_recordings: dict[str, dict] = {}

        # ── 현재 시각 추적 ──
        self._current_timestamp = 0.0
        self._current_frame_idx = 0

        # ── 이벤트 구독 ──
        self.event_bus.subscribe("SIREN_DETECTED", self._on_siren_detected)
        self.event_bus.subscribe("SIREN_ENDED", self._on_siren_ended)
        self.event_bus.subscribe("frame_read", self._on_frame_read)
        self.event_bus.subscribe("detection_result", self._on_detection_result)

        print(f"[PlateEvidence] 초기화 완료")
        print(f"  연속 감지 기준: {self.config['continuous_threshold_sec']}초")
        print(f"  프레임 버퍼: 전 {self.config['pre_buffer_sec']}초 + 후 {self.config['post_buffer_sec']}초")
        print(f"  미감지 허용: {self.config['gap_tolerance_sec']}초")

    # ───────────────────────────────────────────
    # 이벤트 핸들러
    # ───────────────────────────────────────────

    def _on_siren_detected(self, data: dict) -> None:
        """사이렌 감지 → 추적 모드 시작

        이미 추적 중이면 무시.
        추적 시작 시 기존 추적 기록 초기화.

        시간복잡도: O(1)
        """
        if self.is_tracking:
            print(f"  [PlateEvidence] 이미 추적 모드 (무시)")
            return

        self.is_tracking = True
        self.siren_timestamp = data.get("timestamp", 0.0)
        self.tracked_plates.clear()  # 새로운 사이렌 → 추적 기록 초기화

        print(f"\n  [PlateEvidence] 추적 모드 시작 (사이렌 @ {self.siren_timestamp:.1f}s)")
        print(f"     → 번호판 연속 감지 추적 시작")

    def _on_siren_ended(self, data: dict) -> None:
        """사이렌 종료 → 추적 모드 해제

        진행 중인 채증이 있으면 완료까지 유지.
        추적만 중단하고, evidence_recordings는 계속 기록.

        시간복잡도: O(1)
        """
        self.is_tracking = False
        duration = data.get("duration", 0.0)
        active_count = len(self.evidence_recordings)

        print(f"\n  [PlateEvidence] 추적 모드 종료 (사이렌 지속: {duration:.1f}초)")
        if active_count > 0:
            print(f"     → 진행 중인 채증 {active_count}건은 완료까지 유지")
        else:
            print(f"     → 채증 대상 없음")

    def _on_frame_read(self, data: dict) -> None:
        """프레임 읽기 → 순환 버퍼에 저장 + 시각 갱신

        프레임 버퍼는 사이렌과 무관하게 **항상** 돌아감.
        비유: 블랙박스는 시동 걸면 항상 녹화. 사고가 나야 저장할 뿐.

        시간복잡도: O(1) (deque maxlen 초과 시 자동 pop)
        """
        self._current_timestamp = data.get("timestamp", 0.0)
        self._current_frame_idx = data.get("frame_idx", 0)
        frame = data.get("frame")

        if frame is not None:
            # 순환 버퍼에 프레임 저장 (항상)
            entry = {
                "frame": frame.copy(),  # 복사 필수 — 원본은 다음 프레임에서 덮어씀
                "timestamp": self._current_timestamp,
                "frame_idx": self._current_frame_idx,
            }
            if self.config.get("compute_blur_score"):
                entry["blur_score"] = self._blur_score(frame)
            self.frame_buffer.append(entry)

            # 진행 중인 채증의 post-buffer에도 추가
            self._update_evidence_recordings(frame)

    def _on_detection_result(self, data: dict) -> None:
        """감지 결과 수신 → 번호판별 연속 추적 업데이트

        추적 모드가 아니면 무시.

        각 감지된 번호판에 대해:
            - 새 번호판 → PlateTrackRecord 생성
            - 기존 번호판 → update() → 연속 시간 갱신
            - 15초 도달 → 채증 대상 등록

        시간복잡도: O(d), d = 감지된 객체 수
        """
        if not self.is_tracking:
            return

        detections = data.get("detections", [])
        timestamp = data.get("timestamp", self._current_timestamp)
        frame_idx = data.get("frame_idx", self._current_frame_idx)
        gap_tolerance = self.config["gap_tolerance_sec"]
        min_conf = self.config["min_confidence"]
        threshold = self.config["continuous_threshold_sec"]

        for det in detections:
            plate_text = det.get("plate", "")
            confidence = det.get("confidence", 0.0)
            bbox = det.get("bbox", [])

            # 빈 텍스트 또는 낮은 신뢰도 → 무시
            if not plate_text or confidence < min_conf:
                continue

            if plate_text in self.tracked_plates:
                # ── 기존 번호판 업데이트 ──
                record = self.tracked_plates[plate_text]
                record.update(timestamp, frame_idx, bbox, confidence, gap_tolerance)
            else:
                # ── 새 번호판 등록 ──
                record = PlateTrackRecord(plate_text, timestamp, frame_idx, bbox, confidence)
                self.tracked_plates[plate_text] = record

            # ── 15초 도달 체크 ──
            if record.continuous_duration >= threshold and not record.is_evidence_target:
                self._promote_to_evidence(record)

        # ── stale 번호판 정리 (화면에서 사라진 것) ──
        self._cleanup_stale_plates(timestamp)

        # ── 추적 상태 이벤트 발행 (디버그/UI용) ──
        self.event_bus.publish("PLATE_TRACKING_UPDATE", {
            "tracked_count": len(self.tracked_plates),
            "evidence_count": len(self.evidence_targets),
            "plates": {
                text: {
                    "duration": rec.continuous_duration,
                    "is_evidence": rec.is_evidence_target,
                    "bbox": rec.last_bbox,
                }
                for text, rec in self.tracked_plates.items()
            },
            "timestamp": timestamp,
        })

    # ───────────────────────────────────────────
    # 내부 로직
    # ───────────────────────────────────────────

    def _blur_score(self, frame: np.ndarray) -> float:
        """프레임 블러 점수 (Laplacian variance). 낮을수록 흐림.

        시간복잡도: O(w*h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _promote_to_evidence(self, record: PlateTrackRecord) -> None:
        """번호판을 채증 대상으로 승격

        15초 연속 감지 도달 시 호출.
        비유: 출석부에서 15일 연속 출석 → "개근상" 딱지 붙이기

        동작:
            1. is_evidence_target = True
            2. evidence_targets에 등록
            3. pre-buffer 스냅샷 (현재 순환 버퍼의 내용)
            4. post-buffer 녹화 시작
            5. EVIDENCE_STARTED 이벤트 발행

        시간복잡도: O(B), B = pre-buffer 프레임 수 (복사)
        """
        record.is_evidence_target = True
        record.evidence_started_at = self._current_timestamp
        self.evidence_targets[record.plate_text] = record

        # pre-buffer 스냅샷 — 현재 순환 버퍼의 프레임들을 복사
        pre_frames = list(self.frame_buffer)

        # post-buffer 녹화 시작
        post_buffer_frames = int(self.fps * self.config["post_buffer_sec"])
        self.evidence_recordings[record.plate_text] = {
            "pre_frames": pre_frames,
            "post_frames": [],
            "start_time": self._current_timestamp,
            "start_frame": self._current_frame_idx,
            "remaining_frames": post_buffer_frames,
            "record": record,
        }

        # 이벤트 발행
        self.event_bus.publish("EVIDENCE_STARTED", {
            "plate": record.plate_text,
            "continuous_duration": record.continuous_duration,
            "timestamp": self._current_timestamp,
            "frame_idx": self._current_frame_idx,
            "bbox": record.last_bbox,
            "detection_count": record.detection_count,
        })

        elapsed = record.continuous_duration
        print(f"\n  [PlateEvidence] 채증 대상 등록: {record.plate_text}")
        print(f"     연속 감지: {elapsed:.1f}초 (기준: {self.config['continuous_threshold_sec']}초)")
        print(f"     총 감지 횟수: {record.detection_count}회")
        print(f"     → 전 {self.config['pre_buffer_sec']}초 버퍼 확보 ({len(pre_frames)} 프레임)")
        print(f"     → 후 {self.config['post_buffer_sec']}초 녹화 시작")

    def _update_evidence_recordings(self, frame: np.ndarray) -> None:
        """진행 중인 채증의 post-buffer에 프레임 추가

        remaining_frames가 0이 되면 채증 완료.

        시간복잡도: O(E), E = 진행 중인 채증 수 (보통 1~3)
        """
        completed = []

        for plate_text, recording in self.evidence_recordings.items():
            if recording["remaining_frames"] > 0:
                post_entry = {
                    "frame": frame.copy(),
                    "timestamp": self._current_timestamp,
                    "frame_idx": self._current_frame_idx,
                }
                if self.config.get("compute_blur_score"):
                    post_entry["blur_score"] = self._blur_score(frame)
                recording["post_frames"].append(post_entry)
                recording["remaining_frames"] -= 1
            else:
                completed.append(plate_text)

        # 완료된 채증 처리
        for plate_text in completed:
            self._complete_evidence(plate_text)

    def _complete_evidence(self, plate_text: str) -> None:
        """채증 완료 처리

        pre_frames + post_frames = 총 ~1분 영상 데이터.
        이 데이터는 evidence_export.py (계단 5)에서 파일로 저장.

        시간복잡도: O(1)
        """
        recording = self.evidence_recordings.pop(plate_text, None)
        if recording is None:
            return

        record = recording["record"]
        pre_count = len(recording["pre_frames"])
        post_count = len(recording["post_frames"])
        total_count = pre_count + post_count
        total_sec = total_count / self.fps if self.fps > 0 else 0

        # 이벤트 발행 — evidence_export.py가 이걸 받아서 파일 저장
        self.event_bus.publish("EVIDENCE_COMPLETE", {
            "plate": plate_text,
            "pre_frames": recording["pre_frames"],
            "post_frames": recording["post_frames"],
            "start_time": recording["start_time"],
            "start_frame": recording["start_frame"],
            "end_time": self._current_timestamp,
            "end_frame": self._current_frame_idx,
            "total_frames": total_count,
            "total_duration_sec": total_sec,
            "record": {
                "first_seen_time": record.first_seen_time,
                "continuous_duration": record.continuous_duration,
                "detection_count": record.detection_count,
                "last_bbox": record.last_bbox,
                "last_confidence": record.last_confidence,
            },
        })

        print(f"\n  [PlateEvidence] 채증 완료: {plate_text}")
        print(f"     총 {total_count} 프레임 ({total_sec:.1f}초)")
        print(f"     (전 {pre_count}프레임 + 후 {post_count}프레임)")
        print(f"     → EVIDENCE_COMPLETE 이벤트 발행 (계단 5에서 파일 저장)")

    def _cleanup_stale_plates(self, current_time: float) -> None:
        """화면에서 사라진 번호판 정리

        gap_tolerance를 넘도록 감지되지 않은 번호판은 제거.
        단, 채증 대상은 제거하지 않음 (증거 보존).

        시간복잡도: O(P), P = 추적 중인 번호판 수
        """
        gap_tolerance = self.config["gap_tolerance_sec"]
        stale_keys = [
            text for text, rec in self.tracked_plates.items()
            if rec.is_stale(current_time, gap_tolerance) and not rec.is_evidence_target
        ]
        for key in stale_keys:
            del self.tracked_plates[key]

    # ───────────────────────────────────────────
    # 오버레이
    # ───────────────────────────────────────────

    def draw_evidence_overlay(self, frame: np.ndarray) -> np.ndarray:
        """채증 상태 오버레이 — 추적 중인 번호판 정보 표시

        표시 내용:
            - 추적 모드 아니면 → 아무것도 안 그림
            - 추적 중인 번호판 bbox → 노란색 테두리 + 연속 시간
            - 채증 대상 bbox → 빨간색 두꺼운 테두리 + "채증중" 라벨
            - 좌상단 추적 정보 패널

        Args:
            frame: 대상 프레임 (BGR)

        Returns:
            오버레이가 추가된 프레임

        시간복잡도: O(P), P = 추적 중인 번호판 수
        """
        if not self.is_tracking and not self.evidence_recordings:
            return frame

        overlay = frame.copy()
        h, w = frame.shape[:2]

        # ── 추적 중인 번호판에 bbox 오버레이 ──
        for plate_text, record in self.tracked_plates.items():
            bbox = record.last_bbox
            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            duration = record.continuous_duration
            threshold = self.config["continuous_threshold_sec"]

            if record.is_evidence_target:
                # ── 채증 대상 — 빨간색 두꺼운 bbox ──
                color = (0, 0, 255)       # 빨간색
                thickness = 3

                # 채증 진행 상태
                recording = self.evidence_recordings.get(plate_text)
                if recording:
                    remaining = recording["remaining_frames"]
                    remaining_sec = remaining / self.fps if self.fps > 0 else 0
                    label = f"[EVIDENCE] {plate_text} | {remaining_sec:.0f}s left"
                else:
                    label = f"[EVIDENCE DONE] {plate_text}"

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

                # 라벨 배경
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_y = max(y1 - 10, label_size[1] + 4)
                cv2.rectangle(overlay, (x1, label_y - label_size[1] - 4), (x1 + label_size[0] + 4, label_y + 4), (0, 0, 180), -1)
                cv2.putText(overlay, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            else:
                # ── 추적 중 — 노란색 bbox + 경과 시간 ──
                # 15초에 가까울수록 노랑→주황 그라데이션
                progress = min(duration / threshold, 1.0)
                green = int(255 * (1.0 - progress * 0.5))  # 255→128
                color = (0, green, 255)  # 노랑(0,255,255) → 주황(0,128,255)
                thickness = 2

                label = f"{plate_text} [{duration:.1f}s/{threshold:.0f}s]"

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

                # 프로그레스 바 (bbox 하단)
                bar_width = x2 - x1
                filled_width = int(bar_width * progress)
                bar_y = y2 + 2
                cv2.rectangle(overlay, (x1, bar_y), (x1 + bar_width, bar_y + 6), (80, 80, 80), -1)
                cv2.rectangle(overlay, (x1, bar_y), (x1 + filled_width, bar_y + 6), color, -1)

                # 라벨
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                label_y = max(y1 - 8, label_size[1] + 4)
                cv2.rectangle(overlay, (x1, label_y - label_size[1] - 4), (x1 + label_size[0] + 4, label_y + 4), (60, 60, 60), -1)
                cv2.putText(overlay, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # ── 반투명 블렌딩 ──
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        # ── 좌상단 추적 정보 패널 ──
        if self.is_tracking or self.evidence_recordings:
            frame = self._draw_tracking_panel(frame)

        return frame

    def _draw_tracking_panel(self, frame: np.ndarray) -> np.ndarray:
        """좌상단 추적 정보 패널

        표시 내용:
            - 추적 모드 상태
            - 추적 중인 번호판 수
            - 채증 대상 수
            - 각 번호판의 연속 감지 시간

        시간복잡도: O(P), P = 추적 중인 번호판 수
        """
        h, w = frame.shape[:2]

        # 사이렌 오버레이(계단2)가 상단 48px 사용하므로 그 아래에 배치
        panel_y_start = 50
        panel_x = 8
        line_height = 18

        lines = []

        # 헤더
        tracked = len(self.tracked_plates)
        evidence = len(self.evidence_targets)
        active_rec = len(self.evidence_recordings)
        lines.append(f"Tracking: {tracked} plates | Evidence: {evidence} | Recording: {active_rec}")

        # 각 번호판 상태 (상위 5개만, 연속시간 기준 정렬)
        sorted_plates = sorted(
            self.tracked_plates.values(),
            key=lambda r: r.continuous_duration,
            reverse=True,
        )[:5]

        for rec in sorted_plates:
            status = "** EVIDENCE **" if rec.is_evidence_target else ""
            lines.append(f"  {rec.plate_text}: {rec.continuous_duration:.1f}s (x{rec.detection_count}) {status}")

        # 패널 배경
        panel_height = len(lines) * line_height + 12
        panel_width = 420

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x - 4, panel_y_start - 4),
            (panel_x + panel_width, panel_y_start + panel_height),
            (0, 0, 0), -1,
        )
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # 텍스트
        for i, line in enumerate(lines):
            y = panel_y_start + (i + 1) * line_height
            color = (100, 100, 255) if "EVIDENCE" in line else (200, 200, 200)
            cv2.putText(frame, line, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        return frame

    # ───────────────────────────────────────────
    # 외부 API
    # ───────────────────────────────────────────

    def get_evidence_targets(self) -> dict[str, PlateTrackRecord]:
        """현재 채증 대상 목록 반환

        Returns:
            {"번호판텍스트": PlateTrackRecord, ...}

        시간복잡도: O(1) (dict 참조 반환)
        """
        return self.evidence_targets

    def get_all_evidence_frames(self, plate_text: str) -> list[dict] | None:
        """특정 번호판의 채증 영상 프레임 전체 반환 (pre + post)

        녹화 진행 중이면 현재까지의 프레임 반환.
        녹화 완료/미존재면 None.

        Args:
            plate_text: 번호판 텍스트

        Returns:
            [{"frame": np.ndarray, "timestamp": float, "frame_idx": int}, ...] 또는 None

        시간복잡도: O(1) (리스트 참조 반환)
        """
        recording = self.evidence_recordings.get(plate_text)
        if recording is None:
            return None
        return recording["pre_frames"] + recording["post_frames"]

    def get_recording_progress(self, plate_text: str) -> dict | None:
        """특정 번호판의 채증 진행 상태

        Returns:
            {"remaining_frames": int, "remaining_sec": float, "progress": float} 또는 None

        시간복잡도: O(1)
        """
        recording = self.evidence_recordings.get(plate_text)
        if recording is None:
            return None

        total = int(self.fps * self.config["post_buffer_sec"])
        remaining = recording["remaining_frames"]
        elapsed = total - remaining

        return {
            "remaining_frames": remaining,
            "remaining_sec": remaining / self.fps if self.fps > 0 else 0,
            "elapsed_frames": elapsed,
            "elapsed_sec": elapsed / self.fps if self.fps > 0 else 0,
            "progress": elapsed / total if total > 0 else 1.0,
        }
