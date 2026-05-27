"""detection_worker.py — GUI에서 분리된 OCR 워커 스레드.

plate_gui.py의 ``DetectionWorker``를 책임 분리(GUI ↔ 추론) 차원에서 추출.

사용:
    from detection_worker import DetectionWorker

    worker = DetectionWorker(detection_queue, results_queue, recognizer_kwargs)
    worker.start()
    ...
    worker.stop()

설계 메모:
    * Pro/Fast/Unified(``plate_engine_pro``)와 Recognizer(``plate_recognition_4k``) 두 엔진
      경로를 가지며, ML 의존(YOLO/PaddleOCR/CRNN)을 ``run()`` 안에서 지연 import해
      GUI 부팅 비용을 0으로 유지한다.
    * 입력 큐는 ``(frame_idx, frame, timestamp)``, 출력 큐는 결과 ``dict``.
    * 백프레셔 정책: 입력은 *최신 프레임만 처리* (drain), 출력은 *drop-oldest*.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Any, Final, Optional

import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUEUE_POLL_TIMEOUT_SEC: Final[float] = 0.5  # detection_queue.get() 대기
DEFAULT_CAMERA_ID: Final[str] = "CAM01"     # 엔진 호출 시 카메라 식별자


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 결과 포맷 변환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pro_results_to_gui_format(
    pro_results: list[dict[str, Any]],
    frame: np.ndarray,
) -> list[dict[str, Any]]:
    """Pro/Fast/Unified 엔진 결과를 GUI가 기대하는 dict 형식으로 변환.

    bbox 영역의 plate crop을 함께 첨부한다 (GUI 우측 패널 카드에 사용).
    """
    h, w = frame.shape[:2] if frame.size > 0 else (0, 0)
    gui_results: list[dict[str, Any]] = []

    for r in pro_results:
        bbox = r.get("bbox", [0, 0, 0, 0])
        plate_img: Optional[np.ndarray] = None

        if frame.size > 0 and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                plate_img = frame[y1:y2, x1:x2].copy()

        plate_text = r.get("plate", "")
        gui_results.append({
            "text": plate_text,
            "ocr_confidence": r.get("confidence", 0),
            "bbox": r.get("bbox", []),
            "is_valid_plate": bool(plate_text),
            "plate_image": plate_img,
            "pattern_score": r.get("confidence", 0),
            # ── 강화 필드 (PlateEnginePro → GUI 전달) ──
            "confidence_level": r.get("confidence_level", ""),
            "plate_type": r.get("plate_type", ""),
            "vehicle_type": r.get("vehicle_type", ""),
            "frame_count": r.get("frame_count", 1),
        })
    return gui_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DetectionWorker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionWorker(threading.Thread):
    """PlateRecognizer 또는 PlateEnginePro로 번호판 인식.

    ★ 2단계 파이프라인 중 Phase 2(OCR) 담당.
      Phase 1(YOLO bbox)은 별도 Fast YOLO 스레드에서 표시 프레임 기반으로 실행.

    Attributes:
        detection_queue: 입력 큐 ``(frame_idx, frame, timestamp)``.
        results_queue: 출력 큐 (결과 dict).
        ready: 모델 로드 완료 신호 (메인 스레드가 wait 가능).
        stop_flag: 종료 신호.
        loading_msg: 로딩 진행/오류 표시용 메시지.
    """

    def __init__(
        self,
        detection_queue: queue.Queue,
        results_queue: queue.Queue,
        recognizer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(daemon=True)
        self.detection_queue: queue.Queue = detection_queue
        self.results_queue: queue.Queue = results_queue
        self.stop_flag: threading.Event = threading.Event()
        self.ready: threading.Event = threading.Event()
        self.loading_msg: str = ""

        # 엔진 선택 플래그는 kwargs에서 꺼내 recognizer 생성 인자에서 제외
        self.use_pro_engine: bool = recognizer_kwargs.pop("use_pro_engine", False)
        self.engine_mode: str = recognizer_kwargs.pop("engine_mode", "pro")
        self.use_multiframe: bool = recognizer_kwargs.pop("use_multiframe", False)
        self.recognizer_kwargs: dict[str, Any] = recognizer_kwargs

        # 엔진 핸들 (run에서 초기화)
        self.recognizer: Optional[Any] = None
        self.pro_engine: Optional[Any] = None
        self.fast_engine: Optional[Any] = None
        self._process_frame_unified: Optional[Any] = None

    # -- 스레드 진입점 ----------------------------------------------------

    def run(self) -> None:
        if not self._initialize_engines():
            return  # loading_msg에 오류가 기록되어 종료
        self.loading_msg = ""
        self.ready.set()
        self._process_loop()

    def stop(self) -> None:
        self.stop_flag.set()

    # -- 모델 로딩 (Guard Clause로 분기 평탄화) ---------------------------

    def _initialize_engines(self) -> bool:
        if self.use_pro_engine:
            return self._load_pro_engines()
        return self._load_recognizer()

    def _load_pro_engines(self) -> bool:
        self.loading_msg = "Loading Pro/Fast engines..."
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from plate_engine_pro import (
                PlateEnginePro,
                PlateEngineFast,
                process_frame_unified,
            )
            self.pro_engine = PlateEnginePro()
            self.fast_engine = PlateEngineFast()
            self._process_frame_unified = process_frame_unified
        except Exception as e:
            self.loading_msg = f"Pro engine error: {e}"
            return False
        return True

    def _load_recognizer(self) -> bool:
        self.loading_msg = "Loading plate model..."
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from plate_recognition_4k import PlateRecognizer
            self.recognizer = PlateRecognizer(**self.recognizer_kwargs)
        except Exception as e:
            self.loading_msg = f"Model load error: {e}"
            return False
        return True

    # -- 처리 루프 --------------------------------------------------------

    def _process_loop(self) -> None:
        while not self.stop_flag.is_set():
            latest = self._drain_to_latest_frame()
            if latest is None:
                continue
            frame_idx, frame, ts = latest

            data = self._process_frame(frame_idx, frame, ts)
            self._publish_drop_oldest(data)

    def _drain_to_latest_frame(
        self,
    ) -> Optional[tuple[int, np.ndarray, float]]:
        """입력 큐에서 가장 최신 프레임만 꺼낸다 (오래된 프레임은 버림)."""
        try:
            item = self.detection_queue.get(timeout=QUEUE_POLL_TIMEOUT_SEC)
        except queue.Empty:
            return None

        # 큐에 쌓인 오래된 프레임은 모두 버리고 최신만 처리 → 실시간 응답성 확보
        while not self.detection_queue.empty():
            try:
                item = self.detection_queue.get_nowait()
            except queue.Empty:
                break
        return item

    def _publish_drop_oldest(self, data: dict[str, Any]) -> None:
        """결과 큐가 가득 차면 가장 오래된 결과를 버리고 새 결과를 푸시."""
        try:
            self.results_queue.put_nowait(data)
            return
        except queue.Full:
            print("[WORKER] ⚠️ results_queue FULL! 결과 드롭 가능", flush=True)

        try:
            self.results_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.results_queue.put_nowait(data)
        except queue.Full:
            pass  # 그래도 못 넣으면 포기 (다음 프레임에서 재시도)

    # -- 프레임 처리 — Guard Clause로 4갈래 평탄화 ----------------------

    def _process_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        ts: float,
    ) -> dict[str, Any]:
        if self.pro_engine is None:
            return self._process_with_recognizer(frame_idx, frame, ts)
        if self._is_realtime_pro_mode():
            return self._process_with_pro_realtime(frame_idx, frame, ts)
        if self._is_unified_mode():
            return self._process_with_unified(frame_idx, frame, ts)
        return self._process_with_pro(frame_idx, frame, ts)

    def _is_realtime_pro_mode(self) -> bool:
        return (
            getattr(self.pro_engine, "consecutive_required", 1) > 1
            and hasattr(self.pro_engine, "process_frame_realtime")
        )

    def _is_unified_mode(self) -> bool:
        return (
            self.engine_mode in ("fast", "auto")
            and getattr(self.fast_engine, "available", False)
        )

    def _process_with_pro_realtime(
        self,
        frame_idx: int,
        frame: np.ndarray,
        ts: float,
    ) -> dict[str, Any]:
        t0 = time.time()
        pro_results = self.pro_engine.process_frame_realtime(frame, DEFAULT_CAMERA_ID)
        elapsed_ms = (time.time() - t0) * 1000
        return {
            "frame_idx": frame_idx,
            "timestamp": ts,
            "results": _pro_results_to_gui_format(pro_results, frame),
            "process_ms": elapsed_ms,
            "process_ms_pro": elapsed_ms,
            "process_ms_fast": 0.0,
        }

    def _process_with_unified(
        self,
        frame_idx: int,
        frame: np.ndarray,
        ts: float,
    ) -> dict[str, Any]:
        results_list, ms_pro, ms_fast = self._process_frame_unified(
            frame, DEFAULT_CAMERA_ID,
            engine_pro=self.pro_engine,
            engine_fast=self.fast_engine,
            engine_mode=self.engine_mode,
            use_multiframe=self.use_multiframe,
        )
        return {
            "frame_idx": frame_idx,
            "timestamp": ts,
            "results": _pro_results_to_gui_format(results_list, frame),
            "process_ms": max(ms_pro, ms_fast),
            "process_ms_pro": ms_pro,
            "process_ms_fast": ms_fast,
        }

    def _process_with_pro(
        self,
        frame_idx: int,
        frame: np.ndarray,
        ts: float,
    ) -> dict[str, Any]:
        t0 = time.time()
        pro_results = self.pro_engine.process_frame(
            frame, DEFAULT_CAMERA_ID, use_multiframe=self.use_multiframe,
        )
        if pro_results:
            print(f"[WORKER] frame={frame_idx} engine_results={len(pro_results)}", flush=True)
            for pr in pro_results:
                print(
                    f"  engine → plate='{pr.get('plate','')}' "
                    f"conf={pr.get('confidence',0):.2f} "
                    f"bbox={pr.get('bbox',[])}",
                    flush=True,
                )
        elapsed_ms = (time.time() - t0) * 1000
        return {
            "frame_idx": frame_idx,
            "timestamp": ts,
            "results": _pro_results_to_gui_format(pro_results, frame),
            "process_ms": elapsed_ms,
            "process_ms_pro": elapsed_ms,
            "process_ms_fast": 0.0,
        }

    def _process_with_recognizer(
        self,
        frame_idx: int,
        frame: np.ndarray,
        ts: float,
    ) -> dict[str, Any]:
        t0 = time.time()
        results = self.recognizer.process_frame(frame, frame_idx)
        elapsed_ms = (time.time() - t0) * 1000
        return {
            "frame_idx": frame_idx,
            "timestamp": ts,
            "results": results,
            "process_ms": elapsed_ms,
        }
