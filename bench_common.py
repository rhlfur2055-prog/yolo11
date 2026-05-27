# -*- coding: utf-8 -*-
"""bench_common — bench_fps / bench_full / bench_profile 공통 헬퍼.

3개 벤치마크 스크립트의 중복 로직(stdout UTF-8 처리, argparse 패턴,
타이머, OCR 계측 패치, 결과 출력 포맷)을 한 곳에 모은다.
"""
from __future__ import annotations

import argparse
import builtins
import io
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Tuple

# ── Constants ─────────────────────────────────────────────────────────────
DEFAULT_VIDEO: str = r"C:/Users/jomg2/OneDrive/바탕 화면/movie/hiway.mp4"
DEFAULT_MAX_FRAMES: int = 300
DEFAULT_CAMERA_ID: str = "CAM01"
PROGRESS_INTERVAL: int = 50
NOISY_PREFIXES: Tuple[str, ...] = ("[OCR-", "[2STAGE", "[PLATE-", "[FALLBACK")
RESULT_BAR: str = "=" * 60

# OCR fast-path 임계값 — plate_engine_pro._run_ocr 와 동일
OCR_FAST_PATH_MIN_SCORE: float = 0.7
OCR_EARLY_RETURN_MIN_SCORE: float = 0.3
OCR_EARLY_RETURN_MIN_LEN: int = 3
HANGUL_PATTERN: re.Pattern[str] = re.compile(r"[가-힣]")

# YOLO imgsz 분류 기준 — vehicle(416) vs plate(640)
YOLO_VEHICLE_IMGSZ: int = 416


# ── Boot ──────────────────────────────────────────────────────────────────
def configure_utf8_stdout() -> None:
    """Windows 콘솔 UTF-8 강제. 한글 출력 깨짐 방지."""
    if sys.platform != "win32":
        return
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
    except Exception:
        # 이미 래핑됐거나 buffer가 없는 환경 — 무시하고 진행
        pass


def fix_cwd_and_path() -> None:
    """스크립트 위치로 cwd 고정 + sys.path 보강."""
    here = os.path.dirname(os.path.abspath(sys.argv[0] or __file__))
    if here:
        os.chdir(here)
    sys.path.insert(0, ".")


# ── Argparse ──────────────────────────────────────────────────────────────
def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """3개 벤치마크에서 공통으로 쓰는 argparse 빌더."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help="입력 영상 경로",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="처리할 최대 프레임 수",
    )
    parser.add_argument(
        "--camera-id",
        default=DEFAULT_CAMERA_ID,
        help="엔진에 전달할 카메라 ID",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 포맷으로 결과 출력",
    )
    return parser


# ── Quiet print ───────────────────────────────────────────────────────────
@contextmanager
def silence_engine_logs() -> Iterator[Callable[..., None]]:
    """plate_engine_pro 의 [OCR-/[2STAGE/[PLATE-/[FALLBACK 로그 억제.

    yield 되는 callable 은 silenced 상태에서도 강제 출력 가능한 원본 print.
    """
    original = builtins.print

    def quiet(*args: object, **kwargs: object) -> None:
        head = str(args[0]) if args else ""
        if head.startswith(NOISY_PREFIXES):
            return
        original(*args, **kwargs)

    builtins.print = quiet
    try:
        yield original
    finally:
        builtins.print = original


# ── Stats container ───────────────────────────────────────────────────────
@dataclass
class OcrStats:
    """_run_ocr 호출을 분기별로 집계."""
    ocr_calls: int = 0
    fast_path: int = 0
    fallback: int = 0
    early_return: int = 0
    fallback_ms: List[float] = field(default_factory=list)


@dataclass
class YoloStats:
    """YOLO __call__ 소요시간을 vehicle/plate 모델별로 집계."""
    vehicle_ms: List[float] = field(default_factory=list)
    plate_ms: List[float] = field(default_factory=list)


# ── Patches ───────────────────────────────────────────────────────────────
def patch_run_ocr(stats: OcrStats) -> None:
    """plate_engine_pro._run_ocr 를 계측 버전으로 교체.

    원본 로직(rec-only fast path → early return → fallback predict)을
    그대로 따라가며 카운터/지연시간만 채운다.
    """
    import plate_engine_pro as pep

    def instrumented(
        self: object,
        engine_name: str,
        engine: object,
        image: object,
    ) -> Tuple[str, float]:
        stats.ocr_calls += 1
        if engine_name != "paddleocr":
            return "", 0.0

        rec_text, rec_score = _try_fast_path(engine, image, stats)
        if rec_text and rec_score >= OCR_FAST_PATH_MIN_SCORE \
                and HANGUL_PATTERN.search(rec_text):
            return rec_text, rec_score

        if (not rec_text
                or rec_score < OCR_EARLY_RETURN_MIN_SCORE
                or len(rec_text.strip()) < OCR_EARLY_RETURN_MIN_LEN):
            stats.early_return += 1
            if rec_text and rec_score > 0:
                return rec_text, rec_score
            return "", 0.0

        return _run_fallback(engine, image, stats, rec_text, rec_score)

    pep.PlateEnginePro._run_ocr = instrumented  # type: ignore[assignment]


def _try_fast_path(
    engine: object, image: object, stats: OcrStats
) -> Tuple[str, float]:
    """rec-only 경로 시도. 실패 시 ("", 0.0)."""
    try:
        rec_model = engine.paddlex_pipeline.text_rec_model  # type: ignore[attr-defined]
        rec_results = list(rec_model.predict([image]))
        if not rec_results:
            return "", 0.0
        res = rec_results[0]
        rec_text = res.get("rec_text", "")
        rec_score = float(res.get("rec_score", 0))
        if rec_text and rec_score >= OCR_FAST_PATH_MIN_SCORE \
                and HANGUL_PATTERN.search(rec_text):
            stats.fast_path += 1
        return rec_text, rec_score
    except Exception:
        return "", 0.0


def _run_fallback(
    engine: object,
    image: object,
    stats: OcrStats,
    rec_text: str,
    rec_score: float,
) -> Tuple[str, float]:
    """engine.predict() 폴백 경로 + 지연시간 기록."""
    stats.fallback += 1
    started = time.perf_counter()
    try:
        for res in engine.predict(image):  # type: ignore[attr-defined]
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            if texts:
                text = "".join(texts)
                conf = sum(scores) / len(scores) if scores else 0.0
                stats.fallback_ms.append((time.perf_counter() - started) * 1000)
                if text:
                    return text, conf
    except Exception:
        pass
    stats.fallback_ms.append((time.perf_counter() - started) * 1000)
    if rec_text:
        return rec_text, rec_score
    return "", 0.0


def patch_yolo_call(stats: YoloStats) -> None:
    """ultralytics.YOLO.__call__ 을 계측 버전으로 교체."""
    from ultralytics import YOLO

    original = YOLO.__call__

    def timed(self: object, *args: object, **kwargs: object) -> object:
        started = time.perf_counter()
        result = original(self, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000
        imgsz = kwargs.get("imgsz", 640)
        if imgsz == YOLO_VEHICLE_IMGSZ:
            stats.vehicle_ms.append(elapsed_ms)
        else:
            stats.plate_ms.append(elapsed_ms)
        return result

    YOLO.__call__ = timed  # type: ignore[assignment]


# ── Video iteration ───────────────────────────────────────────────────────
@contextmanager
def open_video(path: str) -> Iterator[object]:
    """cv2.VideoCapture context manager — release 보장."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        yield cap
    finally:
        cap.release()


# ── Output helpers ────────────────────────────────────────────────────────
def format_summary_line(frames: int, elapsed: float) -> str:
    """공통 결과 헤더 라인."""
    fps = frames / elapsed if elapsed > 0 else 0.0
    return f"{frames} frames / {elapsed:.1f}s / {fps:.1f} FPS"


def percent_of_wall(value_ms: float, wall_seconds: float) -> float:
    """value(ms) 가 wall(s)의 몇 %인지."""
    if wall_seconds <= 0:
        return 0.0
    return value_ms / (wall_seconds * 1000) * 100
