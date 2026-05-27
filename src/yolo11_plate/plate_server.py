"""plate_server.py — 번호판 인식 HTTP API 서버 (FastAPI + uvicorn).

실행::

    python plate_server.py                 # 기본 포트 (ServerConfig.API_PORT)
    python plate_server.py --port 5000     # 포트 지정

엔드포인트::

    GET  /health           — 헬스체크
    GET  /api/status       — 엔진 상태/통계
    POST /api/detect       — 이미지 → 번호판 인식
    GET  /docs             — Swagger UI (자동 생성)
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, AsyncIterator, Final, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ServerConfig

# Windows 콘솔 한글 깨짐 방지 (uvicorn 진입 전에 적용)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────
# 상수
# ────────────────────────────────────────────────────────────────
API_TITLE: Final[str] = "YOLO11 번호판 인식 API"
API_DESCRIPTION: Final[str] = (
    "한국 차량 번호판 자동 인식 REST API (YOLO11x + PaddleOCR + CRNN 교차검증)"
)
API_VERSION: Final[str] = "3.1.0"
IMAGE_CONTENT_TYPE_PREFIX: Final[str] = "image/"
EMPTY_BBOX: Final[list[int]] = [0, 0, 0, 0]


# ────────────────────────────────────────────────────────────────
# 엔진 상태 (lifespan 에서 채워지고 Depends 로 주입)
# ────────────────────────────────────────────────────────────────
@dataclass
class EngineState:
    """프로세스 단일 인스턴스. lifespan 에서 채운다."""

    engine: Any = None
    ready: bool = False
    error: Optional[str] = None
    startup_sec: float = 0.0


_state: EngineState = EngineState()


def get_engine_state() -> EngineState:
    """FastAPI Depends 용 — 엔진 상태 단일 인스턴스 반환."""
    return _state


def require_ready_engine(
    state: EngineState = Depends(get_engine_state),
) -> EngineState:
    """엔진이 준비된 경우에만 의존성 통과. 그렇지 않으면 503."""
    if not state.ready or state.engine is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=f"엔진 초기화 중입니다. 잠시 후 다시 시도하세요. error={state.error}",
        )
    return state


# ────────────────────────────────────────────────────────────────
# Pydantic 스키마
# ────────────────────────────────────────────────────────────────
class PlateResult(BaseModel):
    """인식된 번호판 1건."""

    plate: str = Field(..., description="인식 텍스트 (예: 01나8060)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 0~1")
    bbox: list[int] = Field(..., description="번호판 좌표 [x1, y1, x2, y2]")
    vehicle_bbox: list[int] = Field(..., description="차량 좌표 [x1, y1, x2, y2]")
    is_alert: bool = Field(False, description="수배 차량 여부")


class DetectResponse(BaseModel):
    success: bool
    plates: list[PlateResult]
    process_ms: float = Field(..., description="인식 처리 시간 (ms)")
    frame_size: list[int] = Field(..., description="입력 프레임 [width, height]")


class StatusResponse(BaseModel):
    ready: bool
    error: Optional[str] = None
    startup_sec: float
    stats: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    ready: bool


# ────────────────────────────────────────────────────────────────
# 라이프스팬: 엔진 초기화/정리
# ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """서버 시작 시 PlateEnginePro 초기화, 종료 시 close()."""
    started_at = time.time()
    print("[서버] PlateEnginePro 초기화 중...", flush=True)
    try:
        from .plate_engine_pro import PlateEnginePro

        _state.engine = PlateEnginePro()
        _state.ready = True
        _state.startup_sec = time.time() - started_at
        print(f"[서버] 엔진 초기화 완료 ({_state.startup_sec:.1f}초)", flush=True)
    except Exception as exc:  # noqa: BLE001
        _state.error = str(exc)
        print(f"[서버] ⚠️ 엔진 초기화 실패: {exc}", flush=True)

    yield

    if _state.engine is not None:
        try:
            _state.engine.close()
        except Exception:
            pass
    print("[서버] 종료됨", flush=True)


# ────────────────────────────────────────────────────────────────
# FastAPI 앱
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────
# 엔드포인트
# ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(state: EngineState = Depends(get_engine_state)) -> HealthResponse:
    """헬스체크 — 엔진 준비 여부 반환."""
    return HealthResponse(
        status="ok" if state.ready else "starting",
        ready=state.ready,
    )


@app.get("/api/status", response_model=StatusResponse, tags=["meta"])
def status(state: EngineState = Depends(get_engine_state)) -> StatusResponse:
    """엔진 상태 및 통계 (인식 누적, 평균 신뢰도, 시작 소요시간 등)."""
    stats: dict[str, Any] = {}
    if state.ready and state.engine is not None:
        try:
            stats = dict(state.engine.stats)
            confs = stats.pop("confidences", [])
            stats["avg_confidence"] = (
                round(float(np.mean(confs)), 3) if confs else 0.0
            )
            stats["total_detections"] = len(confs)
        except Exception:
            pass
    return StatusResponse(
        ready=state.ready,
        error=state.error,
        startup_sec=round(state.startup_sec, 2),
        stats=stats,
    )


def _decode_image(raw: bytes) -> np.ndarray:
    """업로드 바이트 → OpenCV BGR ndarray. 디코드 실패 시 HTTPException."""
    if not raw:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="빈 파일입니다",
        )
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="이미지 디코딩 실패 (손상된 파일)",
        )
    return frame


def _to_plate_result(raw: dict) -> PlateResult:
    """엔진 raw dict → 응답 스키마. 누락 필드는 안전한 기본값."""
    return PlateResult(
        plate=raw.get("plate", ""),
        confidence=round(float(raw.get("confidence", 0.0)), 3),
        bbox=[int(v) for v in raw.get("bbox", EMPTY_BBOX)],
        vehicle_bbox=[int(v) for v in raw.get("vehicle_bbox", EMPTY_BBOX)],
        is_alert=bool(raw.get("is_alert", False)),
    )


@app.post("/api/detect", response_model=DetectResponse, tags=["detect"])
async def detect(
    file: UploadFile = File(..., description="이미지 파일 (jpg/png/bmp)"),
    state: EngineState = Depends(require_ready_engine),
) -> DetectResponse:
    """이미지에서 번호판 인식.

    Parameters
    ----------
    file : UploadFile
        JPEG / PNG / BMP 이미지.

    Returns
    -------
    DetectResponse
        감지된 번호판 목록 + 처리 시간 + 프레임 크기.
    """
    if file.content_type and not file.content_type.startswith(IMAGE_CONTENT_TYPE_PREFIX):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="이미지 파일만 지원됩니다 (jpg/png/bmp)",
        )

    raw_bytes = await file.read()
    frame = _decode_image(raw_bytes)
    h, w = frame.shape[:2]

    started_at = time.time()
    try:
        raw_results = state.engine.process_frame(frame)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"인식 오류: {exc}",
        ) from exc
    process_ms = (time.time() - started_at) * 1000

    plates = [_to_plate_result(r) for r in (raw_results or [])]
    return DetectResponse(
        success=True,
        plates=plates,
        process_ms=round(process_ms, 1),
        frame_size=[w, h],
    )


# ────────────────────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────────────────────
def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=API_TITLE)
    parser.add_argument("--host", default=ServerConfig.HOST,
                        help=f"바인드 주소 (기본: {ServerConfig.HOST})")
    parser.add_argument("--port", type=int, default=ServerConfig.API_PORT,
                        help=f"포트 (기본: {ServerConfig.API_PORT})")
    parser.add_argument("--workers", type=int, default=1,
                        help="워커 수 (GPU 사용 시 1 권장)")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug", "trace"])
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    print(f"[서버] 시작: http://{args.host}:{args.port}")
    print(f"[서버] API 문서: http://localhost:{args.port}/docs")
    uvicorn.run(
        "yolo11_plate.plate_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
