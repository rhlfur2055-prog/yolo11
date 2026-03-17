"""
plate_server.py — 번호판 인식 HTTP API 서버 (FastAPI + uvicorn)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

실행:
    python plate_server.py              # 기본 포트 8765
    python plate_server.py --port 8765  # 포트 지정

엔드포인트:
    GET  /health           — 헬스체크
    GET  /api/status       — 엔진 상태
    POST /api/detect       — 이미지 → 번호판 인식
    GET  /docs             — 자동 API 문서 (Swagger UI)
"""

import sys
import time
import argparse
import io

# Windows 콘솔 한글 깨짐 방지
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import ServerConfig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전역 엔진 (라이프스팬에서 초기화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_engine = None
_engine_ready = False
_engine_error: Optional[str] = None
_startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 PlateEnginePro 초기화, 종료 시 정리"""
    global _engine, _engine_ready, _engine_error, _startup_time
    t0 = time.time()
    print("[서버] PlateEnginePro 초기화 중...", flush=True)
    try:
        from plate_engine_pro import PlateEnginePro
        _engine = PlateEnginePro()
        _engine_ready = True
        _startup_time = time.time() - t0
        print(f"[서버] 엔진 초기화 완료 ({_startup_time:.1f}초)", flush=True)
    except Exception as e:
        _engine_error = str(e)
        print(f"[서버] ⚠️ 엔진 초기화 실패: {e}", flush=True)
    yield
    # 종료 시 정리
    if _engine:
        try:
            _engine.close()
        except Exception:
            pass
    print("[서버] 종료됨", flush=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI 앱
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = FastAPI(
    title="YOLO26 번호판 인식 API",
    description="한국 차량 번호판 자동 인식 REST API (YOLO11x + PaddleOCR)",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS 허용 (웹 클라이언트에서 직접 호출 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 응답 스키마
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PlateResult(BaseModel):
    plate: str
    confidence: float
    bbox: list[int]          # [x1, y1, x2, y2]
    vehicle_bbox: list[int]  # [x1, y1, x2, y2]
    is_alert: bool


class DetectResponse(BaseModel):
    success: bool
    plates: list[PlateResult]
    process_ms: float
    frame_size: list[int]    # [width, height]


class StatusResponse(BaseModel):
    ready: bool
    error: Optional[str]
    startup_sec: float
    stats: dict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 엔드포인트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.get("/health")
def health():
    """헬스체크 — 엔진 준비 여부 반환"""
    return {"status": "ok" if _engine_ready else "starting", "ready": _engine_ready}


@app.get("/api/status", response_model=StatusResponse)
def status():
    """엔진 상태 및 통계"""
    stats = {}
    if _engine_ready and _engine:
        try:
            stats = dict(_engine.stats)
            # confidences 리스트는 평균값으로 압축
            confs = stats.pop("confidences", [])
            stats["avg_confidence"] = round(float(np.mean(confs)), 3) if confs else 0.0
            stats["total_detections"] = len(confs)
        except Exception:
            pass
    return StatusResponse(
        ready=_engine_ready,
        error=_engine_error,
        startup_sec=round(_startup_time, 2),
        stats=stats,
    )


@app.post("/api/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(..., description="이미지 파일 (jpg/png/bmp)")):
    """
    이미지에서 번호판 인식

    - **file**: JPEG / PNG / BMP 이미지
    - **반환**: 감지된 번호판 목록 (plate 텍스트, 신뢰도, bbox 좌표)
    """
    if not _engine_ready:
        raise HTTPException(
            status_code=503,
            detail=f"엔진 초기화 중입니다. 잠시 후 다시 시도하세요. error={_engine_error}"
        )

    # 파일 확장자 검사
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 지원됩니다 (jpg/png/bmp)")

    # 바이트 → numpy 이미지
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다")

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패 (손상된 파일)")

    h, w = frame.shape[:2]

    # 번호판 인식
    t0 = time.time()
    try:
        raw_results = _engine.process_frame(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"인식 오류: {e}")
    proc_ms = (time.time() - t0) * 1000

    # 결과 변환
    plates = []
    for r in (raw_results or []):
        plates.append(PlateResult(
            plate=r.get("plate", ""),
            confidence=round(float(r.get("confidence", 0)), 3),
            bbox=[int(v) for v in r.get("bbox", [0, 0, 0, 0])],
            vehicle_bbox=[int(v) for v in r.get("vehicle_bbox", [0, 0, 0, 0])],
            is_alert=bool(r.get("is_alert", False)),
        ))

    return DetectResponse(
        success=True,
        plates=plates,
        process_ms=round(proc_ms, 1),
        frame_size=[w, h],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 번호판 인식 API 서버")
    parser.add_argument("--host", default=ServerConfig.HOST, help="바인드 주소 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=ServerConfig.API_PORT, help="포트 (기본: 8765)")
    parser.add_argument("--workers", type=int, default=1, help="워커 수 (GPU 사용 시 1 권장)")
    args = parser.parse_args()

    print(f"[서버] 시작: http://{args.host}:{args.port}")
    print(f"[서버] API 문서: http://localhost:{args.port}/docs")

    uvicorn.run(
        "plate_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )
