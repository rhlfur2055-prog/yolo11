# -*- coding: utf-8 -*-
"""
[통합4] Open LPR 스타일 REST API (출처: github.com/faisalthaheem/open-lpr)
FastAPI 기반 번호판 인식 API 서버
- POST /api/v1/detect: 이미지 업로드 → 번호판 반환
- GET /api/v1/health: 서버 상태 확인
"""
import time
import os
import sys
from pathlib import Path

# 프로젝트 루트
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="ANPR API", version="1.0")

# 엔진 로딩 (지연 초기화)
_engine_pro = None
_engine_fast = None


def get_engines():
    global _engine_pro, _engine_fast
    if _engine_pro is None:
        try:
            from plate_engine_pro import PlateEnginePro, PlateEngineFast
            _engine_pro = PlateEnginePro()
            try:
                _engine_fast = PlateEngineFast()
                if not _engine_fast.available:
                    _engine_fast = None
            except Exception:
                _engine_fast = None
        except Exception as e:
            raise RuntimeError(f"엔진 로드 실패: {e}")
    return _engine_pro, _engine_fast


@app.get("/api/v1/health")
def health():
    """서버 상태 확인"""
    return {"status": "ok", "service": "anpr-api"}


@app.post("/api/v1/detect")
async def detect(file: UploadFile = File(...), engine: str = "Pro"):
    """
    이미지 업로드 → 번호판 인식
    engine: "Pro" | "Fast" (기본 Pro)
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ep, ef = get_engines()
    t0 = time.perf_counter()
    results = []
    used_engine = "Pro"
    try:
        if engine == "Fast" and ef is not None and getattr(ef, "available", False):
            results = ef.process_frame(img, "API")
            used_engine = "Fast"
        else:
            results = ep.process_frame(img, "API")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    processing_ms = (time.perf_counter() - t0) * 1000

    # 최고 confidence 1건 반환 (여러 개면 첫 번째)
    plate = ""
    confidence = 0.0
    if results:
        best = max(results, key=lambda r: r.get("confidence", 0))
        plate = best.get("plate", "")
        confidence = best.get("confidence", 0)

    return JSONResponse(content={
        "plate": plate,
        "confidence": round(confidence, 4),
        "engine": used_engine,
        "processing_ms": round(processing_ms, 2),
        "count": len(results),
    })


def run_server(host="0.0.0.0", port=8765):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    run_server(host=args.host, port=args.port)
