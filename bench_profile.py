# -*- coding: utf-8 -*-
"""hiway.mp4 300-frame profiling benchmark — instrument without modifying source"""
import sys, os, io, time, re

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import cv2
import numpy as np
import plate_engine_pro as pep

# ── Counters ──
S = {
    "ocr_calls": 0,
    "fast_path": 0,      # rec-only로 즉시 반환
    "fallback": 0,        # engine.predict 진입
    "early_return": 0,    # 빈 결과/저신뢰 조기 반환
    "fallback_ms": [],
    "yolo_vehicle_ms": [],
    "yolo_plate_ms": [],
}

# ── Patch _run_ocr: 원본 로직 그대로 + 계측만 추가 ──
def _instrumented_run_ocr(self, engine_name, engine, image):
    S["ocr_calls"] += 1
    try:
        if engine_name == "paddleocr":
            # ── Fast path: rec-only ──
            rec_text, rec_score = "", 0.0
            try:
                rec_model = engine.paddlex_pipeline.text_rec_model
                rec_results = list(rec_model.predict([image]))
                if rec_results:
                    res = rec_results[0]
                    rec_text = res.get('rec_text', '')
                    rec_score = float(res.get('rec_score', 0))
                    if rec_text and rec_score >= 0.7 and re.search(r'[가-힣]', rec_text):
                        S["fast_path"] += 1
                        return rec_text, rec_score
            except Exception:
                pass

            if not rec_text or rec_score < 0.3 or len(rec_text.strip()) < 3:
                S["early_return"] += 1
                if rec_text and rec_score > 0:
                    return rec_text, rec_score
                return "", 0.0

            # ── Fallback: engine.predict ──
            S["fallback"] += 1
            t0 = time.perf_counter()
            try:
                for res in engine.predict(image):
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    if texts:
                        text = "".join(texts)
                        conf = sum(scores) / len(scores) if scores else 0.0
                        S["fallback_ms"].append((time.perf_counter() - t0) * 1000)
                        if text:
                            return text, conf
            except Exception:
                pass
            S["fallback_ms"].append((time.perf_counter() - t0) * 1000)

            if rec_text:
                return rec_text, rec_score
    except Exception:
        pass
    return "", 0.0

pep.PlateEnginePro._run_ocr = _instrumented_run_ocr

# ── Patch YOLO __call__ ──
from ultralytics import YOLO as _YOLO
_orig_call = _YOLO.__call__

def _timed_call(self, *args, **kwargs):
    t0 = time.perf_counter()
    result = _orig_call(self, *args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    imgsz = kwargs.get("imgsz", 640)
    if imgsz == 416:
        S["yolo_vehicle_ms"].append(ms)
    else:
        S["yolo_plate_ms"].append(ms)
    return result

_YOLO.__call__ = _timed_call

# ── Run ──
VIDEO = r"C:/Users/jomg2/OneDrive/바탕 화면/movie/hiway.mp4"
MAX_FRAMES = 300

def main():
    from plate_engine_pro import PlateEnginePro
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {VIDEO}")
        return

    engine = PlateEnginePro()

    # suppress debug prints
    import builtins
    _print = builtins.print
    def quiet_print(*a, **kw):
        msg = str(a[0]) if a else ""
        if msg.startswith(("[OCR-", "[2STAGE", "[PLATE-", "[FALLBACK")):
            return
        _print(*a, **kw)
    builtins.print = quiet_print

    _print(f"=== Profiling: {MAX_FRAMES} frames ===", flush=True)

    count = 0
    t0 = time.perf_counter()
    while count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        engine.process_frame(frame, "CAM01")
        count += 1
        if count % 100 == 0:
            _print(f"  [{count}/{MAX_FRAMES}] ...", flush=True)

    elapsed = time.perf_counter() - t0
    cap.release()
    builtins.print = _print

    fps = count / elapsed
    total = S["ocr_calls"]
    fast = S["fast_path"]
    fb = S["fallback"]
    early = S["early_return"]
    fb_ms = S["fallback_ms"]
    yv = S["yolo_vehicle_ms"]
    yp = S["yolo_plate_ms"]

    print(f"\n{'='*60}")
    print(f" Profiling Results  ({count} frames / {elapsed:.1f}s / {fps:.1f} FPS)")
    print(f"{'='*60}")

    print(f"\n 1. _run_ocr 호출 합계:    {total}")
    if total:
        print(f"    Fast path (rec-only):  {fast:4d}  ({fast/total*100:5.1f}%)")
        print(f"    Fallback (predict):    {fb:4d}  ({fb/total*100:5.1f}%)")
        print(f"    Early return (empty):  {early:4d}  ({early/total*100:5.1f}%)")

    print(f"\n 2. Fallback 소요시간:")
    if fb_ms:
        print(f"    평균: {sum(fb_ms)/len(fb_ms):.0f}ms")
        print(f"    총합: {sum(fb_ms):.0f}ms  ({sum(fb_ms)/elapsed/10:.1f}% of wall time)")
    else:
        print(f"    (0 calls)")

    print(f"\n 3. YOLO 소요시간:")
    if yv:
        print(f"    yolov8n (vehicle, 416):  {sum(yv)/len(yv):.1f}ms avg × {len(yv)} calls")
    if yp:
        print(f"    best.pt (plate, 640):    {sum(yp)/len(yp):.1f}ms avg × {len(yp)} calls")
    if yv and yp:
        combo = sum(yv) + sum(yp)
        print(f"    YOLO 합계/프레임:        {combo/count:.1f}ms avg")
        print(f"    YOLO 총합:              {combo:.0f}ms  ({combo/elapsed/10:.1f}% of wall time)")

    print(f"\n 4. 시간 분해 (per frame):")
    yolo_t = sum(yv) + sum(yp)
    fb_t = sum(fb_ms)
    other_t = elapsed * 1000 - yolo_t - fb_t
    print(f"    YOLO:       {yolo_t/count:6.1f}ms  ({yolo_t/elapsed/10:5.1f}%)")
    print(f"    Fallback:   {fb_t/count:6.1f}ms  ({fb_t/elapsed/10:5.1f}%)")
    print(f"    OCR+기타:   {other_t/count:6.1f}ms  ({other_t/elapsed/10:5.1f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
