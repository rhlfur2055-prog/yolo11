# -*- coding: utf-8 -*-
"""hiway.mp4 300-frame: FPS + profiling + 인식 번호판 목록"""
import sys, os, io, time, re
from collections import defaultdict

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
S = {"ocr": 0, "fast": 0, "fallback": 0, "early": 0, "fb_ms": []}

_orig_run_ocr = pep.PlateEnginePro._run_ocr

def _instrumented(self, engine_name, engine, image):
    S["ocr"] += 1
    try:
        if engine_name == "paddleocr":
            rec_text, rec_score = "", 0.0
            try:
                rec_model = engine.paddlex_pipeline.text_rec_model
                rec_results = list(rec_model.predict([image]))
                if rec_results:
                    res = rec_results[0]
                    rec_text = res.get('rec_text', '')
                    rec_score = float(res.get('rec_score', 0))
                    if rec_text and rec_score >= 0.7 and re.search(r'[가-힣]', rec_text):
                        S["fast"] += 1
                        return rec_text, rec_score
            except Exception:
                pass

            if not rec_text or rec_score < 0.3 or len(rec_text.strip()) < 3:
                S["early"] += 1
                if rec_text and rec_score > 0:
                    return rec_text, rec_score
                return "", 0.0

            S["fallback"] += 1
            t0 = time.perf_counter()
            try:
                for res in engine.predict(image):
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    if texts:
                        text = "".join(texts)
                        conf = sum(scores) / len(scores) if scores else 0.0
                        S["fb_ms"].append((time.perf_counter() - t0) * 1000)
                        if text:
                            return text, conf
            except Exception:
                pass
            S["fb_ms"].append((time.perf_counter() - t0) * 1000)
            if rec_text:
                return rec_text, rec_score
    except Exception:
        pass
    return "", 0.0

pep.PlateEnginePro._run_ocr = _instrumented

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

    # suppress noisy debug prints
    import builtins
    _print = builtins.print
    def quiet(*a, **kw):
        msg = str(a[0]) if a else ""
        if msg.startswith(("[OCR-", "[2STAGE", "[PLATE-", "[FALLBACK")):
            return
        _print(*a, **kw)
    builtins.print = quiet

    # Track all recognized plates
    plates_seen = defaultdict(lambda: {"count": 0, "max_conf": 0.0, "frames": []})

    _print(f"=== Benchmark: {MAX_FRAMES} frames ===", flush=True)
    count = 0
    t0 = time.perf_counter()

    while count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        results = engine.process_frame(frame, "CAM01")
        count += 1
        for r in results:
            p = r["plate"]
            c = r["confidence"]
            plates_seen[p]["count"] += 1
            plates_seen[p]["max_conf"] = max(plates_seen[p]["max_conf"], c)
            plates_seen[p]["frames"].append(count)

        if count % 100 == 0:
            _print(f"  [{count}/{MAX_FRAMES}] ...", flush=True)

    elapsed = time.perf_counter() - t0
    cap.release()
    builtins.print = _print

    fps = count / elapsed
    total = S["ocr"]
    fast = S["fast"]
    fb = S["fallback"]
    early = S["early"]
    fb_ms = S["fb_ms"]

    print(f"\n{'='*60}")
    print(f" Results ({count} frames / {elapsed:.1f}s / {fps:.1f} FPS)")
    print(f"{'='*60}")

    print(f"\n [1] OCR Path Distribution ({total} calls)")
    if total:
        print(f"     Fast path (rec-only):  {fast:4d}  ({fast/total*100:5.1f}%)")
        print(f"     Fallback (predict):    {fb:4d}  ({fb/total*100:5.1f}%)")
        print(f"     Early return:          {early:4d}  ({early/total*100:5.1f}%)")

    print(f"\n [2] Fallback timing:")
    if fb_ms:
        print(f"     Avg: {sum(fb_ms)/len(fb_ms):.0f}ms × {len(fb_ms)} calls")
        print(f"     Total: {sum(fb_ms):.0f}ms ({sum(fb_ms)/elapsed/10:.1f}% of wall)")
    else:
        print(f"     (0 calls)")

    print(f"\n [3] Recognized Plates ({len(plates_seen)} unique)")
    print(f"     {'번호판':<20s} {'횟수':>4s} {'최고conf':>8s} {'한글':>4s}  프레임 범위")
    print(f"     {'-'*60}")
    for plate in sorted(plates_seen, key=lambda p: plates_seen[p]["frames"][0]):
        info = plates_seen[plate]
        has_kr = "O" if re.search(r'[가-힣]', plate) else "X"
        frames = info["frames"]
        frange = f"{frames[0]}-{frames[-1]}" if len(frames) > 1 else str(frames[0])
        print(f"     {plate:<20s} {info['count']:>4d} {info['max_conf']:>8.3f} {has_kr:>4s}  {frange}")

    print(f"{'='*60}")

if __name__ == "__main__":
    main()
