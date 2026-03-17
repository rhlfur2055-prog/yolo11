# -*- coding: utf-8 -*-
"""hiway.mp4 300-frame FPS benchmark (headless)"""
import sys, os, time, io

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import cv2
from plate_engine_pro import PlateEnginePro

VIDEO = r"C:/Users/jomg2/OneDrive/바탕 화면/movie/hiway.mp4"
MAX_FRAMES = 300

def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {VIDEO}")
        return

    engine = PlateEnginePro()
    print(f"=== FPS Benchmark: {MAX_FRAMES} frames ===", flush=True)

    frame_count = 0
    t0 = time.perf_counter()

    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        # Run the full pipeline (headless, no display)
        engine.process_frame(frame, "CAM01")
        frame_count += 1
        if frame_count % 50 == 0:
            elapsed = time.perf_counter() - t0
            fps = frame_count / elapsed
            print(f"  [{frame_count}/{MAX_FRAMES}] {fps:.1f} FPS ({elapsed:.1f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    cap.release()

    fps = frame_count / elapsed
    print(f"\n=== Result ===")
    print(f"  Frames: {frame_count}")
    print(f"  Time:   {elapsed:.1f}s")
    print(f"  FPS:    {fps:.1f}")
    print(f"  ms/frame: {elapsed/frame_count*1000:.0f}")

if __name__ == "__main__":
    main()
