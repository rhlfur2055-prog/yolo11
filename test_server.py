# -*- coding: utf-8 -*-
"""
test_server.py - 서버 번호판 인식 테스트
hiway.mp4에서 프레임 추출 → /stream 엔드포인트 + 직접 엔진 테스트
"""
import os
import sys
import time
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "movie", "hiway.mp4")
SERVER = "http://127.0.0.1:5000"

print("=" * 60)
print("  Server Plate Recognition Test")
print("=" * 60)

# 1. 서버 상태 확인
print("\n[1] Server health check...")
try:
    resp = urllib.request.urlopen(f"{SERVER}/api/stream_log", timeout=5)
    data = json.loads(resp.read())
    print(f"  OK - server responding, log entries: {len(data.get('log', []))}")
except Exception as e:
    print(f"  FAIL - {e}")
    sys.exit(1)

# 2. 영상에서 차량이 보이는 프레임 추출 후 직접 엔진 테스트
print(f"\n[2] Direct engine test on hiway.mp4...")
print(f"  Video: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("  FAIL - cannot open video")
    sys.exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"  Resolution: {w}x{h}, FPS: {fps:.1f}, Frames: {total}")

# 엔진 로드
print("\n[3] Loading PlateRecognizer engine...")
t0 = time.time()
sys.path.insert(0, SCRIPT_DIR)
from plate_recognition_4k import PlateRecognizer
recognizer = PlateRecognizer(
    confidence_threshold=0.15,
    use_sahi=False,
    frame_skip=1,
    burst_frames=1,
)
print(f"  Engine loaded in {time.time()-t0:.1f}s")

# 4. 다양한 구간에서 프레임 테스트
print("\n[4] Testing frames from different video sections...")
test_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
all_plates = []
frame_results = []

for pos in test_positions:
    target_frame = int(total * pos)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        continue

    t1 = time.time()
    try:
        results = recognizer.process_frame(frame, target_frame)
    except Exception as e:
        results = []
        print(f"  Frame {target_frame}: ERROR - {e}")
        continue
    elapsed = time.time() - t1

    plates = [r for r in results if r.get("text") and len(r.get("text", "")) >= 3]

    status = f"  Frame {target_frame:5d} ({pos*100:.0f}%): {len(plates)} plates detected ({elapsed*1000:.0f}ms)"
    if plates:
        for p in plates:
            txt = p.get("text", "")
            conf = p.get("pattern_score", 0)
            bbox = p.get("bbox", [])
            status += f"\n    -> [{txt}] score={conf:.2f} bbox={bbox}"
            all_plates.append(txt)
    print(status)
    frame_results.append({"frame": target_frame, "plates": len(plates), "ms": int(elapsed*1000)})

cap.release()

# 5. 전체 스캔 API 테스트
print(f"\n[5] Full scan API test...")
try:
    req = urllib.request.Request(
        f"{SERVER}/api/full_scan",
        data=json.dumps({"source": VIDEO_PATH}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=10)
    scan_data = json.loads(resp.read())
    print(f"  Scan started: {scan_data}")

    # 폴링
    for i in range(30):
        time.sleep(3)
        resp = urllib.request.urlopen(f"{SERVER}/api/full_scan_status", timeout=5)
        status_data = json.loads(resp.read())
        progress = status_data.get("progress", 0)
        status = status_data.get("status", "")
        results_count = len(status_data.get("results", []))
        print(f"  [{i+1}] progress={progress}%, status={status}, plates={results_count}")

        if status in ("done", "error"):
            if status == "done":
                print(f"\n  === Full Scan Results ===")
                for r in status_data.get("results", []):
                    print(f"    {r.get('time', '')} | {r.get('plate', '')} | conf={r.get('conf', 0):.2f}")
            break
except Exception as e:
    print(f"  Scan API: {e}")

# 요약
print(f"\n{'=' * 60}")
print(f"  TEST SUMMARY")
print(f"{'=' * 60}")
print(f"  Frames tested: {len(frame_results)}")
print(f"  Frames with plates: {sum(1 for f in frame_results if f['plates'] > 0)}")
print(f"  Total plates found: {len(all_plates)}")
print(f"  Unique plates: {len(set(all_plates))}")
if all_plates:
    print(f"  Plates: {', '.join(set(all_plates))}")
print(f"  Avg processing time: {sum(f['ms'] for f in frame_results)/max(len(frame_results),1):.0f}ms/frame")
print(f"{'=' * 60}")
