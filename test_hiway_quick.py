#!/usr/bin/env python3
"""hiway.mp4 빠른 테스트 — GT 매칭 확인"""
import cv2, sys, os, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_engine_pro import PlateEnginePro

KNOWN_PLATES = {
    '경기76바7789', '서울70바9203', '경기91바6286',
    '01나8060', '02누2754', '14나3234',
    '36다7117', '48보7062', '55저9392',
    '58두9599', '70버6393', '80부5915',
}

engine = PlateEnginePro()
engine.consecutive_required = 2

cap = cv2.VideoCapture('movie/hiway.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"hiway.mp4: {total_frames} frames, {fps:.1f}fps")

all_detected = {}
frame_idx = 0
t_start = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 3 == 0:
        results = engine.process_frame(frame)
        if results:
            for r in results:
                plate = r.get("plate", "")
                conf = r.get("confidence", 0)
                if plate:
                    if plate not in all_detected:
                        all_detected[plate] = {"ff": frame_idx, "lf": frame_idx, "mc": conf, "cnt": 1}
                    else:
                        d = all_detected[plate]
                        d["lf"] = frame_idx
                        d["mc"] = max(d["mc"], conf)
                        d["cnt"] += 1
cap.release()
elapsed = time.time() - t_start

print(f"\nDone: {elapsed:.1f}s ({frame_idx} frames)")
print(f"\n{'='*60}")
print(f"감지된 번호판: {len(all_detected)}종")
print(f"{'='*60}")

matched = 0
for plate, info in sorted(all_detected.items(), key=lambda x: x[1]["ff"]):
    gt = "✅" if plate in KNOWN_PLATES else "  "
    if plate in KNOWN_PLATES:
        matched += 1
    print(f"  {gt} {plate:<20} conf={info['mc']:.2f}  cnt={info['cnt']:>3}  frames={info['ff']}-{info['lf']}")

print(f"\n{'='*60}")
print(f"GT 매칭: {matched}/{len(KNOWN_PLATES)} ({matched*100//len(KNOWN_PLATES)}%)")
missing = KNOWN_PLATES - set(all_detected.keys())
if missing:
    print(f"미감지 GT: {missing}")
print(f"{'='*60}")
