# -*- coding: utf-8 -*-
"""
extract_frames.py - hiway.mp4 영상에서 프레임 이미지 추출
- 5프레임 간격
- 밝기 임계값 50 이상 필터링 (어두운 프레임 제외)
"""
import cv2
import os
import sys
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "movie", "hiway.mp4")
TRAIN_DIR = os.path.join(SCRIPT_DIR, "dataset", "images", "train")
SAVE_INTERVAL = 5       # 매 5프레임마다 1장
BRIGHTNESS_MIN = 50     # 평균 밝기 임계값

os.makedirs(TRAIN_DIR, exist_ok=True)

video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    print(f"[ERROR] Cannot open: {VIDEO_PATH}")
    sys.exit(1)

total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")
print(f"Save interval: every {SAVE_INTERVAL} frames, brightness >= {BRIGHTNESS_MIN}")

frame_count = 0
save_count = 0
skipped_dark = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    if frame_count % SAVE_INTERVAL == 0:
        # 밝기 필터링: 그레이스케일 평균 밝기 체크
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if brightness >= BRIGHTNESS_MIN:
            filename = os.path.join(TRAIN_DIR, f"frame_{save_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            save_count += 1
        else:
            skipped_dark += 1

    frame_count += 1
    if frame_count % 500 == 0:
        print(f"  {frame_count}/{total} processed, {save_count} saved, {skipped_dark} dark skipped")

video.release()
print(f"\nDone! {save_count} frames saved, {skipped_dark} dark frames skipped")
print(f"Output: {TRAIN_DIR}")
