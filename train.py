# -*- coding: utf-8 -*-
"""
train.py - YOLO26 한국 고속도로 번호판 인식 학습
================================================
RTX 4060 Laptop GPU 기준 최적화
"""
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(SCRIPT_DIR, "dataset", "data.yaml")
BASE_MODEL = os.path.join(SCRIPT_DIR, "yolo26n.pt")

print("=" * 60)
print("  YOLO26 License Plate Training")
print("  GPU: RTX 4060 Laptop")
print("=" * 60)

# YOLO26 nano 모델 로드 (사전학습 가중치)
model = YOLO(BASE_MODEL)

# 학습 실행
results = model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    name="highway_plate",
    patience=20,           # 20 epoch 개선없으면 조기종료
    augment=True,          # 데이터 증강
    device=0,              # GPU
    workers=4,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    mosaic=1.0,            # 모자이크 증강
    mixup=0.1,             # MixUp 증강
    copy_paste=0.1,        # Copy-Paste 증강
    degrees=5.0,           # 회전 증강
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    save=True,
    save_period=10,
    plots=True,
    verbose=True,
)

print("\n" + "=" * 60)
print("  Training Complete!")
print("=" * 60)
print(f"  Best model: runs/detect/highway_plate/weights/best.pt")
print(f"  Last model: runs/detect/highway_plate/weights/last.pt")
