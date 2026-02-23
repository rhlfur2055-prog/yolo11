# -*- coding: utf-8 -*-
# ============================================
# train_plate_detector.py
# YOLO26 번호판 탐지 모델 학습
# ============================================

import os
from pathlib import Path

from ultralytics import YOLO

# ── 모델 로드 (사전학습 가중치) ──
# yolo26n.pt 없으면 yolo11n.pt 등 사용 가능 (ultralytics 기본 모델)
MODEL_NAME = "yolo26n.pt"
if not Path(MODEL_NAME).exists():
    MODEL_NAME = "yolo11n.pt"  # 폴백: Ultralytics 기본 nano
model = YOLO(MODEL_NAME)

# ── 학습 시작 ──
results = model.train(
    data="dataset/yolo_format/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU 0번 사용 (CPU: device="cpu")
    project="runs/plate_detect",
    name="yolo26_plate_v1",
    # ── 학습률 스케줄 ──
    lr0=0.01,
    lrf=0.001,
    warmup_epochs=3,
    # ── 데이터 증강 (야간/우천 대응력 강화) ──
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    hsv_h=0.015,  # 색조 변화
    hsv_s=0.5,  # 채도 변화 (야간 시뮬레이션)
    hsv_v=0.4,  # 명도 변화 (조명 변화 시뮬레이션)
    degrees=5.0,  # 회전 (기울어진 번호판)
    translate=0.1,
    scale=0.3,
    shear=2.0,
    perspective=0.001,
    # ── 조기 종료 ──
    patience=20,
    # ── 저장 ──
    save=True,
    save_period=10,
)

# ── 검증 ──
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

# ── 최적 모델 내보내기 (선택) ──
best_pt = Path("runs/plate_detect/yolo26_plate_v1/weights/best.pt")
if best_pt.exists():
    export_model = YOLO(str(best_pt))
    export_model.export(format="onnx")
    print("[완료] ONNX 내보내기: runs/plate_detect/yolo26_plate_v1/weights/best.onnx")
    # TensorRT (NVIDIA GPU): export_model.export(format="engine")
