#!/usr/bin/env python3
"""jetson_setup.sh와 동일한 검사/설치 — Windows에서도 실행 가능."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
print(f"[Jetson setup] ROOT={ROOT}")

# 1. Python
print(f"[Jetson setup] Python: {sys.executable} {sys.version_info.major}.{sys.version_info.minor}")

# 2. PaddlePaddle
try:
    import paddle
    print("[Jetson setup] PaddlePaddle already installed.")
    print("  use_gpu:", paddle.device.is_compiled_with_cuda())
except Exception as e:
    print("[Jetson setup] PaddlePaddle:", e)
    print("  pip install paddlepaddle  # or paddlepaddle-gpu on Jetson")

# 3. PaddleOCR
try:
    import paddleocr
    print(f"[Jetson setup] PaddleOCR: {getattr(paddleocr, '__version__', 'ok')}")
except Exception as e:
    print("[Jetson setup] PaddleOCR:", e)

# 4. YOLO TensorRT export (optional, Jetson/Linux에서만 권장)
best_pt = ROOT / "best.pt"
if best_pt.exists():
    print("[Jetson setup] best.pt found. TensorRT export: run on Jetson (skip on Windows).")
else:
    print("[Jetson setup] best.pt not found, skip TensorRT export.")

print("[Jetson setup] GStreamer: use nvv4l2decoder/nvvidconv on Jetson for HW decode.")
print("[Jetson setup] Done. Run headless: python -m simulation.simulation_framework movie/hiway.mp4 --goldentime")
