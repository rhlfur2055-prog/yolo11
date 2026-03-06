#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Jetson (JetPack 6.x) 환경 설정 — TensorRT 엔진 내보내기 & Paddle-GPU 최적화
#
# 사용법:
#   chmod +x scripts/jetson_setup.sh
#   ./scripts/jetson_setup.sh
#
# 전제: JetPack 6.x (L4T 36.x), Python 3.10, CUDA/cuDNN 설치됨
# ---------------------------------------------------------------------------
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

echo "[Jetson setup] ROOT=$ROOT"

# ── 1. 가상환경 권장 ──
if command -v python3 &>/dev/null; then
  PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
  echo "[Jetson setup] Python: $(which python3) $PYTHON_VER"
fi

# ── 2. PaddlePaddle GPU (Jetson용) ──
# 공식: https://www.paddlepaddle.org.cn/install/quick
# Jetson: pip install paddlepaddle-gpu (CUDA 10.2/11.2 등 호환 버전 확인)
if python3 -c "import paddle" 2>/dev/null; then
  echo "[Jetson setup] PaddlePaddle already installed."
  python3 -c "import paddle; print('  use_gpu:', paddle.device.is_compiled_with_cuda())"
else
  echo "[Jetson setup] Install PaddlePaddle (GPU). Example (adjust cuda version for your JetPack):"
  echo "  pip3 install paddlepaddle-gpu  # or from paddle dev repo for Jetson"
fi

# ── 3. PaddleOCR ──
pip3 install -q paddlepaddle paddleocr 2>/dev/null || true
echo "[Jetson setup] PaddleOCR: $(python3 -c 'import paddleocr; print(paddleocr.__version__)' 2>/dev/null || 'not found')"

# ── 4. YOLO TensorRT 엔진 내보내기 (선택, 지연 시간 단축) ──
# Ultralytics YOLO: export format=engine device=0
if [[ -f "$ROOT/best.pt" ]]; then
  echo "[Jetson setup] Export YOLO to TensorRT (optional, first run may take minutes)..."
  python3 -c "
from ultralytics import YOLO
m = YOLO('$ROOT/best.pt')
m.export(format='engine', device=0, half=True, simplify=True)
print('  TensorRT engine exported.')
" 2>/dev/null || echo "  Skip TensorRT export (ultralytics or GPU not available)."
else
  echo "[Jetson setup] best.pt not found, skip TensorRT export."
fi

# ── 5. GStreamer HW 디코딩 (nvv4l2decoder) 참고 ──
# CPU 점유율 최소화를 위해 Headless 캡처 시 파이프라인 예:
# gst-launch-1.0 uridecodebin uri=file:///path/to/video.mp4 ! nvvidconv ! video/x-raw,format=BGRx ! fakesink
# Python: cv2.VideoCapture('filesrc location=video.mp4 ! decodebin ! nvvidconv ! video/x-raw,format=BGRx ! appsink')
echo "[Jetson setup] GStreamer: use nvv4l2decoder/nvvidconv in pipeline for HW decode (see docs)."

echo "[Jetson setup] Done. Run headless: ./scripts/jetson_run_headless.sh --video movie/hiway.mp4"
