#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Jetson Orin Nano 실차 headless 실행 스크립트
# - 로그/evidence 자동 저장, 옵션: RTSP·파일 입력, tegrastats 수집
# 사용법:
#   ./scripts/jetson_run_headless.sh --video /data/dashcam/seg.mp4
#   ./scripts/jetson_run_headless.sh --rtsp "rtsp://192.168.1.100:554/stream1" --max-duration 1800
#   ./scripts/jetson_run_headless.sh --video /data/input.mp4 --out-dir /data/jetson_evidence --logs logs
# ---------------------------------------------------------------------------
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-$ROOT/evidence_output}"
LOGS_DIR="${LOGS_DIR:-$ROOT/logs}"
VIDEO=""
RTSP=""
MAX_DURATION=""
MAX_FRAMES=""
TEGRASTATS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --video)       VIDEO="$2"; shift 2 ;;
    --rtsp)        RTSP="$2"; shift 2 ;;
    --max-duration) MAX_DURATION="$2"; shift 2 ;;
    --max-frames)  MAX_FRAMES="$2"; shift 2 ;;
    --out-dir)     OUT_DIR="$2"; shift 2 ;;
    --logs)        LOGS_DIR="$2"; shift 2 ;;
    --tegrastats)  TEGRASTATS=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR" "$LOGS_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOGS_DIR/jetson_headless_${STAMP}.log"
METRICS_CSV="$LOGS_DIR/jetson_metrics_${STAMP}.csv"

# tegrastats 백그라운드 (Jetson 전용)
if [[ -n "$TEGRASTATS" ]] && command -v tegrastats &>/dev/null; then
  TEGRA_LOG="$LOGS_DIR/tegrastats_${STAMP}.log"
  tegrastats --interval 1000 --logfile "$TEGRA_LOG" &
  TEGRA_PID=$!
  trap "kill $TEGRA_PID 2>/dev/null || true" EXIT
fi

echo "[Jetson headless] ROOT=$ROOT OUT_DIR=$OUT_DIR LOG=$LOG_FILE"
export PLATE_CONSECUTIVE_FRAMES="${PLATE_CONSECUTIVE_FRAMES:-1}"
export EVIDENCE_OUTPUT_DIR="$OUT_DIR"

# Python headless 실행
if [[ -f "$ROOT/scripts/jetson_headless_entry.py" ]]; then
  ENTRY_ARGS=(--out-dir "$OUT_DIR" --logs-dir "$LOGS_DIR" --max-duration "${MAX_DURATION:-0}")
  [[ -n "$MAX_FRAMES" ]] && ENTRY_ARGS+=(--max-frames "$MAX_FRAMES")
  if [[ -n "$VIDEO" ]]; then
    python3 scripts/jetson_headless_entry.py --video "$VIDEO" "${ENTRY_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
  else
    python3 scripts/jetson_headless_entry.py --rtsp "$RTSP" "${ENTRY_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
  fi
else
  ARGS=()
  if [[ -n "$VIDEO" ]]; then ARGS+=("$VIDEO"); else ARGS+=("$RTSP"); fi
  [[ -n "$MAX_FRAMES" ]] && ARGS+=(--max-frames "$MAX_FRAMES")
  python3 scripts/run_headless_plate_test.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
fi

echo "[Jetson headless] Done. Evidence: $OUT_DIR Log: $LOG_FILE"
