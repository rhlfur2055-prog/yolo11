"""
plate_gui.py - 실시간 번호판 인식 GUI v3.0 (2단계 파이프라인)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tkinter + OpenCV 하이브리드 데스크톱 애플리케이션.
실행 즉시 영상을 자동 재생하면서 번호판을 실시간 탐지.

★ v3.0 핵심: 1모델 2단계 파이프라인
  Phase 1: YOLO 탐지 즉시 → 빨간 박스 + "탐지중..." (~50ms)
  Phase 2: OCR 완료 후   → 초록 박스 + "서울바9203 (72%)"

사용법:
    python plate_gui.py                         # 기본 영상 자동 실행
    python plate_gui.py video.mp4               # 지정 영상 자동 실행
"""

from __future__ import annotations

import os
import sys
import time
import queue
import argparse
import threading

# Windows 콘솔 한글 깨짐 방지
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import Optional

import cv2
import numpy as np

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageDraw, ImageFont, ImageTk

from config import PathConfig, DisplayConfig
from tracker import PlateTracker  # noqa: F401 — re-export (외부 코드 호환)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VIDEO_DISPLAY_W = 1280
VIDEO_DISPLAY_H = 720
SIDE_PANEL_W = 0  # 우측 패널 제거 → 영상 전체 화면
REFRESH_MS = 33  # ~30 FPS display
DETECTION_SKIP = 1  # 매 프레임 탐지 (즉시 인식)
MAX_DETECTION_CARDS = 12  # 우측 패널 최대 카드 수

FONT_PATH = PathConfig.font_path(bold=True)
FONT_PATH_FALLBACK = PathConfig.font_path(bold=False)

# 다크 테마 색상
C_BG = "#0f1117"
C_PANEL = "#1a1d27"
C_SURFACE = "#252836"
C_BORDER = "#2d3148"
C_TEXT = "#e0e0e0"
C_DIM = "#888888"
C_ACCENT = "#3d5afe"
C_GREEN = "#00e676"
C_ORANGE = "#ff9800"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bbox 검증 — YOLO 오탐지 필터링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_bbox(bbox: list, frame_shape: tuple, confidence: float) -> bool:
    """YOLO bbox가 번호판으로 유효한지 검증.
    엠블럼, 그릴, 프레임 가장자리 등 오탐지를 필터링한다."""
    if len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox[:4]
    w = x2 - x1
    h = y2 - y1

    # 가로세로 비율 검증 (다양한 번호판 형태 대응: 0.5~6.0)
    aspect_ratio = w / max(h, 1)
    if not (0.5 <= aspect_ratio <= 6.0):
        return False

    # 최소 크기
    if w < 45 or h < 15:
        return False

    # 위치 필터 — 상단 3% 제외 (하단 필터 제거: 톨게이트 근접 차량 대응)
    frame_h = frame_shape[0]
    if y1 < frame_h * 0.03:
        return False

    # bbox 크기별 신뢰도 차등 적용
    # ★ PlateEnginePro의 앙상블 투표 신뢰도는 0.5~0.85 범위가 정상
    #   (다중 OCR 엔진 합산 → 단일 OCR보다 낮은 개별 점수)
    #   오탐지 필터는 YOLO confidence 기반이므로 임계값을 엔진 스케일에 맞춤
    if w >= 100:
        min_conf = 0.45
    elif w >= 60:
        min_conf = 0.50
    else:
        min_conf = 0.60
    if confidence < min_conf:
        return False

    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한글 텍스트 오버레이 (PIL 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        for fp in [FONT_PATH, FONT_PATH_FALLBACK]:
            try:
                _font_cache[size] = ImageFont.truetype(fp, size)
                break
            except OSError:
                continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_korean_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_size: int = 22,
    color: tuple[int, int, int] = (0, 255, 0),
    bg: bool = False,
    bg_alpha: float = 0.7,
) -> np.ndarray:
    """BGR 프레임 위에 한글 텍스트를 렌더링한다. frame을 복사해 반환.
    bg=True 이면 검은 반투명 배경을 먼저 그린다."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    x, y = pos
    rgb_color = (color[2], color[1], color[0])  # BGR -> RGB
    # 반투명 배경 박스
    if bg:
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 4
        bg_box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        ov_draw.rectangle(bg_box, fill=(0, 0, 0, int(255 * bg_alpha)))
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
    # 검은 외곽선
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (0, 2), (-2, 0), (2, 0)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VideoReader 스레드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VideoReader(threading.Thread):
    """영상 프레임을 읽어 display_queue / detection_queue에 전달."""

    def __init__(
        self,
        video_path: str,
        display_queue: queue.Queue,
        detection_queue: queue.Queue,
        playing: threading.Event,
    ):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.display_queue = display_queue
        self.detection_queue = detection_queue
        self.playing = playing
        self.stop_flag = threading.Event()
        self.frame_idx = 0
        self._seek_to: Optional[int] = None
        self.lock = threading.Lock()

    def seek(self, frame_idx: int) -> None:
        with self.lock:
            self._seek_to = max(0, min(frame_idx, self.total_frames - 1))

    def stop(self) -> None:
        self.stop_flag.set()

    def run(self) -> None:
        interval = 1.0 / self.fps

        while not self.stop_flag.is_set():
            # pause 대기
            if not self.playing.wait(timeout=0.1):
                continue

            # seek 처리
            with self.lock:
                if self._seek_to is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to)
                    self.frame_idx = self._seek_to
                    self._seek_to = None

            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                # 영상 끝 → 처음으로 되감기 (반복 재생)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                continue

            self.frame_idx += 1
            ts = self.frame_idx / self.fps
            data = (self.frame_idx, frame, ts)

            # display queue (항상)
            try:
                self.display_queue.put_nowait(data)
            except queue.Full:
                try:
                    self.display_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.display_queue.put_nowait(data)
                except queue.Full:
                    pass

            # detection queue (N프레임마다)
            if self.frame_idx % DETECTION_SKIP == 0:
                try:
                    self.detection_queue.put_nowait(data)
                except queue.Full:
                    pass

            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.cap.release()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DetectionWorker — detection_worker.py로 분리 (GUI ↔ 추론 책임 분리)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from detection_worker import DetectionWorker


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 GUI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PlateGUIApp(tk.Tk):
    def __init__(self, cli_args: argparse.Namespace):
        super().__init__()
        self.cli_args = cli_args
        self.title("YOLO11 번호판 인식")
        self.configure(bg=C_BG)
        self.geometry(f"{VIDEO_DISPLAY_W + 15}x{VIDEO_DISPLAY_H + 80}")
        self.minsize(800, 500)

        # 스레드 / 상태
        self.video_reader: Optional[VideoReader] = None
        self.detection_worker: Optional[DetectionWorker] = None
        self.display_queue: Optional[queue.Queue] = None
        self.detection_queue: Optional[queue.Queue] = None
        self.results_queue: Optional[queue.Queue] = None
        self.playing_event = threading.Event()

        # 오버레이 상태
        self._latest_detections: list[dict] = []
        self._detection_lock = threading.Lock()
        self._detection_ts: float = 0.0  # 마지막 감지 결과 수신 시각
        self._DETECTION_DISPLAY_TTL: float = 0.5  # Phase 2 유지 시간 (잔상 방지: 0.5초, 실제 YOLO 빈 결과 시 즉시 클리어)

        # ★ Fast YOLO 스레드: 현재 표시 프레임에 대해 실시간 YOLO 탐지
        self._current_display_frame = None  # 현재 표시 중인 프레임
        self._phase1_bboxes: list[dict] = []  # Fast YOLO Phase 1 결과
        self._phase1_lock = threading.Lock()
        self._phase1_last_active: float = 0.0  # Phase1 마지막 감지 시각
        self._fast_yolo_active = False
        self._detection_history: list[dict] = []
        self._plate_tracker = PlateTracker(iou_threshold=0.35, max_ttl=5)
        self._process_ms = 0.0
        self._process_ms_pro = 0.0
        self._process_ms_fast = 0.0
        self._fast_ms = 0.0  # fast YOLO loop 소요시간 (detect_only)
        self._det_fps_samples: list[float] = []
        self._video_w = 0
        self._video_h = 0

        # CCTV 카드 패널 이미지 참조 (GC 방지)
        self._card_images: list = []
        self._card_slots: list = []  # (frame, img_label, text_label, detail_label, time_label)

        # 로깅 + 자동 저장
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        self._results_dir = os.path.join(self._script_dir, "plate_results_v3")
        os.makedirs(self._results_dir, exist_ok=True)
        self._log_path = os.path.join(self._script_dir, "gui_log.txt")
        self._saved_count = 0
        self._log("=== PlateGUI v3.0 시작 ===")

        # UI 빌드
        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # 디스플레이 루프 시작
        self.after(REFRESH_MS, self._refresh_display)

    # ─── 로깅 ───

    def _log(self, msg: str) -> None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    # ─── UI 빌드 ───

    def _build_ui(self) -> None:
        """SARADA STEELHEAD CCTV 스타일 레이아웃:
        ┌──────────────────┬──────────┐
        │                  │ Detection │
        │   Main Video     │  Cards   │
        │   (CCTV Feed)    │  Panel   │
        │                  │          │
        ├──────────────────┴──────────┤
        │  Control Bar + Status       │
        └─────────────────────────────┘
        """
        # ── 맨 하단: 컨트롤 바 (먼저 pack → 나머지가 위에 채워짐) ──
        bar = tk.Frame(self, bg=C_SURFACE, height=44)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        btn_frame = tk.Frame(bar, bg=C_SURFACE)
        btn_frame.pack(side=tk.LEFT, padx=6, pady=6)

        tk.Button(btn_frame, text="\u25b6 재생", command=self._on_play, bg=C_GREEN, fg="#000",
                  font=("Segoe UI", 9), relief=tk.FLAT, padx=10, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="\u23f8 일시정지", command=self._on_pause, bg=C_ORANGE, fg="#000",
                  font=("Segoe UI", 9), relief=tk.FLAT, padx=10, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="\U0001f4c2 파일 열기", command=self._on_file_open, bg=C_ACCENT, fg="white",
                  font=("Segoe UI", 9), relief=tk.FLAT, padx=10, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="\U0001f4be 저장", command=self._on_save_log, bg=C_PANEL, fg=C_TEXT,
                  font=("Segoe UI", 9), relief=tk.FLAT, padx=10, pady=4).pack(side=tk.LEFT, padx=2)

        self.stats_var = tk.StringVar(value="동영상 파일을 열어주세요.")
        tk.Label(bar, textvariable=self.stats_var, bg=C_SURFACE, fg=C_DIM,
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=12, pady=4, fill=tk.X, expand=True)

        # ── 메인 영역: 좌측 영상 + 우측 감지 패널 ──
        main = tk.Frame(self, bg=C_BG)
        main.pack(fill=tk.BOTH, expand=True)

        # ── 우측 패널 제거 — 영상 전체 화면 ──
        self._det_count_var = tk.StringVar(value="0건")
        self._card_frame = tk.Frame(main, bg=C_PANEL)  # 더미 (호환용)

        # ── 메인 영상 영역 (전체 화면) ──
        video_frame = tk.Frame(main, bg="#000000")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.video_label = tk.Label(video_frame, bg="#000000", anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ── 내부 변수 ──
        self.plate_text_var = tk.StringVar(value="---")
        self.conf_var = tk.StringVar(value="")
        self.engine_choice_var = tk.StringVar(value="Pro 엔진")
        self.multiframe_var = tk.BooleanVar(value=False)
        # Listbox 호환 (기존 코드 참조용)
        self.history_list = None

    def _bind_keys(self) -> None:
        self.bind("<q>", lambda e: self._on_close())
        self.bind("<Q>", lambda e: self._on_close())
        self.bind("<Escape>", lambda e: self._on_close())

    def _on_play(self) -> None:
        self.playing_event.set()
        if self.video_reader:
            self.stats_var.set("재생 중")

    def _on_pause(self) -> None:
        self.playing_event.clear()
        self.stats_var.set("일시정지")

    def _on_file_open(self) -> None:
        path = filedialog.askopenfilename(
            title="동영상 선택",
            filetypes=[("동영상", "*.mp4 *.avi *.mov *.mkv"), ("모든 파일", "*.*")],
        )
        if path:
            self._open_video(path)

    def _on_save_log(self) -> None:
        if not self._detection_history:
            messagebox.showinfo("저장", "저장할 인식 기록이 없습니다.")
            return
        path = filedialog.asksaveasfilename(
            title="인식 로그 저장", defaultextension=".json", initialdir=self._results_dir,
            filetypes=[("JSON", "*.json"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        try:
            import json
            out = []
            for h in self._detection_history:
                rec = {k: v for k, v in h.items()
                       if k not in ("plate_image", "preprocessed_image", "plate_img", "preprocessed")
                       and not isinstance(v, np.ndarray)}
                out.append(rec)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("저장 완료", f"{len(out)}건 저장: {path}")
        except Exception as e:
            messagebox.showerror("저장 오류", str(e))

    # ─── 영상 열기 (자동 시작) ───

    def _open_video(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            self.stats_var.set(f"File not found: {path}")
            return

        self._stop_threads()
        self._latest_detections = []
        self._detection_history = []
        self._plate_tracker.reset()
        # ★ 엔진 내부 트래커도 리셋 (이전 영상 트랙 잔존 방지)
        if self.detection_worker and hasattr(self.detection_worker, 'pro_engine') and self.detection_worker.pro_engine:
            self.detection_worker.pro_engine.reset_state()
        # 패널 초기화
        self._det_count_var.set("0건")
        self.plate_text_var.set("---")
        self.conf_var.set("")

        # 큐 생성
        self.display_queue = queue.Queue(maxsize=3)
        self.detection_queue = queue.Queue(maxsize=2)
        self.results_queue = queue.Queue(maxsize=10)

        # Detection worker
        self.stats_var.set("Loading model...")
        self.update()

        choice = self.engine_choice_var.get()
        engine_mode = "pro" if "Pro" in choice else ("fast" if "Fast" in choice else "auto")
        use_pro = getattr(self.cli_args, "pro_engine", True)
        kwargs = {
            "model_size": self.cli_args.model_size,
            "confidence_threshold": self.cli_args.confidence,
            "use_sahi": False,
            "frame_skip": 1,
            "burst_frames": 1,
            "use_pro_engine": use_pro,
            "engine_mode": engine_mode,
            "use_multiframe": self.multiframe_var.get(),
        }
        self.detection_worker = DetectionWorker(
            self.detection_queue, self.results_queue, recognizer_kwargs=kwargs,
        )
        self.detection_worker.start()

        # Video reader
        self.playing_event = threading.Event()
        self.video_reader = VideoReader(
            path, self.display_queue, self.detection_queue, self.playing_event,
        )
        self._video_w = self.video_reader.width
        self._video_h = self.video_reader.height
        self.video_reader.start()

        self.title(f"YOLO11 번호판 인식 - {os.path.basename(path)} ({self._video_w}x{self._video_h})")

        # 엔진 로딩 대기
        self.stats_var.set("엔진 로딩 중... 잠시 기다려주세요")
        self.update()
        if not self.detection_worker.ready.wait(timeout=60):
            self.stats_var.set("엔진 로딩 타임아웃 - 재생 시작")
        else:
            self.stats_var.set("엔진 로딩 완료! 재생 시작")
            # ★ 영상 모드: consecutive_required=1 (즉시 표시)
            #   Ghost Detection은 PlateTracker가 담당 → 중복 보호 불필요
            #   =2이면 프레임 스킵(3프레임)과 충돌 → 첫 감지 후 3-9초 대기 필요 → Det:0 원인
            if self.detection_worker.pro_engine:
                self.detection_worker.pro_engine.consecutive_required = 1
        self.update()

        self.playing_event.set()
        self.stats_var.set(f"Auto-playing {self._video_w}x{self._video_h} @ {self.video_reader.fps:.0f}fps")

        # ★ Fast YOLO 스레드 시작 (현재 표시 프레임에서 실시간 bbox 탐지)
        self._fast_yolo_active = True
        _fast_t = threading.Thread(target=self._fast_yolo_loop, daemon=True)
        _fast_t.start()

    # ─── Fast YOLO 스레드: 현재 표시 프레임에서 ~8fps YOLO 탐지 ───

    def _fast_yolo_loop(self):
        """현재 표시 프레임: YOLO detect_only → 즉시 bbox 표시 (~3-8fps).
        ★ OCR은 Worker 전용 (PaddleOCR 공유 블로킹 방지).
        Fast loop = YOLO bbox만 → Worker = 전체 OCR + 텍스트."""
        while self._fast_yolo_active:
            frame = self._current_display_frame
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                engine = None
                if self.detection_worker and self.detection_worker.ready.is_set():
                    engine = self.detection_worker.pro_engine
                if engine is None or not hasattr(engine, 'detect_only'):
                    time.sleep(0.1)
                    continue
                # ★ YOLO만 실행 (OCR 없음) → PaddleOCR 블로킹 없이 즉시 bbox
                t0 = time.time()
                results = engine.detect_only(frame)
                _fast_ms = (time.time() - t0) * 1000
                phase1 = []
                for r in results:
                    bbox = r.get("bbox", [])
                    if len(bbox) < 4:
                        continue
                    phase1.append({
                        "bbox": bbox,
                        "confidence": r.get("confidence", 0),
                        "phase": 1,
                        "text": "",
                        "is_valid_plate": False,
                        "ocr_confidence": 0,
                    })
                with self._phase1_lock:
                    self._phase1_bboxes = phase1
                    self._fast_ms = _fast_ms
                    if phase1:  # 감지 성공 시 시각 갱신
                        self._phase1_last_active = time.time()
            except Exception:
                pass
            time.sleep(0.02)  # YOLO ~200-500ms + 최소 딜레이

    # ─── 디스플레이 루프 (30 FPS) ───

    def _refresh_display(self) -> None:
        # 1) 인식 결과 수신 (엔진 트래커가 이미 처리 → GUI 트래커 제거)
        while self.results_queue is not None:
            try:
                data = self.results_queue.get_nowait()
                with self._detection_lock:
                    raw_results = data["results"]
                    frame_idx = data.get("frame_idx", 0)

                    # ── bbox 검증: 오탐지 필터링 ──
                    validated = []
                    for det in raw_results:
                        bbox = det.get("bbox", [])
                        conf = det.get("ocr_confidence", det.get("confidence", 0))
                        frame_shape = (self._video_h, self._video_w)
                        if validate_bbox(bbox, frame_shape, conf):
                            validated.append(det)

                    # ★ GUI PlateTracker 제거 — process_frame_realtime 엔진 내부
                    #   트래커가 이미 Ghost Detection 처리 완료
                    #   이중 트래커는 고스트 텍스트 원인이었음

                    # ★ Phase 2 교체 로직 (잔상 방지):
                    #   - 감지 결과 있으면 → 즉시 교체
                    #   - 빈 결과 + 실제 YOLO 처리(>100ms) → 즉시 클리어 (차량 떠남)
                    #   - 빈 결과 + 캐시(<100ms) → 유지 (YOLO가 아직 재검증 안 함)
                    _proc_ms = data["process_ms"]
                    if validated:
                        self._latest_detections = validated
                        self._detection_ts = time.time()
                    # ★ 빈 결과로 즉시 클리어 하지 않음 — TTL(1.5초)이 만료 담당
                    # 이전: _proc_ms >= 100이면 클리어 → OCR 결과가 다음 빈 프레임에 사라짐
                    # 지금: 새 결과 올 때만 교체, 없으면 TTL까지 유지
                    # 디버그: GUI 수신 결과 추적
                    print(f"[GUI-RX] frame={frame_idx} raw={len(raw_results)} "
                          f"valid={len(validated)} proc={data['process_ms']:.0f}ms "
                          f"latest={len(self._latest_detections)}", flush=True)
                    for v in validated:
                        print(f"  → {v.get('text','')} conf={v.get('ocr_confidence',0):.2f} "
                              f"valid={v.get('is_valid_plate')} bbox={v.get('bbox',[])}",
                              flush=True)

                    if data["process_ms"] >= 50:  # 캐시 결과(<50ms) 제외 → FPS 정확도 개선
                        self._process_ms = data["process_ms"]
                        self._process_ms_pro = data.get("process_ms_pro", 0)
                        self._process_ms_fast = data.get("process_ms_fast", 0)
                        self._det_fps_samples.append(1000.0 / data["process_ms"])
                        if len(self._det_fps_samples) > 10:
                            self._det_fps_samples.pop(0)
                    ts = data.get("timestamp", 0)
                    for det in validated:
                        text = det.get("text", "")
                        conf = det.get("ocr_confidence", det.get("confidence", 0))
                        is_valid = det.get("is_valid_plate", False)
                        # ★ 로그 표시 조건: 화면 표시(Phase 2)와 동일 — is_valid + text
                        if is_valid and text and len(text) >= 2:
                            self._add_to_history(det, ts)
            except (queue.Empty, AttributeError):
                break

        # 2) 비디오 프레임 수신 (최신 것만)
        frame_data = None
        while self.display_queue is not None:
            try:
                frame_data = self.display_queue.get_nowait()
            except (queue.Empty, AttributeError):
                break

        if frame_data is not None:
            frame_idx, frame, ts = frame_data

            # ★ Fast YOLO 스레드용 현재 프레임 저장
            self._current_display_frame = frame

            with self._detection_lock:
                # ★ 감지 결과 TTL: 3초 (거리 기반 매칭이 교차 오염 방지)
                # Phase1 없으면 즉시 클리어, Phase1 있으면 3초까지 유지
                _det_age = time.time() - self._detection_ts
                if _det_age > 3.0:
                    self._latest_detections = []

                phase2_dets = list(self._latest_detections)

            with self._phase1_lock:
                phase1_dets = list(self._phase1_bboxes)

            merged = []

            # ★ 새 차량 진입 감지: Phase1이 0→1+ 전환 시 이전 OCR 클리어
            _has_p1 = len(phase1_dets) > 0
            if not _has_p1:
                self._phase1_empty_since = getattr(self, '_phase1_empty_since', 0) or time.time()
            else:
                _empty_dur = time.time() - getattr(self, '_phase1_empty_since', 0) if getattr(self, '_phase1_empty_since', 0) > 0 else 0
                if _empty_dur > 0.3:
                    # ★ Phase1이 0.3초 이상 비었다가 복귀 → 새 차량 → 이전 OCR 클리어
                    with self._detection_lock:
                        self._latest_detections = []
                        self._detection_ts = 0
                    phase2_dets = []
                self._phase1_empty_since = 0

            # ★ Phase1 없으면 Phase2 표시 안 함 (잔상 완전 차단)
            if not phase1_dets:
                phase2_dets = []

            # ★ Phase1-Phase2 매칭: 가장 가까운 Phase1에 무조건 매칭
            # 거리 제한 제거 — TTL(3초) + 새차량감지(0.3초갭)로 잔상 방지
            _fw = frame.shape[1] if frame is not None else 1920
            _fh = frame.shape[0] if frame is not None else 1080

            # Phase1 유효 bbox 필터링
            _valid_p1 = []
            for i, p1 in enumerate(phase1_dets):
                p1b = p1.get("bbox", [])
                if len(p1b) < 4:
                    continue
                if p1.get("confidence", 1.0) < 0.45:
                    continue
                _pw = p1b[2] - p1b[0]
                _ph = max(p1b[3] - p1b[1], 1)
                if _pw < 40 or _ph < 15:
                    continue
                # Phase1 bbox 중심점
                _cx = (p1b[0] + p1b[2]) / 2
                _cy = (p1b[1] + p1b[3]) / 2
                _valid_p1.append((i, _cx, _cy, p1))

            matched_p1_indices = set()

            # ★ 번호판다운 Phase1 bbox만 추출 (종횡비 1.5 이상)
            _plate_like_p1 = [(i, cx, cy, p1) for i, cx, cy, p1 in _valid_p1
                              if (p1["bbox"][2] - p1["bbox"][0]) /
                                 max(p1["bbox"][3] - p1["bbox"][1], 1) >= 1.5]

            # Phase2 결과 → 가장 가까운 Phase1 bbox에 매칭
            for p2 in phase2_dets:
                p2b = p2.get("bbox", [])
                if len(p2b) < 4:
                    continue
                _p2cx = (p2b[0] + p2b[2]) / 2
                _p2cy = (p2b[1] + p2b[3]) / 2

                # 1차: 거리 기반 매칭
                best_dist = float("inf")
                best_idx = -1
                best_p1 = None
                for idx, cx, cy, p1 in _valid_p1:
                    if idx in matched_p1_indices:
                        continue
                    _dist = ((cx - _p2cx) ** 2 + (cy - _p2cy) ** 2) ** 0.5
                    if _dist < best_dist:
                        best_dist = _dist
                        best_idx = idx
                        best_p1 = p1

                combined = dict(p2)
                # ★ 거리 제한 없이 가장 가까운 Phase1에 무조건 매칭
                # 잔상 방지는 TTL(3초) + 새차량감지(Phase1 0.3초 갭→클리어)로 처리
                if best_p1 is not None:
                    combined["bbox"] = best_p1["bbox"]
                    matched_p1_indices.add(best_idx)
                    merged.append(combined)
                elif (len(_plate_like_p1) >= 1
                      and _det_age < 3.0):
                    # ★ 폴백: valid_p1에 매칭 안 됐지만 plate_like Phase1 존재
                    for _fb_idx, _, _, _fb_p1 in _plate_like_p1:
                        if _fb_idx not in matched_p1_indices:
                            combined["bbox"] = _fb_p1["bbox"]
                            matched_p1_indices.add(_fb_idx)
                            merged.append(combined)
                            break

            # 2) Phase 1 중 Phase 2와 매칭 안 된 것 → "탐지중..." 표시
            # ★ 오감지 필터: 저신뢰, 비율 이상, 프레임 가장자리 bbox 제외
            for i, p1 in enumerate(phase1_dets):
                if i not in matched_p1_indices:
                    if p1.get("confidence", 0) < 0.50:
                        continue
                    p1b = p1.get("bbox", [])
                    if len(p1b) < 4:
                        continue
                    _pw = p1b[2] - p1b[0]
                    _ph = max(p1b[3] - p1b[1], 1)
                    # 번호판 종횡비 필터 (가로 > 세로 × 1.5)
                    if _pw / _ph < 1.5:
                        continue
                    # ★ 프레임 하단 20% 가장자리 bbox 제외 (도로/바닥/워터마크 오탐지)
                    if p1b[3] > _fh * 0.80:
                        continue
                    # ★ 프레임 우측 5% 가장자리 초과 bbox 제외
                    if p1b[0] > _fw * 0.95:
                        continue
                    # ★ 최소 크기 필터 (너무 작은 bbox는 번호판이 아님)
                    if _pw < 50 or _ph < 15:
                        continue
                    merged.append(p1)

            annotated = self._draw_overlay(frame, merged)

            # 리사이즈 (종횡비 유지)
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            if label_w < 100:
                label_w, label_h = VIDEO_DISPLAY_W, VIDEO_DISPLAY_H
            scale = min(label_w / self._video_w, label_h / self._video_h) if self._video_w > 0 else 1.0
            disp_w = int(self._video_w * scale)
            disp_h = int(self._video_h * scale)
            if disp_w > 0 and disp_h > 0:
                display = cv2.resize(annotated, (disp_w, disp_h))
            else:
                display = annotated

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            self.video_label.configure(image=tk_img)
            self.video_label.image = tk_img

            self._update_stats(frame_idx, ts)

        # 모델 로딩 상태 체크
        if (self.detection_worker is not None
                and not self.detection_worker.ready.is_set()
                and self.detection_worker.loading_msg):
            self.stats_var.set(self.detection_worker.loading_msg)

        self.after(REFRESH_MS, self._refresh_display)

    # ─── 오버레이 그리기 (2단계) ───

    # 오버레이 최대 표시 개수 (화면 정리용)
    MAX_PHASE2_DISPLAY = 5   # OCR 확정 박스 최대 표시 수
    MAX_PHASE1_DISPLAY = 5   # YOLO 탐지 즉시 박스 (OCR 전 노란 박스)

    def _draw_overlay(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """MareArts ANPR 스타일 오버레이 — 프로페셔널 UI
        - 좌상단: 시스템 정보 패널 (타이틀 + FPS + 해상도)
        - 번호판 bbox: 두꺼운 초록 테두리 + 상단 배너 라벨
        - 우측: 감지 이력 사이드 패널 (최근 5건)
        """
        result = frame  # ★ copy() 제거 — FPS 극대화
        h, w = result.shape[:2]

        # ── 좌상단 시스템 정보 패널 ──
        fps = 0
        if hasattr(self, '_det_fps_samples') and self._det_fps_samples:
            fps = sum(self._det_fps_samples) / len(self._det_fps_samples)
        # 반투명 배경 패널
        _panel_overlay = result.copy()
        cv2.rectangle(_panel_overlay, (0, 0), (220, 65), (20, 20, 20), -1)
        cv2.addWeighted(_panel_overlay, 0.7, result, 0.3, 0, result)
        # 타이틀 + 초록 액센트 바
        cv2.rectangle(result, (0, 0), (5, 65), (0, 255, 0), -1)
        cv2.putText(result, "YOLO11 ANPR", (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(result, f"FPS: {fps:.0f}  |  {w}x{h}", (14, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # ── 우측 감지 이력 사이드 패널 (최근 5건, PIL 한글 렌더링) ──
        _hist = getattr(self, '_detection_history', [])
        if _hist:
            _show_count = min(5, len(_hist))
            _panel_w = 300
            _row_h = 32
            _panel_h = 30 + _show_count * _row_h
            _px = w - _panel_w - 10
            _py = 10
            # 반투명 배경
            _side_overlay = result.copy()
            cv2.rectangle(_side_overlay, (_px, _py), (_px + _panel_w, _py + _panel_h),
                          (20, 20, 20), -1)
            cv2.addWeighted(_side_overlay, 0.7, result, 0.3, 0, result)
            # 헤더 바
            cv2.rectangle(result, (_px, _py), (_px + _panel_w, _py + 26), (0, 80, 0), -1)
            # PIL로 한글 포함 텍스트 일괄 렌더링
            _pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            _pil_draw = ImageDraw.Draw(_pil_img)
            _font_head = _get_font(16)
            _font_plate = _get_font(15)
            _font_conf = _get_font(13)
            _pil_draw.text((_px + 8, _py + 5), f"Detection Log ({len(_hist)})",
                           font=_font_head, fill=(0, 255, 0))
            for _i in range(_show_count):
                _entry = _hist[_i]
                _plate = (_entry.get("text") or _entry.get("plate", ""))[:14]
                _conf = _entry.get("ocr_confidence", _entry.get("confidence", 0))
                _cnt = _entry.get("count", 1)
                _ey = _py + 30 + _i * _row_h
                # 번호판 텍스트 (흰색)
                _pil_draw.text((_px + 10, _ey + 2), _plate,
                               font=_font_plate, fill=(255, 255, 255))
                # conf + count (시안색)
                _conf_str = f"{_conf:.0%} x{_cnt}"
                _pil_draw.text((_px + _panel_w - 85, _ey + 4), _conf_str,
                               font=_font_conf, fill=(0, 220, 220))
            result = cv2.cvtColor(np.array(_pil_img), cv2.COLOR_RGB2BGR)

        # ── 번호판 bbox + 라벨 (MareArts 스타일: 라벨을 bbox 위에 배치) ──
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            text = det.get("text") or det.get("plate", "")
            is_valid = det.get("is_valid_plate", False)
            ocr_conf = det.get("ocr_confidence", det.get("confidence", 0))
            x1, y1, x2, y2 = [int(v) for v in bbox]

            if is_valid and text and ocr_conf >= 0.70:
                # ★ 확실한 인식: 초록 bbox + 상단 배너 라벨
                det_conf = det.get("pattern_score", ocr_conf)
                d_val = min(99, int(det_conf * 100))
                r_val = min(99, int(ocr_conf * 100))

                # 두꺼운 초록 bbox (3px)
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # 코너 강조 (MareArts 스타일 — 코너에 짧은 굵은 선)
                _corner_len = max(10, (x2 - x1) // 5)
                _ct = 4  # 코너 두께
                cv2.line(result, (x1, y1), (x1 + _corner_len, y1), (0, 255, 0), _ct)
                cv2.line(result, (x1, y1), (x1, y1 + _corner_len), (0, 255, 0), _ct)
                cv2.line(result, (x2, y1), (x2 - _corner_len, y1), (0, 255, 0), _ct)
                cv2.line(result, (x2, y1), (x2, y1 + _corner_len), (0, 255, 0), _ct)
                cv2.line(result, (x1, y2), (x1 + _corner_len, y2), (0, 255, 0), _ct)
                cv2.line(result, (x1, y2), (x1, y2 - _corner_len), (0, 255, 0), _ct)
                cv2.line(result, (x2, y2), (x2 - _corner_len, y2), (0, 255, 0), _ct)
                cv2.line(result, (x2, y2), (x2, y2 - _corner_len), (0, 255, 0), _ct)

                # ★ 라벨을 bbox 위쪽에 배너로 렌더링 (번호만 표시)
                label = text
                bbox_h = y2 - y1
                font_sz = max(20, int(bbox_h * 0.7))
                _label_y = max(font_sz + 8, y1 - 4)
                # ★ 라벨 너비 추정: 한글 1글자 ≈ font_sz * 0.65
                _label_w_est = int(len(label) * font_sz * 0.65)
                _label_x = max(0, min(x1, w - _label_w_est))
                # ★ 라벨이 프레임 밖으로 넘어가면 표시 안 함
                if x1 > w - 20 or x2 < 20:
                    continue
                # ★ 라벨이 화면 상단/좌측 밖으로 나가지 않도록
                _label_top = max(0, _label_y - font_sz - 2)
                result = draw_korean_text(result, label, (_label_x, _label_top),
                                          font_size=font_sz, color=(0, 255, 0), bg=True)

            elif is_valid and text and ocr_conf >= 0.50:
                # ★ 중간 신뢰: 노란 bbox + 번호만 (경계 클램핑)
                if x1 > w - 20 or x2 < 20:
                    continue
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 255), 2)
                bbox_h = y2 - y1
                font_sz = max(18, int(bbox_h * 0.6))
                _label_y = max(font_sz + 8, y1 - 4)
                _label_w_est = int(len(text) * font_sz * 0.65)
                _label_x = max(0, min(x1, w - _label_w_est))
                _label_top = max(0, _label_y - font_sz - 2)
                result = draw_korean_text(result, text, (_label_x, _label_top),
                                          font_size=font_sz, color=(0, 200, 255), bg=True)

            elif det.get("phase") == 1 and det.get("confidence", 0) >= 0.5:
                # ★ Phase 1 탐지중: 얇은 초록 점선 스타일 bbox
                if x1 > w - 20 or x2 < 20:
                    continue
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
                bbox_h = y2 - y1
                font_sz = max(16, min(int(bbox_h * 0.8), 28))
                _label_y = max(font_sz + 8, y1 - 4)
                _label_x = max(0, min(x1, w - int(4 * font_sz * 0.65)))
                _label_top = max(0, _label_y - font_sz - 2)
                result = draw_korean_text(result, "탐지중...", (_label_x, _label_top),
                                          font_size=font_sz, color=(0, 255, 0), bg=True)
            # ★ 미인식/저신뢰: bbox 표시 안 함 (엠블럼 오감지 방지)

        return result

    # ─── Detection Log ───

    def _add_to_history(self, det: dict, timestamp: float = 0) -> None:
        text = (det.get("text") or det.get("plate", "")).strip()
        if not text:
            return
        conf = det.get("ocr_confidence", det.get("confidence", 0))
        # ★ 잔상/오인식 필터: 부분 인식(5자 이하) 및 저신뢰(70% 미만) 차단
        # "대6282", "바6286" 같은 부분 인식이 Detection Log에 누적되는 것 방지
        import re as _re
        _hangul_count = len(_re.findall(r'[가-힣]', text))
        if len(text) < 7 and _hangul_count <= 1:
            return  # 부분 인식 차단 (최소 7자 이상 또는 한글 2자 이상)
        if conf < 0.70:
            return  # 저신뢰 차단
        det_norm = dict(det)
        det_norm["text"] = text
        det_norm["ocr_confidence"] = conf

        for existing in self._detection_history:
            if self._text_similar(existing.get("text", ""), text):
                if conf > existing.get("ocr_confidence", 0):
                    # ★ is_valid_plate=True 보존 (False로 덮어쓰기 방지)
                    was_valid = existing.get("is_valid_plate", False)
                    existing.update(det_norm)
                    if was_valid:
                        existing["is_valid_plate"] = True
                existing["count"] = existing.get("count", 1) + 1
                existing["timestamp"] = timestamp
                self._refresh_history_list()
                return

        entry = dict(det_norm)
        entry["count"] = 1
        entry["timestamp"] = timestamp
        self._detection_history.insert(0, entry)
        self._refresh_history_list()
        self._auto_save_plate(entry)
        self.plate_text_var.set(text)
        self.conf_var.set(f"Conf: {conf:.1%}")

    def _auto_save_plate(self, det: dict) -> None:
        try:
            import json
            idx = self._saved_count
            plate_img = det.get("plate_image")
            if plate_img is not None:
                cv2.imwrite(os.path.join(self._results_dir, f"plate_{idx:04d}.png"), plate_img)
            pre_img = det.get("preprocessed_image")
            if pre_img is not None:
                cv2.imwrite(os.path.join(self._results_dir, f"plate_{idx:04d}_preprocessed.png"), pre_img)

            _EXCLUDE = {"plate_image", "preprocessed_image", "plate_img", "preprocessed", "phase"}
            det_record = {k: v for k, v in det.items() if k not in _EXCLUDE and not isinstance(v, np.ndarray)}
            det_record["index"] = idx

            results_path = os.path.join(self._results_dir, "results.json")
            existing = []
            if os.path.isfile(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(det_record)

            from plate_recognition_4k import NumpyEncoder
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            self._saved_count += 1
            self._log(f"저장 #{idx}: {det.get('text', '?')} conf={det.get('ocr_confidence', 0):.3f}")
        except Exception as e:
            self._log(f"저장 오류: {e}")

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return PlateGUIApp._levenshtein(s2, s1)
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if c1 == c2 else 1)))
            prev = curr
        return prev[-1]

    def _text_similar(self, t1: str, t2: str) -> bool:
        """같은 차량의 유사 인식 결과인지 판단.
        전체 levenshtein ≤ 2 또는 숫자부분 levenshtein ≤ 1 이면 동일 차량."""
        t1 = t1.replace(" ", "")
        t2 = t2.replace(" ", "")
        if t1 == t2:
            return True
        if not t1 or not t2:
            return False
        if abs(len(t1) - len(t2)) > 2:
            return False
        # 전체 텍스트 비교
        if self._levenshtein(t1, t2) <= 2:
            return True
        # ★ 숫자 부분만 비교 (한글 오인식 허용): 숫자가 거의 같으면 같은 차량
        import re
        d1 = re.sub(r'[^0-9]', '', t1)
        d2 = re.sub(r'[^0-9]', '', t2)
        if len(d1) >= 5 and len(d2) >= 5 and self._levenshtein(d1, d2) <= 1:
            return True
        return False

    def _refresh_history_list(self) -> None:
        """인식 기록 카운트만 갱신 (우측 패널 제거됨)."""
        self._det_count_var.set(f"{len(self._detection_history)}건")

    # ─── 상태 표시 ───

    def _update_stats(self, frame_idx: int, ts: float) -> None:
        if self.video_reader is None:
            return
        total = self.video_reader.total_frames
        total_sec = total / self.video_reader.fps if self.video_reader.fps > 0 else 0
        det_fps = sum(self._det_fps_samples) / len(self._det_fps_samples) if self._det_fps_samples else 0.0
        n_det = len(self._latest_detections)
        mins, secs = divmod(int(ts), 60)
        t_mins, t_secs = divmod(int(total_sec), 60)

        # ★ Fast = fast YOLO loop (detect_only), Pro = Worker (process_frame 전체)
        _fast_display = self._fast_ms if self._fast_ms > 0 else self._process_ms_fast
        speed_info = f"Pro:{self._process_ms_pro:.0f}ms / Fast:{_fast_display:.0f}ms"

        status = (
            f"{mins:02d}:{secs:02d}/{t_mins:02d}:{t_secs:02d}  "
            f"F:{frame_idx}/{total}  Det:{n_det}  "
            f"DetFPS:{det_fps:.1f}  {speed_info}  "
            f"Log:{len(self._detection_history)}"
        )
        self.stats_var.set(status)

        if frame_idx % 5 == 0 and det_fps > 0:
            print(f"[F{frame_idx}] DetFPS:{det_fps:.2f} Proc:{self._process_ms:.0f}ms "
                  f"Det:{n_det} Log:{len(self._detection_history)}", flush=True)

    # ─── 종료 ───

    def _stop_threads(self) -> None:
        if self.video_reader is not None:
            self.video_reader.stop()
            self.playing_event.set()
            self.video_reader.join(timeout=2)
            self.video_reader = None
        if self.detection_worker is not None:
            self.detection_worker.stop()
            self.detection_worker.join(timeout=3)
            self.detection_worker = None

    def _on_close(self) -> None:
        self._log(f"=== PlateGUI 종료 === (저장: {self._saved_count}건, 기록: {len(self._detection_history)}건)")
        self._stop_threads()
        self.destroy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 엔트리포인트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_VIDEO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "temp_youtube", "plate_hd.mp4",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time Plate Recognition GUI v3.0")
    parser.add_argument("video", nargs="?", default=None, help="Video file path")
    parser.add_argument("--youtube", metavar="URL", default=None,
                        help="YouTube URL (다운로드 후 재생, yt-dlp 필요)")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"],
                        help="Plate model size (default: n)")
    parser.add_argument("--confidence", type=float, default=0.15,
                        help="Min detection confidence (default: 0.15)")
    parser.add_argument("--pro-engine", action="store_true", default=True,
                        help="Use PlateEnginePro (기본값)")
    parser.add_argument("--no-pro-engine", action="store_false", dest="pro_engine",
                        help="Use 4k recognizer instead of Pro")
    args = parser.parse_args()

    video_path = None
    if args.youtube:
        try:
            from youtube_helper import download_youtube, check_ytdlp
        except ImportError:
            print("[오류] youtube_helper.py를 찾을 수 없습니다.", flush=True)
            sys.exit(1)
        if not check_ytdlp():
            print("[오류] yt-dlp가 설치되지 않았습니다. 설치: pip install yt-dlp", flush=True)
            sys.exit(1)
        try:
            video_path = download_youtube(args.youtube)
            print(f"[YouTube] 로컬 파일로 재생: {video_path}", flush=True)
        except Exception as e:
            print(f"[오류] YouTube 다운로드 실패: {e}", flush=True)
            sys.exit(1)
    elif args.video:
        video_path = args.video
    else:
        video_path = DEFAULT_VIDEO

    app = PlateGUIApp(args)
    app.after(200, lambda: app._open_video(video_path))
    app.mainloop()


if __name__ == "__main__":
    main()
