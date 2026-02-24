"""
plate_server.py - 번호판 인식 웹 서버 (Flask)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용법:
    python plate_server.py
    python plate_server.py --port 5000

기능:
    - http://localhost:5000 에서 영상 업로드
    - 번호판 인식 엔진(plate_recognition_4k)으로 처리
    - 실시간 결과 JSON + 웹 UI 표시
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlparse
import ssl

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify, redirect, send_from_directory, Response
from plate_recognition_4k import PlateRecognizer, NumpyEncoder

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_BASE = os.path.join(os.path.dirname(__file__), "server_results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_BASE, exist_ok=True)

# 진행중인 작업 상태
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# 실시간 스트림 인식 로그 + 번호판 크롭 이미지
_stream_detection_log: list[dict] = []
_stream_current_plate: dict = {}
_stream_log_lock = threading.Lock()
MAX_STREAM_LOG = 100

# ★ 쿨다운 전역 (브라우저 새로고침/스트림 재연결 후에도 유지)
# {plate_text: last_logged_unix_time}
_stream_recent_plates: dict[str, float] = {}
_STREAM_PLATE_COOLDOWN = 30.0

# 기본 재생 소스 (로컬 파일 우선)
_DEFAULT_VIDEO = os.path.join(os.path.dirname(__file__), "youtube_-dlpsCgjCAg.mkv")

# 폰 카메라 실시간 인식용 엔진 (한 번만 로드)
_stream_engine = None
_stream_engine_lock = threading.Lock()


def _get_stream_engine():
    """실시간 스트림 전용 초고속 엔진 (싱글톤 캐시)."""
    global _stream_engine
    with _stream_engine_lock:
        if _stream_engine is None:
            from plate_engine_pro import PlateEnginePro
            _stream_engine = PlateEnginePro()
            # ★ 실시간 고속 설정: 전처리 2가지만, 연속확인 1프레임, conf 낮춤
            _stream_engine.consecutive_required = 1
            # 전처리: original + clahe + sharpen
            _stream_engine.config.PREPROCESS_METHODS = ["original", "clahe", "sharpen"]
            _stream_engine.config.DETECT_CONF = 0.05   # 최대한 낮춰 감지율 향상
            _stream_engine.config.OCR_CONF = 0.30      # 특수문자는 clean_ocr_text에서 차단, conf는 낮춤
            _stream_engine.config.CONSECUTIVE_FRAMES_REQUIRED = 1
            from plate_engine_pro import PlateValidator
            _stream_engine.validator = PlateValidator()
        return _stream_engine

# 기존 결과 디렉토리 자동 스캔
RESULT_DIR_PREFIX = "plate_results"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 보조 유틸리티: ROI 파서 / YouTube 다운로드 / MJPEG 스트림
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_roi_norm(s: str | None) -> list[tuple[float, float]] | None:
    """
    ROI 문자열을 정규화 다각형으로 파싱.
    지원 형식:
      - "x1,y1 x2,y2 x3,y3 x4,y4" (0~1 사이 실수)
      - JSON 배열: [[x,y], [x,y], ...]
    """
    if not s:
        return None
    s = s.strip()
    try:
        if s.startswith("["):
            import json as _json
            arr = _json.loads(s)
            pts = [(float(x), float(y)) for x, y in arr]
            return pts
        parts = s.replace(";", " ").split()
        pts = []
        for p in parts:
            x, y = p.split(",")
            pts.append((float(x), float(y)))
        return pts
    except Exception:
        return None

def _download_youtube(url: str) -> str:
    """yt-dlp로 YouTube 영상을 로컬로 받아 경로 반환."""
    try:
        import yt_dlp
    except Exception as e:
        raise RuntimeError("yt-dlp가 필요합니다: pip install yt-dlp") from e
    dl_dir = os.path.join(os.path.dirname(__file__), "temp_youtube")
    os.makedirs(dl_dir, exist_ok=True)
    opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[height<=720]/best[ext=mp4]/best",
        "outtmpl": os.path.join(dl_dir, "%(title).50s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        if not filename.endswith(".mp4"):
            base, _ = os.path.splitext(filename)
            if os.path.isfile(base + ".mp4"):
                filename = base + ".mp4"
    return filename

def _parse_stream_source(source: str):
    """스트림 소스 파싱: 웹캠은 정수, 그 외는 문자열. 파일 경로는 그대로, 나머지만 YouTube 보정."""
    s = (source or "").strip()
    if s in ("0", "1", "2"):
        return int(s)
    # 파일 경로(백슬래시 또는 C: 등 드라이브)면 YouTube로 바꾸지 않고 그대로 사용 (예: C:\tools\yolo26\youtube_-dlpsCgjCAg)
    if "\\" in s or (len(s) >= 2 and s[1] == ":"):
        return s
    lower = s.lower()
    if "youtube" in lower and "http" not in lower:
        rest = s.split("youtube", 1)[-1].strip()
        if rest and len(rest) >= 5 and all(c.isalnum() or c in "-_" for c in rest):
            s = f"https://www.youtube.com/watch?v={rest}"
    elif "http" not in lower and len(s) >= 10 and all(c.isalnum() or c in "-_" for c in s):
        s = f"https://www.youtube.com/watch?v={s}"
    return s


# ── PIL 한글 폰트 로드 (한번만) ──
_PIL_FONT_SMALL = None
_PIL_FONT_MED   = None
_PIL_FONT_LARGE = None
def _load_pil_fonts():
    global _PIL_FONT_SMALL, _PIL_FONT_MED, _PIL_FONT_LARGE
    if _PIL_FONT_LARGE is not None:
        return
    try:
        from PIL import ImageFont
        for path in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc",
                     "C:/Windows/Fonts/malgunbd.ttf"]:
            if os.path.exists(path):
                _PIL_FONT_SMALL = ImageFont.truetype(path, 16)
                _PIL_FONT_MED   = ImageFont.truetype(path, 22)
                _PIL_FONT_LARGE = ImageFont.truetype(path, 32)
                return
    except Exception:
        pass

def _put_kr_text(frame: np.ndarray, text: str, xy: tuple, color=(0, 255, 150), font=None):
    """PIL로 한글 텍스트를 프레임에 그리기."""
    try:
        from PIL import Image, ImageDraw
        _load_pil_fonts()
        f = font or _PIL_FONT_MED
        if f is None:
            cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text(xy, text, font=f, fill=(color[2], color[1], color[0]))
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _draw_recent_plates_panel(frame: np.ndarray, recent_plates: list[tuple[str, float]], max_entries: int = 12):
    """프레임 우측에 '인식된 번호판' 패널 그리기 (PIL 한글 렌더링)."""
    _load_pil_fonts()
    h, w = frame.shape[:2]
    panel_w = min(280, w // 3)
    x0 = w - panel_w - 20
    y0, row_h = 20, 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w - 10, y0 + 20 + max_entries * row_h), (30, 30, 40), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x0, y0), (w - 10, y0 + 20 + max_entries * row_h), (70, 90, 255), 2)
    cv2.putText(frame, "Detected Plates", (x0 + 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, (text, conf_or_time) in enumerate(recent_plates[:max_entries]):
        if not text:
            continue
        y_pos = y0 + 28 + (i + 1) * row_h
        conf_str = f"{conf_or_time:.0%}" if isinstance(conf_or_time, float) and conf_or_time <= 1.0 else ""
        _put_kr_text(frame, text, (x0 + 12, y_pos - 16), color=(0, 255, 150), font=_PIL_FONT_MED)
        if conf_str:
            cv2.putText(frame, conf_str, (x0 + panel_w - 52, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


def _yield_loading_frame(msg: str = "Loading..."):
    """MJPEG 한 프레임(로딩 메시지) yield. OpenCV는 한글 미지원이므로 영문 사용."""
    w, h = 640, 360
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (40, 42, 52)
    cv2.putText(frame, msg[:50], (max(10, w // 2 - 200), h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Please wait", (w // 2 - 70, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    ok, jpg = cv2.imencode(".jpg", frame)
    if ok:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"


def _yield_error_stream(msg_line1: str, msg_line2: str, duration_sec: int = 60):
    """오류 메시지를 MJPEG로 계속 전송해 검은 화면 대신 안내 표시 (OpenCV는 영문만)."""
    w, h = 640, 360
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (50, 30, 30)
    cv2.putText(frame, msg_line1[:45], (30, h // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.putText(frame, msg_line2[:50], (30, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    ok, jpg = cv2.imencode(".jpg", frame)
    if not ok:
        return
    chunk = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
    for _ in range(max(1, duration_sec * 2)):
        yield chunk
        time.sleep(0.5)


def _mjpeg_stream_from_source(source: str, roi_norm: list[tuple[float, float]] | None):
    """웹 기반 실시간 스트림: 번호판이 영상 위에 바로 오버레이 (가벼운 설정으로 즉시 표시)."""
    from collections import deque
    from plate_engine_pro import PlateEnginePro

    cap_source = _parse_stream_source(source)
    if isinstance(cap_source, str):
        lower = cap_source.lower()
        if "youtube.com" in lower or "youtu.be" in lower:
            download_result = {"path": None, "done": False, "error": None}

            def _do_download():
                try:
                    download_result["path"] = _download_youtube(cap_source)
                except Exception as e:
                    download_result["error"] = str(e)
                finally:
                    download_result["done"] = True

            t = threading.Thread(target=_do_download, daemon=True)
            t.start()
            while not download_result["done"]:
                for _ in _yield_loading_frame("YouTube download... (real-time prep)"):
                    yield _
                time.sleep(0.3)
            if download_result["error"]:
                err_msg = (download_result["error"] or "")[:40].replace("\n", " ")
                for _ in _yield_error_stream("YouTube error", err_msg or "Check yt-dlp", 30):
                    yield _
                return
            cap_source = download_result["path"]

    # Windows 웹캠은 CAP_DSHOW 사용 시 더 안정적
    if isinstance(cap_source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(cap_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened() and isinstance(cap_source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(cap_source)
    # 경로에 확장자가 없으면 .mp4 시도 (예: C:\tools\yolo26\youtube_-dlpsCgjCAg)
    if not cap.isOpened() and isinstance(cap_source, str) and "." not in os.path.basename(cap_source):
        alt = cap_source.rstrip("/\\") + ".mp4"
        if os.path.isfile(alt):
            cap_source = alt
            cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        for _ in _yield_error_stream(
            "Cannot open video. Use: video file path or full YouTube URL",
            "Example: C:\\video.mp4  or  https://www.youtube.com/watch?v=xxxx",
            60,
        ):
            yield _
        return

    try:
        # 엔진은 싱글톤 캐시에서 재사용 (매 요청마다 로드 방지)
        engine = _get_stream_engine()

        frame_idx = 0
        last_results = []
        last_results_age = 0   # last_results가 몇 프레임 됐는지
        # 동영상 FPS에 맞게 sleep 계산 (너무 빠르게 읽지 않도록)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_sleep = max(0.01, 1.0 / fps_src)
        detect_every = 2   # 2프레임에 1번 추론 (정확도↑)
        no_plate_streak = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(cap_source, int):
                    time.sleep(0.1)
                    continue
                break
            frame_idx += 1
            results = []
            if frame_idx % detect_every == 0 and not _scan_active:
                # ★ 추론 해상도 640px + 프레임 선명화로 원거리 번호판 감지력 향상
                h0, w0 = frame.shape[:2]
                max_w = 640
                if w0 > max_w:
                    scale = max_w / w0
                    small = cv2.resize(frame, (max_w, int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                else:
                    scale = 1.0
                    small = frame
                # 프레임 선명화 (번호판 엣지 강화)
                sharp_k = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
                small_sharp = cv2.filter2D(small, -1, sharp_k)
                try:
                    with _engine_use_lock:
                        raw = engine.process_frame(small_sharp, camera_id="STREAM", full_frame=frame)
                        if not raw:
                            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
                            l, a, b_ch = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                            lab = cv2.merge([clahe.apply(l), a, b_ch])
                            small_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                            raw = engine.process_frame(small_clahe, camera_id="STREAM", full_frame=frame)
                        if not raw:
                            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype("float32")
                            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
                            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)
                            small_sat = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
                            raw = engine.process_frame(small_sat, camera_id="STREAM", full_frame=frame)
                    if raw:
                        results = raw
                        last_results = results
                        last_results_age = 0
                        no_plate_streak = 0
                        detect_every = 2
                    else:
                        last_results_age += 1
                        no_plate_streak += 1
                        if no_plate_streak > 15:
                            detect_every = 4
                except Exception:
                    last_results_age += 1
            # ★ 캐시된 이전 결과는 최대 10프레임(약 0.5초)만 재사용
            if not results:
                if last_results_age <= 10:
                    results = last_results
                else:
                    results = []   # 너무 오래된 결과는 버림 (이전 차 잔상 방지)

            for r in results:
                text = (r.get("plate") or r.get("text") or "").strip()
                if not text:
                    continue
                conf = float(r.get("confidence", r.get("ocr_confidence", 0)))

                # 번호판 크롭 Base64
                plate_b64 = ""
                try:
                    bbox = r.get("bbox")
                    if bbox and len(bbox) == 4:
                        x1c2, y1c2, x2c2, y2c2 = [int(v) for v in bbox]
                        h_f, w_f = frame.shape[:2]
                        crop = frame[max(0,y1c2):min(h_f,y2c2), max(0,x1c2):min(w_f,x2c2)]
                        if crop.size > 0:
                            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            if ok:
                                import base64 as _b64
                                plate_b64 = _b64.b64encode(buf.tobytes()).decode()
                except Exception:
                    pass

                # current_plate는 항상 최신값으로 업데이트
                with _stream_log_lock:
                    now_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    _stream_current_plate.update({"plate": text, "time": now_iso, "img": plate_b64})

                # ── Detection Log: 전역 쿨다운 (브라우저 재연결해도 유지) ──
                def _tail_digits(s):
                    d = "".join(c for c in s if c.isdigit())
                    return d[-4:] if len(d) >= 4 else ""

                def _plate_similar(a, b):
                    if a == b:
                        return True
                    if a == b[::-1] or b == a[::-1]:
                        return True
                    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 1:
                        return True
                    # 끝 4자리 숫자 동일 + 길이 차이 3 이하
                    ta, tb = _tail_digits(a), _tail_digits(b)
                    if ta and tb and ta == tb and abs(len(a) - len(b)) <= 3:
                        return True
                    return False

                now_t = time.time()
                # 전역 dict에서 유사 항목 탐색
                similar_key = next(
                    (k for k, t in _stream_recent_plates.items() if _plate_similar(text, k)),
                    None
                )

                def _do_log():
                    with _stream_log_lock:
                        _stream_detection_log.append({"time": now_iso, "plate": text, "status": "성공", "img": plate_b64})
                        if len(_stream_detection_log) > MAX_STREAM_LOG:
                            _stream_detection_log.pop(0)

                if similar_key is None:
                    # 완전히 새 번호판
                    _stream_recent_plates[text] = now_t
                    _do_log()
                elif (now_t - _stream_recent_plates[similar_key]) > _STREAM_PLATE_COOLDOWN:
                    # 쿨다운 경과 → 키 갱신 후 재기록
                    del _stream_recent_plates[similar_key]
                    _stream_recent_plates[text] = now_t
                    _do_log()

            if roi_norm:
                h, w = frame.shape[:2]
                pts = np.array([(int(x * w), int(y * h)) for x, y in roi_norm], dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 140, 0), thickness=2)

            for r in results:
                bbox = r.get("bbox", None)
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = (0, 230, 70)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = (r.get("plate") or r.get("text") or "").strip()
                if label:
                    _put_kr_text(frame, label, (x1, max(5, y1 - 28)),
                                 color=color, font=_PIL_FONT_LARGE)

            # 최근 인식 번호판 패널 (전역 쿨다운 dict → 최신순 최대 12개)
            _recent_sorted = sorted(_stream_recent_plates.items(), key=lambda x: -x[1])
            _draw_recent_plates_panel(frame, [(k, v) for k, v in _recent_sorted])
            cv2.rectangle(frame, (8, 8), (240, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Frame: {frame_idx}  LIVE", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            # 스트리밍 해상도는 960px 이하로 제한 (네트워크 부하 감소)
            h_out, w_out = frame.shape[:2]
            if w_out > 960:
                ratio = 960.0 / w_out
                frame = cv2.resize(frame, (960, int(h_out * ratio)))

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            time.sleep(frame_sleep)
    except Exception as e:
        err = str(e).replace("\n", " ")[:50]
        for _ in _yield_error_stream("Stream error (engine or frame)", err or "Check console", 30):
            yield _
    finally:
        cap.release()


def find_all_result_dirs(base_dir: str = ".") -> list[str]:
    dirs = []
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith(RESULT_DIR_PREFIX):
            json_path = os.path.join(full, "results.json")
            if os.path.isfile(json_path):
                dirs.append(name)
    # server_results 하위도 탐색
    if os.path.isdir(RESULTS_BASE):
        for name in sorted(os.listdir(RESULTS_BASE)):
            full = os.path.join(RESULTS_BASE, name)
            rj = os.path.join(full, "results.json")
            if os.path.isdir(full) and os.path.isfile(rj):
                dirs.append(os.path.join("server_results", name))
    return dirs


def process_video_job(job_id: str, video_path: str, roi_norm: list[tuple[float, float]] | None = None):
    """백그라운드 영상 처리 스레드."""
    result_dir = os.path.join(RESULTS_BASE, job_id)
    os.makedirs(result_dir, exist_ok=True)

    with _jobs_lock:
        _jobs[job_id]["status"] = "loading"
        _jobs[job_id]["message"] = "모델 로딩중..."

    try:
        recognizer = PlateRecognizer()
        if roi_norm:
            recognizer.set_roi_polygon(roi_norm)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        with _jobs_lock:
            _jobs[job_id].update({
                "status": "processing",
                "message": "영상 처리중...",
                "total_frames": total_frames,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "processed_frames": 0,
                "plates_found": 0,
                "plates": [],
            })

        all_results = recognizer.process_video(
            video_path,
            output_dir=result_dir,
            progress_callback=lambda frame_idx, det_count: _update_progress(
                job_id, frame_idx, total_frames, det_count
            ),
        )

        confirmed = recognizer.get_confirmed_plates()

        # 결과 저장
        _EXCLUDE_KEYS = {"plate_image", "preprocessed_image", "plate_img", "preprocessed"}
        clean_results = []
        for r in all_results:
            clean = {k: v for k, v in r.items()
                     if k not in _EXCLUDE_KEYS and not isinstance(v, np.ndarray)}
            clean_results.append(clean)

        with open(os.path.join(result_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        clean_confirmed = []
        for c in confirmed:
            cc = {k: v for k, v in c.items()
                  if k not in _EXCLUDE_KEYS and not isinstance(v, np.ndarray)}
            clean_confirmed.append(cc)

        with _jobs_lock:
            _jobs[job_id].update({
                "status": "completed",
                "message": "완료",
                "plates": clean_results,
                "confirmed": clean_confirmed,
                "total_plates": len(clean_results),
                "total_confirmed": len(clean_confirmed),
            })

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "error",
                "message": str(e),
            })


def _update_progress(job_id: str, frame_idx: int, total: int, det_count: int):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["processed_frames"] = frame_idx
            _jobs[job_id]["plates_found"] = det_count
            pct = frame_idx / total * 100 if total > 0 else 0
            _jobs[job_id]["progress_pct"] = round(pct, 1)


def _download_url(url: str, dest_path: str) -> None:
    """URL을 dest_path에 다운로드 (이미지/영상 공통)."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; PlateServer/1.0)"})
    ctx = ssl.create_default_context()
    if url.lower().startswith("https:"):
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urlopen(req, timeout=30, context=ctx) as resp:
        with open(dest_path, "wb") as f:
            f.write(resp.read())


def process_url_job(job_id: str, url: str, local_path: str):
    """URL에서 받은 파일(이미지 또는 영상)로 번호판 인식."""
    result_dir = os.path.join(RESULTS_BASE, job_id)
    os.makedirs(result_dir, exist_ok=True)
    _EXCLUDE_KEYS = {"plate_image", "preprocessed_image", "plate_img", "preprocessed"}

    with _jobs_lock:
        _jobs[job_id]["status"] = "loading"
        _jobs[job_id]["message"] = "모델 로딩중..."

    try:
        recognizer = PlateRecognizer()
        roi_norm = _parse_roi_norm(_jobs.get(job_id, {}).get("roi"))
        if roi_norm:
            recognizer.set_roi_polygon(roi_norm)
        ext = (Path(local_path).suffix or "").lower()

        # 이미지인지 영상인지: 확장자 또는 OpenCV로 판단
        is_image = ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
        if not is_image:
            cap = cv2.VideoCapture(local_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            is_image = frame_count <= 1

        if is_image:
            frame = cv2.imread(local_path)
            if frame is None:
                frame = cv2.VideoCapture(local_path).read()[1]
            if frame is None:
                raise ValueError(
                    "이미지를 열 수 없습니다. URL이 직접 영상/이미지 링크인지 확인하세요. "
                    "YouTube는 상단 '실시간 스트림 시작' 버튼을 사용하세요."
                )

            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "processing",
                    "message": "이미지 인식중...",
                    "total_frames": 1,
                    "processed_frames": 0,
                    "plates_found": 0,
                    "plates": [],
                })
            results = recognizer.process_frame(frame, 0)
            confirmed = recognizer.get_confirmed_plates()
        else:
            cap = cv2.VideoCapture(local_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "processing",
                    "message": "영상 처리중...",
                    "total_frames": total_frames,
                    "fps": fps,
                    "resolution": f"{w}x{h}",
                    "processed_frames": 0,
                    "plates_found": 0,
                    "plates": [],
                })
            all_results = recognizer.process_video(
                local_path,
                output_dir=result_dir,
                progress_callback=lambda frame_idx, det_count: _update_progress(
                    job_id, frame_idx, total_frames, det_count
                ),
            )
            results = all_results
            confirmed = recognizer.get_confirmed_plates()

        clean_results = []
        for r in results:
            clean = {k: v for k, v in r.items()
                     if k not in _EXCLUDE_KEYS and not isinstance(v, np.ndarray)}
            clean_results.append(clean)
        with open(os.path.join(result_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        clean_confirmed = []
        for c in confirmed:
            cc = {k: v for k, v in c.items()
                  if k not in _EXCLUDE_KEYS and not isinstance(v, np.ndarray)}
            clean_confirmed.append(cc)

        with _jobs_lock:
            _jobs[job_id].update({
                "status": "completed",
                "message": "완료",
                "plates": clean_results,
                "confirmed": clean_confirmed,
                "total_plates": len(clean_results),
                "total_confirmed": len(clean_confirmed),
                "progress_pct": 100.0,
                "processed_frames": (1 if is_image else _jobs[job_id].get("processed_frames", 0)),
            })
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "error",
                "message": str(e),
            })
    finally:
        try:
            if os.path.isfile(local_path) and UPLOAD_DIR in os.path.abspath(local_path):
                os.remove(local_path)
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML 페이지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INDEX_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>번호판 인식 시스템</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', -apple-system, sans-serif; background: #0f1117; color: #e0e0e0; min-height: 100vh; }
  header { background: linear-gradient(135deg, #1a1d27, #252836); padding: 20px 32px; border-bottom: 1px solid #2d3148; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.5rem; font-weight: 700; color: #fff; }
  header .badge { background: #3d5afe; color: #fff; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; }
  .container { max-width: 1000px; margin: 0 auto; padding: 32px; }

  /* 업로드 영역 */
  .upload-box {
    border: 2px dashed #3d5afe; border-radius: 16px; padding: 48px;
    text-align: center; cursor: pointer; transition: all 0.2s;
    background: rgba(61,90,254,0.05); margin-bottom: 32px;
  }
  .upload-box:hover { background: rgba(61,90,254,0.12); border-color: #536dfe; }
  .upload-box.dragover { background: rgba(61,90,254,0.2); border-color: #7c8eff; }
  .upload-box h2 { font-size: 1.3rem; color: #7c8eff; margin-bottom: 8px; }
  .upload-box p { color: #888; font-size: 0.9rem; }
  .upload-box input { display: none; }

  /* URL 입력 */
  .url-box { margin-bottom: 24px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .url-box input[type="url"], .url-box input[type="text"] {
    flex: 1; min-width: 200px; padding: 12px 16px; border-radius: 8px; border: 1px solid #2d3148;
    background: #1a1d27; color: #e0e0e0; font-size: 0.9rem;
  }
  .url-box input::placeholder { color: #666; }
  .url-box button {
    padding: 12px 20px; border-radius: 8px; border: none; background: #3d5afe; color: #fff;
    font-weight: 600; cursor: pointer; white-space: nowrap;
  }
  .url-box button:hover { background: #536dfe; }
  .url-box button:disabled { opacity: 0.6; cursor: not-allowed; }

  /* 진행률 */
  .progress-section { display: none; margin-bottom: 32px; }
  .progress-section.active { display: block; }
  .progress-bar-wrap { background: #1a1d27; border-radius: 8px; height: 24px; overflow: hidden; border: 1px solid #2d3148; }
  .progress-bar { height: 100%; background: linear-gradient(90deg, #3d5afe, #536dfe); border-radius: 8px; transition: width 0.3s; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600; min-width: 40px; }
  .progress-info { display: flex; gap: 24px; margin-top: 12px; flex-wrap: wrap; }
  .progress-info .item { background: #1a1d27; border: 1px solid #2d3148; border-radius: 8px; padding: 12px 20px; }
  .progress-info .item .val { font-size: 1.4rem; font-weight: 700; color: #3d8afe; }
  .progress-info .item .lbl { font-size: 0.75rem; color: #888; margin-top: 2px; }

  /* 결과 */
  .results-section { display: none; }
  .results-section.active { display: block; }
  .results-section h2 { font-size: 1.2rem; color: #aaa; margin-bottom: 16px; }
  .confirmed-list { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }
  .plate-card { background: #1a1d27; border: 1px solid #2d3148; border-radius: 12px; padding: 16px; min-width: 160px; text-align: center; }
  .plate-card .plate-text { font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 4px; letter-spacing: 0.08em; }
  .plate-card .plate-conf { font-size: 0.8rem; color: #4caf50; }
  .plate-card .plate-frames { font-size: 0.7rem; color: #666; margin-top: 4px; }

  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 16px; }
  thead th { background: #1e2133; padding: 10px 12px; text-align: left; color: #888; font-weight: 600; position: sticky; top: 0; }
  tbody tr { border-top: 1px solid #1e2133; }
  tbody tr:hover { background: #1a1d27; }
  td { padding: 8px 12px; }
  .text-col { font-weight: 700; color: #fff; letter-spacing: 0.05em; }
  .conf-high { color: #4caf50; font-weight: 600; }
  .conf-mid { color: #ff9800; }
  .conf-low { color: #f44336; }
  .method-tag { padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
  .method-tag.plate_model { background: #1a3a6b; color: #7eb8f7; }
  .method-tag.log_ocr { background: #1a4a2b; color: #7ef7a8; }

  /* 기존 결과 */
  .history-section { margin-top: 32px; padding-top: 24px; border-top: 1px solid #2d3148; }
  .history-section h2 { font-size: 1.1rem; color: #aaa; margin-bottom: 12px; }
  .dir-links { display: flex; flex-wrap: wrap; gap: 8px; }
  .dir-link { background: #1a1d27; border: 1px solid #2d3148; border-radius: 8px; padding: 10px 20px; color: #e0e0e0; text-decoration: none; font-size: 0.85rem; transition: border-color 0.2s; }
  .dir-link:hover { border-color: #3d5afe; }
  .dir-link span { color: #3d5afe; }

  .footer { text-align: center; padding: 32px; color: #444; font-size: 0.8rem; }
</style>
</head>
<body>

<header>
  <h1>번호판 인식 시스템</h1>
  <span class="badge">v6 AI Hub</span>
  <a class="dir-link" href="/live" style="margin-left:auto; background:#3d5afe; color:#fff;">실시간(홈)</a>
  <a class="dir-link" href="/html/roadmap.html">로드맵</a>
</header>

<div class="container">
  <!-- 업로드 -->
  <div class="upload-box" id="uploadBox" onclick="document.getElementById('fileInput').click()">
    <h2>영상 파일 업로드</h2>
    <p>클릭하거나 파일을 드래그하세요 (.mp4, .avi, .mkv)</p>
    <input type="file" id="fileInput" accept="video/*">
  </div>

  <div class="url-box">
    <input type="url" id="urlInput" placeholder="이미지 또는 영상 URL 입력 (https://...)" />
    <button type="button" id="urlSubmit">URL로 번호판 인식</button>
  </div>

  <div class="url-box" style="border-color:#3d5afe;">
    <p style="margin-bottom:8px; color:#8ab4f8; font-size:0.9rem;"><strong>동영상 번호 인식</strong> — 동영상 파일 경로 또는 YouTube URL 입력 후 재생하면서 번호판 표시</p>
    <input type="text" id="streamSrc" value="" placeholder="동영상 경로 또는 YouTube URL (예: C:\video.mp4)" />
    <input type="text" id="streamRoi" placeholder='ROI (선택)' />
    <button type="button" id="streamBtn">실시간 스트림 시작</button>
  </div>

  <!-- 진행률 -->
  <div class="progress-section" id="progressSection">
    <div class="progress-bar-wrap">
      <div class="progress-bar" id="progressBar" style="width: 0%">0%</div>
    </div>
    <div class="progress-info">
      <div class="item"><div class="val" id="statStatus">대기</div><div class="lbl">상태</div></div>
      <div class="item"><div class="val" id="statFrames">0</div><div class="lbl">처리 프레임</div></div>
      <div class="item"><div class="val" id="statPlates">0</div><div class="lbl">인식 번호판</div></div>
      <div class="item"><div class="val" id="statRes">-</div><div class="lbl">해상도</div></div>
    </div>
  </div>

  <!-- 결과 -->
  <div class="results-section" id="resultsSection">
    <h2>확정 번호판</h2>
    <div class="confirmed-list" id="confirmedList"></div>

    <h2>전체 인식 결과 (<span id="totalCount">0</span>건)</h2>
    <div style="overflow-x:auto; border-radius:12px; border:1px solid #2d3148;">
    <table>
      <thead>
        <tr><th>#</th><th>번호판</th><th>OCR</th><th>탐지</th><th>패턴</th><th>방식</th></tr>
      </thead>
      <tbody id="resultsBody"></tbody>
    </table>
    </div>

    <div style="margin-top:16px;">
      <a id="jsonLink" href="#" target="_blank" style="color:#3d5afe; font-size:0.85rem;">결과 JSON 다운로드</a>
    </div>
  </div>

  <!-- 기존 결과 -->
  <div class="history-section" id="historySection"></div>

  <!-- 사이트 내 HTML 문서 -->
  <div class="history-section" id="htmlSection"></div>
</div>

<div class="footer">plate_server.py Flask v6 | plate_recognition_4k engine</div>

<script>
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');

// 드래그 앤 드롭
uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', e => {
  e.preventDefault(); uploadBox.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length > 0) uploadFile(fileInput.files[0]); });

document.getElementById('urlSubmit').addEventListener('click', function() {
  const url = document.getElementById('urlInput').value.trim();
  if (!url) { alert('URL을 입력하세요.'); return; }
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    alert('http 또는 https로 시작하는 URL을 입력하세요.'); return;
  }
  const btn = this;
  btn.disabled = true;
  progressSection.classList.add('active');
  resultsSection.classList.remove('active');
  document.getElementById('statStatus').textContent = 'URL 다운로드중...';
  fetch('/api/recognize_url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: url, roi: document.getElementById('streamRoi').value.trim() })
  })
    .then(r => r.json())
    .then(data => {
      if (data.job_id) pollProgress(data.job_id);
      else alert('요청 실패: ' + (data.error || '알 수 없는 오류'));
    })
    .catch(err => alert('오류: ' + err))
    .finally(() => { btn.disabled = false; });
});

// 실시간 스트림 열기 (새 창, 비어 있으면 0=웹캠)
document.getElementById('streamBtn').addEventListener('click', function() {
  const src = document.getElementById('streamSrc').value.trim();
  const roi = document.getElementById('streamRoi').value.trim();
  const url = '/stream?source=' + encodeURIComponent(src) + (roi ? '&roi=' + encodeURIComponent(roi) : '');
  window.open(url, '_blank');
});

function uploadFile(file) {
  const form = new FormData();
  form.append('video', file);
  uploadBox.innerHTML = '<h2>업로드중: ' + file.name + '</h2><p>' + (file.size / 1024 / 1024).toFixed(1) + ' MB</p>';
  progressSection.classList.add('active');
  resultsSection.classList.remove('active');
  document.getElementById('statStatus').textContent = '업로드중...';

  fetch('/api/upload', { method: 'POST', body: form })
    .then(r => r.json())
    .then(data => {
      if (data.job_id) pollProgress(data.job_id);
      else alert('업로드 실패: ' + (data.error || '알 수 없는 오류'));
    })
    .catch(err => alert('업로드 오류: ' + err));
}

function pollProgress(jobId) {
  const interval = setInterval(() => {
    fetch('/api/status/' + jobId)
      .then(r => r.json())
      .then(data => {
        const pct = data.progress_pct || 0;
        document.getElementById('progressBar').style.width = pct + '%';
        document.getElementById('progressBar').textContent = pct + '%';
        document.getElementById('statStatus').textContent = data.message || data.status;
        document.getElementById('statFrames').textContent = (data.processed_frames || 0) + '/' + (data.total_frames || '?');
        document.getElementById('statPlates').textContent = data.plates_found || 0;
        document.getElementById('statRes').textContent = data.resolution || '-';

        if (data.status === 'completed') {
          clearInterval(interval);
          document.getElementById('progressBar').style.width = '100%';
          document.getElementById('progressBar').textContent = '100%';
          showResults(data, jobId);
        } else if (data.status === 'error') {
          clearInterval(interval);
          document.getElementById('statStatus').textContent = 'ERROR: ' + data.message;
        }
      });
  }, 1000);
}

function showResults(data, jobId) {
  resultsSection.classList.add('active');
  document.getElementById('jsonLink').href = '/api/results/' + jobId;

  // 확정 번호판
  const confirmed = data.confirmed || [];
  const clist = document.getElementById('confirmedList');
  clist.innerHTML = '';
  confirmed.forEach(c => {
    const card = document.createElement('div');
    card.className = 'plate-card';
    card.innerHTML = '<div class="plate-text">' + (c.text || '-') + '</div>'
      + '<div class="plate-conf">score: ' + (c.pattern_score || 0).toFixed(2) + '</div>'
      + '<div class="plate-frames">' + (c.detection_count || 0) + ' frames</div>';
    clist.appendChild(card);
  });

  // 전체 결과
  const plates = data.plates || [];
  document.getElementById('totalCount').textContent = plates.length;
  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = '';
  plates.forEach((p, i) => {
    const ocr = (p.ocr_confidence || 0);
    const confClass = ocr >= 0.7 ? 'conf-high' : (ocr >= 0.3 ? 'conf-mid' : 'conf-low');
    const method = p.detection_method || '-';
    const methodClass = method === 'plate_model' ? 'plate_model' : (method === 'log_ocr' ? 'log_ocr' : '');
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + i + '</td>'
      + '<td class="text-col">' + (p.text || '-') + '</td>'
      + '<td class="' + confClass + '">' + ocr.toFixed(3) + '</td>'
      + '<td>' + (p.det_confidence || 0).toFixed(3) + '</td>'
      + '<td>' + (p.pattern_score || 0).toFixed(2) + '</td>'
      + '<td><span class="method-tag ' + methodClass + '">' + method + '</span></td>';
    tbody.appendChild(tr);
  });
}

// 기존 결과 디렉토리 로드
fetch('/api/history')
  .then(r => r.json())
  .then(dirs => {
    if (dirs.length === 0) return;
    const sec = document.getElementById('historySection');
    let html = '<h2>기존 인식 결과</h2><div class="dir-links">';
    dirs.forEach(d => {
      html += '<a class="dir-link" href="/view/' + d.name + '">' + d.name + ' <span>(' + d.count + '건)</span></a>';
    });
    html += '</div>';
    sec.innerHTML = html;
  });

// 사이트 내 HTML 문서 (로드맵 등)
fetch('/html')
  .then(r => r.json())
  .then(files => {
    if (files.length === 0) return;
    const sec = document.getElementById('htmlSection');
    const labels = { roadmap: '로드맵', index: '메인' };
    let html = '<h2>문서</h2><div class="dir-links">';
    files.forEach(f => {
      const label = labels[f.name.replace('.html','')] || f.name;
      html += '<a class="dir-link" href="' + f.url + '">' + label + '</a>';
    });
    html += '</div>';
    sec.innerHTML = html;
  })
  .catch(() => {});
</script>
</body>
</html>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Flask 라우트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _realtime_response():
    return Response(
        _load_realtime_html(),
        mimetype="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


_REALTIME_HTML_FILE = Path(__file__).resolve().parent / "realtime_index.html"


@app.route("/")
def index():
    """localhost 메인 = 실시간 번호판 인식 GUI. realtime_index.html 있으면 파일에서 읽음."""
    if _REALTIME_HTML_FILE.is_file():
        try:
            html = _REALTIME_HTML_FILE.read_text(encoding="utf-8")
            return Response(
                html,
                mimetype="text/html; charset=utf-8",
                headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
            )
        except Exception:
            pass
    return _realtime_response()


@app.route("/live")
def live_page():
    """실시간 GUI (동일 내용)."""
    return _realtime_response()


@app.route("/upload")
def upload_page():
    """업로드/URL 배치 처리 페이지 (기존 메인 UI)."""
    return INDEX_HTML


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "video 파일이 없습니다"}), 400

    video = request.files["video"]
    if not video.filename:
        return jsonify({"error": "파일명이 없습니다"}), 400

    job_id = uuid.uuid4().hex[:8]
    ext = Path(video.filename).suffix or ".mp4"
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    video.save(save_path)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "message": "대기중...",
            "filename": video.filename,
            "video_path": save_path,
            "progress_pct": 0,
            "roi": request.form.get("roi", "").strip() if hasattr(request, "form") else "",
        }

    roi_norm = _parse_roi_norm(_jobs[job_id].get("roi"))
    thread = threading.Thread(target=process_video_job, args=(job_id, save_path, roi_norm), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/recognize_url", methods=["POST"])
def api_recognize_url():
    """URL(이미지 또는 영상)로 번호판 인식. JSON: {"url": "https://..."}"""
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url이 없습니다"}), 400
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "http 또는 https URL만 가능합니다"}), 400

    job_id = uuid.uuid4().hex[:8]
    parsed = urlparse(url)
    ext = Path(parsed.path or "").suffix or ".tmp"
    if ext.lower() not in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".mp4", ".avi", ".mkv", ".mov", ".webm"):
        ext = ".tmp"
    save_path = os.path.join(UPLOAD_DIR, f"url_{job_id}{ext}")

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "message": "URL 다운로드중...",
            "filename": url[:80] + ("..." if len(url) > 80 else ""),
            "progress_pct": 0,
            "roi": (data.get("roi") or "").strip(),
        }

    def _run():
        local_file = save_path
        try:
            lower = url.lower()
            if "youtube.com" in lower or "youtu.be" in lower:
                with _jobs_lock:
                    _jobs[job_id]["message"] = "YouTube 다운로드중... (yt-dlp)"
                local_file = _download_youtube(url)
            else:
                _download_url(url, save_path)
            with _jobs_lock:
                _jobs[job_id]["message"] = "다운로드 완료, 인식 시작..."
            process_url_job(job_id, url, local_file)
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["message"] = str(e)
            try:
                if local_file and os.path.isfile(local_file) and UPLOAD_DIR in os.path.abspath(local_file):
                    os.remove(local_file)
            except Exception:
                pass

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다"}), 404
    return Response(
        json.dumps(job, ensure_ascii=False, cls=NumpyEncoder),
        content_type="application/json; charset=utf-8",
    )


@app.route("/api/results/<job_id>")
def api_results(job_id):
    result_path = os.path.join(RESULTS_BASE, job_id, "results.json")
    if not os.path.isfile(result_path):
        return jsonify({"error": "결과 없음"}), 404
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    return Response(
        json.dumps(data, ensure_ascii=False, cls=NumpyEncoder),
        content_type="application/json; charset=utf-8",
    )


# ── 전체 영상 스캔 상태 ──
_scan_state: dict = {"status": "idle", "progress": 0, "total": 0, "results": [], "error": ""}
_scan_lock = threading.Lock()


_engine_use_lock = threading.Lock()   # 스트림/스캔 엔진 직렬 사용
_scan_active = False                  # 스캔 중 스트림 인식 일시정지


def _run_full_scan(video_path: str, step: int = 5):
    """
    백그라운드: 영상 전체를 process_frame()으로 스캔.
    스캔 시작 시 스트림 인식을 일시정지 → 엔진 독점 사용 → 스캔 종료 시 재개.
    """
    global _scan_state, _scan_active

    engine = _get_stream_engine()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with _scan_lock:
            _scan_state.update({"status": "error", "error": f"영상 열기 실패: {video_path}"})
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    with _scan_lock:
        _scan_state.update({"status": "running", "progress": 0, "total": total, "results": [], "error": ""})

    _scan_active = True
    time.sleep(0.3)

    old_consecutive = engine.consecutive_required
    engine.consecutive_required = 1

    seen: dict[str, float] = {}
    found: list[dict] = []
    frame_idx = 0

    def _td(s):
        d = "".join(c for c in s if c.isdigit())
        return d[-4:] if len(d) >= 4 else ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % step != 0:
                continue

            timestamp_sec = frame_idx / fps
            mins, secs = divmod(int(timestamp_sec), 60)
            cur_time_str = f"{mins:02d}:{secs:02d}"

            h0, w0 = frame.shape[:2]
            max_w = 640
            if w0 > max_w:
                scale = max_w / w0
                small = cv2.resize(frame, (max_w, int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
                small = frame

            sharp_k = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
            small_sharp = cv2.filter2D(small, -1, sharp_k)

            raw = []
            try:
                raw = engine.process_frame(small_sharp, camera_id="SCAN", full_frame=frame) or []
                if not raw:
                    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
                    l_ch, a_ch, b_ch = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    lab = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
                    small_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    raw = engine.process_frame(small_clahe, camera_id="SCAN", full_frame=frame) or []
                if not raw:
                    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype("float32")
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)
                    small_sat = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
                    raw = engine.process_frame(small_sat, camera_id="SCAN", full_frame=frame) or []
            except Exception as e:
                with _scan_lock:
                    _scan_state["error"] = f"process_frame: {e}"[:120]

            for r in (raw or []):
                text = (r.get("plate") or r.get("text") or "").strip()
                if not text:
                    continue
                td_new = _td(text)
                dup = next(
                    (k for k, t in seen.items()
                     if k == text or (_td(k) == td_new and td_new and abs(t - timestamp_sec) < 30)),
                    None
                )
                if dup is None:
                    seen[text] = timestamp_sec
                    conf_val = float(r.get("confidence", r.get("ocr_confidence", 0)))
                    found.append({
                        "time": cur_time_str,
                        "plate": text,
                        "conf": round(conf_val, 2),
                    })
                    with _scan_lock:
                        _scan_state["results"] = list(found)

            with _scan_lock:
                _scan_state["progress"] = frame_idx
                _scan_state["cur_time"] = cur_time_str

    except Exception as e:
        with _scan_lock:
            _scan_state["error"] = f"scan error: {e}"[:200]
    finally:
        cap.release()
        engine.consecutive_required = old_consecutive
        engine.recent_plates.clear()
        _scan_active = False
        with _scan_lock:
            if _scan_state["status"] == "running":
                _scan_state["status"] = "done"


@app.route("/api/full_scan", methods=["POST"])
def api_full_scan_start():
    """전체 영상 스캔 시작"""
    data = request.get_json(silent=True) or {}
    video_path = data.get("source") or _DEFAULT_VIDEO
    if not os.path.exists(str(video_path)):
        return jsonify({"error": f"파일 없음: {video_path}"}), 400
    with _scan_lock:
        # 이전 스캔이 "running"이어도 강제 재시작 허용 (stuck 방지)
        pass
    t = threading.Thread(target=_run_full_scan, args=(video_path,), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/full_scan_status")
def api_full_scan_status():
    """스캔 진행률 + 결과 반환"""
    with _scan_lock:
        state = dict(_scan_state)
    pct = int(state["progress"] / state["total"] * 100) if state.get("total") else 0
    return jsonify({
        "status":   state.get("status", "idle"),
        "progress": pct,
        "cur_time": state.get("cur_time", ""),
        "results":  state.get("results", []),
        "error":    state.get("error", ""),
    })


@app.route("/api/stream_log")
def api_stream_log():
    """실시간 스트림에서 인식된 번호판 로그 (Detection Log). 하드코딩 없이 엔진 결과만 반환."""
    with _stream_log_lock:
        current = dict(_stream_current_plate)
        log = list(_stream_detection_log[-50:])
    return jsonify({
        "current_plate": current.get("plate") or "",
        "current_time": current.get("time") or "",
        "current_img": current.get("img") or "",
        "log": log,
    })


@app.route("/api/history")
def api_history():
    dirs = find_all_result_dirs(os.path.dirname(__file__) or ".")
    result = []
    for d in dirs:
        rj = os.path.join(d, "results.json")
        count = 0
        if os.path.isfile(rj):
            try:
                with open(rj, encoding="utf-8") as f:
                    count = len(json.load(f))
            except Exception:
                pass
        result.append({"name": d, "count": count})
    return jsonify(result)


@app.route("/view/<path:dir_name>")
def view_results(dir_name):
    rj = os.path.join(dir_name, "results.json")
    if not os.path.isfile(rj):
        return "결과 없음", 404
    with open(rj, encoding="utf-8") as f:
        data = json.load(f)
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2, cls=NumpyEncoder),
        content_type="application/json; charset=utf-8",
    )


@app.route("/image/<path:filepath>")
def serve_image(filepath):
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    return send_from_directory(directory, filename)

# 실시간 MJPEG 스트림 (동영상 파일 경로, YouTube URL, 또는 웹캠 0)
@app.route("/stream")
def stream_source():
    source = request.args.get("source", "").strip()
    if not source:
        def _empty_source():
            for _ in _yield_error_stream(
                "Enter video file path or YouTube URL",
                "Example: C:\\video.mp4  or  https://www.youtube.com/watch?v=xxxx",
                30,
            ):
                yield _
        return Response(_empty_source(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store, no-cache", "X-Accel-Buffering": "no"})
    roi_norm = _parse_roi_norm(request.args.get("roi"))
    return Response(
        _mjpeg_stream_from_source(source, roi_norm),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache", "X-Accel-Buffering": "no"},
    )


# 실시간 GUI HTML — realtime_index.html 파일에서 로드 (없으면 내장 fallback)
def _load_realtime_html() -> str:
    p = Path(__file__).resolve().parent / "realtime_index.html"
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            pass
    return _REALTIME_HTML_FALLBACK

_REALTIME_HTML_FALLBACK = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>실시간 번호판 인식 | localhost 메인</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', -apple-system, sans-serif; background: #2b2b2b; color: #e0e0e0; min-height: 100vh; }
  .header { background: #1e1e1e; padding: 14px 20px; border-bottom: 1px solid #444; display: flex; align-items: center; gap: 16px; }
  .header h1 { font-size: 1.25rem; font-weight: 700; color: #fff; }
  .header .badge { background: #00c853; color: #000; padding: 4px 10px; border-radius: 6px; font-size: 0.7rem; font-weight: 700; }
  .back a { color: #64b5f6; text-decoration: none; font-size: 0.9rem; }
  .back a:hover { text-decoration: underline; }
  .main { display: flex; gap: 20px; padding: 20px; max-width: 1600px; margin: 0 auto; min-height: calc(100vh - 52px); }
  .left-panel { flex: 1; min-width: 0; }
  .right-panel { width: 380px; flex-shrink: 0; display: flex; flex-direction: column; gap: 16px; }
  .cam-label { background: #333; color: #aaa; padding: 8px 12px; font-size: 0.85rem; border-radius: 8px 8px 0 0; }
  .stream-wrap { background: #000; border-radius: 0 0 8px 8px; overflow: hidden; }
  .stream-wrap img { display: block; width: 100%; height: auto; max-height: 70vh; object-fit: contain; }
  #streamPlaceholder { padding: 80px 24px; text-align: center; color: #666; font-size: 0.95rem; background: #1a1a1a; border-radius: 0 0 8px 8px; }
  .current-plate-box { background: #00c853; color: #000; padding: 24px; border-radius: 12px; text-align: center; }
  .current-plate-box .label { font-size: 0.8rem; color: #333; margin-bottom: 8px; }
  .current-plate-box .plate { font-size: 2rem; font-weight: 800; letter-spacing: 2px; }
  .current-plate-box .time { font-size: 0.75rem; color: #444; margin-top: 8px; }
  .log-section { background: #1e1e1e; border-radius: 12px; overflow: hidden; border: 1px solid #444; flex: 1; min-height: 200px; }
  .log-section h3 { padding: 12px 16px; font-size: 0.95rem; border-bottom: 1px solid #444; }
  .log-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .log-table th { text-align: left; padding: 10px 12px; background: #333; color: #aaa; }
  .log-table td { padding: 8px 12px; border-bottom: 1px solid #333; }
  .log-table tr:hover { background: #2a2a2a; }
  .log-table .status-ok { color: #00c853; }
  .toolbar { margin-bottom: 12px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .toolbar input { padding: 10px 14px; border-radius: 8px; border: 1px solid #444; background: #333; color: #e0e0e0; width: min(100%, 420px); font-size: 0.9rem; }
  .toolbar input::placeholder { color: #666; }
  .toolbar button { padding: 10px 20px; border-radius: 8px; border: none; font-weight: 600; cursor: pointer; font-size: 0.9rem; }
  .toolbar .webcamBtn { background: #00c853; color: #000; }
  .toolbar .goBtn { background: #2196f3; color: #fff; }
  .toolbar .stopBtn { background: #555; color: #fff; }
</style>
</head>
<body>
<div class="header">
  <div class="back"><a href="/upload">&larr; 업로드/배치</a></div>
  <h1>실시간 번호판 인식</h1>
  <span class="badge">LIVE</span>
</div>
<div class="main">
  <div class="left-panel">
    <div class="toolbar">
      <input type="text" id="streamSrc" value="https://www.youtube.com/watch?v=-dlpsCgjCAg" placeholder="동영상 경로 또는 YouTube URL" />
      <button type="button" id="quickBtn" class="goBtn" style="background:#00c853; color:#000;">이 영상으로 바로 시작</button>
      <button type="button" id="goBtn" class="goBtn">동영상 번호판 인식 시작</button>
      <button type="button" id="stopBtn" class="stopBtn">중지</button>
      <button type="button" id="webcamBtn" class="webcamBtn" style="background:#555;">웹캠(0)</button>
    </div>
    <p style="color:#888; font-size:0.8rem; margin:0 0 10px 0;"><strong>바로 시작</strong> — [이 영상으로 바로 시작] 클릭 시 위 URL(-dlpsCgjCAg)로 실시간 인식 (720p)</p>
    <div class="cam-label">동영상 재생 + 번호판 인식</div>
    <div class="stream-wrap">
      <img id="streamImg" src="" alt="실시간 스트림" style="display:none;" />
      <div id="streamPlaceholder">
        <strong>이 영상으로 바로 시작</strong> 클릭 → https://www.youtube.com/watch?v=-dlpsCgjCAg 로 즉시 실시간 인식<br>
        영상 위에 번호판 박스·번호가 나오고, 오른쪽 Detection Log에 기록됩니다.
      </div>
    </div>
  </div>
  <div class="right-panel">
    <div class="current-plate-box">
      <div class="label">현재 인식 번호</div>
      <div class="plate" id="currentPlate">-</div>
      <div class="time" id="currentTime"></div>
    </div>
    <div class="log-section">
      <h3>Detection Log (인식 로그)</h3>
      <table class="log-table">
        <thead><tr><th>시간</th><th>번호판</th><th>상태</th></tr></thead>
        <tbody id="logBody"></tbody>
      </table>
    </div>
  </div>
</div>
<script>
(function(){
  var img = document.getElementById('streamImg');
  var ph = document.getElementById('streamPlaceholder');
  var srcInput = document.getElementById('streamSrc');
  var currentPlateEl = document.getElementById('currentPlate');
  var currentTimeEl = document.getElementById('currentTime');
  var logBody = document.getElementById('logBody');
  function pollLog(){
    fetch('/api/stream_log').then(function(r){ return r.json(); }).then(function(data){
      if (data.current_plate) { currentPlateEl.textContent = data.current_plate; currentTimeEl.textContent = data.current_time || ''; }
      var log = data.log || [];
      logBody.innerHTML = log.slice().reverse().slice(0, 30).map(function(row){
        return '<tr><td>' + (row.time || '') + '</td><td>' + (row.plate || '') + '</td><td class="status-ok">' + (row.status || '성공') + '</td></tr>';
      }).join('') || '<tr><td colspan="3" style="color:#666; padding:20px;">동영상 번호판 인식 시작 후 인식된 번호가 여기 표시됩니다.</td></tr>';
    }).catch(function(){});
  }
  function startStream(s){
    s = (s || '').toString().trim();
    if (!s) {
      alert('동영상 파일 경로 또는 YouTube URL을 입력하세요.\n예: C:\\video.mp4 또는 https://www.youtube.com/watch?v=...');
      return;
    }
    img.src = '/stream?source=' + encodeURIComponent(s);
    img.style.display = 'block';
    ph.style.display = 'none';
  }
  function stopStream(){
    img.src = ''; img.style.display = 'none'; ph.style.display = 'block';
  }
  document.getElementById('quickBtn').onclick = function(){ startStream('https://www.youtube.com/watch?v=-dlpsCgjCAg'); };
  document.getElementById('goBtn').onclick = function(){ startStream(srcInput.value); };
  document.getElementById('webcamBtn').onclick = function(){ startStream('0'); };
  document.getElementById('stopBtn').onclick = stopStream;
  setInterval(pollLog, 500);
})();
</script>
</body>
</html>
"""

REALTIME_HTML = _load_realtime_html()


@app.route("/realtime")
def realtime_page():
    return _realtime_response()

# 업로드 작업 기반 스트림 (/stream/<job_id>)
@app.route("/stream/<job_id>")
def stream_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return "작업을 찾을 수 없습니다", 404
    video_path = job.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        return "영상이 존재하지 않습니다 (URL 작업은 보관되지 않을 수 있음)", 404
    roi_norm = _parse_roi_norm(job.get("roi"))
    return Response(_mjpeg_stream_from_source(video_path, roi_norm),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# HTML 파일 제공 (프로젝트 루트의 .html 접근)
ROOT_DIR = Path(__file__).resolve().parent


@app.route("/html/<path:filename>")
def serve_html(filename):
    """프로젝트 루트 기준 .html 파일 제공. 예: /html/roadmap.html"""
    if ".." in filename or not filename.endswith(".html"):
        return "Not found", 404
    path = ROOT_DIR / filename
    if not path.is_file():
        return "Not found", 404
    return send_from_directory(str(path.parent), path.name)


@app.route("/html")
def html_index():
    """사이트 내 HTML 목록: 로드맵 등"""
    html_files = []
    for p in ROOT_DIR.glob("*.html"):
        name = p.name
        html_files.append({"name": name, "url": f"/html/{name}"})
    if not html_files:
        return jsonify([])
    return jsonify(html_files)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="번호판 인식 웹 서버")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-browser", action="store_true", help="브라우저 자동 열기 안 함")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"

    def _open_browser():
        time.sleep(1.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if not args.no_browser:
        threading.Thread(target=_open_browser, daemon=True).start()

    print()
    print("=" * 50)
    print("  번호판 인식 웹 서버")
    print("=" * 50)
    print(f"  주소: {url}")
    print(f"  엔진: plate_recognition_4k v6 (AI Hub)")
    print()
    print("  기능:")
    print("    - /  → 실시간 GUI (realtime_index.html 또는 내장)")
    print("    - /live, /realtime  → 실시간 번호판 인식")
    print("    - /upload  → 업로드/URL 배치 처리")
    print("    - 인식 결과 JSON API")
    print(f"    - 문서/로드맵: {url}/html/roadmap.html")
    if not args.no_browser:
        print("    - 브라우저 자동 열림 (--no-browser 로 비활성화)")
    print()
    print("  종료: Ctrl+C")
    print("=" * 50)
    print()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
