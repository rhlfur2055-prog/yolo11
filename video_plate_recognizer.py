"""
video_plate_recognizer.py - 비동기 실시간 관제 스타일 번호판 인식기 (리팩토링)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

본 파일은 기존 단일 루프 방식을 전면 개조하여 "메인 스레드=재생/표시 전담, 백그라운드 스레드=딥러닝 분석" 구조를 적용한다.
핵심 목표는 재생 끊김 없는 30fps 유지이며, 분석 지연이 있어도 최신 프레임에 즉시 결과를 반영한다.

[스레드 구성]
- Producer(Main): 프레임 읽기 + 화면 표시. 분석 큐가 비었을 때만 최신 프레임을 넣고, 밀린 프레임은 드롭한다.
- Consumer(Worker): 큐에서 받은 프레임을 PlateEnginePro 또는 PlateRecognizer로 분석하고 결과를 최신값으로 갱신한다.

[UI/UX]
- 화면 우측 패널에 최근 인식된 번호판과 시간 로그를 누적 표시
- 인식된 bbox와 텍스트는 TTL(유지 시간) 동안만 화면에 표시

[사용 예]
  python video_plate_recognizer.py --source "video.mp4" --engine yolo --roi "0.28,0.12 0.72,0.12 0.82,0.98 0.18,0.98"
  python video_plate_recognizer.py --source "https://www.youtube.com/watch?v=-dlpsCgjCAg" --engine pro
  python video_plate_recognizer.py --source 0
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse
from typing import Optional, List, Tuple
import threading
import queue
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_recognition_4k import PlateRecognizer, NumpyEncoder

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수/설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESULTS_DIR = "plate_results_final"
DISPLAY_W = 1280
DISPLAY_H = 720
TTL_SECONDS = 2.5  # bbox/텍스트 표시 유지 시간
LOG_MAX = 20       # 우측 패널 로그 최대 표시 개수


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YouTube 다운로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def is_youtube_url(source: str) -> bool:
    return any(domain in source for domain in [
        "youtube.com", "youtu.be", "youtube.co", "yt.be"
    ])


def download_youtube(url: str) -> str:
    """yt-dlp로 YouTube 영상 다운로드, 로컬 경로 반환."""
    try:
        import yt_dlp
    except ImportError:
        print("[오류] yt-dlp 미설치: pip install yt-dlp")
        sys.exit(1)

    dl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_youtube")
    os.makedirs(dl_dir, exist_ok=True)

    opts = {
        "format": "bestvideo[ext=mp4][height<=2160]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(dl_dir, "%(title).50s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # mp4 확장자 보장
        if not filename.endswith(".mp4"):
            base = os.path.splitext(filename)[0]
            if os.path.isfile(base + ".mp4"):
                filename = base + ".mp4"
    print(f"  [YouTube] 다운로드 완료: {filename}")
    return filename


def open_source(source: str) -> tuple[cv2.VideoCapture, str]:
    """영상 소스를 열고 (cap, source_label)을 반환."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        return cap, f"Webcam:{source}"
    if source.lower().startswith("rtsp://"):
        cap = cv2.VideoCapture(source)
        return cap, f"RTSP"
    if is_youtube_url(source):
        local = download_youtube(source)
        cap = cv2.VideoCapture(local)
        return cap, f"YouTube"
    if not os.path.isfile(source):
        print(f"[오류] 파일을 찾을 수 없습니다: {source}")
        sys.exit(1)
    cap = cv2.VideoCapture(source)
    return cap, os.path.basename(source)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 프레임 전처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    프레임 전처리: 4K 축소/감마 보정
    """
    h, w = frame.shape[:2]

    # 4K 이상이면 2K로 축소
    if w > 2560:
        scale = 1920 / w
        frame = cv2.resize(frame, (1920, int(h * scale)), interpolation=cv2.INTER_AREA)

    # 밝기 자동 보정
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness < 80:
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([
            min(255, int((i / 255.0) ** inv_gamma * 255))
            for i in range(256)
        ], dtype=np.uint8)
        frame = cv2.LUT(frame, table)

    return frame


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한글 텍스트 오버레이 (PIL 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_font_cache = {}

def _get_pil_font(size: int):
    if size not in _font_cache:
        from PIL import ImageFont
        for fp in ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf"]:
            try:
                _font_cache[size] = ImageFont.truetype(fp, size)
                break
            except OSError:
                continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_text_pil(frame, text, pos, font_size=20, color=(0, 255, 0)):
    """BGR 프레임에 한글/영문 텍스트 오버레이."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _get_pil_font(font_size)
    x, y = pos
    rgb = (color[2], color[1], color[0])
    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))
    draw.text((x, y), text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROI 파서
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_roi_norm(s: str | None) -> Optional[List[Tuple[float, float]]]:
    """
    "x1,y1 x2,y2 x3,y3 ..." 형식 또는 JSON 배열 [[x,y],...] 형식의 정규화 좌표 파싱.
    """
    if not s:
        return None
    s = s.strip()
    try:
        if s.startswith("["):
            import json as _json
            arr = _json.loads(s)
            return [(float(x), float(y)) for x, y in arr]
        pts = []
        for p in s.replace(";", " ").split():
            x, y = p.split(",")
            pts.append((float(x), float(y)))
        return pts
    except Exception:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 비동기 구성요소
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectorWorker(threading.Thread):
    """
    백그라운드 Consumer.
    - 큐에서 최신 프레임을 하나만 받아 분석
    - 분석 지연이 있어도 메인 스레드는 영향을 받지 않음
    """
    def __init__(self, input_q: queue.Queue, engine: str = "yolo",
                 roi: Optional[List[Tuple[float,float]]] = None):
        super().__init__(daemon=True)
        self.input_q = input_q
        self.engine = engine
        self.roi = roi
        self.stop_flag = threading.Event()
        self.latest_results: list[dict] = []
        self.last_time = time.time()
        self._init_engine()

    def _init_engine(self):
        if self.engine == "pro":
            try:
                from plate_engine_pro import PlateEnginePro
                self.pro = PlateEnginePro()
                self.recognizer = None
            except Exception as e:
                print(f"[경고] PlateEnginePro 로드 실패 → YOLO 경로로 전환: {e}")
                self.pro = None
                self.recognizer = PlateRecognizer(model_size="n", use_sahi=False, frame_skip=1, burst_frames=1)
        else:
            self.pro = None
            self.recognizer = PlateRecognizer(model_size="n", use_sahi=False, frame_skip=1, burst_frames=1)
        if self.recognizer and self.roi:
            try:
                self.recognizer.set_roi_polygon(self.roi)
            except Exception:
                pass

    def run(self):
        while not self.stop_flag.is_set():
            try:
                item = self.input_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                continue
            frame_idx, frame = item
            try:
                if self.pro is not None:
                    results = self.pro.process_frame(frame, "ASYNC", use_multiframe=False)
                    # pro 결과 → 공통 포맷으로 정규화
                    norm = []
                    for r in results:
                        norm.append({
                            "text": r.get("plate", ""),
                            "bbox": r.get("bbox", []),
                            "pattern_score": r.get("confidence", 0.0),
                            "is_valid_plate": True,
                            "detection_method": "pro",
                            "ocr_confidence": r.get("confidence", 0.0),
                            "detection_confidence": r.get("confidence", 0.0),
                        })
                    self.latest_results = norm
                else:
                    results = self.recognizer.process_frame(frame, int(frame_idx))
                    self.latest_results = results
                self.last_time = time.time()
            except Exception as e:
                print(f"[Worker오류] {e}")
                self.latest_results = []

    def stop(self):
        self.stop_flag.set()


class TTLResultCache:
    """
    분석 결과를 TTL 동안 유지하는 간단한 캐시.
    - 같은 프레임에서 여러 bbox를 보관하고 시간이 지나면 삭제
    """
    def __init__(self, ttl_seconds: float):
        self.ttl = ttl_seconds
        self.items: list[tuple[float, dict]] = []  # (timestamp, result)

    def update_with(self, results: list[dict]):
        ts = time.time()
        for r in results:
            self.items.append((ts, r))
        self._gc(ts)

    def snapshot(self) -> list[dict]:
        ts = time.time()
        self._gc(ts)
        return [r for t, r in self.items]

    def _gc(self, now_ts: float):
        self.items = [(t, r) for (t, r) in self.items if now_ts - t <= self.ttl]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 루프 (Producer/Renderer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(args: argparse.Namespace) -> None:
    cap, source_label = open_source(args.source)
    if not cap.isOpened():
        print(f"[오류] 영상을 열 수 없습니다: {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_stream = total_frames <= 0

    roi = parse_roi_norm(args.roi)

    # 큐는 최대 1 → 분석 스레드가 바쁠 땐 최신 프레임만 유지하고 나머지는 드롭
    q: queue.Queue = queue.Queue(maxsize=1)
    worker = DetectorWorker(q, engine=args.engine, roi=roi)
    worker.start()

    ttl_cache = TTLResultCache(TTL_SECONDS)
    recent_log: list[tuple[str, str]] = []  # (plate, HH:MM:SS)
    seen_set: set[str] = set()

    start = time.time()
    frame_idx = 0

    print(f"\n{'='*50}")
    print(f"  Async Video Plate Recognizer")
    print(f"{'='*50}")
    print(f"  소스: {source_label} ({args.source})")
    print(f"  해상도: {vid_w}x{vid_h} @ {fps:.1f}fps")
    print(f"  엔진: {args.engine}")
    if roi:
        print(f"  ROI: {roi}")
    print(f"{'='*50}\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_stream:
                    time.sleep(0.01)
                    continue
                break
            frame_idx += 1

            # Producer: 큐가 비었을 때만 최신 프레임 투입, 아니면 드롭
            if q.empty():
                try:
                    q.put_nowait((frame_idx, preprocess_frame(frame)))
                except queue.Full:
                    pass

            # JSON Lines 모드: 워커 최신 결과를 즉시 표준출력으로 배출
            if args.json_lines:
                results = worker.latest_results or []
                for r in results:
                    text = r.get("text", "")
                    if not text:
                        continue
                    bbox = r.get("bbox", []) or r.get("xyxy", [])
                    conf = float(r.get("ocr_confidence", r.get("pattern_score", r.get("detection_confidence", 0.0))))
                    obj = {
                        "plate_number": text,
                        "confidence": round(conf, 4),
                        "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if len(bbox) >= 4 else [0,0,0,0],
                    }
                    try:
                        print(json.dumps(obj, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
                continue  # GUI 없이 다음 프레임로op

            # GUI 모드: Worker 최신 결과를 TTL 캐시에 반영
            ttl_cache.update_with(worker.latest_results)

            # 오버레이용 결과 스냅샷
            overlay_results = ttl_cache.snapshot()
            display = frame.copy()

            # ROI 폴리라인 표시
            if roi:
                pts = np.array([(int(x * vid_w), int(y * vid_h)) for x, y in roi], dtype=np.int32)
                cv2.polylines(display, [pts], True, (255, 140, 0), 2)

            # bbox/텍스트 표시 + 로그 갱신
            for r in overlay_results:
                bbox = r.get("bbox", [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    color = (0, 230, 70) if r.get("is_valid_plate", False) else (0, 180, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                text = r.get("text", "")
                if text:
                    ts_str = datetime.now().strftime("%H:%M:%S")
                    try:
                        display = draw_text_pil(display, text, (x1, max(0, y1 - 26)), 22, (0, 230, 70))
                    except Exception:
                        cv2.putText(display, text, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 70), 2)
                    if text not in seen_set:
                        seen_set.add(text)
                        recent_log.append((text, ts_str))
                        if len(recent_log) > LOG_MAX:
                            recent_log.pop(0)

            # 우측 패널 로그
            panel_w = 280
            ph = display.shape[0]
            cv2.rectangle(display, (display.shape[1] - panel_w, 0), (display.shape[1], ph), (16, 18, 24), -1)
            cv2.putText(display, "PLATE LOG", (display.shape[1] - panel_w + 16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
            y = 58
            for plate, tstr in reversed(recent_log[-12:]):
                cv2.putText(display, f"{tstr}  {plate}", (display.shape[1] - panel_w + 12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                y += 22

            # 리사이즈 후 표시
            dh, dw = display.shape[:2]
            if dw > DISPLAY_W:
                scale = DISPLAY_W / dw
                display = cv2.resize(display, (DISPLAY_W, int(dh * scale)))
            cv2.imshow("Async Plate Monitor", display)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                results_dir = os.path.join(script_dir, RESULTS_DIR)
                os.makedirs(results_dir, exist_ok=True)
                cv2.imwrite(os.path.join(results_dir, f"snapshot_{frame_idx:06d}.png"), frame)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n  결과 저장: {results_dir}/")
    print(f"{'='*50}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 결과 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _save_results(
    results_dir: str,
    all_detections: list[dict],
    confirmed: list[dict],
    total_frames: int,
    total_recognized: int,
    total_time: float,
    source_label: str,
) -> None:
    """결과를 JSON + summary.txt로 저장."""
    os.makedirs(results_dir, exist_ok=True)

    # confirmed_plates.json
    confirmed_path = os.path.join(results_dir, "confirmed_plates.json")
    with open(confirmed_path, "w", encoding="utf-8") as f:
        json.dump(confirmed, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    # all_detections.json
    all_path = os.path.join(results_dir, "all_detections.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    # summary.txt
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Video Plate Recognizer - 처리 결과 요약\n")
        f.write(f"{'='*40}\n")
        f.write(f"소스: {source_label}\n")
        f.write(f"총 처리 프레임: {total_frames}\n")
        f.write(f"총 인식 건수: {total_recognized}\n")
        f.write(f"확정 번호판: {len(confirmed)}건\n")
        f.write(f"처리 시간: {total_time:.1f}초\n")
        avg_fps = total_frames / total_time if total_time > 0 else 0
        f.write(f"평균 FPS: {avg_fps:.1f}\n")
        f.write(f"\n확정 번호판 목록:\n")
        for i, cp in enumerate(confirmed, 1):
            text = cp.get("text", "?")
            score = cp.get("pattern_score", 0)
            first = cp.get("first_frame", "?")
            last = cp.get("last_frame", "?")
            count = cp.get("detection_count", 0)
            f.write(f"  {i}. {text} (score={score:.2f}, frames={first}-{last}, count={count})\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 엔트리포인트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="비동기 실시간 관제 번호판 인식기")
    parser.add_argument("--source", required=True,
                        help="영상 소스 (파일경로, YouTube URL, 웹캠 번호, RTSP URL)")
    parser.add_argument("--engine", choices=["yolo", "pro"], default="yolo",
                        help="분석 엔진 선택: yolo(PlateRecognizer) | pro(PlateEnginePro)")
    parser.add_argument("--roi", default="", help="정규화 ROI 다각형: 'x1,y1 x2,y2 x3,y3 ...'")
    parser.add_argument("--json-lines", action="store_true",
                        help="GUI 없이 탐지 결과를 JSON Lines로 표준출력")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
