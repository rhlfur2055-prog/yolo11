"""
video_plate_recognizer.py - 범용 영상 번호판 인식기
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

어떤 영상 소스든 번호판을 인식하는 단일 프로그램.

지원 입력:
    - 로컬 파일 (MP4/AVI/MOV 등)
    - YouTube URL (yt-dlp 자동 다운로드)
    - 웹캠 (source=0)
    - IP카메라 RTSP URL

사용법:
    python video_plate_recognizer.py --source "video.mp4" --show
    python video_plate_recognizer.py --source "https://youtube.com/..." --show
    python video_plate_recognizer.py --source 0 --show
    python video_plate_recognizer.py --source "video.mp4" --no-show
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse
from typing import Optional

import cv2
import numpy as np

# plate_recognition_4k.py 재사용
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_recognition_4k import PlateRecognizer, NumpyEncoder

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESULTS_DIR = "plate_results_final"
CONFIRM_THRESHOLD = 3  # 3프레임 이상 연속 인식 → confirmed
DISPLAY_W = 1280
DISPLAY_H = 720


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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 영상 소스 열기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def open_source(source: str) -> tuple[cv2.VideoCapture, str]:
    """영상 소스를 열고 (cap, source_label)을 반환."""
    # 웹캠
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        return cap, f"Webcam:{source}"

    # RTSP
    if source.lower().startswith("rtsp://"):
        cap = cv2.VideoCapture(source)
        return cap, f"RTSP"

    # YouTube
    if is_youtube_url(source):
        local = download_youtube(source)
        cap = cv2.VideoCapture(local)
        return cap, f"YouTube"

    # 로컬 파일
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
    프레임 전처리:
    - 4K → 2K 리사이즈 (속도)
    - 평균 밝기 < 80 → 감마 보정
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
# 메인 인식 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(args: argparse.Namespace) -> None:
    # 소스 열기
    cap, source_label = open_source(args.source)
    if not cap.isOpened():
        print(f"[오류] 영상을 열 수 없습니다: {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_stream = total_frames <= 0  # 웹캠/RTSP는 총 프레임 0

    print(f"\n{'='*50}")
    print(f"  Video Plate Recognizer")
    print(f"{'='*50}")
    print(f"  소스: {source_label} ({args.source})")
    print(f"  해상도: {vid_w}x{vid_h} @ {fps:.1f}fps")
    if total_frames > 0:
        print(f"  프레임: {total_frames}")
    print(f"  화면 표시: {'ON' if args.show else 'OFF'}")
    print(f"{'='*50}\n")

    # PlateRecognizer 초기화 (plate_recognition_4k.py 재사용)
    print("  모델 로딩 중...")
    recognizer = PlateRecognizer(
        model_size="n",
        confidence_threshold=0.15,
        use_sahi=False,
        frame_skip=1,
        burst_frames=1,
    )
    print("  모델 로딩 완료.\n")

    # 결과 디렉토리
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, RESULTS_DIR)
    crops_dir = os.path.join(results_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # 추적 상태
    all_detections: list[dict] = []
    confirmed_list: list[dict] = []
    frame_idx = 0
    total_recognized = 0
    frame_skip = 1  # 자동 조정
    start_time = time.time()
    fps_samples: list[float] = []
    crop_count = 0

    print("  처리 시작...")
    if args.show:
        print("  [키] q=종료, s=스냅샷 저장\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_stream:
                    time.sleep(0.1)
                    continue
                break

            frame_idx += 1

            # 프레임 스킵
            if frame_idx % frame_skip != 0:
                continue

            t0 = time.time()

            # STEP 1: 프레임 전처리
            try:
                processed_frame = preprocess_frame(frame)
            except Exception:
                processed_frame = frame

            # STEP 2~4: YOLO 탐지 → OCR (PlateRecognizer에 위임)
            try:
                results = recognizer.process_frame(processed_frame, frame_idx)
            except Exception as e:
                print(f"  [오류] 프레임 {frame_idx}: {e}")
                results = []

            # 결과 수집
            for r in results:
                text = r.get("text", "")
                if text and len(text) >= 3:
                    total_recognized += 1
                    det_record = {k: v for k, v in r.items()
                                  if k not in ("plate_img", "preprocessed")}
                    all_detections.append(det_record)

                    # crop 저장
                    plate_img = r.get("plate_img")
                    if plate_img is not None:
                        try:
                            crop_path = os.path.join(crops_dir, f"plate_{crop_count:04d}.png")
                            cv2.imwrite(crop_path, plate_img)
                            crop_count += 1
                        except Exception:
                            pass

            # 처리 시간 측정
            elapsed = time.time() - t0
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_samples.append(current_fps)
            if len(fps_samples) > 30:
                fps_samples.pop(0)
            avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0

            # 프레임 스킵 자동 조정
            if avg_fps < 10 and frame_skip < 3:
                frame_skip = 2
            elif avg_fps >= 15 and frame_skip > 1:
                frame_skip = 1

            # 진행률 (파일인 경우)
            if total_frames > 0 and frame_idx % 30 == 0:
                pct = frame_idx / total_frames * 100
                confirmed = recognizer.get_confirmed_plates()
                print(
                    f"\r  [{pct:5.1f}%] F:{frame_idx}/{total_frames}  "
                    f"인식:{total_recognized}  확정:{len(confirmed)}  "
                    f"FPS:{avg_fps:.1f}  skip:{frame_skip}",
                    end="", flush=True,
                )

            # STEP: 화면 표시
            if args.show:
                display = frame.copy()

                # bbox + 텍스트 오버레이
                for r in results:
                    bbox = r.get("bbox", [])
                    if len(bbox) < 4:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    text = r.get("text", "")
                    score = r.get("pattern_score", 0)
                    is_valid = r.get("is_valid_plate", False)

                    color = (0, 230, 70) if is_valid else (0, 180, 255)
                    thickness = 3 if is_valid else 2
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

                    if text:
                        label = f"{text} ({score:.2f})"
                        try:
                            display = draw_text_pil(display, label,
                                                     (x1, max(0, y1 - 26)), 20, color)
                        except Exception:
                            cv2.putText(display, label, (x1, max(15, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 좌측 상단: 통계
                confirmed = recognizer.get_confirmed_plates()
                stats_lines = [
                    f"FPS: {avg_fps:.1f}",
                    f"Frame: {frame_idx}",
                    f"Recognized: {total_recognized}",
                    f"Confirmed: {len(confirmed)}",
                ]
                for i, line in enumerate(stats_lines):
                    cv2.putText(display, line, (10, 25 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 우측 사이드: 확정 번호판 목록
                confirmed = recognizer.get_confirmed_plates()
                for i, cp in enumerate(confirmed[-10:]):  # 최근 10개
                    cp_text = cp.get("text", "?")
                    y_pos = 25 + i * 22
                    cv2.putText(display, f"[OK] {cp_text}", (display.shape[1] - 250, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 리사이즈
                dh, dw = display.shape[:2]
                if dw > DISPLAY_W:
                    scale = DISPLAY_W / dw
                    display = cv2.resize(display, (DISPLAY_W, int(dh * scale)))

                cv2.imshow("Video Plate Recognizer", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  [종료] q 키 입력")
                    break
                elif key == ord('s'):
                    snap_path = os.path.join(results_dir, f"snapshot_{frame_idx:06d}.png")
                    cv2.imwrite(snap_path, frame)
                    print(f"\n  [스냅샷] {snap_path}")

    except KeyboardInterrupt:
        print("\n  [종료] Ctrl+C")

    # 정리
    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    confirmed = recognizer.get_confirmed_plates()

    print(f"\n\n{'='*50}")
    print(f"  처리 완료")
    print(f"{'='*50}")
    print(f"  처리 프레임: {frame_idx}")
    print(f"  총 인식: {total_recognized}")
    print(f"  확정 번호판: {len(confirmed)}")
    print(f"  처리 시간: {total_time:.1f}초")
    print(f"  평균 FPS: {frame_idx / total_time:.1f}" if total_time > 0 else "")

    # 결과 저장
    _save_results(results_dir, all_detections, confirmed,
                  frame_idx, total_recognized, total_time, source_label)

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
    parser = argparse.ArgumentParser(description="범용 영상 번호판 인식기")
    parser.add_argument("--source", required=True,
                        help="영상 소스 (파일경로, YouTube URL, 웹캠 번호, RTSP URL)")
    parser.add_argument("--show", action="store_true", default=False,
                        help="실시간 화면 표시")
    parser.add_argument("--no-show", action="store_true", default=False,
                        help="화면 없이 저장만")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLO 모델 크기 (기본: n)")
    parser.add_argument("--confidence", type=float, default=0.15,
                        help="탐지 임계값 (기본: 0.15)")
    args = parser.parse_args()

    # --no-show 가 명시되면 show=False
    if args.no_show:
        args.show = False

    run(args)


if __name__ == "__main__":
    main()
