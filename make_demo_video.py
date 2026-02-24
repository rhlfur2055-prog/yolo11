"""
make_demo_video.py - YOLO26 번호판 인식 데모 영상 생성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

temp_youtube/ 안의 모든 mp4 파일을 YOLO 번호판 인식 후
오버레이를 입혀 demo_output/에 저장하고,
최종적으로 하나의 final_demo.mp4로 합치기.

사용법:
    python make_demo_video.py
"""

from __future__ import annotations

import os
import sys
import io
import glob
import json
import time
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "temp_youtube")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "demo_output")
FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
FONT_PATH_FB = "C:/Windows/Fonts/malgun.ttf"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PIL 폰트 캐시
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_fc: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int):
    if size not in _fc:
        for fp in [FONT_PATH, FONT_PATH_FB]:
            try:
                _fc[size] = ImageFont.truetype(fp, size)
                break
            except OSError:
                continue
        else:
            _fc[size] = ImageFont.load_default()
    return _fc[size]


def pil_text(frame: np.ndarray, text: str, pos: tuple, size: int = 20,
             fg=(255, 255, 255), bg=None, outline=True) -> np.ndarray:
    """BGR 프레임에 한글/영문 텍스트 오버레이."""
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _font(size)
    x, y = pos
    rgb = (fg[2], fg[1], fg[0])  # BGR→RGB

    # 배경 박스
    if bg is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 4
        bg_rgb = (bg[2], bg[1], bg[0])
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                        fill=bg_rgb)

    # 외곽선
    if outline:
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_sidebar(frame: np.ndarray, plates: list[dict], confirmed_set: set,
                 sidebar_w: int = 280) -> np.ndarray:
    """우측 반투명 사이드바에 인식된 번호판 목록 표시."""
    h, w = frame.shape[:2]
    # 반투명 오버레이
    overlay = frame.copy()
    x1 = w - sidebar_w
    cv2.rectangle(overlay, (x1, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # 타이틀
    frame = pil_text(frame, "Detected Plates", (x1 + 10, 10), 18,
                     fg=(200, 200, 200), outline=False)

    # 최신 10개
    recent = plates[-10:] if len(plates) > 10 else plates
    for i, p in enumerate(reversed(recent)):
        txt = p.get("text", "?")
        score = p.get("pattern_score", 0)
        count = p.get("count", 1)
        y_pos = 45 + i * 28

        is_confirmed = txt in confirmed_set
        if is_confirmed:
            color = (0, 255, 255)  # 노란색 (BGR: yellow)
            label = f"[OK] {txt} ({score:.0%}) x{count}"
        else:
            color = (200, 200, 200)
            label = f"     {txt} ({score:.0%}) x{count}"

        frame = pil_text(frame, label, (x1 + 10, y_pos), 15,
                         fg=color, outline=False)

    return frame


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단일 영상 처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_video(input_path: str, output_path: str, recognizer) -> dict:
    """단일 영상에 오버레이를 입혀 저장."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [오류] 영상 열기 실패: {input_path}")
        return {"error": True}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    basename = os.path.basename(input_path)
    print(f"\n  [{basename}] {w}x{h} @ {fps:.0f}fps, {total} frames")

    # 추적 상태
    plate_map: dict[str, dict] = {}  # text → {text, pattern_score, count, first_frame}
    confirmed_set: set[str] = set()
    total_det = 0
    best_frame = None
    best_det_count = 0
    frame_idx = 0
    start = time.time()
    recognizer._imgsz = None  # 전략 리셋

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 번호판 탐지
        try:
            results = recognizer.process_frame(frame, frame_idx)
        except Exception:
            results = []

        frame_dets = [r for r in results if r.get("text") and len(r["text"]) >= 3]

        # 추적
        for r in frame_dets:
            txt = r["text"]
            total_det += 1
            if txt in plate_map:
                plate_map[txt]["count"] += 1
                plate_map[txt]["pattern_score"] = max(
                    plate_map[txt]["pattern_score"], r.get("pattern_score", 0))
            else:
                plate_map[txt] = {
                    "text": txt,
                    "pattern_score": r.get("pattern_score", 0),
                    "count": 1,
                    "first_frame": frame_idx,
                }
            if plate_map[txt]["count"] >= 3:
                confirmed_set.add(txt)

        # 최고 프레임 기록
        if len(frame_dets) > best_det_count:
            best_det_count = len(frame_dets)
            best_frame = frame.copy()

        # 오버레이 그리기
        display = frame.copy()

        # 1. bbox + 텍스트
        for r in frame_dets:
            bbox = r.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            txt = r.get("text", "")
            score = r.get("pattern_score", 0)
            is_valid = r.get("is_valid_plate", False)

            color = (0, 230, 70) if is_valid else (0, 180, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)

            if txt:
                label = f"{txt} {score:.0%}"
                display = pil_text(display, label, (x1, max(0, y1 - 28)), 18,
                                   fg=(0, 0, 0), bg=(255, 255, 255), outline=False)

        # 2. 좌상단 HUD
        elapsed = time.time() - start
        current_fps = frame_idx / elapsed if elapsed > 0 else 0
        hud_lines = [
            "YOLO26 License Plate Recognition",
            f"FPS: {current_fps:.1f}  Det: {total_det}  Time: {elapsed:.0f}s",
        ]
        for i, line in enumerate(hud_lines):
            y_pos = 30 + i * 28
            sz = 22 if i == 0 else 16
            display = pil_text(display, line, (10, y_pos), sz,
                               fg=(0, 255, 0), outline=True)

        # 3. 우측 사이드바
        sorted_plates = sorted(plate_map.values(), key=lambda x: x["count"], reverse=True)
        display = draw_sidebar(display, sorted_plates, confirmed_set)

        writer.write(display)

        # 진행률
        if frame_idx % 60 == 0 or frame_idx == total:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"\r  [{pct:5.1f}%] {frame_idx}/{total}  det:{total_det}  "
                  f"confirmed:{len(confirmed_set)}", end="", flush=True)

    cap.release()
    writer.release()
    total_time = time.time() - start
    print(f"\n  완료: {total_det}건 인식, {len(confirmed_set)}건 확정, {total_time:.1f}초")

    return {
        "input": input_path,
        "output": output_path,
        "frames": frame_idx,
        "detections": total_det,
        "confirmed": len(confirmed_set),
        "time": total_time,
        "best_frame": best_frame,
        "best_det_count": best_det_count,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 영상 합치기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def concat_videos_ffmpeg(video_list: list[str], output_path: str) -> bool:
    """ffmpeg concat demuxer로 영상 합치기."""
    try:
        list_file = os.path.join(OUTPUT_DIR, "concat_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for v in video_list:
                f.write(f"file '{os.path.abspath(v)}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        os.remove(list_file)

        if result.returncode == 0:
            print(f"  [ffmpeg] final_demo.mp4 생성 완료")
            return True
        else:
            print(f"  [ffmpeg] 실패: {result.stderr[:200]}")
            return False
    except FileNotFoundError:
        print("  [ffmpeg] ffmpeg 미설치 → OpenCV 폴백")
        return False
    except Exception as e:
        print(f"  [ffmpeg] 오류: {e} → OpenCV 폴백")
        return False


def concat_videos_opencv(video_list: list[str], output_path: str) -> bool:
    """OpenCV로 영상 합치기 (ffmpeg 폴백)."""
    if not video_list:
        return False

    # 첫 영상 기준으로 해상도/FPS 설정
    cap0 = cv2.VideoCapture(video_list[0])
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for vp in video_list:
        cap = cv2.VideoCapture(vp)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 해상도 다르면 리사이즈
            if vw != w or vh != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        cap.release()

    writer.release()
    print(f"  [OpenCV] final_demo.mp4 생성 완료")
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 50)
    print("  YOLO26 Demo Video Generator")
    print("=" * 50)

    # 입력 파일
    mp4_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4")))
    if not mp4_files:
        print(f"  [오류] {INPUT_DIR} 에 mp4 파일 없음")
        return

    print(f"\n  입력 영상 {len(mp4_files)}개:")
    for f in mp4_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"    - {os.path.basename(f)} ({size_mb:.1f}MB)")

    # 출력 디렉토리
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # PlateRecognizer 초기화
    print("\n  모델 로딩 중...")
    from plate_recognition_4k import PlateRecognizer
    recognizer = PlateRecognizer(
        model_size="n",
        confidence_threshold=0.15,
        use_sahi=False,
        frame_skip=1,
        burst_frames=1,
    )
    print("  모델 로딩 완료.\n")

    # 각 영상 처리
    all_results = []
    output_files = []
    global_best_frame = None
    global_best_count = 0

    for mp4 in mp4_files:
        basename = os.path.splitext(os.path.basename(mp4))[0]
        out_path = os.path.join(OUTPUT_DIR, f"demo_{basename}.mp4")
        result = process_video(mp4, out_path, recognizer)
        all_results.append(result)
        if not result.get("error"):
            output_files.append(out_path)
            if result.get("best_det_count", 0) > global_best_count:
                global_best_count = result["best_det_count"]
                global_best_frame = result.get("best_frame")

    # 영상 합치기
    if len(output_files) > 1:
        print("\n  영상 합치기...")
        final_path = os.path.join(OUTPUT_DIR, "final_demo.mp4")
        ok = concat_videos_ffmpeg(output_files, final_path)
        if not ok:
            concat_videos_opencv(output_files, final_path)
    elif len(output_files) == 1:
        final_path = os.path.join(OUTPUT_DIR, "final_demo.mp4")
        shutil.copy2(output_files[0], final_path)
        print(f"\n  영상 1개 → final_demo.mp4 복사")

    # 썸네일 생성
    if global_best_frame is not None:
        thumb_path = os.path.join(OUTPUT_DIR, "thumbnail.jpg")
        cv2.imwrite(thumb_path, global_best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  썸네일 저장: thumbnail.jpg (최다 {global_best_count}건 인식 프레임)")

    # 결과 요약
    print(f"\n{'=' * 50}")
    print(f"  demo_output/ 파일 목록:")
    print(f"{'=' * 50}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"    {f:40s} {size_mb:8.1f} MB")

    total_det = sum(r.get("detections", 0) for r in all_results)
    total_conf = sum(r.get("confirmed", 0) for r in all_results)
    total_time = sum(r.get("time", 0) for r in all_results)
    print(f"\n  총 인식: {total_det}건, 확정: {total_conf}건, 처리시간: {total_time:.0f}초")
    print(f"\n  ✅ 데모 영상 완성")


if __name__ == "__main__":
    main()
