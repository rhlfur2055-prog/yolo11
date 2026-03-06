# -*- coding: utf-8 -*-
"""
crop_cctv_video.py — CCTV 소프트웨어 UI를 제거하고 영상 부분만 크롭
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용법:
    python crop_cctv_video.py input.mp4                    # 자동 크롭
    python crop_cctv_video.py input.mp4 --preview          # 크롭 영역 미리보기만
    python crop_cctv_video.py input.mp4 --x1 10 --y1 30 --x2 900 --y2 680  # 수동 지정

YouTube에서 다운로드 후 사용:
    python youtube_helper.py "https://youtube.com/watch?v=..."
    python crop_cctv_video.py temp_youtube/영상.mp4
    python plate_gui.py movie/cropped_cctv.mp4
"""

import os
import sys
import argparse
import cv2
import numpy as np


def detect_video_region(frame: np.ndarray) -> tuple[int, int, int, int]:
    """CCTV 소프트웨어 UI에서 실제 영상 영역을 자동 감지.

    SARADA STEELHEAD 스타일 레이아웃:
    - 좌측: 메인 CCTV 영상 (약 70%)
    - 우측: 감지 카드 패널
    - 상단: 메뉴바
    - 하단: 로그 패널

    Returns:
        (x1, y1, x2, y2) 크롭 좌표
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 방법 1: 엣지 기반으로 UI 패널 경계 찾기
    edges = cv2.Canny(gray, 50, 150)

    # 세로 방향 엣지 히스토그램 → 우측 패널 경계 감지
    v_hist = np.sum(edges, axis=0)

    # 우측 30% 영역에서 가장 강한 세로선 찾기
    right_start = int(w * 0.60)
    right_region = v_hist[right_start:]
    if len(right_region) > 0 and np.max(right_region) > 0:
        # 임계값 이상인 첫 번째 위치
        threshold = np.max(right_region) * 0.5
        candidates = np.where(right_region > threshold)[0]
        if len(candidates) > 0:
            x2 = right_start + candidates[0] - 5  # 약간 여유
        else:
            x2 = int(w * 0.70)
    else:
        x2 = int(w * 0.70)

    # 가로 방향 엣지 히스토그램 → 상단 메뉴바 감지
    h_hist = np.sum(edges, axis=1)

    # 상단 10%에서 메뉴바 하단 경계 찾기
    top_region = h_hist[:int(h * 0.10)]
    if len(top_region) > 0 and np.max(top_region) > 0:
        threshold = np.max(top_region) * 0.3
        candidates = np.where(top_region > threshold)[0]
        if len(candidates) > 0:
            y1 = candidates[-1] + 2
        else:
            y1 = 0
    else:
        y1 = 0

    # 하단: 로그 패널 또는 진행바 위치
    bottom_region = h_hist[int(h * 0.85):]
    if len(bottom_region) > 0 and np.max(bottom_region) > 0:
        threshold = np.max(bottom_region) * 0.3
        candidates = np.where(bottom_region > threshold)[0]
        if len(candidates) > 0:
            y2 = int(h * 0.85) + candidates[0] - 5
        else:
            y2 = h
    else:
        y2 = h

    x1 = 0

    # 안전 범위 보정
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # 최소 크기 보장
    if (x2 - x1) < w * 0.50 or (y2 - y1) < h * 0.50:
        print(f"[경고] 자동 감지 영역이 너무 작음 ({x2-x1}x{y2-y1}). 기본값 사용.")
        x1, y1 = 0, 30
        x2, y2 = int(w * 0.70), int(h * 0.92)

    return x1, y1, x2, y2


def preview_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """크롭 영역을 미리보기로 표시."""
    display = frame.copy()
    h, w = display.shape[:2]

    # 크롭 영역 외부를 어둡게
    mask = np.zeros_like(display)
    mask[y1:y2, x1:x2] = display[y1:y2, x1:x2]
    dark = cv2.addWeighted(display, 0.3, mask, 0.7, 0)

    # 크롭 영역 내부는 원본
    dark[y1:y2, x1:x2] = display[y1:y2, x1:x2]

    # 초록 사각형으로 크롭 영역 표시
    cv2.rectangle(dark, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 크기 정보 표시
    crop_w, crop_h = x2 - x1, y2 - y1
    info = f"Crop: ({x1},{y1})-({x2},{y2}) = {crop_w}x{crop_h}"
    cv2.putText(dark, info, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 미리보기 표시
    scale = min(1280 / w, 720 / h, 1.0)
    if scale < 1.0:
        dark = cv2.resize(dark, (int(w * scale), int(h * scale)))

    cv2.imshow("Crop Preview (Press any key to continue, ESC to cancel)", dark)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == 27:  # ESC
        print("[취소] 사용자가 크롭을 취소했습니다.")
        sys.exit(0)


def crop_video(input_path: str, output_path: str,
               x1: int, y1: int, x2: int, y2: int) -> str:
    """영상을 지정 영역으로 크롭하여 저장."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    crop_w = x2 - x1
    crop_h = y2 - y1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"출력 파일을 생성할 수 없습니다: {output_path}")

    print(f"[크롭] 입력: {input_path}")
    print(f"[크롭] 출력: {output_path}")
    print(f"[크롭] 영역: ({x1},{y1})-({x2},{y2}) = {crop_w}x{crop_h}")
    print(f"[크롭] FPS: {fps:.1f}, 총 프레임: {total_frames}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 크롭
        cropped = frame[y1:y2, x1:x2]
        writer.write(cropped)

        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"\r[크롭] {frame_idx}/{total_frames} ({pct:.0f}%)", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\n[완료] 크롭 영상 저장: {output_path} ({crop_w}x{crop_h}, {frame_idx}프레임)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="CCTV 영상 UI 제거 크롭")
    parser.add_argument("input", help="입력 영상 경로")
    parser.add_argument("-o", "--output", help="출력 경로 (기본: movie/cropped_cctv.mp4)")
    parser.add_argument("--preview", action="store_true", help="크롭 영역 미리보기")
    parser.add_argument("--x1", type=int, default=None, help="좌측 X 좌표")
    parser.add_argument("--y1", type=int, default=None, help="상단 Y 좌표")
    parser.add_argument("--x2", type=int, default=None, help="우측 X 좌표")
    parser.add_argument("--y2", type=int, default=None, help="하단 Y 좌표")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[오류] 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    # 첫 프레임 읽기
    cap = cv2.VideoCapture(args.input)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print(f"[오류] 영상을 읽을 수 없습니다: {args.input}")
        sys.exit(1)

    h, w = first_frame.shape[:2]
    print(f"[정보] 입력 영상: {w}x{h}")

    # 크롭 좌표 결정
    if all(v is not None for v in [args.x1, args.y1, args.x2, args.y2]):
        x1, y1, x2, y2 = args.x1, args.y1, args.x2, args.y2
        print(f"[정보] 수동 크롭: ({x1},{y1})-({x2},{y2})")
    else:
        x1, y1, x2, y2 = detect_video_region(first_frame)
        print(f"[정보] 자동 감지 크롭: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}")

    # 미리보기
    if args.preview:
        preview_crop(first_frame, x1, y1, x2, y2)

    # 출력 경로
    if args.output:
        output_path = args.output
    else:
        movie_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movie")
        os.makedirs(movie_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(movie_dir, f"{base}_cropped.mp4")

    # 크롭 실행
    crop_video(args.input, output_path, x1, y1, x2, y2)

    print(f"\n[사용법] 번호판 인식 실행:")
    print(f"  python plate_gui.py \"{output_path}\"")


if __name__ == "__main__":
    main()
