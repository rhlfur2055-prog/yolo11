# -*- coding: utf-8 -*-
"""
plate_gui 헤드리스 테스트: GUI 없이 영상만 돌려서 번호판 감지 결과만 출력.
(백그라운드/타임아웃 제한 없이 로컬에서 실행용)

사용법:
  set PLATE_CONSECUTIVE_FRAMES=1
  python scripts/run_headless_plate_test.py
  python scripts/run_headless_plate_test.py "C:\path\to\vehicle_plate_test.mp4"
  python scripts/run_headless_plate_test.py --max-frames 100
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 이미지 슬라이드형 영상(1장=1프레임)이면 연속 1프레임만 있어도 표시
if "PLATE_CONSECUTIVE_FRAMES" not in os.environ:
    os.environ["PLATE_CONSECUTIVE_FRAMES"] = "1"


def find_test_video():
    for name in ("vehicle_plate_test.mp4", "plate_hd.mp4"):
        for d in [ROOT, ROOT / "dataset", ROOT / "temp_youtube"]:
            p = d / name
            if p.exists():
                return str(p)
    for ext in ("*.mp4", "*.avi"):
        for f in ROOT.rglob(ext):
            if f.stat().st_size < 200 * 1024 * 1024:  # 200MB 미만
                return str(f)
    return None


def main():
    video = None
    max_frames = None
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--max-frames" and i + 1 < len(sys.argv):
            max_frames = int(sys.argv[i + 1])
            break
        if not a.startswith("-") and (a.endswith(".mp4") or a.endswith(".avi")):
            video = a
            break
    if not video:
        video = find_test_video()
    if not video or not os.path.isfile(video):
        print("사용법: python scripts/run_headless_plate_test.py [영상경로] [--max-frames N]")
        print("영상이 없습니다. 경로를 지정하거나 vehicle_plate_test.mp4 를 프로젝트에 두세요.")
        return

    import cv2
    from plate_engine_pro import PlateEnginePro

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"영상 열기 실패: {video}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    limit = max_frames if max_frames else total
    print("=" * 50)
    print("헤드리스 번호판 테스트")
    print("=" * 50)
    print(f"영상: {video}")
    print(f"해상도: {w}x{h}, {total}프레임 (처리: 최대 {limit}프레임)")
    print(f"PLATE_CONSECUTIVE_FRAMES={os.environ.get('PLATE_CONSECUTIVE_FRAMES', '1')}")
    print()

    engine = PlateEnginePro()
    cap = cv2.VideoCapture(video)
    n = 0
    plates_seen = []
    t0 = time.perf_counter()
    while n < limit:
        ret, frame = cap.read()
        if not ret:
            break
        results = engine.process_frame(frame, "HEADLESS", use_multiframe=False)
        for r in results:
            p = r.get("plate", "")
            if p and p not in plates_seen:
                plates_seen.append(p)
        n += 1
        if n % 50 == 0:
            print(f"  {n}/{limit} 프레임, 인식 {len(plates_seen)}건...", flush=True)
    cap.release()
    elapsed = time.perf_counter() - t0

    print()
    print("결과")
    print("-" * 50)
    print(f"처리 프레임: {n}")
    print(f"번호판 인식 건수: {engine.stats['plates_shown']}")
    print(f"고유 번호판 수: {len(plates_seen)}")
    if plates_seen:
        for i, p in enumerate(plates_seen[:30]):
            print(f"  {i+1}. {p}")
        if len(plates_seen) > 30:
            print(f"  ... 외 {len(plates_seen)-30}건")
    print(f"소요 시간: {elapsed:.1f}초, FPS: {n/elapsed:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
