# -*- coding: utf-8 -*-
"""
가벼운 통합 검증 (OCR 없음, 30초 이내)
- 영상 1개 로드 확인
- YOLO 번호판 감지 5프레임만 (EasyOCR/PaddleOCR 미사용)
- API 서버 8765 포트 응답 확인
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

EXCLUDE = {"highway_kr.mp4", "d09a4761.mp4", "d819ca75.mp4"}
MAX_MB = 100
MAX_FRAMES = 5


def find_video():
    for ext in ("*.mp4", "*.avi"):
        for f in ROOT.rglob(ext):
            if f.name in EXCLUDE:
                continue
            try:
                if 100_000 < f.stat().st_size <= MAX_MB * 1024 * 1024:
                    return str(f)
            except OSError:
                continue
    return None


def main():
    print("=" * 50)
    print("통합 검증 (YOLO만, OCR 없음)")
    print("=" * 50)
    t_start = time.perf_counter()

    # 1) 영상 파일 1개 로드 확인
    video = find_video()
    if not video or not os.path.isfile(video):
        print("[FAIL] 영상 없음 (100MB 이하 .mp4/.avi)")
        return
    size_mb = os.path.getsize(video) / (1024 * 1024)
    print(f"\n[1] 영상 로드: {os.path.basename(video)} ({size_mb:.2f} MB)")

    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("[FAIL] 영상 열기 실패")
        return
    print("     -> 열기 OK")

    # 2) YOLO 번호판 감지 5프레임만 (OCR 로딩 없음)
    from ultralytics import YOLO
    model_path = ROOT / "yolo26.pt"
    if not model_path.exists():
        model_path = ROOT / "runs/plate_detect/yolo26_plate_v1/weights/best.pt"
    if not model_path.exists():
        model_path = "yolo11n.pt"
    print(f"\n[2] YOLO 감지 (5프레임): 모델={model_path.name}")
    t0 = time.perf_counter()
    model = YOLO(str(model_path))
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"     -> 모델 로드 {load_ms:.0f}ms")

    total_det = 0
    for i in range(MAX_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=0.45, verbose=False)
        n = len(results[0].boxes) if results and results[0].boxes is not None else 0
        total_det += n
    cap.release()
    det_ms = (time.perf_counter() - t0) * 1000
    print(f"     -> 5프레임 감지 완료: {total_det}개 박스, {det_ms:.0f}ms")

    # 3) API 서버 8765 응답 확인
    print("\n[3] API 서버 (8765)")
    api_ok = False
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8765/api/v1/health")
        with urllib.request.urlopen(req, timeout=2) as r:
            api_ok = r.status == 200
    except Exception as e:
        print(f"     -> 응답 없음 ({e})")
    if api_ok:
        print("     -> OK (동작 중)")
    else:
        print("     -> 중지됨 (python api_server.py --port 8765)")

    # 4) 결과 출력
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 50)
    print("결과")
    print("=" * 50)
    print(f"  영상: {os.path.basename(video)} / {size_mb:.2f} MB")
    print(f"  YOLO 5프레임 감지: {total_det}개")
    print(f"  API 8765: {'OK' if api_ok else 'FAIL'}")
    print(f"  총 소요: {elapsed:.1f}초")
    print("=" * 50)


if __name__ == "__main__":
    main()
