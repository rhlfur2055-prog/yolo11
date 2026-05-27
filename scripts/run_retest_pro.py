# -*- coding: utf-8 -*-
"""
STEP 3 리테스트: PlateEnginePro로 영상 처리 후 결과 요약 출력
- 사용 영상, 총 감지 프레임, 번호판 인식 성공, 오인식 필터 제거, 평균 confidence
사용법:
  python scripts/run_retest_pro.py "C:\path\to\video.mp4"
  python scripts/run_retest_pro.py   # scripts/list_videos.py 첫 100MB 이하 영상 사용
"""
import os
import sys
import io
from pathlib import Path

# Windows 콘솔 한글 깨짐 방지
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# 프로젝트 루트
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import cv2
from yolo11_plate.plate_engine_pro import PlateEnginePro


def get_video_size_mb(path: str) -> float:
    try:
        return path and os.path.getsize(path) / (1024 * 1024) or 0
    except OSError:
        return 0


def find_first_small_video():
    """100MB 이하 첫 .mp4/.avi 찾기 (제외 목록 적용)"""
    exclude = {"highway_kr.mp4", "d09a4761.mp4", "d819ca75.mp4"}
    for ext in ("*.mp4", "*.avi"):
        for f in ROOT.rglob(ext):
            if f.name in exclude:
                continue
            try:
                if f.stat().st_size <= 100 * 1024 * 1024:
                    return str(f)
            except OSError:
                continue
    return None


def main():
    video_path = None
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    if not video_path or not os.path.isfile(video_path):
        video_path = find_first_small_video()
    if not video_path:
        print("[에러] 100MB 이하 영상이 없습니다. 경로를 지정하세요: python scripts/run_retest_pro.py <video.mp4>")
        return

    size_mb = get_video_size_mb(video_path)
    print(f"[리테스트] PlateEnginePro 품질 개선 버전")
    print(f"  영상: {video_path}")
    print(f"  크기: {size_mb:.2f} MB")
    print()

    engine = PlateEnginePro()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[에러] 영상 열기 실패: {video_path}")
        return

    total_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        engine.process_frame(frame, "RETEST")
    cap.release()

    s = engine.stats
    filtered_total = (
        s["filtered_by_length"]
        + s["filtered_by_pattern"]
        + s["filtered_by_confidence"]
    )
    avg_conf = (
        sum(s["confidences"]) / len(s["confidences"])
        if s["confidences"]
        else 0
    )

    print()
    print("═══════════════════════════════════")
    print("결과 요약")
    print("═══════════════════════════════════")
    print(f"  사용 영상: {os.path.basename(video_path)} / {size_mb:.2f} MB")
    print(f"  총 감지 프레임: {s['frames_processed']}개")
    print(f"  번호판 인식 성공: {s['plates_shown']}개")
    print(f"  오인식 필터 제거: {filtered_total}개 (길이:{s['filtered_by_length']} 패턴:{s['filtered_by_pattern']} 신뢰도:{s['filtered_by_confidence']})")
    print(f"  평균 confidence: {avg_conf*100:.1f}%")
    print("═══════════════════════════════════")


if __name__ == "__main__":
    main()
