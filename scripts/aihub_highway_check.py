# -*- coding: utf-8 -*-
"""
AI Hub "교통문제 해결을 위한 CCTV 교통 영상" 검증 스크립트

사용법:
  1. AI Hub 웹에서 "교통문제 해결을 위한 CCTV 교통 영상" 데이터셋 접속
  2. 파일 목록에서 각 파일 크기 확인 → 100MB 초과는 절대 다운로드 금지
  3. 100MB 이하 파일만 다운로드 후 아래 폴더에 저장
  4. python scripts/aihub_highway_check.py [폴더경로]

폴더 기본값: C:/tools/yolo26/dataset/aihub_traffic
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

MAX_MB = 100
DEFAULT_DIR = ROOT / "dataset" / "aihub_traffic"


def get_video_files(folder):
    """폴더 내 .mp4, .avi 목록 + 크기(MB) 반환"""
    if not folder or not os.path.isdir(folder):
        return []
    out = []
    for ext in ("*.mp4", "*.avi"):
        for f in Path(folder).rglob(ext):
            try:
                mb = f.stat().st_size / (1024 * 1024)
                out.append((str(f), f.name, mb))
            except OSError:
                continue
    return out


def analyze_first_frame(video_path):
    """첫 프레임 캡처 후 해상도/각도/차선수/번호판 분석"""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return {"resolution": (w, h), "angle": "미확인", "lanes": 0, "plate": "미확인"}

    # 각도 추정: 부감이면 하단에 도로가 넓게 보이고 상단에 수평선(수평선 상단 1/3 이내)
    # 하단 절반 평균 밝기 vs 상단 절반 → 하단이 더 넓고 도로색이면 부감 가능
    hh, ww = frame.shape[:2]
    lower = frame[int(hh * 0.5) :, :]
    upper = frame[: int(hh * 0.5), :]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 수평선 근처 에지: 부감이면 상단 1/4~1/3에 수평선
    edge_row = np.mean(gray, axis=1)
    horizon_peak = np.argmax(np.gradient(edge_row)[: int(hh * 0.4)]) if hh > 0 else 0
    # 상단 30% 안에 peak 있으면 부감 가능
    angle = "부감" if horizon_peak < hh * 0.35 else ("정면" if horizon_peak < hh * 0.6 else "측면")

    # 차선: HoughLinesP로 수평에 가까운 선 개수 추정
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=ww // 8, maxLineGap=20)
    lane_count = 0
    if lines is not None:
        ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle_deg) < 15 or abs(angle_deg - 180) < 15:  # 수평에 가까운 선
                ys.append((y1 + y2) // 2)
        if ys:
            ys.sort()
            prev = -999
            for y in ys:
                if y - prev > 30:
                    lane_count += 1
                    prev = y
        lane_count = max(1, min(lane_count, 8))  # 1~8개로 제한

    # 번호판: YOLO 감지 (모델 있으면 실행)
    plate_visible = "미확인"
    try:
        from ultralytics import YOLO
        for name in ["yolo26.pt", "runs/plate_detect/yolo26_plate_v1/weights/best.pt", "yolo11n.pt"]:
            p = ROOT / name
            if p.exists():
                model = YOLO(str(p))
                res = model(frame, conf=0.35, verbose=False)
                n = len(res[0].boxes) if res and res[0].boxes is not None else 0
                plate_visible = "보임" if n > 0 else "안보임"
                break
    except Exception:
        pass

    suitable = (
        angle == "부감"
        and lane_count >= 2
        and plate_visible in ("보임", "미확인")
    )
    return {
        "resolution": (w, h),
        "angle": angle,
        "lanes": lane_count,
        "plate": plate_visible,
        "suitable": "YES" if suitable else "NO",
    }


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_DIR)
    os.makedirs(folder, exist_ok=True)

    print("=" * 60)
    print("AI Hub 교통 CCTV 영상 검증 (100MB 이하만)")
    print("데이터셋: 교통문제 해결을 위한 CCTV 교통 영상")
    print(f"폴더: {folder}")
    print("=" * 60)

    files = get_video_files(folder)
    if not files:
        print("\n[안내] 폴더에 영상이 없습니다.")
        print("  - AI Hub에서 100MB 이하 파일만 다운로드해 아래 경로에 넣어주세요.")
        print(f"  - 경로: {folder}")
        print("  - 다운로드 전 반드시 파일 크기를 확인하고, 100MB 초과는 다운로드 금지.")
        return

    # 표: 파일명 | 크기 | 다운로드 여부
    print("\n파일명 | 크기 | 다운로드 여부")
    print("-------|------|-------------")
    under = []
    for path, name, mb in sorted(files, key=lambda x: -x[2]):
        if mb <= MAX_MB:
            status = "다운로드(검증 대상)"
            under.append((path, name, mb))
        else:
            status = "100MB 초과 금지"
        print(f"{name} | {mb:.1f}MB | {status}")

    # 100MB 이하 파일에 대해 첫 프레임 분석
    print("\n" + "=" * 60)
    print("고속도로 CCTV 적합성 확인 (첫 프레임 기준)")
    print("=" * 60)

    for path, name, mb in under:
        print(f"\n- 파일명: {name}")
        print(f"- 크기: {mb:.1f} MB")
        info = analyze_first_frame(path)
        if info is None:
            print("- 해상도: 열기 실패")
            print("- 고속도로 CCTV 적합: NO")
            continue
        w, h = info["resolution"]
        print(f"- 해상도: {w}x{h}")
        print(f"- 각도: {info['angle']}")
        print(f"- 차선수: {info['lanes']}개")
        print(f"- 번호판: {info['plate']}")
        print(f"- 고속도로 CCTV 적합: {info['suitable']}")


if __name__ == "__main__":
    main()
