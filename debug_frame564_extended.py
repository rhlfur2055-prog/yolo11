"""
프레임 564 주변 확장 분석 - 버스 번호판 위치 파악
"""
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "C:/tool/yolo26-main/movie/hiway.mp4"
MODEL_PATH = "C:/tool/yolo26-main/best.pt"

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    # 1. 프레임 564에서 유일한 탐지 분석
    cap.set(cv2.CAP_PROP_POS_FRAMES, 564)
    ret, frame = cap.read()

    print("=" * 70)
    print("[프레임 564 유일한 탐지 분석]")
    print(f"탐지 bbox: (1636,654)-(1887,810), 크기 250x156")
    print(f"위치: 프레임 우측 하단 (x=85-98%, y=61-75%)")
    print(f"이것은 우측 차선 차량의 번호판으로 보임")

    # 탐지된 번호판 크롭 저장
    crop = frame[654:810, 1636:1887]
    cv2.imwrite("C:/tool/yolo26-main/debug_det0_crop.jpg", crop)
    print(f"탐지 크롭 저장: debug_det0_crop.jpg")

    # 2. 프레임 범위 스캔 (550-580) - 버스가 어디에 있는지 확인
    print("\n" + "=" * 70)
    print("[프레임 550-580 스캔 - 모든 탐지]")
    print(f"{'Frame':>6} {'#Det':>5} {'Detections (x1,y1,x2,y2,conf)':50}")
    print("-" * 70)

    for fidx in range(550, 581, 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, f = cap.read()
        if not ret:
            continue
        results = model.predict(f, conf=0.05, max_det=20, verbose=False)
        boxes = results[0].boxes
        det_str = ""
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            det_str += f"({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) c={conf:.2f}  "
        print(f"{fidx:6d} {len(boxes):5d}  {det_str if det_str else '없음'}")

    # 3. 더 넓은 범위 스캔 (500-650) - 버스 진입/이탈 시점 확인
    print("\n" + "=" * 70)
    print("[프레임 500-650 스캔 (10프레임 간격) - 탐지 패턴]")
    print(f"{'Frame':>6} {'#Det':>5} {'Detections':50}")
    print("-" * 70)

    for fidx in range(500, 651, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, f = cap.read()
        if not ret:
            continue
        results = model.predict(f, conf=0.05, max_det=20, verbose=False)
        boxes = results[0].boxes
        det_str = ""
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            w = x2 - x1
            h = y2 - y1
            det_str += f"({x1:.0f},{y1:.0f}) {w:.0f}x{h:.0f} c={conf:.2f}  "
        print(f"{fidx:6d} {len(boxes):5d}  {det_str if det_str else '없음'}")

    # 4. 프레임 564 전체 영역 분석 - 버스가 있는지 확인
    cap.set(cv2.CAP_PROP_POS_FRAMES, 564)
    ret, frame = cap.read()

    # 프레임의 다양한 영역 크롭 저장
    print("\n" + "=" * 70)
    print("[프레임 564 영역별 크롭 저장]")

    regions = {
        'left': (0, 300, 640, 800),
        'center': (480, 300, 1440, 800),
        'right': (1280, 300, 1920, 800),
        'full_lower': (0, 400, 1920, 1080),
    }

    for name, (x1, y1, x2, y2) in regions.items():
        roi = frame[y1:y2, x1:x2]
        path = f"C:/tool/yolo26-main/debug_region_{name}.jpg"
        cv2.imwrite(path, roi)
        print(f"  {name}: ({x1},{y1})-({x2},{y2}) → {path}")

    cap.release()
    print("\n완료!")

if __name__ == "__main__":
    main()
