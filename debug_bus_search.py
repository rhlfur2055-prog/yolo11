"""
버스 번호판 탐색 - 큰 bbox를 찾아서 버스 위치 확인
"""
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "C:/tool/yolo26-main/movie/hiway.mp4"
MODEL_PATH = "C:/tool/yolo26-main/best.pt"

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 전체 비디오에서 큰 번호판 탐지 찾기 (버스 = 큰 번호판)
    print("=" * 70)
    print("[전체 비디오 스캔 - 큰 번호판 탐지 (W>200 또는 H>100)]")
    print(f"{'Frame':>6} {'Time':>7} {'#Det':>5} {'BBox':>30} {'Size':>12} {'Conf':>6}")
    print("-" * 70)

    big_plates = []
    for fidx in range(0, total, 5):  # 5프레임 간격
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, f = cap.read()
        if not ret:
            continue
        results = model.predict(f, conf=0.05, max_det=20, verbose=False)
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            w = x2 - x1
            h = y2 - y1
            if w > 200 or h > 100:
                t = fidx / fps
                print(f"{fidx:6d} {t:6.1f}s {len(boxes):5d}  ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  {w:.0f}x{h:.0f}  {conf:.3f}")
                big_plates.append((fidx, x1, y1, x2, y2, w, h, conf))

    print(f"\n총 큰 번호판: {len(big_plates)}개")

    # 2프레임 이상 연속 멀티 탐지 찾기
    print("\n" + "=" * 70)
    print("[멀티 탐지 프레임 (2개 이상 동시 탐지)]")
    print("-" * 70)

    for fidx in range(0, total, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, f = cap.read()
        if not ret:
            continue
        results = model.predict(f, conf=0.10, max_det=20, verbose=False)
        boxes = results[0].boxes
        if len(boxes) >= 2:
            t = fidx / fps
            det_str = ""
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                w = x2 - x1
                h = y2 - y1
                det_str += f"({x1:.0f},{y1:.0f}) {w:.0f}x{h:.0f} c={conf:.2f}  "
            print(f"F{fidx:5d} ({t:5.1f}s) #{len(boxes):d}: {det_str}")

    cap.release()
    print("\n완료!")

if __name__ == "__main__":
    main()
