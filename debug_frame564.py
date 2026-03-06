"""
디버그 스크립트: hiway.mp4 프레임 564에서 YOLO 번호판 탐지 분석
"""
import cv2
import numpy as np
import sys
import os

# YOLO 모델 로드
from ultralytics import YOLO

VIDEO_PATH = "C:/tool/yolo26-main/movie/hiway.mp4"
MODEL_PATH = "C:/tool/yolo26-main/best.pt"
FRAME_IDX = 564

def main():
    # 1. 비디오 열기 및 프레임 추출
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: 비디오 열기 실패")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"비디오 정보: {w}x{h}, {fps:.2f}fps, 총 {total} 프레임")
    print(f"프레임 {FRAME_IDX} = 약 {FRAME_IDX/fps:.2f}초")
    print("=" * 70)

    # 프레임 시크
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: 프레임 읽기 실패")
        return

    print(f"프레임 shape: {frame.shape}")

    # 프레임 저장 (디버그용)
    out_path = "C:/tool/yolo26-main/debug_frame564.jpg"
    cv2.imwrite(out_path, frame)
    print(f"프레임 저장: {out_path}")
    print("=" * 70)

    # 2. YOLO 모델 로드 및 추론
    print(f"\nYOLO 모델 로드: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 모델 클래스 정보
    names = model.names
    print(f"모델 클래스: {names}")
    print("=" * 70)

    # 낮은 confidence로 전체 탐지
    print(f"\n[추론] conf=0.10, max_det=20")
    results = model.predict(
        frame,
        conf=0.10,
        max_det=20,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    print(f"\n총 탐지 수: {len(boxes)}")
    print("-" * 70)

    if len(boxes) == 0:
        print("탐지 없음! conf를 더 낮춰봅니다...")
        # conf=0.01로 재시도
        results2 = model.predict(frame, conf=0.01, max_det=50, verbose=False)
        boxes2 = results2[0].boxes
        print(f"\n[재시도] conf=0.01, max_det=50 → 탐지 수: {len(boxes2)}")
        if len(boxes2) > 0:
            boxes = boxes2
            result = results2[0]

    # 3. 모든 탐지 출력
    print(f"\n{'#':>3} {'Class':>10} {'Conf':>6} {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6} {'W':>5} {'H':>5} {'Area':>8}")
    print("-" * 70)

    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, f"cls{cls_id}")
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        detections.append({
            'idx': i, 'cls': cls_name, 'conf': conf,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'w': bw, 'h': bh, 'area': area
        })

        print(f"{i:3d} {cls_name:>10} {conf:6.3f} {x1:6.0f} {y1:6.0f} {x2:6.0f} {y2:6.0f} {bw:5.0f} {bh:5.0f} {area:8.0f}")

    print("=" * 70)

    # 4. 버스 번호판 영역 분석 (중앙-우측, 하단부)
    # 1920x1080 프레임에서 버스 번호판은 대략 중앙~우측, 중하단
    bus_region = {
        'x_min': 600, 'x_max': 1400,
        'y_min': 400, 'y_max': 900
    }

    print(f"\n[버스 번호판 영역 분석] x:{bus_region['x_min']}-{bus_region['x_max']}, y:{bus_region['y_min']}-{bus_region['y_max']}")
    found_in_region = False
    for d in detections:
        cx = (d['x1'] + d['x2']) / 2
        cy = (d['y1'] + d['y2']) / 2
        if (bus_region['x_min'] <= cx <= bus_region['x_max'] and
            bus_region['y_min'] <= cy <= bus_region['y_max']):
            print(f"  → 영역 내 탐지: #{d['idx']} {d['cls']} conf={d['conf']:.3f} "
                  f"bbox=({d['x1']:.0f},{d['y1']:.0f})-({d['x2']:.0f},{d['y2']:.0f}) "
                  f"size={d['w']:.0f}x{d['h']:.0f}")
            found_in_region = True

    if not found_in_region:
        print("  → 해당 영역에 탐지 없음!")

    # 5. 프레임 밝기/특성 분석 (해당 영역)
    roi = frame[bus_region['y_min']:bus_region['y_max'],
                bus_region['x_min']:bus_region['x_max']]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    print(f"\n[영역 이미지 특성]")
    print(f"  BGR mean: {roi.mean(axis=(0,1))}")
    print(f"  HSV mean: {hsv.mean(axis=(0,1))}")
    print(f"  밝기(V) min/max: {hsv[:,:,2].min()} / {hsv[:,:,2].max()}")

    # 6. 탐지 시각화 저장
    annotated = frame.copy()
    for d in detections:
        color = (0, 255, 0) if d['conf'] >= 0.25 else (0, 165, 255)  # 녹색: 고신뢰, 주황: 저신뢰
        cv2.rectangle(annotated,
                      (int(d['x1']), int(d['y1'])),
                      (int(d['x2']), int(d['y2'])),
                      color, 2)
        label = f"{d['cls']} {d['conf']:.2f}"
        cv2.putText(annotated, label,
                    (int(d['x1']), int(d['y1']) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 버스 영역 표시 (파란색 점선 대신 실선)
    cv2.rectangle(annotated,
                  (bus_region['x_min'], bus_region['y_min']),
                  (bus_region['x_max'], bus_region['y_max']),
                  (255, 0, 0), 1)

    out_annotated = "C:/tool/yolo26-main/debug_frame564_annotated.jpg"
    cv2.imwrite(out_annotated, annotated)
    print(f"\n시각화 저장: {out_annotated}")

    # 7. 추가: 여러 confidence threshold 비교
    print("\n" + "=" * 70)
    print("[다양한 confidence threshold 비교]")
    for conf_th in [0.50, 0.25, 0.10, 0.05, 0.01]:
        res = model.predict(frame, conf=conf_th, max_det=50, verbose=False)
        n = len(res[0].boxes)
        print(f"  conf={conf_th:.2f} → 탐지 {n}개")

    print("\n완료!")

if __name__ == "__main__":
    main()
