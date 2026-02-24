# -*- coding: utf-8 -*-
"""
auto_label.py - YOLO 모델로 번호판 자동 어노테이션
==================================================
1차: yolo26n.pt (COCO) 로 차량 탐지 → 하단 영역에서 번호판 후보 추출
2차: 번호판 전용 모델이 있으면 직접 사용

결과: dataset/labels/train/ 에 YOLO 형식 .txt 생성
      class_id cx cy w h (정규화 좌표)
      class 0 = license_plate
"""
import os
import sys
import glob
import cv2
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG_DIR = os.path.join(SCRIPT_DIR, "dataset", "images", "train")
TRAIN_LBL_DIR = os.path.join(SCRIPT_DIR, "dataset", "labels", "train")
os.makedirs(TRAIN_LBL_DIR, exist_ok=True)

# ── 번호판 탐지 전략 ──
# HuggingFace 번호판 전용 모델 시도 → 없으면 COCO 차량 기반 휴리스틱
PLATE_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolo11x_plate.pt")
COCO_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolo26n.pt")

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def detect_plates_with_plate_model(model, img_path, img):
    """번호판 전용 모델로 직접 탐지"""
    h, w = img.shape[:2]
    results = model(img, conf=0.25, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            # YOLO 정규화 좌표
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if 0.01 < bw < 0.5 and 0.01 < bh < 0.3:  # 합리적 크기
                boxes.append((cx, cy, bw, bh))
    return boxes


def detect_plates_via_vehicles(model, img_path, img):
    """COCO 모델로 차량 탐지 → 하단에서 번호판 영역 추정"""
    h, w = img.shape[:2]
    results = model(img, conf=0.30, verbose=False)
    boxes = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            vx1, vy1, vx2, vy2 = [int(v) for v in box.xyxy[0]]
            vw = vx2 - vx1
            vh = vy2 - vy1

            if vw < 60 or vh < 40:
                continue

            # 차량 하단 30~40% 영역에 번호판이 위치한다고 가정
            # 번호판 크기: 차량 폭의 25~45%, 높이의 8~15%
            plate_w = int(vw * 0.35)
            plate_h = int(vh * 0.10)
            plate_cx = vx1 + vw // 2
            plate_cy = vy2 - int(vh * 0.15)

            px1 = max(0, plate_cx - plate_w // 2)
            py1 = max(0, plate_cy - plate_h // 2)
            px2 = min(w, plate_cx + plate_w // 2)
            py2 = min(h, plate_cy + plate_h // 2)

            # 번호판 영역에서 에지 기반 정밀 위치 보정
            plate_region = img[vy2 - int(vh * 0.35):vy2, vx1:vx2]
            if plate_region.size > 0:
                refined = refine_plate_location(plate_region, vx1, vy2 - int(vh * 0.35), w, h)
                if refined:
                    boxes.append(refined)
                    continue

            # 정밀 보정 실패시 기본 위치 사용
            cx = ((px1 + px2) / 2) / w
            cy = ((py1 + py2) / 2) / h
            bw = (px2 - px1) / w
            bh = (py2 - py1) / h
            if bw > 0.01 and bh > 0.005:
                boxes.append((cx, cy, bw, bh))

    return boxes


def refine_plate_location(region, offset_x, offset_y, img_w, img_h):
    """차량 하단 영역에서 에지/컨투어 기반 번호판 위치 정밀 탐색"""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    # CLAHE + 이진화
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 50, 150)

    # 형태학 닫힘으로 번호판 사각형 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rh, rw = region.shape[:2]
    best = None
    best_score = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        area = cw * ch

        # 번호판 aspect ratio: 2.0 ~ 6.0, 최소 면적
        if 1.8 < aspect < 6.5 and area > (rw * rh * 0.02):
            score = area * (1.0 if 2.5 < aspect < 5.0 else 0.5)
            if score > best_score:
                best_score = score
                abs_x1 = offset_x + x
                abs_y1 = offset_y + y
                abs_x2 = abs_x1 + cw
                abs_y2 = abs_y1 + ch
                # 패딩
                pad_x = int(cw * 0.1)
                pad_y = int(ch * 0.15)
                abs_x1 = max(0, abs_x1 - pad_x)
                abs_y1 = max(0, abs_y1 - pad_y)
                abs_x2 = min(img_w, abs_x2 + pad_x)
                abs_y2 = min(img_h, abs_y2 + pad_y)

                cx = ((abs_x1 + abs_x2) / 2) / img_w
                cy = ((abs_y1 + abs_y2) / 2) / img_h
                bw = (abs_x2 - abs_x1) / img_w
                bh = (abs_y2 - abs_y1) / img_h
                best = (cx, cy, bw, bh)

    return best


def main():
    print("=" * 60)
    print("  Auto-Labeling: License Plate Detection")
    print("=" * 60)

    # 모델 선택
    use_plate_model = os.path.isfile(PLATE_MODEL_PATH)
    if use_plate_model:
        print(f"  Using plate-specific model: {PLATE_MODEL_PATH}")
        model = YOLO(PLATE_MODEL_PATH)
        detect_fn = detect_plates_with_plate_model
    else:
        print(f"  Using COCO model + vehicle heuristic: {COCO_MODEL_PATH}")
        model = YOLO(COCO_MODEL_PATH)
        detect_fn = detect_plates_via_vehicles

    # 이미지 목록
    images = sorted(
        glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))
        + glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png"))
    )
    print(f"  Images to label: {len(images)}")

    labeled = 0
    empty = 0

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        boxes = detect_fn(model, img_path, img)

        # 라벨 파일 작성
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(TRAIN_LBL_DIR, f"{basename}.txt")

        if boxes:
            with open(lbl_path, "w") as f:
                for (cx, cy, bw, bh) in boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            labeled += 1
        else:
            # 빈 라벨 파일 (배경=negative sample)
            with open(lbl_path, "w") as f:
                pass
            empty += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(images)}] labeled={labeled}, empty={empty}")

    print(f"\nDone! labeled={labeled}, empty(background)={empty}")
    print(f"Labels saved to: {TRAIN_LBL_DIR}")


if __name__ == "__main__":
    main()
