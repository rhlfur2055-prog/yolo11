"""
OCR 정밀 디버깅: 실제 영상에서 번호판 크롭 → 전처리 전/후 비교 → EasyOCR 결과 분석
"""
import cv2
import numpy as np
import os

# 영상 로드
VIDEO = "temp_youtube/KakaoTalk_20260219_123507625.mp4"
cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"영상: {VIDEO}, 총 {total}프레임")

# YOLO 번호판 탐지
print("\n=== 1단계: YOLO 번호판 탐지 ===")
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

hf_path = hf_hub_download(repo_id="Koushim/yolov8-license-plate-detection", filename="best.pt")
model = YOLO(hf_path)
print(f"모델 로드 완료: {hf_path}")

# 여러 프레임에서 번호판 크롭
test_frames = [150, 300, 450, 600, 750, 900]
crops = []
for fidx in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if not ret:
        continue
    results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
    if results and results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size > 0:
                crops.append({"frame": fidx, "crop": plate_crop, "conf": conf, "bbox": (x1,y1,x2,y2)})
                # 저장
                os.makedirs("debug_crops", exist_ok=True)
                cv2.imwrite(f"debug_crops/f{fidx}_plate{i}_raw.png", plate_crop)
                print(f"  Frame {fidx}: plate {i} size={plate_crop.shape[:2]} conf={conf:.3f} bbox=({x1},{y1},{x2},{y2})")

cap.release()
print(f"\n총 {len(crops)}개 번호판 크롭 수집")

if not crops:
    print("번호판 탐지 실패!")
    exit()

# === 2단계: 전처리 비교 ===
print("\n=== 2단계: 전처리 비교 ===")

def preprocess_clahe_binary(img):
    """현재 코드: CLAHE + 이진화"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary

def preprocess_raw_gray(img):
    """원본 그레이스케일만"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img.copy()

def preprocess_clahe_only(img):
    """CLAHE만 (이진화 없음)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def resize_min_height(img, min_h=140):
    h, w = img.shape[:2]
    if h >= min_h:
        return img
    scale = min_h / h
    return cv2.resize(img, (int(w*scale), min_h), interpolation=cv2.INTER_CUBIC)

# === 3단계: EasyOCR 테스트 ===
print("\n=== 3단계: EasyOCR 테스트 ===")
import easyocr
reader = easyocr.Reader(["en"], gpu=True)

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

for ci, crop_info in enumerate(crops[:8]):
    plate = crop_info["crop"]
    fidx = crop_info["frame"]
    print(f"\n--- Crop {ci} (Frame {fidx}, size={plate.shape[:2]}, det_conf={crop_info['conf']:.3f}) ---")

    # 크기 정규화
    plate_resized = resize_min_height(plate, 140)
    print(f"  리사이즈 후: {plate_resized.shape[:2]}")

    # 4가지 전처리 방식 비교
    variants = {
        "A_원본_컬러": plate_resized,
        "B_원본_그레이": preprocess_raw_gray(plate_resized),
        "C_CLAHE만": preprocess_clahe_only(plate_resized),
        "D_CLAHE+이진화": preprocess_clahe_binary(plate_resized),
    }

    for name, img in variants.items():
        # EasyOCR는 BGR 필요
        if len(img.shape) == 2:
            ocr_input = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            ocr_input = img

        result = reader.readtext(ocr_input, detail=1, paragraph=False, allowlist=ALLOWLIST)
        if result:
            result.sort(key=lambda r: r[0][0][0])
            texts = [r[1] for r in result if r[1].strip()]
            confs = [r[2] for r in result if r[1].strip()]
            combined = "".join(texts)
            avg_conf = sum(confs)/len(confs) if confs else 0
            print(f"  {name:20s} → '{combined}' conf={avg_conf:.3f} (parts={texts})")
        else:
            print(f"  {name:20s} → (인식 실패)")

        # 저장
        cv2.imwrite(f"debug_crops/f{fidx}_c{ci}_{name}.png", img if len(img.shape)==3 else img)

    # allowlist 없이도 테스트
    result_noallow = reader.readtext(plate_resized, detail=1, paragraph=False)
    if result_noallow:
        texts_na = [r[1] for r in result_noallow if r[1].strip()]
        confs_na = [r[2] for r in result_noallow if r[1].strip()]
        combined_na = "".join(texts_na)
        avg_na = sum(confs_na)/len(confs_na) if confs_na else 0
        print(f"  {'E_allowlist없음':20s} → '{combined_na}' conf={avg_na:.3f}")
    else:
        print(f"  {'E_allowlist없음':20s} → (인식 실패)")

print("\n=== 디버깅 완료 ===")
print(f"크롭 이미지 저장: debug_crops/")
