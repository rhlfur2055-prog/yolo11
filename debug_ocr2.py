"""
OCR 정밀 비교: 업스케일 방식별 EasyOCR 결과
"""
import cv2, numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import easyocr

VIDEO = "temp_youtube/KakaoTalk_20260219_123507625.mp4"
cap = cv2.VideoCapture(VIDEO)

hf_path = hf_hub_download(repo_id="Koushim/yolov8-license-plate-detection", filename="best.pt")
model = YOLO(hf_path)
reader = easyocr.Reader(["en"], gpu=True)
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def upscale_cubic(img, min_h=64):
    h, w = img.shape[:2]
    if h >= min_h: return img
    s = min_h / h
    return cv2.resize(img, (int(w*s), min_h), interpolation=cv2.INTER_CUBIC)

def upscale_lanczos_sharp(img, min_h=64):
    h, w = img.shape[:2]
    if h >= min_h: return img
    s = min_h / h
    up = cv2.resize(img, (int(w*s), min_h), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def upscale_lanczos_sharp_big(img, min_h=100):
    h, w = img.shape[:2]
    if h >= min_h: return img
    s = min_h / h
    up = cv2.resize(img, (int(w*s), min_h), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def ocr_test(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    r = reader.readtext(img, detail=1, paragraph=False, allowlist=ALLOW)
    if r:
        r.sort(key=lambda x: x[0][0][0])
        texts = [x[1] for x in r if x[1].strip()]
        confs = [x[2] for x in r if x[1].strip()]
        return "".join(texts), sum(confs)/len(confs) if confs else 0
    return "", 0

# 실제 번호판이 보이는 프레임 (스크린샷 기준)
test_frames = [150, 300, 450, 600, 750]
print("=" * 80)
print(f"{'Frame':>6} | {'Size':>8} | {'Conf':>5} | {'Raw':>15} | {'Cubic64':>15} | {'Lanczos64':>15} | {'Lanczos100':>15}")
print("=" * 80)

for fidx in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if not ret: continue
    results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
    if not results or results[0].boxes is None: continue
    for i, box in enumerate(results[0].boxes):
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue

        raw_text, raw_c = ocr_test(crop)
        cubic_text, cubic_c = ocr_test(upscale_cubic(crop))
        lanc_text, lanc_c = ocr_test(upscale_lanczos_sharp(crop))
        lanc100_text, lanc100_c = ocr_test(upscale_lanczos_sharp_big(crop))

        h, w = crop.shape[:2]
        print(f"F{fidx:>4} | {h:>3}x{w:<3} | {conf:.2f} | {raw_text:>15} | {cubic_text:>15} | {lanc_text:>15} | {lanc100_text:>15}")

cap.release()
print("\n완료!")
