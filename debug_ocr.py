from ultralytics import YOLO
from plate_engine_pro import PlateEnginePro, PlateEngineConfig
import cv2, sys

engine = PlateEnginePro()
engine.consecutive_required = 1
engine.config.PREPROCESS_METHODS = ["original"]
engine.config.DETECT_CONF = 0.15
engine.config.OCR_CONF = 0.0

cap = cv2.VideoCapture("youtube_-dlpsCgjCAg.mkv")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 각 감지 프레임에서 OCR 결과 직접 출력
m = YOLO("yolo11x_plate.pt")

for frame_no in range(0, total, 30):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        continue
    res = m(frame, conf=0.15, verbose=False)
    if len(res[0].boxes) == 0:
        continue

    b = res[0].boxes[0]
    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
    crop = frame[y1:y2, x1:x2]

    # EasyOCR 직접
    raw_text = ""
    for eng_name, eng in engine.ocr_engines.items():
        try:
            if eng_name == "easyocr":
                results = eng.readtext(crop, detail=1)
                texts = [r[1] for r in results]
                raw_text = " ".join(texts)
            elif eng_name == "paddleocr":
                results = eng.ocr(crop, cls=True)
                if results and results[0]:
                    texts = [line[1][0] for line in results[0] if line]
                    raw_text = " ".join(texts)
        except Exception as e:
            raw_text = f"ERROR: {e}"
        print(f"frame={frame_no} [{eng_name}] raw='{raw_text}' yolo_conf={float(b.conf[0]):.2f}")

cap.release()
