from ultralytics import YOLO
from plate_engine_pro import PlateEnginePro
import cv2

engine = PlateEnginePro()
engine.consecutive_required = 1
engine.config.PREPROCESS_METHODS = ["original"]
engine.config.DETECT_CONF = 0.20
engine.config.OCR_CONF = 0.0
print("[엔진] OCR:", list(engine.ocr_engines.keys()))
print("[엔진] consecutive_required:", engine.consecutive_required)

cap = cv2.VideoCapture("youtube_-dlpsCgjCAg.mkv")
found = 0
for frame_no in range(80, 250, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        continue
    results = engine.process_frame(frame, camera_id="TEST")
    if results:
        found += 1
        for r in results:
            plate = r.get("plate", "")
            conf = r.get("confidence", 0)
            print(f"  frame {frame_no}: [{plate}] conf={conf:.2f}")
        if found >= 5:
            break
    else:
        # YOLO만 단독 확인
        m = YOLO("yolo11x_plate.pt")
        res = m(frame, conf=0.20, verbose=False)
        if len(res[0].boxes) > 0:
            b = res[0].boxes[0]
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            print(f"  frame {frame_no}: YOLO감지O conf={float(b.conf[0]):.2f} but OCR실패")
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(f"debug_crop_{frame_no}.jpg", crop)

cap.release()
if found == 0:
    print(">> 인식 결과 없음!")
else:
    print(f">> 총 {found}건 인식 성공")
