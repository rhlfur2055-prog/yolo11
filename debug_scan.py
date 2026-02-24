"""스캔 0개 원인 추적 스크립트"""
import sys, cv2, numpy as np
sys.path.insert(0, "C:/tools/yolo26")

VIDEO = "C:/tools/yolo26/youtube_-dlpsCgjCAg.mkv"
STEP  = 15

print("=== YOLO 단독 테스트 ===")
from ultralytics import YOLO
import os

model_path = "C:/tools/yolo26/yolo11x_plate.pt"
if not os.path.exists(model_path):
    print(f"모델 없음: {model_path}")
    sys.exit(1)

model = YOLO(model_path)
cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"영상: {total}프레임, {fps}fps")

yolo_detections = 0
frame_idx = 0
sample_frames = []  # YOLO 감지된 프레임 저장

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % STEP != 0:
        continue

    h0, w0 = frame.shape[:2]
    if w0 > 640:
        sc = 640 / w0
        small = cv2.resize(frame, (640, int(h0 * sc)))
    else:
        small = frame

    sharp_k = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
    small_sharp = cv2.filter2D(small, -1, sharp_k)

    res = model(small_sharp, conf=0.05, verbose=False)
    boxes = res[0].boxes if res else []
    n = len(boxes) if boxes is not None else 0
    if n > 0:
        yolo_detections += 1
        t = frame_idx / fps
        m, s = divmod(int(t), 60)
        confs = [float(b.conf[0]) for b in boxes]
        print(f"  [{m:02d}:{s:02d}] 프레임{frame_idx}: YOLO {n}개 감지, conf={[f'{c:.2f}' for c in confs]}")
        if len(sample_frames) < 3:
            sample_frames.append((frame_idx, small_sharp, boxes))

cap.release()
print(f"\nYOLO 감지 프레임 수: {yolo_detections} / {total//STEP}")

if yolo_detections == 0:
    print(">> YOLO가 아무것도 못 잡음 → 모델 문제 또는 conf 너무 낮음")
    sys.exit(0)

print("\n=== OCR 테스트 (첫 번째 감지 프레임) ===")
from plate_engine_pro import PlateEnginePro
engine = PlateEnginePro()
engine.consecutive_required = 1
engine.config.DETECT_CONF = 0.05
engine.config.OCR_CONF = 0.01  # OCR conf 최저로 낮춰서 테스트

frame_no, test_frame, _ = sample_frames[0]
raw = engine.process_frame(test_frame, camera_id="TEST")
print(f"프레임 {frame_no} process_frame 결과: {raw}")
if not raw:
    print(">> OCR/검증 단계에서 탈락")

# OCR 원본 출력 확인
print("\n=== OCR 원본 텍스트 직접 확인 ===")
from ultralytics import YOLO as YOLO2
import cv2 as cv2b
m2 = YOLO2("yolo11x_plate.pt")
cap2 = cv2b.VideoCapture(VIDEO)
cap2.set(cv2b.CAP_PROP_POS_FRAMES, frame_no - 1)
_, f2 = cap2.read()
cap2.release()
h0, w0 = f2.shape[:2]
sc = 640/w0 if w0>640 else 1.0
sm = cv2b.resize(f2,(int(w0*sc),int(h0*sc))) if sc!=1.0 else f2
sharp_k = __import__('numpy').array([[0,-0.5,0],[-0.5,3,-0.5],[0,-0.5,0]],dtype='float32')
sm = cv2b.filter2D(sm,-1,sharp_k)
res2 = m2(cv2b.cvtColor(sm,cv2b.COLOR_BGR2RGB),conf=0.05,verbose=False)
for bi, box in enumerate(res2[0].boxes):
    x1,y1,x2,y2=[int(v) for v in box.xyxy[0].tolist()]
    crop = sm[max(0,y1):y2, max(0,x1):x2]
    if crop.size==0: continue
    ch,cw=crop.shape[:2]
    print(f"  bbox{bi}: {cw}x{ch}px, yolo_conf={float(box.conf[0]):.2f}")
    # 크롭 저장
    cv2b.imwrite(f"debug_crop_f{frame_no}_b{bi}.jpg", crop)
    print(f"  → debug_crop_f{frame_no}_b{bi}.jpg 저장됨")
    # OCR 직접 실행
    for eng_name, eng_obj in engine.ocr_engines.items():
        t, c = engine._run_ocr(eng_name, eng_obj, crop)
        cleaned = engine.validator.clean_ocr_text(t) if t else ""
        ok, result = engine.validator.validate(cleaned) if cleaned else (False,"")
        print(f"  [{eng_name}] raw='{t}' conf={c:.2f} → cleaned='{cleaned}' → validate={'OK:'+result if ok else 'FAIL'}")
