"""
process_frame 파이프라인 디버그 스크립트
어디서 번호판이 사라지는지 단계별로 추적
"""
import multiprocessing
multiprocessing.freeze_support()

import cv2
import sys
import re
import numpy as np

# 영상에서 번호판이 있는 프레임(135번) 추출
cap = cv2.VideoCapture(r"C:\tool\yolo26-main\movie\hiway.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 135)
ret, frame = cap.read()
cap.release()
print(f"Frame shape: {frame.shape}")

# 엔진 생성
from plate_engine_pro import PlateEnginePro, PlateEngineConfig
engine = PlateEnginePro()
print(f"_is_plate_model: {engine._is_plate_model}")
print(f"Model names: {engine.model.names}")
print(f"OCR engines: {list(engine.ocr_engines.keys())}")
print(f"DETECT_CONF: {engine.config.DETECT_CONF}")
print(f"OCR_CONF: {engine.config.OCR_CONF}")
print()

# Step 1: YOLO 탐지
_fh, _fw = frame.shape[:2]
if _fw >= 3840:
    _imgsz = 1920
elif _fw >= 1920:
    _imgsz = 1280
elif _fw >= 1280:
    _imgsz = 960
else:
    _imgsz = 640
print(f"imgsz: {_imgsz}")

detections = engine.model(frame, conf=engine.config.DETECT_CONF, imgsz=_imgsz, verbose=False)
boxes = detections[0].boxes
print(f"Step 1 - YOLO detections: {len(boxes)} boxes")

for i, det in enumerate(boxes):
    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
    conf = float(det.conf[0])
    cls_id = int(det.cls[0])
    w, h = x2-x1, y2-y1
    print(f"  Box {i}: cls={cls_id}, conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2}), size={w}x{h}")

if len(boxes) == 0:
    print("NO DETECTIONS - exiting")
    sys.exit(1)

# Step 2: 첫 번째 탐지에 대해 파이프라인 추적
det = boxes[0]
x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
conf = float(det.conf[0])
print(f"\nStep 2 - Processing first detection: bbox=({x1},{y1},{x2},{y2}), conf={conf:.3f}")

# _is_plate_model 체크
if not engine._is_plate_model:
    cls_id = int(det.cls[0])
    print(f"  NOT plate model, cls_id={cls_id}, vehicle_ids={engine._vehicle_class_ids}")
    if cls_id not in engine._vehicle_class_ids:
        print(f"  FILTERED: cls_id {cls_id} not in vehicle classes!")
else:
    print(f"  Plate model - no class filter applied")

# 크기 필터
det_w = x2 - x1
det_h = y2 - y1
print(f"  Size: {det_w}x{det_h}, min: 15x8")
if det_w < 15 or det_h < 8:
    print(f"  FILTERED: too small!")

# ROI 크롭
crop_src = frame
ch_full, cw_full = crop_src.shape[:2]
sx, sy = 1.0, 1.0  # same frame
ox1, oy1, ox2, oy2 = x1, y1, x2, y2

margin_x = int((ox2 - ox1) * 0.35)
margin_y = int((oy2 - oy1) * 0.40)
rx1 = max(0, ox1 - margin_x)
ry1 = max(0, oy1 - margin_y)
rx2 = min(cw_full, ox2 + margin_x)
ry2 = min(ch_full, oy2 + margin_y)
roi = crop_src[ry1:ry2, rx1:rx2]
print(f"  ROI: ({rx1},{ry1})-({rx2},{ry2}), shape={roi.shape}")

# ROI 저장
cv2.imwrite("debug_roi.jpg", roi)
print(f"  Saved debug_roi.jpg")

# 업스케일
roi_for_ocr = roi
roi_h, roi_w = roi.shape[:2]
target_w = 500
if roi_w < target_w:
    scale = target_w / roi_w
    if roi_w < 60:
        scale = max(scale, 6.0)
    elif roi_w < 120:
        scale = max(scale, 4.0)
else:
    scale = 1.0
print(f"  Upscale: roi_w={roi_w}, scale={scale:.2f}")

if scale > 1.0:
    roi_for_ocr = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    roi_for_ocr = cv2.filter2D(roi_for_ocr, -1, kernel)

# 패딩
pad = max(10, int(roi_for_ocr.shape[0] * 0.15))
roi_for_ocr = cv2.copyMakeBorder(roi_for_ocr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
print(f"  After upscale+pad: {roi_for_ocr.shape}")
cv2.imwrite("debug_roi_processed.jpg", roi_for_ocr)
print(f"  Saved debug_roi_processed.jpg")

# Step 3: OCR 앙상블
print(f"\nStep 3 - OCR ensemble")
all_candidates = []
roi_inv = cv2.bitwise_not(roi_for_ocr)

for method in engine.config.PREPROCESS_METHODS + ["_inverted"]:
    try:
        if method == "original":
            processed = roi_for_ocr.copy()
        elif method == "_inverted":
            processed = roi_inv.copy()
        else:
            proc_func = getattr(engine.preprocessor, method, None)
            if proc_func is None:
                continue
            processed = proc_func(roi_for_ocr.copy())

        for engine_name, eng in engine.ocr_engines.items():
            text, ocr_conf = engine._run_ocr(engine_name, eng, processed)
            raw_text = text

            if not text or ocr_conf < 0.15:
                if text:
                    print(f"  [{method}][{engine_name}] raw='{text}' conf={ocr_conf:.3f} -> FILTERED (conf<0.15)")
                continue

            cleaned = engine.validator.clean_ocr_text(text)
            valid_len = engine.validator.is_valid_length(cleaned)

            if not valid_len:
                print(f"  [{method}][{engine_name}] raw='{raw_text}' -> cleaned='{cleaned}' len={len(cleaned)} -> FILTERED (length)")
                continue

            is_valid, final_text = engine.validator.validate(cleaned)
            if not is_valid:
                print(f"  [{method}][{engine_name}] raw='{raw_text}' -> cleaned='{cleaned}' -> FILTERED (pattern)")
                continue

            print(f"  [{method}][{engine_name}] raw='{raw_text}' -> cleaned='{cleaned}' -> final='{final_text}' conf={ocr_conf:.3f} -> PASSED!")
            all_candidates.append((final_text, ocr_conf))
    except Exception as e:
        print(f"  [{method}] Error: {e}")

print(f"\nTotal candidates: {len(all_candidates)}")
for t, c in all_candidates:
    print(f"  '{t}' conf={c:.3f}")

if not all_candidates:
    print("\nNO CANDIDATES PASSED! OCR/Validation pipeline is blocking everything.")
    print("Trying raw OCR on roi without validation...")
    for engine_name, eng in engine.ocr_engines.items():
        text, ocr_conf = engine._run_ocr(engine_name, eng, roi_for_ocr)
        print(f"  [{engine_name}] raw='{text}' conf={ocr_conf:.3f}")
        text2, ocr_conf2 = engine._run_ocr(engine_name, eng, roi_inv)
        print(f"  [{engine_name}] inverted raw='{text2}' conf={ocr_conf2:.3f}")
