import cv2, time, sys
sys.path.insert(0, '.')
from plate_engine_pro import PlateEnginePro

engine = PlateEnginePro()
cap = cv2.VideoCapture('movie/hiway.mp4')

print("=== YOLO26 파이프라인 검증 ===")
print(f"모델: {engine.model.ckpt_path}")
print(f"DETECT_CONF: {engine.config.DETECT_CONF}")
print(f"consecutive: {engine.consecutive_required}")
print(f"OCR엔진: {list(engine.ocr_engines.keys())}")
print(f"ROI: X({engine.config.ROI_X1}~{engine.config.ROI_X2}) Y({engine.config.ROI_Y1}~{engine.config.ROI_Y2})")

total_det = 0
total_ms = 0
plates = {}

for fnum in range(0, 1800, 30):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
    ret, frame = cap.read()
    if not ret: break
    t0 = time.time()
    results = engine.process_frame(frame, 'CAM01')
    ms = (time.time() - t0) * 1000
    total_ms += ms
    if results:
        total_det += len(results)
        for r in results:
            p = r.get('plate', '')
            c = r.get('confidence', 0)
            print(f"  F{fnum}: {p} ({c:.0%}) [{ms:.0f}ms]")
            if p not in plates or c > plates[p]:
                plates[p] = c

cap.release()
n = 1800 // 30
avg = total_ms / max(n, 1)
print(f"\n=== 결과 ===")
print(f"프레임: {n}, 탐지: {total_det}건, 평균: {avg:.0f}ms, FPS: {1000/max(avg,1):.1f}")
print(f"\n번호판 {len(plates)}종:")
for p, c in sorted(plates.items(), key=lambda x: -x[1]):
    print(f"  {p}: {c:.0%}")
