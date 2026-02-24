"""
번호판 인식 전체 파이프라인 테스트
- 영상에서 프레임 샘플 추출
- YOLO 감지 → 크롭 이미지 저장
- OCR 실행 → 결과 출력
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, "C:/tools/yolo26")

VIDEO   = "C:/tools/yolo26/youtube_-dlpsCgjCAg.mkv"
OUT_DIR = "C:/tools/yolo26/test_results"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1: 번호판 크롭 추출")
print("=" * 60)

from ultralytics import YOLO
model = YOLO("yolo11x_plate.pt")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# 영상 전체에서 20개 프레임 균등 샘플
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_every = total // 20
saved_crops = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % sample_every != 0:
        continue

    # 전처리
    h0, w0 = frame.shape[:2]
    sc = 640 / w0 if w0 > 640 else 1.0
    small = cv2.resize(frame, (int(w0*sc), int(h0*sc)))
    k = np.array([[0,-0.5,0],[-0.5,3,-0.5],[0,-0.5,0]], dtype=np.float32)
    small = cv2.filter2D(small, -1, k)

    results = model(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), conf=0.05, verbose=False)
    boxes = results[0].boxes if results else []

    for bi, box in enumerate(boxes or []):
        yconf = float(box.conf[0])
        if yconf < 0.08:
            continue
        # ★ 원본 해상도로 역변환 후 원본 프레임에서 크롭
        x1,y1,x2,y2 = [int(v / sc) for v in box.xyxy[0].tolist()]
        x1=max(0,x1); y1=max(0,y1); x2=min(w0,x2); y2=min(h0,y2)
        crop = frame[y1:y2, x1:x2]   # 원본 고해상도
        if crop.size == 0 or crop.shape[1] < 20:
            continue

        t_sec = frame_idx / fps
        m, s = divmod(int(t_sec), 60)
        fname = f"{OUT_DIR}/f{frame_idx:04d}_{m:02d}m{s:02d}s_b{bi}_conf{yconf:.2f}.jpg"
        cv2.imwrite(fname, crop)
        saved_crops.append((frame_idx, t_sec, crop, fname, yconf))
        print(f"  [{m:02d}:{s:02d}] 크롭 저장: {os.path.basename(fname)} ({crop.shape[1]}x{crop.shape[0]}px)")

cap.release()
print(f"\n총 {len(saved_crops)}개 크롭 저장 완료\n")

if not saved_crops:
    print("YOLO 감지 없음!")
    sys.exit(1)

print("=" * 60)
print("STEP 2: OCR + 검증 결과")
print("=" * 60)

from plate_engine_pro import PlateEnginePro, PlateValidator
engine = PlateEnginePro()
engine.consecutive_required = 1
engine.config.OCR_CONF = 0.01

validator = PlateValidator()
results_table = []

for frame_no, t_sec, crop, fname, yconf in saved_crops:
    m, s = divmod(int(t_sec), 60)

    # 크롭 확대
    ch, cw = crop.shape[:2]
    scale_up = max(1, 200 // max(cw, 1))
    if scale_up > 1:
        crop_up = cv2.resize(crop, (cw*scale_up, ch*scale_up), interpolation=cv2.INTER_CUBIC)
    else:
        crop_up = crop

    ocr_results = []
    for eng_name, eng_obj in engine.ocr_engines.items():
        try:
            text, conf = engine._run_ocr(eng_name, eng_obj, crop_up)
            if text:
                cleaned = validator.clean_ocr_text(text)
                ok, final = validator.validate(cleaned)
                ocr_results.append((eng_name, text, conf, cleaned, ok, final))
        except Exception as e:
            ocr_results.append((eng_name, f"ERROR:{e}", 0, "", False, ""))

    print(f"\n[{m:02d}:{s:02d}] 프레임{frame_no}  크롭={cw}x{ch}px  YOLO={yconf:.2f}")
    if not ocr_results:
        print("  OCR 엔진 없음")
    for eng_name, raw_text, oconf, cleaned, ok, final in ocr_results:
        status = f"✅ {final}" if ok else "❌ 검증실패"
        print(f"  [{eng_name}] raw='{raw_text}'  conf={oconf:.2f}  clean='{cleaned}'  → {status}")
    
    # 최종 결과
    valid_results = [(f, c) for _, _, c, _, ok, f in ocr_results if ok]
    if valid_results:
        best = max(valid_results, key=lambda x: x[1])
        results_table.append({"time": f"{m:02d}:{s:02d}", "plate": best[0], "conf": best[1], "yolo": yconf})

print("\n" + "=" * 60)
print("최종 인식 결과 요약")
print("=" * 60)
if results_table:
    print(f"{'시각':8} {'번호판':15} {'신뢰도':8} {'YOLO':6}")
    print("-" * 45)
    for r in results_table:
        print(f"{r['time']:8} {r['plate']:15} {r['conf']:.2f}     {r['yolo']:.2f}")
else:
    print("인식된 번호판 없음 → 패턴 검증 전부 실패")
print(f"\n크롭 이미지 위치: {OUT_DIR}")
