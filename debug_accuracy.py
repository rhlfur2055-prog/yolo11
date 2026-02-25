# -*- coding: utf-8 -*-
"""12개 번호판 이미지 단계별 인식 디버그"""
import os, sys, cv2, numpy as np, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

IMG_DIR = r'C:\tool\yolo26-main\22'
GROUND_TRUTH = {
    '01나8060.png': '01나8060',
    '02누2754.png': '02누2754',
    '14니3234.png': '14니3234',
    '36다7117.png': '36다7117',
    '48보7062.png': '48보7062',
    '55저9392.png': '55저9392',
    '58두9599.png': '58두9599',
    '70버6393.png': '70버6393',
    '80부5915.png': '80부5915',
    '경기76바7789.png': '경기76바7789',
    '서울바9203.png': '서울바9203',
    '트럭 경기91바6286.png': '경기91바6286',
}

def imread_unicode(fpath):
    """한글 경로 대응 imread"""
    buf = np.fromfile(fpath, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def main():
    # --- 엔진 로드 ---
    print("=" * 70)
    print("  12장 번호판 단계별 디버그 테스트")
    print("=" * 70)

    from plate_engine_pro import PlateEnginePro, PlateValidator
    engine = PlateEnginePro()
    engine.consecutive_required = 1
    engine.config.DETECT_CONF = 0.05
    engine.config.OCR_CONF = 0.15
    engine.config.CONSECUTIVE_FRAMES_REQUIRED = 1
    validator = engine.validator

    from plate_ocr_postfilter_v2 import clean_ocr_text_v2

    exact = 0
    digit_ok = 0
    total = 0

    for fname, answer in GROUND_TRUTH.items():
        fpath = os.path.join(IMG_DIR, fname)
        img = imread_unicode(fpath)
        if img is None:
            print(f"\n  [SKIP] {fname} - 로드 실패")
            continue
        total += 1
        h, w = img.shape[:2]

        print(f"\n{'─'*70}")
        print(f"  [{total}] {fname}  ({w}x{h})  정답: {answer}")
        print(f"{'─'*70}")

        # 1) YOLO 탐지
        detections = engine.model(img, conf=0.05, verbose=False)
        boxes = []
        for det in detections:
            for box in det.boxes:
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                boxes.append((x1,y1,x2,y2,conf))
        print(f"  YOLO 탐지: {len(boxes)}개 박스")
        for i, (x1,y1,x2,y2,c) in enumerate(boxes):
            print(f"    box[{i}]: ({x1},{y1})-({x2},{y2}) conf={c:.2f} size={x2-x1}x{y2-y1}")

        # 2) 각 박스에서 OCR 실행 (raw + 교정)
        best_text = ""
        best_conf = 0.0
        for i, (x1,y1,x2,y2,det_conf) in enumerate(boxes):
            # 마진 추가
            margin = 5
            cx1 = max(0, x1 - margin)
            cy1 = max(0, y1 - margin)
            cx2 = min(w, x2 + margin)
            cy2 = min(h, y2 + margin)
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # CLAHE 전처리
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            # 업스케일
            ch, cw = crop.shape[:2]
            if cw < 200:
                scale = 200 / cw
                crop_up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                enhanced_up = cv2.resize(enhanced_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                crop_up = crop
                enhanced_up = enhanced_bgr

            # 각 엔진별 OCR
            for engine_name, eng in engine.ocr_engines.items():
                for label, test_img in [("원본", crop_up), ("CLAHE", enhanced_up)]:
                    raw_text, raw_conf = engine._run_ocr(engine_name, eng, test_img)
                    cleaned = validator.clean_ocr_text(raw_text)
                    v2_cleaned = clean_ocr_text_v2(raw_text)
                    if raw_text.strip():
                        print(f"    box[{i}] {engine_name:10} [{label}] raw='{raw_text}' conf={raw_conf:.2f} → cleaned='{cleaned}' v2='{v2_cleaned}'")
                    if cleaned and raw_conf > best_conf:
                        best_conf = raw_conf
                        best_text = cleaned

        # 3) 엔진 전체 처리 (process_frame)
        results = engine.process_frame(img, camera_id="TEST", use_multiframe=False) or []
        engine_best = ""
        engine_conf = 0.0
        for r in results:
            t = (r.get("plate") or r.get("text") or "").strip()
            c = float(r.get("confidence", r.get("ocr_confidence", 0)))
            if c > engine_conf and t:
                engine_conf = c
                engine_best = t

        # 비교
        digits_answer = "".join(c for c in answer if c.isdigit())
        digits_result = "".join(c for c in engine_best if c.isdigit())
        hangul_answer = "".join(c for c in answer if '\uac00' <= c <= '\ud7a3')
        hangul_result = "".join(c for c in engine_best if '\uac00' <= c <= '\ud7a3')

        is_exact = answer == engine_best
        is_digit = digits_answer and digits_answer == digits_result and len(digits_answer) >= 4
        is_hangul = hangul_answer == hangul_result

        if is_exact:
            exact += 1
            mark = "EXACT ✓"
        elif is_digit:
            digit_ok += 1
            mark = "DIGIT_OK"
        else:
            mark = "FAIL ✗"

        print(f"  >>> 엔진 최종: '{engine_best}' (conf={engine_conf:.2f})")
        print(f"  >>> 정답: '{answer}'")
        print(f"  >>> 판정: {mark}  숫자={'OK' if is_digit else 'NG'}  한글={'OK' if is_hangul else f'NG({hangul_answer}→{hangul_result})'}")

    print(f"\n{'='*70}")
    print(f"  종합 결과: 완전일치={exact}/{total} ({exact/total*100:.0f}%)")
    print(f"            숫자일치={exact+digit_ok}/{total} ({(exact+digit_ok)/total*100:.0f}%)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
