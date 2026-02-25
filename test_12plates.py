# -*- coding: utf-8 -*-
"""12장 스크린샷으로 번호판 인식 정확도 테스트"""
import os
import sys
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 정답 매핑 (파일명 → 정답 번호판)
GROUND_TRUTH = {
    "01나8060.png": "01나8060",
    "02누2754.png": "02누2754",
    "14니3234.png": "14니3234",
    "36다7117.png": "36다7117",
    "48보7062.png": "48보7062",
    "55저9392.png": "55저9392",
    "58두9599.png": "58두9599",
    "70버6393.png": "70버6393",
    "80부5915.png": "80부5915",
    "경기76바7789.png": "경기76바7789",
    "서울바9203.png": "서울바9203",
    "트럭 경기91바6286.png": "경기91바6286",
}

IMG_DIR = r"C:\Users\jomg2\OneDrive\바탕 화면\22"


def test_with_engine_pro():
    """plate_engine_pro.py의 PlateEnginePro로 테스트"""
    print("=" * 60)
    print("  [테스트 1] PlateEnginePro (실시간 서버 엔진)")
    print("=" * 60)
    try:
        from plate_engine_pro import PlateEnginePro
        engine = PlateEnginePro()
        engine.consecutive_required = 1
        engine.config.DETECT_CONF = 0.05
        engine.config.OCR_CONF = 0.20
        engine.config.CONSECUTIVE_FRAMES_REQUIRED = 1
    except Exception as e:
        print(f"  엔진 로드 실패: {e}")
        return 0, 0

    correct = 0
    total = 0
    for fname, answer in GROUND_TRUTH.items():
        fpath = os.path.join(IMG_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP] {fname} 파일 없음")
            continue
        img = cv2.imread(fpath)
        if img is None:
            continue
        total += 1

        # 엔진으로 인식
        results = engine.process_frame(img, camera_id="TEST") or []
        # 고해상도 원본도 전달
        if not results:
            # 리사이즈 후 재시도
            h0, w0 = img.shape[:2]
            if w0 > 640:
                scale = 640 / w0
                small = cv2.resize(img, (640, int(h0 * scale)))
            else:
                small = img
            sharp_k = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
            small_sharp = cv2.filter2D(small, -1, sharp_k)
            results = engine.process_frame(small_sharp, camera_id="TEST", full_frame=img) or []

        if not results:
            # CLAHE 시도
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            results = engine.process_frame(enhanced_bgr, camera_id="TEST") or []

        best_text = ""
        best_conf = 0.0
        for r in results:
            text = (r.get("plate") or r.get("text") or "").strip()
            conf = float(r.get("confidence", r.get("ocr_confidence", 0)))
            if conf > best_conf and text:
                best_conf = conf
                best_text = text

        # 정답 비교
        digits_answer = "".join(c for c in answer if c.isdigit())
        digits_result = "".join(c for c in best_text if c.isdigit())
        hangul_answer = "".join(c for c in answer if '\uac00' <= c <= '\ud7a3')
        hangul_result = "".join(c for c in best_text if '\uac00' <= c <= '\ud7a3')

        exact_match = answer == best_text
        digit_match = digits_answer and digits_answer == digits_result and len(digits_answer) >= 4
        hangul_match = hangul_answer and hangul_answer == hangul_result

        if exact_match:
            correct += 1
            mark = "EXACT"
        elif digit_match:
            mark = "DIGIT_OK"
        else:
            mark = "FAIL"

        hangul_mark = "한글OK" if hangul_match else f"한글X({hangul_answer}→{hangul_result})"
        print(f"  [{mark}] {fname}")
        print(f"        정답: {answer}")
        print(f"        인식: {best_text or '(없음)'} (conf={best_conf:.2f}) [{hangul_mark}]")

    print(f"\n  결과: 완전일치={correct}/{total} ({correct/total*100:.0f}%)" if total > 0 else "  파일 없음")
    return correct, total


def test_with_4k():
    """plate_recognition_4k.py의 PlateRecognizer로 테스트"""
    print("\n" + "=" * 60)
    print("  [테스트 2] PlateRecognizer (plate_recognition_4k)")
    print("=" * 60)
    try:
        from plate_recognition_4k import PlateRecognizer
        rec = PlateRecognizer()
    except Exception as e:
        print(f"  엔진 로드 실패: {e}")
        return 0, 0

    correct = 0
    total = 0
    for fname, answer in GROUND_TRUTH.items():
        fpath = os.path.join(IMG_DIR, fname)
        if not os.path.exists(fpath):
            continue
        img = cv2.imread(fpath)
        if img is None:
            continue
        total += 1

        results = rec.process_frame(img, frame_index=0)
        best_text = ""
        best_conf = 0.0
        for r in results:
            text = r.get("text", "").strip()
            conf = float(r.get("ocr_confidence", 0))
            if conf > best_conf and text:
                best_conf = conf
                best_text = text

        digits_answer = "".join(c for c in answer if c.isdigit())
        digits_result = "".join(c for c in best_text if c.isdigit())
        match = answer in best_text or best_text in answer or (
            digits_answer and digits_answer == digits_result and len(digits_answer) >= 4
        )
        mark = "OK" if match else "FAIL"
        if match:
            correct += 1
        print(f"  [{mark}] {fname}")
        print(f"        정답: {answer}")
        print(f"        인식: {best_text or '(없음)'} (conf={best_conf:.2f})")

    print(f"\n  결과: {correct}/{total} ({correct/total*100:.0f}%)" if total > 0 else "  파일 없음")
    return correct, total


if __name__ == "__main__":
    c1, t1 = test_with_engine_pro()
    c2, t2 = test_with_4k()
    print("\n" + "=" * 60)
    print(f"  종합: PlateEnginePro={c1}/{t1}, PlateRecognizer={c2}/{t2}")
    print("=" * 60)
