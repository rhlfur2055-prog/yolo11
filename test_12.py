# -*- coding: utf-8 -*-
"""22/ 디렉토리 12장 정확도 테스트"""
import sys, os, io, re, time
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import cv2
import numpy as np
from yolo11_plate.plate_engine_pro import PlateEnginePro

def extract_gt(filename):
    name = os.path.splitext(filename)[0]
    # "트럭 경기91바6286" → "경기91바6286"
    if " " in name:
        name = name.split(" ", 1)[1]
    return name.strip()

def normalize(text):
    return re.sub(r"[\s\-\.\,]", "", text).upper()

def main():
    img_dir = "22"
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"=== 12장 정확도 테스트 ({len(files)}장) ===\n")

    engine = PlateEnginePro()

    correct = 0
    for fname in files:
        gt = extract_gt(fname)
        gt_norm = normalize(gt)

        fpath = os.path.join(img_dir, fname)
        img_data = np.fromfile(fpath, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"{fname} → 이미지 로드 실패")
            continue

        # 프레임 스킵 캐시 무효화 (독립 이미지이므로)
        engine._frame_counter = 0
        engine._cached_results = None

        t0 = time.perf_counter()
        results = engine.process_frame(img, "CAM01")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if results:
            plate = results[0]["plate"]
            conf = results[0]["confidence"]
        else:
            plate = ""
            conf = 0.0

        plate_norm = normalize(plate)
        # 마지막 4자리 숫자 비교도 허용
        gt_digits = re.findall(r"\d+", gt_norm)
        plate_digits = re.findall(r"\d+", plate_norm)
        gt_last4 = gt_digits[-1][-4:] if gt_digits else ""
        plate_last4 = plate_digits[-1][-4:] if plate_digits else ""

        match = (plate_norm == gt_norm) or (gt_last4 and plate_last4 and gt_last4 == plate_last4 and len(plate) >= 4)
        if match:
            correct += 1
            mark = "OK"
        else:
            mark = "FAIL"

        print(f"{fname} → {plate or '(없음)'} ({elapsed_ms:.0f}ms) [{mark}]  정답={gt}")

    print(f"\n=== 결과: {correct}/{len(files)} ===")

if __name__ == "__main__":
    main()
