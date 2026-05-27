# -*- coding: utf-8 -*-
"""
번호판 인식 정확도 벤치마크
aihub_data/images/ 파일명(=정답)과 OCR 결과를 비교하여 정확도 측정
"""
import os
import sys
import io
import re
import random
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import cv2
import numpy as np
from yolo11_plate.plate_engine_pro import ImagePreprocessor, PlateEngineConfig, PlateValidator

# ── OCR 엔진 로드 ──
paddle_ocr = None
_paddle_v34 = False
try:
    from paddleocr import PaddleOCR
    # v3.4.0+ 호환: use_textline_orientation
    try:
        paddle_ocr = PaddleOCR(lang="korean", use_textline_orientation=False)
        _paddle_v34 = True
    except (TypeError, ValueError):
        # 구버전 fallback
        paddle_kwargs = dict(lang="korean", use_angle_cls=True)
        _paddle_model_root = Path("C:/tools/paddleocr_models")
        if _paddle_model_root.exists():
            paddle_kwargs["det_model_dir"] = str(_paddle_model_root / "det/ml/Multilingual_PP-OCRv3_det_infer")
            paddle_kwargs["rec_model_dir"] = str(_paddle_model_root / "rec/korean/korean_PP-OCRv4_rec_infer")
            paddle_kwargs["cls_model_dir"] = str(_paddle_model_root / "cls/ch_ppocr_mobile_v2.0_cls_infer")
        paddle_ocr = PaddleOCR(**paddle_kwargs)
except Exception:
    pass



def extract_gt(filename: str) -> str:
    """파일명에서 정답 번호판 추출 (확장자, -2 등 제거)"""
    name = Path(filename).stem
    name = re.sub(r"-\d+$", "", name)  # 경기75바1067-2 → 경기75바1067
    return name.strip()


def ocr_image(img, preprocessor, validator, config):
    """19종 전처리 × PaddleOCR → 앙상블 투표"""
    from collections import Counter
    all_candidates = []  # [(cleaned_text, confidence), ...]

    for method in config.PREPROCESS_METHODS:
        try:
            if method == "original":
                processed = img.copy()
            else:
                proc_func = getattr(preprocessor, method, None)
                if proc_func is None:
                    continue
                processed = proc_func(img.copy())

            # PaddleOCR
            if paddle_ocr is not None:
                try:
                    if _paddle_v34:
                        # v3.4.0+: OCRResult 딕셔너리
                        result = paddle_ocr.ocr(processed)
                        if result and result[0]:
                            r = result[0]
                            if hasattr(r, '__getitem__') and 'rec_texts' in r:
                                texts = r['rec_texts']
                                confs = r['rec_scores']
                                text = "".join(texts)
                                conf = float(np.mean(confs)) if confs else 0
                                cleaned = validator.clean_ocr_text(text)
                                if conf > 0.3 and len(cleaned) >= 4:
                                    all_candidates.append((cleaned, conf))
                    else:
                        # 구버전: [[bbox, (text, conf)], ...]
                        result = paddle_ocr.ocr(processed, cls=False)
                        if result and result[0]:
                            texts = [line[1][0] for line in result[0]]
                            confs = [line[1][1] for line in result[0]]
                            text = "".join(texts)
                            conf = float(np.mean(confs))
                            cleaned = validator.clean_ocr_text(text)
                            if conf > 0.3 and len(cleaned) >= 4:
                                all_candidates.append((cleaned, conf))
                except Exception:
                    pass

        except Exception:
            continue

    if not all_candidates:
        return "", 0.0

    # 투표: 최다 득표 선택
    counter = Counter(t for t, _ in all_candidates)
    best_text = counter.most_common(1)[0][0]
    confs = [c for t, c in all_candidates if t == best_text]
    best_conf = sum(confs) / len(confs)
    return best_text, best_conf


def normalize_for_compare(text: str) -> str:
    """비교용 정규화: 공백/특수문자 제거, 대문자"""
    return re.sub(r"[\s\-\.\,]", "", text).upper()


def main():
    sample_n = 100
    if len(sys.argv) >= 2:
        try:
            sample_n = int(sys.argv[1])
        except ValueError:
            pass

    img_dir = ROOT / "aihub_data" / "images"
    all_files = [f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")]
    if not all_files:
        print("[에러] aihub_data/images/ 에 이미지가 없습니다.")
        return

    random.seed(42)
    sample = random.sample(all_files, min(sample_n, len(all_files)))

    preprocessor = ImagePreprocessor()
    validator = PlateValidator()
    config = PlateEngineConfig()

    ocr_names = []
    if paddle_ocr: ocr_names.append("PaddleOCR")

    print("=" * 60)
    print(f"번호판 인식 정확도 벤치마크 ({len(sample)}장 샘플)")
    print(f"전처리: {len(config.PREPROCESS_METHODS)}종, OCR: {'+'.join(ocr_names)} 앙상블")
    print("=" * 60)

    correct = 0
    partial = 0
    fail = 0
    errors = []
    t0 = time.perf_counter()

    for i, fpath in enumerate(sample):
        gt = extract_gt(fpath.name)
        gt_norm = normalize_for_compare(gt)

        # 한글 파일명 호환: numpy로 읽기
        img_data = np.fromfile(str(fpath), dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            errors.append((fpath.name, gt, "", "이미지 로드 실패"))
            fail += 1
            continue

        # 이미지가 작으면 2배 업스케일
        h, w = img.shape[:2]
        if w < 200:
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        pred, conf = ocr_image(img, preprocessor, validator, config)
        pred_norm = normalize_for_compare(pred)

        if pred_norm == gt_norm:
            correct += 1
            status = "OK"
        elif gt_norm in pred_norm or pred_norm in gt_norm:
            partial += 1
            status = "PARTIAL"
        else:
            fail += 1
            status = "FAIL"
            errors.append((fpath.name, gt, pred, f"conf={conf:.2f}"))

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            acc = correct / (i + 1) * 100
            print(f"  [{i+1}/{len(sample)}] 정확={correct} 부분={partial} 실패={fail} "
                  f"정확도={acc:.1f}% ({elapsed:.0f}초)", flush=True)

    elapsed = time.perf_counter() - t0
    total = len(sample)
    acc_exact = correct / total * 100
    acc_partial = (correct + partial) / total * 100

    print()
    print("=" * 60)
    print("결과")
    print("=" * 60)
    print(f"  샘플 수:      {total}")
    print(f"  완전 일치:    {correct} ({acc_exact:.1f}%)")
    print(f"  부분 일치:    {partial} ({partial/total*100:.1f}%)")
    print(f"  실패:         {fail} ({fail/total*100:.1f}%)")
    print(f"  정확도(완전): {acc_exact:.1f}%")
    print(f"  정확도(부분포함): {acc_partial:.1f}%")
    print(f"  소요 시간:    {elapsed:.1f}초 ({elapsed/total:.2f}초/장)")
    print("=" * 60)

    if errors:
        print()
        print(f"실패 목록 (상위 20건):")
        print("-" * 60)
        for fname, gt, pred, note in errors[:20]:
            print(f"  {fname}")
            print(f"    정답: {gt}  →  인식: {pred or '(없음)'}  ({note})")
    print()


if __name__ == "__main__":
    main()
