# -*- coding: utf-8 -*-
"""
배치 벤치마크 실행기
Usage:
  python run_benchmark.py --n 100     # 100장 샘플
  python run_benchmark.py --n 0       # 전체 10000장
"""
import os, sys, time, csv, re, json, zipfile, random
sys.stdout.reconfigure(encoding='utf-8')
os.environ['HOME'] = r'C:\tools\yolo26'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100, help='0=all')
parser.add_argument('--img-dir', default=r'aihub_data\images')
parser.add_argument('--label-zip',
                    default=r'103.자동차_차종-연식-번호판_인식용_데이터\추가_데이터_보완_건_211229(폴더구조수정)\2.Validation\라벨링데이터\자동차번호판OCR_validation.zip')
parser.add_argument('--output-dir', default='aihub_data')
args = parser.parse_args()

# Import pipeline
from plate_ocr_pipeline import (
    load_labels, imread_kr, preprocess_plate, ensemble_ocr,
    extract_num4, is_valid_plate, correct_hangul, correct_region,
    find_region_in_text, REGION_SET
)

import numpy as np, cv2

os.makedirs(args.output_dir, exist_ok=True)

print("Loading labels...", flush=True)
labels = load_labels(args.label_zip)

# 이미지 목록
all_images = sorted([f for f in os.listdir(args.img_dir) if f.endswith('.jpg')])
if args.n > 0 and args.n < len(all_images):
    random.seed(42)
    all_images = random.sample(all_images, args.n)
    all_images.sort()

print("Images: {} (sample={})".format(len(all_images), args.n if args.n > 0 else 'ALL'), flush=True)

print("Loading EasyOCR...", flush=True)
import easyocr
easy_reader = easyocr.Reader(['ko', 'en'], gpu=True)

print("Loading PaddleOCR...", flush=True)
from paddleocr import PaddleOCR
paddle_reader = PaddleOCR(
    use_angle_cls=True, lang='korean', show_log=False,
    det_model_dir=r'C:\tools\yolo26\.paddleocr\det',
    rec_model_dir=r'C:\tools\yolo26\.paddleocr\rec',
    cls_model_dir=r'C:\tools\yolo26\.paddleocr\cls',
)
print("Models loaded.\n", flush=True)

results = []
error_hangul = {}   # 한글 오류 패턴 분석
t0 = time.time()

for i, fname in enumerate(all_images, 1):
    fpath = os.path.join(args.img_dir, fname)
    gt = labels.get(fname, 'N/A')
    img = imread_kr(fpath)
    if img is None:
        results.append(dict(
            no=i, file=fname, gt=gt, ocr_result='READ_ERROR',
            confidence='0', method='none',
            exact_match='X', body_match='X', num4_match='X'
        ))
        continue

    h, w = img.shape[:2]
    img_prep = preprocess_plate(img)
    plate_text, conf, method = ensemble_ocr(easy_reader, paddle_reader, img, img_prep)

    gt_clean = gt.replace(' ', '')
    exact = 'O' if gt_clean == plate_text else 'X'
    gt_no_region = re.sub(r'^[가-힣]{2}', '', gt_clean)
    ocr_no_region = re.sub(r'^[가-힣]{2}', '', plate_text)
    body_match = 'O' if gt_no_region == ocr_no_region else 'X'
    gt_n4 = extract_num4(gt_clean)
    ocr_n4 = extract_num4(plate_text)
    partial = 'O' if gt_n4 and gt_n4 == ocr_n4 else 'X'

    row = dict(
        no=i, file=fname, resolution="{}x{}".format(w, h), gt=gt,
        ocr_result=plate_text, confidence="{:.3f}".format(conf),
        method=method, exact_match=exact, body_match=body_match, num4_match=partial,
    )
    results.append(row)

    # 한글 오류 분석: GT 한글 vs OCR 한글
    if exact == 'X':
        gt_hangul = re.findall(r'[가-힣]', gt_clean)
        ocr_hangul = re.findall(r'[가-힣]', plate_text)
        for gh in gt_hangul:
            if gh not in ''.join(ocr_hangul):
                error_hangul[gh] = error_hangul.get(gh, 0) + 1

    # 진행률 출력 (매 50개)
    if i % 50 == 0 or i == len(all_images):
        elapsed = time.time() - t0
        speed = elapsed / i
        eta = speed * (len(all_images) - i)
        e_cnt = sum(1 for r in results if r['exact_match'] == 'O')
        b_cnt = sum(1 for r in results if r.get('body_match') == 'O')
        n_cnt = sum(1 for r in results if r['num4_match'] == 'O')
        print("[{:>5}/{:>5}] exact:{:.1f}% body:{:.1f}% num4:{:.1f}% | {:.1f}s/img ETA:{:.0f}s".format(
            i, len(all_images),
            e_cnt/i*100, b_cnt/i*100, n_cnt/i*100,
            speed, eta
        ), flush=True)

elapsed = time.time() - t0

# CSV 저장
csv_name = 'ocr_benchmark_{}.csv'.format(args.n if args.n > 0 else 'full')
csv_path = os.path.join(args.output_dir, csv_name)
with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

# 최종 집계
total = len(results)
exact_ok = sum(1 for r in results if r['exact_match'] == 'O')
body_ok = sum(1 for r in results if r.get('body_match') == 'O')
num4_ok = sum(1 for r in results if r['num4_match'] == 'O')

# 메서드별 집계
method_stats = {}
for r in results:
    m = r.get('method', 'none')
    if m not in method_stats:
        method_stats[m] = {'total': 0, 'exact': 0, 'body': 0}
    method_stats[m]['total'] += 1
    if r['exact_match'] == 'O':
        method_stats[m]['exact'] += 1
    if r.get('body_match') == 'O':
        method_stats[m]['body'] += 1

sep = '=' * 60
print("\n" + sep, flush=True)
print("  총 이미지:     {}".format(total), flush=True)
print("  소요시간:      {:.1f}s ({:.2f}s/img)".format(elapsed, elapsed/total), flush=True)
print("  완전일치:      {}/{} ({:.1f}%)".format(exact_ok, total, exact_ok/total*100), flush=True)
print("  본문일치:      {}/{} ({:.1f}%)  (지역명 제외)".format(body_ok, total, body_ok/total*100), flush=True)
print("  숫자4자리:     {}/{} ({:.1f}%)".format(num4_ok, total, num4_ok/total*100), flush=True)
print(sep, flush=True)

print("\n--- 메서드별 성능 ---", flush=True)
for m, s in sorted(method_stats.items(), key=lambda x: -x[1]['total']):
    print("  {:<15} total:{:>4}  exact:{:.1f}%  body:{:.1f}%".format(
        m, s['total'],
        s['exact']/s['total']*100 if s['total'] else 0,
        s['body']/s['total']*100 if s['total'] else 0,
    ), flush=True)

if error_hangul:
    print("\n--- 한글 오류 패턴 TOP 10 ---", flush=True)
    for ch, cnt in sorted(error_hangul.items(), key=lambda x: -x[1])[:10]:
        print("  '{}' 인식 실패: {}회".format(ch, cnt), flush=True)

# 오류 샘플
errors = [r for r in results if r['exact_match'] == 'X' and r.get('body_match') == 'X']
if errors:
    print("\n--- 본문도 틀린 오류 (최대 15건) ---", flush=True)
    for r in errors[:15]:
        print("  GT:{:<16} OCR:{:<16} ({})".format(
            r['gt'], r['ocr_result'], r.get('method', '')), flush=True)

print("\nCSV: {}".format(csv_path), flush=True)
