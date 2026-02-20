# -*- coding: utf-8 -*-
"""
개선판 벤치마크 실행기 v2.0
plate_recognition_4k.py의 개선된 OCR 파이프라인으로 AI Hub 10,000장 테스트

Usage:
  python run_benchmark_v2.py --n 100     # 100장 샘플
  python run_benchmark_v2.py --n 0       # 전체 10000장
"""
import os, sys, time, csv, re, json, zipfile, random
sys.stdout.reconfigure(encoding='utf-8')
os.environ['HOME'] = r'C:\tools\yolo26'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100, help='0=all')
parser.add_argument('--img-dir', default=r'aihub_data\images')
_LABEL_ZIP_DEFAULT = os.path.join(
    '103.\uc790\ub3d9\ucc28_\ucc28\uc885-\uc5f0\uc2dd-\ubc88\ud638\ud310_\uc778\uc2dd\uc6a9_\ub370\uc774\ud130',
    '\ucd94\uac00_\ub370\uc774\ud130_\ubcf4\uc644_\uac74_211229(\ud3f4\ub354\uad6c\uc870\uc218\uc815)',
    '2.Validation',
    '\ub77c\ubca8\ub9c1\ub370\uc774\ud130',
    '\uc790\ub3d9\ucc28\ubc88\ud638\ud310OCR_validation.zip',
)
parser.add_argument('--label-zip', default=_LABEL_ZIP_DEFAULT)
parser.add_argument('--output-dir', default='aihub_results')
args = parser.parse_args()

import numpy as np, cv2

# plate_recognition_4k.py 에서 개선된 함수 임포트
from plate_recognition_4k import (
    validate_korean_plate, validate_plate_format,
    correct_ocr_hangul, correct_hangul_similarity,
    _HANGUL_PLATE_CORRECTION, _VALID_PLATE_HANGUL_ALL,
    _correct_region, _find_region_in_text, _correct_single_hangul,
    _DIGIT_CORRECTION, _REGION_SET,
)

os.makedirs(args.output_dir, exist_ok=True)


def imread_kr(fpath):
    with open(fpath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_labels(vzip_path):
    labels = {}
    with zipfile.ZipFile(vzip_path) as z:
        for n in z.namelist():
            if n.endswith('.json'):
                with z.open(n) as f:
                    data = json.load(f)
                    labels[data['imagePath']] = data['value']
    return labels


def extract_num4(text):
    m = re.search(r'(\d{4})', text)
    return m.group(1) if m else ''


def preprocess_enhanced(img):
    """개선2: CLAHE(3.0, 4x4) + 샤프닝 컬러 전처리"""
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # LAB 컬러 CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # 샤프닝
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def deskew_plate(img):
    """개선2: HoughLines + minAreaRect 기울기 보정"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    angle = 0.0
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min(img.shape[1] // 3, 80))
    if lines is not None and len(lines) > 0:
        angles = []
        for rho, theta in lines[:20, 0]:
            deg = np.degrees(theta) - 90
            if abs(deg) < 20:
                angles.append(deg)
        if angles:
            angle = float(np.median(angles))
    if abs(angle) < 0.3:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(thresh)
        if coords is not None and len(coords) > 10:
            rect = cv2.minAreaRect(coords)
            (_, _), (w_r, h_r), rect_angle = rect
            if w_r < h_r:
                rect_angle = rect_angle + 90
            if abs(rect_angle) > 45:
                rect_angle = rect_angle - 90 if rect_angle > 0 else rect_angle + 90
            angle = rect_angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def upscale_if_small(img):
    """개선2: LANCZOS4 업스케일"""
    h, w = img.shape[:2]
    short_side = min(h, w)
    if short_side < 100:
        scale = max(6, 100 / short_side)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif w < 300:
        img = cv2.resize(img, (w * 6, h * 6), interpolation=cv2.INTER_LANCZOS4)
    return img


def reassemble_plate(ocr_entries):
    """개선4: bbox y중심 정렬 + 패턴 재조합 + 한글 교정"""
    if not ocr_entries:
        return '', 0.0

    entries_with_y = []
    for text, bbox, conf in ocr_entries:
        y_center = 0.0
        if bbox is not None:
            try:
                if isinstance(bbox[0], (list, tuple)):
                    y_center = sum(p[1] for p in bbox) / len(bbox)
                elif len(bbox) >= 4:
                    y_center = (bbox[1] + bbox[3]) / 2
            except (IndexError, TypeError):
                pass
        entries_with_y.append((text, bbox, conf, y_center))

    entries_with_y.sort(key=lambda x: x[3])
    all_text = ''.join(e[0] for e in entries_with_y)
    avg_conf = sum(e[2] for e in entries_with_y) / len(entries_with_y) if entries_with_y else 0

    cleaned = re.sub(r'[^가-힣0-9a-zA-Z]', '', all_text)
    cleaned = ''.join(_DIGIT_CORRECTION.get(c, c) for c in cleaned)

    # 패턴A: 지역명+숫자2~3+한글1+숫자4
    m = re.search(r'([가-힣]{2})(\d{2,3})([가-힣])(\d{4})', cleaned)
    if m:
        region, nf, usage, nb = m.groups()
        region = _correct_region(region)
        usage = _correct_single_hangul(usage)
        return region + nf + usage + nb, avg_conf

    # 패턴B: 숫자2~3+한글1+숫자4
    m = re.search(r'(\d{2,3})([가-힣])(\d{4})', cleaned)
    if m:
        nf, usage, nb = m.groups()
        usage = _correct_single_hangul(usage)
        prefix = cleaned[:m.start()]
        region = _find_region_in_text(prefix)
        if region:
            return region + nf + usage + nb, avg_conf
        return nf + usage + nb, avg_conf

    # 패턴C: 2줄 번호판
    if len(entries_with_y) >= 2:
        y_values = [e[3] for e in entries_with_y]
        y_mid = (min(y_values) + max(y_values)) / 2
        upper = ''.join(re.sub(r'[^가-힣0-9a-zA-Z]', '', e[0]) for e in entries_with_y if e[3] < y_mid)
        lower = ''.join(re.sub(r'[^가-힣0-9a-zA-Z]', '', e[0]) for e in entries_with_y if e[3] >= y_mid)
        upper = ''.join(_DIGIT_CORRECTION.get(c, c) for c in upper)
        lower = ''.join(_DIGIT_CORRECTION.get(c, c) for c in lower)
        lower_nums = re.findall(r'\d{4}', lower)
        upper_nums = re.findall(r'\d{2,3}', upper)
        upper_hangul = re.findall(r'[가-힣]', upper)
        if lower_nums and upper_nums and upper_hangul:
            usage = _correct_single_hangul(upper_hangul[-1])
            region = _find_region_in_text(upper)
            base = upper_nums[0] + usage + lower_nums[0]
            if region:
                base = region + base
            return base, avg_conf

    # 패턴D: 조각 재조합
    all_nums = re.findall(r'\d+', cleaned)
    all_hangul = re.findall(r'[가-힣]', cleaned)
    if all_nums and all_hangul:
        d4 = [n for n in all_nums if len(n) == 4]
        d23 = [n for n in all_nums if len(n) in (2, 3)]
        if d4 and d23:
            usage = _correct_single_hangul(all_hangul[-1] if len(all_hangul) == 1 else all_hangul[0])
            return d23[0] + usage + d4[0], avg_conf

    return cleaned, avg_conf


def run_paddle(reader, img):
    try:
        results = reader.ocr(img, cls=True)
        entries = []
        if results and results[0]:
            for line in results[0]:
                entries.append((line[1][0], line[0], line[1][1]))
        return entries
    except Exception:
        return []


def run_easy(reader, img):
    try:
        results = reader.readtext(img, text_threshold=0.3, low_text=0.3)
        entries = []
        for bbox, text, conf in results:
            entries.append((text, bbox, conf))
        return entries
    except Exception:
        return []


def postprocess(text, conf):
    """한글 교정 + 패턴 검증 + 형식 교정"""
    if not text or len(text) < 4:
        return text, conf, False, 0.0
    cleaned = re.sub(r"[^가-힣0-9A-Za-z]", "", text)
    corrected = correct_ocr_hangul(cleaned)
    corrected = correct_hangul_similarity(corrected)
    is_valid, normalized, score = validate_korean_plate(corrected)
    if normalized and len(normalized) >= 4:
        fmt_corrected, fmt_score = validate_plate_format(normalized)
        if fmt_score > 0:
            normalized = fmt_corrected
            score = max(score, fmt_score)
            is_valid = True
        has_hangul = any('\uac00' <= c <= '\ud7a3' for c in normalized)
        if has_hangul:
            score = max(score, 0.90)
        return normalized, conf, is_valid or has_hangul, score
    return text, conf, False, 0.0


def preprocess_binary(img):
    """v1 스타일 이진화 전처리 (원본 파이프라인과 동일)"""
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) < 400:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def is_valid_plate(text):
    """v1 호환 엄격 유효성 검증"""
    if re.fullmatch(r'[가-힣]{2}\d{2,3}[가-힣]\d{4}', text):
        return True
    if re.fullmatch(r'\d{2,3}[가-힣]\d{4}', text):
        return True
    return False


def ensemble_ocr_v2(easy_reader, paddle_reader, img_original, img_enhanced):
    """개선3: v1 선택 로직 + 개선된 교정 테이블 + 다중 전처리"""
    candidates = []

    # v1 스타일 이진화 전처리 (가장 효과적)
    img_prep = preprocess_binary(img_original)

    def _process_candidate(method_name, entries):
        """OCR 결과 → 재조합 + 교정 → 후보"""
        if not entries:
            return
        text, conf = reassemble_plate(entries)
        if not text:
            return
        # 개선4: 확장된 교정 테이블 적용
        cleaned = re.sub(r"[^가-힣0-9A-Za-z]", "", text)
        corrected = correct_ocr_hangul(cleaned)
        corrected = correct_hangul_similarity(corrected)
        _, normalized, _ = validate_korean_plate(corrected)
        if not normalized:
            normalized = corrected
        fmt, _ = validate_plate_format(normalized)
        candidates.append((method_name, fmt, conf, is_valid_plate(fmt)))

    # 4 후보 (v1과 동일 구조)
    _process_candidate('paddle_orig', run_paddle(paddle_reader, img_original))
    if img_prep is not None:
        _process_candidate('paddle_prep', run_paddle(paddle_reader, img_prep))
    _process_candidate('easy_orig', run_easy(easy_reader, img_original))
    if img_prep is not None:
        _process_candidate('easy_prep', run_easy(easy_reader, img_prep))

    if not candidates:
        return '', 0.0, 'none'

    # ── v1 선택 로직: 유효 패턴 우선 → 신뢰도 ──
    valid = [c for c in candidates if c[3]]
    if valid:
        valid.sort(key=lambda x: -x[2])  # confidence 기준
        best = valid[0]
        return best[1], best[2], best[0]

    # 유효 없으면: 가장 긴 결과 + 높은 신뢰도
    candidates.sort(key=lambda x: (-len(x[1]), -x[2]))
    best = candidates[0]
    return best[1], best[2], best[0]


# ===== 메인 실행 =====

print("Loading labels...", flush=True)
labels = load_labels(args.label_zip)

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
error_hangul = {}
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

    # 전처리: 소형 이미지만 업스케일 (대형 이미지는 원본 유지)
    h_img, w_img = img.shape[:2]
    if min(h_img, w_img) < 80:
        img_processed = upscale_if_small(img)
    else:
        img_processed = img
    img_enhanced = preprocess_enhanced(img_processed)

    plate_text, conf, method = ensemble_ocr_v2(easy_reader, paddle_reader, img_processed, img_enhanced)

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

    if exact == 'X':
        gt_hangul = re.findall(r'[가-힣]', gt_clean)
        ocr_hangul = re.findall(r'[가-힣]', plate_text)
        for gh in gt_hangul:
            if gh not in ''.join(ocr_hangul):
                error_hangul[gh] = error_hangul.get(gh, 0) + 1

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

csv_name = 'ocr_benchmark_v2_{}.csv'.format(args.n if args.n > 0 else 'full')
csv_path = os.path.join(args.output_dir, csv_name)
with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

total = len(results)
exact_ok = sum(1 for r in results if r['exact_match'] == 'O')
body_ok = sum(1 for r in results if r.get('body_match') == 'O')
num4_ok = sum(1 for r in results if r['num4_match'] == 'O')

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
print("  [개선판 v2.0 벤치마크 결과]", flush=True)
print("  총 이미지:     {}".format(total), flush=True)
print("  소요시간:      {:.1f}s ({:.2f}s/img)".format(elapsed, elapsed/total), flush=True)
print("  완전일치:      {}/{} ({:.1f}%)".format(exact_ok, total, exact_ok/total*100), flush=True)
print("  본문일치:      {}/{} ({:.1f}%)  (지역명 제외)".format(body_ok, total, body_ok/total*100), flush=True)
print("  숫자4자리:     {}/{} ({:.1f}%)".format(num4_ok, total, num4_ok/total*100), flush=True)
print(sep, flush=True)

print("\n--- 메서드별 성능 ---", flush=True)
for m, s in sorted(method_stats.items(), key=lambda x: -x[1]['total']):
    print("  {:<20} total:{:>4}  exact:{:.1f}%  body:{:.1f}%".format(
        m, s['total'],
        s['exact']/s['total']*100 if s['total'] else 0,
        s['body']/s['total']*100 if s['total'] else 0,
    ), flush=True)

if error_hangul:
    print("\n--- 한글 오류 패턴 TOP 10 ---", flush=True)
    for ch, cnt in sorted(error_hangul.items(), key=lambda x: -x[1])[:10]:
        print("  '{}' 인식 실패: {}회".format(ch, cnt), flush=True)

errors = [r for r in results if r['exact_match'] == 'X' and r.get('body_match') == 'X']
if errors:
    print("\n--- 본문도 틀린 오류 (최대 15건) ---", flush=True)
    for r in errors[:15]:
        print("  GT:{:<16} OCR:{:<16} ({})".format(
            r['gt'], r['ocr_result'], r.get('method', '')), flush=True)

print("\nCSV: {}".format(csv_path), flush=True)
