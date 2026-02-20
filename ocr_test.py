# -*- coding: utf-8 -*-
import easyocr, json, zipfile, os, sys, cv2, numpy as np, re, csv
sys.stdout.reconfigure(encoding='utf-8')
os.environ['HOME'] = r'C:\tools\yolo26'

from paddleocr import PaddleOCR

img_dir = r'aihub_data\images'
vzip = r'103.자동차_차종-연식-번호판_인식용_데이터\추가_데이터_보완_건_211229(폴더구조수정)\2.Validation\라벨링데이터\자동차번호판OCR_validation.zip'

def imread_kr(fpath):
    with open(fpath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def reassemble_plate(ocr_texts):
    combined = ''.join(ocr_texts).replace(' ', '')
    for ch in "'\"._-,!()":
        combined = combined.replace(ch, '')

    # 패턴1: 지역+숫자2~3+한글1+숫자4
    m = re.search(r'([가-힣]{2})(\d{2,3})([가-힣])(\d{4})', combined)
    if m:
        return m.group(0)

    # 패턴2: 숫자2~3+한글1+숫자4
    m = re.search(r'(\d{2,3})([가-힣])(\d{4})', combined)
    if m:
        return m.group(0)

    # 패턴3: 뒤집힌 순서 보정
    nums = re.findall(r'\d+', combined)
    kors = re.findall(r'[가-힣]', combined)
    if len(nums) >= 2 and kors:
        d4 = [n for n in nums if len(n) == 4]
        d23 = [n for n in nums if len(n) in (2, 3)]
        if d4 and d23:
            return d23[0] + kors[0] + d4[0]

    return combined

# ===== 라벨 로드 =====
labels = {}
with zipfile.ZipFile(vzip) as z:
    for n in z.namelist():
        if n.endswith('.json'):
            with z.open(n) as f:
                data = json.load(f)
                labels[data['imagePath']] = data['value']

selected = [
    '경기37바5577-5.jpg', '경기37바1945.jpg', '경기37바2355.jpg',
    '경기37바1621.jpg', '경기37바5193-2.jpg', '경기37바1945-4.jpg',
    '경기37바5587-4.jpg', '경기37바1745-2.jpg', '경기37바6041-2.jpg',
    '경기37바5294-2.jpg',
]

print('Loading EasyOCR...', flush=True)
easy_reader = easyocr.Reader(['ko', 'en'], gpu=True)

print('Loading PaddleOCR...', flush=True)
paddle_reader = PaddleOCR(
    use_angle_cls=True, lang='korean', show_log=False,
    det_model_dir=r'C:\tools\yolo26\.paddleocr\det',
    rec_model_dir=r'C:\tools\yolo26\.paddleocr\rec',
    cls_model_dir=r'C:\tools\yolo26\.paddleocr\cls',
)
print('Both loaded.\n', flush=True)

results = []
for i, fname in enumerate(selected, 1):
    fpath = os.path.join(img_dir, fname)
    gt = labels.get(fname, 'N/A')
    img = imread_kr(fpath)
    h, w = img.shape[:2]

    # --- EasyOCR ---
    easy_raw = easy_reader.readtext(img)
    easy_texts = [r[1] for r in easy_raw]
    easy_combined = ' '.join(easy_texts)
    easy_conf = max((r[2] for r in easy_raw), default=0)
    easy_reassembled = reassemble_plate(easy_texts)

    # --- PaddleOCR ---
    paddle_raw = paddle_reader.ocr(img, cls=True)
    paddle_texts = []
    paddle_conf = 0
    if paddle_raw and paddle_raw[0]:
        for line in paddle_raw[0]:
            paddle_texts.append(line[1][0])
            paddle_conf = max(paddle_conf, line[1][1])
    paddle_combined = ' '.join(paddle_texts)
    paddle_reassembled = reassemble_plate(paddle_texts)

    # --- 매칭 ---
    gt_clean = gt.replace(' ', '')
    easy_exact = 'O' if gt_clean == easy_reassembled.replace(' ', '') else 'X'
    paddle_exact = 'O' if gt_clean == paddle_reassembled.replace(' ', '') else 'X'

    gt_num4 = re.search(r'\d{4}', gt_clean)
    e_num4 = re.search(r'\d{4}', easy_reassembled)
    p_num4 = re.search(r'\d{4}', paddle_reassembled)
    easy_partial = 'O' if gt_num4 and e_num4 and gt_num4.group() == e_num4.group() else 'X'
    paddle_partial = 'O' if gt_num4 and p_num4 and gt_num4.group() == p_num4.group() else 'X'

    row = dict(
        no=i, file=fname, resolution="{0}x{1}".format(w, h), gt=gt,
        easy_raw=easy_combined, easy_reassembled=easy_reassembled,
        easy_conf="{:.2f}".format(easy_conf), easy_exact=easy_exact, easy_partial=easy_partial,
        paddle_raw=paddle_combined, paddle_reassembled=paddle_reassembled,
        paddle_conf="{:.2f}".format(paddle_conf), paddle_exact=paddle_exact, paddle_partial=paddle_partial,
    )
    results.append(row)

    print("[{:>2}/10] {}".format(i, fname), flush=True)
    print("  GT:     {}".format(gt), flush=True)
    print("  Easy:   {:30s} -> {:20s} [{}] (num4:{})".format(
        easy_combined, easy_reassembled, easy_exact, easy_partial), flush=True)
    print("  Paddle: {:30s} -> {:20s} [{}] (num4:{})".format(
        paddle_combined, paddle_reassembled, paddle_exact, paddle_partial), flush=True)

# ===== CSV 저장 =====
csv_path = 'aihub_data/ocr_comparison.csv'
with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

# ===== 집계 =====
e_ex = sum(1 for r in results if r['easy_exact'] == 'O')
p_ex = sum(1 for r in results if r['paddle_exact'] == 'O')
e_pt = sum(1 for r in results if r['easy_partial'] == 'O')
p_pt = sum(1 for r in results if r['paddle_partial'] == 'O')

sep = '=' * 50
print("\n" + sep, flush=True)
print("              EasyOCR    PaddleOCR", flush=True)
print("완전일치:      {}/10       {}/10".format(e_ex, p_ex), flush=True)
print("숫자4자리:     {}/10       {}/10".format(e_pt, p_pt), flush=True)
print(sep, flush=True)
print("CSV: {}".format(csv_path), flush=True)
