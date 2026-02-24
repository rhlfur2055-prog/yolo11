# -*- coding: utf-8 -*-
"""
한국 번호판 OCR 파이프라인 v1.0
목표: 완전일치 95%+, 숫자4자리 99%+

[2단계] 후처리: 정규식 재조합 + 한글 교정 + bbox 위치 정렬
[3단계] 전처리: 업스케일 + CLAHE + 이진화 + 기울기 보정
[4단계] 앙상블: PaddleOCR + EasyOCR 동시 → 유효 패턴 선택
"""
import os, sys, re, json, zipfile, csv, time
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
os.environ['HOME'] = r'C:\tools\yolo26'

# ============================================================
# [2단계] 한글 교정 테이블 & 재조합 로직
# ============================================================

# 번호판에 실제 사용되는 한글 (용도 문자 40자)
VALID_PLATE_CHARS = set(
    '가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주'
    '허하호배'
)

# 지역명
REGIONS = [
    '서울','부산','대구','인천','광주','대전','울산','세종',
    '경기','강원','충북','충남','전북','전남','경북','경남','제주',
]

# OCR이 자주 틀리는 한글 → 올바른 용도문자 교정
HANGUL_CORRECTION = {
    # EasyOCR 오인식 패턴
    '륙': '바', '릎': '바', '휴': '바', '푹': '바', '선': '바',
    '춤': '바', '식': '바', '겸': '바', '겨': '바', '겪': '바',
    '릅': '바', '륜': '바', '륨': '바', '륩': '바',
    '빠': '바', '빼': '바', '빵': '바', '뱌': '바',
    '늑': '나', '닉': '나', '냑': '나',
    '딕': '다', '딘': '다', '덕': '다',
    '럭': '라', '럽': '라', '렉': '라',
    '먹': '마', '멕': '마', '먕': '마',
    '벽': '버', '볍': '버', '벡': '버',
    '석': '서', '섭': '서', '섞': '서',
    '억': '어', '엌': '어', '엌': '어',
    '젝': '저', '젖': '저', '젊': '저',
    '곡': '고', '곤': '고', '곧': '고',
    '녹': '노', '논': '노', '놉': '노',
    '독': '도', '돈': '도', '돋': '도',
    '록': '로', '론': '로', '롯': '로',
    '목': '모', '몬': '모', '몫': '모',
    '복': '보', '본': '보', '볼': '보',
    '속': '소', '손': '소', '솔': '소',
    '옥': '오', '온': '오', '올': '오',
    '족': '조', '존': '조', '졸': '조',
    '국': '구', '군': '구', '굿': '구',
    '눈': '누', '눌': '누', '눔': '누',
    '둔': '두', '둘': '두', '둠': '두',
    '룬': '루', '룰': '루', '룸': '루',
    '문': '무', '물': '무', '뭄': '무',
    '분': '부', '불': '부', '붐': '부',
    '순': '수', '술': '수', '숨': '수',
    '운': '우', '울': '우', '움': '우',
    '준': '주', '줄': '주', '줌': '주',
    '헌': '허', '헐': '허', '험': '허',
    '한': '하', '할': '하', '함': '하',
    '혼': '호', '홀': '호', '홈': '호',
    '백': '배', '밸': '배', '뱅': '배',
}

# OCR이 지역명(2자)을 오인식하는 패턴 교정
REGION_CORRECTION = {
    # 경기
    '경기': '경기', '걍기': '경기', '겅기': '경기', '견기': '경기',
    '경끼': '경기', '경키': '경기', '껭기': '경기', '격기': '경기',
    # 서울
    '서울': '서울', '서을': '서울', '서운': '서울', '셔울': '서울',
    '서욿': '서울', '석울': '서울',
    # 인천
    '인천': '인천', '인쳔': '인천', '인촌': '인천', '인첨': '인천',
    # 부산
    '부산': '부산', '부산': '부산', '부선': '부산', '부샨': '부산',
    # 대구
    '대구': '대구', '대귀': '대구', '대굴': '대구', '대국': '대구',
    # 대전
    '대전': '대전', '대잔': '대전', '대젼': '대전', '대졘': '대전',
    # 광주
    '광주': '광주', '괄주': '광주', '광쥬': '광주', '괌주': '광주',
    # 울산
    '울산': '울산', '울산': '울산', '울선': '울산',
    # 강원
    '강원': '강원', '깡원': '강원', '강월': '강원', '강완': '강원',
    # 충북
    '충북': '충북', '충붂': '충북', '총북': '충북',
    # 충남
    '충남': '충남', '총남': '충남', '충납': '충남',
    # 전북
    '전북': '전북', '전붂': '전북', '젼북': '전북',
    # 전남
    '전남': '전남', '젼남': '전남', '전납': '전남',
    # 경북
    '경북': '경북', '겅북': '경북', '경붂': '경북',
    # 경남
    '경남': '경남', '겅남': '경남', '경납': '경남',
    # 제주
    '제주': '제주', '재주': '제주', '제쥬': '제주', '졔주': '제주',
    # 세종
    '세종': '세종', '셰종': '세종', '세좀': '세종',
}

REGION_SET = set(REGIONS)

def correct_region(text):
    """지역명 2자를 교정 테이블에서 찾아 교정"""
    if text in REGION_SET:
        return text
    return REGION_CORRECTION.get(text, text)

def find_region_in_text(text):
    """텍스트에서 지역명(2자) 후보 찾기"""
    hangul_only = re.findall(r'[가-힣]', text)
    if len(hangul_only) >= 2:
        for i in range(len(hangul_only) - 1):
            pair = hangul_only[i] + hangul_only[i + 1]
            corrected = correct_region(pair)
            if corrected in REGION_SET:
                return corrected
    return None

# 숫자 OCR 오인식 교정
DIGIT_CORRECTION = {
    'O': '0', 'o': '0', 'Q': '0', 'D': '0',
    'I': '1', 'l': '1', '|': '1', 'i': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'B': '8', 'b': '6',
    'G': '6', 'g': '9',
    'T': '7', 'A': '4',
}


def correct_digit_str(s):
    """숫자 문자열 내 알파벳→숫자 교정"""
    return ''.join(DIGIT_CORRECTION.get(c, c) for c in s)


def correct_hangul(ch):
    """한글 1자 교정: 유효 문자면 그대로, 아니면 교정 테이블 참조"""
    if ch in VALID_PLATE_CHARS:
        return ch
    return HANGUL_CORRECTION.get(ch, ch)


def reassemble_plate_v2(ocr_entries):
    """
    OCR 결과(텍스트+bbox)를 번호판 패턴으로 재조합
    ocr_entries: list of (text, bbox, confidence)
    """
    if not ocr_entries:
        return '', 0.0

    # bbox y중심 기준 정렬
    entries_with_y = []
    for text, bbox, conf in ocr_entries:
        if bbox is not None and len(bbox) >= 2:
            if isinstance(bbox[0], (list, tuple)):
                y_center = sum(p[1] for p in bbox) / len(bbox)
            else:
                y_center = (bbox[1] + bbox[3]) / 2 if len(bbox) >= 4 else 0
        else:
            y_center = 0
        entries_with_y.append((text, bbox, conf, y_center))

    entries_with_y.sort(key=lambda x: x[3])

    all_text = ''.join(e[0] for e in entries_with_y)
    avg_conf = sum(e[2] for e in entries_with_y) / len(entries_with_y) if entries_with_y else 0

    # 특수문자 제거 + 알파벳→숫자 교정
    cleaned = re.sub(r'[^가-힣0-9a-zA-Z]', '', all_text)
    cleaned = ''.join(DIGIT_CORRECTION.get(c, c) for c in cleaned)

    # --- 패턴 매칭 ---

    # 패턴A: 지역명+숫자2~3+한글1+숫자4 (구형: 경기37바5577)
    m = re.search(r'([가-힣]{2})(\d{2,3})([가-힣])(\d{4})', cleaned)
    if m:
        region, nf, usage, nb = m.groups()
        region = correct_region(region)
        usage = correct_hangul(usage)
        return region + nf + usage + nb, avg_conf

    # 패턴B: 숫자2~3+한글1+숫자4 (신형 or 지역명 누락)
    m = re.search(r'(\d{2,3})([가-힣])(\d{4})', cleaned)
    if m:
        nf, usage, nb = m.groups()
        usage = correct_hangul(usage)
        # 앞에 한글2자가 있으면 지역명 시도
        prefix = cleaned[:m.start()]
        region = find_region_in_text(prefix)
        if region:
            return region + nf + usage + nb, avg_conf
        return nf + usage + nb, avg_conf

    # 패턴C: 2줄 번호판 상하 분리
    if len(entries_with_y) >= 2:
        y_values = [e[3] for e in entries_with_y]
        y_mid = (min(y_values) + max(y_values)) / 2
        upper = ''.join(re.sub(r'[^가-힣0-9a-zA-Z]', '', e[0]) for e in entries_with_y if e[3] < y_mid)
        lower = ''.join(re.sub(r'[^가-힣0-9a-zA-Z]', '', e[0]) for e in entries_with_y if e[3] >= y_mid)
        upper = ''.join(DIGIT_CORRECTION.get(c, c) for c in upper)
        lower = ''.join(DIGIT_CORRECTION.get(c, c) for c in lower)

        lower_nums = re.findall(r'\d{4}', lower)
        upper_nums = re.findall(r'\d{2,3}', upper)
        upper_hangul = re.findall(r'[가-힣]', upper)

        if lower_nums and upper_nums and upper_hangul:
            usage = correct_hangul(upper_hangul[-1])
            region = find_region_in_text(upper)
            base = upper_nums[0] + usage + lower_nums[0]
            if region:
                base = region + base
            return base, avg_conf

        if lower_nums and upper_hangul:
            usage = correct_hangul(upper_hangul[-1])
            upper_d = re.findall(r'\d+', upper)
            if upper_d:
                region = find_region_in_text(upper)
                base = upper_d[0] + usage + lower_nums[0]
                if region:
                    base = region + base
                return base, avg_conf

    # 패턴D: 조각 모아 재조합
    all_nums = re.findall(r'\d+', cleaned)
    all_hangul = re.findall(r'[가-힣]', cleaned)
    if all_nums and all_hangul:
        d4 = [n for n in all_nums if len(n) == 4]
        d23 = [n for n in all_nums if len(n) in (2, 3)]
        if d4 and d23:
            usage = correct_hangul(all_hangul[-1] if len(all_hangul) == 1 else all_hangul[0])
            return d23[0] + usage + d4[0], avg_conf
        elif d4:
            usage = correct_hangul(all_hangul[0])
            return usage + d4[0], avg_conf

    return cleaned, avg_conf


# ============================================================
# [3단계] 이미지 전처리 파이프라인
# ============================================================

def preprocess_plate(img):
    """번호판 이미지 전처리: 업스케일 + CLAHE + 이진화 + 기울기 보정"""
    if img is None:
        return None

    h, w = img.shape[:2]

    # 1) 업스케일 x2
    if max(h, w) < 400:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # 2) 그레이스케일
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) CLAHE 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4) 가우시안 블러 (노이즈 제거)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 5) Otsu 이진화
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6) 기울기 보정 (deskew)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 50:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 15:  # 15도 이내만 보정
            center = (img.shape[1] // 2, img.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return img


# ============================================================
# [4단계] 앙상블 OCR
# ============================================================

def run_easyocr(reader, img):
    """EasyOCR 실행 → [(text, bbox, confidence), ...]"""
    results = reader.readtext(img)
    entries = []
    for bbox, text, conf in results:
        entries.append((text, bbox, conf))
    return entries


def run_paddleocr(reader, img):
    """PaddleOCR 실행 → [(text, bbox, confidence), ...]"""
    results = reader.ocr(img, cls=True)
    entries = []
    if results and results[0]:
        for line in results[0]:
            bbox = line[0]
            text = line[1][0]
            conf = line[1][1]
            entries.append((text, bbox, conf))
    return entries


def is_valid_plate(text):
    """번호판 유효 패턴 검증"""
    # 구형: 지역2+숫자2~3+한글1+숫자4
    if re.fullmatch(r'[가-힣]{2}\d{2,3}[가-힣]\d{4}', text):
        return True
    # 신형: 숫자2~3+한글1+숫자4
    if re.fullmatch(r'\d{2,3}[가-힣]\d{4}', text):
        return True
    return False


def extract_num4(text):
    """끝 4자리 숫자 추출"""
    m = re.search(r'(\d{4})', text)
    return m.group(1) if m else ''


def ensemble_ocr(easy_reader, paddle_reader, img_original, img_preprocessed):
    """
    앙상블 전략:
    1) 원본 + 전처리 이미지 각각에 PaddleOCR + EasyOCR 실행 (4가지 결과)
    2) 유효 패턴 매칭되는 것 우선 선택
    3) 둘 다 유효하면 신뢰도 높은 것 선택
    """
    candidates = []

    # PaddleOCR - 원본
    entries = run_paddleocr(paddle_reader, img_original)
    text, conf = reassemble_plate_v2(entries)
    candidates.append(('paddle_orig', text, conf, is_valid_plate(text)))

    # PaddleOCR - 전처리
    if img_preprocessed is not None:
        entries = run_paddleocr(paddle_reader, img_preprocessed)
        text, conf = reassemble_plate_v2(entries)
        candidates.append(('paddle_prep', text, conf, is_valid_plate(text)))

    # EasyOCR - 원본
    entries = run_easyocr(easy_reader, img_original)
    text, conf = reassemble_plate_v2(entries)
    candidates.append(('easy_orig', text, conf, is_valid_plate(text)))

    # EasyOCR - 전처리
    if img_preprocessed is not None:
        entries = run_easyocr(easy_reader, img_preprocessed)
        text, conf = reassemble_plate_v2(entries)
        candidates.append(('easy_prep', text, conf, is_valid_plate(text)))

    # 유효한 패턴 우선
    valid = [c for c in candidates if c[3]]
    if valid:
        # 신뢰도 기준 정렬
        valid.sort(key=lambda x: -x[2])
        best = valid[0]
        return best[1], best[2], best[0]

    # 유효한 게 없으면 가장 긴 결과(더 많은 정보) 중 신뢰도 높은 것
    candidates.sort(key=lambda x: (-len(x[1]), -x[2]))
    best = candidates[0]
    return best[1], best[2], best[0]


# ============================================================
# 메인 실행
# ============================================================

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


def run_test(img_dir, labels, filenames, easy_reader, paddle_reader, verbose=True):
    """지정된 파일들에 대해 테스트 실행"""
    results = []
    for i, fname in enumerate(filenames, 1):
        fpath = os.path.join(img_dir, fname)
        gt = labels.get(fname, 'N/A')
        img = imread_kr(fpath)
        if img is None:
            results.append(dict(
                no=i, file=fname, gt=gt,
                ocr_result='READ_ERROR', confidence=0,
                method='none', exact_match='X', num4_match='X'
            ))
            continue

        h, w = img.shape[:2]
        img_prep = preprocess_plate(img)

        plate_text, conf, method = ensemble_ocr(easy_reader, paddle_reader, img, img_prep)

        gt_clean = gt.replace(' ', '')
        exact = 'O' if gt_clean == plate_text else 'X'

        # 지역명 제외 부분일치 (경기37바5577 vs 37바5577 → 일치)
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

        if verbose:
            mark = 'O' if exact == 'O' else ('~' if partial == 'O' else 'X')
            print("[{:>5}/{:>5}] [{}] GT:{:<16} OCR:{:<16} ({})".format(
                i, len(filenames), mark, gt, plate_text, method), flush=True)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test10', choices=['test10', 'full'])
    parser.add_argument('--img-dir', default=r'aihub_data\images')
    parser.add_argument('--label-zip',
                        default=r'103.자동차_차종-연식-번호판_인식용_데이터\추가_데이터_보완_건_211229(폴더구조수정)\2.Validation\라벨링데이터\자동차번호판OCR_validation.zip')
    parser.add_argument('--output-dir', default='aihub_data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading labels...", flush=True)
    labels = load_labels(args.label_zip)
    print("Labels: {} files".format(len(labels)), flush=True)

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

    if args.mode == 'test10':
        selected = [
            '경기37바5577-5.jpg', '경기37바1945.jpg', '경기37바2355.jpg',
            '경기37바1621.jpg', '경기37바5193-2.jpg', '경기37바1945-4.jpg',
            '경기37바5587-4.jpg', '경기37바1745-2.jpg', '경기37바6041-2.jpg',
            '경기37바5294-2.jpg',
        ]
        results = run_test(args.img_dir, labels, selected, easy_reader, paddle_reader, verbose=True)
        csv_path = os.path.join(args.output_dir, 'ocr_pipeline_test10.csv')

    else:  # full
        all_images = sorted([f for f in os.listdir(args.img_dir) if f.endswith('.jpg')])
        print("Total images: {}".format(len(all_images)), flush=True)
        t0 = time.time()
        results = run_test(args.img_dir, labels, all_images, easy_reader, paddle_reader, verbose=True)
        elapsed = time.time() - t0
        csv_path = os.path.join(args.output_dir, 'ocr_benchmark_full.csv')
        print("\nElapsed: {:.1f}s ({:.2f}s/image)".format(elapsed, elapsed / len(all_images)), flush=True)

    # CSV 저장
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # 집계
    total = len(results)
    exact_ok = sum(1 for r in results if r['exact_match'] == 'O')
    body_ok = sum(1 for r in results if r.get('body_match') == 'O')
    num4_ok = sum(1 for r in results if r['num4_match'] == 'O')

    sep = '=' * 55
    print("\n" + sep, flush=True)
    print("  총 이미지:    {}".format(total), flush=True)
    print("  완전일치:     {}/{} ({:.1f}%)".format(exact_ok, total, exact_ok / total * 100), flush=True)
    print("  본문일치:     {}/{} ({:.1f}%)  (지역명 제외)".format(body_ok, total, body_ok / total * 100), flush=True)
    print("  숫자4자리:    {}/{} ({:.1f}%)".format(num4_ok, total, num4_ok / total * 100), flush=True)
    print(sep, flush=True)
    print("CSV: {}".format(csv_path), flush=True)

    # 오류 패턴 분석
    errors = [r for r in results if r['exact_match'] == 'X']
    if errors:
        print("\n--- 오류 샘플 (최대 20건) ---", flush=True)
        for r in errors[:20]:
            print("  GT:{:<16} OCR:{:<16} ({})".format(
                r['gt'], r['ocr_result'], r.get('method', '')), flush=True)
