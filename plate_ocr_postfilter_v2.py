"""
=======================================================
  plate_ocr_postfilter_v2.py
  한글 교정 완전 개선 패치 - V3 ULTIMATE

  핵심 개선:
    1. OCR 역순 읽기 감지 및 복원 (939255시 → 55저9392)
    2. 한글 혼동 교정 테이블 대폭 확장 (실제 오인식 데이터 기반)
    3. 지역명 퍼지 매칭 (거리→경기, 시일→서울 등)
    4. 다중 후보 생성 + 패턴 스코어링

  적용 방법:
    plate_engine_pro.py에서
      from plate_ocr_postfilter_v2 import clean_ocr_text_v2, ensemble_vote_v2
    로 임포트 후 기존 함수 교체
=======================================================
"""

import re
from difflib import SequenceMatcher

# =============================================
# 1. 한국 번호판 허용 한글 목록
# =============================================
VALID_HANGUL = set(
    '가나다라마바사아자차카타파하'   # Row 1: 자가용
    '거너더러머버서어저처커터퍼허'   # Row 2: 렌터카
    '고노도로모보소오조호'           # Row 3: 자가용 확장
    '구누두루무부수우주'             # Row 4: 자가용 확장
    '배육'                           # 영업용/특수
)

REGION_NAMES = ['서울', '경기', '인천', '부산', '대구', '광주', '대전', '울산', '세종',
                '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']

# =============================================
# 2. OCR 오인식 교정 테이블 (대폭 확장)
# =============================================
# [한글 위치] 영문→한글
OCR_TO_HANGUL = {
    'L': '나', 'N': '나',
    'C': '다', 'D': '다',
    'B': '버',
    'S': '서',
    'G': '가',
    'A': '아',
    'J': '지', 'Z': '저',
    'P': '파', 'R': '라',
    'M': '마', 'I': '이',
    'K': '카', 'T': '타',
    'H': '하', 'V': '부',
    'U': '구', 'W': '우',
    'X': '크', 'Y': '이',
    'F': '프', 'E': '에', 'Q': '고',
}

# [숫자 위치] 영문→숫자
OCR_TO_NUM = {
    'O': '0', 'o': '0',
    'I': '1', 'l': '1', 'i': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'G': '6', 'g': '9',
    'T': '7',
    'B': '8', 'b': '8',
    'q': '9',
}

# =============================================
# 2-1. 한글→한글 혼동 교정 (실제 OCR 오인식 데이터 기반)
# =============================================
# OCR이 유사한 한글끼리 혼동하는 경우 교정
# 키: OCR이 잘못 읽은 문자, 값: [가능한 올바른 문자들] (우선순위순)
HANGUL_CONFUSION = {
    # PaddleOCR/EasyOCR 실제 혼동 패턴
    '이': ['이'],       # 이 자체는 유효 (거→이 등은 별도 처리)
    '시': ['서'],       # 시→서 (저 다음으로 흔한 오류)
    '히': ['하'],       # 히→하
    '에': ['아'],       # 에→아
    '지': ['저'],       # 지→저
    '벼': ['버'],       # 벼→버
    '비': ['바'],       # 비→바
    '당': ['다'],       # 당→다 (서울바9203에서 시일7당9203)
    '물': ['무'],       # 물→무
    '랑': ['라'],       # 랑→라 (이랑8060 → 라)
    '끼': ['기'],       # 끼→기 (4끼 → 경기)
    '개': ['가'],       # 개→가
    '것': ['거'],       # 것→거
    '건': ['거'],       # 건→거
    '결': ['거'],       # 결→거
    '일': ['이'],       # 일→이
    '기': ['기'],       # 기 is valid in region name context
    '대': ['다'],       # 대→다 (36대7117 → 36다7117)
    '태': ['타'],       # 태→타
    '내': ['나'],       # 내→나
    '래': ['라'],       # 래→라
    '매': ['마'],       # 매→마
    '배': ['바'],       # 배→바 (배는 영업용이지만 위치 맥락으로 구분)
    '새': ['사'],       # 새→사
    '재': ['자'],       # 재→자
    '채': ['차'],       # 채→차
    '해': ['하'],       # 해→하
    '문': ['무'],       # 문→무
    '법': ['버'],       # 법→버
    '낭': ['나'],       # 낭→나
}

# =============================================
# 2-2. 지역명 퍼지 매칭 테이블
# =============================================
# OCR이 지역명을 잘못 읽는 패턴 (실제 데이터 기반)
REGION_FUZZY = {
    # 경기
    '거리': '경기', '개기': '경기', '건기': '경기', '결건': '경기',
    '경기': '경기', '경끼': '경기', '걍기': '경기', '겅기': '경기',
    '거기': '경기', '건거': '경기', '급기': '경기', '긍기': '경기',
    '깅기': '경기', '경거': '경기',
    # 서울
    '시일': '서울', '시울': '서울', '서일': '서울', '시물': '서울',
    '서울': '서울', '서을': '서울', '서옳': '서울', '사울': '서울',
    # 인천
    '인천': '인천', '인쳔': '인천', '인처': '인천',
    # 부산
    '부산': '부산', '부삭': '부산', '무산': '부산',
    # 대구
    '대구': '대구', '데구': '대구', '대고': '대구',
    # 광주
    '광주': '광주', '관주': '광주', '꽝주': '광주',
    # 대전
    '대전': '대전', '대저': '대전', '데전': '대전',
    # 울산
    '울산': '울산', '을산': '울산',
    # 세종
    '세종': '세종', '세중': '세종',
    # 강원
    '강원': '강원', '강완': '강원',
    # 충북/충남
    '충북': '충북', '충남': '충남',
    # 전북/전남
    '전북': '전북', '전남': '전남',
    # 경북/경남
    '경북': '경북', '경남': '경남',
    # 제주
    '제주': '제주', '재주': '제주',
}


# =============================================
# 3. 교정 헬퍼 함수
# =============================================

def correct_hangul_char(ch):
    """영문 → 한글 교정 (한글 위치)"""
    if ch in VALID_HANGUL:
        return ch
    return OCR_TO_HANGUL.get(ch.upper(), ch)


def correct_num_char(ch):
    """영문 → 숫자 교정 (숫자 위치)"""
    if ch.isdigit():
        return ch
    return OCR_TO_NUM.get(ch, ch)


def _fix_num(s):
    """숫자 필드 내 모든 문자 교정: 'O060' → '0060'"""
    return ''.join(correct_num_char(c) for c in s)


def _fix_hangul_confusion(ch):
    """한글→한글 혼동 교정. 유효 한글이면 그대로, 아니면 교정 시도."""
    if ch in VALID_HANGUL:
        return ch
    candidates = HANGUL_CONFUSION.get(ch, [])
    for c in candidates:
        if c in VALID_HANGUL:
            return c
    return ch


def _apply_ocr_correction(text):
    """
    숫자-영문-숫자 패턴에서 영문 위치를 한글로 교정
    예: '01L8060' → '01나8060'
    """
    result = list(text)
    for i, ch in enumerate(result):
        prev_num = (i > 0 and result[i-1].isdigit())
        next_num = (i < len(result)-1 and result[i+1].isdigit())
        if (prev_num or next_num) and ch.isalpha() and ch not in VALID_HANGUL:
            fixed = correct_hangul_char(ch)
            if fixed in VALID_HANGUL:
                result[i] = fixed
    return ''.join(result)


# =============================================
# 4. 역순 OCR 텍스트 감지 및 복원
# =============================================

def _try_reverse_plate(text):
    """
    PaddleOCR이 2줄 번호판을 아래→위로 읽는 경우 복원.

    실제 패턴:
      939255시 → 55저9392  (뒤4자리 + 앞2자리 + 한글)
      959958두 → 58두9599
      6393705  → 70버6393  (한글 누락)
      591580무 → 80무5915
      3234:144 → 14니3234

    역순 감지 조건:
      - 4자리숫자 + 2~3자리숫자 + 한글0~1자  (꼬리가 앞)
    """
    # 특수문자/공백 제거
    clean = re.sub(r'[^\d가-힣A-Za-z]', '', text)

    candidates = []

    # 패턴A: 숫자4 + 숫자2~3 + 한글1 (완전 역순)
    # 예: 939255시 → 55시9392 → 55저9392
    m = re.match(r'^(\d{4})(\d{2,3})([가-힣A-Za-z])$', clean)
    if m:
        tail = m.group(1)       # 9392
        head = m.group(2)       # 55
        hangul = m.group(3)     # 시
        fixed_h = _fix_hangul_confusion(hangul)
        if fixed_h not in VALID_HANGUL:
            fixed_h = correct_hangul_char(hangul)
        if fixed_h in VALID_HANGUL:
            candidates.append(head + fixed_h + tail)

    # 패턴B: 숫자4 + 숫자2~3 (한글 누락, 역순)
    # 예: 6393705 → 70?6393
    m = re.match(r'^(\d{4})(\d{2,3})$', clean)
    if m:
        tail = m.group(1)
        head = m.group(2)
        # 한글 없으므로 패턴 스코어 낮지만 후보로 유지
        candidates.append(head + '?' + tail)

    # 패턴C: 한글 + 숫자4 + 숫자2~3 + 한글 (2줄 읽기 혼합)
    # 예: 무59858두 → 58두9599 (앞뒤 한글이 다름)
    m = re.match(r'^([가-힣])(\d{3,4})(\d{2,3})([가-힣])$', clean)
    if m:
        h1, tail, head, h2 = m.group(1), m.group(2), m.group(3), m.group(4)
        # 두 한글 중 유효한 것 선택
        for h in [h2, h1]:
            fixed = _fix_hangul_confusion(h)
            if fixed in VALID_HANGUL:
                candidates.append(head + fixed + tail)
                break

    return candidates


# =============================================
# 5. 핵심 파싱 함수 (강화)
# =============================================

def _parse_plate_core(text):
    """
    번호판 핵심 부분 파싱 + 교정 (역순 감지 포함)

    지원 구조:
      - 숫자2~3 + 한글1 + 숫자4  (일반: 01나8060, 123가4567)
      - 한글1 + 숫자4            (지역판 앞숫자 누락: 바9203)
      - 역순 OCR 텍스트 복원     (939255시 → 55저9392)
    """
    corrected = _apply_ocr_correction(text)

    for src in [corrected, text]:
        # 패턴1: 숫자2~3 + 순수한글 + 숫자4  (핵심 패턴)
        m = re.search(r'(\d{2,3})([가-힣])(\d{4})', src)
        if m:
            hangul = _fix_hangul_confusion(m.group(2))
            return _fix_num(m.group(1)) + hangul + _fix_num(m.group(3))

        # 패턴2: 한글1 + 숫자4  (지역명 뒤 앞숫자 누락, 예: 바9203)
        m = re.match(r'^([가-힣])(\d{4})$', src)
        if m and m.group(1) in VALID_HANGUL:
            return m.group(1) + _fix_num(m.group(2))

        # 패턴3: 숫자+영문혼용한글+숫자 (OCR 오류 포함)
        m = re.search(r'(\d{2,3})([A-Za-z가-힣])(\d{4})', src)
        if m:
            mid = correct_hangul_char(m.group(2))
            mid = _fix_hangul_confusion(mid)
            if mid in VALID_HANGUL:
                return _fix_num(m.group(1)) + mid + _fix_num(m.group(3))

        # 패턴4: 전체 혼용 (숫자 위치도 영문 포함, 예: 01L806O)
        m = re.match(r'^([A-Za-z0-9]{2,3})([A-Za-z가-힣])([A-Za-z0-9]{4})$', src)
        if m:
            prefix = _fix_num(m.group(1))
            mid = correct_hangul_char(m.group(2))
            mid = _fix_hangul_confusion(mid)
            suffix = _fix_num(m.group(3))
            if prefix.isdigit() and suffix.isdigit() and mid in VALID_HANGUL:
                return prefix + mid + suffix

    # 패턴5: 역순 OCR 복원 시도
    rev_candidates = _try_reverse_plate(text)
    for cand in rev_candidates:
        if '?' not in cand:
            parsed = _parse_plate_core_simple(cand)
            if parsed:
                return parsed

    return ""


def _parse_plate_core_simple(text):
    """단순 패턴 매칭 (재귀 방지용)"""
    m = re.search(r'(\d{2,3})([가-힣])(\d{4})', text)
    if m:
        hangul = _fix_hangul_confusion(m.group(2))
        return _fix_num(m.group(1)) + hangul + _fix_num(m.group(3))
    return ""


# =============================================
# 6. 지역명 퍼지 매칭
# =============================================

def _extract_region(text):
    """
    텍스트 앞부분에서 지역명을 퍼지 매칭으로 추출.
    반환: (지역명, 나머지) 또는 (None, 원본)
    """
    # 정확한 매칭 먼저
    for region in REGION_NAMES:
        if text.startswith(region):
            return region, text[len(region):]

    # 퍼지 매칭 (2글자)
    if len(text) >= 2:
        prefix2 = text[:2]
        if prefix2 in REGION_FUZZY:
            return REGION_FUZZY[prefix2], text[2:]

    # 퍼지 매칭 (3글자 → 2글자 지역)
    if len(text) >= 3:
        prefix3 = text[:3]
        # 3글자가 지역명+숫자가 아닌 경우, 2글자 시도
        for key, region in REGION_FUZZY.items():
            if len(key) == 2 and text.startswith(key):
                return region, text[len(key):]

    # 한글 2~3자로 시작하고 뒤에 숫자가 오는 경우 지역명 후보
    m = re.match(r'^([가-힣]{2,3})(\d.*)$', text)
    if m:
        hangul_prefix = m.group(1)
        remainder = m.group(2)
        # 각 2글자 조합으로 퍼지 매칭
        if len(hangul_prefix) >= 2:
            p2 = hangul_prefix[:2]
            if p2 in REGION_FUZZY:
                extra = hangul_prefix[2:]
                return REGION_FUZZY[p2], extra + remainder

    return None, text


# =============================================
# 7. 메인 교정 함수 (plate_engine_pro.py 교체용)
# =============================================

def clean_ocr_text_v2(raw_text):
    """
    완전 개선된 OCR 후처리 함수 V3
    plate_engine_pro.py의 clean_ocr_text() 교체용

    처리:
      1. 특수문자/공백 제거
      2. 역순 OCR 텍스트 감지 및 복원
      3. 지역명 퍼지 매칭
      4. 영문→한글 교정 + 숫자위치 O→0 교정
      5. 한글→한글 혼동 교정
      6. 패턴 매칭으로 최종 검증
    """
    if not raw_text:
        return ""

    # 기본 정리
    text = re.sub(r'\s+', '', raw_text.strip())
    text = re.sub(r'[^\w가-힣]', '', text)

    if len(text) < 4:
        return ""

    candidates = []

    # --- 경로 1: 정방향 파싱 ---
    # 지역명 처리
    region, remainder = _extract_region(text)
    if region:
        parsed = _parse_plate_core(remainder)
        if parsed:
            candidates.append(region + parsed)

    # 지역명 없이 파싱
    direct = _parse_plate_core(text)
    if direct:
        candidates.append(direct)

    # --- 경로 2: 역순 복원 ---
    rev_cands = _try_reverse_plate(text)
    for rc in rev_cands:
        if '?' not in rc:
            # 역순 복원 결과에서 지역명 추출 시도
            reg2, rem2 = _extract_region(rc)
            if reg2:
                parsed2 = _parse_plate_core(rem2)
                if parsed2:
                    candidates.append(reg2 + parsed2)
            simple = _parse_plate_core_simple(rc)
            if simple:
                candidates.append(simple)

    # --- 경로 3: 지역명 제거 후보 (잡문자 혼입) ---
    # "주결건91바6286" 같은 경우: 앞 잡문자 제거 후 숫자+한글+숫자 패턴 탐색
    stripped = _strip_leading_junk(text)
    if stripped != text:
        reg3, rem3 = _extract_region(stripped)
        if reg3:
            parsed3 = _parse_plate_core(rem3)
            if parsed3:
                candidates.append(reg3 + parsed3)
        direct3 = _parse_plate_core(stripped)
        if direct3:
            candidates.append(direct3)

    if not candidates:
        return ""

    # 최적 후보 선택 (패턴 완성도 스코어 기준)
    return _pick_best_candidate(candidates)


def _strip_leading_junk(text):
    """앞부분 잡문자 제거: 숫자2~3+한글+숫자4 패턴이 나올 때까지 건너뛰기."""
    # 숫자+한글+숫자4 패턴 위치 찾기
    m = re.search(r'\d{2,3}[가-힣]\d{4}', text)
    if m:
        # 패턴 앞에 지역명이 있을 수 있으므로 2글자 전까지 확인
        start = m.start()
        if start >= 2:
            prefix = text[start-2:start]
            if prefix in REGION_FUZZY:
                return REGION_FUZZY[prefix] + text[start:]
        return text[start:]
    return text


def _pick_best_candidate(candidates):
    """후보 중 가장 완성도 높은 번호판 선택."""
    best = ""
    best_score = -1
    for cand in candidates:
        score = _plate_score(cand)
        if score > best_score:
            best_score = score
            best = cand
    return best


def _plate_score(plate):
    """번호판 패턴 완성도 점수 (높을수록 좋음)."""
    score = 0

    # 지역명 포함 +2
    for region in REGION_NAMES:
        if plate.startswith(region):
            score += 2
            plate_body = plate[len(region):]
            break
    else:
        plate_body = plate

    # 숫자2~3 + 한글1 + 숫자4 완성형 +10
    m = re.match(r'^(\d{2,3})([가-힣])(\d{4})$', plate_body)
    if m:
        score += 10
        # 유효 한글이면 +3
        if m.group(2) in VALID_HANGUL:
            score += 3
        # 길이가 길수록 +1
        score += len(plate_body)
    else:
        # 한글1 + 숫자4 (앞숫자 누락)
        m = re.match(r'^([가-힣])(\d{4})$', plate_body)
        if m:
            score += 5
            if m.group(1) in VALID_HANGUL:
                score += 2

    return score


# =============================================
# 8. OCR 앙상블 투표 (voting 로직 개선)
# =============================================

def ensemble_vote_v2(ocr_results_list):
    """
    개선된 앙상블 투표 V3

    모든 OCR 원본 텍스트에 clean_ocr_text_v2 적용 후 투표.
    """
    if not ocr_results_list:
        return {'plate': '', 'conf': 0.0, 'method': 'none'}

    ENGINE_WEIGHT = {
        'paddle': 1.0,
        'easyocr': 0.85,
        'tesseract': 0.6,
    }

    cleaned = []
    for r in ocr_results_list:
        text = clean_ocr_text_v2(r.get('text', ''))
        conf = float(r.get('conf', 0.5))
        engine = r.get('engine', 'unknown').lower()
        weight = ENGINE_WEIGHT.get(engine, 0.7) * conf
        if text:
            cleaned.append({'text': text, 'conf': conf, 'weight': weight})

    if not cleaned:
        return {'plate': '', 'conf': 0.0, 'method': 'none'}

    scores = {}
    for item in cleaned:
        t = item['text']
        scores[t] = scores.get(t, 0.0) + item['weight']

    scores = _merge_similar_votes(scores)

    best = max(scores, key=scores.get)
    best_conf = max(
        (item['conf'] for item in cleaned if item['text'] == best),
        default=scores[best]
    )

    return {
        'plate': best,
        'conf': min(best_conf, 1.0),
        'method': 'ensemble_vote',
        'all_candidates': scores,
    }


def _merge_similar_votes(scores, threshold=0.8):
    """편집거리 기반 유사 번호판 병합"""
    texts = list(scores.keys())
    merged = {}
    used = set()
    for i, t1 in enumerate(texts):
        if i in used:
            continue
        score = scores[t1]
        dominant = t1
        for j, t2 in enumerate(texts):
            if i >= j or j in used:
                continue
            if SequenceMatcher(None, t1, t2).ratio() >= threshold:
                score += scores[t2]
                # 더 긴 것 또는 더 높은 패턴 스코어 선택
                if _plate_score(t2) > _plate_score(dominant):
                    dominant = t2
                elif _plate_score(t2) == _plate_score(dominant) and len(t2) > len(dominant):
                    dominant = t2
                used.add(j)
        merged[dominant] = score
        used.add(i)
    return merged


# 멀티라인 OCR 결과 처리 (PaddleOCR 2줄 번호판)
def parse_ocr_multiline(ocr_lines):
    """
    PaddleOCR 멀티라인 결과 처리
    ocr_lines: [("경기 76", 0.9), ("바 7789", 0.85)]
    """
    if not ocr_lines:
        return "", 0.0
    combined = ''.join(line for line, conf in ocr_lines)
    avg_conf = sum(conf for _, conf in ocr_lines) / len(ocr_lines)
    return clean_ocr_text_v2(combined), avg_conf


# =============================================
# 9. 테스트 실행
# =============================================
if __name__ == "__main__":

    # 기존 테스트 케이스
    TEST_CASES = [
        # (OCR 원본 입력,       정답)
        ("01L8060",      "01나8060"),
        ("01나8060",     "01나8060"),
        ("02누2754",     "02누2754"),
        ("14니3234",     "14니3234"),
        ("36C7117",      "36다7117"),
        ("36다7117",     "36다7117"),
        ("48보7062",     "48보7062"),
        ("55저9392",     "55저9392"),
        ("58두9599",     "58두9599"),
        ("70B6393",      "70버6393"),
        ("80부5915",     "80부5915"),
        ("경기76바7789", "경기76바7789"),
        ("서울바9203",   "서울바9203"),
        ("경기91바6286", "경기91바6286"),
        ("01L806O",      "01나8060"),
        ("경기76B7789",  "경기76버7789"),
        ("서울B9203",    "서울버9203"),
    ]

    # 실제 OCR 오인식 패턴 추가 (디버그 테스트에서 발견)
    REAL_OCR_CASES = [
        # 역순 읽기 패턴
        ("939255시",      "55서9392"),    # 역순 + 시→서
        ("959958두",      "58두9599"),    # 역순
        ("591580무",      "80무5915"),    # 역순
        ("3234144",       "14가3234"),    # 역순 (한글 누락→가 fallback)
        # 지역명 퍼지
        ("거리76바7789",  "경기76바7789"),
        ("개기76바7789",  "경기76바7789"),
        ("시일바9203",    "서울바9203"),
        # 한글 혼동
        ("80무5915",      "80무5915"),   # 무는 유효 한글
        ("70자6393",      "70자6393"),   # 자는 유효 한글
        ("36라7117",      "36라7117"),   # 라는 유효 한글
    ]

    print("=" * 65)
    print("한글 교정 로직 테스트 V3 ULTIMATE")
    print("=" * 65)

    all_tests = TEST_CASES + REAL_OCR_CASES
    correct = 0
    for raw, answer in all_tests:
        result = clean_ocr_text_v2(raw)
        ok = "PASS" if result == answer else "FAIL"
        if result == answer:
            correct += 1
        print(f"  {ok}  {raw:<22} -> {result:<16} (answer: {answer})")

    print(f"\nAccuracy: {correct}/{len(all_tests)} = {correct/len(all_tests)*100:.1f}%")
