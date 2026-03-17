# -*- coding: utf-8 -*-
"""
CMD 3: OCR 인식 워커 프로세스 (PaddleOCR 단독)
plate_engine_pro.py process_frame()에서 OCR 파이프라인 부분을 추출하여
별도 프로세스에서 실행. multiprocessing.Queue로 IPC 통신.

★ EasyOCR 완전 제거 — PaddleOCR + Tesseract(영문 fallback) 단독 구조
"""
import re
import time
import traceback
import numpy as np
import cv2
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Queue

from pipeline_common import CMD_OCR_STOP, CMD_FLUSH
from config import PathConfig

# plate_engine_pro.py에서 클래스/함수 import (코드 중복 방지)
from plate_engine_pro import (
    ImagePreprocessor,
    PlateValidator,
    HangulClassifier,
    PlateEngineConfig,
    _CRNNModel,
)

# ── OCR 엔진: PaddleOCR 단독 + Tesseract 영문 fallback ──
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except Exception:
    HAS_PADDLEOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

try:
    import torch as _torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. OCR 엔진 초기화 (PaddleOCR 단독)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _init_ocr_engines():
    """PaddleOCR + Tesseract + CRNN 모델 초기화 (워커 시작 시 1회)

    Returns:
        ocr_engines: dict {name: engine}
        crnn_model: _CRNNModel 또는 None
        hangul_clf: HangulClassifier
        kr_allowlist: str (허용 문자)
    """
    ocr_engines = {}

    # ── PaddleOCR (한국어 특화, 유일한 OCR 엔진) ──
    if HAS_PADDLEOCR:
        paddle_kwargs = dict(lang="korean", use_angle_cls=True, show_log=False, use_gpu=False)
        _paddle_model_root = PathConfig.paddle_model_dir()
        if _paddle_model_root.exists():
            _det = _paddle_model_root / "det/ml/Multilingual_PP-OCRv3_det_infer"
            _rec = _paddle_model_root / "rec/korean/korean_PP-OCRv4_rec_infer"
            _cls = _paddle_model_root / "cls/ch_ppocr_mobile_v2.0_cls_infer"
            if _det.exists():
                paddle_kwargs["det_model_dir"] = str(_det)
            if _rec.exists():
                paddle_kwargs["rec_model_dir"] = str(_rec)
            if _cls.exists():
                paddle_kwargs["cls_model_dir"] = str(_cls)
        try:
            ocr_engines["paddleocr"] = PaddleOCR(**paddle_kwargs)
        except TypeError:
            paddle_kwargs.pop("show_log", None)
            try:
                ocr_engines["paddleocr"] = PaddleOCR(**paddle_kwargs)
            except Exception as e:
                print(f"[CMD3-OCR] PaddleOCR 초기화 실패: {e}")
        except Exception as e:
            print(f"[CMD3-OCR] PaddleOCR 초기화 실패: {e}")

    # ── Tesseract (영문 fallback) ──
    if HAS_TESSERACT:
        try:
            _tess_langs = pytesseract.get_languages()
            if 'kor' in _tess_langs:
                ocr_engines["tesseract"] = "tesseract"
            else:
                print("[CMD3-OCR] Tesseract: 한국어 언어팩(kor) 없음 — 건너뜀")
        except Exception:
            ocr_engines["tesseract"] = "tesseract"

    # ── CRNN 모델 (학습된 번호판 전용) ──
    crnn_model = None
    _crnn_path = Path(__file__).resolve().parent / "plate_ocr_crnn.pth"
    if HAS_TORCH and _crnn_path.exists():
        try:
            crnn_model = _CRNNModel(str(_crnn_path))
            print(f"[CMD3-OCR] CRNN 모델 로드: {_crnn_path}")
        except Exception as e:
            print(f"[CMD3-OCR] CRNN 모델 로드 실패: {e}")

    # 한글 분류기
    hangul_clf = HangulClassifier()

    # 허용 문자 목록 (PaddleOCR allowlist용)
    kr_allowlist = (
        "0123456789"
        "가나다라마바사아자차카타파하"
        "거너더러머버서어저처커터퍼허"
        "고노도로모보소오조호"
        "구누두루무부수우주"
        "배육"
        "서울부산대구인천광주대전울산세종"
        "경기강원충북충남전북전남경북경남제주"
        "전기외교"
    )

    print(f"[CMD3-OCR] OCR 엔진: {list(ocr_engines.keys())} (PaddleOCR 단독)")
    return ocr_engines, crnn_model, hangul_clf, kr_allowlist


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. ROI 업스케일
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _upscale_roi(roi):
    """ROI를 500px 목표로 업스케일 + 선명화 + 흰색 패딩"""
    roi_h, roi_w = roi.shape[:2]
    target_w = 500
    if roi_w < target_w:
        scale = target_w / roi_w
        if roi_w < 60:
            scale = max(scale, 9.0)
        elif roi_w < 120:
            scale = max(scale, 4.0)
    else:
        scale = 1.0

    if scale > 1.0:
        _interp = cv2.INTER_LANCZOS4 if roi_w < 80 else cv2.INTER_CUBIC
        roi_for_ocr = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=_interp)
        if roi_w < 80:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        else:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        roi_for_ocr = cv2.filter2D(roi_for_ocr, -1, kernel)
    else:
        roi_for_ocr = roi

    pad = max(10, int(roi_for_ocr.shape[0] * 0.15))
    roi_for_ocr = cv2.copyMakeBorder(
        roi_for_ocr, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    return roi_for_ocr, scale, pad


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. 색상 검증 + 녹색 판별
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_color(roi):
    """ROI의 HSV 색상 분석 → 색상 페널티 + 녹색 번호판 여부"""
    color_conf_penalty = 1.0
    is_green_plate = False
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_avg = np.mean(hsv[:, :, 0])
        s_avg = np.mean(hsv[:, :, 1])
        v_avg = np.mean(hsv[:, :, 2])
        _is_white = s_avg < 50 and v_avg > 80
        _is_yellow = 15 < h_avg < 35 and s_avg > 60
        _is_green = 35 < h_avg < 85 and s_avg > 40
        _is_blue = 90 < h_avg < 130 and s_avg > 40
        if not (_is_white or _is_yellow or _is_green or _is_blue):
            color_conf_penalty = 0.85
    except Exception:
        pass

    try:
        hsv_det = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_det, (35, 40, 40), (85, 255, 255))
        is_green_plate = (np.sum(green_mask > 0) / green_mask.size) > 0.20
    except Exception:
        pass

    return color_conf_penalty, is_green_plate


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D. CRNN 모델 인식
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _run_crnn(crnn_model, roi, validator):
    """CRNN 모델로 원본 ROI 인식 → 유효 후보 리스트"""
    candidates = []
    if crnn_model is None:
        return candidates
    try:
        crnn_text, crnn_conf = crnn_model.recognize(roi)
        if crnn_text and crnn_conf > 0.3:
            cleaned = validator.clean_ocr_text(crnn_text)
            if validator.is_valid_length(cleaned):
                is_valid, final_text = validator.validate(cleaned)
                if is_valid:
                    candidates.append((final_text, crnn_conf))
    except Exception:
        pass
    return candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E. Extra crops 생성 (구형 2줄 번호판)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _make_extra_crops(roi_for_ocr, roi_h, roi_w, is_green_plate):
    """구형 2줄 번호판 상단/하단 크롭 + 녹색 번호판 추가 크롭"""
    extra_crops = []
    if roi_h <= roi_w * 0.45:
        return extra_crops

    top_crop = roi_for_ocr[:int(roi_for_ocr.shape[0] * 0.5), :]
    bot_crop = roi_for_ocr[int(roi_for_ocr.shape[0] * 0.4):, :]

    top_h, top_w = top_crop.shape[:2]
    if top_w < 500:
        sc_top = 500 / top_w
        top_crop = cv2.resize(top_crop, None, fx=sc_top, fy=sc_top, interpolation=cv2.INTER_CUBIC)

    top_inv = cv2.bitwise_not(top_crop)

    extra_crops = [("top", top_inv), ("bot", bot_crop)]

    if is_green_plate:
        _inv_lab = cv2.cvtColor(top_inv, cv2.COLOR_BGR2LAB)
        _l, _a, _b = cv2.split(_inv_lab)
        _clahe_s = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        _l = _clahe_s.apply(_l)
        _inv_clahe = cv2.cvtColor(cv2.merge([_l, _a, _b]), cv2.COLOR_LAB2BGR)
        extra_crops.append(("top", _inv_clahe))

        _inv_gray = cv2.cvtColor(top_inv, cv2.COLOR_BGR2GRAY)
        _clahe_g = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        _inv_enh = _clahe_g.apply(_inv_gray)
        _, _inv_bin = cv2.threshold(_inv_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        extra_crops.append(("top", cv2.cvtColor(_inv_bin, cv2.COLOR_GRAY2BGR)))

        _g_ch = top_crop[:, :, 1]
        _g_inv = cv2.bitwise_not(_g_ch)
        _, _g_bin = cv2.threshold(_g_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        extra_crops.append(("top", cv2.cvtColor(_g_bin, cv2.COLOR_GRAY2BGR)))

        top_hsv = cv2.cvtColor(top_crop, cv2.COLOR_BGR2HSV)
        _, _, top_v = cv2.split(top_hsv)
        _, _v_low = cv2.threshold(top_v, 100, 255, cv2.THRESH_BINARY)
        extra_crops.append(("top", cv2.cvtColor(_v_low, cv2.COLOR_GRAY2BGR)))

        extra_crops.append(("bot", cv2.bitwise_not(bot_crop)))

    return extra_crops


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# F. OCR 실행 (PaddleOCR + Tesseract만)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _run_ocr(engine_name, engine, image, kr_allowlist=None):
    """단일 OCR 엔진 실행 → (text, conf). PaddleOCR 또는 Tesseract."""
    try:
        if engine_name == "paddleocr":
            result = engine.ocr(image, cls=True)
            if result and result[0]:
                lines = sorted(result[0], key=lambda l: l[0][0][1])
                texts = [l[1][0] for l in lines]
                confs = [l[1][1] for l in lines]
                if texts:
                    return "".join(texts), float(np.mean(confs))
        elif engine_name == "tesseract":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            custom_config = r'--oem 3 --psm 7 -l kor+eng'
            text = pytesseract.image_to_string(gray, config=custom_config)
            text = text.strip().replace('\n', '').replace(' ', '')
            if text:
                try:
                    data = pytesseract.image_to_data(
                        gray, config=custom_config,
                        output_type=pytesseract.Output.DICT
                    )
                    confs = [int(c) for c in data['conf'] if int(c) > 0]
                    conf = float(np.mean(confs)) / 100.0 if confs else 0.5
                except Exception:
                    conf = 0.5
                return text, conf
    except Exception:
        pass
    return "", 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G. OCR 파이프라인: Tier1 + Tier2 (PaddleOCR 단독)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 구형 번호판 정규식 (컴파일 1회)
_RE_OLD_PLATE = re.compile(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$')


def _run_ocr_pipeline(roi_for_ocr, ocr_engines, preprocessor, validator,
                      is_green_plate, kr_allowlist):
    """Tier1 + Tier2 OCR 앙상블 (PaddleOCR 단독)"""
    _TIER1_METHODS = ["original", "clahe", "sharpen"]
    _TIER2_METHODS = ["_inverted", "bilateral", "auto_contrast"]

    all_candidates = []
    roi_inv = cv2.bitwise_not(roi_for_ocr)
    tier1_consensus = False
    _paddle_engine = ocr_engines.get("paddleocr")

    # ── Tier 1: PaddleOCR × 핵심 3개 전처리 ──
    if _paddle_engine:
        for method in _TIER1_METHODS:
            try:
                if method == "original":
                    processed = roi_for_ocr.copy()
                else:
                    proc_func = getattr(preprocessor, method, None)
                    if proc_func is None:
                        continue
                    processed = proc_func(roi_for_ocr.copy())
                text, ocr_conf = _run_ocr("paddleocr", _paddle_engine, processed)
                if not text or ocr_conf < 0.20:
                    continue
                cleaned = validator.clean_ocr_text(text)
                if not validator.is_valid_length(cleaned):
                    continue
                is_valid, final_text = validator.validate(cleaned)
                if not is_valid:
                    continue
                _v2_has_region = bool(_RE_OLD_PLATE.match(cleaned))
                weight = 4 if _v2_has_region else 1
                for _ in range(weight):
                    all_candidates.append((final_text, ocr_conf))

                # 매 method 후 즉시 합의 체크 → 조기 종료
                if len(all_candidates) >= 2:
                    _t1_counter = Counter(t for t, c in all_candidates)
                    _t1_top, _t1_cnt = _t1_counter.most_common(1)[0]
                    _t1_confs = [c for t, c in all_candidates if t == _t1_top]
                    if _t1_cnt >= 2 and float(np.mean(_t1_confs)) > 0.6:
                        tier1_consensus = True
                        break
            except Exception:
                continue

        if not tier1_consensus and all_candidates:
            _t1_counter = Counter(t for t, c in all_candidates)
            _t1_top, _t1_cnt = _t1_counter.most_common(1)[0]
            _t1_confs = [c for t, c in all_candidates if t == _t1_top]
            if _t1_cnt >= 2 and float(np.mean(_t1_confs)) > 0.6:
                tier1_consensus = True

    # ── Tier 2: PaddleOCR 확장 전처리 (합의 실패 시) ──
    mid_consensus = False
    if not tier1_consensus and _paddle_engine:
        for method in _TIER2_METHODS:
            try:
                if method == "_inverted":
                    processed = roi_inv.copy()
                else:
                    proc_func = getattr(preprocessor, method, None)
                    if proc_func is None:
                        continue
                    processed = proc_func(roi_for_ocr.copy())
                text, ocr_conf = _run_ocr("paddleocr", _paddle_engine, processed)
                if not text or ocr_conf < 0.20:
                    continue
                cleaned = validator.clean_ocr_text(text)
                if not validator.is_valid_length(cleaned):
                    continue
                is_valid, final_text = validator.validate(cleaned)
                if not is_valid:
                    continue
                _v2_has_region = bool(_RE_OLD_PLATE.match(cleaned))
                weight = 4 if _v2_has_region else 1
                for _ in range(weight):
                    all_candidates.append((final_text, ocr_conf))
            except Exception:
                continue

            # 매 메서드 후 조기종료 체크
            if all_candidates:
                _t2_counter = Counter(t for t, c in all_candidates)
                _t2_top, _t2_cnt = _t2_counter.most_common(1)[0]
                _t2_confs = [c for t, c in all_candidates if t == _t2_top]
                if _t2_cnt >= 4 and float(np.mean(_t2_confs)) > 0.70:
                    break

        # mid_consensus 판정
        if all_candidates:
            _mid_counter = Counter(t for t, c in all_candidates)
            _mid_top, _mid_cnt = _mid_counter.most_common(1)[0]
            _mid_confs = [c for t, c in all_candidates if t == _mid_top]
            if _mid_cnt >= 4 and float(np.mean(_mid_confs)) > 0.60:
                mid_consensus = True

        # 녹색 번호판: PaddleOCR 반전+강화 추가 전처리
        if is_green_plate and not mid_consensus and _paddle_engine:
            _green_extras = []
            _fi_lab = cv2.cvtColor(roi_inv, cv2.COLOR_BGR2LAB)
            _fi_l, _fi_a, _fi_b = cv2.split(_fi_lab)
            _fi_clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
            _fi_l = _fi_clahe.apply(_fi_l)
            _green_extras.append(cv2.cvtColor(cv2.merge([_fi_l, _fi_a, _fi_b]), cv2.COLOR_LAB2BGR))
            _fi_gray = cv2.cvtColor(roi_inv, cv2.COLOR_BGR2GRAY)
            _fi_c2 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
            _fi_enh = _fi_c2.apply(_fi_gray)
            _, _fi_bin = cv2.threshold(_fi_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _green_extras.append(cv2.cvtColor(_fi_bin, cv2.COLOR_GRAY2BGR))

            for _ge in _green_extras:
                try:
                    text, ocr_conf = _run_ocr("paddleocr", _paddle_engine, _ge)
                    if text and ocr_conf > 0.20:
                        cleaned = validator.clean_ocr_text(text)
                        if validator.is_valid_length(cleaned):
                            is_valid, final_text = validator.validate(cleaned)
                            if is_valid:
                                _v2r = bool(_RE_OLD_PLATE.match(cleaned))
                                w = 4 if _v2r else 1
                                for _ in range(w):
                                    all_candidates.append((final_text, ocr_conf))
                except Exception:
                    pass

    return all_candidates, tier1_consensus, mid_consensus


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# H. Extra crops 처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _process_extra_crops(extra_crops, ocr_engines, preprocessor, validator,
                         tier1_consensus, mid_consensus):
    """구형 번호판 상단+하단 결합 → 유효 후보 생성"""
    extra_candidates = []
    if not extra_crops or tier1_consensus or mid_consensus:
        return extra_candidates

    _paddle_engine = ocr_engines.get("paddleocr")
    top_texts, bot_texts = [], []
    top_confs, bot_confs = [], []

    for crop_name, crop_img in extra_crops:
        if _paddle_engine is None:
            break
        t, c = _run_ocr("paddleocr", _paddle_engine, crop_img)
        if t and c > 0.20:
            raw = re.sub(r'[^0-9가-힣a-zA-Z]', '', t.strip())
            _num_map = {'O': '0', 'o': '0', 'I': '1', 'l': '1', 'Z': '2', 'S': '5', 'B': '8', 'D': '0', 'Q': '0', 'G': '6'}
            _han_map = {'L': '나', 'H': '하', 'T': '타', 'U': '우', 'P': '파'}
            corrected = []
            for ch in raw:
                if ch.isalpha() and not ('가' <= ch <= '힣'):
                    corrected.append(_num_map.get(ch.upper(), _han_map.get(ch.upper(), ch)))
                else:
                    corrected.append(ch)
            cleaned_t = ''.join(corrected)
            fixed_chars = []
            for ch in cleaned_t:
                if '가' <= ch <= '힣':
                    fixed_chars.append(validator._KR_CONFUSION.get(ch, ch))
                else:
                    fixed_chars.append(ch)
            cleaned_t = ''.join(fixed_chars)
            if cleaned_t:
                if crop_name == "top":
                    top_texts.append(cleaned_t)
                    top_confs.append(c)
                else:
                    bot_texts.append(cleaned_t)
                    bot_confs.append(c)

    for tt in (top_texts or [""]):
        for bt in (bot_texts or [""]):
            combined = (tt + bt).strip()
            norm = validator._normalize_for_validation(combined)
            if validator.is_valid_length(norm):
                is_v, final = validator.validate(norm)
                if is_v:
                    avg_c = float(np.mean((top_confs or [0.3]) + (bot_confs or [0.3])))
                    weight = 6 if _RE_OLD_PLATE.match(final) else 2
                    for _ in range(weight):
                        extra_candidates.append((final, avg_c))

    return extra_candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# I. CRNN 적응 가중치
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _adjust_crnn_weight(crnn_candidates, all_candidates):
    """CRNN 결과를 all_candidates에 추가 (합의 기반 적응 가중)"""
    extra = []
    for ct, cc in crnn_candidates:
        _crnn_weight = 2
        if all_candidates:
            _crnn_digits = re.sub(r'[^0-9]', '', ct)
            _digit_matches = 0
            for _oc_text, _ in all_candidates:
                _oc_digits = re.sub(r'[^0-9]', '', _oc_text)
                if _crnn_digits and _oc_digits:
                    if _crnn_digits[-4:] == _oc_digits[-4:]:
                        _digit_matches += 1
            _pool_small = len(all_candidates) < 15
            if _digit_matches > len(all_candidates) * 0.2:
                _crnn_weight = 35 if _pool_small else 25
            elif _digit_matches > 0:
                _crnn_weight = 10 if _pool_small else 8
        for _ in range(_crnn_weight):
            extra.append((ct, cc))
    return extra


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# J. 위치별 투표
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _position_voting(all_candidates, roi_for_ocr, bbox, roi_crop_offset,
                     clf_scale, clf_pad, hangul_clf, ocr_engines, validator):
    """위치별 분리 투표 → best_text, best_conf"""
    if not all_candidates:
        return "", 0.0

    _re_split = re.compile(r'^(\d{2,3})([가-힣])(\d{4})$')
    _re_split_old = re.compile(r'^([가-힣]{2,3})(\d{1,2})([가-힣])(\d{4})$')

    new_parts, old_parts, other_candidates = [], [], []

    for txt, c in all_candidates:
        m_new = _re_split.match(txt)
        m_old = _re_split_old.match(txt)
        if m_new:
            new_parts.append((m_new.group(1), m_new.group(2), m_new.group(3), c))
        elif m_old:
            old_parts.append((m_old.group(1), m_old.group(2), m_old.group(3), m_old.group(4), c))
        else:
            other_candidates.append((txt, c))

    combined_best, combined_conf = "", 0.0

    # 신형 번호판 위치별 투표
    if new_parts:
        prefix_counter, suffix_counter = Counter(), Counter()
        hangul_confs = defaultdict(list)
        for pfx, hg, sfx, c in new_parts:
            prefix_counter[pfx] += 1
            suffix_counter[sfx] += 1
            hangul_confs[hg].append(c)
        best_pfx = prefix_counter.most_common(1)[0][0]
        best_hg = max(hangul_confs.keys(), key=lambda k: sum(hangul_confs[k]))

        # 한글 초성 교차검증 (HangulClassifier)
        ox1, oy1, ox2, oy2 = bbox
        rx1, ry1 = roi_crop_offset
        if hangul_clf._ready and 'paddleocr' in ocr_engines and clf_scale >= 1.0:
            try:
                _sc, _pd = clf_scale, clf_pad
                _px1 = _pd + (ox1 - rx1) * _sc
                _py1 = _pd + (oy1 - ry1) * _sc
                _pw = (ox2 - ox1) * _sc
                _ph = (oy2 - oy1) * _sc
                _rh, _rw = roi_for_ocr.shape[:2]
                _hx1 = max(0, int(_px1 + _pw * 0.26))
                _hx2 = min(_rw, int(_px1 + _pw * 0.52))
                _hy1 = max(0, int(_py1 + _ph * 0.20))
                _hy2 = min(_rh, int(_py1 + _ph * 0.80))
                if _hx2 > _hx1 + 10 and _hy2 > _hy1 + 10:
                    _hcrop = roi_for_ocr[_hy1:_hy2, _hx1:_hx2]
                    _new_hg, _changed = hangul_clf.check_override(
                        best_hg, _hcrop, ocr_engines['paddleocr'], ocr_engines)
                    if _changed:
                        best_hg = _new_hg
            except Exception:
                pass

        best_sfx = suffix_counter.most_common(1)[0][0]
        synth = best_pfx + best_hg + best_sfx
        is_v, final_synth = validator.validate(synth)
        if is_v:
            combined_best = final_synth
            combined_conf = sum(c for _, _, _, c in new_parts) / len(new_parts)
        else:
            counter_new = Counter(p + h + s for p, h, s, _ in new_parts)
            top = counter_new.most_common(1)[0][0]
            combined_best = top
            combined_conf = sum(c for p, h, s, c in new_parts if p + h + s == top) / counter_new[top]

    # 구형 번호판 위치별 투표
    if old_parts:
        region_counter, num_counter, suffix_counter = Counter(), Counter(), Counter()
        hangul_confs_old = defaultdict(list)
        for rg, nm, hg, sfx, c in old_parts:
            region_counter[rg] += 1
            num_counter[nm] += 1
            suffix_counter[sfx] += 1
            hangul_confs_old[hg].append(c)
        synth_old = (region_counter.most_common(1)[0][0] +
                     num_counter.most_common(1)[0][0] +
                     max(hangul_confs_old.keys(), key=lambda k: sum(hangul_confs_old[k])) +
                     suffix_counter.most_common(1)[0][0])
        is_v, final_old = validator.validate(synth_old)
        if is_v:
            old_conf = sum(c for _, _, _, _, c in old_parts) / len(old_parts)
            if len(old_parts) > len(new_parts) or not combined_best:
                combined_best, combined_conf = final_old, old_conf

    if other_candidates and not combined_best:
        counter_other = Counter(t for t, _ in other_candidates)
        top_other = counter_other.most_common(1)[0][0]
        combined_best = top_other
        combined_conf = sum(c for t, c in other_candidates if t == top_other) / counter_other[top_other]

    # 전체 투표
    counter_all = Counter(t for t, _ in all_candidates)
    whole_best = counter_all.most_common(1)[0][0]
    whole_count = counter_all[whole_best]
    whole_confs = [c for t, c in all_candidates if t == whole_best]
    whole_conf = sum(whole_confs) / len(whole_confs)

    if combined_best:
        best_text = combined_best
        _bt_confs = [c for t, c in all_candidates if t == combined_best]
        best_conf = (max(_bt_confs) * 0.6 + sum(_bt_confs) / len(_bt_confs) * 0.4) if _bt_confs else combined_conf
    else:
        best_text = whole_best
        best_conf = max(whole_confs) * 0.6 + whole_conf * 0.4

    # 합의 강도 보너스
    _total_votes = sum(counter_all.values())
    if _total_votes > 0:
        _top_ratio = whole_count / _total_votes
        if _top_ratio >= 0.80 and whole_count >= 3:
            best_conf = min(best_conf * 1.15, 1.0)
        elif _top_ratio >= 0.60 and whole_count >= 2:
            best_conf = min(best_conf * 1.08, 1.0)

    # 지역명 교정
    _re_withregion = re.compile(r'^([가-힣]{2,3})(\d{2}[가-힣]\d{4})$')
    _re_noregion = re.compile(r'^\d{2,3}[가-힣]\d{4}$')
    _VALID_REGIONS = set(PlateValidator._REGION_PREFIXES)

    valid_region_counts = {}
    for cand_t, cand_c in all_candidates:
        m_r = _re_withregion.match(cand_t)
        if m_r and m_r.group(1) in _VALID_REGIONS:
            valid_region_counts[m_r.group(1)] = valid_region_counts.get(m_r.group(1), 0) + 1

    if _re_noregion.match(best_text) and valid_region_counts:
        top_region = max(valid_region_counts, key=valid_region_counts.get)
        is_v, final = validator.validate(top_region + best_text)
        if is_v:
            best_text = final
    elif _re_withregion.match(best_text):
        m_cur = _re_withregion.match(best_text)
        cur_region, cur_suffix = m_cur.group(1), m_cur.group(2)
        if cur_region not in _VALID_REGIONS and valid_region_counts:
            top_region = max(valid_region_counts, key=valid_region_counts.get)
            is_v, final = validator.validate(top_region + cur_suffix)
            if is_v:
                best_text = final

    return best_text, best_conf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# K. 신뢰도 조정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _adjust_confidence(best_text, best_conf, bbox_conf_penalty, color_conf_penalty,
                       is_small_plate, det_w):
    """신뢰도 페널티 적용 + 최종 필터"""
    ocr_conf_threshold = PlateEngineConfig.OCR_CONF
    if not best_text or best_conf < ocr_conf_threshold:
        return "", 0.0

    _num_range_penalty = 1.0
    _m_prefix = re.match(r'^(?:[가-힣]{2,3})?(\d{2,3})[가-힣]\d{4}$', best_text)
    if _m_prefix:
        _pnum = int(_m_prefix.group(1))
        if len(_m_prefix.group(1)) == 3 and (_pnum < 100 or _pnum > 997):
            _num_range_penalty = 0.70

    if best_conf >= 0.90:
        _bbox_relief = 0.6
    elif best_conf >= 0.80:
        _bbox_relief = 0.4
    elif best_conf >= 0.70:
        _bbox_relief = 0.2
    else:
        _bbox_relief = 0.0
    adj_bbox_penalty = bbox_conf_penalty + (1.0 - bbox_conf_penalty) * _bbox_relief

    best_conf *= adj_bbox_penalty * color_conf_penalty * _num_range_penalty

    _raw = best_conf / max(adj_bbox_penalty * color_conf_penalty * _num_range_penalty, 0.01)
    if _raw >= 0.95:
        _floor = 0.92 if not is_small_plate else 0.90
    elif _raw >= 0.85:
        _floor = 0.85 if not is_small_plate else 0.80
    elif _raw >= 0.75:
        _floor = 0.78 if not is_small_plate else 0.72
    else:
        _floor = 0.70 if not is_small_plate else 0.60
    best_conf = max(best_conf, _floor)

    _is_large = det_w >= 150
    _thr = 0.50 if is_small_plate else (0.55 if _is_large else 0.60)
    if best_conf < _thr:
        return "", 0.0

    return best_text, best_conf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# L. 번호판 분류
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _classify_plate(text, roi, is_green_plate, det_w, det_h, validator):
    """번호판 색상/줄 수/유형 분류"""
    aspect = det_w / max(det_h, 1)
    plate_lines = 1 if aspect > 2.5 else 2
    plate_color = "흰색바탕_검은글씨"
    if is_green_plate:
        plate_color = "초록색바탕_흰글씨"
    else:
        try:
            _hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            _h = np.mean(_hsv[:, :, 0])
            _s = np.mean(_hsv[:, :, 1])
            if 20 < _h < 35 and _s > 60:
                plate_color = "노란색바탕_검은글씨"
            elif 95 < _h < 130 and _s > 40:
                plate_color = "파란색바탕_흰글씨"
        except Exception:
            pass
    return validator.classify_plate_type(text), validator.classify_vehicle_type(text), plate_lines, plate_color


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# M. 단건 OCR 처리 통합
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _process_single_ocr(request, ocr_engines, preprocessor, validator,
                        crnn_model, hangul_clf, kr_allowlist):
    """OCR 요청 1건 → OcrResult dict"""
    t0 = time.perf_counter()

    roi = request["roi"]
    bbox = request["bbox"]
    roi_crop_offset = request["roi_crop_offset"]
    det_w = request["det_w"]
    det_h = request["det_h"]
    bbox_conf_penalty = request["bbox_conf_penalty"]
    is_small_plate = request["is_small_plate"]
    roi_h, roi_w = roi.shape[:2]

    _fast_text = request.get("fast_ocr_text", "")
    _fast_conf = request.get("fast_ocr_conf", 0.0)
    _fast_shortcut = bool(_fast_text and _fast_conf > 0.7)

    roi_for_ocr, clf_scale, clf_pad = _upscale_roi(roi)
    color_conf_penalty, is_green_plate = _check_color(roi)

    all_candidates, tier1_consensus, mid_consensus = _run_ocr_pipeline(
        roi_for_ocr, ocr_engines, preprocessor, validator, is_green_plate, kr_allowlist)

    # CRNN + extra crops: tier1 합의 실패 시 실행 (fast_shortcut 시에도 보강)
    if not tier1_consensus or not _fast_shortcut:
        crnn_candidates = _run_crnn(crnn_model, roi, validator)
        extra_crops = _make_extra_crops(roi_for_ocr, roi_h, roi_w, is_green_plate)
        extra_cands = _process_extra_crops(extra_crops, ocr_engines, preprocessor, validator,
                                           tier1_consensus, mid_consensus)
        all_candidates.extend(extra_cands)
        crnn_extra = _adjust_crnn_weight(crnn_candidates, all_candidates)
        all_candidates.extend(crnn_extra)

    best_text, best_conf = _position_voting(
        all_candidates, roi_for_ocr, bbox, roi_crop_offset,
        clf_scale, clf_pad, hangul_clf, ocr_engines, validator)

    best_text, best_conf = _adjust_confidence(
        best_text, best_conf, bbox_conf_penalty, color_conf_penalty, is_small_plate, det_w)

    if best_text:
        plate_type, vehicle_type, plate_lines, plate_color = _classify_plate(
            best_text, roi, is_green_plate, det_w, det_h, validator)
        is_valid_format = True
    else:
        plate_type, vehicle_type, plate_lines, plate_color = "", "", 1, "흰색바탕_검은글씨"
        is_valid_format = False

    ocr_ms = (time.perf_counter() - t0) * 1000

    return {
        "frame_id": request["frame_id"],
        "det_index": request["det_index"],
        "best_text": best_text,
        "best_conf": best_conf,
        "bbox": bbox,
        "is_small_plate": is_small_plate,
        "det_w": det_w,
        "is_green_plate": is_green_plate,
        "plate_type": plate_type,
        "vehicle_type": vehicle_type,
        "plate_lines": plate_lines,
        "plate_color": plate_color,
        "is_valid_format": is_valid_format,
        "ocr_ms": ocr_ms,
        "timestamp": time.time(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# N. 워커 메인 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ocr_worker_loop(ocr_queue: Queue, result_queue: Queue, cmd_queue: Queue):
    """CMD 3: OCR 워커 프로세스 메인 루프 (PaddleOCR 단독)"""
    print("[CMD3-OCR] 워커 프로세스 시작 — PaddleOCR 단독 엔진 로딩 중...")

    ocr_engines, crnn_model, hangul_clf, kr_allowlist = _init_ocr_engines()
    preprocessor = ImagePreprocessor()
    validator = PlateValidator()

    # 워밍업
    _warmup_t0 = time.perf_counter()
    _dummy_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    if "paddleocr" in ocr_engines:
        try:
            ocr_engines["paddleocr"].ocr(_dummy_img, cls=True)
        except Exception:
            pass
    _warmup_ms = (time.perf_counter() - _warmup_t0) * 1000
    print(f"[CMD3-OCR] 엔진 로드 완료 — 워밍업 {_warmup_ms:.0f}ms — 요청 대기 중")

    processed_count = 0

    while True:
        try:
            cmd = cmd_queue.get_nowait()
            if cmd == CMD_FLUSH:
                while not ocr_queue.empty():
                    try:
                        ocr_queue.get_nowait()
                    except Exception:
                        break
                while not result_queue.empty():
                    try:
                        result_queue.get_nowait()
                    except Exception:
                        break
                print(f"[CMD3-OCR] FLUSH 완료")
            elif cmd == CMD_OCR_STOP:
                print(f"[CMD3-OCR] STOP 수신 → 종료 (처리: {processed_count}건)")
                break
        except Exception:
            pass

        try:
            request = ocr_queue.get(timeout=0.1)
        except Exception:
            continue

        _age = time.time() - request.get("timestamp", 0)
        if _age > 3.0:
            result_queue.put({
                "frame_id": request.get("frame_id", -1),
                "det_index": request.get("det_index", -1),
                "best_text": "", "best_conf": 0.0,
                "bbox": request.get("bbox", [0, 0, 0, 0]),
                "is_small_plate": False, "det_w": 0,
                "is_green_plate": False, "plate_type": "", "vehicle_type": "",
                "plate_lines": 1, "plate_color": "흰색바탕_검은글씨",
                "is_valid_format": False, "ocr_ms": 0.0, "timestamp": time.time(),
            })
            continue

        try:
            result = _process_single_ocr(
                request, ocr_engines, preprocessor, validator,
                crnn_model, hangul_clf, kr_allowlist)
            result_queue.put(result)
            processed_count += 1
            if result["best_text"]:
                print(f"  [CMD3] #{result['frame_id']:04d}/{result['det_index']} "
                      f"→ {result['best_text']} (conf={result['best_conf']:.2f}, "
                      f"{result['ocr_ms']:.0f}ms)")
        except Exception as e:
            print(f"[CMD3-OCR] 처리 오류: {e}")
            traceback.print_exc()
            result_queue.put({
                "frame_id": request.get("frame_id", -1),
                "det_index": request.get("det_index", -1),
                "best_text": "", "best_conf": 0.0,
                "bbox": request.get("bbox", [0, 0, 0, 0]),
                "is_small_plate": False, "det_w": 0,
                "is_green_plate": False, "plate_type": "", "vehicle_type": "",
                "plate_lines": 1, "plate_color": "흰색바탕_검은글씨",
                "is_valid_format": False, "ocr_ms": 0.0, "timestamp": time.time(),
            })

    print("[CMD3-OCR] 워커 프로세스 종료")
