# -*- coding: utf-8 -*-
"""
CMD 6: OCR 인식 워커 프로세스
plate_engine_pro.py process_frame()에서 OCR 파이프라인 부분을 추출하여
별도 프로세스에서 실행. multiprocessing.Queue로 IPC 통신.

추출 원본: plate_engine_pro.py L1605-2278, L2468-2522
"""
import os
import re
import time
import traceback
import numpy as np
import cv2
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Queue

from pipeline_common import CMD_OCR_STOP

# plate_engine_pro.py에서 클래스/함수 import (코드 중복 방지)
# 모듈 레벨에서 모델 인스턴스화 없음 — 안전
from plate_engine_pro import (
    ImagePreprocessor,
    PlateValidator,
    HangulClassifier,
    PlateEngineConfig,
    _CRNNModel,
    normalize,
    _deskew_and_otsu,
)

# OCR 엔진 임포트
try:
    import easyocr
    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False

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
# A. OCR 엔진 초기화 (plate_engine_pro.py L1310-1377 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _init_ocr_engines():
    """PaddleOCR, EasyOCR, CRNN 모델 초기화 (워커 시작 시 1회)

    Returns:
        ocr_engines: dict {name: engine}
        crnn_model: _CRNNModel 또는 None
        hangul_clf: HangulClassifier
        kr_allowlist: str (EasyOCR 허용 문자)
    """
    ocr_engines = {}

    if HAS_PADDLEOCR:
        paddle_kwargs = dict(lang="korean", use_angle_cls=True, show_log=False, use_gpu=False)
        # Windows 한글 경로 우회: 영문 경로에 모델이 있으면 직접 지정
        _paddle_model_root = None
        for _pdir in [
            Path("C:/paddle_models/.paddleocr/whl"),
            Path("C:/tools/paddleocr_models"),
        ]:
            if _pdir.exists():
                _paddle_model_root = _pdir
                break
        if _paddle_model_root is not None:
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
                print(f"[CMD6-OCR] PaddleOCR 초기화 실패: {e}")
        except Exception as e:
            print(f"[CMD6-OCR] PaddleOCR 초기화 실패: {e}")

    if HAS_EASYOCR:
        ocr_engines["easyocr"] = easyocr.Reader(["ko", "en"], gpu=True)

    if HAS_TESSERACT:
        try:
            _tess_langs = pytesseract.get_languages()
            if 'kor' in _tess_langs:
                ocr_engines["tesseract"] = "tesseract"
            else:
                print("[CMD6-OCR] Tesseract: 한국어 언어팩(kor) 없음 — 건너뜀")
        except Exception:
            ocr_engines["tesseract"] = "tesseract"

    # CRNN 모델 로드
    crnn_model = None
    _crnn_path = Path(__file__).resolve().parent / "plate_ocr_crnn.pth"
    if HAS_TORCH and _crnn_path.exists():
        try:
            crnn_model = _CRNNModel(str(_crnn_path))
            print(f"[CMD6-OCR] CRNN 모델 로드: {_crnn_path}")
        except Exception as e:
            print(f"[CMD6-OCR] CRNN 모델 로드 실패: {e}")

    # 한글 분류기
    hangul_clf = HangulClassifier()

    # EasyOCR 허용 문자
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

    print(f"[CMD6-OCR] OCR 엔진: {list(ocr_engines.keys())}")
    return ocr_engines, crnn_model, hangul_clf, kr_allowlist


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. ROI 업스케일 (plate_engine_pro.py L1621-1653 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _upscale_roi(roi):
    """ROI를 500px 목표로 업스케일 + 선명화 + 흰색 패딩

    Returns:
        roi_for_ocr: 업스케일된 ROI
        scale: 업스케일 배율
        pad: 패딩 크기 (px)
    """
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
        # 업스케일 후 선명화
        if roi_w < 80:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        else:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        roi_for_ocr = cv2.filter2D(roi_for_ocr, -1, kernel)
    else:
        roi_for_ocr = roi

    # 흰색 테두리 패딩
    pad = max(10, int(roi_for_ocr.shape[0] * 0.15))
    roi_for_ocr = cv2.copyMakeBorder(
        roi_for_ocr, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    return roi_for_ocr, scale, pad


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. 색상 검증 + 녹색 판별 (plate_engine_pro.py L1655-1692 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_color(roi):
    """ROI의 HSV 색상 분석 → 색상 페널티 + 녹색 번호판 여부

    Returns:
        color_conf_penalty: float (1.0 = 정상)
        is_green_plate: bool
    """
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
# D. CRNN 모델 인식 (plate_engine_pro.py L1672-1683 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _run_crnn(crnn_model, roi, validator):
    """CRNN 모델로 원본 ROI 인식 → 유효 후보 리스트

    Returns:
        crnn_candidates: list of (text, conf)
    """
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
# E. Extra crops 생성 (plate_engine_pro.py L1694-1753 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _make_extra_crops(roi_for_ocr, roi_h, roi_w, is_green_plate):
    """구형 2줄 번호판 상단/하단 크롭 + 녹색 번호판 추가 크롭

    Returns:
        extra_crops: list of (label, image) — label은 "top" 또는 "bot"
    """
    extra_crops = []
    if roi_h <= roi_w * 0.45:
        return extra_crops

    top_crop = roi_for_ocr[:int(roi_for_ocr.shape[0] * 0.5), :]
    bot_crop = roi_for_ocr[int(roi_for_ocr.shape[0] * 0.4):, :]

    # 상단 크롭 강화: 500px 확대
    top_h, top_w = top_crop.shape[:2]
    if top_w < 500:
        sc_top = 500 / top_w
        top_crop = cv2.resize(top_crop, None, fx=sc_top, fy=sc_top, interpolation=cv2.INTER_CUBIC)

    # 반전 버전
    top_inv = cv2.bitwise_not(top_crop)
    # CLAHE 강화 버전
    top_gray = cv2.cvtColor(top_crop, cv2.COLOR_BGR2GRAY)
    clahe_obj = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    top_clahe = clahe_obj.apply(top_gray)
    top_clahe_bgr = cv2.cvtColor(top_clahe, cv2.COLOR_GRAY2BGR)
    # HSV V-channel 이진화
    top_hsv = cv2.cvtColor(top_crop, cv2.COLOR_BGR2HSV)
    _, _, top_v = cv2.split(top_hsv)
    _, top_val_mask = cv2.threshold(top_v, 150, 255, cv2.THRESH_BINARY)
    top_val_bgr = cv2.cvtColor(top_val_mask, cv2.COLOR_GRAY2BGR)
    # 선명화 800px 버전
    sc_800 = 800 / top_crop.shape[1] if top_crop.shape[1] < 800 else 1.0
    if sc_800 > 1.0:
        top_800 = cv2.resize(top_inv, None, fx=sc_800, fy=sc_800, interpolation=cv2.INTER_CUBIC)
        sharp_k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        top_sharp800 = cv2.filter2D(top_800, -1, sharp_k)
    else:
        top_sharp800 = top_inv

    extra_crops = [
        ("top", top_inv),
        ("bot", bot_crop),
    ]

    # 녹색 번호판 추가 전처리
    if is_green_plate:
        # 반전 + 강화 CLAHE (clipLimit 8.0)
        _inv_lab = cv2.cvtColor(top_inv, cv2.COLOR_BGR2LAB)
        _l, _a, _b = cv2.split(_inv_lab)
        _clahe_s = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        _l = _clahe_s.apply(_l)
        _inv_clahe = cv2.cvtColor(cv2.merge([_l, _a, _b]), cv2.COLOR_LAB2BGR)
        extra_crops.append(("top", _inv_clahe))
        # 반전 + Otsu 이진화
        _inv_gray = cv2.cvtColor(top_inv, cv2.COLOR_BGR2GRAY)
        _clahe_g = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        _inv_enh = _clahe_g.apply(_inv_gray)
        _, _inv_bin = cv2.threshold(_inv_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        extra_crops.append(("top", cv2.cvtColor(_inv_bin, cv2.COLOR_GRAY2BGR)))
        # 녹색 채널 추출 → 반전 → Otsu
        _g_ch = top_crop[:, :, 1]
        _g_inv = cv2.bitwise_not(_g_ch)
        _, _g_bin = cv2.threshold(_g_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        extra_crops.append(("top", cv2.cvtColor(_g_bin, cv2.COLOR_GRAY2BGR)))
        # V채널 낮은 임계값
        _, _v_low = cv2.threshold(top_v, 100, 255, cv2.THRESH_BINARY)
        extra_crops.append(("top", cv2.cvtColor(_v_low, cv2.COLOR_GRAY2BGR)))
        # 하단 반전 추가
        extra_crops.append(("bot", cv2.bitwise_not(bot_crop)))

    return extra_crops


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# F. OCR 실행 (plate_engine_pro.py L2468-2522 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _run_ocr(engine_name, engine, image, kr_allowlist=None):
    """단일 OCR 엔진 실행 → (text, conf)"""
    try:
        if engine_name == "paddleocr":
            result = engine.ocr(image, cls=True)
            if result and result[0]:
                lines = sorted(result[0], key=lambda l: l[0][0][1])
                texts = [l[1][0] for l in lines]
                confs = [l[1][1] for l in lines]
                if texts:
                    return "".join(texts), float(np.mean(confs))
        elif engine_name == "easyocr":
            ocr_kwargs = dict(detail=1, paragraph=False)
            if kr_allowlist:
                ocr_kwargs["allowlist"] = kr_allowlist
            result = engine.readtext(image, **ocr_kwargs)
            if result:
                result_sorted = sorted(result, key=lambda r: r[0][0][1])
                texts = [r[1] for r in result_sorted]
                confs = [r[2] for r in result_sorted]
                combined = "".join(texts)
                avg_conf = float(np.mean(confs))
                return combined, avg_conf
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
        elif engine_name == "crnn":
            text, conf = engine.recognize(image)
            if text:
                return text, conf
    except Exception:
        pass
    return "", 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G. OCR 파이프라인: Tier1 + Tier2 + 녹색 (L1757-1934 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _run_ocr_pipeline(roi_for_ocr, ocr_engines, preprocessor, validator,
                      is_green_plate, kr_allowlist):
    """Tier1 + Tier2 OCR 앙상블 실행

    Returns:
        all_candidates: list of (text, weighted_conf)
        tier1_consensus: bool
        mid_consensus: bool
    """
    _ENGINE_WEIGHT = {'paddleocr': 1.0, 'easyocr': 0.85, 'tesseract': 0.6}
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
                weighted_conf = ocr_conf * 1.0
                _v2_has_region = bool(re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', cleaned))
                weight = 4 if _v2_has_region else 1
                for _ in range(weight):
                    all_candidates.append((final_text, weighted_conf))
            except Exception:
                continue

        # Tier 1 합의 판정
        if all_candidates:
            _t1_counter = Counter(t for t, c in all_candidates)
            _t1_top, _t1_cnt = _t1_counter.most_common(1)[0]
            _t1_confs = [c for t, c in all_candidates if t == _t1_top]
            if _t1_cnt >= 2 and float(np.mean(_t1_confs)) > 0.6:
                tier1_consensus = True

    # ── Tier 2: PaddleOCR + EasyOCR 교차검증 ──
    mid_consensus = False
    if not tier1_consensus:
        _tier2_break = False
        for _t2_idx, method in enumerate(_TIER2_METHODS):
            if _tier2_break:
                break
            try:
                if method == "_inverted":
                    processed = roi_inv.copy()
                else:
                    proc_func = getattr(preprocessor, method, None)
                    if proc_func is None:
                        continue
                    processed = proc_func(roi_for_ocr.copy())
                _skip_easy = False
                for engine_name, engine in ocr_engines.items():
                    if engine_name == "tesseract":
                        continue
                    if engine_name == "easyocr" and _skip_easy:
                        continue
                    text, ocr_conf = _run_ocr(engine_name, engine, processed, kr_allowlist)
                    if not text or ocr_conf < 0.20:
                        continue
                    cleaned = validator.clean_ocr_text(text)
                    if not validator.is_valid_length(cleaned):
                        continue
                    is_valid, final_text = validator.validate(cleaned)
                    if not is_valid:
                        continue
                    _ew = _ENGINE_WEIGHT.get(engine_name, 0.7)
                    weighted_conf = ocr_conf * _ew
                    _v2_has_region = bool(re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', cleaned))
                    weight = 4 if _v2_has_region else 1
                    for _ in range(weight):
                        all_candidates.append((final_text, weighted_conf))
                    # PaddleOCR 합의 시 EasyOCR 스킵
                    if engine_name == "paddleocr" and all_candidates:
                        _pe_counter = Counter(t for t, c in all_candidates)
                        _pe_top, _pe_cnt = _pe_counter.most_common(1)[0]
                        _pe_confs = [c for t, c in all_candidates if t == _pe_top]
                        if _pe_cnt >= 4 and float(np.mean(_pe_confs)) > 0.70:
                            _skip_easy = True
            except Exception:
                continue
            # 매 메서드 후 조기종료
            if all_candidates:
                _t2_counter = Counter(t for t, c in all_candidates)
                _t2_top, _t2_cnt = _t2_counter.most_common(1)[0]
                _t2_confs = [c for t, c in all_candidates if t == _t2_top]
                if _t2_cnt >= 4 and float(np.mean(_t2_confs)) > 0.70:
                    _tier2_break = True

        # mid_consensus 판정
        if all_candidates:
            _mid_counter = Counter(t for t, c in all_candidates)
            _mid_top, _mid_cnt = _mid_counter.most_common(1)[0]
            _mid_confs = [c for t, c in all_candidates if t == _mid_top]
            if _mid_cnt >= 4 and float(np.mean(_mid_confs)) > 0.60:
                mid_consensus = True

        # 녹색 번호판: 반전+강화 추가 전처리
        if is_green_plate and not mid_consensus:
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
                _ge_skip = False
                if _paddle_engine:
                    try:
                        text, ocr_conf = _run_ocr("paddleocr", _paddle_engine, _ge)
                        if text and ocr_conf > 0.20:
                            cleaned = validator.clean_ocr_text(text)
                            if validator.is_valid_length(cleaned):
                                is_valid, final_text = validator.validate(cleaned)
                                if is_valid:
                                    _gwc = ocr_conf * 1.0
                                    _v2r = bool(re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', cleaned))
                                    w = 4 if _v2r else 1
                                    for _ in range(w):
                                        all_candidates.append((final_text, _gwc))
                                    if ocr_conf >= 0.80:
                                        _ge_skip = True
                    except Exception:
                        pass
                if not _ge_skip:
                    for engine_name, engine in ocr_engines.items():
                        if engine_name == "paddleocr":
                            continue
                        try:
                            text, ocr_conf = _run_ocr(engine_name, engine, _ge, kr_allowlist)
                            if text and ocr_conf > 0.20:
                                cleaned = validator.clean_ocr_text(text)
                                if validator.is_valid_length(cleaned):
                                    is_valid, final_text = validator.validate(cleaned)
                                    if is_valid:
                                        _gew = _ENGINE_WEIGHT.get(engine_name, 0.7)
                                        _gwc = ocr_conf * _gew
                                        _v2r = bool(re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', cleaned))
                                        w = 4 if _v2r else 1
                                        for _ in range(w):
                                            all_candidates.append((final_text, _gwc))
                        except Exception:
                            continue

    return all_candidates, tier1_consensus, mid_consensus


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# H. Extra crops 처리 (plate_engine_pro.py L1936-2000 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _process_extra_crops(extra_crops, ocr_engines, preprocessor, validator,
                         tier1_consensus, mid_consensus):
    """구형 번호판 상단+하단 결합 → 유효 후보 생성

    Returns:
        extra_candidates: list of (text, conf)
    """
    extra_candidates = []
    if not extra_crops or tier1_consensus or mid_consensus:
        return extra_candidates

    _paddle_engine = ocr_engines.get("paddleocr")
    top_texts, bot_texts = [], []
    top_confs, bot_confs = [], []
    _crop_methods = ["original"]

    for crop_name, crop_img in extra_crops:
        for cm in _crop_methods:
            if cm == "original":
                proc_crop = crop_img
            else:
                pfn = getattr(preprocessor, cm, None)
                if pfn is None:
                    continue
                try:
                    proc_crop = pfn(crop_img.copy())
                except Exception:
                    continue
            _crop_engines = [("paddleocr", _paddle_engine)] if _paddle_engine else []
            for eng_name, eng in _crop_engines:
                t, c = _run_ocr(eng_name, eng, proc_crop)
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
                    # 한글 혼동 교정
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

    # 상단 + 하단 조합
    for tt in (top_texts or [""]):
        for bt in (bot_texts or [""]):
            combined = (tt + bt).strip()
            norm = validator._normalize_for_validation(combined)
            if validator.is_valid_length(norm):
                is_v, final = validator.validate(norm)
                if is_v:
                    avg_c = float(np.mean((top_confs or [0.3]) + (bot_confs or [0.3])))
                    weight = 6 if re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', final) else 2
                    for _ in range(weight):
                        extra_candidates.append((final, avg_c))

    return extra_candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# I. CRNN 적응 가중치 (plate_engine_pro.py L2002-2027 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _adjust_crnn_weight(crnn_candidates, all_candidates):
    """CRNN 결과를 all_candidates에 추가 (합의 기반 적응 가중)

    Returns:
        추가할 후보 리스트 [(text, conf), ...]
    """
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
# J. 위치별 투표 (plate_engine_pro.py L2029-2228 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _position_voting(all_candidates, roi_for_ocr, bbox, roi_crop_offset,
                     clf_scale, clf_pad, hangul_clf, ocr_engines, validator):
    """위치별 분리 투표 → best_text, best_conf

    Returns:
        best_text: str
        best_conf: float
    """
    if not all_candidates:
        return "", 0.0

    _re_split = re.compile(r'^(\d{2,3})([가-힣])(\d{4})$')
    _re_split_old = re.compile(r'^([가-힣]{2,3})(\d{1,2})([가-힣])(\d{4})$')

    new_parts = []
    old_parts = []
    other_candidates = []

    for txt, c in all_candidates:
        m_new = _re_split.match(txt)
        m_old = _re_split_old.match(txt)
        if m_new:
            new_parts.append((m_new.group(1), m_new.group(2), m_new.group(3), c))
        elif m_old:
            old_parts.append((m_old.group(1), m_old.group(2), m_old.group(3), m_old.group(4), c))
        else:
            other_candidates.append((txt, c))

    combined_best = ""
    combined_conf = 0.0

    # 신형 번호판 위치별 투표
    if new_parts:
        prefix_counter = Counter()
        hangul_counter = Counter()
        suffix_counter = Counter()
        prefix_confs = defaultdict(list)
        hangul_confs = defaultdict(list)
        suffix_confs = defaultdict(list)
        for pfx, hg, sfx, c in new_parts:
            prefix_counter[pfx] += 1
            hangul_counter[hg] += 1
            suffix_counter[sfx] += 1
            prefix_confs[pfx].append(c)
            hangul_confs[hg].append(c)
            suffix_confs[sfx].append(c)
        best_pfx = prefix_counter.most_common(1)[0][0]
        best_hg = max(hangul_confs.keys(), key=lambda k: sum(hangul_confs[k]))

        # 한글 초성 교차검증
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
                        best_hg, _hcrop, ocr_engines['paddleocr'],
                        ocr_engines
                    )
                    if _changed:
                        best_hg = _new_hg
            except Exception:
                pass

        best_sfx = suffix_counter.most_common(1)[0][0]
        synth = best_pfx + best_hg + best_sfx
        is_v, final_synth = validator.validate(synth)
        if is_v:
            combined_best = final_synth
            all_c = [c for _, _, _, c in new_parts]
            combined_conf = sum(all_c) / len(all_c)
        else:
            counter_new = Counter(p + h + s for p, h, s, _ in new_parts)
            top = counter_new.most_common(1)[0][0]
            combined_best = top
            combined_conf = sum(c for p, h, s, c in new_parts if p + h + s == top) / counter_new[top]

    # 구형 번호판 위치별 투표
    if old_parts:
        region_counter = Counter()
        num_counter = Counter()
        hangul_counter = Counter()
        suffix_counter = Counter()
        hangul_confs_old = defaultdict(list)
        for rg, nm, hg, sfx, c in old_parts:
            region_counter[rg] += 1
            num_counter[nm] += 1
            hangul_counter[hg] += 1
            suffix_counter[sfx] += 1
            hangul_confs_old[hg].append(c)
        best_rg = region_counter.most_common(1)[0][0]
        best_nm = num_counter.most_common(1)[0][0]
        best_hg = max(hangul_confs_old.keys(), key=lambda k: sum(hangul_confs_old[k]))
        best_sfx = suffix_counter.most_common(1)[0][0]
        synth_old = best_rg + best_nm + best_hg + best_sfx
        is_v, final_old = validator.validate(synth_old)
        if is_v:
            old_conf = sum(c for _, _, _, _, c in old_parts) / len(old_parts)
            if len(old_parts) > len(new_parts) or not combined_best:
                combined_best = final_old
                combined_conf = old_conf

    # 기타 후보 포함 전체 투표
    if other_candidates and not combined_best:
        counter_other = Counter(t for t, _ in other_candidates)
        top_other = counter_other.most_common(1)[0][0]
        combined_best = top_other
        combined_conf = sum(c for t, c in other_candidates if t == top_other) / counter_other[top_other]

    # 전체 투표 결과
    counter_all = Counter(t for t, _ in all_candidates)
    whole_best = counter_all.most_common(1)[0][0]
    whole_count = counter_all[whole_best]
    whole_confs = [c for t, c in all_candidates if t == whole_best]
    whole_conf = sum(whole_confs) / len(whole_confs)

    if combined_best:
        best_text = combined_best
        _bt_confs = [c for t, c in all_candidates if t == combined_best]
        if _bt_confs:
            _c_max = max(_bt_confs)
            _c_mean = sum(_bt_confs) / len(_bt_confs)
            best_conf = _c_max * 0.6 + _c_mean * 0.4
        else:
            best_conf = combined_conf
    else:
        best_text = whole_best
        _w_max = max(whole_confs)
        best_conf = _w_max * 0.6 + whole_conf * 0.4

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
        if m_r:
            region = m_r.group(1)
            if region in _VALID_REGIONS:
                valid_region_counts[region] = valid_region_counts.get(region, 0) + 1

    if _re_noregion.match(best_text):
        if valid_region_counts:
            top_region = max(valid_region_counts, key=valid_region_counts.get)
            candidate_text = top_region + best_text
            is_v, final = validator.validate(candidate_text)
            if is_v:
                best_text = final
    elif _re_withregion.match(best_text):
        m_cur = _re_withregion.match(best_text)
        cur_region = m_cur.group(1)
        cur_suffix = m_cur.group(2)
        if cur_region not in _VALID_REGIONS and valid_region_counts:
            top_region = max(valid_region_counts, key=valid_region_counts.get)
            new_text = top_region + cur_suffix
            is_v, final = validator.validate(new_text)
            if is_v:
                best_text = final
        elif cur_region in _VALID_REGIONS and valid_region_counts:
            for region, count in valid_region_counts.items():
                if region != cur_region and count > valid_region_counts.get(cur_region, 0):
                    new_text = region + cur_suffix
                    is_v, final = validator.validate(new_text)
                    if is_v:
                        best_text = final
                        break

    return best_text, best_conf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# K. 신뢰도 조정 (plate_engine_pro.py L2238-2290 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _adjust_confidence(best_text, best_conf, bbox_conf_penalty, color_conf_penalty,
                       is_small_plate, det_w):
    """신뢰도 페널티 적용 + 최종 필터

    Returns:
        (best_text, best_conf) 또는 ("", 0.0) (필터됨)
    """
    ocr_conf_threshold = PlateEngineConfig.OCR_CONF
    if not best_text or best_conf < ocr_conf_threshold:
        return "", 0.0

    # 번호 범위 페널티
    _num_range_penalty = 1.0
    _m_prefix = re.match(r'^(?:[가-힣]{2,3})?(\d{2,3})[가-힣]\d{4}$', best_text)
    if _m_prefix:
        _pnum = int(_m_prefix.group(1))
        if len(_m_prefix.group(1)) == 3:
            if _pnum < 100 or _pnum > 997:
                _num_range_penalty = 0.70

    # OCR 신뢰도 기반 bbox 페널티 완화
    if best_conf >= 0.90:
        _bbox_relief = 0.6
    elif best_conf >= 0.80:
        _bbox_relief = 0.4
    elif best_conf >= 0.70:
        _bbox_relief = 0.2
    else:
        _bbox_relief = 0.0
    adj_bbox_penalty = bbox_conf_penalty + (1.0 - bbox_conf_penalty) * _bbox_relief

    # 페널티 적용
    best_conf *= adj_bbox_penalty * color_conf_penalty * _num_range_penalty

    # 적응형 신뢰도 floor
    _raw_conf_before_penalty = best_conf / max(adj_bbox_penalty * color_conf_penalty * _num_range_penalty, 0.01)
    if _raw_conf_before_penalty >= 0.95:
        _floor = 0.92 if not is_small_plate else 0.90
    elif _raw_conf_before_penalty >= 0.85:
        _floor = 0.85 if not is_small_plate else 0.80
    elif _raw_conf_before_penalty >= 0.75:
        _floor = 0.78 if not is_small_plate else 0.72
    else:
        _floor = 0.70 if not is_small_plate else 0.60
    best_conf = max(best_conf, _floor)

    # 최종 필터
    _is_large_plate = det_w >= 150
    if is_small_plate:
        _conf_threshold = 0.50
    elif _is_large_plate:
        _conf_threshold = 0.55
    else:
        _conf_threshold = 0.60
    if best_conf < _conf_threshold:
        return "", 0.0

    return best_text, best_conf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# L. 번호판 분류 (plate_engine_pro.py L2401-2423 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _classify_plate(text, roi, is_green_plate, det_w, det_h, validator):
    """번호판 색상/줄 수/유형 분류

    Returns:
        plate_type, vehicle_type, plate_lines, plate_color
    """
    bbox_w = det_w
    bbox_h = det_h
    aspect = bbox_w / max(bbox_h, 1)
    plate_lines = 1 if aspect > 2.5 else 2

    plate_color = "흰색바탕_검은글씨"
    if is_green_plate:
        plate_color = "초록색바탕_흰글씨"
    else:
        try:
            _hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            _h_mean = np.mean(_hsv_roi[:, :, 0])
            _s_mean = np.mean(_hsv_roi[:, :, 1])
            if 20 < _h_mean < 35 and _s_mean > 60:
                plate_color = "노란색바탕_검은글씨"
            elif 95 < _h_mean < 130 and _s_mean > 40:
                plate_color = "파란색바탕_흰글씨"
        except Exception:
            pass

    plate_type = validator.classify_plate_type(text)
    vehicle_type = validator.classify_vehicle_type(text)

    return plate_type, vehicle_type, plate_lines, plate_color


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# M. 단건 OCR 처리 통합 (신규)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _process_single_ocr(request, ocr_engines, preprocessor, validator,
                        crnn_model, hangul_clf, kr_allowlist):
    """OCR 요청 1건 → OcrResult dict 생성

    Args:
        request: OcrRequest dict
    Returns:
        OcrResult dict
    """
    t0 = time.perf_counter()

    roi = request["roi"]
    bbox = request["bbox"]
    roi_crop_offset = request["roi_crop_offset"]
    det_w = request["det_w"]
    det_h = request["det_h"]
    bbox_conf_penalty = request["bbox_conf_penalty"]
    is_small_plate = request["is_small_plate"]

    roi_h, roi_w = roi.shape[:2]

    # B. ROI 업스케일
    roi_for_ocr, clf_scale, clf_pad = _upscale_roi(roi)

    # C. 색상 검증
    color_conf_penalty, is_green_plate = _check_color(roi)

    # D. CRNN
    crnn_candidates = _run_crnn(crnn_model, roi, validator)

    # E. Extra crops
    extra_crops = _make_extra_crops(roi_for_ocr, roi_h, roi_w, is_green_plate)

    # G. OCR 파이프라인
    all_candidates, tier1_consensus, mid_consensus = _run_ocr_pipeline(
        roi_for_ocr, ocr_engines, preprocessor, validator,
        is_green_plate, kr_allowlist
    )

    # H. Extra crops 처리
    extra_cands = _process_extra_crops(
        extra_crops, ocr_engines, preprocessor, validator,
        tier1_consensus, mid_consensus
    )
    all_candidates.extend(extra_cands)

    # I. CRNN 가중치
    crnn_extra = _adjust_crnn_weight(crnn_candidates, all_candidates)
    all_candidates.extend(crnn_extra)

    # J. 위치별 투표
    best_text, best_conf = _position_voting(
        all_candidates, roi_for_ocr, bbox, roi_crop_offset,
        clf_scale, clf_pad, hangul_clf, ocr_engines, validator
    )

    # K. 신뢰도 조정
    best_text, best_conf = _adjust_confidence(
        best_text, best_conf, bbox_conf_penalty, color_conf_penalty,
        is_small_plate, det_w
    )

    # L. 분류
    if best_text:
        plate_type, vehicle_type, plate_lines, plate_color = _classify_plate(
            best_text, roi, is_green_plate, det_w, det_h, validator
        )
        is_valid_format = True
    else:
        plate_type = ""
        vehicle_type = ""
        plate_lines = 1
        plate_color = "흰색바탕_검은글씨"
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
# N. 워커 메인 루프 (신규)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ocr_worker_loop(ocr_queue: Queue, result_queue: Queue, cmd_queue: Queue):
    """OCR 워커 프로세스 메인 루프

    ocr_queue에서 OcrRequest를 꺼내 OCR 처리 → result_queue에 OcrResult 전송.
    cmd_queue에서 CMD_OCR_STOP 수신 시 종료.
    """
    print("[CMD6-OCR] 워커 프로세스 시작 — 엔진 로딩 중...")

    # OCR 엔진 초기화 (1회, ~5-8초)
    ocr_engines, crnn_model, hangul_clf, kr_allowlist = _init_ocr_engines()
    preprocessor = ImagePreprocessor()
    validator = PlateValidator()

    # 워밍업: 더미 이미지로 첫 호출 지연 제거
    _warmup_t0 = time.perf_counter()
    _dummy_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    if "paddleocr" in ocr_engines:
        try:
            ocr_engines["paddleocr"].ocr(_dummy_img, cls=True)
        except Exception:
            pass
    if "easyocr" in ocr_engines:
        try:
            ocr_engines["easyocr"].readtext(_dummy_img)
        except Exception:
            pass
    _warmup_ms = (time.perf_counter() - _warmup_t0) * 1000
    print(f"[CMD6-OCR] 엔진 로드 완료 — 워밍업 {_warmup_ms:.0f}ms — 요청 대기 중")

    processed_count = 0

    while True:
        # 제어 명령 확인
        try:
            cmd = cmd_queue.get_nowait()
            if cmd == CMD_OCR_STOP:
                print(f"[CMD6-OCR] STOP 수신 → 종료 (처리: {processed_count}건)")
                break
        except Exception:
            pass

        # OCR 요청 가져오기 (블로킹, 타임아웃 0.1초)
        try:
            request = ocr_queue.get(timeout=0.1)
        except Exception:
            continue

        try:
            result = _process_single_ocr(
                request, ocr_engines, preprocessor, validator,
                crnn_model, hangul_clf, kr_allowlist
            )
            result_queue.put(result)
            processed_count += 1

            if result["best_text"]:
                print(f"  [CMD6] #{result['frame_id']:04d}/{result['det_index']} "
                      f"→ {result['best_text']} (conf={result['best_conf']:.2f}, "
                      f"{result['ocr_ms']:.0f}ms)")
        except Exception as e:
            print(f"[CMD6-OCR] 처리 오류: {e}")
            traceback.print_exc()
            # 오류 시 빈 결과 전송
            result_queue.put({
                "frame_id": request.get("frame_id", -1),
                "det_index": request.get("det_index", -1),
                "best_text": "",
                "best_conf": 0.0,
                "bbox": request.get("bbox", [0, 0, 0, 0]),
                "is_small_plate": request.get("is_small_plate", False),
                "det_w": request.get("det_w", 0),
                "is_green_plate": False,
                "plate_type": "",
                "vehicle_type": "",
                "plate_lines": 1,
                "plate_color": "흰색바탕_검은글씨",
                "is_valid_format": False,
                "ocr_ms": 0.0,
                "timestamp": time.time(),
            })

    print("[CMD6-OCR] 워커 프로세스 종료")
