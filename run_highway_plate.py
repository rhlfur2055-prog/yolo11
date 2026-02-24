# -*- coding: utf-8 -*-
"""
run_highway_plate.py - 저해상도 톨게이트 영상 번호판 인식 전용 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
640x360 하이패스 CCTV 영상에서 번호판을 최대한 인식하기 위한 특화 파이프라인.

핵심 전략:
1. ROI 집중: 톨게이트 차선 영역만 크롭 (불필요 영역 제거)
2. 초공격적 업스케일: 번호판 영역 8~10x 확대
3. 18단계 전처리: CLAHE, 샤프닝, 적응형 이진화, 형태학 등
4. EasyOCR 한국어 + 패턴 검증
5. 시간축 투표: 동일 차량 다중 프레임 합산

사용법:
    python run_highway_plate.py
"""

import os
import sys
import re
import time
import json
from collections import Counter, defaultdict
from datetime import datetime

# Windows 콘솔 한글 깨짐 방지
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "temp_youtube", "highway_plate.mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "plate_results_highway")

# ROI: 톨게이트 차선 중앙 영역 (정규화 좌표)
# 차량이 진입하는 도로 영역에 집중
ROI_X1, ROI_Y1 = 0.20, 0.05   # 좌상단
ROI_X2, ROI_Y2 = 0.75, 0.95   # 우하단

# 업스케일 설정
UPSCALE_FACTOR = 8        # 번호판 크롭 확대 배율
MIN_PLATE_WIDTH = 15      # 최소 번호판 폭 (저해상도이므로 매우 낮게)
MIN_PLATE_HEIGHT = 8

# YOLO 설정
DETECT_CONF = 0.30        # 저해상도이므로 탐지 임계값 낮춤
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# OCR 설정
MIN_OCR_CONF = 0.15       # 저해상도이므로 OCR 임계값도 낮춤

# 한국 번호판 패턴
KR_PATTERNS = [
    re.compile(r"\d{2,3}[가-힣]\d{4}"),          # 신형: 123가4567
    re.compile(r"[가-힣]{2}\d{1,2}[가-힣]\d{4}"), # 구형: 서울12가1234
    re.compile(r"[가-힣]{2,3}\d{1,2}[가-힣]\d{4}"),# 지역포함
    re.compile(r"\d{2,3}[가-힣]\d{3}"),           # 부분인식
    re.compile(r"\d{4,}"),                         # 숫자만 4자리+
    re.compile(r"[가-힣]+\d{2,}"),                 # 한글+숫자
    re.compile(r"\d{2,}[가-힣]+\d{2,}"),           # 숫자+한글+숫자
]

# 한글 OCR 오인식 보정 맵
HANGUL_CONFUSE = {
    "2": "가", "7": "나", "4": "라", "0": "오",
    "3": "가", "1": "이", "5": "마", "6": "바",
    "8": "바", "9": "자",
    "A": "아", "B": "바", "C": "소", "D": "다",
    "E": "어", "H": "하", "J": "자", "K": "카",
    "L": "나", "M": "마", "N": "나", "O": "오",
    "P": "파", "R": "라", "S": "서", "T": "다",
}

KOREAN_PLATE_HANGUL = (
    "가나다라마바사아자차카타파하"
    "거너더러머버서어저처커터퍼허"
    "고노도로모보소오조초코토포호"
    "구누두루무부수우주추쿠투푸후"
    "배"
    "서울부산대구인천광주대전울산세종"
    "경기강원충북충남전북전남경북경남제주"
)
ALLOWLIST = "0123456789" + KOREAN_PLATE_HANGUL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 이미지 전처리 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_preprocessed_variants(plate_crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """번호판 크롭에 대해 다양한 전처리 변형 생성"""
    variants = []
    h, w = plate_crop.shape[:2]

    # 0. 대폭 업스케일 (저해상도 핵심)
    scale = max(UPSCALE_FACTOR, 300 // max(w, 1))
    upscaled = cv2.resize(plate_crop, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    variants.append(("upscaled", upscaled))

    # 그레이스케일 기반 변형
    if len(upscaled.shape) == 3:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = upscaled.copy()

    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    variants.append(("clahe", cl))

    # 2. CLAHE + 샤프닝
    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(cl, -1, kernel_sharp)
    variants.append(("clahe_sharp", sharp))

    # 3. Otsu 이진화
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    # 4. Otsu 반전
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(("otsu_inv", otsu_inv))

    # 5. 적응형 평균 이진화
    adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 15, 5)
    variants.append(("adaptive_mean", adaptive_mean))

    # 6. 적응형 가우시안 이진화
    adaptive_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, 5)
    variants.append(("adaptive_gauss", adaptive_gauss))

    # 7. 양방향 필터 + CLAHE
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    bl_clahe = clahe.apply(bilateral)
    variants.append(("bilateral_clahe", bl_clahe))

    # 8. 감마 보정 (밝게)
    gamma = 0.6
    inv_gamma = 1.0 / gamma
    table = np.array([min(255, int((i / 255.0) ** inv_gamma * 255)) for i in range(256)], dtype=np.uint8)
    gamma_img = cv2.LUT(gray, table)
    variants.append(("gamma_bright", gamma_img))

    # 9. 감마 보정 (어둡게) + CLAHE
    gamma2 = 1.8
    table2 = np.array([min(255, int((i / 255.0) ** (1.0/gamma2) * 255)) for i in range(256)], dtype=np.uint8)
    gamma_dark = cv2.LUT(gray, table2)
    gd_clahe = clahe.apply(gamma_dark)
    variants.append(("gamma_dark_clahe", gd_clahe))

    # 10. 형태학 열림 (노이즈 제거)
    kernel_morph = np.ones((2, 2), np.uint8)
    morph_open = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel_morph)
    variants.append(("morph_open", morph_open))

    # 11. 형태학 닫힘 (글자 연결)
    morph_close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_morph)
    variants.append(("morph_close", morph_close))

    # 12. 히스토그램 균등화
    hist_eq = cv2.equalizeHist(gray)
    variants.append(("hist_eq", hist_eq))

    # 13. 미디언 필터 + 샤프닝
    median = cv2.medianBlur(gray, 3)
    med_sharp = cv2.filter2D(median, -1, kernel_sharp)
    variants.append(("median_sharp", med_sharp))

    # 14. 언샤프 마스크
    gaussian = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
    variants.append(("unsharp_mask", unsharp))

    # 15. 대비 스트레칭
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 > p2:
        stretched = np.clip((gray.astype(float) - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    else:
        stretched = gray.copy()
    variants.append(("contrast_stretch", stretched))

    # 16. 대비 스트레칭 + 이진화
    _, stretch_bin = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("stretch_otsu", stretch_bin))

    # 17. 슈퍼 샤프닝 (강한 커널)
    kernel_super = np.array([[-1, -1, -1, -1, -1],
                              [-1,  2,  2,  2, -1],
                              [-1,  2,  8,  2, -1],
                              [-1,  2,  2,  2, -1],
                              [-1, -1, -1, -1, -1]]) / 8.0
    super_sharp = cv2.filter2D(cl, -1, kernel_super)
    variants.append(("super_sharp", super_sharp))

    return variants


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 번호판 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_plate(text: str) -> tuple[bool, str, float]:
    """한국 번호판 패턴 검증"""
    cleaned = re.sub(r"[^가-힣0-9A-Za-z]", "", text)
    if not cleaned or len(cleaned) < 4:
        return False, "", 0.0

    scores = [1.0, 0.95, 0.90, 0.88, 0.45, 0.40, 0.35]
    for i, pat in enumerate(KR_PATTERNS):
        m = pat.search(cleaned)
        if m:
            score = scores[i] if i < len(scores) else 0.3
            return True, m.group(), score

    # 패턴 불일치여도 길이 기반 점수
    fallback = min(len(cleaned) / 10.0, 0.25)
    return False, cleaned, fallback


def correct_plate_hangul(text: str) -> str:
    """신형 번호판(숫자2-3 + 한글1 + 숫자4) 한글 오인식 보정"""
    m = re.match(r"^(\d{2,3})([가-힣0-9A-Z])(\d{4})$", text)
    if not m:
        return text
    prefix, mid, suffix = m.groups()
    # 이미 한글이면 그대로
    if '\uAC00' <= mid <= '\uD7A3':
        return text
    # 오인식 보정
    corrected = HANGUL_CONFUSE.get(mid.upper(), mid)
    return prefix + corrected + suffix


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  저해상도 톨게이트 번호판 인식 시스템")
    print("  (640x360 하이패스 CCTV 최적화)")
    print("=" * 60)

    if not os.path.isfile(VIDEO_PATH):
        print(f"[오류] 영상 파일 없음: {VIDEO_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    crops_dir = os.path.join(OUTPUT_DIR, "crops")
    preprocessed_dir = os.path.join(OUTPUT_DIR, "preprocessed")
    frames_dir = os.path.join(OUTPUT_DIR, "annotated_frames")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # ── YOLO 모델 로드 ──
    print("\n[1/4] YOLO 모델 로딩...")
    from ultralytics import YOLO

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_candidates = [
        os.path.join(script_dir, "yolo11x_plate.pt"),
        os.path.join(script_dir, "yolo11n_plate.pt"),
        os.path.join(script_dir, "yolo26n.pt"),
        os.path.join(script_dir, "yolo26.pt"),
        os.path.join(script_dir, "yolo11n.pt"),
        "yolov8n.pt",  # 자동 다운로드 폴백
    ]

    model = None
    model_name = ""
    is_plate_model = False
    for mp in model_candidates:
        try:
            model = YOLO(mp)
            model_name = os.path.basename(mp)
            is_plate_model = "plate" in mp.lower()
            print(f"  모델 로드 성공: {model_name} (번호판전용={is_plate_model})")
            break
        except Exception:
            continue

    if model is None:
        print("[오류] YOLO 모델 로드 실패")
        return

    # ── EasyOCR 로드 ──
    print("\n[2/4] EasyOCR 로딩 (한국어+영어)...")
    import easyocr
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    print("  EasyOCR 준비 완료")

    # ── 영상 열기 ──
    print(f"\n[3/4] 영상 처리 시작: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  해상도: {vid_w}x{vid_h} @ {fps:.1f}fps, 총 {total_frames}프레임")

    # ROI 계산 (픽셀 좌표)
    roi_px1 = int(vid_w * ROI_X1)
    roi_py1 = int(vid_h * ROI_Y1)
    roi_px2 = int(vid_w * ROI_X2)
    roi_py2 = int(vid_h * ROI_Y2)
    print(f"  ROI: ({roi_px1},{roi_py1}) ~ ({roi_px2},{roi_py2})")

    # ── 프레임별 처리 ──
    print(f"\n[4/4] 프레임 분석 중...")

    all_detections = []
    plate_counter = Counter()   # 번호판 텍스트 → 등장 횟수
    plate_frames = defaultdict(list)  # 번호판 텍스트 → 프레임 번호 리스트
    plate_best_conf = {}        # 번호판 텍스트 → 최고 신뢰도
    plate_best_crop = {}        # 번호판 텍스트 → 가장 좋은 크롭 이미지

    frame_idx = 0
    skip = 2  # 2프레임마다 분석 (30fps → 15fps 분석)
    t_start = time.time()
    detection_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % skip != 0:
            continue

        # 진행률
        if frame_idx % 60 == 0:
            pct = frame_idx / total_frames * 100
            elapsed = time.time() - t_start
            print(f"  진행: {pct:.0f}% ({frame_idx}/{total_frames}) "
                  f"| 인식: {detection_count}건 | {elapsed:.1f}초")

        # ROI 크롭
        roi_frame = frame[roi_py1:roi_py2, roi_px1:roi_px2]

        # 밝기 보정
        gray_check = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        mean_bright = gray_check.mean()
        if mean_bright < 80:
            gamma = 1.5
            table = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                              for i in range(256)], dtype=np.uint8)
            roi_frame = cv2.LUT(roi_frame, table)

        # YOLO 탐지
        results = model(roi_frame, conf=DETECT_CONF, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # 번호판 전용 모델: 모든 탐지 사용
                # COCO 모델: 차량 클래스만
                if not is_plate_model and cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                bw, bh = x2 - x1, y2 - y1

                if is_plate_model:
                    # 번호판 직접 탐지 → 패딩 추가 후 크롭
                    pad_h = int(bw * 0.3)
                    pad_v = int(bh * 0.2)
                    cx1 = max(0, x1 - pad_h)
                    cy1 = max(0, y1 - pad_v)
                    cx2 = min(roi_frame.shape[1], x2 + pad_h)
                    cy2 = min(roi_frame.shape[0], y2 + pad_v)
                    plate_crop = roi_frame[cy1:cy2, cx1:cx2]

                    if plate_crop.size == 0:
                        continue

                    # OCR 수행
                    plate_text, ocr_conf = run_ocr_ensemble(reader, plate_crop)
                    if plate_text:
                        plate_text = correct_plate_hangul(plate_text)
                        is_valid, normalized, score = validate_plate(plate_text)

                        det = {
                            "frame": frame_idx,
                            "text": normalized if is_valid else plate_text,
                            "raw_text": plate_text,
                            "is_valid": is_valid,
                            "pattern_score": score,
                            "ocr_confidence": ocr_conf,
                            "detection_confidence": conf,
                            "bbox_roi": [x1, y1, x2, y2],
                            "method": "plate_direct",
                        }
                        all_detections.append(det)
                        key = normalized if is_valid else plate_text
                        plate_counter[key] += 1
                        plate_frames[key].append(frame_idx)
                        if ocr_conf > plate_best_conf.get(key, 0):
                            plate_best_conf[key] = ocr_conf
                            plate_best_crop[key] = plate_crop.copy()
                        detection_count += 1
                else:
                    # COCO 차량 탐지 → 하단 번호판 영역 추출
                    # 차량 하단 40%에서 번호판 탐색
                    vehicle_crop = roi_frame[y1:y2, x1:x2]
                    if vehicle_crop.size == 0 or bw < 30 or bh < 20:
                        continue

                    # 차량 하단 40% 크롭 (번호판은 보통 아래에 있음)
                    plate_region_y = int(bh * 0.5)
                    plate_region = vehicle_crop[plate_region_y:, :]
                    if plate_region.size == 0:
                        plate_region = vehicle_crop

                    # 전체 차량에서도 시도
                    for region_name, region in [("lower", plate_region), ("full", vehicle_crop)]:
                        plate_text, ocr_conf = run_ocr_ensemble(reader, region)
                        if plate_text:
                            plate_text = correct_plate_hangul(plate_text)
                            is_valid, normalized, score = validate_plate(plate_text)

                            if score >= 0.3 or is_valid:
                                det = {
                                    "frame": frame_idx,
                                    "text": normalized if is_valid else plate_text,
                                    "raw_text": plate_text,
                                    "is_valid": is_valid,
                                    "pattern_score": score,
                                    "ocr_confidence": ocr_conf,
                                    "detection_confidence": conf,
                                    "bbox_roi": [x1, y1, x2, y2],
                                    "method": f"vehicle_{region_name}",
                                }
                                all_detections.append(det)
                                key = normalized if is_valid else plate_text
                                plate_counter[key] += 1
                                plate_frames[key].append(frame_idx)
                                if ocr_conf > plate_best_conf.get(key, 0):
                                    plate_best_conf[key] = ocr_conf
                                    plate_best_crop[key] = region.copy()
                                detection_count += 1

                                # 어노테이션 프레임 저장 (처음 5건)
                                if detection_count <= 10:
                                    ann = frame.copy()
                                    abs_x1 = roi_px1 + x1
                                    abs_y1 = roi_py1 + y1
                                    abs_x2 = roi_px1 + x2
                                    abs_y2 = roi_py1 + y2
                                    cv2.rectangle(ann, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                                    label = f"{plate_text} ({ocr_conf:.2f})"
                                    cv2.putText(ann, label, (abs_x1, max(15, abs_y1 - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    cv2.imwrite(os.path.join(frames_dir, f"det_{detection_count:03d}_f{frame_idx}.png"), ann)

                                break  # 하나 찾으면 다음 차량으로

    cap.release()
    elapsed = time.time() - t_start

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 결과 정리 및 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'=' * 60}")
    print(f"  처리 완료!")
    print(f"{'=' * 60}")
    print(f"  총 프레임: {total_frames} | 분석 프레임: {frame_idx // skip}")
    print(f"  처리 시간: {elapsed:.1f}초")
    print(f"  총 탐지: {detection_count}건")

    # 2회 이상 등장한 번호판 = 확정
    confirmed = []
    for plate_text, count in plate_counter.most_common():
        is_valid, normalized, score = validate_plate(plate_text)
        frames_list = plate_frames[plate_text]
        best_conf = plate_best_conf.get(plate_text, 0)

        entry = {
            "plate": plate_text,
            "count": count,
            "is_valid_pattern": is_valid,
            "pattern_score": score,
            "best_ocr_confidence": best_conf,
            "first_frame": min(frames_list),
            "last_frame": max(frames_list),
            "frame_span": max(frames_list) - min(frames_list),
        }
        confirmed.append(entry)

        # 크롭 이미지 저장
        if plate_text in plate_best_crop:
            crop = plate_best_crop[plate_text]
            safe_name = re.sub(r'[^\w가-힣]', '_', plate_text)
            cv2.imwrite(os.path.join(crops_dir, f"{safe_name}_x{count}.png"), crop)

            # 전처리 변형도 저장 (상위 5개 번호판)
            if len(confirmed) <= 5:
                variants = generate_preprocessed_variants(crop)
                plate_pp_dir = os.path.join(preprocessed_dir, safe_name)
                os.makedirs(plate_pp_dir, exist_ok=True)
                for vname, vimg in variants:
                    cv2.imwrite(os.path.join(plate_pp_dir, f"{vname}.png"), vimg)

    # 결과 출력
    print(f"\n  === 인식된 번호판 목록 ===")
    for i, entry in enumerate(confirmed[:20], 1):
        valid_mark = "V" if entry["is_valid_pattern"] else " "
        print(f"  {i:2d}. [{valid_mark}] {entry['plate']:<15s} "
              f"x{entry['count']:2d}회  "
              f"점수={entry['pattern_score']:.2f}  "
              f"OCR={entry['best_ocr_confidence']:.2f}  "
              f"프레임={entry['first_frame']}-{entry['last_frame']}")

    # JSON 저장
    result_json = {
        "video": VIDEO_PATH,
        "resolution": f"{vid_w}x{vid_h}",
        "total_frames": total_frames,
        "analyzed_frames": frame_idx // skip,
        "processing_time_sec": round(elapsed, 1),
        "total_detections": detection_count,
        "confirmed_plates": confirmed,
        "all_detections": all_detections,
    }

    json_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)

    # 요약 텍스트
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"저해상도 톨게이트 번호판 인식 결과\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"영상: {VIDEO_PATH}\n")
        f.write(f"해상도: {vid_w}x{vid_h}\n")
        f.write(f"처리시간: {elapsed:.1f}초\n")
        f.write(f"총 탐지: {detection_count}건\n\n")
        f.write(f"확정 번호판:\n")
        for i, entry in enumerate(confirmed, 1):
            f.write(f"  {i}. {entry['plate']} (x{entry['count']}, "
                    f"score={entry['pattern_score']:.2f})\n")

    print(f"\n  결과 저장: {OUTPUT_DIR}/")
    print(f"  - results.json: 전체 결과")
    print(f"  - summary.txt: 요약")
    print(f"  - crops/: 번호판 크롭 이미지")
    print(f"  - preprocessed/: 전처리 변형 이미지")
    print(f"  - annotated_frames/: 탐지 프레임")
    print(f"{'=' * 60}")


def run_ocr_ensemble(reader, plate_crop: np.ndarray) -> tuple[str, float]:
    """
    EasyOCR + 18단계 전처리 앙상블 투표.
    가장 많이 나온 텍스트를 반환.
    """
    if plate_crop.size == 0:
        return "", 0.0

    variants = generate_preprocessed_variants(plate_crop)
    text_counter = Counter()
    conf_map = {}

    for vname, vimg in variants:
        try:
            results = reader.readtext(
                vimg,
                allowlist=ALLOWLIST,
                detail=1,
                paragraph=False,
                min_size=5,
                text_threshold=0.3,
                low_text=0.3,
                width_ths=1.5,
            )
            for bbox, text, conf in results:
                cleaned = re.sub(r"[^가-힣0-9A-Za-z]", "", text)
                if len(cleaned) >= 3:
                    text_counter[cleaned] += 1
                    if conf > conf_map.get(cleaned, 0):
                        conf_map[cleaned] = conf
        except Exception:
            continue

    if not text_counter:
        return "", 0.0

    # 가장 많이 나온 텍스트
    best_text, best_count = text_counter.most_common(1)[0]
    best_conf = conf_map.get(best_text, 0)

    return best_text, best_conf


if __name__ == "__main__":
    main()
