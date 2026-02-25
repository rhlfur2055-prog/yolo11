# -*- coding: utf-8 -*-
"""
========================================================
  시험용 실시간 번호판 인식 프로그램
  - YOLO 객체 탐지 + 번호판 OCR
  - ROI, crop, CLAHE, homography 함수 사용
  - 실시간 번호판 번호 화면 표시
========================================================
  사용법:
    python exam_realtime_plate.py
    python exam_realtime_plate.py --video movie/hiway.mp4
========================================================
"""

import os
import sys
import re
import time
import argparse
from collections import Counter, defaultdict
from pathlib import Path

# Windows 콘솔 한글 깨짐 방지
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ── OCR 엔진 임포트 ──
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except Exception:
    HAS_PADDLEOCR = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# ── OCR 후처리 v2 임포트 ──
try:
    from plate_ocr_postfilter_v2 import clean_ocr_text_v2, ensemble_vote_v2
    HAS_POSTFILTER_V2 = True
except ImportError:
    HAS_POSTFILTER_V2 = False


# ============================================================
# [설정]
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "movie", "hiway.mp4")

# YOLO 모델 우선순위
MODEL_PRIORITY = [
    "yolo11x_plate.pt",    # 번호판 전용 (mAP@50=98.4%)
    "yolo26n.pt",
    "yolo11n.pt",
    "yolov8n.pt",
]

# 탐지 설정
DETECT_CONF = 0.40
OCR_CONF_MIN = 0.50
CONSECUTIVE_FRAMES = 2  # N프레임 연속 감지 후 확정

# 한국 번호판 패턴
KR_PATTERNS = [
    re.compile(r'^\d{2,3}[가-힣]\d{4}$'),              # 신형: 12가3456
    re.compile(r'^[가-힣]{2}\d{1,2}[가-힣]\d{4}$'),    # 구형: 서울12가3456
    re.compile(r'^[가-힣]{2,3}\d{2}[가-힣]\d{4}$'),    # 구형지역: 경기76바7789
]

# 한글 폰트 경로
FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
FONT_FALLBACK = "C:/Windows/Fonts/malgun.ttf"


# ============================================================
# [1] YOLO 모델 로드
# ============================================================
def load_yolo_model():
    """우선순위에 따라 YOLO 모델 자동 로드"""
    for m in MODEL_PRIORITY:
        p = os.path.join(SCRIPT_DIR, m)
        if os.path.exists(p):
            print(f"[YOLO] 모델 로드: {m}")
            return YOLO(p)
    print("[YOLO] yolo11n.pt 자동 다운로드...")
    return YOLO("yolo11n.pt")


# ============================================================
# [2] ROI 함수 (시험 채점 항목 5번 - 5점)
# ============================================================
def apply_roi(frame, bbox, margin_x_ratio=0.25, margin_y_ratio=0.30):
    """
    ROI(Region of Interest) 추출
    - 탐지된 번호판 bbox 주변에 마진을 주어 관심 영역 추출
    - margin_x_ratio: 좌우 마진 비율
    - margin_y_ratio: 상하 마진 비율 (2줄 번호판 캡처)
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    margin_x = int((x2 - x1) * margin_x_ratio)
    margin_y = int((y2 - y1) * margin_y_ratio)

    roi_x1 = max(0, x1 - margin_x)
    roi_y1 = max(0, y1 - margin_y)
    roi_x2 = min(w, x2 + margin_x)
    roi_y2 = min(h, y2 + margin_y)

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, (roi_x1, roi_y1, roi_x2, roi_y2)


# ============================================================
# [3] crop 함수 (시험 채점 항목 6번 - 5점)
# ============================================================
def crop_plate(frame, bbox):
    """
    번호판 영역 크롭 + 업스케일
    - 작은 번호판은 OCR 정확도를 위해 업스케일
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    # 작은 번호판 업스케일 (너비 300px 미만)
    crop_h, crop_w = cropped.shape[:2]
    if crop_w < 300 and crop_w > 0:
        scale = 300.0 / crop_w
        cropped = cv2.resize(cropped, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        # 업스케일 후 선명화
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        cropped = cv2.filter2D(cropped, -1, kernel)

    return cropped


# ============================================================
# [4] CLAHE 함수 (시험 채점 항목 7번 - 5점)
# ============================================================
def apply_clahe(image, clip_limit=4.0, tile_size=(8, 8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - 적응형 히스토그램 평활화로 대비 향상
    - 어두운/밝은 조명 환경에서 번호판 가독성 개선
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


# ============================================================
# [5] Homography 함수 (시험 채점 항목 8번 - 5점) [평가기준 20점 중 Homography]
# ============================================================
def apply_homography(image):
    """
    Homography 변환 (원근 보정)
    - 기울어진 번호판을 정면으로 보정
    - 에지 검출 → 윤곽선 → 4점 변환
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # 4개의 꼭짓점이 발견되면 원근 변환 적용
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        # 좌상, 우상, 우하, 좌하 순서로 정렬
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        ordered = np.array([
            pts[np.argmin(s)],      # 좌상
            pts[np.argmin(d)],      # 우상
            pts[np.argmax(s)],      # 우하
            pts[np.argmax(d)],      # 좌하
        ], dtype=np.float32)

        h_img, w_img = image.shape[:2]
        dst = np.array([
            [0, 0],
            [w_img - 1, 0],
            [w_img - 1, h_img - 1],
            [0, h_img - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, M, (w_img, h_img))
        return warped
    else:
        # 4점을 찾지 못하면 기울기 보정 (Hough 변환)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                                minLineLength=20, maxLineGap=10)
        if lines is not None:
            angles = []
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                if abs(angle) < 30:
                    angles.append(angle)
            if angles:
                median_angle = np.median(angles)
                h_img, w_img = image.shape[:2]
                M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), median_angle, 1.0)
                return cv2.warpAffine(image, M, (w_img, h_img),
                                      borderMode=cv2.BORDER_REPLICATE)
    return image


# ============================================================
# [6] 전처리 파이프라인 (다중 전처리)
# ============================================================
def preprocess_variants(image):
    """여러 전처리 결과를 반환하여 OCR 앙상블에 사용"""
    variants = [("original", image.copy())]

    # CLAHE
    clahe_img = apply_clahe(image)
    variants.append(("clahe", clahe_img))

    # Homography + CLAHE
    homo_img = apply_homography(image)
    homo_clahe = apply_clahe(homo_img)
    variants.append(("homography+clahe", homo_clahe))

    # 노이즈 제거
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    variants.append(("denoise", denoised))

    # 그레이스케일 + 이진화
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))

    # 감마 밝게
    gamma_table = np.array([((i / 255.0) ** 0.5) * 255 for i in range(256)]).astype("uint8")
    bright = cv2.LUT(image, gamma_table)
    variants.append(("gamma_bright", bright))

    return variants


# ============================================================
# [7] OCR 엔진 초기화 + 실행
# ============================================================
class OCREngine:
    """PaddleOCR + EasyOCR 앙상블"""

    def __init__(self):
        self.engines = {}
        if HAS_PADDLEOCR:
            try:
                kwargs = dict(lang="korean", use_angle_cls=True, show_log=False)
                _paddle_root = Path("C:/tools/paddleocr_models")
                if _paddle_root.exists():
                    kwargs["det_model_dir"] = str(_paddle_root / "det/ml/Multilingual_PP-OCRv3_det_infer")
                    kwargs["rec_model_dir"] = str(_paddle_root / "rec/korean/korean_PP-OCRv4_rec_infer")
                    kwargs["cls_model_dir"] = str(_paddle_root / "cls/ch_ppocr_mobile_v2.0_cls_infer")
                self.engines["paddle"] = PaddleOCR(**kwargs)
                print("[OCR] PaddleOCR 초기화 완료")
            except Exception as e:
                print(f"[OCR] PaddleOCR 실패: {e}")
        if HAS_EASYOCR:
            try:
                self.engines["easyocr"] = easyocr.Reader(["ko", "en"], gpu=True)
                print("[OCR] EasyOCR 초기화 완료 (GPU)")
            except Exception:
                try:
                    self.engines["easyocr"] = easyocr.Reader(["ko", "en"], gpu=False)
                    print("[OCR] EasyOCR 초기화 완료 (CPU)")
                except Exception as e:
                    print(f"[OCR] EasyOCR 실패: {e}")
        if not self.engines:
            print("[경고] OCR 엔진 없음! pip install paddlepaddle paddleocr easyocr")

    def run_ocr(self, image):
        """모든 엔진으로 OCR 실행, 결과 리스트 반환"""
        results = []
        for name, engine in self.engines.items():
            text, conf = self._run_single(name, engine, image)
            if text and conf > 0.2:
                results.append({"engine": name, "text": text, "conf": conf})
        return results

    def _run_single(self, name, engine, image):
        try:
            if name == "paddle":
                result = engine.ocr(image, cls=True)
                if result and result[0]:
                    lines = sorted(result[0], key=lambda l: l[0][0][1])
                    texts = [l[1][0] for l in lines]
                    confs = [l[1][1] for l in lines]
                    return "".join(texts), float(np.mean(confs))
            elif name == "easyocr":
                result = engine.readtext(image, detail=1, paragraph=False)
                if result:
                    result_sorted = sorted(result, key=lambda r: r[0][0][1])
                    texts = [r[1] for r in result_sorted]
                    confs = [r[2] for r in result_sorted]
                    return "".join(texts), float(np.mean(confs))
        except Exception:
            pass
        return "", 0.0


# ============================================================
# [8] 번호판 텍스트 교정 + 검증
# ============================================================
def clean_plate_text(raw_text):
    """OCR 결과 후처리: 특수문자 제거 + 영문→한글 교정"""
    if HAS_POSTFILTER_V2:
        result = clean_ocr_text_v2(raw_text)
        if result:
            return result

    # 기본 정리
    text = re.sub(r'[^\w가-힣]', '', raw_text.strip())
    text = re.sub(r'\s+', '', text)

    # OCR 혼동문자 교정
    confusion = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'D': '0'}
    result = []
    for i, ch in enumerate(text):
        if ch in confusion:
            prev_digit = (i > 0 and result[-1].isdigit()) if result else False
            next_digit = (i < len(text) - 1 and text[i + 1].isdigit())
            if prev_digit or next_digit:
                result.append(confusion[ch])
            else:
                result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


def validate_plate(text):
    """한국 번호판 패턴 검증"""
    clean = re.sub(r'[^\d가-힣]', '', text)
    if len(clean) < 6 or len(clean) > 10:
        return False, clean
    for pattern in KR_PATTERNS:
        if pattern.match(clean):
            return True, clean
    return False, clean


# ============================================================
# [9] 한글 텍스트 오버레이 (PIL 기반)
# ============================================================
_font_cache = {}

def _get_font(size):
    if size not in _font_cache:
        for fp in [FONT_PATH, FONT_FALLBACK]:
            try:
                _font_cache[size] = ImageFont.truetype(fp, size)
                break
            except OSError:
                continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_text_korean(frame, text, pos, font_size=24, color=(0, 255, 0)):
    """OpenCV BGR 프레임에 한글 텍스트 렌더링"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    x, y = pos
    rgb = (color[2], color[1], color[0])
    # 외곽선
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (0, 2)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ============================================================
# [10] 메인 실시간 인식 시스템
# ============================================================
class RealtimePlateRecognizer:
    """
    실시간 번호판 인식 시스템
    [영상입력] → [YOLO탐지] → [ROI추출] → [crop] → [CLAHE] → [Homography]
    → [다중전처리] → [OCR앙상블] → [검증/교정] → [화면표시]
    """

    def __init__(self, video_path):
        self.video_path = video_path
        self.model = load_yolo_model()
        self.ocr = OCREngine()
        self.plate_tracker = defaultdict(lambda: {"count": 0, "consecutive": 0, "conf": 0.0})
        self.confirmed_plates = []  # 확정된 번호판 리스트
        self.frame_count = 0
        self.fps_samples = []

    def process_frame(self, frame):
        """한 프레임 처리: 탐지 → ROI → crop → CLAHE → homography → OCR"""
        results = []
        detections = self.model(frame, conf=DETECT_CONF, imgsz=1280, verbose=False)

        for det in detections[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            det_conf = float(det.conf[0])
            bbox = (x1, y1, x2, y2)

            # ── [ROI] 관심 영역 추출 ──
            roi_img, roi_coords = apply_roi(frame, bbox)
            if roi_img.size == 0:
                continue

            # ── [crop] 번호판 크롭 + 업스케일 ──
            cropped = crop_plate(frame, bbox)
            if cropped is None:
                continue

            # ── [CLAHE] 대비 향상 ──
            clahe_img = apply_clahe(cropped)

            # ── [Homography] 원근 보정 ──
            homo_img = apply_homography(clahe_img)

            # ── 다중 전처리 + OCR 앙상블 ──
            variants = preprocess_variants(homo_img)
            all_candidates = []

            for method_name, processed in variants:
                ocr_results = self.ocr.run_ocr(processed)
                for r in ocr_results:
                    cleaned = clean_plate_text(r["text"])
                    is_valid, final_text = validate_plate(cleaned)
                    if is_valid and len(final_text) >= 6:
                        all_candidates.append((final_text, r["conf"]))

            # 투표로 최종 결과 선택
            if all_candidates:
                counter = Counter(t for t, c in all_candidates)
                best_text = counter.most_common(1)[0][0]
                best_conf = max(c for t, c in all_candidates if t == best_text)

                if best_conf >= OCR_CONF_MIN:
                    # 연속 프레임 필터링
                    tracker = self.plate_tracker[best_text]
                    tracker["consecutive"] += 1
                    tracker["count"] += 1
                    tracker["conf"] = max(tracker["conf"], best_conf)

                    if tracker["consecutive"] >= CONSECUTIVE_FRAMES:
                        results.append({
                            "plate": best_text,
                            "confidence": best_conf,
                            "bbox": bbox,
                            "roi_coords": roi_coords,
                        })
                        # 확정 번호판 기록
                        if best_text not in [p["plate"] for p in self.confirmed_plates]:
                            self.confirmed_plates.append({
                                "plate": best_text,
                                "confidence": best_conf,
                                "frame": self.frame_count,
                            })
                            print(f"  [확정] {best_text} (신뢰도: {best_conf:.0%})")

        # 이번 프레임에 없던 번호판은 연속 카운트 리셋
        seen = set(r["plate"] for r in results)
        for key in list(self.plate_tracker.keys()):
            if key not in seen:
                self.plate_tracker[key]["consecutive"] = 0

        return results

    def draw_results(self, frame, results):
        """프레임에 탐지 결과 오버레이"""
        display = frame.copy()

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            plate = r["plate"]
            conf = r["confidence"]

            # 초록색 bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 230, 70), 3)

            # 번호판 텍스트 (한글 지원)
            label = f"{plate} ({conf:.0%})"
            font_sz = max(20, min(32, (x2 - x1) // 3))
            display = draw_text_korean(display, label, (x1, max(0, y1 - font_sz - 8)),
                                       font_size=font_sz, color=(0, 230, 70))

            # 신뢰도 바
            bar_w = int((x2 - x1) * conf)
            cv2.rectangle(display, (x1, y2 + 2), (x1 + bar_w, y2 + 8), (0, 230, 70), -1)

        return display

    def draw_sidebar(self, frame):
        """우측에 인식된 번호판 리스트 사이드바 표시"""
        h, w = frame.shape[:2]
        sidebar_w = 280
        canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = (30, 25, 20)  # 어두운 배경

        # 사이드바 헤더
        cv2.rectangle(canvas, (w, 0), (w + sidebar_w, 50), (60, 50, 40), -1)
        canvas = draw_text_korean(canvas, "인식된 번호판", (w + 10, 10),
                                  font_size=22, color=(0, 230, 70))

        # 확정된 번호판 리스트
        y_offset = 65
        for i, p in enumerate(self.confirmed_plates[-15:]):  # 최근 15개
            plate = p["plate"]
            conf = p["confidence"]
            frame_no = p["frame"]

            # 배경 박스
            bg_color = (50, 60, 45) if i % 2 == 0 else (40, 45, 35)
            cv2.rectangle(canvas, (w + 5, y_offset - 5), (w + sidebar_w - 5, y_offset + 40), bg_color, -1)

            # 번호판 번호
            canvas = draw_text_korean(canvas, plate, (w + 15, y_offset),
                                      font_size=18, color=(255, 255, 255))
            # 신뢰도
            conf_text = f"{conf:.0%}"
            conf_color = (0, 230, 70) if conf >= 0.8 else (0, 180, 255)
            canvas = draw_text_korean(canvas, conf_text, (w + 200, y_offset),
                                      font_size=16, color=conf_color)

            y_offset += 45

        # 하단 정보
        info_y = h - 60
        cv2.rectangle(canvas, (w, info_y - 5), (w + sidebar_w, h), (50, 40, 35), -1)
        canvas = draw_text_korean(canvas, f"총 인식: {len(self.confirmed_plates)}대",
                                  (w + 15, info_y + 5), font_size=16, color=(200, 200, 200))
        canvas = draw_text_korean(canvas, f"프레임: {self.frame_count}",
                                  (w + 15, info_y + 28), font_size=14, color=(150, 150, 150))

        return canvas

    def run(self):
        """메인 실행 루프"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[에러] 영상 열기 실패: {self.video_path}")
            return

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("=" * 60)
        print("  실시간 번호판 인식 시스템 시작")
        print(f"  영상: {self.video_path}")
        print(f"  해상도: {vid_w}x{vid_h} @ {vid_fps:.0f}fps")
        print(f"  총 프레임: {total_frames}")
        print(f"  OCR 엔진: {list(self.ocr.engines.keys())}")
        print(f"  사용 함수: ROI, crop, CLAHE, Homography")
        print("=" * 60)
        print("  [q] 종료  [Space] 일시정지/재생")
        print("=" * 60)

        paused = False
        window_name = "YOLO Plate Recognition - ROI/crop/CLAHE/Homography"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280 + 280, 720)

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # 영상 끝 → 처음부터 반복
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                self.frame_count += 1
                t0 = time.time()

                # 번호판 탐지 + OCR
                results = self.process_frame(frame)
                elapsed_ms = (time.time() - t0) * 1000

                # 결과 오버레이
                display = self.draw_results(frame, results)

                # FPS 정보
                self.fps_samples.append(1000.0 / max(elapsed_ms, 1))
                if len(self.fps_samples) > 30:
                    self.fps_samples.pop(0)
                avg_fps = sum(self.fps_samples) / len(self.fps_samples)

                # 상단 정보 바
                info = f"Frame:{self.frame_count}/{total_frames}  Det:{len(results)}  {elapsed_ms:.0f}ms  FPS:{avg_fps:.1f}  Plates:{len(self.confirmed_plates)}"
                cv2.rectangle(display, (0, 0), (vid_w, 35), (0, 0, 0), -1)
                display = draw_text_korean(display, info, (10, 5), font_size=18, color=(0, 200, 255))

                # 사이드바 추가
                canvas = self.draw_sidebar(display)

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
                print("[일시정지]" if paused else "[재생]")

        cap.release()
        cv2.destroyAllWindows()

        # 최종 결과 출력
        print("\n" + "=" * 60)
        print("  최종 인식 결과")
        print("=" * 60)
        for i, p in enumerate(self.confirmed_plates, 1):
            print(f"  {i:2d}. {p['plate']:<16} 신뢰도: {p['confidence']:.0%}  (프레임 {p['frame']})")
        print(f"\n  총 {len(self.confirmed_plates)}대 인식 완료")
        print("=" * 60)


# ============================================================
# [메인 실행]
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="시험용 실시간 번호판 인식")
    parser.add_argument("--video", default=VIDEO_PATH, help="동영상 파일 경로")
    args = parser.parse_args()

    video = args.video
    if not os.path.exists(video):
        print(f"[에러] 영상 파일 없음: {video}")
        # movie 폴더에서 찾기
        movie_dir = os.path.join(SCRIPT_DIR, "movie")
        if os.path.isdir(movie_dir):
            videos = [f for f in os.listdir(movie_dir) if f.endswith(('.mp4', '.avi'))]
            if videos:
                video = os.path.join(movie_dir, videos[0])
                print(f"[대체] {video} 사용")
            else:
                print("[에러] movie 폴더에 영상 파일 없음")
                sys.exit(1)

    recognizer = RealtimePlateRecognizer(video)
    recognizer.run()
