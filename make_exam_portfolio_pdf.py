# -*- coding: utf-8 -*-
"""
시험 제출용 포트폴리오 PDF 생성
- 평가기준표 기반: 100점 만점
- 코드 + 결과 캡쳐 + 기능 설명 포함
- 실행: py make_exam_portfolio_pdf.py
"""
import os
import sys
import textwrap

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from PIL import Image as PILImage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(SCRIPT_DIR, "번호판인식_포트폴리오.pdf")
W, H = A4  # 595.27 x 841.89

# ── 한글 폰트 등록 ──
FONT_NAME = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
for path, name in [
    ("C:/Windows/Fonts/malgun.ttf", "Malgun"),
    ("C:/Windows/Fonts/malgunbd.ttf", "MalgunBold"),
]:
    if os.path.isfile(path):
        try:
            pdfmetrics.registerFont(TTFont(name, path))
        except Exception:
            pass

if "Malgun" in [f.fontName for f in pdfmetrics.getRegisteredFontNames() if hasattr(f, 'fontName')] or True:
    try:
        pdfmetrics.getFont("Malgun")
        FONT_NAME = "Malgun"
    except Exception:
        pass
    try:
        pdfmetrics.getFont("MalgunBold")
        FONT_BOLD = "MalgunBold"
    except Exception:
        FONT_BOLD = FONT_NAME

# ── 색상 ──
COLOR_PRIMARY = HexColor("#1a73e8")
COLOR_DARK = HexColor("#202124")
COLOR_GRAY = HexColor("#5f6368")
COLOR_LIGHT_BG = HexColor("#f8f9fa")
COLOR_GREEN = HexColor("#1e8e3e")
COLOR_ACCENT = HexColor("#ea4335")
COLOR_CODE_BG = HexColor("#f0f0f0")

# ── 이미지 경로 ──
IMG_DIR = os.path.join(SCRIPT_DIR, "zip_ref_images")
RESULT_IMAGES = [
    os.path.join(IMG_DIR, "KakaoTalk_20260225_093640981.jpg"),     # 서울바9203
    os.path.join(IMG_DIR, "KakaoTalk_20260225_093640981_01.jpg"),  # 경기91바6286
    os.path.join(IMG_DIR, "KakaoTalk_20260225_093640981_02.jpg"),  # 01나8060
]
# image/ 폴더 결과 이미지
IMG_FOLDER_IMAGES = [
    os.path.join(SCRIPT_DIR, "image", "img1.png.png"),
    os.path.join(SCRIPT_DIR, "image", "img2.png.png"),
    os.path.join(SCRIPT_DIR, "image", "img3.png.png"),
]


def draw_header(c, y, title, subtitle=None):
    """페이지 상단 헤더"""
    # 파란 상단바
    c.setFillColor(COLOR_PRIMARY)
    c.rect(0, H - 45, W, 45, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont(FONT_BOLD, 16)
    c.drawString(30, H - 32, "AI 번호판 인식 시스템 - 시험 포트폴리오")
    c.setFont(FONT_NAME, 9)
    c.drawRightString(W - 30, H - 32, "교통안전을 위한 AI 솔루션 | 2026-02-25")
    return H - 55


def draw_section_title(c, y, number, title, points=None):
    """섹션 제목 (번호 + 제목 + 배점)"""
    if y < 80:
        c.showPage()
        y = draw_header(c, H, "")
        y -= 10

    # 배경 박스
    c.setFillColor(HexColor("#e8f0fe"))
    c.roundRect(25, y - 22, W - 50, 28, 4, fill=1, stroke=0)

    c.setFillColor(COLOR_PRIMARY)
    c.setFont(FONT_BOLD, 13)
    c.drawString(35, y - 16, f"{number}. {title}")

    if points:
        c.setFillColor(COLOR_ACCENT)
        c.setFont(FONT_BOLD, 11)
        c.drawRightString(W - 35, y - 16, f"[배점: {points}]")

    c.setFillColor(COLOR_DARK)
    return y - 35


def draw_body_text(c, y, text, font_size=10, indent=35, bold=False):
    """본문 텍스트"""
    if y < 60:
        c.showPage()
        y = draw_header(c, H, "")
        y -= 10
    c.setFont(FONT_BOLD if bold else FONT_NAME, font_size)
    c.setFillColor(COLOR_DARK)
    c.drawString(indent, y, text)
    return y - (font_size + 5)


def draw_code_block(c, y, code_lines, title=None):
    """코드 블록 (파일명 + 코드)"""
    line_h = 10
    needed_h = len(code_lines) * line_h + 30
    if y - needed_h < 50:
        c.showPage()
        y = draw_header(c, H, "")
        y -= 10

    if title:
        c.setFont(FONT_BOLD, 9)
        c.setFillColor(COLOR_GRAY)
        c.drawString(40, y, f"# {title}")
        y -= 14

    block_h = len(code_lines) * line_h + 12
    c.setFillColor(COLOR_CODE_BG)
    c.roundRect(35, y - block_h + 5, W - 70, block_h, 3, fill=1, stroke=0)
    c.setStrokeColor(HexColor("#dadce0"))
    c.roundRect(35, y - block_h + 5, W - 70, block_h, 3, fill=0, stroke=1)

    c.setFont("Courier", 7.5)
    c.setFillColor(COLOR_DARK)
    code_y = y - 5
    for line in code_lines:
        # 줄이 너무 길면 잘라냄
        display_line = line[:95] + "..." if len(line) > 95 else line
        c.drawString(42, code_y, display_line)
        code_y -= line_h

    return code_y - 8


def draw_image(c, y, img_path, caption, max_w=260, max_h=200):
    """이미지 삽입"""
    if not os.path.isfile(img_path):
        return y
    if y - max_h - 30 < 50:
        c.showPage()
        y = draw_header(c, H, "")
        y -= 10

    try:
        pil = PILImage.open(img_path)
        iw, ih = pil.size
        ratio = min(max_w / iw, max_h / ih)
        dw, dh = int(iw * ratio), int(ih * ratio)

        img_x = (W - dw) / 2
        img_y = y - dh
        c.drawImage(ImageReader(pil), img_x, img_y, width=dw, height=dh)

        # 캡션
        c.setFont(FONT_NAME, 8)
        c.setFillColor(COLOR_GRAY)
        c.drawCentredString(W / 2, img_y - 14, caption)

        return img_y - 25
    except Exception as e:
        print(f"[이미지 오류] {img_path}: {e}")
        return y - 20


def draw_three_images_row(c, y, images_info):
    """3개 이미지를 한 줄에 배치"""
    max_h = 180
    if y - max_h - 40 < 50:
        c.showPage()
        y = draw_header(c, H, "")
        y -= 10

    img_w = (W - 80) / 3  # 각 이미지 폭
    start_x = 35
    lowest_y = y

    for i, (img_path, caption) in enumerate(images_info):
        if not os.path.isfile(img_path):
            continue
        try:
            pil = PILImage.open(img_path)
            iw, ih = pil.size
            ratio = min(img_w / iw, max_h / ih)
            dw, dh = int(iw * ratio), int(ih * ratio)

            img_x = start_x + i * (img_w + 8)
            img_y = y - dh
            c.drawImage(ImageReader(pil), img_x, img_y, width=dw, height=dh)

            c.setFont(FONT_NAME, 7)
            c.setFillColor(COLOR_GRAY)
            c.drawCentredString(img_x + dw / 2, img_y - 10, caption)

            lowest_y = min(lowest_y, img_y - 18)
        except Exception:
            pass

    return lowest_y - 10


def build_pdf():
    c = canvas.Canvas(OUT_PDF, pagesize=A4)

    # ================================================================
    # PAGE 1: 프로젝트 개요 + 파이프라인 (표지 삭제)
    # ================================================================
    y = draw_header(c, H, "")
    y = draw_section_title(c, y, 1, "프로젝트 개요")
    y = draw_body_text(c, y, "이동하는 차량들의 파일을 입력받아 차량의 차량번호를 인지하여 표시하는 AI 프로그램 구현", 10)
    y -= 5
    y = draw_body_text(c, y, "- 입력: 차량 이동 영상 (movie/hiway.mp4) 또는 이미지 파일")
    y = draw_body_text(c, y, "- 출력: 번호판 영역 탐지 + 번호판 텍스트 인식 + 화면 표시")
    y = draw_body_text(c, y, "- 기술: YOLO 객체탐지, PaddleOCR + EasyOCR 앙상블, OpenCV 영상처리")
    y -= 10

    y = draw_section_title(c, y, 2, "전체 파이프라인")
    pipeline = [
        "1. Roboflow에서 차량 이미지 업로드 → 번호판 영역 라벨링 (Bounding Box)",
        "2. 라벨링된 데이터를 train/val/test 비율로 데이터셋 생성 (data.yaml)",
        "3. YOLO 모델 학습 → 최적 모델 best.pt 생성 (mAP@50 = 98.4%)",
        "4. Python에서 best.pt 로드 → 입력 영상/이미지에서 차량 번호판 detect",
        "5. ROI (Region of Interest) 추출 → 번호판 주변 관심 영역 분리",
        "6. CROP → 번호판 영역만 크롭 + 업스케일 (OCR 정확도 향상)",
        "7. CLAHE → 적응형 히스토그램 평활화로 대비 향상",
        "8. Homography → 기울어진 번호판 원근 보정 (4점 변환)",
        "9. 다중 전처리 + PaddleOCR/EasyOCR 앙상블 → 번호판 텍스트 인식",
        "10. 검증/교정 → 한국 번호판 패턴 매칭 + 투표 → 최종 결과 화면 표시",
    ]
    for line in pipeline:
        y = draw_body_text(c, y, line, 9)
    y -= 5

    # 파이프라인 흐름도
    c.setFont(FONT_BOLD, 9)
    c.setFillColor(COLOR_PRIMARY)
    flow = "[영상입력] → [YOLO탐지] → [ROI] → [CROP] → [CLAHE] → [Homography] → [OCR앙상블] → [검증] → [화면표시]"
    c.drawCentredString(W / 2, y, flow)
    y -= 15

    c.showPage()

    # ================================================================
    # PAGE 3: 라벨링 + 데이터셋 + best.pt (5+5+5 = 15점)
    # ================================================================
    y = draw_header(c, H, "")
    y = draw_section_title(c, y, 3, "Roboflow 라벨링 작업", "5점")
    y = draw_body_text(c, y, "- Roboflow에서 차량 이미지를 업로드하여 Object Detection 프로젝트 생성")
    y = draw_body_text(c, y, "- 번호판 영역에 Bounding Box로 라벨링 작업 수행")
    y = draw_body_text(c, y, "- 라벨링 포맷: YOLO v5/v8 형식 (class x_center y_center width height)")
    y -= 10

    y = draw_section_title(c, y, 4, "데이터셋 생성", "5점")
    y = draw_body_text(c, y, "- 라벨링된 자료를 적절한 비율로 train/val/test 데이터셋 생성")
    y = draw_body_text(c, y, "- data.yaml 설정: train/val/test 경로 + 클래스 정의")

    code_dataset = [
        "# split_dataset.py - 데이터셋 분할",
        "# train: 70%, val: 20%, test: 10%",
        "data_yaml = {",
        "    'train': './train/images',",
        "    'val':   './valid/images',",
        "    'test':  './test/images',",
        "    'nc': 1,  # 클래스 수: 번호판 1개",
        "    'names': ['plate']",
        "}",
    ]
    y = draw_code_block(c, y, code_dataset, "data.yaml 구조")
    y -= 5

    y = draw_section_title(c, y, 5, "best.pt 모델 학습/생성", "5점")
    y = draw_body_text(c, y, "- YOLO를 이용하여 학습자료를 학습시켜 best.pt 모델 생성")
    y = draw_body_text(c, y, "- 학습 결과: mAP@50 = 98.4% (yolo11x_plate.pt)")

    code_train = [
        "# train.py - YOLO 모델 학습",
        "from ultralytics import YOLO",
        "",
        "model = YOLO('yolo11x.pt')  # 사전학습 모델 로드",
        "results = model.train(",
        "    data='data.yaml',",
        "    epochs=100,",
        "    imgsz=640,",
        "    batch=16,",
        "    name='plate_detector'",
        ")",
        "# 결과: runs/detect/plate_detector/weights/best.pt",
    ]
    y = draw_code_block(c, y, code_train, "train.py")
    y -= 5

    y = draw_section_title(c, y, 6, "입력파일 차량 detect 표시", "5점")
    y = draw_body_text(c, y, "- best.pt 모델로 외부파일에서 받아온 차량의 객체를 인지")
    y = draw_body_text(c, y, "- Bounding Box로 탐지된 번호판 영역 화면 표시")

    code_detect = [
        "# exam_realtime_plate.py - YOLO 탐지",
        "def load_yolo_model():",
        "    for m in MODEL_PRIORITY:",
        "        p = os.path.join(SCRIPT_DIR, m)",
        "        if os.path.exists(p):",
        "            return YOLO(p)",
        "",
        "# 프레임별 탐지 실행",
        "detections = self.model(frame, conf=0.40, imgsz=1280, verbose=False)",
        "for det in detections[0].boxes:",
        "    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())",
        "    cv2.rectangle(display, (x1,y1), (x2,y2), (0,230,70), 3)",
    ]
    y = draw_code_block(c, y, code_detect, "exam_realtime_plate.py - 차량 detect")

    c.showPage()

    # ================================================================
    # PAGE 4: ROI + CROP + CLAHE + Homography (5+5+5+5 = 20점)
    # ================================================================
    y = draw_header(c, H, "")
    y = draw_section_title(c, y, 7, "ROI 함수 (Region of Interest)", "5점")
    y = draw_body_text(c, y, "- 탐지된 번호판 bbox 주변에 마진을 주어 관심 영역 추출")

    code_roi = [
        "# exam_realtime_plate.py - [2] ROI 함수 (시험 채점 항목 5번 - 5점)",
        "def apply_roi(frame, bbox, margin_x_ratio=0.25, margin_y_ratio=0.30):",
        '    """ROI(Region of Interest) 추출"""',
        "    h, w = frame.shape[:2]",
        "    x1, y1, x2, y2 = bbox",
        "    margin_x = int((x2 - x1) * margin_x_ratio)",
        "    margin_y = int((y2 - y1) * margin_y_ratio)",
        "    roi_x1 = max(0, x1 - margin_x)",
        "    roi_y1 = max(0, y1 - margin_y)",
        "    roi_x2 = min(w, x2 + margin_x)",
        "    roi_y2 = min(h, y2 + margin_y)",
        "    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]",
        "    return roi, (roi_x1, roi_y1, roi_x2, roi_y2)",
    ]
    y = draw_code_block(c, y, code_roi, "exam_realtime_plate.py:105-124")
    y -= 5

    y = draw_section_title(c, y, 8, "CROP 함수 (번호판 크롭 + 업스케일)", "5점")
    y = draw_body_text(c, y, "- 번호판 영역을 정확히 잘라내고, 작은 번호판은 업스케일하여 OCR 정확도 향상")

    code_crop = [
        "# exam_realtime_plate.py - [3] crop 함수 (시험 채점 항목 6번 - 5점)",
        "def crop_plate(frame, bbox):",
        '    """번호판 영역 크롭 + 업스케일"""',
        "    x1, y1, x2, y2 = bbox",
        "    cropped = frame[y1:y2, x1:x2]",
        "    if cropped.size == 0: return None",
        "    crop_h, crop_w = cropped.shape[:2]",
        "    if crop_w < 300 and crop_w > 0:",
        "        scale = 300.0 / crop_w",
        "        cropped = cv2.resize(cropped, None, fx=scale, fy=scale,",
        "                             interpolation=cv2.INTER_CUBIC)",
        "        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)",
        "        cropped = cv2.filter2D(cropped, -1, kernel)  # 선명화",
        "    return cropped",
    ]
    y = draw_code_block(c, y, code_crop, "exam_realtime_plate.py:130-154")
    y -= 5

    y = draw_section_title(c, y, 9, "CLAHE 함수 (대비 향상)", "5점")
    y = draw_body_text(c, y, "- 적응형 히스토그램 평활화로 어두운/밝은 환경에서 번호판 가독성 개선")

    code_clahe = [
        "# exam_realtime_plate.py - [4] CLAHE 함수 (시험 채점 항목 7번 - 5점)",
        "def apply_clahe(image, clip_limit=4.0, tile_size=(8, 8)):",
        '    """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""',
        "    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)",
        "    l_channel, a_channel, b_channel = cv2.split(lab)",
        "    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)",
        "    l_enhanced = clahe.apply(l_channel)",
        "    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])",
        "    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)",
        "    return enhanced_bgr",
    ]
    y = draw_code_block(c, y, code_clahe, "exam_realtime_plate.py:160-174")

    c.showPage()

    # ================================================================
    # PAGE 5: Homography + OCR 앙상블
    # ================================================================
    y = draw_header(c, H, "")
    y = draw_section_title(c, y, 10, "Homography 함수 (원근 보정)", "5점")
    y = draw_body_text(c, y, "- 기울어진 번호판을 정면으로 보정 (에지 검출 → 윤곽선 → 4점 변환)")

    code_homo = [
        "# exam_realtime_plate.py - [5] Homography 함수 (시험 채점 항목 8번 - 5점)",
        "def apply_homography(image):",
        '    """Homography 변환 (원근 보정)"""',
        "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)",
        "    blurred = cv2.GaussianBlur(gray, (5, 5), 0)",
        "    edges = cv2.Canny(blurred, 50, 150)",
        "    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,",
        "                                    cv2.CHAIN_APPROX_SIMPLE)",
        "    largest = max(contours, key=cv2.contourArea)",
        "    approx = cv2.approxPolyDP(largest, epsilon, True)",
        "    if len(approx) == 4:  # 4점 원근 변환",
        "        M = cv2.getPerspectiveTransform(ordered, dst)",
        "        warped = cv2.warpPerspective(image, M, (w_img, h_img))",
        "        return warped",
        "    else:  # Hough 변환으로 기울기 보정",
        "        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, ...)",
        "        M = cv2.getRotationMatrix2D((w/2, h/2), median_angle, 1.0)",
        "        return cv2.warpAffine(image, M, (w, h))",
    ]
    y = draw_code_block(c, y, code_homo, "exam_realtime_plate.py:180-239")
    y -= 5

    y = draw_section_title(c, y, 11, "OCR 앙상블 + 번호판 텍스트 인식")
    y = draw_body_text(c, y, "- PaddleOCR + EasyOCR 듀얼 엔진으로 한국어 번호판 인식")
    y = draw_body_text(c, y, "- 다중 전처리 (original, CLAHE, homography+CLAHE, 이진화, 감마) + 투표")

    code_ocr = [
        "# OCR 앙상블 엔진",
        "class OCREngine:",
        "    def __init__(self):",
        "        self.engines['paddle'] = PaddleOCR(lang='korean', ...)",
        "        self.engines['easyocr'] = easyocr.Reader(['ko', 'en'], gpu=True)",
        "",
        "    def run_ocr(self, image):",
        "        results = []",
        "        for name, engine in self.engines.items():",
        "            text, conf = self._run_single(name, engine, image)",
        "            if text and conf > 0.2:",
        "                results.append({'engine':name, 'text':text, 'conf':conf})",
        "        return results",
        "",
        "# 프레임 처리: ROI → crop → CLAHE → Homography → 다중전처리 → OCR",
        "variants = preprocess_variants(homo_img)  # 6가지 전처리",
        "for method_name, processed in variants:",
        "    ocr_results = self.ocr.run_ocr(processed)  # 각 엔진 결과",
        "counter = Counter(t for t, c in all_candidates)",
        "best_text = counter.most_common(1)[0][0]  # 투표로 최종 선택",
    ]
    y = draw_code_block(c, y, code_ocr, "exam_realtime_plate.py - OCR 앙상블")

    c.showPage()

    # ================================================================
    # PAGE 6: 차량번호 인식 결과 (20점) - 이미지 포함
    # ================================================================
    y = draw_header(c, H, "")
    y = draw_section_title(c, y, 12, "차량번호 인식 표시 결과", "20점")
    y = draw_body_text(c, y, "- 실제 고속도로 CCTV 영상에서 이동 차량의 번호판을 실시간 인식한 결과:", 10)
    y -= 5

    # 결과 이미지 3장 한 줄 배치 (image/ 폴더 실제 인식 결과)
    images_info = [
        (IMG_FOLDER_IMAGES[0], "img1 - 서울바9203 (72%)"),
        (IMG_FOLDER_IMAGES[1], "img2 - 경기91바6286 (79%)"),
        (IMG_FOLDER_IMAGES[2], "img3 - 01나8060 (71%)"),
    ]
    y = draw_three_images_row(c, y, images_info)
    y -= 5

    # 결과 텍스트 요약
    y = draw_body_text(c, y, "인식된 번호판 결과:", 10, bold=True)
    results_data = [
        ("서울바9203", "72%", "구형 지역번호판 (서울)"),
        ("경기91바6286", "79%", "구형 지역번호판 (경기)"),
        ("01나8060", "71%", "신형 번호판"),
    ]

    # 결과 테이블
    table_y = y - 5
    c.setFillColor(COLOR_PRIMARY)
    c.rect(40, table_y - 2, W - 80, 18, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont(FONT_BOLD, 9)
    c.drawString(55, table_y + 2, "번호판")
    c.drawString(200, table_y + 2, "신뢰도")
    c.drawString(310, table_y + 2, "유형")

    table_y -= 20
    for plate, conf, ptype in results_data:
        c.setFillColor(COLOR_LIGHT_BG if results_data.index((plate, conf, ptype)) % 2 == 0 else white)
        c.rect(40, table_y - 3, W - 80, 17, fill=1, stroke=0)
        c.setFillColor(COLOR_DARK)
        c.setFont(FONT_BOLD, 10)
        c.drawString(55, table_y + 1, plate)
        c.setFont(FONT_NAME, 10)
        c.setFillColor(COLOR_GREEN)
        c.drawString(200, table_y + 1, conf)
        c.setFillColor(COLOR_GRAY)
        c.drawString(310, table_y + 1, ptype)
        table_y -= 18

    y = table_y - 10

    # 화면 표시 코드
    y = draw_body_text(c, y, "화면 표시 코드 (한글 번호판 오버레이):", 10, bold=True)
    code_display = [
        "# 결과 화면 표시",
        "def draw_results(self, frame, results):",
        "    for r in results:",
        "        x1, y1, x2, y2 = r['bbox']",
        "        plate = r['plate']",
        "        conf = r['confidence']",
        "        # 초록색 Bounding Box",
        "        cv2.rectangle(display, (x1,y1), (x2,y2), (0,230,70), 3)",
        "        # 한글 번호판 텍스트 오버레이 (PIL)",
        "        label = f'{plate} ({conf:.0%})'",
        "        display = draw_text_korean(display, label, (x1, y1-font_sz-8))",
        "        # 신뢰도 바",
        "        bar_w = int((x2-x1) * conf)",
        "        cv2.rectangle(display, (x1,y2+2), (x1+bar_w,y2+8), (0,230,70), -1)",
    ]
    y = draw_code_block(c, y, code_display, "exam_realtime_plate.py:501-522 - 결과 화면표시")

    c.showPage()

    # ================================================================
    # PAGE 7: 웹 서버 + 실시간 시스템
    # ================================================================
    y = draw_header(c, H, "")
    y = draw_section_title(c, y, 13, "웹 서버 기반 실시간 인식 시스템 (추가 구현)")

    y = draw_body_text(c, y, "- Flask 웹 서버를 통한 영상 업로드 + 실시간 인식 결과 표시")
    y = draw_body_text(c, y, "- http://localhost:5000 에서 웹 UI를 통해 번호판 인식 가능")
    y -= 5

    code_server = [
        "# plate_server.py - Flask 웹 서버",
        "from flask import Flask, request, jsonify, Response",
        "from plate_recognition_4k import PlateRecognizer",
        "",
        "app = Flask(__name__)",
        "",
        "@app.route('/api/upload', methods=['POST'])",
        "def upload_video():",
        "    file = request.files['file']",
        "    # 영상 저장 → 백그라운드 인식 시작",
        "    recognizer = PlateRecognizer(model_path='best.pt')",
        "    results = recognizer.process_video(filepath)",
        "    return jsonify({'plates': results})",
        "",
        "# 실시간 스트리밍 인식",
        "@app.route('/stream/realtime')",
        "def stream_realtime():",
        "    return Response(generate_frames(), mimetype='multipart/x-mixed-replace')",
    ]
    y = draw_code_block(c, y, code_server, "plate_server.py - 웹 서버")
    y -= 10

    y = draw_section_title(c, y, 14, "사용된 핵심 파일 목록")
    files_info = [
        ("exam_realtime_plate.py", "시험용 실시간 번호판 인식 (ROI/crop/CLAHE/Homography 포함)"),
        ("plate_engine_pro.py", "상용급 번호판 인식 엔진 (고급 파이프라인)"),
        ("plate_server.py", "Flask 웹 서버 (업로드 + 실시간 스트림)"),
        ("plate_recognition_4k.py", "4K 고해상도 번호판 인식 + 한글 교정"),
        ("plate_ocr_postfilter_v2.py", "OCR 후처리 v2 (clean + ensemble vote)"),
        ("train.py", "YOLO 모델 학습 스크립트"),
        ("split_dataset.py", "데이터셋 분할 (train/val/test)"),
        ("auto_label.py", "자동 라벨링 도구"),
    ]

    for fname, desc in files_info:
        c.setFont(FONT_BOLD, 9)
        c.setFillColor(COLOR_PRIMARY)
        c.drawString(45, y, fname)
        c.setFont(FONT_NAME, 8)
        c.setFillColor(COLOR_GRAY)
        c.drawString(220, y, desc)
        y -= 15

    y -= 10

    # 실행 방법
    y = draw_section_title(c, y, 15, "실행 방법")
    code_run = [
        "# 1. 실시간 번호판 인식 실행",
        "python exam_realtime_plate.py",
        "python exam_realtime_plate.py --video movie/hiway.mp4",
        "",
        "# 2. 웹 서버 실행 (http://localhost:5000)",
        "python plate_server.py",
        "",
        "# 3. 이미지 인식 테스트",
        "python test_plate.py",
    ]
    y = draw_code_block(c, y, code_run, "실행 명령어")

    c.showPage()

    # ================================================================
    # PAGE 8: 전체 처리 흐름 + 마무리
    # ================================================================
    y = draw_header(c, H, "")
    y = draw_section_title(c, y, 16, "process_frame() - 전체 처리 흐름 코드")
    y = draw_body_text(c, y, "한 프레임이 처리되는 전체 과정 (ROI → crop → CLAHE → Homography → OCR):", 9)

    code_full = [
        "def process_frame(self, frame):",
        '    """한 프레임 처리: 탐지 → ROI → crop → CLAHE → homography → OCR"""',
        "    results = []",
        "    detections = self.model(frame, conf=DETECT_CONF, imgsz=1280)",
        "",
        "    for det in detections[0].boxes:",
        "        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())",
        "        bbox = (x1, y1, x2, y2)",
        "",
        "        # [ROI] 관심 영역 추출",
        "        roi_img, roi_coords = apply_roi(frame, bbox)",
        "",
        "        # [CROP] 번호판 크롭 + 업스케일",
        "        cropped = crop_plate(frame, bbox)",
        "",
        "        # [CLAHE] 대비 향상",
        "        clahe_img = apply_clahe(cropped)",
        "",
        "        # [Homography] 원근 보정",
        "        homo_img = apply_homography(clahe_img)",
        "",
        "        # 다중 전처리 + OCR 앙상블",
        "        variants = preprocess_variants(homo_img)",
        "        all_candidates = []",
        "        for method_name, processed in variants:",
        "            ocr_results = self.ocr.run_ocr(processed)",
        "            for r in ocr_results:",
        "                cleaned = clean_plate_text(r['text'])",
        "                is_valid, final_text = validate_plate(cleaned)",
        "                if is_valid and len(final_text) >= 6:",
        "                    all_candidates.append((final_text, r['conf']))",
        "",
        "        # 투표로 최종 결과 선택",
        "        if all_candidates:",
        "            counter = Counter(t for t, c in all_candidates)",
        "            best_text = counter.most_common(1)[0][0]",
        "            best_conf = max(c for t,c in all_candidates if t==best_text)",
        "            if best_conf >= OCR_CONF_MIN:",
        "                results.append({'plate':best_text, 'confidence':best_conf,",
        "                                'bbox':bbox})",
        "    return results",
    ]
    y = draw_code_block(c, y, code_full, "exam_realtime_plate.py:426-499 - process_frame()")
    y -= 10

    # 마무리
    y = draw_section_title(c, y, 17, "요약")
    summary_items = [
        "- YOLO (best.pt) 모델로 영상/이미지에서 번호판 영역 탐지 (detect)",
        "- ROI, CROP 함수로 번호판 관심 영역 추출 및 크롭",
        "- CLAHE 함수로 대비 향상 → 어두운 환경 대응",
        "- Homography 함수로 기울어진 번호판 원근 보정",
        "- PaddleOCR + EasyOCR 앙상블 + 다중 전처리로 한국어 번호판 텍스트 인식",
        "- 투표 기반 검증으로 최종 번호판 번호 확정 + 화면 표시",
        "- 인식 결과: 서울바9203(72%), 경기91바6286(79%), 01나8060(71%)",
    ]
    for item in summary_items:
        y = draw_body_text(c, y, item, 9)

    # 하단 서명
    y -= 25
    c.setStrokeColor(HexColor("#dadce0"))
    c.line(40, y + 10, W - 40, y + 10)
    c.setFont(FONT_NAME, 8)
    c.setFillColor(COLOR_GRAY)
    c.drawCentredString(W / 2, y - 5, "교통안전을 위한 AI 솔루션 | 포트폴리오 평가 (100점)")
    c.drawCentredString(W / 2, y - 18, "2026-02-25 제출")

    c.showPage()
    c.save()
    print(f"[완료] PDF 생성: {OUT_PDF}")
    return OUT_PDF


if __name__ == "__main__":
    path = build_pdf()
    print(f"\n파일 위치: {path}")
