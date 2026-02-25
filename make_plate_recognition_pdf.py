# -*- coding: utf-8 -*-
"""
번호판 인식 프로젝트 발표 PDF 생성
- 평가 기준(라벨링, 데이터셋, best.pt, detect, ROI/CROP/CLAHE/Homography, 차량번호 표시) 반영
- 실행: python make_plate_recognition_pdf.py
- 결과: 번호판인식_발표.pdf (프로젝트 루트)
"""
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

# 한글 폰트 (Windows)
def _get_canvas_font(c):
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        for path in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"]:
            if os.path.isfile(path):
                name = "KoreanFont"
                pdfmetrics.registerFont(TTFont(name, path))
                return name
    except Exception:
        pass
    return "Helvetica"


def draw_slide_title(c, y, text, font_size=22):
    font = _get_canvas_font(c)
    c.setFont(font, font_size)
    c.drawCentredString(A4[0] / 2, y, text)


def draw_slide_text(c, y, lines, font_size=11):
    font = _get_canvas_font(c)
    c.setFont(font, font_size)
    for line in lines:
        c.drawString(40, y, line)
        y -= (font_size + 4)
    return y


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "번호판인식_발표.pdf")
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4

    # Slide 1: 표지
    c.setFont(_get_canvas_font(c), 28)
    c.drawCentredString(w / 2, h - 80, "AI License Plate Recognition")
    c.setFont(_get_canvas_font(c), 18)
    c.drawCentredString(w / 2, h - 120, "Moving Vehicle Input -> Plate Recognition & Display")
    c.setFont(_get_canvas_font(c), 12)
    c.drawCentredString(w / 2, h - 160, "Regular Course / AI Solution for Traffic Safety")
    c.showPage()

    # Slide 2: 프로젝트 목표
    draw_slide_title(c, h - 50, "Project Goal", 20)
    draw_slide_text(c, h - 90, [
        "- Input: Moving vehicle files (video / images)",
        "- Output: Recognize and display license plate numbers",
        "- Path: C:\\tool\\yolo26-main\\image (e.g. img1.png, img2.png, img3.png)",
    ], 12)
    c.showPage()

    # Slide 3: 진행 순서
    draw_slide_title(c, h - 50, "Pipeline", 20)
    draw_slide_text(c, h - 85, [
        "1. Roboflow: Upload & label vehicle/plate images",
        "2. Dataset: Generate train/val/test from labeled data",
        "3. YOLO: Train model -> best.pt",
        "4. Python: Load best.pt, run detection on input files",
        "5. ROI / CROP: Extract plate region from frame",
        "6. CLAHE: Contrast enhancement for OCR",
        "7. Homography: Perspective correction (exam_realtime_plate.py)",
        "8. OCR: PaddleOCR + EasyOCR -> plate text",
        "9. Display: Bbox + recognized plate number on screen",
    ], 11)
    c.showPage()

    # Slide 4: 평가 기준 반영
    draw_slide_title(c, h - 50, "Evaluation Criteria (100 pts)", 20)
    draw_slide_text(c, h - 85, [
        "- Self-assessment report: 40 pts",
        "- Labeling work: 5 pts  (Roboflow -> dataset)",
        "- Dataset creation: 5 pts (data.yaml, train/val split)",
        "- Vehicle/plate detect on input file: 10 pts (YOLO bbox)",
        "- ROI, CROP, CLAHE, Homography: 20 pts (plate_engine_pro.py, exam_realtime_plate.py)",
        "- License plate recognition display: 20 pts (plate text + confidence on screen)",
        "",
        "Code tags: [평가기준 10점], [평가기준 20점] in plate_engine_pro.py, plate_server.py",
    ], 11)
    c.showPage()

    # Slide 5: 결과 예시
    draw_slide_title(c, h - 50, "Result Example", 20)
    draw_slide_text(c, h - 90, [
        "Recognized plates (confidence):",
        "  - 01나8060 (71%)",
        "  - 경기91바6286 (79%)",
        "  - 서울바9203 (72%)",
        "",
        "Output: Red/green bbox on plate + plate text above.",
        "See: plate_server.py (localhost:5000), plate_engine_pro.process_frame()",
    ], 12)
    c.showPage()

    # Slide 6: 마무리
    draw_slide_title(c, h - 70, "Summary", 20)
    draw_slide_text(c, h - 120, [
        "- YOLO (best.pt) detects plate region -> ROI/CROP",
        "- CLAHE + optional Homography improve readability",
        "- PaddleOCR + EasyOCR ensemble -> Korean plate text",
        "- Validation & voting -> final plate string",
        "- Display: plate number + confidence on video/image.",
        "",
        "Files: plate_engine_pro.py, plate_server.py, exam_realtime_plate.py",
        "Mapping: 평가기준_기능매핑.md",
    ], 11)
    c.showPage()

    c.save()
    print(f"[OK] PDF saved: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
