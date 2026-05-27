"""영상 번호판 인식 — 포트폴리오용 결과 영상 생성 (PIL 한글 렌더링)"""
import sys
import cv2
import os
import re
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

from yolo11_plate.plate_engine_pro import PlateEnginePro

VIDEO_PATH = r'C:\Users\jomg2\OneDrive\바탕 화면\文档\카카오톡 받은 파일\7777777777777777777777.mp4'
OUTPUT_PATH = r'C:\tool\yolo26-main\master-clean\result_portfolio.mp4'

# 한글 폰트
FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

_font_cache = {}

def get_font(size):
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
        except:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_korean_label(frame, text, x1, y1, x2, y2, font_size=40):
    """bbox 위에 한글 텍스트 — 배경 없이 흰색 텍스트 + 검정 외곽선"""
    # 바운딩박스 (흰색 테두리)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = get_font(font_size)

    # 텍스트 위치 (bbox 위)
    bbox_t = draw.textbbox((0, 0), text, font=font)
    th = bbox_t[3] - bbox_t[1]
    lx = x1
    ly = max(y1 - th - 10, 2)

    # 검정 외곽선 (가독성)
    for dx, dy in [(-2,-2),(2,-2),(-2,2),(2,2),(0,-2),(0,2),(-2,0),(2,0)]:
        draw.text((lx + dx, ly + dy), text, font=font, fill=(0, 0, 0))

    # 흰색 텍스트
    draw.text((lx, ly), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def normalize_plate(text):
    """표준 번호판 패턴 추출"""
    m = re.search(r'(\d{2,3}[가-힣]\d{4})', text)
    if m:
        return m.group(1)
    return text


def run_pass(engine, show_progress=True):
    """영상 1회 패스 → {norm_plate: 누적점수} 및 프레임별 결과 반환"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    votes = Counter()
    frame_data = []  # [(text, conf, bbox), ...]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        results = engine.process_frame(frame)
        best = None
        for r in results:
            text = r.get('plate') or r.get('text', '')
            conf = r.get('confidence') or r.get('conf', 0)
            bbox = r.get('bbox', [])
            if text and len(text) >= 6 and bbox:
                norm = normalize_plate(text)
                votes[norm] += conf
                if best is None or conf > best[1]:
                    best = (text, conf, bbox)
        frame_data.append(best)
        if show_progress and frame_idx % 60 == 0:
            print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)")

    cap.release()
    return votes, frame_data


def main():
    print("[1단계] 인식 중...")
    engine = PlateEnginePro()
    votes, frame_data = run_pass(engine)

    if not votes:
        print("[경고] 인식된 번호판 없음")
        return

    print("\n[전체 후보]")
    for plate, score in votes.most_common(8):
        print(f"  {plate}: {score:.1f}점")

    # 상위 후보들 — 2점수 이상 차이면 단독 승자, 아니면 둘 다 표시
    top = votes.most_common(3)
    print(f"\n[최종 상위 번호판]: {[p for p,s in top]}")

    # 결과 영상 생성
    print("\n[2단계] 결과 영상 생성...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    font_size = max(30, h // 35)  # 화면 크기에 맞는 폰트

    for idx, best in enumerate(frame_data):
        ret, frame = cap.read()
        if not ret:
            break

        if best:
            text, conf, bbox = best
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                label = f"{text}  {conf:.0%}"
                frame = draw_korean_label(frame, label, x1, y1, x2, y2, font_size)

        out.write(frame)

    cap.release()
    out.release()

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\n결과 영상 저장 완료: {OUTPUT_PATH} ({size_mb:.1f}MB)")


if __name__ == '__main__':
    main()
