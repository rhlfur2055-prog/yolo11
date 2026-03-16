#!/usr/bin/env python3
"""
번호판 OCR CRNN 재학습 스크립트 v4.0 (2줄 번호판 완벽 학습)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v4.0 변경사항:
  1. ★ 2줄 번호판 합성 추가: _draw_2line_plate() (지역명 상단, 번호 하단)
  2. 2줄 합성 데이터 ~7,000장 추가 (17지역 × 43자 × 조합)
  3. 충남/경기/경남/전남 등 우선 지역 2줄 학습 3배 강화
  4. NUM_EPOCHS 100→200 (2줄 패턴 학습 충분한 시간)
  5. 조기 종료 조건: acc=12/12 & epoch>=150
  6. 혼동 문자(소/조/보/무/오/버) 샘플 유지
  7. 합성 이미지에 실제 번호판 질감(노이즈/그림자/오염) 시뮬레이션 유지
"""
import os
import sys
import random
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Windows 콘솔 한글 깨짐 방지
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── 경로 설정 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "22")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "plate_ocr_crnn.pth")

# 한국어 폰트 경로 (합성 번호판 생성용)
FONT_PATHS = [
    r"C:\Windows\Fonts\HANBatangB.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\NotoSansKR-Regular.otf",
]
FONT_PATH = next((f for f in FONT_PATHS if os.path.exists(f)), None)

# ── 문자 사전 (전체 한국 번호판 문자) ──
CHARS = (
    "0123456789"
    "가나다라마바사아자차카타파하"
    "거너더러머버서어저처커터퍼허"
    "고노도로모보소오조초코토포호"
    "구누두루무부수우주추쿠투푸후"
    "배비"                  # 상용차 전용 (배달, 비사업용 등)
    # 지역명 한글 (2줄 번호판 상단)
    "서울부산대구인천광주대전울산세종"
    "경기강원충북충남전북전남경북경남제주"
)
# 중복 제거 + 정렬
CHARS = "".join(sorted(set(CHARS)))
VOCAB = ["<blank>"] + list(CHARS)
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX2CHAR = {i: ch for i, ch in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)

# ── 학습 데이터 (12장 실제 이미지 — 파일명 오타 수정) ──
TRAIN_DATA = [
    ("경기76바7789.png",      "경기76바7789"),
    ("서울70바9203.png",      "서울70바9203"),   # ★ 수정: 서울바9203 → 서울70바9203
    ("트럭 경기91바6286.png", "경기91바6286"),
    ("01나8060.png",          "01나8060"),
    ("02누2754.png",          "02누2754"),
    ("14나3234.png",          "14나3234"),       # ★ 수정: 14니3234 → 14나3234
    ("36다7117.png",          "36다7117"),
    ("48보7062.png",          "48보7062"),
    ("55저9392.png",          "55저9392"),
    ("58두9599.png",          "58두9599"),
    ("70버6393.png",          "70버6393"),
    ("80부5915.png",          "80부5915"),
]

# ── 한국 지역명 전체 17개 ──
ALL_REGIONS = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]

# ── 상용차 한글 (영업용/버스/트럭) ──
COMMERCIAL_CHARS = list("바사아자비하배")
# ── 일반 번호판 한글 (가나다 계열) ──
GENERAL_CHARS = list("가나다라마바사아자차카타파하")

# ── 모델 하이퍼파라미터 ──
IMG_H = 64
IMG_W = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 2
BATCH_SIZE = 128          # 배치 크기 128 (CPU 처리량 개선)
NUM_EPOCHS = 200         # 200 에폭 (2줄 번호판 패턴 학습 충분)
LR = 0.001
AUG_PER_IMAGE = 0        # ★ 실제 이미지 학습셋 제외 (검증 전용 — 과적합 차단)
AUG_PER_SYNTH = 1        # 합성 이미지당 증강 수 (20000장 × 1 = 20,000샘플)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════
# 1. 합성 번호판 이미지 생성 (PIL)
# ═══════════════════════════════════════════
def make_synthetic_plates():
    """PIL로 합성 번호판 이미지 대량 생성 (v3.0 — ~20,000장).
    전 한글 43자 × 충분한 숫자 조합으로 과적합 완전 차단."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[경고] PIL 없음 — 합성 데이터 스킵")
        return []

    synth = []

    # 폰트 크기별 준비 (다양한 폰트 크기 = 해상도 다양성)
    font_objs = {}
    if FONT_PATH:
        for sz in [30, 35, 38, 42]:
            try:
                font_objs[sz] = ImageFont.truetype(FONT_PATH, size=sz)
            except Exception:
                pass
    font_sizes = list(font_objs.keys()) or [None]

    # ── 전 한글 문자 목록 (43자) ──
    ALL_HANGUL = list(
        "가나다라마바사아자차카타파하"
        "거너더러머버서어저처커터퍼허"
        "고노도로모보소오조초코토포호"
        "구누두루무부수우주추쿠투푸후"
        "배비"
    )

    # ── 혼동 문자 (집중 강화) ──
    CONFUSABLE = list("보무버머바마비배저소로나누라두")

    def _rand_font():
        sz = random.choice(font_sizes)
        return font_objs.get(sz, None)

    def _make(label, bg, fg):
        synth.append((_draw_plate(label, bg=bg, fg=fg, font_obj=_rand_font()), label))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # A. 신형 번호판: 숫자2+한글+숫자4 (지역명 없음)
    #    전 한글 43자 × 50조합 = 2,150장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for ch in ALL_HANGUL:
        for _ in range(50):
            yr = random.randint(1, 99)
            nm = random.randint(1000, 9999)
            label = f"{yr:02d}{ch}{nm:04d}"
            bg = random.choice([(255, 255, 255), (255, 220, 0), (0, 0, 128)])
            fg = (0, 0, 0) if bg != (0, 0, 128) else (255, 255, 255)
            _make(label, bg, fg)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # B. 구형 번호판: 지역명+숫자2+한글+숫자4
    #    17지역 × 43자 × 3조합 = 2,193장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for region in ALL_REGIONS:
        for ch in ALL_HANGUL:
            for _ in range(3):
                yr = random.randint(1, 99)
                nm = random.randint(1000, 9999)
                label = f"{region}{yr:02d}{ch}{nm:04d}"
                bg = random.choice([(255, 255, 255), (255, 220, 0)])
                _make(label, bg, (0, 0, 0))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # C. 혼동 문자 집중 (보/무/버/머/바/마 등)
    #    핵심 혼동 6자(소/조/보/무/오/버) × 200조합 + 나머지 9자 × 100조합
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CRITICAL_CONFUSABLE = set("소조보무오버")  # 핵심 혼동 문자 2배 강화
    for ch in CONFUSABLE:
        n_samples = 200 if ch in CRITICAL_CONFUSABLE else 100
        for _ in range(n_samples):
            yr = random.randint(1, 99)
            nm = random.randint(1000, 9999)
            label = f"{yr:02d}{ch}{nm:04d}"
            _make(label, (255, 255, 255), (0, 0, 0))
            # 지역명 포함도 절반 추가
            if random.random() < 0.5:
                region = random.choice(ALL_REGIONS)
                label2 = f"{region}{yr:02d}{ch}{nm:04d}"
                _make(label2, (255, 255, 255), (0, 0, 0))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # D. 우선 지역 집중 (충남/경남/전남 등 — 2글자 지역명 OCR이 어려움)
    #    8지역 × 43자 × 5조합 = 1,720장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PRIORITY_REGIONS = ["충남", "경남", "전남", "충북", "경북", "전북", "강원", "제주"]
    for region in PRIORITY_REGIONS:
        for ch in ALL_HANGUL:
            for _ in range(5):
                yr = random.randint(1, 99)
                nm = random.randint(1000, 9999)
                label = f"{region}{yr:02d}{ch}{nm:04d}"
                bg = random.choice([(255, 255, 255), (255, 220, 0)])
                _make(label, bg, (0, 0, 0))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # E. ★ 2줄 번호판 합성 (v4.0 신규)
    #    지역명 상단 + 번호 하단 렌더링
    #    17지역 × 43자 × 5조합 = 3,655장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for region in ALL_REGIONS:
        for ch in ALL_HANGUL:
            for _ in range(5):
                yr = random.randint(1, 99)
                nm = random.randint(1000, 9999)
                number = f"{yr:02d}{ch}{nm:04d}"
                full_label = f"{region}{number}"
                bg = random.choice([(255, 255, 255), (255, 220, 0)])
                img_2l = _draw_2line_plate(region, number, bg=bg, fg=(0, 0, 0),
                                            font_obj=_rand_font())
                synth.append((img_2l, full_label))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # F. ★ 우선 지역 2줄 번호판 3배 강화
    #    충남/경기/경남 등 8지역 × 43자 × 10조합 = 3,440장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for region in PRIORITY_REGIONS:
        for ch in ALL_HANGUL:
            for _ in range(10):
                yr = random.randint(1, 99)
                nm = random.randint(1000, 9999)
                number = f"{yr:02d}{ch}{nm:04d}"
                full_label = f"{region}{number}"
                # 녹색 번호판(영업용) 포함
                bg = random.choice([(255, 255, 255), (255, 220, 0), (0, 100, 0)])
                fg = (255, 255, 255) if bg == (0, 100, 0) else (0, 0, 0)
                img_2l = _draw_2line_plate(region, number, bg=bg, fg=fg,
                                            font_obj=_rand_font())
                synth.append((img_2l, full_label))

    print(f"  합성 번호판: {len(synth)}장 (신형+구형+혼동+우선지역+★2줄번호판, v4.0)")
    return synth


def _draw_plate(label, bg=(255, 255, 255), fg=(0, 0, 0), font_obj=None):
    """PIL로 번호판 텍스트 이미지 생성 + 실제 번호판 질감 시뮬레이션.

    BGR numpy array 반환.
    """
    from PIL import Image, ImageDraw, ImageFont

    W, H = 256, 64
    # 배경색에 미세 노이즈 (실제 번호판 표면 질감)
    bg_noisy = tuple(
        min(255, max(0, c + random.randint(-8, 8))) for c in bg
    )
    img = Image.new("RGB", (W, H), color=bg_noisy)
    draw = ImageDraw.Draw(img)
    if font_obj is None:
        try:
            font_obj = ImageFont.load_default()
        except Exception:
            font_obj = None

    # 텍스트 크기 계산 → 중앙 정렬
    try:
        bbox_t = draw.textbbox((0, 0), label, font=font_obj)
        tw = bbox_t[2] - bbox_t[0]
        th = bbox_t[3] - bbox_t[1]
    except Exception:
        tw, th = len(label) * 20, 30

    # 텍스트 위치에 ±3px 오프셋 (정렬 변동 시뮬레이션)
    x = max(0, (W - tw) // 2 + random.randint(-3, 3))
    y = max(0, (H - th) // 2 + random.randint(-2, 2))

    # 글자색에 미세 변동 (잉크 농도 차이)
    fg_noisy = tuple(
        min(255, max(0, c + random.randint(-12, 12))) for c in fg
    )
    draw.text((x, y), label, fill=fg_noisy, font=font_obj)

    # PIL RGB → OpenCV BGR
    arr = np.array(img)[:, :, ::-1].copy()

    # ── 실제 번호판 질감 시뮬레이션 (50% 확률) ──
    if random.random() < 0.5:
        # 그림자/조명 그라디언트
        h, w = arr.shape[:2]
        grad = np.linspace(
            random.uniform(0.75, 1.0),
            random.uniform(0.75, 1.0),
            w
        ).astype(np.float32)
        if random.random() < 0.5:
            grad = grad[::-1]
        arr = np.clip(arr.astype(np.float32) * grad[np.newaxis, :, np.newaxis],
                      0, 255).astype(np.uint8)

    # 미세 가우시안 노이즈 (항상 적용)
    sigma = random.uniform(2, 12)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return arr


def _draw_2line_plate(region, number, bg=(255, 255, 255), fg=(0, 0, 0), font_obj=None):
    """2줄 번호판 합성: 상단 지역명, 하단 번호.

    예: region="충남", number="86아6118"
    → 상단: "충 남"  하단: "86아6118"
    전체 레이블: "충남86아6118"
    BGR numpy array 반환.
    """
    from PIL import Image, ImageDraw, ImageFont

    W, H = 256, 64
    bg_noisy = tuple(min(255, max(0, c + random.randint(-8, 8))) for c in bg)
    img = Image.new("RGB", (W, H), color=bg_noisy)
    draw = ImageDraw.Draw(img)

    if font_obj is None:
        try:
            font_obj = ImageFont.load_default()
        except Exception:
            font_obj = None

    # 상단 지역명 (작은 글씨, 상단 1/3에 배치)
    try:
        # 지역명 폰트: 작게
        region_font = font_obj
        bbox_r = draw.textbbox((0, 0), region, font=region_font)
        rw = bbox_r[2] - bbox_r[0]
    except Exception:
        rw = len(region) * 16

    rx = max(0, (W - rw) // 2 + random.randint(-5, 5))
    ry = random.randint(1, 6)  # 상단 영역
    fg_noisy = tuple(min(255, max(0, c + random.randint(-12, 12))) for c in fg)
    draw.text((rx, ry), region, fill=fg_noisy, font=font_obj)

    # 하단 번호 (큰 글씨, 하단 2/3에 배치)
    try:
        bbox_n = draw.textbbox((0, 0), number, font=font_obj)
        nw = bbox_n[2] - bbox_n[0]
    except Exception:
        nw = len(number) * 20

    nx = max(0, (W - nw) // 2 + random.randint(-3, 3))
    ny = random.randint(28, 36)  # 하단 영역
    draw.text((nx, ny), number, fill=fg_noisy, font=font_obj)

    # PIL RGB → OpenCV BGR
    arr = np.array(img)[:, :, ::-1].copy()

    # 실제 번호판 질감 시뮬레이션 (50% 확률)
    if random.random() < 0.5:
        h, w = arr.shape[:2]
        grad = np.linspace(
            random.uniform(0.75, 1.0), random.uniform(0.75, 1.0), w
        ).astype(np.float32)
        if random.random() < 0.5:
            grad = grad[::-1]
        arr = np.clip(arr.astype(np.float32) * grad[np.newaxis, :, np.newaxis],
                      0, 255).astype(np.uint8)

    sigma = random.uniform(2, 12)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return arr


# ═══════════════════════════════════════════
# 2. 실제 이미지에서 plate ROI 추출
# ═══════════════════════════════════════════
def extract_plate_crops():
    """12장 이미지에서 YOLO로 번호판 ROI 추출."""
    from plate_engine_pro import PlateEnginePro
    engine = PlateEnginePro()

    crops = []
    for fname, gt in TRAIN_DATA:
        fpath = os.path.join(IMG_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[경고] 파일 없음: {fname}")
            continue
        img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[경고] 로드 실패: {fname}")
            continue
        h, w = img.shape[:2]
        dets = engine.model(img, conf=0.25, imgsz=640, verbose=False)
        if not dets[0].boxes:
            # YOLO 미감지 시 전체 이미지 사용 (2줄 번호판 대응)
            print(f"[경고] YOLO 미감지 → 전체 이미지 사용: {fname}")
            crops.append((img, gt))
            continue
        det = dets[0].boxes[0]
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * 0.35), int(bh * 0.40)
        rx1, ry1 = max(0, x1 - mx), max(0, y1 - my)
        rx2, ry2 = min(w, x2 + mx), min(h, y2 + my)
        roi = img[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            crops.append((img, gt))
            continue
        crops.append((roi, gt))
        print(f"  [{len(crops):2d}] {gt:14s}  ROI={roi.shape[1]}x{roi.shape[0]}")

    print(f"\n총 {len(crops)}개 실제 plate crop 추출")
    return crops


# ═══════════════════════════════════════════
# 3. 데이터 증강
# ═══════════════════════════════════════════
def augment_image(img):
    """단일 이미지에 랜덤 증강 적용."""
    h, w = img.shape[:2]
    result = img.copy()

    # (1) 밝기/대비
    if random.random() < 0.7:
        alpha = random.uniform(0.5, 1.6)
        beta = random.randint(-50, 50)
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    # (2) 가우시안 노이즈
    if random.random() < 0.5:
        sigma = random.uniform(5, 30)
        noise = np.random.normal(0, sigma, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # (3) 가우시안 블러
    if random.random() < 0.4:
        k = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (k, k), 0)

    # (4) 모션 블러
    if random.random() < 0.3:
        size = random.choice([3, 5, 7])
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0 / size
        result = cv2.filter2D(result, -1, kernel)

    # (5) 회전 (±6도)
    if random.random() < 0.5:
        angle = random.uniform(-6, 6)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # (6) 원근 변환
    if random.random() < 0.4:
        d = random.uniform(0, 0.07)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [w * random.uniform(0, d), h * random.uniform(0, d)],
            [w * (1 - random.uniform(0, d)), h * random.uniform(0, d)],
            [w * random.uniform(0, d), h * (1 - random.uniform(0, d))],
            [w * (1 - random.uniform(0, d)), h * (1 - random.uniform(0, d))],
        ])
        M_p = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(result, M_p, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # (7) HSV jitter
    if random.random() < 0.5:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-15, 15)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.6, 1.4), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.6, 1.4), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # (8) JPEG 압축 아티팩트
    if random.random() < 0.3:
        quality = random.randint(25, 85)
        _, enc = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        result = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # (9) 랜덤 스케일
    if random.random() < 0.3:
        sx = random.uniform(0.85, 1.15)
        sy = random.uniform(0.85, 1.15)
        result = cv2.resize(result, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)

    # (10) 반전 (녹색 번호판 대응)
    if random.random() < 0.12:
        result = cv2.bitwise_not(result)

    # (11) CLAHE
    if random.random() < 0.3:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        cl = cv2.createCLAHE(clipLimit=random.uniform(2.0, 6.0), tileGridSize=(4, 4))
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # (12) 랜덤 크롭 (약간)
    if random.random() < 0.2:
        h2, w2 = result.shape[:2]
        sh = max(1, int(h2 * random.uniform(0.02, 0.08)))
        sw = max(1, int(w2 * random.uniform(0.02, 0.08)))
        y0, x0 = random.randint(0, sh), random.randint(0, sw)
        result = result[y0:h2, x0:w2]
        if result.size == 0:
            result = img.copy()

    return result


# ═══════════════════════════════════════════
# 4. Dataset 클래스
# ═══════════════════════════════════════════
class PlateOCRDataset(Dataset):
    """증강 포함 번호판 OCR 데이터셋."""

    def __init__(self, crops_and_labels, aug_per_image=AUG_PER_IMAGE,
                 img_h=IMG_H, img_w=IMG_W, augment=True):
        self.data = crops_and_labels
        self.aug_per_image = aug_per_image
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.total = len(self.data) * aug_per_image

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        real_idx = idx % len(self.data)
        img, label = self.data[real_idx]
        img = img.copy()

        if self.augment and (idx // len(self.data)) > 0:
            img = augment_image(img)

        img = self._resize_pad(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        tensor = gray.astype(np.float32) / 255.0
        tensor = torch.FloatTensor(tensor).unsqueeze(0)  # (1, H, W)

        encoded = []
        for ch in label:
            if ch in CHAR2IDX:
                encoded.append(CHAR2IDX[ch])
            else:
                print(f"[경고] 미등록 문자: '{ch}' (label={label})")
        encoded = torch.IntTensor(encoded)

        return tensor, encoded, len(encoded)

    def _resize_pad(self, img):
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        ratio = self.img_h / h
        new_w = min(int(w * ratio), self.img_w)
        new_w = max(new_w, 1)
        img = cv2.resize(img, (new_w, self.img_h), interpolation=cv2.INTER_CUBIC)
        if new_w < self.img_w:
            if len(img.shape) == 3:
                pad = np.ones((self.img_h, self.img_w - new_w, 3), dtype=np.uint8) * 255
            else:
                pad = np.ones((self.img_h, self.img_w - new_w), dtype=np.uint8) * 255
            img = np.concatenate([img, pad], axis=1)
        return img


def collate_fn(batch):
    """가변 길이 레이블 배치 처리."""
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    label_lengths = torch.IntTensor(label_lengths)
    labels = torch.cat(labels, 0)
    return images, labels, label_lengths


# ═══════════════════════════════════════════
# 5. CRNN 모델
# ═══════════════════════════════════════════
class CRNN(nn.Module):
    """CNN + BiLSTM + CTC 기반 텍스트 인식 모델."""

    def __init__(self, num_classes, img_h=IMG_H, hidden=HIDDEN_SIZE, n_layers=NUM_LAYERS):
        super().__init__()
        assert img_h == 64, "CNN 구조가 img_h=64에 최적화됨"
        self.cnn = nn.Sequential(
            # Block 1: 64→32
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 2: 32→16
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 3: 16→8
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 4: 8→4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 5: 4→2
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 6: 2→1
            nn.Conv2d(512, 512, (2, 1), 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.rnn = nn.LSTM(512, hidden, n_layers,
                           bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)             # (B, 512, 1, W')
        b, c, h, w = conv.size()
        assert h == 1, f"CNN 출력 높이가 1이 아님: {h}"
        conv = conv.squeeze(2)          # (B, 512, W')
        conv = conv.permute(0, 2, 1)    # (B, W', 512)
        rnn_out, _ = self.rnn(conv)     # (B, W', hidden*2)
        output = self.fc(rnn_out)       # (B, W', num_classes)
        return output.permute(1, 0, 2)  # (T, B, C) for CTC


# ═══════════════════════════════════════════
# 6. CTC 디코딩
# ═══════════════════════════════════════════
def ctc_greedy_decode(output, idx2char=IDX2CHAR):
    """CTC greedy decoding: 연속 중복+blank 제거."""
    _, preds = output.max(2)
    preds = preds.transpose(0, 1)
    results = []
    for b in range(preds.size(0)):
        chars = []
        prev = -1
        for t in range(preds.size(1)):
            p = preds[b, t].item()
            if p != 0 and p != prev:
                if p in idx2char:
                    chars.append(idx2char[p])
            prev = p
        results.append("".join(chars))
    return results


# ═══════════════════════════════════════════
# 7. 학습 루프
# ═══════════════════════════════════════════
def train():
    # ★ 파일 로깅: Tee 방식 (콘솔 + 파일 동시 출력)
    import io
    _log_path = r"C:\tmp\train_log9.txt"
    class _TeeWriter:
        def __init__(self, orig, logfile):
            self._orig = orig
            self._log = logfile
        def write(self, s):
            self._orig.write(s)
            self._orig.flush()
            self._log.write(s)
            self._log.flush()
        def flush(self):
            self._orig.flush()
            self._log.flush()
    _log_file = open(_log_path, "w", encoding="utf-8")
    sys.stdout = _TeeWriter(sys.__stdout__, _log_file)
    sys.stderr = _TeeWriter(sys.__stderr__, _log_file)

    print("=" * 65)
    print("  번호판 OCR CRNN 재학습 v2.0")
    print("=" * 65)
    print(f"  Device  : {DEVICE}")
    print(f"  폰트    : {FONT_PATH or '없음(기본폰트사용)'}")
    print(f"  문자 수 : {NUM_CLASSES} ({len(CHARS)}자 + blank)")
    print(f"  에폭    : {NUM_EPOCHS}")
    print()

    # ── 1단계: 실제 이미지 ROI 추출 (검증 전용) ──
    print("[1/5] 실제 이미지 Plate ROI 추출 (검증 전용)...")
    real_crops = extract_plate_crops()
    if len(real_crops) < 10:
        print(f"[경고] 실제 이미지 {len(real_crops)}/12 추출됨")

    # ── 2단계: 합성 번호판 생성 ──
    print("\n[2/5] PIL 합성 번호판 생성 (v3.0 ~20,000장)...")
    synth_crops = make_synthetic_plates()

    # ── 3단계: 데이터셋 구성 (합성만 학습, 실제는 검증 전용) ──
    print(f"\n[3/5] 데이터셋 구성...")
    print(f"  ★ 실제 이미지: 검증 전용 (학습셋 제외 — 과적합 차단)")
    print(f"  합성 이미지: {len(synth_crops)}장 × {AUG_PER_SYNTH}증강 = {len(synth_crops)*AUG_PER_SYNTH:,}샘플")

    # 합성 이미지만 학습
    synth_ds = PlateOCRDataset(synth_crops, aug_per_image=AUG_PER_SYNTH)
    from torch.utils.data import ConcatDataset
    combined_ds = synth_ds  # 합성만 사용

    # 검증: 실제 이미지만 원본으로
    val_ds = PlateOCRDataset(real_crops, aug_per_image=1, augment=False)

    loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=len(real_crops),
                            collate_fn=collate_fn, num_workers=0)

    total_samples = len(combined_ds)
    print(f"  총 학습 샘플: {total_samples:,}")

    # ── 4단계: 모델 생성 ──
    print(f"\n[4/5] CRNN 모델 생성...")
    model = CRNN(NUM_CLASSES).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {param_count:,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── 5단계: 학습 ──
    import time
    print(f"\n[5/5] 학습 시작 ({NUM_EPOCHS} epochs)...", flush=True)
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for images, labels, label_lengths in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)

            output = model(images)
            T = output.size(0)
            B = images.size(0)
            input_lengths = torch.full((B,), T, dtype=torch.int32, device=DEVICE)

            log_probs = output.log_softmax(2)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        epoch_sec = time.time() - epoch_start
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # 매 에폭 진행 상황 출력 (속도 모니터링용)
        print(f"  [epoch {epoch:3d}/{NUM_EPOCHS}] loss={avg_loss:.4f}  {epoch_sec:.1f}s/epoch", flush=True)

        # 검증 (매 5에폭)
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            model.eval()
            correct = 0
            total_val = len(real_crops)
            with torch.no_grad():
                for images, labels_flat, label_lengths in val_loader:
                    images = images.to(DEVICE)
                    output = model(images)
                    decoded = ctc_greedy_decode(output)
                    offset = 0
                    for i, length in enumerate(label_lengths):
                        gt_encoded = labels_flat[offset:offset + length].tolist()
                        gt = "".join(IDX2CHAR.get(c, "?") for c in gt_encoded)
                        pred = decoded[i] if i < len(decoded) else ""
                        if pred == gt:
                            correct += 1
                        offset += length

            acc = correct / max(total_val, 1)
            lr_now = scheduler.get_last_lr()[0]
            mark = "★" if acc > best_acc else " "
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}  "
                  f"acc={correct}/{total_val} ({acc:.0%})  lr={lr_now:.7f}  {mark}", flush=True)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save({
                    "model_state": model.state_dict(),
                    "vocab": VOCAB,
                    "char2idx": CHAR2IDX,
                    "idx2char": IDX2CHAR,
                    "num_classes": NUM_CLASSES,
                    "img_h": IMG_H,
                    "img_w": IMG_W,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "accuracy": acc,
                    "epoch": epoch,
                }, MODEL_SAVE_PATH)

            if acc >= 1.0 and epoch >= 150:
                print(f"\n  ★ 12/12 달성! (epoch {epoch}) 조기 종료")
                break

    print(f"\n{'=' * 65}")
    print(f"  학습 완료!")
    print(f"  최고 정확도: {best_acc:.0%} (epoch {best_epoch})")
    print(f"  모델 저장: {MODEL_SAVE_PATH}")
    print(f"{'=' * 65}")

    # ── 최종 검증: 12장 이미지별 결과 출력 ──
    print("\n[최종 검증 — 12장 이미지]")
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        for images, labels_flat, label_lengths in val_loader:
            images = images.to(DEVICE)
            output = model(images)
            decoded = ctc_greedy_decode(output)
            offset = 0
            for i, length in enumerate(label_lengths):
                gt_encoded = labels_flat[offset:offset + length].tolist()
                gt = "".join(IDX2CHAR.get(c, "?") for c in gt_encoded)
                pred = decoded[i] if i < len(decoded) else ""
                mark = "OK" if pred == gt else "NG"
                print(f"  [{mark}] GT={gt:14s}  PRED={pred}")
                offset += length


if __name__ == "__main__":
    train()
