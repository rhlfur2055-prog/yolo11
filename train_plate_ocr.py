#!/usr/bin/env python3
"""
번호판 OCR 전용 CRNN 학습 스크립트
- 12장 테스트 이미지에서 plate ROI 추출
- 강력한 데이터 증강 (회전, 노이즈, 블러, 색상 변환, 원근 변환 등)
- CRNN (CNN + BiLSTM + CTC) 모델 학습
- plate_ocr_crnn.pth 저장
"""
import os
import sys
import random
import time
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── 설정 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "22")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "plate_ocr_crnn.pth")

# ── 문자 사전 ──
CHARS = (
    "0123456789"
    "가나다라마바사아자차카타파하"
    "거너더러머버서어저처커터퍼허"
    "고노도로모보소오조호"
    "구누두루무부수우주"
    "배육"
    # 지역명 한글
    "울산대인천광전세종경기강원충북남제외교"
)
# 중복 제거 + 정렬
CHARS = "".join(sorted(set(CHARS)))
BLANK = ""  # CTC blank → index 0
VOCAB = ["<blank>"] + list(CHARS)
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX2CHAR = {i: ch for i, ch in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)

# ── 학습 데이터 (12장) ──
TRAIN_DATA = [
    ("경기76바7789.png",      "경기76바7789"),
    ("서울바9203.png",        "서울70바9203"),
    ("트럭 경기91바6286.png", "경기91바6286"),
    ("01나8060.png",          "01나8060"),
    ("02누2754.png",          "02누2754"),
    ("14니3234.png",          "14나3234"),
    ("36다7117.png",          "36다7117"),
    ("48보7062.png",          "48보7062"),
    ("55저9392.png",          "55저9392"),
    ("58두9599.png",          "58두9599"),
    ("70버6393.png",          "70버6393"),
    ("80부5915.png",          "80부5915"),
]

# ── 모델 하이퍼파라미터 ──
IMG_H = 64       # 입력 이미지 높이 (고정)
IMG_W = 256      # 입력 이미지 너비 (고정, 패딩)
HIDDEN_SIZE = 256
NUM_LAYERS = 2
BATCH_SIZE = 16
NUM_EPOCHS = 150
LR = 0.001
AUG_PER_IMAGE = 400   # 이미지당 증강 수
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════
# 1. 데이터 추출: YOLO로 plate ROI 크롭
# ═══════════════════════════════════════════
def extract_plate_crops():
    """12장 이미지에서 YOLO로 번호판 ROI를 추출."""
    from plate_engine_pro import PlateEnginePro
    engine = PlateEnginePro()

    crops = []  # [(bgr_image, label), ...]
    for fname, gt in TRAIN_DATA:
        fpath = os.path.join(IMG_DIR, fname)
        img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[경고] 이미지 로드 실패: {fpath}")
            continue
        h, w = img.shape[:2]
        dets = engine.model(img, conf=0.25, imgsz=640, verbose=False)
        if not dets[0].boxes:
            print(f"[경고] 번호판 미감지: {fname}")
            continue
        det = dets[0].boxes[0]  # 첫 번째 감지 사용
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * 0.35), int(bh * 0.40)
        rx1, ry1 = max(0, x1 - mx), max(0, y1 - my)
        rx2, ry2 = min(w, x2 + mx), min(h, y2 + my)
        roi = img[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue
        crops.append((roi, gt))
        print(f"  [{len(crops):2d}] {gt:14s}  ROI={roi.shape[1]}x{roi.shape[0]}")

    print(f"\n총 {len(crops)}개 plate crop 추출 완료")
    return crops


# ═══════════════════════════════════════════
# 2. 데이터 증강
# ═══════════════════════════════════════════
def augment_image(img):
    """단일 이미지에 랜덤 증강 적용."""
    h, w = img.shape[:2]
    result = img.copy()

    # (1) 밝기/대비 조절
    if random.random() < 0.7:
        alpha = random.uniform(0.6, 1.5)   # 대비
        beta = random.randint(-40, 40)      # 밝기
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    # (2) 가우시안 노이즈
    if random.random() < 0.5:
        sigma = random.uniform(5, 25)
        noise = np.random.normal(0, sigma, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # (3) 가우시안 블러
    if random.random() < 0.4:
        k = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (k, k), 0)

    # (4) 모션 블러 (수평)
    if random.random() < 0.3:
        size = random.choice([3, 5, 7])
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0 / size
        result = cv2.filter2D(result, -1, kernel)

    # (5) 회전 (±5도)
    if random.random() < 0.5:
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)

    # (6) 원근 변환 (약간의 기울임)
    if random.random() < 0.4:
        d = random.uniform(0, 0.06)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [w * random.uniform(0, d), h * random.uniform(0, d)],
            [w * (1 - random.uniform(0, d)), h * random.uniform(0, d)],
            [w * random.uniform(0, d), h * (1 - random.uniform(0, d))],
            [w * (1 - random.uniform(0, d)), h * (1 - random.uniform(0, d))],
        ])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(result, M_persp, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)

    # (7) 색상 변환 (HSV jitter)
    if random.random() < 0.5:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # (8) JPEG 압축 아티팩트
    if random.random() < 0.3:
        quality = random.randint(30, 80)
        _, enc = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        result = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # (9) 랜덤 스케일 (약간)
    if random.random() < 0.3:
        sx = random.uniform(0.9, 1.1)
        sy = random.uniform(0.9, 1.1)
        result = cv2.resize(result, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)

    # (10) 반전 (녹색 번호판 등 대응)
    if random.random() < 0.15:
        result = cv2.bitwise_not(result)

    # (11) CLAHE
    if random.random() < 0.3:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        cl = cv2.createCLAHE(clipLimit=random.uniform(2.0, 6.0), tileGridSize=(4, 4))
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result


# ═══════════════════════════════════════════
# 3. Dataset 클래스
# ═══════════════════════════════════════════
class PlateOCRDataset(Dataset):
    """증강 포함 번호판 OCR 데이터셋."""

    def __init__(self, crops_and_labels, aug_per_image=AUG_PER_IMAGE,
                 img_h=IMG_H, img_w=IMG_W, augment=True):
        self.data = crops_and_labels   # [(bgr, label), ...]
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

        # 증강 적용 (첫 번째는 원본 유지)
        if self.augment and (idx // len(self.data)) > 0:
            img = augment_image(img)

        # 리사이즈 + 패딩
        img = self._resize_pad(img)

        # 그레이스케일 → 정규화
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        tensor = gray.astype(np.float32) / 255.0
        tensor = torch.FloatTensor(tensor).unsqueeze(0)  # (1, H, W)

        # 레이블 인코딩
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
        ratio = self.img_h / h
        new_w = min(int(w * ratio), self.img_w)
        img = cv2.resize(img, (new_w, self.img_h), interpolation=cv2.INTER_CUBIC)
        # 우측 흰색 패딩
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
# 4. CRNN 모델
# ═══════════════════════════════════════════
class CRNN(nn.Module):
    """CNN + BiLSTM + CTC 기반 텍스트 인식 모델."""

    def __init__(self, num_classes, img_h=IMG_H, hidden=HIDDEN_SIZE, n_layers=NUM_LAYERS):
        super().__init__()
        assert img_h == 64, "CNN 구조가 img_h=64에 최적화됨"

        # CNN backbone: 64 → 32 → 16 → 8 → 4 → 2 → 1 (height reduction)
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
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # h/2, w 유지
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

        # RNN
        self.rnn = nn.LSTM(512, hidden, n_layers,
                           bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, 64, W)
        conv = self.cnn(x)            # (B, 512, 1, W')
        b, c, h, w = conv.size()
        assert h == 1, f"CNN 출력 높이가 1이 아님: {h}"
        conv = conv.squeeze(2)         # (B, 512, W')
        conv = conv.permute(0, 2, 1)   # (B, W', 512)
        rnn_out, _ = self.rnn(conv)    # (B, W', hidden*2)
        output = self.fc(rnn_out)      # (B, W', num_classes)
        # CTC 입력: (T, B, C)
        output = output.permute(1, 0, 2)
        return output


# ═══════════════════════════════════════════
# 5. CTC 디코딩
# ═══════════════════════════════════════════
def ctc_greedy_decode(output, idx2char=IDX2CHAR):
    """CTC greedy decoding: 연속 중복 제거 + blank 제거."""
    # output: (T, B, C) → argmax
    _, preds = output.max(2)       # (T, B)
    preds = preds.transpose(0, 1)  # (B, T)
    results = []
    for b in range(preds.size(0)):
        chars = []
        prev = -1
        for t in range(preds.size(1)):
            p = preds[b, t].item()
            if p != 0 and p != prev:  # 0 = blank
                if p in idx2char:
                    chars.append(idx2char[p])
            prev = p
        results.append("".join(chars))
    return results


# ═══════════════════════════════════════════
# 6. 학습 루프
# ═══════════════════════════════════════════
def train():
    print("=" * 60)
    print("  번호판 OCR CRNN 학습")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  문자 수: {NUM_CLASSES} ({len(CHARS)} 문자 + blank)")
    print(f"  증강 배수: {AUG_PER_IMAGE}x/image")
    print(f"  에폭: {NUM_EPOCHS}")
    print()

    # ── 데이터 추출 ──
    print("[1/4] Plate ROI 추출 중...")
    crops = extract_plate_crops()
    if len(crops) < 12:
        print(f"[경고] {len(crops)}/12 이미지만 추출됨")

    # ── 데이터셋 생성 ──
    print(f"\n[2/4] 데이터셋 생성 (총 {len(crops) * AUG_PER_IMAGE} 샘플)...")
    dataset = PlateOCRDataset(crops, aug_per_image=AUG_PER_IMAGE)
    # Validation: 원본만 (증강 없음)
    val_dataset = PlateOCRDataset(crops, aug_per_image=1, augment=False)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(crops),
                            collate_fn=collate_fn, num_workers=0)

    # ── 모델 생성 ──
    print(f"\n[3/4] CRNN 모델 생성...")
    model = CRNN(NUM_CLASSES).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {param_count:,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── 학습 ──
    print(f"\n[4/4] 학습 시작 ({NUM_EPOCHS} epochs)...")
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for images, labels, label_lengths in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)
            # Forward
            output = model(images)  # (T, B, C)
            T = output.size(0)
            B = images.size(0)
            input_lengths = torch.full((B,), T, dtype=torch.int32, device=DEVICE)

            # CTC Loss
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

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # ── 검증 (매 5에폭) ──
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            model.eval()
            correct = 0
            total = len(crops)
            with torch.no_grad():
                for images, labels_flat, label_lengths in val_loader:
                    images = images.to(DEVICE)
                    output = model(images)
                    decoded = ctc_greedy_decode(output)
                    # 레이블 복원
                    offset = 0
                    for i, length in enumerate(label_lengths):
                        gt_encoded = labels_flat[offset:offset + length].tolist()
                        gt = "".join(IDX2CHAR.get(c, "?") for c in gt_encoded)
                        pred = decoded[i] if i < len(decoded) else ""
                        if pred == gt:
                            correct += 1
                        offset += length

            acc = correct / total
            lr_now = scheduler.get_last_lr()[0]
            status = "★" if acc > best_acc else " "
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}  "
                  f"acc={correct}/{total} ({acc:.0%})  lr={lr_now:.6f}  {status}")

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

            if acc >= 1.0:
                print(f"\n  ★ 12/12 달성! (epoch {epoch})")
                break

    print(f"\n{'=' * 60}")
    print(f"  학습 완료!")
    print(f"  최고 정확도: {best_acc:.0%} (epoch {best_epoch})")
    print(f"  모델 저장: {MODEL_SAVE_PATH}")
    print(f"{'=' * 60}")

    # ── 최종 검증: 각 이미지별 결과 ──
    print(f"\n[최종 검증]")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
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
                mark = "✅" if pred == gt else "❌"
                print(f"  {mark} GT={gt:14s}  PRED={pred}")
                offset += length


if __name__ == "__main__":
    train()
