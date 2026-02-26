"""
=============================================================================
  한국 번호판 YOLO 학습 파이프라인 (3,000건+ 목표)
=============================================================================
  경로: C:\\tool\\yolo26-main\\train_korean_plate_3000.py

  [데이터 소스]
    1) 기존 dataset/       → 159건 (라벨 있는 것)
    2) AI Hub 다운로드     → 2,000~3,000건 (JSON→YOLO 변환)
    3) hiway.mp4 등 영상   → 프레임 추출 + auto-label
    4) Augmentation        → 데이터 증강 (x3~5배)

  [사용법] - 한 단계씩 실행하세요!
    python train_korean_plate_3000.py --step 1  # AI Hub JSON → YOLO 변환
    python train_korean_plate_3000.py --step 2  # 영상 → 프레임 추출
    python train_korean_plate_3000.py --step 3  # 기존 빈 라벨 정리
    python train_korean_plate_3000.py --step 4  # 데이터셋 통합 + train/val 분리
    python train_korean_plate_3000.py --step 5  # Augmentation 증강
    python train_korean_plate_3000.py --step 6  # data.yaml 생성 + 검증
    python train_korean_plate_3000.py --step 7  # YOLO 학습 시작!

    python train_korean_plate_3000.py --check    # 현재 데이터 현황 확인
=============================================================================
"""
import os
import sys
import json
import glob
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

# ============================================================
# 설정 (본인 환경에 맞게 수정!)
# ============================================================
BASE_DIR = Path(r"C:\tool\yolo26-main")
DATASET_DIR = BASE_DIR / "dataset_3k"          # 새 데이터셋 폴더
EXISTING_DATASET = BASE_DIR / "dataset"         # 기존 데이터셋
AIHUB_DIR = BASE_DIR / "aihub_raw"              # AI Hub 다운로드 폴더

# AI Hub에서 받은 영상 폴더 (여러 개 가능)
VIDEO_DIR = BASE_DIR / "videos"                 # 영상 파일 모아놓는 폴더

# YOLO 학습 설정
BASE_MODEL = "yolo11n.pt"      # 기본 모델 (yolo26n.pt도 가능)
EPOCHS = 100
IMGSZ = 640
BATCH = 16                     # VRAM 부족하면 8로 낮추기
PATIENCE = 20
DEVICE = 0                     # GPU 번호 (CPU면 "cpu")

# 한국 번호판 클래스
NC = 1
CLASS_NAMES = ["plate"]

# Augmentation 배수
AUG_MULTIPLIER = 3             # 원본 1장 → 3장 추가 생성


# ============================================================
# Step 1: AI Hub JSON → YOLO 라벨 변환
# ============================================================
def step1_convert_aihub():
    """
    AI Hub '자동차 차종/연식/번호판 인식용 영상' 데이터를 YOLO 형식으로 변환

    AI Hub 구조 (일반적):
      aihub_raw/
      ├── Training/
      │   ├── images/
      │   │   └── *.jpg
      │   └── labels/           # 또는 annotations/
      │       └── *.json
      └── Validation/
          ├── images/
          └── labels/

    JSON 형식 (AI Hub 번호판 데이터):
      {
        "image": {"file_name": "xxx.jpg", "width": 1920, "height": 1080},
        "annotations": [
          {
            "class": "license_plate",
            "bbox": [x_min, y_min, x_max, y_max]   ← 또는 [x, y, w, h]
            // 또는 "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
          }
        ]
      }
    """
    print("=" * 60)
    print("  Step 1: AI Hub JSON → YOLO 변환")
    print("=" * 60)

    if not AIHUB_DIR.exists():
        print(f"\n  ⚠️  AI Hub 폴더가 없습니다: {AIHUB_DIR}")
        print(f"  → 먼저 AI Hub에서 데이터를 다운받아 이 폴더에 넣으세요!")
        print(f"  → https://aihub.or.kr 에서 '번호판 인식용 영상' 검색")
        AIHUB_DIR.mkdir(parents=True, exist_ok=True)
        return

    # JSON 파일 찾기
    json_files = list(AIHUB_DIR.rglob("*.json"))
    print(f"\n  발견된 JSON 파일: {len(json_files)}개")

    if not json_files:
        print("  ⚠️  JSON 파일이 없습니다.")
        print(f"  → AI Hub 데이터를 {AIHUB_DIR}에 압축 해제하세요")
        return

    # 출력 폴더
    out_img = DATASET_DIR / "images" / "aihub"
    out_lbl = DATASET_DIR / "labels" / "aihub"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    errors = 0

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            # ─── AI Hub 형식 자동 감지 ───
            img_info = None
            annotations = []
            img_w, img_h = 0, 0

            # 형식 A: {"image": {...}, "annotations": [...]}
            if "image" in data and "annotations" in data:
                img_info = data["image"]
                annotations = data["annotations"]
                img_w = img_info.get("width", 0)
                img_h = img_info.get("height", 0)

            # 형식 B: {"images": [...], "annotations": [...]}  (COCO-like)
            elif "images" in data and "annotations" in data:
                if data["images"]:
                    img_info = data["images"][0]
                    img_w = img_info.get("width", 0)
                    img_h = img_info.get("height", 0)
                    annotations = data["annotations"]

            # 형식 C: 직접 bbox가 있는 경우
            elif "bbox" in data or "bounding_box" in data:
                annotations = [data]
                # 이미지 크기를 파일에서 읽어야 함

            if not annotations or img_w == 0 or img_h == 0:
                skipped += 1
                continue

            # 이미지 파일 찾기
            img_filename = img_info.get("file_name", "")
            if not img_filename:
                img_filename = jf.stem + ".jpg"

            # 이미지 경로 탐색 (같은 폴더, 상위 images 폴더 등)
            img_path = None
            candidates = [
                jf.parent / img_filename,
                jf.parent.parent / "images" / img_filename,
                jf.parent.parent / "image" / img_filename,
                AIHUB_DIR / "images" / img_filename,
            ]
            for c in candidates:
                if c.exists():
                    img_path = c
                    break

            if img_path is None:
                # 확장자 바꿔서 시도
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    for c_base in [jf.parent, jf.parent.parent / "images"]:
                        candidate = c_base / (jf.stem + ext)
                        if candidate.exists():
                            img_path = candidate
                            break
                    if img_path:
                        break

            if img_path is None:
                skipped += 1
                continue

            # ─── YOLO 라벨 변환 ───
            yolo_lines = []
            for ann in annotations:
                bbox = None

                # bbox 형식 감지
                if "bbox" in ann:
                    bbox = ann["bbox"]  # [x, y, w, h] 또는 [x1, y1, x2, y2]
                elif "bounding_box" in ann:
                    bb = ann["bounding_box"]
                    if isinstance(bb, dict):
                        bbox = [bb.get("x", 0), bb.get("y", 0),
                                bb.get("w", bb.get("width", 0)),
                                bb.get("h", bb.get("height", 0))]
                    else:
                        bbox = bb
                elif "points" in ann:
                    # 폴리곤 → 바운딩 박스
                    pts = ann["points"]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

                if bbox is None or len(bbox) < 4:
                    continue

                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

                # [x1, y1, x2, y2] 형식인지 확인 (w, h가 너무 크면 x2, y2로 간주)
                if w > img_w * 0.5 and h > img_h * 0.5:
                    # x1, y1, x2, y2 형식
                    x2, y2 = w, h
                    w = x2 - x
                    h = y2 - y

                # YOLO 형식: class x_center y_center width height (정규화)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # 범위 체크
                if 0 < x_center < 1 and 0 < y_center < 1 and 0 < w_norm < 1 and 0 < h_norm < 1:
                    yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            if not yolo_lines:
                skipped += 1
                continue

            # 파일 복사 + 라벨 저장
            new_name = f"aihub_{converted:05d}"
            dst_img = out_img / f"{new_name}{img_path.suffix}"
            dst_lbl = out_lbl / f"{new_name}.txt"

            shutil.copy2(img_path, dst_img)
            with open(dst_lbl, "w") as f:
                f.write("\n".join(yolo_lines))

            converted += 1

            if converted % 100 == 0:
                print(f"    변환 중... {converted}건 완료")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    ⚠️ 에러: {jf.name} → {e}")

    print(f"\n  ✅ AI Hub 변환 완료!")
    print(f"     변환 성공: {converted}건")
    print(f"     스킵: {skipped}건")
    print(f"     에러: {errors}건")
    print(f"     저장 위치: {out_img}")


# ============================================================
# Step 2: 영상 → 프레임 추출
# ============================================================
def step2_extract_frames():
    """
    hiway.mp4 등 영상에서 프레임을 추출하여 이미지로 저장
    → 이후 라벨링 도구(CVAT, Roboflow)로 라벨링 필요!
    """
    print("=" * 60)
    print("  Step 2: 영상 → 프레임 추출")
    print("=" * 60)

    try:
        import cv2
    except ImportError:
        print("  ⚠️  pip install opencv-python 먼저 실행하세요!")
        return

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # 영상 파일 찾기
    video_exts = ["*.mp4", "*.avi", "*.mkv", "*.mov"]
    videos = []
    for ext in video_exts:
        videos.extend(VIDEO_DIR.glob(ext))
        videos.extend(BASE_DIR.glob(ext))  # 루트에도 찾기

    if not videos:
        print(f"\n  ⚠️  영상 파일이 없습니다!")
        print(f"  → {VIDEO_DIR} 또는 {BASE_DIR}에 영상을 넣으세요")
        print(f"  → hiway.mp4 같은 파일을 여기 복사하세요")
        return

    out_dir = DATASET_DIR / "images" / "frames_unlabeled"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    EXTRACT_FPS = 2  # 초당 2프레임 추출

    for video_path in videos:
        print(f"\n  처리 중: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"    ⚠️ 열 수 없음: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps if fps > 0 else 0

        print(f"    해상도: {w}x{h}, FPS: {fps:.1f}, 길이: {duration:.1f}초")

        frame_interval = max(1, int(fps / EXTRACT_FPS))
        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                fname = f"frame_{video_path.stem}_{saved:05d}.jpg"
                cv2.imwrite(str(out_dir / fname), frame)
                saved += 1

            frame_idx += 1

        cap.release()
        total_frames += saved
        print(f"    → {saved}장 추출 완료")

    print(f"\n  ✅ 총 {total_frames}장 추출 완료!")
    print(f"     저장 위치: {out_dir}")
    print()
    print("  ┌──────────────────────────────────────────────┐")
    print("  │  ⚠️  중요! 이 프레임들은 라벨링이 안 되어있음!  │")
    print("  │                                              │")
    print("  │  방법 1: Roboflow (추천, 무료 1000장)          │")
    print("  │    → https://app.roboflow.com                │")
    print("  │    → 이미지 업로드 → 자동 라벨링 → YOLO 내보내기│")
    print("  │                                              │")
    print("  │  방법 2: CVAT (무제한 무료)                    │")
    print("  │    → https://app.cvat.ai                     │")
    print("  │    → 프로젝트 생성 → 번호판 박스 그리기         │")
    print("  │    → YOLO 형식 내보내기                        │")
    print("  │                                              │")
    print("  │  방법 3: 기존 YOLO 모델로 auto-label           │")
    print("  │    → python train_korean_plate_3000.py --auto │")
    print("  │    → 기존 yolo11n.pt로 자동 라벨 생성          │")
    print("  │    → ⚠️ 정확도 낮음, 수동 검수 필수!           │")
    print("  └──────────────────────────────────────────────┘")


# ============================================================
# Step 2.5: Auto-label (기존 모델로 자동 라벨링)
# ============================================================
def step_auto_label():
    """
    기존 YOLO 모델(yolo11n.pt 등)으로 추출된 프레임에 자동 라벨 생성
    ⚠️ 정확도가 완벽하지 않으므로 수동 검수 권장!
    """
    print("=" * 60)
    print("  Step 2.5: Auto-Label (기존 모델로 자동 라벨링)")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ⚠️  pip install ultralytics 먼저 실행하세요!")
        return

    # 모델 로딩 (우선순위대로 시도)
    model_candidates = [
        BASE_DIR / "yolo11x_plate.pt",
        BASE_DIR / "yolo11n_plate.pt",
        BASE_DIR / "yolo26n.pt",
        BASE_DIR / "yolo11n.pt",
        BASE_DIR / "yolov8n.pt",
        "yolo11n.pt",  # ultralytics 기본 다운로드
    ]

    model = None
    for mp in model_candidates:
        mp = Path(mp)
        if mp.exists():
            print(f"  모델 로딩: {mp}")
            model = YOLO(str(mp))
            break

    if model is None:
        print("  ⚠️ 사용 가능한 YOLO 모델이 없습니다!")
        print(f"  → {BASE_DIR}에 .pt 파일을 넣으세요")
        return

    # 라벨 안 된 프레임 폴더
    frame_dir = DATASET_DIR / "images" / "frames_unlabeled"
    if not frame_dir.exists():
        print(f"  ⚠️ 프레임 폴더 없음: {frame_dir}")
        print("  → Step 2를 먼저 실행하세요!")
        return

    out_img = DATASET_DIR / "images" / "auto_labeled"
    out_lbl = DATASET_DIR / "labels" / "auto_labeled"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    print(f"  처리할 이미지: {len(images)}장")

    labeled = 0
    for img_path in images:
        results = model(str(img_path), verbose=False, conf=0.3)

        yolo_lines = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                # 번호판 클래스만 필터 (모델에 따라 클래스 ID 다를 수 있음)
                conf = float(box.conf[0])
                if conf < 0.3:
                    continue

                # YOLO xywhn 형식
                xywhn = box.xywhn[0].tolist()
                yolo_lines.append(
                    f"0 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}"
                )

        if yolo_lines:
            # 이미지 복사 + 라벨 저장
            shutil.copy2(img_path, out_img / img_path.name)
            lbl_path = out_lbl / (img_path.stem + ".txt")
            with open(lbl_path, "w") as f:
                f.write("\n".join(yolo_lines))
            labeled += 1

    print(f"\n  ✅ Auto-label 완료!")
    print(f"     번호판 감지된 이미지: {labeled}장 / {len(images)}장")
    print(f"     저장 위치: {out_img}")
    print(f"\n  ⚠️  자동 라벨은 오류가 있을 수 있습니다!")
    print(f"  → {out_lbl} 폴더의 .txt 파일을 수동 검수하세요")


# ============================================================
# Step 3: 기존 데이터셋 정리 (빈 라벨 처리)
# ============================================================
def step3_clean_existing():
    """
    기존 dataset/ 폴더에서:
    - 빈 라벨 파일 + 대응 이미지 → 20개만 남기고 삭제 (negative sample)
    - 라벨 있는 파일 → 새 데이터셋으로 복사
    """
    print("=" * 60)
    print("  Step 3: 기존 데이터셋 정리")
    print("=" * 60)

    if not EXISTING_DATASET.exists():
        print(f"  ⚠️ 기존 데이터셋 없음: {EXISTING_DATASET}")
        return

    out_img = DATASET_DIR / "images" / "existing"
    out_lbl = DATASET_DIR / "labels" / "existing"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    labeled_count = 0
    empty_count = 0
    negative_kept = 0
    MAX_NEGATIVE = 20  # negative sample 최대 개수

    for split in ["train", "val"]:
        lbl_dir = EXISTING_DATASET / "labels" / split
        img_dir = EXISTING_DATASET / "images" / split

        if not lbl_dir.exists():
            continue

        for lbl_file in lbl_dir.glob("*.txt"):
            img_file = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = img_dir / (lbl_file.stem + ext)
                if candidate.exists():
                    img_file = candidate
                    break

            if img_file is None:
                continue

            # 라벨 내용 확인
            content = lbl_file.read_text().strip()

            if content:
                # 라벨 있음 → 복사
                new_name = f"exist_{labeled_count:05d}"
                shutil.copy2(img_file, out_img / f"{new_name}{img_file.suffix}")
                shutil.copy2(lbl_file, out_lbl / f"{new_name}.txt")
                labeled_count += 1
            else:
                # 빈 라벨 → 제한적으로 유지
                empty_count += 1
                if negative_kept < MAX_NEGATIVE:
                    new_name = f"neg_{negative_kept:05d}"
                    shutil.copy2(img_file, out_img / f"{new_name}{img_file.suffix}")
                    shutil.copy2(lbl_file, out_lbl / f"{new_name}.txt")
                    negative_kept += 1

    print(f"\n  ✅ 기존 데이터셋 정리 완료!")
    print(f"     라벨 있는 이미지: {labeled_count}장 → 전부 복사")
    print(f"     빈 라벨 이미지: {empty_count}장 → {negative_kept}장만 유지")
    print(f"     저장 위치: {out_img}")


# ============================================================
# Step 4: 데이터셋 통합 + Train/Val 분리
# ============================================================
def step4_merge_split():
    """
    aihub/, existing/, auto_labeled/ 등 모든 소스를 합쳐서
    train/val (80/20)로 분리
    """
    print("=" * 60)
    print("  Step 4: 데이터셋 통합 + Train/Val 분리 (80:20)")
    print("=" * 60)

    # 최종 폴더
    final_train_img = DATASET_DIR / "images" / "train"
    final_train_lbl = DATASET_DIR / "labels" / "train"
    final_val_img = DATASET_DIR / "images" / "val"
    final_val_lbl = DATASET_DIR / "labels" / "val"

    for d in [final_train_img, final_train_lbl, final_val_img, final_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # 모든 소스에서 이미지-라벨 쌍 수집
    sources = ["aihub", "existing", "auto_labeled"]
    all_pairs = []

    for src in sources:
        img_dir = DATASET_DIR / "images" / src
        lbl_dir = DATASET_DIR / "labels" / src

        if not img_dir.exists():
            print(f"  ℹ️ {src} 폴더 없음 (스킵)")
            continue

        count = 0
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if lbl_file.exists():
                all_pairs.append((img_file, lbl_file, src))
                count += 1

        print(f"  {src}: {count}건")

    print(f"\n  총 수집: {len(all_pairs)}건")

    if not all_pairs:
        print("  ⚠️ 데이터가 없습니다! Step 1~3을 먼저 실행하세요.")
        return

    # 셔플 + 분리
    random.seed(42)
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    # 파일 복사
    for i, (img, lbl, src) in enumerate(train_pairs):
        new_name = f"train_{i:05d}"
        shutil.copy2(img, final_train_img / f"{new_name}{img.suffix}")
        shutil.copy2(lbl, final_train_lbl / f"{new_name}.txt")

    for i, (img, lbl, src) in enumerate(val_pairs):
        new_name = f"val_{i:05d}"
        shutil.copy2(img, final_val_img / f"{new_name}{img.suffix}")
        shutil.copy2(lbl, final_val_lbl / f"{new_name}.txt")

    print(f"\n  ✅ 분리 완료!")
    print(f"     Train: {len(train_pairs)}건")
    print(f"     Val:   {len(val_pairs)}건")
    print(f"     비율:  {len(train_pairs)/(len(all_pairs))*100:.0f}% / {len(val_pairs)/(len(all_pairs))*100:.0f}%")


# ============================================================
# Step 5: Augmentation (데이터 증강)
# ============================================================
def step5_augmentation():
    """
    학습 데이터를 AUG_MULTIPLIER배로 증강
    번호판 특화: 밝기, 블러, 비, 안개, 약간 회전 (좌우반전 X!)
    """
    print("=" * 60)
    print(f"  Step 5: Augmentation (x{AUG_MULTIPLIER}배 증강)")
    print("=" * 60)

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  ⚠️  pip install opencv-python numpy 먼저 실행하세요!")
        return

    try:
        import albumentations as A
        HAS_ALBUM = True
    except ImportError:
        print("  ℹ️  albumentations 없음 → OpenCV 기본 증강 사용")
        print("  (더 나은 결과: pip install albumentations)")
        HAS_ALBUM = False

    train_img_dir = DATASET_DIR / "images" / "train"
    train_lbl_dir = DATASET_DIR / "labels" / "train"

    if not train_img_dir.exists():
        print("  ⚠️ train 폴더 없음. Step 4를 먼저 실행하세요!")
        return

    images = sorted([f for f in train_img_dir.iterdir()
                     if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    original_count = len(images)
    print(f"  원본 이미지: {original_count}장")
    print(f"  목표: +{original_count * AUG_MULTIPLIER}장 (총 {original_count * (1 + AUG_MULTIPLIER)}장)")

    if HAS_ALBUM:
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
            A.RandomRain(slant_lower=-5, slant_upper=5, p=0.15),
            A.Rotate(limit=5, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.RandomScale(scale_limit=0.15, p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))

    aug_count = 0
    for img_path in images:
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 라벨 읽기
        bboxes = []
        class_labels = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    bboxes.append(bbox)
                    class_labels.append(cls)

        for aug_i in range(AUG_MULTIPLIER):
            try:
                if HAS_ALBUM and bboxes:
                    result = transform(
                        image=img,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    aug_img = result['image']
                    aug_bboxes = result['bboxes']
                    aug_classes = result['class_labels']
                else:
                    # OpenCV 기본 증강
                    aug_img = img.copy()
                    # 밝기/대비 변경
                    alpha = random.uniform(0.7, 1.3)
                    beta = random.randint(-30, 30)
                    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
                    # 가우시안 블러
                    if random.random() < 0.3:
                        ksize = random.choice([3, 5])
                        aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)
                    aug_bboxes = bboxes
                    aug_classes = class_labels

                if not aug_bboxes:
                    continue

                # 저장
                new_name = f"{img_path.stem}_aug{aug_i}"
                cv2.imwrite(str(train_img_dir / f"{new_name}.jpg"), aug_img)

                with open(train_lbl_dir / f"{new_name}.txt", "w") as f:
                    for cls, bbox in zip(aug_classes, aug_bboxes):
                        f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                aug_count += 1

            except Exception as e:
                continue

        if (aug_count) % 100 == 0 and aug_count > 0:
            print(f"    증강 중... {aug_count}장 생성")

    final_count = len(list(train_img_dir.glob("*.*")))
    print(f"\n  ✅ Augmentation 완료!")
    print(f"     추가 생성: {aug_count}장")
    print(f"     Train 총: {final_count}장")


# ============================================================
# Step 6: data.yaml 생성 + 데이터 검증
# ============================================================
def step6_create_yaml():
    """
    학습용 data.yaml 생성 + 전체 데이터 무결성 검증
    """
    print("=" * 60)
    print("  Step 6: data.yaml 생성 + 검증")
    print("=" * 60)

    # yaml 생성
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_content = f"""# 한국 번호판 인식 YOLO 학습 데이터셋
# 생성: train_korean_plate_3000.py
path: {str(DATASET_DIR).replace(chr(92), '/')}
train: images/train
val: images/val

nc: {NC}
names: {CLASS_NAMES}
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"  ✅ {yaml_path} 생성 완료")
    print(f"\n  --- 내용 ---")
    print(yaml_content)

    # 데이터 검증
    print("  --- 데이터 검증 ---")

    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split

        if not img_dir.exists():
            print(f"  ⚠️ {split} 폴더 없음!")
            continue

        imgs = set(f.stem for f in img_dir.iterdir()
                   if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"])
        lbls = set(f.stem for f in lbl_dir.iterdir() if f.suffix == ".txt")

        matched = imgs & lbls
        img_only = imgs - lbls
        lbl_only = lbls - imgs

        # 라벨 내용 검증
        empty_labels = 0
        valid_labels = 0
        total_boxes = 0
        for lbl_stem in matched:
            content = (lbl_dir / f"{lbl_stem}.txt").read_text().strip()
            if content:
                valid_labels += 1
                total_boxes += len(content.split("\n"))
            else:
                empty_labels += 1

        print(f"\n  [{split.upper()}]")
        print(f"    이미지: {len(imgs)}장")
        print(f"    라벨:   {len(lbls)}개")
        print(f"    매칭됨: {len(matched)}건")
        print(f"    라벨 있음: {valid_labels}건 (총 bbox: {total_boxes}개)")
        print(f"    빈 라벨: {empty_labels}건")
        if img_only:
            print(f"    ⚠️ 라벨 없는 이미지: {len(img_only)}장")
        if lbl_only:
            print(f"    ⚠️ 이미지 없는 라벨: {len(lbl_only)}개")

    # 목표 달성 여부
    train_count = len(list((DATASET_DIR / "images" / "train").glob("*.*"))) if (DATASET_DIR / "images" / "train").exists() else 0
    val_count = len(list((DATASET_DIR / "images" / "val").glob("*.*"))) if (DATASET_DIR / "images" / "val").exists() else 0
    total = train_count + val_count

    print(f"\n  ─────────────────────────────")
    print(f"  총 데이터: {total}건 (목표: 3,000건)")
    if total >= 3000:
        print(f"  목표 달성! 학습 준비 완료!")
    elif total >= 1000:
        print(f"  ⚠️  1,000건 이상이면 학습 가능하지만, 더 많을수록 좋습니다")
    else:
        print(f"  데이터가 부족합니다. AI Hub 데이터를 더 추가하세요!")


# ============================================================
# Step 7: YOLO 학습!
# ============================================================
def step7_train():
    """
    드디어 학습!
    """
    print("=" * 60)
    print("  Step 7: YOLO 학습 시작!")
    print("=" * 60)

    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print("  ⚠️ data.yaml 없음! Step 6을 먼저 실행하세요!")
        return

    # 데이터 수 확인
    train_dir = DATASET_DIR / "images" / "train"
    if train_dir.exists():
        train_count = len(list(train_dir.glob("*.*")))
    else:
        train_count = 0

    if train_count == 0:
        print("  ⚠️ 학습 데이터가 없습니다!")
        return

    print(f"\n  학습 설정:")
    print(f"    모델:     {BASE_MODEL}")
    print(f"    데이터:   {yaml_path}")
    print(f"    Train:    {train_count}장")
    print(f"    Epochs:   {EPOCHS}")
    print(f"    ImgSize:  {IMGSZ}")
    print(f"    Batch:    {BATCH}")
    print(f"    Patience: {PATIENCE}")
    print(f"    Device:   {DEVICE}")
    print()

    # 학습 코드 생성 (직접 실행하지 않음!)
    train_script = DATASET_DIR / "run_train.py"
    script_content = f'''"""
YOLO 번호판 학습 스크립트
실행: python {train_script}
"""
from ultralytics import YOLO

# 모델 로딩 (전이학습)
model = YOLO("{BASE_MODEL}")

# 학습 시작!
results = model.train(
    data=r"{yaml_path}",
    epochs={EPOCHS},
    imgsz={IMGSZ},
    batch={BATCH},
    name="plate_korean_3k",
    patience={PATIENCE},
    device={DEVICE},

    # 한국 번호판 최적화 옵션
    lr0=0.01,          # 초기 학습률
    lrf=0.01,          # 최종 학습률 비율
    warmup_epochs=3,   # 워밍업
    hsv_h=0.015,       # 색상 변환 (번호판은 낮게)
    hsv_s=0.4,         # 채도 변환
    hsv_v=0.4,         # 밝기 변환
    degrees=5.0,       # 회전 (번호판은 작게!)
    translate=0.1,     # 이동
    scale=0.3,         # 스케일
    flipud=0.0,        # 상하반전 OFF (번호판!)
    fliplr=0.0,        # 좌우반전 OFF (번호판!)
    mosaic=1.0,        # 모자이크 증강
    mixup=0.1,         # MixUp 증강

    # 저장
    save=True,
    save_period=10,    # 10에폭마다 체크포인트
    plots=True,        # 학습 그래프 저장
)

print()
print("=" * 50)
print("  학습 완료!")
print(f"  best.pt: runs/detect/plate_korean_3k/weights/best.pt")
print("=" * 50)
'''

    with open(train_script, "w", encoding="utf-8") as f:
        f.write(script_content)

    print(f"  ✅ 학습 스크립트 생성: {train_script}")
    print()
    print("  ┌──────────────────────────────────────────────┐")
    print("  │  학습 시작 명령어:                             │")
    print(f"  │  python {train_script}    │")
    print("  │                                              │")
    print("  │  학습 후 결과:                                 │")
    print("  │  runs/detect/plate_korean_3k/weights/best.pt │")
    print("  │                                              │")
    print("  │  best.pt를 C:\\tool\\yolo26-main\\에 복사하면     │")
    print("  │  기존 코드가 자동으로 사용합니다!               │")
    print("  └──────────────────────────────────────────────┘")


# ============================================================
# 현황 확인
# ============================================================
def check_status():
    """현재 데이터 현황 확인"""
    print("=" * 60)
    print("  현재 데이터 현황")
    print("=" * 60)

    sources = {
        "기존 dataset": EXISTING_DATASET,
        "AI Hub 원본": AIHUB_DIR,
        "새 데이터셋": DATASET_DIR,
    }

    for name, path in sources.items():
        print(f"\n  [{name}] {path}")
        if not path.exists():
            print(f"    → 폴더 없음")
            continue

        # 이미지 수
        img_count = 0
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            img_count += len(list(path.rglob(ext)))

        # 라벨 수
        lbl_count = len(list(path.rglob("*.txt")))
        json_count = len(list(path.rglob("*.json")))

        print(f"    이미지: {img_count}장")
        print(f"    라벨(txt): {lbl_count}개")
        if json_count:
            print(f"    라벨(json): {json_count}개")

    # 영상
    print(f"\n  [영상 파일]")
    for ext in ["*.mp4", "*.avi", "*.mkv"]:
        for v in BASE_DIR.glob(ext):
            print(f"    → {v.name} ({v.stat().st_size / 1024 / 1024:.1f}MB)")

    # .pt 모델
    print(f"\n  [YOLO 모델 (.pt)]")
    for pt in BASE_DIR.glob("*.pt"):
        print(f"    → {pt.name} ({pt.stat().st_size / 1024 / 1024:.1f}MB)")


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한국 번호판 YOLO 학습 파이프라인")
    parser.add_argument("--step", type=int, help="실행할 단계 (1~7)")
    parser.add_argument("--check", action="store_true", help="현재 현황 확인")
    parser.add_argument("--auto", action="store_true", help="Auto-label 실행")
    args = parser.parse_args()

    if args.check:
        check_status()
    elif args.auto:
        step_auto_label()
    elif args.step == 1:
        step1_convert_aihub()
    elif args.step == 2:
        step2_extract_frames()
    elif args.step == 3:
        step3_clean_existing()
    elif args.step == 4:
        step4_merge_split()
    elif args.step == 5:
        step5_augmentation()
    elif args.step == 6:
        step6_create_yaml()
    elif args.step == 7:
        step7_train()
    else:
        print(__doc__)
        print("\n  사용법:")
        print("    python train_korean_plate_3000.py --check   # 현황 확인")
        print("    python train_korean_plate_3000.py --step 1  # AI Hub 변환")
        print("    python train_korean_plate_3000.py --step 2  # 영상 프레임 추출")
        print("    python train_korean_plate_3000.py --auto    # 자동 라벨링")
        print("    python train_korean_plate_3000.py --step 3  # 기존 데이터 정리")
        print("    python train_korean_plate_3000.py --step 4  # 통합 + 분리")
        print("    python train_korean_plate_3000.py --step 5  # Augmentation")
        print("    python train_korean_plate_3000.py --step 6  # yaml + 검증")
        print("    python train_korean_plate_3000.py --step 7  # 학습 스크립트 생성")
