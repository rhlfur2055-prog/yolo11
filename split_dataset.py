# -*- coding: utf-8 -*-
"""
split_dataset.py - train/val 분할 (80/20) + data.yaml 생성
"""
import os
import sys
import glob
import random
import shutil

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
TRAIN_IMG = os.path.join(DATASET_DIR, "images", "train")
VAL_IMG = os.path.join(DATASET_DIR, "images", "val")
TRAIN_LBL = os.path.join(DATASET_DIR, "labels", "train")
VAL_LBL = os.path.join(DATASET_DIR, "labels", "val")

os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)

# train 이미지 목록 (jpg + png)
images = sorted(
    glob.glob(os.path.join(TRAIN_IMG, "*.jpg"))
    + glob.glob(os.path.join(TRAIN_IMG, "*.png"))
)
print(f"Total images: {len(images)}")

# 20% val 분할
random.seed(42)
random.shuffle(images)
val_count = max(1, len(images) // 5)  # 20%
val_images = images[:val_count]
train_images = images[val_count:]

print(f"Train: {len(train_images)}, Val: {val_count}")

# val 이미지+라벨 이동
moved = 0
for img_path in val_images:
    basename = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(TRAIN_LBL, f"{basename}.txt")

    # 이미지 이동
    dst_img = os.path.join(VAL_IMG, os.path.basename(img_path))
    shutil.move(img_path, dst_img)

    # 라벨 이동
    if os.path.exists(lbl_path):
        dst_lbl = os.path.join(VAL_LBL, f"{basename}.txt")
        shutil.move(lbl_path, dst_lbl)

    moved += 1

print(f"Moved {moved} images to val/")

# data.yaml 생성
data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")
yaml_content = f"""# Korean Highway License Plate Dataset
# Auto-generated for YOLO26 training

path: {DATASET_DIR}
train: images/train
val: images/val

nc: 1
names: ['license_plate']
"""

with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\ndata.yaml created: {data_yaml_path}")

# 최종 통계
train_remaining = len(glob.glob(os.path.join(TRAIN_IMG, "*.jpg"))) + len(glob.glob(os.path.join(TRAIN_IMG, "*.png")))
val_final = len(glob.glob(os.path.join(VAL_IMG, "*.jpg"))) + len(glob.glob(os.path.join(VAL_IMG, "*.png")))
train_labels = len(glob.glob(os.path.join(TRAIN_LBL, "*.txt")))
val_labels = len(glob.glob(os.path.join(VAL_LBL, "*.txt")))
print(f"\nFinal split:")
print(f"  Train: {train_remaining} images, {train_labels} labels")
print(f"  Val:   {val_final} images, {val_labels} labels")
