# -*- coding: utf-8 -*-
# ============================================
# aihub_data_organizer.py
# AI허브 데이터 → YOLO 학습 형태로 변환
# ============================================

import os
import json
import shutil
from pathlib import Path

# ── 경로 설정 ──
AIHUB_RAW = r"C:\tools\yolo26\dataset\aihub_raw"
OUTPUT_DIR = r"C:\tools\yolo26\dataset\yolo_format"


def convert_bbox_to_yolo(bbox, img_w, img_h):
    """AI허브 bbox [x, y, w, h] → YOLO [cx, cy, w, h] 정규화"""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def organize_dataset():
    """AI허브 JSON → YOLO format 변환"""
    # YOLO 디렉토리 생성
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    raw_path = Path(AIHUB_RAW)
    if not raw_path.exists():
        print(f"[경고] {AIHUB_RAW} 가 없습니다. AI허브에서 데이터를 다운로드한 뒤 다시 실행하세요.")
        return

    json_files = list(raw_path.rglob("*.json"))
    print(f"[INFO] 총 {len(json_files)}개 JSON 파일 발견")

    for idx, jf in enumerate(json_files):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 이미지 정보 추출
            img_info = data.get("image", data.get("Image_Info", {}))
            img_name = img_info.get("file_name", img_info.get("filename", ""))
            img_w = img_info.get("width", 1920)
            img_h = img_info.get("height", 1080)

            # 어노테이션 추출
            annotations = data.get("annotations", data.get("Annotation", []))

            # 이미지 파일 찾기
            img_path = jf.parent / img_name
            if not img_path.exists():
                img_path = jf.with_suffix(".jpg")
            if not img_path.exists():
                img_path = jf.with_suffix(".png")
            if not img_path.exists():
                continue

            # train/val/test 분배 (8:1:1)
            if idx % 10 < 8:
                split = "train"
            elif idx % 10 == 8:
                split = "val"
            else:
                split = "test"

            # 이미지 복사
            dst_img = f"{OUTPUT_DIR}/images/{split}/{img_path.name}"
            shutil.copy2(img_path, dst_img)

            # YOLO 라벨 생성
            label_name = img_path.stem + ".txt"
            dst_label = f"{OUTPUT_DIR}/labels/{split}/{label_name}"

            with open(dst_label, "w", encoding="utf-8") as lf:
                for ann in annotations:
                    bbox = ann.get("bbox", ann.get("coord", []))
                    if len(bbox) == 4:
                        cx, cy, nw, nh = convert_bbox_to_yolo(bbox, img_w, img_h)
                        # class 0 = plate
                        lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            if idx % 10000 == 0 and idx > 0:
                print(f"[진행] {idx}/{len(json_files)} 완료")

        except Exception as e:
            print(f"[에러] {jf}: {e}")
            continue

    print("[완료] 데이터 변환 완료!")


if __name__ == "__main__":
    organize_dataset()
