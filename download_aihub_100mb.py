# -*- coding: utf-8 -*-
# ============================================
# download_aihub_100mb.py
# AI허브 100MB 이하 분할 다운로드 + YOLO 변환 도구
# ============================================
#
# 사용법:
#   python download_aihub_100mb.py --guide       # 다운로드 방법 안내
#   python download_aihub_100mb.py --scan        # 데이터 구조 스캔
#   python download_aihub_100mb.py --convert     # aihub_raw → YOLO 변환만
#   python download_aihub_100mb.py --all         # 스캔 + 변환 + data.yaml 확인
#   python download_aihub_100mb.py --stats       # 변환 결과 통계
# ============================================

import os
import sys
import io
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Windows cp949 콘솔 인코딩 문제 해결
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 경로 설정 ──
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
AIHUB_DOWNLOAD = DATASET_DIR / "aihub_download"
AIHUB_RAW = DATASET_DIR / "aihub_raw"
YOLO_DIR = DATASET_DIR / "yolo_format"
DATA_YAML = YOLO_DIR / "data.yaml"

DATASET_KEY = "27727"  # AI허브 자동차 차종/연식/번호판 인식용 데이터


def ensure_dirs():
    """필수 디렉토리 생성"""
    for d in [DATASET_DIR, AIHUB_DOWNLOAD, AIHUB_RAW,
              YOLO_DIR / "images" / "train",
              YOLO_DIR / "images" / "val",
              YOLO_DIR / "images" / "test",
              YOLO_DIR / "labels" / "train",
              YOLO_DIR / "labels" / "val",
              YOLO_DIR / "labels" / "test"]:
        d.mkdir(parents=True, exist_ok=True)


def show_guide():
    """다운로드 방법 안내"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         AI허브 100MB 이하 분할 다운로드 가이드               ║
╚══════════════════════════════════════════════════════════════╝

📋 데이터셋: 자동차 차종/연식/번호판 인식용 영상 (ID: {dataset_key})

────────────────────────────────────────────────────────
🅰  방법 1: 웹 브라우저 수동 다운로드 (가장 간단)
────────────────────────────────────────────────────────
  1. https://aihub.or.kr 로그인
  2. 마이페이지 → 데이터 신청 내역
  3. '자동차 차종/연식/번호판 인식용 영상' 선택
  4. 100MB 이하 파일들을 개별 다운로드
  5. 다운받은 파일을 아래 경로에 저장:
     → {download_dir}
  6. ZIP 파일 압축 해제:
     → {raw_dir}

────────────────────────────────────────────────────────
🅱  방법 2: aihubshell CLI 도구 (WSL/Linux)
────────────────────────────────────────────────────────
  # WSL 터미널에서:

  # 1) aihubshell 설치
  curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do
  chmod +x aihubshell && sudo cp aihubshell /usr/local/bin/

  # 2) 파일 목록 조회 (filekey 확인)
  aihubshell -mode l -datasetkey {dataset_key}

  # 3) 100MB 이하 파일 다운로드 (filekey로 지정)
  cd /mnt/c/tools/yolo26/dataset/aihub_download
  aihubshell -mode d \\
    -datasetkey {dataset_key} \\
    -filekey <파일키> \\
    -aihubapikey '<본인_API키>'

  # 4) 여러 파일 동시 다운로드 (콤마 구분)
  aihubshell -mode d \\
    -datasetkey {dataset_key} \\
    -filekey 12345,12346,12347 \\
    -aihubapikey '<본인_API키>'

────────────────────────────────────────────────────────
🅲  방법 3: Windows PowerShell (Invoke-WebRequest)
────────────────────────────────────────────────────────
  # AI허브 웹에서 직접 다운로드 링크 복사 후:
  Invoke-WebRequest -Uri "<다운로드URL>" `
    -OutFile "{download_dir}\\파일명.zip"

════════════════════════════════════════════════════════
⚠️  다운로드 후 다음 단계:
════════════════════════════════════════════════════════
  1. ZIP 압축 해제 → {raw_dir}
  2. python download_aihub_100mb.py --scan    # 구조 확인
  3. python download_aihub_100mb.py --convert # YOLO 변환
  4. python train_plate_detector.py           # 학습 시작

⚠️  주의: dataset/ 폴더는 .gitignore에 포함됨 (개인정보 보호)
""".format(
        dataset_key=DATASET_KEY,
        download_dir=AIHUB_DOWNLOAD,
        raw_dir=AIHUB_RAW,
    ))


def scan_data():
    """다운로드/원본 데이터 구조 스캔"""
    print("\n" + "=" * 60)
    print("  데이터 구조 스캔")
    print("=" * 60)

    # aihub_download 스캔
    print(f"\n📁 다운로드 폴더: {AIHUB_DOWNLOAD}")
    if AIHUB_DOWNLOAD.exists():
        files = list(AIHUB_DOWNLOAD.iterdir())
        if files:
            total_size = 0
            for f in sorted(files):
                size = f.stat().st_size if f.is_file() else 0
                total_size += size
                size_str = _format_size(size)
                icon = "📦" if f.suffix == ".zip" else "📄"
                print(f"  {icon} {f.name} ({size_str})")
            print(f"  ── 합계: {len(files)}개, {_format_size(total_size)}")
        else:
            print("  (비어있음) → --guide 로 다운로드 방법 확인")
    else:
        print("  (폴더 없음)")

    # aihub_raw 스캔
    print(f"\n📁 원본 데이터 폴더: {AIHUB_RAW}")
    if AIHUB_RAW.exists():
        json_files = list(AIHUB_RAW.rglob("*.json"))
        img_files = list(AIHUB_RAW.rglob("*.jpg")) + list(AIHUB_RAW.rglob("*.png"))
        subdirs = [d for d in AIHUB_RAW.iterdir() if d.is_dir()]
        print(f"  📂 하위 폴더: {len(subdirs)}개")
        print(f"  📋 JSON 파일: {len(json_files)}개")
        print(f"  🖼️  이미지 파일: {len(img_files)}개")

        # JSON 구조 샘플링
        if json_files:
            sample = json_files[0]
            print(f"\n  📋 JSON 샘플: {sample.name}")
            try:
                with open(sample, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _print_json_structure(data, indent=4)
            except Exception as e:
                print(f"    (읽기 실패: {e})")
    else:
        print("  (폴더 없음)")

    # 기존 YOLO 데이터
    print(f"\n📁 YOLO 변환 데이터: {YOLO_DIR}")
    if YOLO_DIR.exists():
        for split in ["train", "val", "test"]:
            img_dir = YOLO_DIR / "images" / split
            lbl_dir = YOLO_DIR / "labels" / split
            n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
            n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            print(f"  {split:>5}: 이미지 {n_img}개, 라벨 {n_lbl}개")
    print()


def _format_size(size_bytes):
    """바이트 → 사람 읽기용 크기"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def _print_json_structure(obj, indent=0, max_depth=3, depth=0):
    """JSON 구조를 트리 형태로 출력"""
    prefix = " " * indent
    if depth >= max_depth:
        print(f"{prefix}...")
        return
    if isinstance(obj, dict):
        for key in list(obj.keys())[:15]:
            val = obj[key]
            if isinstance(val, dict):
                print(f"{prefix}📎 {key}: {{...}} ({len(val)}개 키)")
                if depth < max_depth - 1:
                    _print_json_structure(val, indent + 4, max_depth, depth + 1)
            elif isinstance(val, list):
                print(f"{prefix}📎 {key}: [...] ({len(val)}개 항목)")
                if val and depth < max_depth - 1:
                    _print_json_structure(val[0], indent + 4, max_depth, depth + 1)
            else:
                val_str = str(val)[:60]
                print(f"{prefix}📎 {key}: {val_str}")
    elif isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict):
            print(f"{prefix}[0]: {{...}} ({len(first)}개 키)")
            _print_json_structure(first, indent + 4, max_depth, depth + 1)
        else:
            print(f"{prefix}[0]: {str(first)[:60]}")


def unzip_downloads():
    """aihub_download의 ZIP 파일들을 aihub_raw로 압축 해제"""
    import zipfile

    if not AIHUB_DOWNLOAD.exists():
        print("[경고] 다운로드 폴더가 없습니다:", AIHUB_DOWNLOAD)
        return 0

    zip_files = list(AIHUB_DOWNLOAD.glob("*.zip"))
    if not zip_files:
        print("[INFO] 압축 해제할 ZIP 파일이 없습니다.")
        return 0

    AIHUB_RAW.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for zf in zip_files:
        print(f"[압축해제] {zf.name} → {AIHUB_RAW}")
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(AIHUB_RAW)
            extracted += 1
        except Exception as e:
            print(f"  [에러] {zf.name}: {e}")

    print(f"[완료] {extracted}/{len(zip_files)} 파일 압축 해제")
    return extracted


def merge_part_files():
    """분할 다운로드 .part 파일 병합"""
    if not AIHUB_DOWNLOAD.exists():
        return

    part_groups = defaultdict(list)
    for f in AIHUB_DOWNLOAD.iterdir():
        if ".part" in f.name:
            base = f.name.split(".part")[0]
            part_groups[base].append(f)

    for base_name, parts in part_groups.items():
        parts.sort(key=lambda x: x.name)
        output = AIHUB_DOWNLOAD / base_name
        if output.exists():
            print(f"[건너뜀] 이미 병합됨: {base_name}")
            continue
        print(f"[병합] {len(parts)}개 파트 → {base_name}")
        with open(output, "wb") as out:
            for part in parts:
                with open(part, "rb") as inp:
                    shutil.copyfileobj(inp, out)
        print(f"  → {_format_size(output.stat().st_size)}")


def convert_to_yolo():
    """AI허브 JSON + 이미지 → YOLO format 변환

    지원 데이터 형태:
      A) 크롭형: {imagePath, value, id} - 번호판만 크롭된 이미지
         → 이미지 전체를 bbox(0.5, 0.5, 1.0, 1.0)로 처리
      B) 좌표형: {image:{width,height}, annotations:[{bbox}]} - 전체 차량 이미지+좌표
         → bbox를 YOLO 정규화 좌표로 변환
    """
    try:
        import numpy as np
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False

    ensure_dirs()

    if not AIHUB_RAW.exists() or not any(AIHUB_RAW.iterdir()):
        print(f"[경고] {AIHUB_RAW} 가 비어있습니다.")
        print("  -> --guide 로 다운로드 방법을 확인하세요.")
        return 0

    json_files = list(AIHUB_RAW.rglob("*.json"))
    print(f"[INFO] 총 {len(json_files)}개 JSON 파일 발견")

    if not json_files:
        print("[경고] JSON 파일이 없습니다.")
        return 0

    # 랜덤 셔플 후 분배 (재현성 위해 시드 고정)
    random.seed(42)
    random.shuffle(json_files)

    converted = 0
    skipped_no_image = 0
    skipped_no_bbox = 0
    errors = 0
    type_a_count = 0  # 크롭형
    type_b_count = 0  # 좌표형
    stats = {"train": 0, "val": 0, "test": 0}

    for idx, jf in enumerate(json_files):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            # ── 데이터 형태 자동 감지 ──
            if "value" in data and "imagePath" in data:
                # ═══ A) 크롭형: {imagePath, value, id} ═══
                img_name = data["imagePath"]
                img_path = jf.parent / img_name
                if not img_path.exists():
                    img_path = jf.with_suffix(".jpg")
                if not img_path.exists():
                    img_path = jf.with_suffix(".png")
                if not img_path.exists():
                    skipped_no_image += 1
                    continue

                # 한글 파일명 대응: shutil.copy2 사용 (cv2보다 안전)
                # train/val/test 분배 (8:1:1)
                if idx % 10 < 8:
                    split = "train"
                elif idx % 10 == 8:
                    split = "val"
                else:
                    split = "test"

                # 파일명을 인덱스 기반으로 변경 (한글 파일명 문제 방지)
                safe_name = f"plate_{idx:06d}"
                dst_img = YOLO_DIR / "images" / split / f"{safe_name}.jpg"
                dst_label = YOLO_DIR / "labels" / split / f"{safe_name}.txt"

                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

                # 크롭 이미지 전체가 번호판 → bbox = 이미지 전체
                # YOLO format: class cx cy w h (정규화)
                with open(dst_label, "w", encoding="utf-8") as lf:
                    lf.write("0 0.500000 0.500000 1.000000 1.000000\n")

                stats[split] += 1
                converted += 1
                type_a_count += 1

            else:
                # ═══ B) 좌표형: {image, annotations} ═══
                img_info = (data.get("image")
                           or data.get("Image_Info")
                           or data.get("images")
                           or {})
                if isinstance(img_info, list) and img_info:
                    img_info = img_info[0]

                img_name = (img_info.get("file_name")
                           or img_info.get("filename")
                           or img_info.get("FILE_NAME")
                           or "")
                img_w = img_info.get("width", img_info.get("WIDTH", 1920))
                img_h = img_info.get("height", img_info.get("HEIGHT", 1080))

                annotations = (data.get("annotations")
                              or data.get("Annotation")
                              or data.get("annotation")
                              or data.get("ANNOTATION")
                              or [])
                if isinstance(annotations, dict):
                    annotations = [annotations]

                bboxes = []
                for ann in annotations:
                    bbox = (ann.get("bbox")
                           or ann.get("coord")
                           or ann.get("BBOX")
                           or ann.get("plate_bbox")
                           or ann.get("Plate_bbox")
                           or [])
                    if isinstance(bbox, dict):
                        bbox = [bbox.get("x", 0), bbox.get("y", 0),
                                bbox.get("w", bbox.get("width", 0)),
                                bbox.get("h", bbox.get("height", 0))]
                    if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                        bboxes.append(bbox)

                if not bboxes:
                    skipped_no_bbox += 1
                    continue

                # 이미지 파일 찾기
                img_path = None
                if img_name:
                    candidate = jf.parent / img_name
                    if candidate.exists():
                        img_path = candidate
                if not img_path:
                    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                        candidate = jf.with_suffix(ext)
                        if candidate.exists():
                            img_path = candidate
                            break

                has_image = img_path is not None and img_path.exists()
                if not has_image:
                    skipped_no_image += 1
                    continue

                if idx % 10 < 8:
                    split = "train"
                elif idx % 10 == 8:
                    split = "val"
                else:
                    split = "test"

                safe_name = f"plate_{idx:06d}"
                dst_img = YOLO_DIR / "images" / split / f"{safe_name}{img_path.suffix}"
                dst_label = YOLO_DIR / "labels" / split / f"{safe_name}.txt"

                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

                with open(dst_label, "w", encoding="utf-8") as lf:
                    for bbox in bboxes:
                        x, y, w, h = bbox
                        cx = (x + w / 2) / img_w
                        cy = (y + h / 2) / img_h
                        nw = w / img_w
                        nh = h / img_h
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        nw = max(0.0, min(1.0, nw))
                        nh = max(0.0, min(1.0, nh))
                        if nw > 0.001 and nh > 0.001:
                            lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                stats[split] += 1
                converted += 1
                type_b_count += 1

            if (idx + 1) % 10000 == 0:
                print(f"  [진행] {idx + 1}/{len(json_files)} 처리 완료...")

        except json.JSONDecodeError:
            errors += 1
            continue
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [에러] {jf.name}: {e}")
            continue

    # ── 결과 출력 ──
    print("\n" + "=" * 50)
    print("  변환 완료!")
    print("=" * 50)
    print(f"  변환 성공: {converted}개")
    print(f"    - 크롭형 (이미지=번호판): {type_a_count}개")
    print(f"    - 좌표형 (bbox 포함):     {type_b_count}개")
    print(f"  건너뜀 (이미지 없음): {skipped_no_image}개")
    print(f"  건너뜀 (bbox 없음):   {skipped_no_bbox}개")
    print(f"  에러: {errors}개")
    print(f"  ────────────────────────")
    print(f"  Train: {stats['train']}개")
    print(f"  Val:   {stats['val']}개")
    print(f"  Test:  {stats['test']}개")
    print()

    return converted


def write_data_yaml():
    """data.yaml 생성/갱신"""
    yaml_content = f"""# ============================================
# data.yaml - YOLO 학습 데이터 설정
# ============================================
path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['plate']
"""
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[OK] data.yaml 생성: {DATA_YAML}")


def show_stats():
    """변환 결과 통계"""
    print("\n" + "=" * 50)
    print("  YOLO 데이터셋 통계")
    print("=" * 50)

    if not YOLO_DIR.exists():
        print("  (데이터 없음)")
        return

    total_imgs = 0
    total_lbls = 0
    total_bboxes = 0

    for split in ["train", "val", "test"]:
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split

        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

        # bbox 수 카운트
        n_bbox = 0
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob("*.txt"):
                try:
                    with open(lbl_file, "r") as f:
                        n_bbox += sum(1 for line in f if line.strip())
                except Exception:
                    pass

        total_imgs += n_img
        total_lbls += n_lbl
        total_bboxes += n_bbox
        print(f"  {split:>5}: 이미지 {n_img:>6}개 | 라벨 {n_lbl:>6}개 | bbox {n_bbox:>7}개")

    print(f"  {'합계':>5}: 이미지 {total_imgs:>6}개 | 라벨 {total_lbls:>6}개 | bbox {total_bboxes:>7}개")

    # data.yaml 확인
    print(f"\n  data.yaml: {'✅ 존재' if DATA_YAML.exists() else '❌ 없음'}")
    if DATA_YAML.exists():
        print(f"  경로: {DATA_YAML}")

    # 학습 준비 상태
    if total_lbls > 0:
        print(f"\n  🟢 학습 준비 완료!")
        print(f"     python train_plate_detector.py")
    else:
        print(f"\n  🔴 학습 데이터 없음 - 먼저 데이터를 변환하세요")
        print(f"     python download_aihub_100mb.py --guide")
    print()


def run_all():
    """전체 파이프라인: 병합 → 압축해제 → 변환 → yaml → 통계"""
    print("\n🚀 전체 파이프라인 실행")
    print("=" * 50)

    # 1) .part 파일 병합
    print("\n[1/5] 분할 파일 병합...")
    merge_part_files()

    # 2) ZIP 압축 해제
    print("\n[2/5] ZIP 압축 해제...")
    unzip_downloads()

    # 3) 데이터 구조 스캔
    print("\n[3/5] 데이터 구조 스캔...")
    scan_data()

    # 4) YOLO 변환
    print("\n[4/5] YOLO 포맷 변환...")
    converted = convert_to_yolo()

    # 5) data.yaml 확인/생성
    print("\n[5/5] data.yaml 확인...")
    write_data_yaml()

    # 결과 통계
    show_stats()

    if converted > 0:
        print("=" * 50)
        print("🟢 즉시 학습 시작 가능!")
        print("   python train_plate_detector.py")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="AI허브 100MB 이하 분할 다운로드 + YOLO 변환 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  %(prog)s --guide       다운로드 방법 안내
  %(prog)s --scan        데이터 구조 확인
  %(prog)s --convert     JSON → YOLO 라벨 변환
  %(prog)s --all         전체 파이프라인 (압축해제+변환+확인)
  %(prog)s --stats       변환 결과 통계
        """)
    parser.add_argument("--guide", action="store_true",
                        help="다운로드 방법 안내 표시")
    parser.add_argument("--scan", action="store_true",
                        help="데이터 구조 스캔")
    parser.add_argument("--convert", action="store_true",
                        help="aihub_raw → YOLO 변환")
    parser.add_argument("--all", action="store_true",
                        help="전체 파이프라인 실행")
    parser.add_argument("--stats", action="store_true",
                        help="변환 결과 통계 표시")
    parser.add_argument("--unzip", action="store_true",
                        help="다운로드 ZIP 파일 압축 해제만")

    args = parser.parse_args()

    # 인자 없으면 도움말 표시
    if not any(vars(args).values()):
        parser.print_help()
        print("\n💡 처음이면 --guide 옵션으로 시작하세요!")
        return

    ensure_dirs()

    if args.guide:
        show_guide()
    elif args.scan:
        scan_data()
    elif args.unzip:
        merge_part_files()
        unzip_downloads()
    elif args.convert:
        convert_to_yolo()
        write_data_yaml()
        show_stats()
    elif args.all:
        run_all()
    elif args.stats:
        show_stats()


if __name__ == "__main__":
    main()
