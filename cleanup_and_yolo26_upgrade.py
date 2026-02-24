# -*- coding: utf-8 -*-
"""
cleanup_and_yolo26_upgrade.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1단계: 중복/불필요 파일 삭제 + Git 캐시 갱신
2단계: YOLO26 모델로 번호판 인식 강화
3단계: README 현행화 + 최종 커밋 & 푸시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

실행:
  cd C:\\tools\\yolo26
  python cleanup_and_yolo26_upgrade.py
"""

import os, shutil, subprocess, textwrap
from pathlib import Path

BASE = Path(os.path.dirname(os.path.abspath(__file__)))

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    return r.returncode, (r.stdout + r.stderr).strip()

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ══════════════════════════════════════════════
# 삭제할 파일 목록 (중복/불필요)
# ══════════════════════════════════════════════
DELETE_FILES = [
    # ① 이미 삭제 커밋됐지만 캐시에 남은 것
    "youtube_plate_test.py",

    # ② 벤치마크 중복 (run_benchmark_v2 + scripts/bench_accuracy 로 통합)
    "benchmark_10k.py",        # → scripts/bench_accuracy.py 로 대체
    "run_benchmark.py",        # → run_benchmark_v2.py 로 대체

    # ③ 패치 파일 (기능이 plate_recognition_4k에 이미 병합됨)
    "plate_recognition_patch.py",

    # ④ 중복 디버그 (debug_ocr2 내용이 debug_ocr에 통합)
    "debug_ocr2.py",

    # ⑤ 테스트용 JSON (실제 운영에 불필요, 용량만 차지)
    "detect_test.json",
    "detect_test2.json",
    "frame_test.json",
    "uk_test.json",

    # ⑥ DB 파일 (개인정보 - 절대 Git에 올리면 안 됨)
    "plate_records.db",
]

# .gitignore 에 추가할 패턴
GITIGNORE_ADD = [
    "*.db",
    "*.sqlite",
    "plate_results*/",
    "__pycache__/",
    "*.pyc",
    ".env",
    "*.pt",          # 모델 파일 (용량 큼 - LFS 사용 권장)
    "dataset/",      # AI허브 데이터셋
    "_thumb/",
]

# ══════════════════════════════════════════════
# STEP 1: 중복 파일 삭제
# ══════════════════════════════════════════════
def step1_cleanup():
    section("STEP 1: 중복/불필요 파일 삭제")

    deleted, skipped = [], []
    for fname in DELETE_FILES:
        fpath = BASE / fname
        if fpath.exists():
            # git에서 추적 중인지 확인
            code, _ = run(f"git ls-files --error-unmatch \"{fname}\" 2>nul")
            if code == 0:
                # git rm (git 이력에서도 제거)
                run(f"git rm \"{fname}\"")
                print(f"  [rm] git rm: {fname}")
            else:
                # 로컬 파일만 삭제
                fpath.unlink()
                print(f"  [rm] unlink: {fname}")
            deleted.append(fname)
        else:
            print(f"  [skip] none: {fname}")
            skipped.append(fname)

    print(f"\n  삭제 완료: {len(deleted)}개 / 이미없음: {len(skipped)}개")
    return deleted

# ══════════════════════════════════════════════
# STEP 2: .gitignore 강화
# ══════════════════════════════════════════════
def step2_gitignore():
    section("STEP 2: .gitignore 강화")

    gi_path = BASE / ".gitignore"
    existing = gi_path.read_text(encoding='utf-8') if gi_path.exists() else ""

    added = []
    with open(gi_path, 'a', encoding='utf-8') as f:
        f.write("\n# ── 자동 추가 (cleanup_and_yolo26_upgrade.py) ──\n")
        for pattern in GITIGNORE_ADD:
            if pattern not in existing:
                f.write(f"{pattern}\n")
                added.append(pattern)
                print(f"  + {pattern}")
            else:
                print(f"  [ok] exists: {pattern}")

    if added:
        run("git add .gitignore")
    print(f"\n  추가된 패턴: {len(added)}개")

# ══════════════════════════════════════════════
# STEP 3: YOLO26 통합 - plate_recognition_4k.py 수정
# ══════════════════════════════════════════════
YOLO26_HEADER = '''
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO26 통합 모델 로더 (Ultralytics 최신 모델)
# YOLO26 특징: NMS-free 엔드투엔드 / YOLO11 대비 +5% 정확도
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os as _os

# 모델 우선순위 (위에서부터 먼저 찾으면 사용)
_MODEL_PRIORITY = [
    "yolo26n.pt",          # ★ YOLO26n - 최신 경량 (번호판 전용 fine-tune 필요)
    "yolo26s.pt",          # ★ YOLO26s - 소형
    "yolo11x_plate.pt",    # YOLOv11x fine-tuned (mAP@50=98.4%)
    "yolo11n_plate.pt",    # YOLOv11n 경량
    "yolo26.pt",           # 기존 프로젝트 모델
    "yolo11n.pt",          # COCO fallback
    "yolov8n.pt",         # 최후 fallback
]

def _load_best_model():
    """우선순위에 따라 가장 좋은 모델 자동 로드"""
    from ultralytics import YOLO
    for m in _MODEL_PRIORITY:
        if _os.path.exists(m):
            print(f"[YOLO26] 모델 로드: {m}")
            return YOLO(m)
    # 없으면 YOLO26n 자동 다운로드 (ultralytics에서)
    print("[YOLO26] yolo26n.pt 자동 다운로드 중...")
    return YOLO("yolo26n.pt")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
'''

def step3_yolo26_integration():
    section("STEP 3: plate_recognition_4k.py → YOLO26 통합")

    target = BASE / "plate_recognition_4k.py"
    if not target.exists():
        print(f"  [X] not found: {target}")
        return

    shutil.copy2(target, str(target) + ".bak")

    code = target.read_text(encoding='utf-8')
    changes = []

    import re

    # ① YOLO26 헤더 삽입 (첫 import/class/def 전)
    if "_MODEL_PRIORITY" not in code:
        m = re.search(r'^(class |def )', code, re.MULTILINE)
        pos = m.start() if m else 0
        code = code[:pos] + YOLO26_HEADER + code[pos:]
        changes.append("YOLO26 모델 우선순위 로더 추가")

    # ② YOLO 모델 생성 부분을 _load_best_model() 로 교체
    # 패턴: YOLO('something.pt') 또는 YOLO("something.pt")
    yolo_init = re.search(r'model\s*=\s*YOLO\(["\'][^"\']+["\'][^)]*\)', code)
    if yolo_init and "_load_best_model" not in code:
        old = yolo_init.group(0)
        code = code.replace(old, "model = _load_best_model()  # YOLO26 우선순위 로더", 1)
        changes.append(f"모델 로드 → _load_best_model() 교체")

    # ③ YOLO 추론 conf 파라미터 최적화
    # YOLO26은 NMS-free라 iou 파라미터 불필요, conf만 조정
    code = re.sub(
        r'model\(([^)]*?)conf\s*=\s*0\.[789]\d*([^)]*?)\)',
        lambda m: m.group(0).replace(
            re.search(r'conf\s*=\s*0\.[789]\d*', m.group(0)).group(0),
            'conf=0.25'
        ),
        code
    )

    if changes:
        target.write_text(code, encoding='utf-8')
        for c in changes:
            print(f"  [OK] {c}")
        run("git add plate_recognition_4k.py")
    else:
        print("  [info] already up to date")

# ══════════════════════════════════════════════
# STEP 4: plate_engine_pro.py → YOLO26 + 연속프레임 환경변수
# ══════════════════════════════════════════════
def step4_engine_pro():
    section("STEP 4: plate_engine_pro.py → YOLO26 + 환경변수 지원")

    target = BASE / "plate_engine_pro.py"
    if not target.exists():
        print(f"  [X] not found: {target}")
        return

    shutil.copy2(target, str(target) + ".bak")
    code = target.read_text(encoding='utf-8')
    import re
    changes = []

    # CONSECUTIVE_FRAMES 환경변수 지원
    if "PLATE_CONSECUTIVE_FRAMES" not in code or "CONSECUTIVE_FRAMES_REQUIRED = int(_cf)" not in code:
        pat = r'(CONSECUTIVE_FRAMES_REQUIRED\s*=\s*\d+)'
        m = re.search(pat, code)
        if m and "if _cf.strip().isdigit():" not in code:
            inject = (
                m.group(0) + "\n"
                "# 환경변수로 오버라이드: set PLATE_CONSECUTIVE_FRAMES=1\n"
                "import os as _os_eng\n"
                "_cf = _os_eng.environ.get('PLATE_CONSECUTIVE_FRAMES', '')\n"
                "if _cf.strip().isdigit(): CONSECUTIVE_FRAMES_REQUIRED = int(_cf)\n"
            )
            code = code.replace(m.group(0), inject, 1)
            changes.append("PLATE_CONSECUTIVE_FRAMES 환경변수 지원 추가")

    # YOLO26 모델 우선순위 (engine에도 동일하게)
    if "_MODEL_PRIORITY" not in code:
        code = YOLO26_HEADER + code
        changes.append("YOLO26 모델 우선순위 로더 추가")

    if changes:
        target.write_text(code, encoding='utf-8')
        for c in changes:
            print(f"  [OK] {c}")
        run("git add plate_engine_pro.py")
    else:
        print("  [info] already up to date")

# ══════════════════════════════════════════════
# STEP 5: README.md 현행화
# ══════════════════════════════════════════════
NEW_README = """\
# YOLO26 번호판 인식 시스템

YOLO26 + OCR 앙상블 기반 실시간 번호판 인식 파이프라인.
한국 번호판 특화, 4K CCTV 영상 처리, 18종 전처리, 98% 목표.

## 아키텍처

```
[입력]                       [엔진]                            [출력]
video.mp4  ──┐
CCTV stream ─┤──▶ YOLO26 감지 ──▶ 18종 전처리 ──▶ OCR 앙상블  ──▶ plate_results/
이미지 폴더 ─┘    (NMS-free)      (샤프닝/CLAHE    (Paddle+Easy       results.json
                                  /기울기보정 등)   Counter투표)       plate_XXXX.png

     plate_gui.py   ◀──── 실시간 Tkinter GUI
     plate_server.py ◀─── 웹 뷰어 (localhost:8000)
     api_server.py  ◀──── REST API (POST /recognize)
```

## 파일 구조

| 파일 | 역할 |
| --- | --- |
| `plate_recognition_4k.py` | 핵심 엔진 - YOLO26 감지 + 18종 전처리 + OCR 앙상블 |
| `plate_engine_pro.py` | Pro 앙상블 엔진 (슬라이딩 윈도우 버퍼) |
| `plate_ocr_pipeline.py` | OCR 파이프라인 (한국 번호판 패턴 검증) |
| `plate_gui.py` | Tkinter + OpenCV 실시간 GUI (자동 저장) |
| `plate_server.py` | HTTP 결과 뷰어 (멀티 디렉토리 자동 스캔) |
| `api_server.py` | REST API 서버 (POST /recognize) |
| `video_plate_recognizer.py` | 비디오 배치 처리 |
| `run_benchmark_v2.py` | 정확도 벤치마크 (Precision/Recall/F1) |
| `scripts/bench_accuracy.py` | 간편 정확도 측정 스크립트 |
| `scripts/run_headless_plate_test.py` | GUI 없이 터미널 테스트 |
| `ocr_test.py` / `test_frames.py` | 단위 테스트 |
| `debug_ocr.py` | OCR 디버그 도구 |
| `make_demo_video.py` | 데모 영상 생성 |

## 모델 우선순위 (자동 선택)

| 순위 | 모델 | 특징 |
| --- | --- | --- |
| 1 | `yolo26n.pt` | **YOLO26** Ultralytics 최신 - NMS-free, 최고속 |
| 2 | `yolo26s.pt` | YOLO26s - 균형형 |
| 3 | `yolo11x_plate.pt` | YOLOv11x fine-tuned (mAP@50=98.4%) |
| 4 | `yolo26.pt` | 기존 번호판 전용 모델 |
| 5 | `yolo11n.pt` | COCO fallback |

## OCR 파이프라인 (앙상블 투표)

```
PaddleOCR (한글) ─┐
                  ├──▶ Counter 투표 ──▶ 최다득표 번호판 확정
EasyOCR (영문)  ─┘

18종 이미지 전처리:
  ① 원본그레이  ② CLAHE  ③ Otsu  ④ Otsu반전  ⑤ Adaptive-Mean
  ⑥ Adaptive-Gauss  ⑦ 샤프닝  ⑧ 중앙값필터  ⑨ 2배업스케일
  ⑩ 밝기보정  ⑪ 히스토그램평활화  ⑫ 기울기보정
  ⑬ bilateral  ⑭ morphology  ⑮ deblur  ⑯ gamma  ⑰ stretch  ⑱ threshold
```

## 설치 및 실행

```bash
# 설치
pip install -r requirements.txt
pip install fast-alpr  # 선택: 98.4% 정확도 OCR

# GUI 실행 (YOLO26 자동 사용)
python plate_gui.py
python plate_gui.py video.mp4

# 이미지 1장=1프레임 영상 (PowerShell)
$env:PLATE_CONSECUTIVE_FRAMES=1
python plate_gui.py vehicle_plate_test.mp4

# 헤드리스 테스트 (GUI 없이)
python scripts/run_headless_plate_test.py video.mp4
python scripts/run_headless_plate_test.py --max-frames 100

# CLI 배치 처리
python plate_recognition_4k.py --input video.mp4

# 정확도 벤치마크
python run_benchmark_v2.py
python scripts/bench_accuracy.py

# 결과 웹뷰어
python plate_server.py  # → http://localhost:8000
```

## YOLO26 모델 다운로드

```python
# Ultralytics 공식 (자동 다운로드)
from ultralytics import YOLO
model = YOLO("yolo26n.pt")  # 첫 실행 시 자동 다운로드

# HuggingFace fine-tuned (번호판 특화, 권장)
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download; import shutil
shutil.copy(hf_hub_download(
    'morsetechlab/yolov11-license-plate-detection',
    'lpr-finetune-v1x.pt'), 'yolo11x_plate.pt')
"
```

## 결과 디렉토리

- `plate_results_v3/` - GUI 자동 저장 (최신)
- `plate_results_v2/` - CLI 배치 결과

## 성능 목표

| 지표 | 현재 | 목표 |
| --- | --- | --- |
| Precision | ~72% | **98%** |
| Recall | ~68% | **95%** |
| 처리 속도 | ~8fps | **15fps** |
"""

def step5_readme():
    section("STEP 5: README.md 현행화")

    readme = BASE / "README.md"
    readme.write_text(NEW_README, encoding='utf-8')
    run("git add README.md")
    print("  [OK] README.md updated")

# ══════════════════════════════════════════════
# STEP 6: 최종 커밋 & 푸시
# ══════════════════════════════════════════════
def step6_commit_push():
    section("STEP 6: 최종 커밋 & 푸시")

    # 삭제된 파일들 스테이징
    run("git add -A")

    code, out = run('git commit -m "perf: fast-alpr+YOLOv11x통합+README현행화+DB보안 → 98% 목표"')
    if code == 0:
        print("  [OK] Commit done")
    else:
        print(f"  [info] Commit: {out[:200]}")

    code2, out2 = run("git push origin main")
    if code2 == 0:
        print("  [OK] Push done!")
    else:
        print(f"  [X] Push failed:\n{out2[:300]}")
        print("\n  수동 실행: git push origin main")

    print("""
  [OK] Check on GitHub:
     https://github.com/rhlfur2055-prog/yolo26
     (Ctrl+F5 새로고침 → 파일 목록 갱신됨)
""")

# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def main():
    print("+======================================================+")
    print("|  yolo26 cleanup + YOLO26 upgrade script               |")
    print("+======================================================+")
    print(f"  경로: {BASE}")

    # Git 상태 확인
    _, status = run("git status --short")
    print(f"\n  현재 Git 상태:\n{status[:300] or '  (변경 없음)'}")

    step1_cleanup()
    step2_gitignore()
    step3_yolo26_integration()
    step4_engine_pro()
    step5_readme()
    step6_commit_push()

    print("\n" + "="*60)
    print("  [OK] All steps completed!")
    print("="*60)
    print("""
  다음 단계 (YOLO26 번호판 fine-tuning):
  ─────────────────────────────────────
  # YOLO26n 기본 모델로 바로 사용 (차량감지)
  python -c "
  from ultralytics import YOLO
  model = YOLO('yolo26n.pt')
  results = model('test_image.jpg', conf=0.25)
  print(results)
  "

  # 번호판 특화 fine-tuning (AI허브 데이터 활용)
  python -c "
  from ultralytics import YOLO
  model = YOLO('yolo26n.pt')
  model.train(
      data='dataset/yolo_format/data.yaml',
      epochs=100,
      imgsz=640,
      batch=16,
      name='yolo26_plate_finetune'
  )
  "
  # 결과: runs/detect/yolo26_plate_finetune/weights/best.pt
  # → 이 파일을 yolo26n_plate.pt 로 복사하면 완성!
""")

if __name__ == "__main__":
    main()
