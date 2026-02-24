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
