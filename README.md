# YOLO26 — 한국 차량 번호판 자동 인식 시스템

YOLO 객체 탐지 + 다중 OCR 앙상블 기반 한국 차량 번호판 인식 파이프라인.
정적 이미지 **12/12 (100%)**, 실시간 영상 Ghost Detection **5/5 (100%)** 달성.

## 아키텍처

```
입력 (이미지/영상)
  │
  ▼
YOLO11x 번호판 탐지 (mAP@50 = 98.4%)
  │
  ▼
ROI 크롭 (35%/40% 마진) → 500px 업스케일
  │
  ▼
18가지 전처리 × 2 OCR 엔진 (PaddleOCR + EasyOCR)
  │
  ▼
위치 기반 분해 투표 → PlateValidator → HangulClassifier (초성 교차검증)
  │
  ▼
PlateTracker (IoU 기반 차량 추적, Ghost Detection 방지)
```

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `plate_engine_pro.py` | 메인 OCR 엔진 — YOLO 탐지 + 18종 전처리 + 앙상블 투표 + 추적 |
| `plate_gui.py` | Tkinter GUI + 실시간 영상 처리 |
| `plate_ocr_postfilter_v2.py` | OCR 후처리/정제 필터 |
| `test_ocr_accuracy.py` | 정적 이미지 12장 정확도 테스트 |
| `test_ghost_detection.py` | 실시간 영상 Ghost Detection 테스트 |
| `train_plate_ocr.py` | CRNN OCR 모델 학습 스크립트 |

## 실행 방법

### GUI 실행 (실시간 영상)

```bash
python plate_gui.py movie/hiway.mp4
```

기본 영상 없이 실행하면 내장 테스트 영상을 자동으로 사용합니다:

```bash
python plate_gui.py
```

### 정확도 테스트

```bash
# 정적 이미지 OCR 정확도 (12장, Python 3.10 필수)
python test_ocr_accuracy.py

# 실시간 Ghost Detection 테스트
python test_ghost_detection.py
```

## 설치

```bash
pip install -r requirements.txt
```

### 의존성

- Python 3.10+
- ultralytics (YOLO), opencv-python, numpy, Pillow
- paddleocr, easyocr (OCR 앙상블)
- torch, torchvision (CRNN 모델)
- tkinter (GUI)

## 모델 파일

| 모델 | 용도 |
|------|------|
| `best.pt` | YOLO11x 번호판 전용 fine-tuned (mAP@50=98.4%) |
| `yolo11x_plate.pt` | YOLO11x 번호판 전용 |
| `plate_ocr_crnn.pth` | CRNN OCR 모델 (42MB) |
| `yolo26n.pt` | YOLO26 nano fallback |
| `yolo11n.pt` | YOLOv11 nano fallback |

## 현재 성능

| 테스트 | 결과 |
|--------|------|
| 정적 이미지 OCR 정확도 | **12/12 (100%)** |
| 실시간 Ghost Detection | **5/5 (100%)** |

## 테스트 데이터

- `22/` — 번호판 이미지 12장 (파일명 = Ground Truth)
- `movie/hiway.mp4` — 고속도로 실시간 테스트 영상

## 알려진 한계

- **CCTV급 소형 번호판** — 해상도 제약으로 OCR 정확도 저하 가능
- **흰색/은색 번호판** — 그림자/음영에 의한 HSV 임계값 오판 가능성
- **OCR 처리 시간** — CPU 기준 이미지당 5~16초 (GPU 미사용 시)

## 라이선스

이 프로젝트는 교육/연구 목적으로 개발되었습니다.
