# YOLO11 — 한국 차량 번호판 인식 파이프라인

YOLO11x 객체 탐지 + PaddleOCR + CRNN 교차검증 기반의 한국 차량 번호판 실시간 인식 시스템.
정적 이미지 회귀 테스트 **11/12 (91.7%)** — 실패 케이스 1건을 README에 공개합니다. 실시간 영상 Ghost Detection 방지.

---

## 파이프라인 (7단계)

```
[1] 영상 입력 → [2] YOLO11x 탐지 → [3] ROI 크롭 + 18종 전처리
→ [4] PaddleOCR 인식 → [5] CRNN 교차검증
→ [6] 위치 기반 투표 + 형식 검증 → [7] PlateTracker 추적 + GUI 출력
```

### 1단계 — 영상 입력
프레임 캡처. 파일 영상(`movie/hiway.mp4`) 또는 라이브 캠 스트림.

| 항목 | 값 |
|------|----|
| 입력 해상도 | 1920×1080 권장 (4K 지원) |
| 프레임 처리 | OpenCV `VideoCapture` |
| 진입점 | `plate_gui.py` |

### 2단계 — YOLO11x 번호판 탐지
번호판 위치를 bounding box로 탐지.

| 항목 | 값 |
|------|----|
| 모델 | `best.pt` (YOLO11x fine-tuned) |
| 정확도 | mAP@50 = 98.4% |
| NMS | Ultralytics 기본값(IoU 0.7) — `plate_engine_pro.py`에 explicit 미설정 |
| 최소 박스 크기 | 폭 35px / 높이 16px (`config.py:202-203`) |
| 종횡비 필터 | 2.0 ≤ w/h ≤ 6.0 (`PLATE_MIN/MAX_ASPECT`) |
| 면적 비율 상한 | 프레임의 8% (`PLATE_MAX_AREA_RATIO`) |

### 3단계 — ROI 크롭 + 18종 전처리
탐지 박스를 잘라낸 뒤 OCR이 잘 읽도록 18가지 방식으로 변형.

| 항목 | 값 |
|------|----|
| 크롭 마진 | 좌우 35%, 상하 40% |
| 업스케일 | 500px |
| 전처리 종류 | 원본, 흑백, CLAHE, 샤프닝, 이진화, 컬러 채널별 등 18종 |
| 목적 | 조명·각도·해상도 변동에 대한 robustness 확보 |

### 4단계 — PaddleOCR 문자 인식
한국어 특화 PaddleOCR로 18장 전처리 이미지 각각에 대해 OCR 수행.

| 항목 | 값 |
|------|----|
| 모델 | PaddleOCR (한국어 단독) |
| 신뢰도 필터 | 0.40 이상만 후보 채택 |
| 출력 | 18개의 (text, confidence) 후보 |

### 5단계 — CRNN 교차검증
PaddleOCR이 잘 틀리는 한글 부분(`나↔라`, `버↔아` 등)을 CRNN으로 재판독.

| 항목 | 값 |
|------|----|
| 모델 | `plate_ocr_crnn.pth` (10.5M params) |
| 구조 | CNN(6블록) + BiLSTM(2층) + CTC |
| 학습 | 실제 132장 × 10증강(1,320) + 합성 20,647장 = **21,967 샘플** / 200 epoch / RTX 4060 |
| 검증 | 131/132 (99%) |
| 부가 기능 | 2줄 번호판(구형) 지역명 복원 |

### 6단계 — 위치 기반 투표 + 형식 검증
자릿수별 독립 투표 후 한국 번호판 규칙으로 최종 확정.

| 컴포넌트 | 역할 |
|----------|------|
| 위치 기반 분해 투표 | 자릿수마다 18개 후보 중 다수결 (예: 15/18이 `7` → `7` 확정) |
| `PlateValidator` | 한국 번호판 형식(7자리/2줄) 검증 |
| `HangulClassifier` | 초성 기준 한글 교차검증 |

### 7단계 — PlateTracker 추적 + GUI 출력
같은 차량의 번호판을 프레임 간 IoU로 묶어 안정화하고 화면에 표시.

| 항목 | 값 |
|------|----|
| 매칭 | IoU 기반 트랙 + 텍스트 기반 트랙 병합 |
| TTL | 30프레임 |
| Grace period | 3프레임 |
| Ghost Detection | 5/5 PASS (사라진 차량 잔상 제거) |
| 출력 | Tkinter GUI 실시간 표시 |

---

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `plate_engine_pro.py` | OCR 엔진 본체 (YOLO + PaddleOCR + CRNN + 투표 + 추적) |
| `plate_gui.py` | Tkinter GUI + 실시간 영상 루프 |
| `plate_recognition_4k.py` | 한글 교정 함수 라이브러리 |
| `test_ocr_accuracy.py` | 정확도 회귀 테스트 (12장) |
| `best.pt` | YOLO11x 번호판 탐지 모델 |
| `plate_ocr_crnn.pth` | CRNN 한글 교차검증 모델 |

---

## 실행

```bash
# 패키지 설치
pip install -r requirements.txt

# GUI 실행 (기본 영상)
python plate_gui.py

# GUI 실행 (특정 영상)
python plate_gui.py movie/hiway.mp4

# 정확도 회귀 테스트
python test_ocr_accuracy.py
```

기대 결과: `11/12 = 91.7%` (현재 실측 — 실패 #10 `58두9599` 1건 공개, [성능] 섹션 참고)

---

## 성능

| 항목 | 값 |
|------|----|
| 정적 이미지 12장 (회귀 테스트) | **11/12 (91.7%)** — `test_ocr_accuracy.py` 실측 |
| 이미지당 평균 추론 시간 | 약 1.35초 (CPU, 0.57s~4.62s 분포) |
| 실시간 영상 GT 매칭 | 12/12 (100%) |
| Ghost Detection | 5/5 PASS |
| OCR 처리 시간 | 0.6 – 3.2초 / 이미지 (CPU) |

### 알려진 실패 케이스 (정직성 공개)

| # | 파일 | 정답 | OCR 결과 | 원인 |
|---|------|------|---------|------|
| 10 | `58두9599.png` | `58두9599` | `(미인식)` | PaddleOCR이 한글 `두` 한 글자 누락 → `589599`(숫자 6자리) 반환. 현재 CRNN 폴백은 4자리 숫자 패턴만 트리거하여 6자리 케이스에서 보정 실패. **개선 진행 중** (CRNN 디지트 폴백 패턴 확장 예정). |

## 지원 번호판

| 종류 | 예시 | 처리 단계 |
|------|------|-----------|
| 신형 1줄 (7자리) | `01나8060`, `86오1144` | 5,6단계로 충분 |
| 구형 2줄 (지역명) | `서울70바9203`, `경기91바6286` | 5단계 CRNN 복원 |
| 녹색 영업용 | `36다7117` | 3단계 컬러 전처리 |
| 노란 번호판 | `경기76바7789` | 3단계 컬러 전처리 |

## 원거리 인식 한계

| 번호판 크기 비율 | 결과 |
|-----------------|------|
| < 5% | 인식 불가 (번호판 ~50px) |
| 5–10% | 숫자 4자리만 |
| 10–25% | 지역명 오독 가능 |
| 25% 이상 | 완전 인식 |

---

## 의존성

- Python 3.10–3.12 (3.13은 PaddlePaddle 미지원)
- `ultralytics` (YOLO11)
- `opencv-python`
- `paddlepaddle` + `paddleocr` (한국어 단독, **`paddleocr` 단독으론 실행 불가** — `paddlepaddle` 백엔드 필수)
- `torch`, `torchvision` (CRNN)
- `numpy`, `Pillow`
- `tkinter` (Python 기본 내장)

---

## 절대 규칙

1. 변경 후 `python test_ocr_accuracy.py` 회귀 테스트로 baseline(현재 11/12) 대비 회귀 없는지 반드시 확인
2. `best.pt`, `plate_ocr_crnn.pth` 모델 파일 수정 금지
3. `plate_engine_pro.py` 수정 시 regression 주의
