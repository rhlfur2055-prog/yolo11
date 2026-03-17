# YOLO26 — 한국 차량 번호판 자동 인식 시스템

YOLO 객체 탐지 + PaddleOCR + CRNN 기반 한국 차량 번호판 실시간 인식 시스템

## ⚠️ 팀원 필독 — 핵심 엔진 & 최소 실행 가이드

### 이것만은 알아야 한다

| 구분 | 파일 | 역할 | 중요도 |
|------|------|------|--------|
| **🔴 핵심 엔진** | `plate_engine_pro.py` | YOLO 탐지 + PaddleOCR + CRNN 교차검증 + 투표 시스템 | ★★★ 절대 수정 금지 |
| **🔴 YOLO 모델** | `best.pt` | 번호판 탐지 전용 YOLO11x (mAP@50=98.4%) | ★★★ 수정 금지 |
| **🔴 CRNN 모델** | `plate_ocr_crnn.pth` | 한글 교차검증 + 2줄 번호판 복원 (132장 학습) | ★★★ 수정 금지 |
| **🟡 GUI** | `plate_gui.py` | 실시간 영상 처리 + 화면 표시 | ★★ |
| **🟡 후처리** | `plate_ocr_postfilter_v2.py` | OCR 결과 정제/교정 | ★★ |
| **🟢 테스트** | `test_ocr_accuracy.py` | 12장 정확도 검증 (반드시 100% 유지) | ★ |

### 최소 실행 환경 (이것만 설치하면 된다)

```bash
# 1. Python 3.10+ 설치 후
pip install ultralytics opencv-python paddleocr torch torchvision numpy Pillow

# 2. 필수 파일 확인 (이 4개 없으면 실행 불가)
#    best.pt              — YOLO 모델 (번호판 탐지)
#    plate_ocr_crnn.pth   — CRNN 모델 (한글 교차검증)
#    plate_engine_pro.py  — 핵심 OCR 엔진
#    plate_gui.py         — GUI 화면

# 3. 실행
python plate_gui.py                    # 기본 영상으로 실행
python plate_gui.py movie/hiway.mp4   # 특정 영상 지정

# 4. 정확도 테스트 (변경 후 반드시 실행!)
python test_ocr_accuracy.py
# 결과: 12/12 = 100.0% 나와야 정상
```

### 절대 하지 말 것 ❌
1. **`.pt`, `.pth` 모델 파일 삭제/수정** — 시스템 전체 작동 불가
2. **`plate_engine_pro.py` 함부로 수정** — regression 발생 시 복구 어려움
3. **테스트 안 돌리고 커밋** — `python test_ocr_accuracy.py` 반드시 12/12 확인

---

## 번호판 인식 과정 (이렇게 인식된다)

### 전체 흐름
```
영상 입력 → YOLO11x 번호판 탐지 → ROI 크롭 → 전처리 → PaddleOCR 문자 인식
→ CRNN 교차검증 → 투표 → 결과 출력
```

### 1단계: YOLO로 번호판 찾기
- **YOLO11x** 모델이 프레임에서 번호판 위치를 탐지 (mAP@50=98.4%)
- 중복 탐지 제거 (IoU > 0.65)
- 작은 번호판 필터: 70px 미만은 신뢰도 낮춤

### 2단계: 번호판 이미지 전처리
- 탐지된 번호판을 크롭 + 여유 마진 추가 (좌우 35%, 상하 40%)
- 500px로 업스케일 (해상도 확보)
- **18가지 전처리** 적용: 원본, 흑백, CLAHE, 샤프닝, 이진화 등
  → 각각 OCR 돌려서 가장 좋은 결과 선택

### 3단계: PaddleOCR로 문자 인식
- **PaddleOCR** (한국어 특화) 단독 사용
- 18가지 전처리 결과에서 각각 OCR 실행
- 신뢰도 0.40 이상만 후보로 채택

### 4단계: CRNN 교차검증
- PaddleOCR이 한글을 잘못 읽을 수 있음 (예: 나→라, 버→아)
- **CRNN 모델**이 번호판 전체를 다시 읽어서 교차 검증
- 특히 한글 부분 정확도를 높여줌
- 2줄 번호판(구형)도 지역명 복원 가능

### 5단계: 투표 + 최종 결과
- **위치 기반 분해 투표**: 각 자릿수별로 독립 투표
  - 예: 18가지 전처리 중 15개가 "7"이면 → "7" 확정
- **PlateValidator**: 한국 번호판 형식 맞는지 검증
- **HangulClassifier**: 초성 기준 한글 교차검증

### 6단계: 차량 추적 (실시간 영상)
- **PlateTracker**: IoU 기반으로 같은 차량 추적
- 같은 번호판이 여러 프레임에서 인식되면 신뢰도 누적
- Ghost Detection 방지 (사라진 차량의 잔상 제거)

---

## 정확도

| 테스트 | 결과 |
|--------|------|
| 정적 이미지 12장 | **12/12 (100%)** ✅ |
| 실시간 영상 GT 매칭 | **12/12 (100%)** ✅ |
| Ghost Detection | **5/5 PASS** ✅ |
| OCR 처리 시간 | 0.6~3.2초/이미지 (CPU) |

## 인식 가능 번호판 종류

| 종류 | 예시 | 지원 |
|------|------|------|
| 신형 1줄 (7자리) | `01나8060`, `86오1144` | ✅ 완전 지원 |
| 구형 2줄 (지역명) | `서울70바9203`, `경기91바6286` | ✅ CRNN 복원 |
| 녹색 영업용 | `36다7117` | ✅ 컬러 전처리 |
| 노란 번호판 | `경기76바7789` | ✅ 컬러 전처리 |

## 원거리 인식 한계

| 번호판 크기 비율 | 인식 수준 | 설명 |
|-----------------|-----------|------|
| < 5% | ❌ 인식 불가 | 번호판 ~50px, 해상도 한계 |
| 5-10% | ⚠️ 숫자만 | 4자리 숫자만, 지역명 복원 시도 |
| 10-25% | 🟡 부분 인식 | 지역명 오독 가능성 |
| 25%+ | ✅ 완전 인식 | 전체 정확 인식 |

## 파일 구조

```
YOLO26/
├── plate_engine_pro.py          # 🔴 핵심 OCR 엔진 (YOLO + PaddleOCR + CRNN)
├── plate_gui.py                 # 🟡 GUI + 실시간 영상 처리
├── plate_ocr_postfilter_v2.py   # 🟡 OCR 후처리/정제
├── plate_recognition_4k.py      # 한글 교정 함수 라이브러리
├── train_plate_ocr.py           # CRNN 학습 스크립트
├── test_ocr_accuracy.py         # 🟢 정확도 테스트 (12장)
│
├── best.pt                      # 🔴 YOLO 모델 (번호판 탐지)
├── plate_ocr_crnn.pth           # 🔴 CRNN 모델 (한글 교차검증)
├── 22/                          # 테스트 이미지 12장
└── movie/                       # 테스트 영상
```

## CRNN 모델 정보

| 항목 | 내용 |
|------|------|
| 구조 | CNN(6블록) + BiLSTM(2층) + CTC |
| 파라미터 | 10,587,862개 |
| 학습 데이터 | 실제 132장 + 합성 20,647장 = 21,967 샘플 |
| 학습 장비 | RTX 4060 (CUDA) |
| 학습 에폭 | 200 |
| 검증 정확도 | 131/132 (99%) |

## 의존성

```bash
pip install ultralytics opencv-python paddleocr torch torchvision numpy Pillow
```

- Python 3.10+
- ultralytics — YOLO11x 객체 탐지
- opencv-python — 이미지/영상 처리
- paddleocr — 한국어 OCR (핵심)
- torch, torchvision — CRNN 모델
- numpy, Pillow — 이미지 처리
- tkinter — GUI (Python 기본 내장)
