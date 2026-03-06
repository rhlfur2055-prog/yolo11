# YOLO26 — SafePlate & GoldenTime

한국 차량 번호판 자동 인식 + 긴급차량 미양보 탐지 + 물피도주 탐지 시스템

## 기능 개요

| 시스템 | 목적 | 트리거 | 판정 | 신고 대상 |
|--------|------|--------|------|-----------|
| **GoldenTime 2.0** | 긴급차량 진로 방해 차량 자동 채증 | 사이렌 감지 (S키) | 15초 연속 감지 + 거리 위반 | 안전신문고, 119 |
| **SafePlate 4K** | 물피도주(접촉사고 후 도주) 차량 자동 채증 | 충격 감지 (S키) | 충격 + 차량 이탈 | 경찰서, 보험사 |

## 핵심 아키텍처

```
YOLO11x 탐지 → ROI 크롭 (35%/40% 마진) → 500px 업스케일
→ 18가지 전처리 × PaddleOCR (한국어 특화 단독)
→ 위치 기반 분해 투표 → PlateValidator → HangulClassifier (초성 교차검증)
→ PlateTracker (IoU 기반 차량 추적)
→ EventBus 이벤트 → 시뮬레이션 모듈 (트리거 → 판정 → 증거)
```

## 정확도

- 정적 이미지: **12/12 (100%)**
- 실시간 영상 GT 매칭: **12/12 (100%)**
- Ghost Detection: **5/5 PASS**

## 실행 방법

### 골든타임 2.0 — 긴급차량 미양보 탐지

```bash
# GUI 모드 (대화형)
python -m simulation.simulation_framework movie/hiway.mp4 --goldentime

# 조작: S=사이렌 시작, E=사이렌 종료, SPACE=일시정지, Q=종료
```

### SafePlate 4K — 물피도주 탐지

```bash
# GUI 모드
python -m simulation.safeplate.run_safeplate --video movie/hiway.mp4

# 야간 모드 강제 ON
python -m simulation.safeplate.run_safeplate --video movie/night.mp4 --night

# Headless 자동 테스트
python -m simulation.safeplate.run_safeplate --video movie/hiway.mp4 --headless --shock-at 150

# 조작: S=충격 감지, E=충격 종료, N=야간 모드 토글, SPACE=일시정지, Q=종료
```

### 번호판 OCR 정확도 테스트

```bash
python test_ocr_accuracy.py
# 기대 결과: 12/12 = 100.0%
```

### 증거 대시보드

```bash
python build_dashboard.py
# 브라우저에서 evidence_dashboard.html 열기
```

## 파일 구조

```
YOLO26/
├── plate_engine_pro.py          # 메인 OCR 엔진 (YOLO + PaddleOCR 단독)
├── plate_gui.py                 # Tkinter GUI + 실시간 영상 처리
├── plate_ocr_postfilter_v2.py   # OCR 후처리/정제
├── test_ocr_accuracy.py         # 12장 정확도 테스트
├── build_dashboard.py           # 증거 대시보드 HTML 생성기
├── config.py                    # 중앙 설정
│
├── simulation/                  # 시뮬레이션 프레임워크 (4,826줄)
│   ├── simulation_framework.py  #   EventBus + SimulationFramework (541줄)
│   │
│   ├── goldentime/              #   골든타임 2.0 — 긴급차량 미양보
│   │   ├── siren_trigger.py     #     사이렌 감지 시뮬레이터 (264줄)
│   │   ├── plate_evidence.py    #     연속 감지 채증 (664줄)
│   │   ├── distance_checker.py  #     거리 판정 (604줄)
│   │   └── evidence_export.py   #     증거 패키지 생성 (549줄)
│   │
│   └── safeplate/               #   SafePlate 4K — 물피도주 탐지
│       ├── shock_simulator.py   #     충격 감지 시뮬레이터 (352줄)
│       ├── departure_detector.py#     이탈 차량 감지기 (419줄)
│       ├── evidence_package.py  #     증거 패키지 생성 (410줄)
│       ├── night_enhancer.py    #     야간 모드 보정 (557줄)
│       └── run_safeplate.py     #     통합 실행 스크립트 (460줄)
│
├── evidence_output/             # 증거 패키지 저장 폴더
│   ├── evidence_*               #   골든타임 증거 (3건)
│   └── safeplate_*              #   SafePlate 증거 (85건 한국 + 10,192건 외국)
│
├── .claude/skills/              # Claude Code 스킬 (5개)
│   ├── yolo26-dev/              #   개발 규칙 + EventBus 패턴
│   ├── yolo26-test/             #   테스트 자동화 + regression
│   ├── yolo26-simulation/       #   시뮬레이션 시나리오
│   ├── yolo26-evidence/         #   증거 패키지 + 대시보드
│   └── yolo26-ocr/              #   OCR 엔진 비교 + 최적화
│
├── best.pt                      # YOLO11x 번호판 전용 모델 (mAP@50=98.4%)
├── plate_ocr_crnn.pth           # CRNN OCR 모델 (42MB)
├── 22/                          # 테스트 이미지 12장 (Ground Truth = 파일명)
└── movie/                       # 테스트 영상
```

## 증거 패키지 구조

### 골든타임 2.0

```
evidence_YYYYMMDD_HHMMSS_번호판/
├── video.mp4           # 채증 영상 클립
├── plates.json         # 번호판 + 위반 데이터
├── screenshots/        # 핵심 순간 스크린샷 (6장)
└── report.txt          # 한글 신고 리포트 (안전신문고 + 119)
```

### SafePlate 4K

```
safeplate_YYYYMMDD_HHMMSS_번호판/
├── video.mp4           # 충격 전후 영상 클립
├── plates.json         # 번호판 + 이탈 정보 + departure_info
├── screenshots/        # 충격/이탈/종료 시점 스크린샷
└── report.txt          # 한글 신고 리포트 (경찰서 + 보험사)
```

### plates.json 주요 키

| 키 | 설명 |
|----|------|
| `version` | 데이터 형식 버전 |
| `system` | GoldenTime 또는 SafePlate |
| `plate` | 인식된 번호판 텍스트 |
| `evidence` | 채증 시간, 프레임 수, 감지 횟수 |
| `distance_violations` | 거리 위반 기록 (골든타임) |
| `departure_info` | 이탈 방향, 마지막 위치 (SafePlate) |
| `submission_targets` | 신고 대상 기관 |

## SafePlate 야간 모드

저조도 환경에서 번호판 인식률을 높이기 위한 자동 영상 보정:

- **자동 감지**: 프레임 밝기(V채널) 평균 80 이하 시 자동 활성화
- **CLAHE**: V채널 적응형 히스토그램 평활화
- **감마 보정**: LUT 기반 밝기 증가 (기본 1.8)
- **디노이징**: fastNlMeansDenoisingColored 야간 노이즈 제거
- **전조등 억제**: 과노출 영역 선택적 밝기 감소
- **히스테리시스**: 80 이하 ON / 95 이상 OFF (깜빡임 방지)

```bash
--night                  # 강제 활성화
--night-auto             # 자동 감지 (기본)
--night-threshold 80     # 밝기 임계값
--night-gamma 1.8        # 감마 보정 값
```

## EventBus 이벤트 흐름

### 골든타임 2.0

```
frame_read → SIREN_DETECTED → detection_result → DISTANCE_VIOLATION
                                                       ↓
                                              EVIDENCE_STARTED → EVIDENCE_COMPLETE
                                                                       ↓
                                                                  SIREN_ENDED
```

### SafePlate 4K

```
frame_read → SHOCK_DETECTED → detection_result → DEPARTURE_DETECTED
                                                       ↓
                                              EVIDENCE_EXPORTED → SHOCK_ENDED
```

## Claude Code 스킬

| 스킬 | 설명 | 자동 적용 조건 |
|------|------|---------------|
| `yolo26-dev` | 개발 규칙 + EventBus 패턴 | Python 코드 작성, simulation/ 작업 |
| `yolo26-test` | 테스트 자동화 + regression | headless 테스트, YOLO 스캔 |
| `yolo26-simulation` | 시뮬레이션 시나리오 | 골든타임/SafePlate 모듈 추가 |
| `yolo26-evidence` | 증거 패키지 + 대시보드 | evidence_output/ 작업 |
| `yolo26-ocr` | OCR 엔진 비교 + 최적화 | PaddleOCR/EasyOCR 설정 |

## 의존성

```
Python 3.10+
ultralytics          # YOLO11x
opencv-python        # OpenCV
paddleocr            # PaddleOCR (한국어)
easyocr              # EasyOCR (한국어+영어)
torch, torchvision   # CRNN 모델
numpy, Pillow        # 이미지 처리
tkinter              # GUI
```

## 커밋 이력

| 커밋 | 내용 |
|------|------|
| `2215934` | SafePlate 4K 완성 — 5모듈 + 야간모드 + 증거패키지 + 스킬5개 |
| `761e469` | 골든타임 2.0 시뮬레이션 6계단 파이프라인 + 증거 대시보드 |
| `6a84c2a` | 불필요한 로그/구버전 파일 정리 |
| `f9ecd49` | FLUSH 파이프라인 완성 + config.py 중앙 설정 통합 |
| `a7e959d` | initial commit |
