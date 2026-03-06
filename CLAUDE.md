# YOLO26 한국 차량 번호판 인식 프로젝트

## 프로젝트 개요
YOLO 객체 탐지 + PaddleOCR(한국어 특화) 단독 기반 한국 차량 번호판 자동 인식 시스템.
정적 이미지 12/12 (100%) 정확도 달성. 실시간 영상 Ghost Detection 해결 중.

## 핵심 아키텍처
```
YOLO 탐지 → ROI 크롭 (35%/40% 마진) → 500px 업스케일
→ 18가지 전처리 × PaddleOCR (한국어 특화 단독)
→ 위치 기반 분해 투표 → PlateValidator → HangulClassifier (초성 교차검증)
→ PlateTracker (IoU 기반 차량 추적, Ghost Detection 방지)
```

## 파일 맵 + 소유권
| 파일 | 역할 | 소유 터미널 | 줄 수 |
|------|------|------------|-------|
| `plate_engine_pro.py` | 메인 OCR 엔진 (탐지+인식+추적) | 터미널 3 | ~2330 |
| `plate_gui.py` | Tkinter GUI + 실시간 영상 처리 | 터미널 2 | ~972 |
| `plate_ocr_postfilter_v2.py` | OCR 후처리/정제 | 터미널 3 | ~650 |
| `test_ocr_accuracy.py` | 12장 정확도 테스트 | 터미널 4 | ~142 |
| `plate_recognition_4k.py` | 한글 교정 함수 라이브러리 | 읽기 전용 | ~2925 |
| `plate_ocr_postfilter.py` | 구버전 후처리 (레거시) | 수정 금지 | ~263 |
| `train_plate_ocr.py` | CRNN 학습 스크립트 | 수정 금지 | ~493 |
| `youtube_helper.py` | YouTube 영상 다운로드 | 수정 금지 | ~149 |
| `CLAUDE.md` | 프로젝트 규칙 (이 파일) | 터미널 1 | - |

## 모델 파일 (수정 금지)
- `best.pt` — YOLO11x 번호판 전용 fine-tuned (mAP@50=98.4%)
- `yolo11x_plate.pt` — YOLO11x 번호판 전용
- `plate_ocr_crnn.pth` — CRNN OCR 모델 (42MB)
- `yolo26n.pt`, `yolo11n.pt` — fallback 모델

## 테스트 데이터
- `22/` 폴더: 12장 번호판 이미지 (Ground Truth = 파일명)
- `movie/hiway.mp4` — 실시간 테스트 영상

## 절대 규칙
1. **Regression 금지** — 현재 12/12 통과 케이스 절대 깨뜨리지 않기
2. **변경 후 테스트 필수** — `python test_ocr_accuracy.py` 반드시 실행
3. **파일 소유권 준수** — 다른 터미널 담당 파일은 읽기만 가능
4. **주석은 한글** — 모든 코드 주석 한글로 작성
5. **수정 전 기존 로직 읽기** — 반드시 현재 코드를 읽은 후 수정
6. **Mock/가짜 코드 금지** — 실제 동작하는 코드만 작성
7. **모델 파일 수정 금지** — .pt, .pth 파일 절대 건드리지 않기

## 현재 이슈: Ghost Detection (실시간 영상)
### 문제
- 이전 차량 번호판이 다음 차량에 표시됨 (예: 58주9599 → 01나8060)
- YOLO bbox가 번호판이 아닌 엠블럼/그릴 감지
- 차량 간 구분 부족

### 현재 대응 (plate_engine_pro.py)
- `PlateTracker` 클래스: IoU 기반 bbox 매칭 (threshold=0.30)
- TTL 프레임 만료 (30프레임 미감지 시 트랙 삭제)
- `consecutive_required`: 연속 N프레임 감지 시 표시

### 알려진 이슈
- 흰색/은색 번호판: V > 80 임계값에서 그림자/음영 포함 문제
- 한글 파일명: `cv2.imread` 인코딩 문제 → `os.listdir` + 매칭으로 우회

## 의존성
- Python 3.10+
- ultralytics (YOLO), opencv-python, numpy, Pillow
- paddleocr (OCR 단독, 한국어 특화)
- torch, torchvision (CRNN)
- tkinter (GUI)

## 실행 방법
```bash
# 정확도 테스트
python test_ocr_accuracy.py

# Ghost Detection 테스트
python test_ghost_detection.py

# GUI 실행
python plate_gui.py [video.mp4]

# 기본 영상 자동 실행
python plate_gui.py
```

## 변경 이력

| 날짜 | 작업 | 내용 |
|------|------|------|
| 2025-02-24 | 초기 커밋 | YOLO + OCR 기본 파이프라인 구축 |
| 2025-02-25 | OCR 엔진 업데이트 | CRNN 모델/학습 스크립트 추가, 18종 전처리 파이프라인, 앙상블 투표 |
| 2025-02-25 | 리포지토리 정리 | GUI 핵심 9개 파일만 유지 |
| 2025-02-26 | 한글 초성 교차검증 | HangulClassifier 추가 (정적 12/12 달성) |
| 2025-02-26 | OCR 엔진 고도화 | CRNN 모델/디버그 이미지 추가, plate_engine_pro.py 확장 |
| 2025-02-27 | Ghost Detection | PlateTracker IoU 기반 차량 추적, Ghost Detection 5/5 달성 |
| 2025-02-27 | 문서화/정리 | 불필요 파일 정리, README.md 재작성, CLAUDE.md 이력 추가 |
