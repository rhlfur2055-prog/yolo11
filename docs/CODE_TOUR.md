# CODE_TOUR — 10분 코드베이스 산책 (Code Index Tour)

> **이 문서는 코드베이스를 10분 안에 한 바퀴 돌고 싶은 분을 위한 것입니다.**
> [`ARCHITECTURE.md`](ARCHITECTURE.md)가 전체 다이어그램(개념)이라면, 이 파일은 **"어디서 무엇이 일어나는가"를 코드 위치로 짚는 인덱스**입니다.
> 각 섹션은 `파일:라인` 형태로 적혀 있어 IDE에서 바로 점프할 수 있습니다.

---

## 0. 누구를 위한 문서인가 (Audience)

| 독자 | 이 문서에서 얻는 것 |
|---|---|
| 신입 엔지니어 | 첫날 10분 안에 7단계 파이프라인 코드 위치 전부 파악 |
| 면접관 / 코드 리뷰어 | 5분 안에 "이 코드베이스가 뭘 하는지" 큰 그림 + 진입점 |
| PR 리뷰 시작자 | 변경된 단계가 7단계 중 어디인지 즉시 매핑 |

---

## 1. 진입점 3가지 (Entry Points)

| 명령 | 파일 | 진입 라인 | 역할 |
|---|---|---|---|
| `plate-gui` 또는 `python -m yolo11_plate.plate_gui` | [`src/yolo11_plate/plate_gui.py`](../src/yolo11_plate/plate_gui.py) | `plate_gui.py:1109` (`def main`) | 실시간 영상 GUI (Tkinter). CLI argparse는 같은 함수 안 |
| `plate-server` 또는 `python -m yolo11_plate.plate_server` | [`src/yolo11_plate/plate_server.py`](../src/yolo11_plate/plate_server.py) | `plate_server.py:152` (`app = FastAPI(...)`) / `plate_server.py:291` (`def main`) | REST API 서버 (FastAPI + uvicorn) |
| `pytest tests/test_ocr_accuracy.py` | [`tests/test_ocr_accuracy.py`](../tests/test_ocr_accuracy.py) | `test_ocr_accuracy.py:302` (`def main`) | 정적 12장 회귀 테스트 — baseline **11/12 (91.7%)** |

```bash
# GUI 모드 (실시간 영상 / Real-time video)
plate-gui movie/hiway.mp4
# 또는: python -m yolo11_plate.plate_gui movie/hiway.mp4

# 서버 모드 (REST API)
plate-server --port 5000
# 또는: python -m yolo11_plate.plate_server --port 5000

# 회귀 모드 (regression)
pytest tests/ -v
```

---

## 2. 7단계 파이프라인 코드 투어 (Pipeline Code Tour)

[`ARCHITECTURE.md`](ARCHITECTURE.md)의 mermaid 다이어그램과 1:1 매핑됩니다.

### 단계 1 — 영상 입력 (Frame Capture)

| 위치 | 역할 |
|---|---|
| [`src/yolo11_plate/plate_gui.py:494`](../src/yolo11_plate/plate_gui.py) `class VideoReader` | OpenCV `VideoCapture` 래퍼 — 별도 스레드에서 프레임을 읽어 큐에 넣음 |
| [`src/yolo11_plate/detection_worker.py:175`](../src/yolo11_plate/detection_worker.py) `def _process_loop` | 큐에서 최신 프레임만 꺼내서 (`_drain_to_latest_frame`) 추론으로 넘김 — 실시간 응답성을 위해 오래된 프레임은 버림 |
| [`src/yolo11_plate/plate_gui.py:592`](../src/yolo11_plate/plate_gui.py) `class PlateGUIApp` | Tkinter 루트. `main()`에서 인스턴스화 → `mainloop()` |

**산출물:** `np.ndarray (H, W, 3) uint8 BGR`

### 단계 2 — YOLO 탐지 (YOLO Detection)

| 위치 | 역할 |
|---|---|
| [`src/yolo11_plate/plate_engine_pro.py:1291`](../src/yolo11_plate/plate_engine_pro.py) `def process_frame` | **2-Stage 파이프라인 진입점.** Stage1(차량)→Stage2(번호판) |
| [`src/yolo11_plate/plate_engine_pro.py:1334`](../src/yolo11_plate/plate_engine_pro.py) Stage 1 (vehicle) | `self.model_vehicle` (`yolov8n.pt`, COCO classes 2/5/7) |
| [`src/yolo11_plate/plate_engine_pro.py:1376`](../src/yolo11_plate/plate_engine_pro.py) Stage 2 (plate) | `self.model` (`best.pt`, fine-tuned 번호판) |
| [`src/yolo11_plate/plate_engine_pro.py:1267`](../src/yolo11_plate/plate_engine_pro.py) `def detect_only` | bbox만 필요한 경우의 경량 진입점 |

**산출물:** `list[dict{bbox: (x1,y1,x2,y2), conf: float}]`

### 단계 3 — ROI 크롭 + 18종 전처리 (ROI Crop + 18-variant Preprocessing)

| 위치 | 역할 |
|---|---|
| [`src/yolo11_plate/preprocessor.py:59`](../src/yolo11_plate/preprocessor.py) `class ImagePreprocessor` | 18종 변형 메서드 모음 (정적 메서드, stateless) |
| [`src/yolo11_plate/preprocessor.py:69`](../src/yolo11_plate/preprocessor.py) `gray_threshold` / `clahe` / `sharpen` / `deblur` … | 단일 책임: 한 변형 = 한 메서드 |
| [`src/yolo11_plate/preprocessor.py:213`](../src/yolo11_plate/preprocessor.py) `green_plate` / `yellow_plate` | 색상별 번호판(영업용/노란판) 특화 |

**관찰:** `preprocessor.py`는 프로젝트 내부 모듈을 전혀 import하지 않음 — **self-contained / 재사용 가능**.
**산출물:** `dict[str, np.ndarray]` (18장 변형 이미지)

### 단계 4 — PaddleOCR 추론 (Multi-variant OCR)

| 위치 | 역할 |
|---|---|
| [`src/yolo11_plate/plate_engine_pro.py:580`](../src/yolo11_plate/plate_engine_pro.py) `def _ocr_plate_roi` | ROI에서 OCR — 다중 프레임 합성 / 업스케일 / 샤프닝 처리 |
| [`src/yolo11_plate/plate_engine_pro.py:2421`](../src/yolo11_plate/plate_engine_pro.py) `def _run_ocr` | PaddleOCR 엔진 호출 + 결과 파싱. `conf ≥ 0.40` 필터 |

**산출물:** `list[tuple[str, float]]` (텍스트 후보 + 신뢰도)

### 단계 5 — CRNN 한글 교차검증 (Korean Hangul Cross-Verification)

| 위치 | 역할 |
|---|---|
| [`src/yolo11_plate/plate_engine_pro.py:2253`](../src/yolo11_plate/plate_engine_pro.py) `def _verify_korean_with_crnn` | **PaddleOCR 숫자 + CRNN 한글 = 최적 조합.** CRNN 숫자가 PaddleOCR과 크게 다르면 교정 거부 (과적합 방어) |
| [`src/yolo11_plate/plate_recognition_4k.py:282`](../src/yolo11_plate/plate_recognition_4k.py) `def correct_ocr_hangul` | 한글 자모 교정 함수 (순수 함수) |
| [`src/yolo11_plate/plate_recognition_4k.py:349`](../src/yolo11_plate/plate_recognition_4k.py) `_HANGUL_PLATE_CORRECTION` | 잘못 읽힌 한글 → 올바른 한글 dict 테이블 |

**산출물:** `str` (교정된 plate)

### 단계 6 — 투표 + 형식 검증 (Voting + Format Validation)

| 위치 | 역할 |
|---|---|
| [`src/yolo11_plate/plate_engine_pro.py:1127`](../src/yolo11_plate/plate_engine_pro.py) `def _deduplicate_results` | 자릿수별 분해 투표 (digit-position majority vote) |
| [`src/yolo11_plate/validator.py:27`](../src/yolo11_plate/validator.py) `class PlateValidator` | 한국 번호판 형식·자모 검증 + 패턴 복원 |
| [`src/yolo11_plate/validator.py:99`](../src/yolo11_plate/validator.py) `def validate` | `(passed: bool, normalized_text: str)` 반환 — reject 가능 |

**산출물:** `str` (확정 plate) 또는 reject

### 단계 7 — 추적 + 출력 (Tracking + Display)

| 위치 | 역할 |
|---|---|
| [`src/yolo11_plate/plate_gui.py:118`](../src/yolo11_plate/plate_gui.py) `class PlateTracker` | IoU 매칭 + TTL 기반 트래커 (`max_ttl=8`) |
| [`src/yolo11_plate/plate_engine_pro.py:281`](../src/yolo11_plate/plate_engine_pro.py) `def _make_track_key` | bbox → track key 매핑 |
| [`src/yolo11_plate/plate_engine_pro.py:297`](../src/yolo11_plate/plate_engine_pro.py) `def _should_skip_ocr` | 같은 트랙은 OCR 재실행 안 함 (FPS 최적화) |
| [`src/yolo11_plate/plate_engine_pro.py:356`](../src/yolo11_plate/plate_engine_pro.py) `def _update_ocr_cache` | grace=3 / frames_absent로 ghost 방지 |
| [`src/yolo11_plate/plate_gui.py:592`](../src/yolo11_plate/plate_gui.py) `class PlateGUIApp` | Tkinter GUI 렌더 (`video_label` = 영상 / 우측 패널 = 결과 리스트) |
| [`src/yolo11_plate/db.py:62`](../src/yolo11_plate/db.py) `def record_plate` | 확정된 plate → SQLite 기록 |

**산출물:** `dict{plate, confidence, bbox, track_id, ttl, grace}` + GUI 표시 + SQLite 행

---

## 3. FAQ — 자주 묻는 질문 (Frequently Asked Questions)

### Q1. 한글 오인식 어떻게 교정해요? (How to fix Hangul misreads?)

- 함수: [`src/yolo11_plate/plate_recognition_4k.py:282`](../src/yolo11_plate/plate_recognition_4k.py) `def correct_ocr_hangul`
- 데이터: [`src/yolo11_plate/plate_recognition_4k.py:349`](../src/yolo11_plate/plate_recognition_4k.py) `_HANGUL_PLATE_CORRECTION` dict
  - 자주 헷갈리는 자모/매핑(예: `잘 → 차`, `밤 → 바`)을 여기에 추가하면 됩니다.

### Q2. OCR 임계값(threshold) 어디서 바꿔요?

- 클래스: [`src/yolo11_plate/config.py:252`](../src/yolo11_plate/config.py) `class OCRConfig`
- 함께 보면 좋은 것: [`src/yolo11_plate/config.py:187`](../src/yolo11_plate/config.py) `class ThresholdConfig` (탐지/ROI 임계값)

### Q3. YOLO 모델(`best.pt`) 교체는?

1. 새 `best.pt`를 프로젝트 루트에 배치
2. (선택) 다른 경로 사용 시 [`src/yolo11_plate/config.py:39`](../src/yolo11_plate/config.py) `class PathConfig`의 `find_best_model()`이 자동 탐색 — 폴백 순서는 [`docs/SETUP.md` §3.2](SETUP.md) 참조

### Q4. 새 전처리 방법 추가는?

1. [`src/yolo11_plate/preprocessor.py:59`](../src/yolo11_plate/preprocessor.py) `class ImagePreprocessor`에 `@staticmethod` 추가
2. [`src/yolo11_plate/plate_engine_pro.py:580`](../src/yolo11_plate/plate_engine_pro.py) `_ocr_plate_roi` (혹은 변형 dict 생성부)에서 호출 등록
3. `python test_ocr_accuracy.py`로 회귀 확인 — **반드시 `≥ 11/12`**

### Q5. DB 스키마 수정은?

- 위치: [`src/yolo11_plate/db.py:29`](../src/yolo11_plate/db.py) `def _create_tables`
- 두 테이블: `plate_records`(기록), `alert_list`(수배차량)
- 참고: 스펙 표기는 `_init_db`였지만 실제 함수명은 `_create_tables`입니다.

### Q6. GUI에 새 위젯 추가는?

- 위치: [`src/yolo11_plate/plate_gui.py:593`](../src/yolo11_plate/plate_gui.py) `class PlateGUIApp.__init__`
- 영상 영역: `self.video_label` ([`src/yolo11_plate/plate_gui.py:707`](../src/yolo11_plate/plate_gui.py))
- 새 패널 / 버튼은 같은 `__init__` 안에서 `tk.Frame`/`tk.Label` 등을 추가

---

## 4. 공통 함정 (Common Gotchas)

| # | 함정 | 회피 |
|---|---|---|
| 1 | `paddleocr` 단독 설치 → `ModuleNotFoundError: paddle` | `pip install paddlepaddle>=3.0` 필수 (`paddleocr`는 백엔드 미포함) |
| 2 | `best.pt` / `plate_ocr_crnn.pth`가 클론에 없음 | 둘 다 `.gitignore` 처리됨 — 별도 다운로드 필요. 자세한 절차는 [`docs/SETUP.md` §3`](SETUP.md) |
| 3 | `test_ocr_accuracy.py` 결과가 **11/12 ↔ 12/12** 사이에서 흔들림 | OCR 타임아웃에 의한 비결정성. baseline은 `11/12` 고정 — 다른 케이스가 깨지면 회귀로 간주 |
| 4 | Linux에서 `import tkinter` 실패 | `sudo apt install python3-tk` (Windows는 Python 표준 포함) |

---

## 5. 다음 단계 학습 자료 (Next Steps)

| 목적 | 문서 |
|---|---|
| 7단계 전체 그림 + 의존성 그래프 | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) |
| 설치 / 모델 배치 / 트러블슈팅 | [`docs/SETUP.md`](SETUP.md) |
| 변경 이력 / 리팩터링 여정 | [`../CHANGELOG.md`](../CHANGELOG.md) |
| 외부 시각 (소개·성능·라이선스) | [`../README.md`](../README.md) |

> **추천 학습 순서:** README (외부 시각) → CODE_TOUR (코드 위치 인덱스) → ARCHITECTURE (다이어그램 / 깊은 이해) → SETUP (직접 실행).
