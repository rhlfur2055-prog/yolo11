# SETUP — 설치 및 실행 가이드

> **목표:** 클론 → 가상환경 → 패키지 → 회귀 테스트 통과까지 **10분 이내**.
> 회귀 baseline: 정적 12장 **11/12 (91.7%)**.

---

## 1. 시스템 요구 사항

| 항목 | 요구 | 비고 |
|---|---|---|
| Python | **3.10 / 3.11 / 3.12** | 3.13 ❌ (`paddlepaddle` 미지원) |
| OS | Windows 11 ✅ 테스트됨 / Linux / macOS | 한글 폰트 경로만 차이 (§5 참조) |
| RAM | **16 GB 권장** (최소 8 GB) | PaddleOCR + CRNN + YOLO11x 동시 로드 |
| 디스크 | **5 GB 이상 여유** | 모델 ~160 MB + PaddleOCR 캐시 ~1 GB |
| GPU | 선택 (CUDA 11.8+) | CPU로도 동작 (이미지당 ~1.35초) |
| 외부 도구 | `git` 필수, `tesseract` 선택 | tesseract는 보조 OCR (없어도 동작) |

---

## 2. 빠른 시작 (3분)

```bash
# 1) 클론
git clone https://github.com/rhlfur2055-prog/yolo11.git
cd yolo11

# 2) 가상환경 생성/활성화
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell: .venv\Scripts\Activate.ps1)
# source .venv/bin/activate     # Linux / macOS

# 3) 의존성 설치 (3~5분 소요, PaddleOCR가 큼)
pip install --upgrade pip
pip install -r requirements.txt

# 4) 회귀 테스트 (모델·데이터 배치 후 실행)
python test_ocr_accuracy.py
#   기대 결과: 11/12 (91.7%) — §4 참조
```

> ⚠️ **3-4 사이 누락 단계가 있음** — `best.pt` / `plate_ocr_crnn.pth` / `22/` 폴더는 `.gitignore` 처리되어 **클론에 포함되지 않는다**. §3에서 배치 방법을 안내한다.

---

## 3. 모델 / 테스트 데이터 배치

`.gitignore`로 제외된 파일을 프로젝트 루트에 직접 두어야 한다.

### 3.1 필수 파일

| 파일 | 크기 | 위치 (프로젝트 루트 기준) | 출처 |
|---|---|---|---|
| `best.pt` | 114 MB | `./best.pt` | YOLO11x 번호판 fine-tuned (별도 전달, §3.4 참조) |
| `plate_ocr_crnn.pth` | 42 MB | `./plate_ocr_crnn.pth` | CRNN 한글 교차검증 (별도 전달, §3.4 참조) |
| `yolov8n.pt` | 6.5 MB | `./yolov8n.pt` | COCO fallback (선택, `best.pt` 부재 시 안전망) |
| `22/` 폴더 | 3.4 MB | `./22/*.png` | 회귀 테스트 12장 + 확장 6장 (총 18장) |
| `result_portfolio.mp4` | 18 MB | `./result_portfolio.mp4` | 데모 영상 (`docs/demo.gif` 추출 원본) |

**배치 후 확인:**
```bash
python -c "from pathlib import Path; \
  print('YOLO :', Path('best.pt').exists()); \
  print('CRNN :', Path('plate_ocr_crnn.pth').exists()); \
  print('22/  :', len(list(Path('22').glob('*.png'))), '장')"
# 기대: YOLO=True, CRNN=True, 22/=12장
```

### 3.2 자동 폴백 (best.pt 부재 시)

`best.pt`가 없으면 `PathConfig.find_best_model()`이 자동으로 다음 순서를 시도한다 ([config.py:138](../config.py#L138)):

1. `./best.pt`
2. `runs/detect/plate_korean_3k_v2/weights/best.pt`
3. `runs/detect/plate_korean_3k/weights/best.pt`
4. **HuggingFace 자동 다운로드** — `morsetechlab/yolov11-license-plate-detection` ([config.py:61](../config.py#L61))

> CRNN (`plate_ocr_crnn.pth`)에는 자동 폴백이 없다. **반드시 직접 배치**.

### 3.3 선택 — 실시간 영상

```bash
mkdir movie
# movie/hiway.mp4 등 테스트용 영상을 배치
python plate_gui.py movie/hiway.mp4
```

### 3.4 모델 / 데이터 입수 경로 (3가지 옵션)

§3.1의 파일들을 어떻게 확보할지에 대한 안내. 외부인 / 면접관 / 신규 합류자가 이 저장소만 받았을 때 따라갈 수 있는 경로를 명시한다.

#### 옵션 1 — 자체 호스팅 (TODO)

> **상태:** Google Drive / HuggingFace 링크 제공 예정.
> **연락:** 프로젝트 소유자에게 요청. 인계 시 다음 5개 파일을 함께 전달:
>
> - `best.pt` (114 MB)
> - `plate_ocr_crnn.pth` (42 MB)
> - `yolov8n.pt` (6.5 MB, 선택)
> - `22/` 디렉토리 (3.4 MB, 18장)
> - `result_portfolio.mp4` (18 MB, 데모 GIF 원본)

#### 옵션 2 — 자체 학습

CRNN 모델을 직접 학습해 `plate_ocr_crnn.pth`를 재생산하는 경로.

- 학습 스크립트 `train_plate_ocr.py`는 v3+ 이력에 있으나 **현재 리포지토리는 정리되어 미포함**.
- 학습 데이터셋(실제 132장 + 합성 20,647장 = 21,967 샘플)도 **별도 확보 필요**.
- 학습 환경: RTX 4060 (CUDA), 200 epoch, BiLSTM(2층) + CTC.
- YOLO `best.pt`는 별도 fine-tuning 셋업이 필요(라벨링된 번호판 박스 데이터셋).

> 면접/포트폴리오 용도라면 옵션 1을 권장. 옵션 2는 시간 투자가 크다(데이터셋만 수일~수주).

#### 옵션 3 — 유사 모델 대체 (코드 동작만 확인)

`best.pt` 없이도 파이프라인이 죽지 않도록 폴백 동작을 확인하고 싶을 때.

```bash
# Ultralytics 공식 YOLO11n (COCO 80클래스, ~6MB)
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
# → ~/.cache/.../yolo11n.pt에 자동 다운로드
```

| 항목 | 결과 |
|---|---|
| YOLO 박스 탐지 | ✅ 동작 (자동차/트럭/버스 등 COCO 클래스) |
| 번호판 특정 탐지 | ❌ 동작 안 함 (COCO에는 `license_plate` 클래스 없음) |
| PaddleOCR 인식 | ❌ 의미 없음 (번호판 ROI가 안 들어옴) |
| CRNN 교차검증 | ❌ 의미 없음 (입력 없음) |

> **결론:** `best.pt` 없이는 **plate-specific 인식 불가**. §3.2의 HuggingFace 자동 폴백(`morsetechlab/yolov11-license-plate-detection`)이 현실적인 차선책.

---

## 4. 검증 (회귀 테스트)

### 4.1 SRP 모듈 스모크 테스트 (~1초)

```bash
pip install pytest
pytest tests/test_modules_smoke.py -v
# preprocessor / validator / db 3개 모듈의 self-contained 검증
# YOLO·PaddleOCR·CRNN 로드 안 함 → 1초 내 끝남
```

### 4.2 정확도 회귀 테스트 (~16초)

```bash
python test_ocr_accuracy.py
```

**기대 출력:**
```
정확도 : 11/12 (91.7%)
평균 시간: 약 1,350 ms / 이미지
```

| 항목 | 임계값 |
|---|---|
| Baseline 통과 | `passed ≥ 11` |
| 알려진 실패 | `58두9599.png` 1건 (README 공개) |
| 다른 케이스 실패 | **즉시 회귀로 간주 — 머지 금지** |

**유용한 옵션:**
```bash
python test_ocr_accuracy.py --verbose          # 단계별 로그
python test_ocr_accuracy.py --json report.json # 실패 케이스 JSON 리포트
```

### 4.3 경량 통합 검증 (~30초, 선택)

```bash
python scripts/verify_integrations.py
# 영상 로드 + YOLO 5프레임 감지 + 서버 헬스체크 (OCR 미실행)
```

---

## 5. 실행 모드

### 5.1 GUI (실시간 영상)

```bash
python plate_gui.py                         # 기본 영상 자동 로드
python plate_gui.py movie/hiway.mp4         # 특정 영상
python plate_gui.py --youtube <URL>         # YouTube 영상
python plate_gui.py --no-pro-engine         # 경량 엔진(PlateEngineFast) 사용
```

### 5.2 REST API 서버

```bash
python plate_server.py --port 5000
# 또는 Windows: run_plate_server.bat / run_plate_server.ps1

# 헬스 체크
curl http://localhost:5000/health
```

### 5.3 단일 이미지 디버그

```python
import cv2
from plate_engine_pro import PlateEnginePro

engine = PlateEnginePro()
frame = cv2.imread("22/01나8060.png")
result = engine.process_frame(frame)
print(result)   # [{'plate': '01나8060', 'confidence': 0.92, 'bbox': (...)}]
```

---

## 6. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: paddle` | `paddleocr`만 설치, `paddlepaddle` 누락 | `pip install paddlepaddle>=3.0` |
| Python 3.13에서 설치 실패 | `paddlepaddle` 미지원 | Python 3.10–3.12 재설치 |
| `FileNotFoundError: best.pt` | 모델 미배치 | §3.1 배치 또는 §3.2 폴백 활성화 (인터넷 필요) |
| `FileNotFoundError: 22/` | 회귀 데이터 미배치 | §3.1 12장 배치 |
| 한글이 `?` 또는 깨짐 (Linux) | 한글 폰트 부재 | `sudo apt install fonts-nanum` 또는 `FONT_PATH` 환경변수 |
| PaddleOCR 첫 실행 매우 느림 | 모델 자동 다운로드 중 (~1 GB) | 1회만, `~/.paddleocr` 캐시 후 빨라짐 |
| `cv2.imread` 가 한글 파일명 반환 None | Windows 인코딩 이슈 | 이미 우회 처리됨 (`np.fromfile + cv2.imdecode`) |
| GPU 인식 안 됨 (CUDA) | CPU 휠 설치됨 | torch: <https://pytorch.org/get-started/locally/> / paddle: `pip install paddlepaddle-gpu` |
| `tesseract not found` 경고 | 보조 OCR 미설치 | 무시 가능 (메인 파이프라인 영향 없음). 설치 시: Windows 설치 후 `TESSERACT_CMD` 환경변수 |
| OCR 결과가 baseline보다 낮음 | PaddleOCR / torch 버전 차이 | `requirements.txt` 버전 핀 준수 |

### 6.1 환경변수 (선택)

| 변수 | 용도 | 예 |
|---|---|---|
| `PADDLE_MODEL_DIR` | PaddleOCR 모델 캐시 위치 강제 | `C:\paddle_cache` |
| `FONT_PATH` | 한글 폰트 강제 지정 | `/usr/share/fonts/.../NanumGothicBold.ttf` |
| `TESSERACT_CMD` | tesseract 실행 파일 경로 | `C:\Program Files\Tesseract-OCR\tesseract.exe` |
| `PLATE_CONSECUTIVE_FRAMES` | 같은 번호판 확정 프레임 수 | `1` (기본) |

---

## 7. 다음 단계

- 🏗️ 아키텍처 / 7단계 파이프라인 → [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- 📊 회귀 baseline 상세 / 알려진 실패 케이스 → [`README.md` § 성능](../README.md)
- 🧪 새 변경사항 검증 순서:
  ```bash
  pytest tests/test_modules_smoke.py && python test_ocr_accuracy.py
  ```
