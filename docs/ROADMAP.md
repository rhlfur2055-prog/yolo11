# Roadmap — 후속 작업 + 미해결 조사 항목

리팩터링 진행 중 발견된 후속 작업과 회귀 조사 항목을 추적합니다.

---

## [v3.2.0 — Pending] src/ 패키지 레이아웃 도입 (CRNN lazy import 회귀 후속 조사)

### 배경
2026-05-27 세션에서 `src/yolo11_plate/` 패키지 레이아웃 전환을 시도했으나 **회귀 발생으로 revert**.

### Revert된 커밋

| 커밋 | 내용 |
|------|------|
| `4c51b685` | src/ 패키지 레이아웃 전환 — 루트 클린업 + 콘솔 진입점 + MIT License |
| `91ce2385` | 루트 .py 모듈 24개 제거 — src/ 패키지 레이아웃 이동 완료 |

복구 커밋:
- `3779c132` Revert "refactor: 루트 .py 모듈 24개 제거…"
- `afde8116` Revert "refactor: src/ 패키지 레이아웃 전환…"

### 회귀 증상

| 측정 | 변경 전 | 변경 후 | Δ |
|------|--------|--------|---|
| `test_ocr_accuracy.py` | 12/12 (100%) | **1/12 (8.3%)** | -92%p |
| smoke `pytest tests/` | 25/25 | 25/25 | 변화 없음 |
| import 검증 | OK | OK | 변화 없음 |
| 증상 | — | **한글 글자 보정이 silent fail** (CRNN 미동작) | |

PaddleOCR raw 결과는 정상 반환됐으나(한글 자리에 오인식 글자 들어옴), CRNN cross-check가 그 한글 자리를 보정하지 못함. PaddleOCR 환경 문제가 아니라 **refactor가 일으킨 회귀**.

### 가설 (조사 우선순위 순)

1. **CRNN 모델 경로 깨짐** — `plate_ocr_crnn.pth`가 gitignored 루트에 있음. `src/yolo11_plate/`에서 cwd 의존 상대 경로로 모델 로드 시 not found
   - 확인 명령: `grep -rn "plate_ocr_crnn.pth" --include="*.py"`
2. **PEP 562 lazy re-export 깨짐** — `plate_recognition_4k.py:__getattr__`가 `from .plate_recognizer import PlateRecognizer`로 변경되면서 import path 해석 시점 문제
3. **try/except silent fail** — CRNN 로드 실패가 조용히 삼켜져 호출만 빈 결과 반환
   - 확인 명령: `grep -rn "except.*pass" --include="*.py" | head -20`
4. **`__init__.py` 누락 또는 import 시 모듈 부분 초기화** — `src/yolo11_plate/__init__.py`가 `__version__`만 정의해 패키지 첫 import 시 의도된 side effect 누락

### 검증 게이트 (재시도 시 필수)

| 게이트 | 명령 | 기대값 |
|--------|------|--------|
| import | `python -c "from yolo11_plate.crnn_verifier import CrnnVerifier; v = CrnnVerifier('./plate_ocr_crnn.pth'); print('CRNN OK')"` | 모델 실제 로드 (예외 없음) |
| smoke | `pytest tests/test_modules_smoke.py` | 25/25 |
| **회귀** | `pytest tests/test_ocr_accuracy.py` | **12/12 (이게 핵심)** |
| 콘솔 진입점 | `plate-gui --help` | 정상 출력 |

**12/12 통과 못 하면 머지 금지.** smoke만 통과한 상태로 머지하면 안 됨 (이번 사고가 그 케이스).

### v3.2.0 마일스톤 구성

- [ ] 위 4개 가설 중 진짜 원인 식별
- [ ] 별도 브랜치 `investigate/src-layout-regression`에서 재시도
- [ ] `tests/test_ocr_accuracy.py`를 패키지 import 형식에 맞게 갱신
- [ ] CI 워크플로우에 12/12 게이트 추가 (모델 파일 mock 또는 실제 로드)
- [ ] LICENSE + 콘솔 진입점 + pyproject 설정은 보존 (별도 PR로 분리 가능)

### 분리 PR 후보 (회귀 없는 부분만 우선 머지 가능)

| 작업 | 회귀 위험 | 분리 머지 가능 |
|------|----------|---------------|
| LICENSE (MIT) 추가 | 0 | |
| `bench/`, `tools/` 디렉토리 분리 | 낮음 (외부에서 import 안 됨) | |
| `.bat` 스크립트 → `scripts/` 이동 | 0 | |
| README 셀링 포인트 갱신 | 0 | (이미 머지됨) |
| **`src/yolo11_plate/` 패키지** | **높음 (CRNN 회귀)** | — v3.2.0 |

---

## 기타 후속 작업

### CRNN 디지트 폴백 패턴 확장 (test #10 `58두9599`)

README의 "알려진 실패 케이스" 참조. PaddleOCR이 한글 `두`를 누락해 `589599`(6자리)만 반환 시 현재 CRNN 폴백은 4자리만 트리거. 6자리 케이스도 트리거하도록 패턴 확장.

### 외부 검증셋 확보 (YOLO recall/precision)

README "알려진 측정 공백" 박스 참조. 현재 `22/` 12장은 모두 close-up(area ratio 약 43%)이라 YOLO에 매우 쉬운 케이스. AIHub 한국 번호판 데이터셋 등 외부 검증셋으로 실제 도로 영상의 recall/precision 측정 필요.

### Test-time augmentation 도입

`docs/DATA_AUGMENTATION.md` 향후 개선 섹션 참조. 12장 × 5종 변형으로 60 케이스 평가하면 회귀 baseline 강도 향상.

---

## 관련 문서

- [`CHANGELOG.md`](../CHANGELOG.md) — 모든 머지된 변경 이력
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — 현재 아키텍처
- [`docs/CODE_TOUR.md`](CODE_TOUR.md) — 코드 위치 인덱스
- [`docs/DATA_AUGMENTATION.md`](DATA_AUGMENTATION.md) — 데이터 증강 전략
- [`docs/SETUP.md`](SETUP.md) — 설치/실행 가이드
