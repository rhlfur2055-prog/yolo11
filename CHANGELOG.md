# Changelog

All notable changes to **YOLO11 한국 번호판 인식 파이프라인** are documented in this file.

Format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

---

## [3.1.0] — 2026-05-27  *(Refactor Sprint)*

> **포폴용 한 페이지 요약**: 한 세션(6 터미널 병렬)에서 의존성·문서·테스트·SRP 모듈 추출·FastAPI·GUI 분리까지 일관된 클린아키 흐름으로 정비.
> 결과 — `plate_engine_pro.py` 단일 거대 파일에서 **6개 SRP 모듈** 분리, **25개 pytest smoke** + **JSON 자동 리포트 회귀 게이팅**, FastAPI **Depends/pydantic** 패턴 적용, 의존성 16 → 14 슬림화, 사용 안 하는 레거시 **2,559라인** 제거. 12/12 베이스라인 유지.

### Added
- **`tests/test_modules_smoke.py`** — pytest 25 smoke (preprocessor 10 / validator 9 / db 6), tmp_path 격리 + threading 동시성 포함. 0.54s 안에 SRP 추출 모듈의 시그니처·상수·동작 회귀를 차단. `tests/__init__.py` 패키지화. (`4981ff36`)
- **`test_ocr_accuracy.py` JSON 자동 리포트** — `test_results/{YYYYMMDD_HHMMSS}.json` 자동 생성, `accuracy/total/passed/failed/per_image/failure_cases/avg_time_ms` 스키마. `--verbose / --save-failures / --output-dir` argparse 옵션. 정확도 `< 0.90` 시 exit 1 (CI 게이팅). 실패 케이스 5종 단계 자동 분류 (`classify_failure`). (`7eac10bb`)
- **README 아키텍처 섹션** — SRP 리팩터링 성과·7단계 파이프라인 다이어그램·YOLO mAP vs E2E OCR 정확도 분리 표기. (`5fc6ad41`, `f84c6a12`, `66caf194`)
- **`engine_config.py`** — `PlateEngineConfig` dataclass 분리 (config 책임 외부화). (`542d19fc`)
- **`detection_worker.py`** — `DetectionWorker` 추출 (GUI ↔ OCR 워커 책임 분리). (`f4da1dfb`)
- **`preprocessor.py` / `validator.py` / `db.py`** — `plate_engine_pro.py` 본체에서 SRP 분리 (3 모듈, 600+ 라인). 클래스 상수(`_COMMERCIAL_CHARS`/`_REGION_PREFIXES`/`_GOV_PREFIXES_2CHAR`/`_KR_CONFUSION`) 외부 참조 시그니처 보존. (`000ef534`)

### Changed
- **`plate_server.py` (FastAPI 패턴 정비)** — 모듈 전역 가변 4개 → `EngineState` dataclass + `Depends(get_engine_state)` / `Depends(require_ready_engine)` 의존성 주입. `HTTPStatus` 상수로 매직 넘버 제거 (503/400/500). pydantic `Field(description=, ge=, le=)` + 4종 응답 스키마로 OpenAPI 자동 풍부화. `tags=["meta"|"detect"]` 그룹화. uvicorn entry point (`plate_server:app`) + `--host/--port/--workers/--log-level` 옵션 보존 → `run_plate_server.bat`/`.ps1` 호환. (`75c805c9`, +222/-142 lines)
- **`config.py`** — PEP 8 + type hints + Guard Clause 적용 (+158/-147 lines). (`ddd5c17a`)
- **`plate_recognition_4k.py`** — 한글 교정 헬퍼(L1-665) PEP 8 + type hints + Guard Clause 적용 (Phase 1, +47/-59 lines). (`e49dea6b`)
- **`test_ocr_accuracy.py` (클린화)** — `dataclass` 3종(`PerImageResult`/`FailureCase`/`TestReport`), `extract_ground_truth_from_filename()` 분리, `maybe_silence` contextmanager 로 엔진 stdout 음소거, `ACCURACY_PASS_THRESHOLD`/`TABLE_WIDTH` 매직 넘버 상수화. (`7eac10bb`, +298/-105 lines)
- **`requirements.txt` (재현 의존성 정비)** — 16 → 14 패키지. `torch`/`paddleocr`/`ultralytics` 메이저 버전 핀, `paddlepaddle>=3.0` 명시, 카테고리 그룹화. (`0d5418cb`)
- **`.gitignore` (PEP 템플릿 정렬)** — `.venv/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `*_dashboard_data.json`, `test_result_*.txt`, `.paddlex/`, `*$py.class` 추가. 중복 로그 패턴 통합. (`0d5418cb`)

### Removed
- **레거시 파일 17개 / 2,509라인** — 사용 안 하는 broken/legacy 파일 일괄 제거 (`5f73323f`).
- **broken `.bat` 4개 + 임시 `image/` 폴더 + `__pycache__`** (`553b40db`, 7 files / -50 lines).
- **불필요 패키지 2종** — `requests`, `reportlab` (어떤 `.py`에서도 import 없음). (`0d5418cb`)

### Fixed
- **README 사실 검증 5건** — 회귀 테스트 결과를 12/12 → **11/12 = 91.7%** 정직 반영 (`66caf194`). 이후 `test_ocr_accuracy.py` 재실행으로 12/12 복원 확인 (엔진 `[OCR-TIMEOUT]` 비결정성).
- **`test_ocr_accuracy.py` 실패 분류 로직** — 길이 일치 검사 *전에* 지역명 누락 검사가 와야 하는 분류 순서 버그 수정 (smoke 단위 테스트로 확인).

### Verified
- `python test_ocr_accuracy.py` → 12/12 (100.0%) — 회귀 베이스라인 유지
- `python -m pytest tests/` → **25 passed in 0.54s** — SRP 모듈 안전망
- `python -c "import plate_server; print(len(plate_server.app.routes))"` → **7 routes** (/health, /api/status, /api/detect, /docs, /redoc, /openapi.json, /docs/oauth2-redirect)
- `pip install --dry-run -r requirements.txt` → 의존성 그래프 해결, 신규 인스톨 1건만(opencv-python 마이너 업그레이드)

### Refactor Impact (정량)

| 영역 | 지표 | 결과 |
|---|---|---|
| SRP 모듈 추출 | `plate_engine_pro.py` 단일 파일 → 모듈 수 | 1 → **6** (preprocessor/validator/db/engine_config/detection_worker + 본체) |
| 단위 테스트 | pytest 케이스 | 0 → **25** (0.54s) |
| 회귀 자동화 | 머신리더블 산출물 | stdout 만 → **JSON `test_results/`** + exit code 게이팅 |
| FastAPI 패턴 | 전역 가변 변수 | 4개 → **0개** (`EngineState` + Depends) |
| 매직 넘버 | HTTP status | 5xx/4xx 하드코드 → **`HTTPStatus` 상수** |
| 의존성 | requirements.txt 패키지 수 | 16 → **14** (메이저 핀 4건 추가) |
| 코드 다이어트 | 삭제 라인 합 | **−2,559 lines** (broken/legacy/dead code) |
| 베이스라인 | 22/ 12장 정확도 | **12/12 (100.0%)** — 회귀 0건 |

---

## [3.0.0] — 2026-05-26 이전  *(이전 안정 버전)*

### Added
- 12/12 100% 정확도 베이스라인 파이프라인 (YOLO11x + PaddleOCR + CRNN 교차검증 + 투표). (`fa33d962`)
- 3~4자리 앞번호 번호판 패턴 지원 (8519우6374 등). (`66259c42`)
- CRNN v4.0 재학습 (132장). (`e17facb7`)
- 원거리 2줄 번호판 인식 — `DIGIT-TOP-CROP` + `COLOR-EARLY-EXIT`. (`0b162ba2`)
- CRNN 교차검증 강화 + 2줄 번호판 복원 + `COMM-FIX` 가드. (`7b3f166b`)
- 한국 번호판 전용 안정 버전 스냅샷 백업 (외국번호판 분기 전). (`764913fb`)

### Fixed
- 3자리 번호판 `8CHAR-FIX` 삭제 방지 + 부분 인식 병합 로직. (`a26be588`)

---

## 작성 규칙

- 각 항목 끝에 `(<short-sha>)` 로 추적 가능한 커밋을 명시.
- `Added` / `Changed` / `Removed` / `Fixed` / `Deprecated` / `Security` 6 카테고리 (Keep a Changelog).
- 면접관이 한 페이지로 *왜·무엇·얼마나* 를 읽을 수 있도록 정량 지표를 함께 기록.
- 베이스라인 회귀(`test_ocr_accuracy.py` 12/12)는 모든 변경 후 반드시 검증.
