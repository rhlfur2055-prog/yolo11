# 2026 업무 매뉴얼 — 파일 유지/삭제 완전 분류

**기준:** 현재 `C:\tool\yolo11`의 모든 파일/폴더
**원칙:** 애매하면 살린다 (conservative)
**상태:** 2026-05-27, commit 9f921eae 기준
**파이프라인:** 7단계 LPR (영상→YOLO→전처리→PaddleOCR→CRNN→투표→추적)

---

## 한눈에 보기

| 분류 | 개수 | 정의 |
|------|------|------|
| 필수 (Critical) | 11 | 이 중 하나라도 없으면 파이프라인 작동 불가 |
| 유지 권장 (Keep) | 22 | 포폴 가치 / 도달 경로 있음 / 방어적 fallback |
| 경계선 (Borderline) | 10 | 사용자 판단 필요 — 추가 정보 제공 |
| 삭제 권장 (Drop) | 6 | 다른 4개 세션 검증으로 100% dead 확인 |

**총 절감 가능 (만 적용 시): 약 2KB + 6개 항목** — 핵심은 정리가 아니라 **명확한 매뉴얼화**.

---

## 필수 파일 (절대 삭제 금지) — 11개

| 파일 | 역할 | 의존 관계 | 비고 |
|------|------|----------|------|
| `plate_gui.py` | GUI 진입점 (Tkinter, 실시간 루프) | `config`, `plate_engine_pro` import | 77KB |
| `plate_engine_pro.py` | OCR 엔진 본체 (단계 2~7) | `config`, `plate_recognition_4k` 함수들 import + `best.pt` + `plate_ocr_crnn.pth` 로드 | 176KB, 3212줄 (TASK F: God class, 리팩토링 후보) |
| `plate_recognition_4k.py` | 한글 교정 라이브러리 + 대체 엔진 | `config` import | 120KB, `--no-pro-engine` 시 PlateRecognizer 도달 |
| `config.py` | 전역 설정 (경로/임계값) | logging 만 | 16KB |
| `test_ocr_accuracy.py` | 회귀 테스트 (12장) | `plate_engine_pro` | **TASK B: 현재 11/12=91.7%** (#10 58두9599 실패) |
| `best.pt` | YOLO11x 번호판 탐지 모델 | config.py:35에서 자동 로드 | 114MB, **gitignored (로컬 전용)** |
| `plate_ocr_crnn.pth` | CRNN 한글 교차검증 모델 | plate_engine_pro 내부 로드 | 42MB |
| `requirements.txt` | 의존성 목록 | pip install -r | **TASK A: paddlepaddle 누락 — 추가 필요** |
| `README.md` | 포폴 진입점 (7단계 파이프라인 문서) | — | **TASK A: 5가지 수정 사항 존재 (아직 미적용)** |
| `.gitignore` | Git 제외 규칙 | — | `*.pt`, `*.mp4`, `uploads/` 등 보호 |
| `22/` | 테스트 이미지 18장 (Ground Truth = 파일명) | `test_ocr_accuracy.py`가 읽음 | **gitignored**, 12장 + 충86다6118 6장 |

---

## 유지 권장 — 22개 (포폴 가치 / 도달 경로 있음)

### 모델/데이터 fallback
| 파일 | 사유 |
|------|------|
| `yolov8n.pt` (6.5MB) | best.pt 누락 시 COCO fallback. **gitignored.** |
| `result_portfolio.mp4` (18.6MB) | 포폴 데모 영상. **gitignored.** |
| `plate_records.db` (28KB) | sqlite 런타임 DB. **gitignored.** |

### 보조 진입점 (CLI 옵션/별도 워크플로우)
| 파일 | 도달 경로 / 가치 |
|------|------------------|
| `youtube_helper.py` | `python plate_gui.py --youtube <URL>` 시 lazy import |
| `plate_server.py` (8.4KB) | FastAPI 서버 모드 (포폴: API 제공 역량) |
| `run_goldentime.py` (10KB) | 시뮬레이션 러너 (포폴: 시나리오 검증 역량) |
| `train_plate_ocr.py` (38KB) | CRNN 학습 스크립트 (포폴: ML 학습 파이프라인) |

### 테스트 (커버리지 표현)
| 파일 | 가치 |
|------|------|
| `test_ghost_detection.py` (12KB) | Ghost Detection 회귀 (README 5/5 PASS 근거) |
| `test_benchmark.py` (16KB) | 성능 측정 (포폴: 정량 평가) |
| `test_full_report.py` (7KB) | 종합 리포트 |
| `test_ttl_ghost_fix.py` (14KB) | TTL 버그 회귀 |
| `test_hiway_quick.py`, `test_hiway_video.py` | 영상 테스트 |
| `test_12.py` (2.6KB) | 빠른 12장 변형 테스트 |

### 벤치/분석 도구
| 파일 | 가치 |
|------|------|
| `bench_fps.py` (1.7KB) | FPS 측정 |
| `bench_full.py` (5.6KB) | 전체 벤치 |
| `bench_profile.py` (6.4KB) | 프로파일링 |
| `analyze_video.py` (4.8KB) | 영상 분석 유틸 |
| `build_dashboard.py` (28KB) | 대시보드 생성 (산출물은 삭제됐지만 스크립트는 보존) |

### 진입 스크립트 (TASK E: keep)
| 파일 | 사유 |
|------|------|
| `run_plate_server.bat` (417B) | 메인 데모 진입점, `%~dp0` 상대경로 |
| `run_plate_server.ps1` (725B) | PowerShell 변형, `$PSScriptRoot` 사용, robust |
| `run_test.bat` (8.9KB) | 환경 점검 6단계 (면접 시 환경검증 유용) |

### 폴더
| 폴더 | 사유 |
|------|------|
| `simulation/` (14 files, 255KB) | 시뮬레이션 프레임워크 (포폴: 시나리오/시뮬레이션 역량) |
| `scripts/` (13 files, 68KB) | **Jetson 엣지 배포 + AIHub 데이터셋 처리** (포폴 핵심) |
| `22_v5/` (18 files, 3.4MB) | 2줄 번호판 디버그 데이터셋 (충86다6118 6장 포함) |

---

## 경계선 — 10개 (사용자 판단 필요)

### 데이터셋 (대용량)
| 항목 | 정보 | 결정 포인트 |
|------|------|-------------|
| `dataset_3k/` | **546MB, 1,838 파일** | 학습 데이터 — GitHub gitignored일 가능성 있음. 로컬 디스크 부담. **포폴 GitHub에 안 올라가면 로컬 보관, 디스크 압박이면 삭제** |
| `train_2line/` | 12MB, 12 파일 | 2줄 번호판 학습용. 학습 안 돌릴 거면 불필요 |
| `dataset/` | 1KB, 1 파일 | 거의 빈 폴더. data.yaml 정도일 가능성 |

### 빈 폴더 (런타임 산출물용 placeholder)
| 폴더 | 처리 |
|------|------|
| `models/` (empty) | placeholder. 삭제해도 코드 자동 생성 가능. 보존 무해 |
| `uploads/` (empty) | 서버 업로드용. plate_server 실행 시 필요 |
| `plate_results_final/` (empty) | 런타임 출력용 |

### 워커 (TASK A: broken, 의존 모듈 없음)
| 파일 | 상태 |
|------|------|
| `cmd5_yolo_worker.py` (26KB) | TASK A 검증: `pipeline_common` 모듈 import (HEAD에 없음) → **import 시점에 깨짐**. 워커 분산 처리 컨셉 보존 vs broken 코드 정리 |
| `cmd6_ocr_worker.py` (40KB) | 같은 이유로 broken |

### 산출물/배포
| 파일 | 결정 포인트 |
|------|-------------|
| `test_result_20260527_090017.txt` (10KB) | TASK B 실행 결과. 보관해서 회귀 기준점 vs 매번 새로 생성 |
| `docker-compose.yml` (574B), `setup.sh` (2.9KB) | Docker 배포 helper. 사용 안 한다고 했지만 포폴 가치는 있음 |

---

## 삭제 권장 — 6개 (TASK E 4세션 검증 완료)

| 파일/폴더 | 크기 | 삭제 근거 |
|----------|------|----------|
| `image/` | 992KB, 3 파일 | `img1.png.png`, `img2.png.png`, `img3.png.png` — **이중 확장자, 의미없는 이름**. 어디서도 참조 없음 (grep "image/img" → 0건). 명백한 임시 파일 |
| `run_plate_server_live.bat` | 412B | TASK E: `cd /d C:\tools\yolo26` **하드코딩 절대경로** + `run_plate_server.bat`와 기능 중복 |
| `run_realtime_gui.bat` | 321B | TASK E: 첫 줄 `ㅁㅁ@echo off` **오타** + 절대경로 깨짐 + 중복 |
| `run_realtime_plate.bat` | 320B | TASK E: 대상 `realtime_plate.py` **파일 자체가 없음** (broken reference) |
| `실시간_GUI_localhost.bat` | 351B | TASK E: 절대경로 깨짐 + 한글 파일명 인코딩 이슈 + 중복 |
| `__pycache__/` | (auto) | Python 빌드 캐시. `.gitignore`에 이미 포함, 로컬에서도 안전 삭제 가능 |

**총 절감:** 약 1MB + 5개 .bat 파일 + 1개 폴더

---

## 별도 처리 필요 (TASK 결과 미적용)

| 항목 | 상태 | 조치 필요 |
|------|------|----------|
| **TASK A README 수정** | stranded 커밋 b39a9a7e가 yolo26-main 브랜치에 남음 | C:\tool\yolo11에 5가지 수정 재적용 후 origin/main 푸시 (postfilter_v2 줄 제거, CRNN 학습 데이터 산수 수정, NMS IoU 0.65 문구 교체, 70px 감산 문구 교체, paddlepaddle 추가) |
| **TASK B 11/12 실패** | #10 58두9599 PaddleOCR 한글 '두' 누락 → CRNN 폴백 미트리거 | (1) README "12/12 100%" 주장 수정 OR (2) 6자리 숫자-only 패턴을 [DIGIT-CRNN] 분기에 추가 |
| **TASK F 리팩토링** | plate_engine_pro.py 3212줄 God class, 9개 모듈로 분리 가능 | 단계적 분리 PR 계획 수립 (필수 아님, 가독성 개선) |

---

## 결정 가이드 (한 줄 요약)

1. **필수 11개:** 묻지도 따지지도 말고 보존
2. **유지 22개:** 포폴 가치 있음, 손대지 않음
3. **경계선 10개:** `dataset_3k/`(546MB) 디스크 부담 시 검토 / 워커 2개는 broken이지만 컨셉 보존 여부 결정 / 빈 폴더는 무해
4. **삭제 6개:** 안전하게 즉시 삭제 가능
5. **후속 작업 3개:** TASK A 푸시 / TASK B 실패 대응 / TASK F 리팩토링 계획

**다음 액션 추천 순서:**
1. 6개 삭제 + commit + push (5분)
2. TASK A README 수정 재적용 + push (10분) — 사실 검증된 README가 포폴의 첫인상
3. TASK B 실패 결정: "11/12 = 91.7%" 솔직히 README에 적기 vs CRNN 분기 보완
4. `dataset_3k/` 거취 결정
5. TASK F 리팩토링은 별도 작업으로 분리 (큰 작업)
