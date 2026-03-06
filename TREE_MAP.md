# YOLO26 프로젝트 트리맵

> 프로젝트 루트: `c:\tool\yolo26-main`

---

## 전체 디렉터리 트리

```
yolo26-main/
├── .claude/
│   ├── launch.json
│   ├── settings.local.json
│   └── skills/
│       ├── yolo26-dev/      SKILL.md   (개발 스킬)
│       ├── yolo26-test/     SKILL.md   (테스트 자동화)
│       ├── yolo26-simulation/ SKILL.md (시뮬레이션 시나리오)
│       ├── yolo26-evidence/ SKILL.md   (증거 패키지·대시보드)
│       └── yolo26-ocr/      SKILL.md   (OCR 엔진 선택·튜닝)
│
├── .git/
│
├── 22/                          # Regression 테스트용 번호판 이미지 (12장, GT=파일명)
│   ├── *.png                    # 예: 경기76바7789.png, 01나8060.png
│   └── _zip_extracted/
│
├── crops/                       # 크롭 결과 등
├── crops_test/
├── dataset/                     # 학습/평가 데이터
├── dataset_3k/
│   ├── data.yaml
│   ├── images/
│   ├── labels/
│   ├── run_train.py
│   └── run_train_v2.py
├── debug_crops/
├── evidence_output/             # 채증 결과 (골든타임 + SafePlate)
│   ├── evidence_YYYYMMDD_HHMMSS_번호판/   # 골든타임 패키지
│   │   ├── video.mp4
│   │   ├── plates.json
│   │   ├── report.txt
│   │   └── screenshots/
│   └── safeplate_YYYYMMDD_HHMMSS_번호판/ # SafePlate 패키지
│       ├── video.mp4
│       ├── plates.json
│       ├── report.txt
│       └── screenshots/
├── image/
├── models/
├── movie/                       # 테스트 영상
│   ├── hiway.mp4                # 실시간 테스트용
│   ├── emergency_test.mp4
│   └── emergency_sample_*.jpg
├── plate_results_final/
├── plate_results_highway/
├── plate_results_v3/
├── runs/
├── scripts/
│   ├── run_headless_plate_test.py
│   ├── run_retest_pro.py
│   ├── list_videos.py
│   ├── bench_accuracy.py
│   ├── verify_integrations.py
│   ├── aihub_highway_check.py
│   └── setup_aihub_dataset.ps1
├── server_results/
├── simulation/                  # 시뮬레이션 프레임워크 (EventBus 기반)
│   ├── __init__.py
│   ├── simulation_framework.py  # EventBus, SimulationFramework
│   ├── goldentime/              # 골든타임 2.0 (긴급차량 방해)
│   │   ├── __init__.py
│   │   ├── siren_trigger.py
│   │   ├── distance_checker.py
│   │   ├── plate_evidence.py
│   │   └── evidence_export.py
│   └── safeplate/               # SafePlate 4K (물피도주)
│       ├── __init__.py
│       ├── shock_simulator.py   # ShockSimulator
│       ├── departure_detector.py # DepartureDetector
│       ├── evidence_package.py   # EvidencePackage
│       ├── night_enhancer.py
│       └── run_safeplate.py     # 통합 실행 (python -m simulation.safeplate.run_safeplate)
├── temp_youtube/
├── test_results/
├── uploads/
└── __pycache__/
```

---

## 루트 파일 (핵심)

### 실행·엔트리
| 파일 | 역할 |
|------|------|
| `plate_gui.py` | Tkinter GUI + 실시간 영상 (수정 금지) |
| `run_goldentime.py` | 골든타임 2.0 통합 실행 |
| `build_dashboard.py` | evidence_output 스캔 → evidence_dashboard.html 생성 |

### 엔진·파이프라인 (수정 주의/금지)
| 파일 | 역할 |
|------|------|
| `plate_engine_pro.py` | 메인 OCR 엔진 (YOLO+OCR+추적, 수정 금지) |
| `pipeline_common.py` | 파이프라인 공통 |
| `plate_ocr_postfilter_v2.py` | OCR 후처리 |
| `plate_recognition_4k.py` | 한글 교정 라이브러리 (읽기 전용) |
| `config.py` | 설정 |

### 워커·유틸
| 파일 | 역할 |
|------|------|
| `cmd5_yolo_worker.py` | YOLO 워커 |
| `cmd6_ocr_worker.py` | OCR 큐 워커 |
| `youtube_helper.py` | YouTube 영상 다운로드 (수정 금지) |

### 테스트
| 파일 | 역할 |
|------|------|
| `test_ocr_accuracy.py` | 12장 Regression (필수 실행) |
| `test_ghost_detection.py` | Ghost Detection 테스트 |
| `test_goldentime_*.py` | 골든타임 시나리오 테스트 |
| `test_safeplate_australia.py` | SafePlate 테스트 |
| `test_pipeline*.py`, `test_recognition.py`, `test_benchmark.py`, `test_realtime_stability.py` | 기타 테스트 |

### 학습·파이프라인 스크립트
| 파일 | 역할 |
|------|------|
| `train_plate_ocr.py` | CRNN 학습 (수정 금지) |
| `pipeline_stage1.py`, `pipeline_stage2.py` | 스테이지 파이프라인 |
| `extract_pipeline.py`, `auto_eval.py`, `analyze_project.py` | 추출/평가/분석 |

### 디버그·기타
| 파일 | 역할 |
|------|------|
| `debug_*.py`, `debug_*.jpg` | 디버그용 |
| `crop_cctv_video.py`, `patch_marearts_style.py`, `two_stage_test.py` | 유틸/패치 |

### 모델 (수정 금지)
| 파일 | 용도 |
|------|------|
| `best.pt` | YOLO11x 번호판 전용 (mAP@50=98.4%) |
| `yolo11x_plate.pt` | YOLO11x 번호판 |
| `plate_ocr_crnn.pth` | CRNN OCR (42MB) |
| `yolo26n.pt`, `yolo11n.pt`, `yolov8n.pt` | fallback |

### 문서·설정
| 파일 | 역할 |
|------|------|
| `CLAUDE.md` | 프로젝트 규칙 |
| `README.md` | 프로젝트 소개 |
| `work_status.md` | 작업 현황 |
| `requirements.txt` | 의존성 |

### 출력·대시보드
| 파일 | 역할 |
|------|------|
| `evidence_dashboard.html` | build_dashboard.py 생성 결과 |
| `_dashboard_data.json` | 대시보드용 데이터 |
| `realtime_index.html`, `roadmap.html` | 기타 HTML |
| `plate_records.db` | DB |
| `*.bat`, `*.ps1` | 실행 배치/스크립트 |

---

## simulation/ 상세

```
simulation/
├── simulation_framework.py   # EventBus, SimulationFramework (오버레이 등록·프레임 루프)
├── goldentime/               # 사이렌 감지 → 거리 위반 → 증거 패키지
│   ├── siren_trigger.py      # SIREN_DETECTED / SIREN_ENDED
│   ├── distance_checker.py  # DISTANCE_VIOLATION
│   ├── plate_evidence.py    # EVIDENCE_STARTED / EVIDENCE_COMPLETE
│   └── evidence_export.py   # evidence_YYYYMMDD_HHMMSS_번호판/ 생성
└── safeplate/                # 충격 감지 → 이탈 차량 → 증거 패키지
    ├── shock_simulator.py   # SHOCK_DETECTED / SHOCK_ENDED, 프레임 버퍼
    ├── departure_detector.py # DEPARTURE_DETECTED (IoU 추적, 경계 이탈)
    ├── evidence_package.py  # safeplate_YYYYMMDD_HHMMSS_번호판/ 생성
    ├── night_enhancer.py    # 야간 전처리
    └── run_safeplate.py     # CLI: --video, --headless, --shock-at 등
```

---

## 실행 요약

```bash
# Regression (필수)
python test_ocr_accuracy.py

# GUI
python plate_gui.py
python plate_gui.py movie/hiway.mp4

# 골든타임
python run_goldentime.py --video movie/hiway.mp4

# SafePlate
python -m simulation.safeplate.run_safeplate --video movie/hiway.mp4

# 대시보드
python build_dashboard.py   # → evidence_dashboard.html
```

---

*트리맵 생성: 프로젝트 현재 파일 기준.*
