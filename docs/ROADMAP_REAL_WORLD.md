# Phase 1: 실차 탑재 로드맵 (Road-Legal)

**프로젝트**: [yolo26](https://github.com/rhlfur2055-prog/yolo26) — 한국 차량 번호판 OCR (YOLO11x + PaddleOCR) + GoldenTime + SafePlate  
**목표**: Jetson Orin Nano 8GB + dashcam 실차 테스트를 1주일 단위로 실행·반복하여 road-legal 검증.

---

## 0. Phase 1 개요

| 항목 | 내용 |
|------|------|
| 기간 | 1주 (Day 1~7) |
| 하드웨어 | Jetson Orin Nano 8GB, dashcam (1080p 15fps) |
| 스택 | YOLO11x TensorRT FP16, PaddleOCR 단독 (EasyOCR 제거), OpenCV headless |
| 산출물 | 로그/evidence 자동 저장, 메트릭 CSV·JSON, 안전 체크리스트, 일일 iterate (Issue/PR) |

---

## 1. Jetson Orin Nano 8GB 권장 설정

### 1.1 환경 요약

| 항목 | 권장 값 | 비고 |
|------|---------|------|
| JetPack | 6.0 / 6.1 / 6.2 | L4T R36.x, Ubuntu 22.04 |
| Python | 3.10 | JetPack 6 기본 |
| 입력 | 1080p @ 15fps | GStreamer / V4L2 / RTSP |
| YOLO 입력 해상도 | 640×640 (필요 시 960) | 1080p 크롭·리사이즈 후 |
| YOLO 포맷 | TensorRT FP16 (`.engine`) | inference 최적화 |
| PaddleOCR | CPU 또는 GPU 1 stream | 8GB 고려, `use_gpu=False` fallback |
| 메모리·전력 | nvpmodel 15W, FP16, headless, GStreamer | tegrastats로 모니터링 |

### 1.2 YOLO 최적화 코드

```python
# 1) TensorRT export (1회, 호스트 또는 Jetson)
from ultralytics import YOLO
model = YOLO("yolo11x_plate.pt")
model.export(
    format="engine",
    device=0,
    half=True,
    simplify=True,
    workspace=4,
    imgsz=640,
)

# 2) 추론 (Jetson)
model = YOLO("yolo11x_plate.engine")
results = model(
    frame,
    imgsz=640,
    conf=0.25,
    half=True,
    stream=False,
    verbose=False,
)
```

### 1.3 GStreamer / RTSP 캡처 예시

```python
# GStreamer: USB 또는 CSI (1080p 15fps)
GST_V4L2 = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1920,height=1080,framerate=15/1 ! "
    "jpegdec ! videoconvert ! appsink"
)
cap = cv2.VideoCapture(GST_V4L2, cv2.CAP_GSTREAMER)

# RTSP (dashcam 스트리밍 시)
RTSP_URL = "rtsp://192.168.1.100:554/stream1"
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 지연 최소화

# 파일 (오프라인 검증)
cap = cv2.VideoCapture("/data/dashcam/segment_001.mp4")
```

### 1.4 메모리·전력 절약

| 조치 | 명령/설정 |
|------|-----------|
| nvpmodel | `sudo nvpmodel -m 1` (15W) |
| PaddleOCR fallback | `use_gpu=False` (OOM 시) |
| OpenCV | `opencv-python-headless` |
| TensorRT workspace | `workspace=4` (GB) |

---

## 2. Phase 1 — 1주일 단위 로드맵

| Day | 목표 | 산출물 | Fail-fast |
|-----|------|--------|-----------|
| **Day 1** | Jetson 셋업 | JetPack 6, Python 3.10, venv, YOLO engine export, PaddleOCR 설치 | 설치 실패 시 Issue + 로그 첨부 |
| **Day 2** | 코드 포팅·테스트 | repo clone, `scripts/jetson_run_headless.sh` 실행, 12/12 regression (가능 시), headless 1영상 통과 | PR: Jetson 전용 config/경로 |
| **Day 3** | 실차 데이터 수집 1 | 고속도로 갓길 + 주차장 (GoldenTime·SafePlate) 영상·evidence | 메트릭 CSV 1차 확인, 이슈로 FPS/메모리 기록 |
| **Day 4** | 실차 데이터 수집 2 | 터널·도심 야간·골목 구간 | blur·latency 메트릭 추가 |
| **Day 5** | 실차 데이터 수집 3 | 버스 전용차로 인접·재촬영 구간 | 일일 메트릭 요약 Issue |
| **Day 6** | 분석·iterate | 메트릭 집계, FP rate·OCR hit rate·latency 대시보드(간이), 파라미터 튜닝 | PR: threshold/config 변경 |
| **Day 7** | 문서화 | ROADMAP 업데이트, SAFETY_CHECKLIST 서명, README 실차 섹션 | PR: docs + scripts 정리 |

---

## 3. 1일 실차 테스트 구간 우선순위

| 순위 | 구간 | 시간 | 목적 |
|------|------|------|------|
| 1 | 고속도로 갓길 | 20~30분 | GoldenTime 미양보 |
| 2 | 주차장(실내/옥외) | 30~40분 | SafePlate 후진·접촉, GoldenTime 양보 |
| 3 | 터널 진입/출구 | 15~20분 | Ghost·블러·밝기 변화 |
| 4 | 도심 주요로(밤) | 20~30분 | 블러·야간·전조등 반사 |
| 5 | 좁은 골목/주정차 밀집 | 15~20분 | 한국 특화, ROI·오탐 |
| 6 | 버스 전용차로 인접 | 10~15분 | GoldenTime (진입 금지 준수) |

### 3.1 1일 예시 스케줄 (총 2~3시간, 휴식 포함)

| 시간 | 구간 | 비고 |
|------|------|------|
| 09:00~09:15 | 장비 점검·경로 확인 | 안전 체크리스트 |
| 09:15~09:50 | 주차장 | SafePlate + GoldenTime 후진 |
| 10:00~10:15 | 휴식 | 로그 백업 |
| 10:15~10:45 | 고속도로 갓길 | GoldenTime 메인 |
| 11:00~11:20 | 터널 | Ghost·블러 |
| 11:20~11:40 | 휴식 | 점심 |
| 13:00~13:20 | 골목/주정차 | ROI·오탐 |
| 22:00~22:30 | 도심 야간 | 블러·야간 OCR |

---

## 4. 메트릭 트래킹

### 4.1 수집 메트릭

| 메트릭 | 설명 | 단위/형식 |
|--------|------|-----------|
| violation_latency | 이벤트 발생 → evidence export 완료 | 초 (float) |
| false_positive_rate | 잘못 잡은 violation / 총 프레임 | 비율 (0~1) |
| ocr_accuracy_wild | (GT 있을 때) 실차 번호판 정확률 | 비율 또는 N/M |
| blur_correlation | blur_score vs OCR 실패 여부 | 상관계수 또는 구간별 실패율 |
| power_watts | tegrastats 등 | W (평균/최대) |
| inference_fps | 프레임당 처리 시간 역수 | fps |
| memory_used_mb | 런타임 메모리 | MB |

### 4.2 로그 형식 (CSV/JSON)

**CSV (세션 요약)**  
`logs/jetson_metrics_YYYYMMDD_HHMMSS.csv`:

```csv
timestamp_utc,session_id,violation_latency_sec,fp_count,total_frames,fps_avg,memory_mb,power_watt,blur_affected_frames,evidence_count
2026-03-07T09:30:00Z,sess_001,2.3,0,4500,14.2,4200,12.1,12,2
```

**JSON (이벤트 단위)**  
`logs/events_YYYYMMDD.json`:

```json
{
  "session_id": "sess_001",
  "start": "2026-03-07T09:15:00Z",
  "end": "2026-03-07T09:50:00Z",
  "violations": [
    {
      "event_time": "2026-03-07T09:32:01Z",
      "export_done_time": "2026-03-07T09:32:03.3Z",
      "latency_sec": 2.3,
      "plate": "경기76바7789",
      "type": "DISTANCE_VIOLATION"
    }
  ],
  "metrics": {
    "fp_rate": 0.0,
    "ocr_hit_rate": 0.92,
    "blur_correlation_fail_rate": 0.15
  }
}
```

### 4.3 tegrastats (전력·메모리)

```bash
# 백그라운드 로깅 (jetson_run_headless.sh에서 호출)
tegrastats --interval 1000 --logfile logs/tegrastats_$(date +%Y%m%d_%H%M%S).log
```

---

## 5. 자동화 스크립트

- **Jetson headless 실행**: `scripts/jetson_run_headless.sh` (및 Python 래퍼)  
  - 로그 디렉터리: `logs/`  
  - evidence 디렉터리: `evidence_output/` 또는 `data/jetson_evidence/`  
  - 메트릭 CSV·JSON 출력 경로를 인자로 지정 가능.

- **메트릭 수집**: `scripts/jetson_metrics_logger.py`  
  - violation latency, FP rate, OCR hit rate, blur 구간, (선택) tegrastats 파싱.

실행 예:

```bash
# Jetson에서 (영상 파일로 테스트)
./scripts/jetson_run_headless.sh --video /data/dashcam/segment_001.mp4 --out-dir /data/jetson_evidence --logs logs

# RTSP 라이브 (dashcam 스트리밍)
./scripts/jetson_run_headless.sh --rtsp "rtsp://192.168.1.100:554/stream1" --max-duration 1800 --out-dir /data/jetson_evidence
```

---

## 6. 안전·법적 체크리스트

상세 항목은 `docs/SAFETY_CHECKLIST_REAL_WORLD.md` 참고. 요약:

| 구분 | 항목 |
|------|------|
| **안전** | 부기사만 장비 조작, 운전 중 조작 금지, 갓길 정차 시 법적 요건 충족 |
| **촬영** | 연구·시스템 검증 목적 명시, 실제 신고 시 별도 법적·개인정보 검토 |
| **데이터** | 번호판 공개 시 모자이크 옵션, 저장 기간·위치 규정, 동의 필요 시 동의 취득 |
| **사이렌·접촉** | 사이렌 재현 시 소음·오인 주의, 접촉 시뮬은 전용 장소·가상 시뮬만 |

---

## 7. Fail-fast 루프 (매일)

| 단계 | 행동 |
|------|------|
| 1 | 당일 테스트 후 메트릭·로그 확인 |
| 2 | 이슈 요약: FPS, OOM, FP 증가, OCR 저하 구간 등 |
| 3 | GitHub Issue 생성 (라벨: `phase1-real-world`, `day-N`) |
| 4 | 설정/코드 변경 시 PR로 제안, 리뷰 후 merge |
| 5 | 다음 날 같은 실패 반복 최소화 (체크리스트·스크립트 보강) |

---

## 8. 참고 파일

| 파일 | 역할 |
|------|------|
| `scripts/jetson_run_headless.sh` | Jetson headless 실행 + 로그/evidence 경로 |
| `scripts/jetson_metrics_logger.py` | 메트릭 CSV/JSON + (선택) tegrastats |
| `scripts/run_headless_plate_test.py` | 기존 headless 테스트 (로컬/Jetson 공용) |
| `docs/SAFETY_CHECKLIST_REAL_WORLD.md` | 안전·법적 체크리스트 |
| `simulation/goldentime/`, `simulation/safeplate/` | GoldenTime·SafePlate 시나리오 |

---

*실차 테스트 시 안전·법규를 준수하고, 촬영 목적과 데이터 보관 규정을 명확히 하세요.*
