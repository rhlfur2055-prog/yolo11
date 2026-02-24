# YOLO26 번호판 인식 — 실행 명령지시 (RUN.md)

> 경로: `c:\tools\yolo26\RUN.md`
> 최종 갱신: 2026-02-24

---

## ① plate_gui.py — GUI 실시간 인식 (메인)

```bash
# 기본 실행 (파일 다이얼로그로 동영상 선택)
python plate_gui.py

# 동영상 직접 지정
python plate_gui.py video.mp4

# YouTube URL → 다운로드 후 재생
python plate_gui.py --youtube "https://youtube.com/watch?v=..."
```

**창 구성 (960×860, 최소 900×600):**
```
┌─ YOLO26 번호판 인식 ─────────────────────────┐
│  [영상 재생 영역] 960×540 전체 너비           │
│    bbox + 번호판 텍스트 오버레이              │
├───────────────────────────────────────────────┤
│  Detection Log (시간 | 번호판 | 신뢰도)       │
│  🟢 00:12.3  12가3456   95%                  │
│  🟡 00:08.7  서울56다1234  78%               │
├───────────────────────────────────────────────┤
│  [▶재생] [⏸일시정지] [📂열기] [💾저장] [API]  │
│  FPS: 24 | 탐지: 2 | 총 인식: 4              │
└───────────────────────────────────────────────┘
```

**--youtube 동작 흐름:**
```
1. yt-dlp로 temp_youtube/에 mp4 다운로드
2. 다운로드 완료 → _open_video(로컬 경로) 자동 호출
3. 이미 다운로드된 파일은 재다운로드 스킵 (.archive.txt)
```

**내부 엔진:**
- `PlateRecognizer` (plate_recognition_4k.py)
- 모델 우선순위: yolo26.engine → yolo26.onnx → HuggingFace → yolo26.pt → yolo11n.pt
- OCR: EasyOCR (ko, en)
- 결과 키: `text`, `ocr_confidence`, `bbox`, `is_valid_plate`, `pattern_score`

---

## ② plate_server.py — Flask 웹 서버

```bash
# 기본 실행 (포트 5000, 브라우저 자동 열림)
python plate_server.py

# 포트 변경
python plate_server.py --port 8000

# 브라우저 자동 열기 안 함
python plate_server.py --no-browser

# 외부 접속 허용
python plate_server.py --host 0.0.0.0 --port 5000
```

**웹 페이지:**
| URL | 설명 |
|-----|------|
| `http://127.0.0.1:5000/` | 메인 (업로드 + 결과 뷰어) |
| `http://127.0.0.1:5000/realtime` | 실시간 인식 (웹캠/YouTube/파일) |
| `http://127.0.0.1:5000/html` | HTML 문서 목록 |

**API 엔드포인트:**
| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/upload` | 영상 파일 업로드 → 비동기 인식 |
| POST | `/api/recognize_url` | URL → 다운로드 후 인식 |
| GET | `/api/status/<job_id>` | 작업 상태 조회 |
| GET | `/api/results/<job_id>` | 인식 결과 JSON |
| GET | `/api/history` | 전체 결과 디렉토리 목록 |
| GET | `/stream?source=0` | 실시간 MJPEG 스트림 |
| GET | `/stream/<job_id>` | 작업 기반 영상 스트림 |

**실시간 스트림 엔진 설정 (경량):**
```python
# plate_server.py 내부 (60~62줄)
_stream_engine.consecutive_required = 1       # 즉시 표시
_stream_engine.config.PREPROCESS_METHODS = ["original", "clahe"]  # 2종만
_stream_engine.config.DETECT_CONF = 0.35      # 낮은 임계값
```

---

## ③ plate_engine_pro.py — Pro 엔진 CLI

```bash
# 웹캠 실시간
python plate_engine_pro.py

# 동영상 파일
python plate_engine_pro.py --input video.mp4

# RTSP 스트림
python plate_engine_pro.py --input "rtsp://192.168.1.100/stream"

# 카메라 ID + 화면 표시 안 함
python plate_engine_pro.py --input 0 --camera CAM01 --no-show

# 수배 차량 등록
python plate_engine_pro.py --alert-add "12가3456"
```

**결과 키:** `plate`, `confidence`, `bbox`, `is_alert`
**종료:** `q` 키 / Ctrl+C

---

## ④ plate_recognition_4k.py — 배치 처리 CLI

```bash
# 기본
python plate_recognition_4k.py video.mp4

# 전체 옵션
python plate_recognition_4k.py video.mp4 \
    -o ./plate_results \
    --model-size n \
    --no-sahi \
    --frame-skip 3 \
    --burst-frames 10 \
    --confidence 0.3
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-o, --output` | `./plate_results` | 결과 출력 디렉토리 |
| `--model` | 자동 | 사용자 지정 .pt 경로 |
| `--model-size` | `n` | n/s/m/l/x |
| `--confidence` | `0.5` | 탐지 최소 신뢰도 |
| `--no-sahi` | OFF | SAHI 타일링 비활성화 |
| `--frame-skip` | `5` | SCANNING 프레임 스킵 |
| `--burst-frames` | `10` | CAPTURING 버스트 수 |

**출력:**
```
plate_results/
├── results.json
├── plate_0000.png
├── plate_0000_preprocessed.png
└── ...
```

---

## ⑤ make_demo_video.py — 데모 영상 생성

```bash
# temp_youtube/에 mp4 넣고 실행
python make_demo_video.py
```

**출력:**
```
demo_output/
├── demo_영상이름.mp4      ← 오버레이 포함 개별 영상
├── final_demo.mp4          ← 전체 합본
└── thumbnail.jpg           ← 최다 인식 프레임 썸네일
```

---

## ⑥ 환경변수

```bash
# Windows
set PLATE_CONSECUTIVE_FRAMES=1    # 연속 프레임 필터 (1=즉시 표시)
set HOME=C:\tools\yolo26          # PaddleOCR 홈 디렉토리

# Linux/Mac
export PLATE_CONSECUTIVE_FRAMES=1
export HOME=/home/user/yolo26
```

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PLATE_CONSECUTIVE_FRAMES` | `3` | N프레임 연속 감지 후 표시 (1=즉시) |
| `HOME` | 시스템 기본 | PaddleOCR 모델 캐시 경로 |

---

## ⑦ 필수 패키지 설치

```bash
# 전체 설치 (requirements.txt)
pip install -r requirements.txt

# 개별 설치
pip install ultralytics            # YOLO26 (탐지 엔진)
pip install easyocr                # EasyOCR (한국어+영어 OCR)
pip install paddlepaddle paddleocr # PaddleOCR (Pro 엔진용, 선택)
pip install flask                  # 웹 서버
pip install huggingface_hub       # HuggingFace 모델 다운로드
pip install opencv-python          # OpenCV (GUI용, headless 아님)
pip install pillow numpy           # 이미지 처리
pip install yt-dlp                 # YouTube 다운로드 (--youtube 옵션용)
```

**GPU 가속 (선택):**
```bash
# CUDA 지원 PyTorch (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# fast-alpr ONNX GPU (PlateEngineFast용)
pip install fast-alpr[onnx-gpu]
```

---

## 🎯 추천 실행 순서

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1단계: GUI 실시간 인식 (메인 목적)
  python plate_gui.py video.mp4
  python plate_gui.py --youtube "https://youtube.com/watch?v=..."

2단계: 웹 서버 (폰 카메라 / 원격 접속)
  python plate_server.py
  → http://127.0.0.1:5000/realtime

3단계: 배치 처리 (대량 영상)
  python plate_recognition_4k.py video.mp4 -o ./results

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📁 파일 역할 요약

| 파일 | 역할 | 실행 |
|------|------|------|
| `plate_gui.py` | GUI 실시간 인식 (메인) | `python plate_gui.py` |
| `plate_recognition_4k.py` | 핵심 인식 엔진 (2833줄) | CLI 또는 import |
| `plate_engine_pro.py` | 상용급 Pro 엔진 (869줄) | CLI 또는 import |
| `plate_ocr_pipeline.py` | OCR 후처리 파이프라인 | import 전용 |
| `plate_ocr_postfilter.py` | OCR 결과 필터 | import 전용 |
| `plate_server.py` | Flask 웹 서버 (1122줄) | `python plate_server.py` |
| `youtube_helper.py` | YouTube 다운로드 헬퍼 | import 또는 단독 |
| `make_demo_video.py` | 데모 영상 생성 | `python make_demo_video.py` |
| `ocr_test.py` | OCR 엔진 비교 테스트 | 개발용 |
| `run_test.bat` | 실행 테스트 (환경 확인 + 메뉴) | 더블클릭, 또는 PowerShell: `.\run_test.bat` |
