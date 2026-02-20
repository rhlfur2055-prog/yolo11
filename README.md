# YOLO26 번호판 인식 시스템

YOLO + OCR 기반 실시간 번호판 인식 파이프라인. 한국/영국 번호판 지원, 4K CCTV 영상 처리.

## 아키텍처

```
[입력]                     [엔진]                        [출력]
video.mp4  ──┐
YouTube URL ─┤──▶ plate_recognition_4k.py ──▶ plate_results_v3/
CCTV stream ─┘    (YOLO탐지 → OCR → 검증)     ├─ results.json
                                                ├─ plate_XXXX.png
     plate_gui.py ◀──── 실시간 GUI              └─ plate_XXXX_preprocessed.png
     plate_server.py ◀─ 웹 뷰어 (localhost:8000)
```

## 파일 구조

| 파일 | 역할 |
|------|------|
| `plate_recognition_4k.py` | 핵심 엔진 - YOLO 탐지 + 멀티 OCR + 7종 전처리 |
| `plate_gui.py` | Tkinter + OpenCV 실시간 GUI (자동 저장) |
| `plate_server.py` | HTTP 결과 뷰어 (멀티 디렉토리 자동 스캔) |
| `youtube_plate_test.py` | YouTube 4K 영상 다운로드 → 인식 테스트 |
| `test_frames.py` | 프레임별 빠른 QA 테스트 |
| `yolo26.pt` | 번호판 전용 YOLO 모델 (5.3MB) |
| `yolo11n.pt` | COCO fallback 모델 (5.4MB) |

## 모델 우선순위

1. **로컬 `yolo26.pt`** - 번호판 전용 fine-tuned 모델
2. **HuggingFace** - 온라인 번호판 모델 자동 다운로드
3. **COCO `yolo11n.pt`** - 차량 탐지 후 crop → OCR fallback

## OCR 파이프라인

PaddleOCR (한글) → EasyOCR (영문) → Tesseract (UK PSM7)

7종 이미지 전처리: deblur, gamma, stretch, CLAHE, bilateral, morphology, threshold

## 사용법

```bash
# 설치
pip install -r requirements.txt

# GUI 실행
python plate_gui.py
python plate_gui.py video.mp4

# CLI 배치 처리
python plate_recognition_4k.py --input video.mp4

# YouTube 테스트
python youtube_plate_test.py "https://youtube.com/watch?v=..."

# 결과 뷰어 (자동 스캔)
python plate_server.py
```

## 결과 디렉토리

- `plate_results_v3/` - GUI 자동 저장 (최신)
- `plate_results_v2/` - CLI 배치 결과
- `plate_results_highway/` - 고속도로 테스트
