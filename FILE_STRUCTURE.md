# 번호판 인식 프로젝트 파일 구조

## 현재 디렉터리 구조 요약

```
(프로젝트 루트)
├── plate_server.py          # 웹 서버 (localhost:5000) → PlateEnginePro 사용
├── plate_engine_pro.py      # 번호판 인식 엔진 (YOLO 탐지 + OCR)
├── plate_recognition_4k.py  # 한글 교정 / 포맷 검증
├── plate_ocr_postfilter_v2.py
├── 실시간_GUI_localhost.bat # 서버 실행: python plate_server.py --port 5000
├── train.py                 # 번호판 YOLO 학습 → runs/detect/highway_plate/weights/best.pt 생성
├── dataset/
│   └── yolo_format/
│       └── data.yaml        # YOLO 데이터 설정 (path 등)
├── movie/                   # (없을 수 있음) 기본 영상 hiway.mp4
├── uploads/                 # 서버 업로드 파일
├── server_results/          # 서버 처리 결과
└── *.pt                     # ★ 여기에 번호판 전용 YOLO 가중치가 있어야 함
```

## 번호판이 “전혀 인식되지 않는” 이유

1. **번호판 전용 YOLO 모델이 없음**  
   엔진은 아래 순서로만 모델을 찾습니다.
   - `yolo11x_plate.pt` (config 기본값) → 없음
   - `yolo26.pt` (fallback) → 없음
   - `yolo11n.pt` → Ultralytics가 **COCO 80클래스** 모델을 자동 다운로드

   **COCO 모델에는 ‘번호판(plate)’ 클래스가 없습니다.**  
   사람/차는 잡혀도, 번호판 영역을 탐지하지 못해 인식 건수가 0으로 나옵니다.

2. **실행 경로**  
   `실시간_GUI_localhost.bat`는 `cd /d C:\tools\yolo26` 로 다른 폴더를 기준으로 할 수 있어,  
   현재 이 프로젝트 폴더에 `.pt`를 넣어도 “다른 쪽”에서 실행하면 그쪽 CWD 기준으로만 찾습니다.

3. **기본 영상**  
   서버 기본 영상은 `movie/hiway.mp4` 입니다.  
   `movie` 폴더가 없거나 파일이 없으면 소스 선택/업로드가 필요합니다.

## 해결 방법 (인식이 되게 하려면)

### A. 학습으로 번호판 모델 만들기 (권장)

1. **데이터 준비**  
   - `dataset/yolo_format/` 에 YOLO 형식으로 이미지와 라벨 구성  
   - `dataset/data.yaml` 이 없으면 `dataset/yolo_format/data.yaml` 내용을 참고해  
     `dataset/data.yaml` 을 만들거나, `train.py` 안의 `DATA_YAML` 경로를  
     `dataset/yolo_format/data.yaml` 로 맞춥니다.

2. **사전학습 가중치**  
   - `train.py`는 `yolo26n.pt` 를 기준으로 학습합니다.  
   - 없으면 Ultralytics에서 받거나, `yolo11n.pt` 등으로 대체할 수 있도록  
     `train.py`의 `BASE_MODEL` 경로를 확인합니다.

3. **학습 실행**  
   ```bash
   python train.py
   ```  
   학습이 끝나면 다음 경로에 best 모델이 생성됩니다.  
   - `runs/detect/highway_plate/weights/best.pt`

4. **엔진이 이 모델을 쓰도록**  
   - `plate_engine_pro.py`는 **프로젝트 루트 기준**으로  
     `runs/detect/highway_plate/weights/best.pt` 를 우선 사용하도록 수정할 수 있습니다.  
   - 또는 `best.pt` 를 프로젝트 루트로 복사해  
     `yolo11x_plate.pt` / `yolo26n.pt` 등으로 이름을 바꿔 두면,  
     현재 엔진의 “우선순위 목록”에 따라 자동으로 사용됩니다.

### B. 이미 있는 번호판 전용 .pt 사용

- 번호판으로 학습된 `.pt` 파일이 있다면  
  **서버를 실행하는 작업 디렉터리(또는 스크립트 위치)** 에 두고  
  다음 이름 중 하나로 두면 엔진이 우선 사용합니다.  
  - `yolo11x_plate.pt`  
  - `yolo26n.pt`  
  - `yolo26.pt`  
  (엔진 코드의 모델 우선순위 목록에 따라 결정)

## 정리

- **파일 구조**는 위와 같고,  
- **인식이 0건인 이유**는 “번호판 전용 YOLO 가중치가 없어서 COCO용 `yolo11n.pt`만 쓰이기 때문”입니다.  
- **실제로 번호판이 나오게 하려면**  
  - 이 프로젝트에서 `train.py`로 학습한 `best.pt`를 쓰거나,  
  - 외부에서 받은 번호판 전용 `.pt`를 프로젝트(및 실행 CWD)에 넣어야 합니다.

엔진 쪽 수정으로 “스크립트 기준 경로”와 “runs/.../best.pt 우선 사용”을 적용해 두면,  
같은 프로젝트에서 학습한 모델을 바로 쓸 수 있습니다.
