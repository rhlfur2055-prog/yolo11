# 골든타임 신규 물리 연산·채증 품질 검증 보고서

시뮬레이션/QA 검증: 핀홀 거리, 후진 양보, 블러 점수, 회귀·이벤트 시퀀스.

---

## 1. 핀홀 모델 거리 연산 정밀도 (distance_checker.py)

### 1.1 고정 상수 및 해상도별 타당성

- **공식**: \( d = \frac{f \cdot W}{w} \)  
  - \(f\): 초점거리(px), \(W\): 번호판 실제 폭(m), \(w\): 이미지 내 번호판 폭(px).

- **현재 기본값**  
  - `plate_width_m = 0.52` (520mm, 한국 번호판)  
  - `focal_length_px = 0` (비활성), 활성화 시 예: **1200**  
  - `close_distance_m = 5.0`

- **1080p (1920×1080)에서의 f 추정**  
  - 일반 와이드 CCTV: 초점거리 약 3~6mm, FOV 수평 약 90°~120°.  
  - \(f_{px} \approx \frac{w_{px}}{2 \tan(\theta/2)}\). \(\theta \approx 90°\)이면 \(f \approx 0.5 \times 1920 \approx 960\); \(\theta \approx 70°\)이면 \(f \approx 1300\).  
  - **1200px**는 1080p 기준 물리적으로 타당한 범위(대략 900~1500).

- **720p (1280×720)**  
  - 같은 광학계면 \(f_{720} = f_{1080} \times (1280/1920) \approx 800\).  
  - 현재 구현은 **해상도별 f 자동 보정 없음**. 720p 입력에 1200을 쓰면 \(d\)가 약 1.5배 과대 추정됨.  
  - **권장**: 해상도 변경 시 `focal_length_px`를 비율로 조정하거나, 초기 1프레임에서 `frame_width` 기준 스케일링.

### 1.2 오차 분석: 화면 중앙 vs 외곽(렌즈 왜곡)

- **핀홀 모델 가정**: 직선 투영, 왜곡 없음.  
- **실제 와이드 렌즈**: 방사형 왜곡으로 **외곽으로 갈수록 w가 과대 추정**되기 쉬움 → \(d = fW/w\)는 **과소 추정**(더 가깝게 나옴).  
- **영향**: 차량이 화면 중앙→코너로 이동 시 \(w\)가 튀어서 \(d\)가 순간적으로 튀는 현상 가능.  
- **왜곡 보정**: 현재 **미구현**. 정밀도 요구 시 카메라 캘리브레이션 후 undistort 적용 후 bbox 사용 권장.

### 1.3 임계값 테스트: close_distance_m=5.0m ↔ bbox 비율 상관관계

- \(d = 5\text{m}\), \(W = 0.52\text{m}\), \(f = 1200\)이면  
  \(w = \frac{fW}{d} = \frac{1200 \times 0.52}{5} = 124.8 \approx 125\) px (가로).

- **1080p**에서 번호판 세로를 약 0.21×가로(110/520)로 가정하면  
  면적 ≈ \(125 \times 26 \approx 3250\) px²,  
  프레임 면적 = 1920×1080 = 2,073,600 → **비율 ≈ 0.00157 (0.157%)**.

- **위반 전이**:  
  - `close_distance_m = 5.0` 사용 시: **픽셀 가로폭 w ≥ 125px(가정)** 구간이 `violation_duration_sec` 이상 유지되면 위반.  
  - 비율 기준으로는 **bbox 면적 비율 ≳ 0.15%~0.16%**가 5m 전이 구간에 대응한다고 볼 수 있음.  
  - 기존 `close_ratio_threshold = 0.0008`(0.08%)는 5m보다 더 먼 거리(~8~10m급)에 대응하므로, 핀홀 모드와 비율 모드는 **서로 다른 절대 거리 기준**을 가짐.

---

## 2. 후진 양보(Yield) 판별 알고리즘 견고성

### 2.1 히스토리 윈도우(15프레임) 분석

- **BBOX_HISTORY_LEN = 15** (distance_checker.py PlateDistanceRecord).  
- 30fps 기준 **0.5초** 구간.  
- **노이즈 억제**: 3~5프레임만 쓰면 jitter에 민감; 15프레임이면 단순 선형 추세(첫–끝)로 2%/s 수준의 면적 변화율을 안정적으로 추정 가능.  
- **반응 속도**: 0.5초 지연은 “양보 시작 후 최대 0.5초 뒤에 인식” 수준으로, 긴급차 시나리오에서 수용 가능.  
- **결론**: 15프레임은 **jitter 억제와 반응 속도 균형**에 적절. 더 빠른 인식이 필요하면 8~10으로 줄이되, 노이즈로 인한 오양보 증가 가능성 있음.

### 2.2 면적 변화율(ΔArea/Δt) 임계값

- **YIELD_AREA_RATE_MIN = 0.02** (2%/초).  
  - `rate = (a1 - a0) / (delta_t * a0)`, `delta_t` 최소 0.3초.

- **천천히 후진 vs 정지**  
  - 정지: 측정 오차·jitter로 rate가 ±1%/s 수준 나올 수 있음.  
  - 2%/s는 “의도된 느린 후진”과 “정지+jitter”를 구분하는 **최소 유의 변화율**로 적절.  
  - 더 느린 후진(예: 1%/s)까지 인정하려면 0.01로 낮출 수 있으나, 정지 오탐 증가.

- **권장**:  
  - 현장 데이터가 쌓이기 전까지 **0.02 유지**.  
  - “아주 천천히 후진”까지 포착하려면 **0.01**로 완화 검토(정지 구간 필터 강화와 함께).

### 2.3 복합 기동 및 front_roi와의 간섭

- **면적만으로 is_yielding**:  
  - 후진+좌/우 꺾임 시 bbox는 **회전·시야 변화**로 면적이 일시 감소할 수 있음.  
  - 그러면 해당 구간에서 `is_yielding`이 False로 바뀌고, **그 구간만** 위반 누적이 다시 켜질 수 있음.  
  - “후진으로 양보했다”는 전체 맥락에서는 일부 구간이 위반으로 잡히는 **엣지 케이스** 가능.

- **front_roi_y_min_ratio와의 관계**  
  - 전방 ROI는 “bbox 중심이 화면 하단 비율 이내인지”만 필터.  
  - `is_yielding`은 **같은 record**에 대해 bbox_history로 계산되며, ROI 필터는 **거리 판정 대상 여부**만 결정.  
  - 따라서 **직접적인 논리 충돌은 없음**. 다만 ROI로 걸러진 차량은 애초에 거리/양보 판정 대상이 아니게 됨.

- **권장**: 복합 기동 시 오탐을 줄이려면 **이동 벡터(dx, dy)**를 보조 지표로 사용(예: dy > 0인 구간 비중이 높을 때만 양보로 인정)하는 확장을 고려.

---

## 3. 채증 품질(Blur Score) 및 증거 신뢰도

### 3.1 Laplacian 임계값(50) 적정성

- **evidence_export.py**: `BLUR_THRESHOLD = 50.0` (blur_score < 50 → 흐린 프레임).  
- **blur_score**: `cv2.Laplacian(gray, cv2.CV_64F).var()` (선명도 지표).

- **일반적 경험**  
  - Laplacian variance < 50: 상당히 흐린 영상.  
  - 50~100: 다소 흐림.  
  - 100 이상: 대체로 선명.  
  - OCR 인식률은 영상·전처리에 따라 다르나, **50 미만 구간에서 급격히 떨어지는 경우가 많음**.

- **상관계수 분석**  
  - 공식 실험은 **동일 영상·동일 OCR 파이프라인**으로 프레임별 blur_score와 OCR 정답 여부를 수집해 상관계수 산출해야 함.  
  - 여기서는 **권장만 제시**:  
    - **50**: “흐린 프레임” 분류용으로 무난.  
    - **너무 엄격(예: 100)**: 흐린 프레임이 과다 분류 → `blur_affected_frames` 과다, 증거가 대부분 “contains_blur”로 나와 활용도 저하.  
    - **너무 느슨(예: 20)**: 심하게 흐린 프레임이 “정상”으로 넘어가 OCR 오독 위험.

- **결론**: 50은 **증거 누락과 오독 사이의 타협점**으로 타당. 프로젝트별로 샘플 검증 후 30~70 범위에서 미세 조정 권장.

### 3.2 evidence_quality 전달 방식

- **현재**: `evidence_quality`: `"normal"` | `"contains_blur"`, `blur_affected_frames` (흐린 프레임 수).  
- **plates.json**에 포함되며, 리포트(report.txt)에서도 언급 가능.

- **법적 신뢰 수준 전달**  
  - “contains_blur”만으로는 **어느 정도 신뢰할 수 있는지**가 사용자에게 불명확할 수 있음.  
  - **권장**:  
    - `evidence_quality`를 유지하되, **report.txt**에 예시 문구 추가:  
      - “본 채증 영상 중 흐린 프레임이 N장 포함되어 있습니다. 핵심 순간 스크린샷(evidence_start, midpoint, evidence_end)은 선명도 기준으로 선별되었을 수 있습니다.”  
    - 또는 `evidence_reliability` 같은 필드로 “high / medium / low”를 `blur_affected_frames` 비율로 산출해 제공.

---

## 4. 회귀 테스트 및 이벤트 연동

### 4.1 12/12 유지(Zero-impact) 검증

- **신규 옵션 기본값**  
  - `use_pinhole_distance`: False  
  - `front_roi_y_min_ratio`: 0.0  
  - `compute_blur_score`: False  

- **test_ocr_accuracy.py**  
  - **plate_engine_pro**, **22/** 정적 이미지, **plate_ocr_postfilter_v2** 등만 사용.  
  - 골든타임·시뮬레이션 프레임워크는 **호출하지 않음**.  
  - 따라서 **옵션 on/off와 무관하게 12/12 결과는 동일**해야 함 (Zero-impact).

- **검증 방법**:  
  - `python test_ocr_accuracy.py` 실행 → 12/12 통과 확인.  
  - (선택) run_goldentime에서 `compute_blur_score=True`, `use_pinhole_distance=True` 등으로 실행해도, **같은 영상·같은 구간**에서 plate_engine_pro가 반환하는 detection_result는 동일함. 채증 품질·거리 판정만 추가 정보가 붙을 뿐.

### 4.2 YIELD_DETECTED 발행 시 위반 누적 중단 시퀀스

- **흐름 요약**  
  1. `_on_detection_result`에서 `record.update_bbox(..., bbox=bbox)` 호출.  
  2. `update_bbox` 내부에서 `_update_yielding()`로 `is_yielding` 갱신.  
  3. **is_yielding == True이면**  
     - `close_start_time = None`, `close_duration = 0.0`, **is_close = False** 로 설정됨 (비율 모드·핀홀 모드 공통).  
  4. 그 다음 `check_violation(violation_duration, timestamp)` 호출.  
     - `check_violation`은 `self.is_close and self.close_duration >= violation_duration` 일 때만 위반 판정.  
     - **is_close가 False**이므로 **위반으로 전이되지 않음** (즉시 위반 누적 중단).  
  5. `is_yielding`이 True인 동안 **YIELD_DETECTED**는 최초 1회만 발행(`_yield_emitted` 플래그).

- **세션 종료**  
  - “세션 정상 종료”는 **SIREN_ENDED**로 이뤄짐.  
  - YIELD_DETECTED는 **위반 누적만 중단**할 뿐, 세션(사이렌 구간) 자체를 끝내지는 않음.  
  - 따라서 “YIELD_DETECTED 발행 → 위반 누적 중단”까지가 검증 대상이며, **동일 프레임 내에서** 위반 체크가 yield 이후에 수행되므로 **즉시 중단**이 보장됨.

---

## 5. 검증 체크리스트

| 항목 | 내용 | 상태 |
|------|------|------|
| 핀홀 상수 | 1080p에서 f=1200, W=0.52 타당 | 문서 검증 |
| 해상도 | 720p 시 f 스케일 미적용 → 별도 설정 필요 | 권장 사항 반영 |
| 왜곡 | 렌즈 왜곡 미보정, 코너에서 d 튐 가능 | 문서 명시 |
| 5m 전이 | w≈125px(1080p), 면적 비율 ≈0.16% | 수식 도출 |
| 15프레임 | 0.5초, jitter/반응 균형 적절 | 분석 완료 |
| 2%/s 임계값 | 정지 vs 느린 후진 구분에 적절 | 권장 유지 |
| 복합 기동 | 면적만으로는 회전 구간 오탐 가능 | 권장: 벡터 보조 |
| Blur 50 | 흐림 분류·OCR 상관 타협점 | 권장 유지 |
| evidence_quality | report에 신뢰 수준 문구 보강 권장 | 권장 사항 |
| 12/12 | 골든타임 옵션과 무관 Zero-impact | 검증 방법 명시 |
| YIELD 시퀀스 | is_yielding → is_close=False → check_violation 미판정 | 코드 추적 완료 |

---

## 6. 검증 스크립트 실행

```bash
# 단위 검증만 (핀홀 공식, YIELD 시퀀스)
python simulation/goldentime/test_verification.py

# 회귀 12/12 포함 (YOLO/OCR 로딩으로 1~2분 소요)
RUN_REGRESSION=1 python simulation/goldentime/test_verification.py
```

- `test_pinhole_formula`: d = f×W/w, 5m/10m 대응 w 검증.
- `test_yield_stops_violation_accumulation`: 면적 증가 시 is_yielding → 위반 미누적 확인.
- `test_regression_ocr_accuracy`: RUN_REGRESSION=1일 때 test_ocr_accuracy.py 실행 후 12/12 확인.

---

*문서 위치: simulation/goldentime/VERIFICATION_GOLDENTIME.md*
