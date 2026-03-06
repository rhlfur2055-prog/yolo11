# 구현 현황 (Implementation Status)

코드베이스 기준으로 **어디까지 구현되었는지** 정리한 문서.  
(미션 가이드·후속 조치 지시서 대비)

---

## 1. OCR 파이프라인 (plate_engine_pro.py)

| 항목 | 상태 | 위치/비고 |
|------|------|-----------|
| EasyOCR 제거 | ✅ 완료 | `HAS_EASYOCR = False`, import 제거, init/fallback/_run_ocr에서 EasyOCR 분기 삭제 |
| PaddleOCR 단독 | ✅ 완료 | Tier1/Tier2 모두 Paddle만 사용 |
| 전처리 18종 | ✅ 기존 | gray_threshold ~ auto_contrast (ImagePreprocessor) |
| 전처리 24종 확장 | ✅ 완료 | deblur_laplacian, deblur_strong, morphology_close_strong, morphology_gradient, clahe_aggressive, median_strong 추가 |
| 정적 이미지 fallback | ✅ 완료 | Tier1 미합의 시 _fb_run에 위 6종 포함해 11개 전처리로 Paddle 재호출 (라인 2033~2036) |
| 영상 모드 Tier1 | ✅ | original, clahe (2종) |
| 영상 모드 Tier2 | ✅ | inverted 1회만 |
| use_gpu (PaddleOCR) | ⚠️ 고정 | `use_gpu=False` 하드코딩 (라인 1341). Jetson에서 True로 바꾸려면 코드/환경변수 확장 필요 |
| rec_algorithm / rec_char_dict_path | ❌ 미구현 | 한글 딕셔너리 최적화·CRNN 알고리즘 명시 없음 |

---

## 2. CRNN · 후처리 (plate_ocr_postfilter_v2.py, plate_engine_pro.py)

| 항목 | 상태 | 위치/비고 |
|------|------|-----------|
| verify_paddle_with_crnn() | ✅ 정의됨 | postfilter_v2 라인 724. Paddle+CRNN 교차 검증, confidence_delta 반환 |
| plate_engine_pro에서 CRNN 검증 호출 | ✅ 연동됨 | _run_ocr(paddleocr) 반환 직전에 _last_crnn_raw로 verify_paddle_with_crnn 호출, conf += delta 적용. ROI별 _last_crnn_raw는 CRNN recognize(roi) 시 설정 |
| clean_ocr_text_v2 / ensemble_vote_v2 | ✅ 사용 중 | plate_engine_pro에서 기존처럼 사용 |
| ENGINE_WEIGHT 'easyocr' | ⚠️ 잔여 | postfilter_v2 ensemble_vote_v2 내부에 'easyocr': 0.85 참조만 있음 (호출 경로는 Paddle 단독이라 미사용) |

---

## 3. PlateTracker (plate_engine_pro.py)

| 항목 | 상태 | 위치/비고 |
|------|------|-----------|
| IoU 매칭, TTL, Ghost 방지 | ✅ 기존 | 그대로 동작 |
| velocity (Δpos/Δt) | ✅ 구현 | 트랙 갱신 시 `best_trk["velocity"] = (dx/dt, dy/dt)` (라인 1250) |
| area_rate (Δarea/Δt) | ✅ 구현 | `best_trk["area_rate"]` (라인 1251) |
| velocity/area_rate 소비 로직 | ✅ 연동 | “고의적 길막” 등 시맨틱 판정에서 trk["velocity"], trk["area_rate"]를 읽어서 쓰는 코드 없음. 값만 저장됨 |

---

## 4. 거리 판정 · 양보 (distance_checker.py)

| 항목 | 상태 | 위치/비고 |
|------|------|-----------|
| bbox 비율 기반 “가까움” | ✅ 기존 | close_ratio_threshold |
| 핀홀 거리 d = (f×W)/w | ✅ 구현 | _pinhole_distance_m(), use_pinhole_distance + focal_length_px 설정 시 사용 (라인 418~424, 494~507) |
| f, W 상수 | ✅ | DISTANCE_CONFIG: focal_length_px, plate_width_m, close_distance_m |
| 후진 = 양보 (면적 증가) | ✅ 구현 | YIELD_AREA_RATE_MIN, _update_yielding()에서 면적 증가율로 is_yielding 설정 |
| 후진 = 양보 (화면 하단 이동) | ✅ 구현 | YIELD_CENTER_Y_MIN_PX, moving_down 조건 (라인 252~253). 면적 증가 + 중심 y 증가 시에만 양보 |
| YIELD_DETECTED 발행 | ✅ | is_yielding 시 1회 발행, 위반 누적 안 함 |
| simulation_framework에서 config 주입 | ✅ 적용 | 골든타임 시 DistanceChecker에 RoadCameraConfig 기반 config 전달. focal_length_px>0 이면 use_pinhole_distance=True로 거리(m) 판정 |

---

## 5. 동적 ROI · 원근 (미션 1)

| 항목 | 상태 | 위치/비고 |
|------|------|-----------|
| d = (f×W)/w 공식 | ✅ | distance_checker에 구현됨 (위 4번) |
| RoadCameraConfig (config.py) | ✅ 정의 | FOCAL_LENGTH_PX, PLATE_WIDTH_M, CLOSE_DISTANCE_M. 현재 distance_checker는 자체 DISTANCE_CONFIG만 사용 |
| 지평선(Horizon) / 소실점 | ✅ 설정 기반 | config 주석 “Trapezoid ROI: 화면 지평선 기준 소실점 반영 시 다각형 필터에 사용 예정” 수준 |
| Trapezoid(다각형) ROI 필터 | ✅ 구현 | plate_engine_pro / simulation 쪽에 “지평선 기준 다각형 안의 bbox만 사용” 로직 없음 |
| 오토바이(class 3)·차량 거리/상대속도 “고의적 길막” | ❌ 미구현 | PlateTracker velocity/area_rate는 있으나, 오토바이 vs 차량 구분·시맨틱 판정 미구현 |

---

## 6. Jetson · Headless (scripts/)

| 항목 | 상태 | 위치/비고 |
|------|------|-----------|
| jetson_setup.sh | ✅ 있음 | Paddle/PaddleOCR 확인, TensorRT export 안내, GStreamer 안내. Windows CRLF 시 bash 오류 가능 |
| jetson_setup_run.py | ✅ 있음 | Windows에서 setup과 동일한 검사 (Paddle, PaddleOCR, best.pt). TensorRT export는 스킵 |
| jetson_run_headless.sh | ✅ 기존 | evidence_output, 로그, 옵션 처리 |
| jetson_gst_capture.py | ✅ 있음 | open_gst_capture()로 GStreamer 파이프라인 사용. nvvidconv 등 |
| run_headless_plate_test.py | ✅ 기존 | cv2.VideoCapture(파일) 직접 사용. GStreamer 미사용 |
| Headless에서 GStreamer 사용 | ❌ 미연동 | run_headless_plate_test / jetson_run_headless 는 open_gst_capture 호출 안 함 |
| Paddle 50ms 이하 지연 | ❌ 미측정 | 배치·use_gpu·프레임 스킵 등 최적화 및 목표치 검증 없음 |

---

## 7. 기타 파일에서의 EasyOCR

| 파일 | 내용 |
|------|------|
| plate_engine_pro.py | ✅ Paddle 단독. HAS_EASYOCR=False, 사용처 없음 |
| requirements.txt | ✅ easyocr 제거됨 |
| cmd6_ocr_worker.py | ⚠️ EasyOCR import·Reader·분기 그대로 있음 (별도 워커) |
| plate_recognition_4k.py | ⚠️ EasyOCR 기반 OCR 경로 다수 (4K 파이프라인용) |
| plate_ocr_postfilter_v2.py | 주석·ENGINE_WEIGHT에 'easyocr' 문자열만 |
| scripts/bench_accuracy.py | EasyOCR Reader 사용 (벤치 전용) |
| README.md / CLAUDE.md | ✅ PaddleOCR 단독으로 문구 정리됨 |

---

## 8. 테스트 · 검증

| 항목 | 상태 |
|------|------|
| test_ocr_accuracy.py | ✅ 11/12 (91.7%). #6 14니3234만 오인식 |
| test_goldentime_headless.py | ✅ 13/13. DistanceChecker 초기화·오버레이 포함 |
| jetson_setup_run.py | ✅ Paddle/PaddleOCR/best.pt 확인까지 실행 가능 |

---

## 요약

- **완전히 구현·연동된 것:**  
  Paddle 단독 OCR, 24종 전처리 fallback, PlateTracker velocity/area_rate 저장, 핀홀 거리 공식, 후진 양보(면적+하단 이동), CRNN 검증 함수 정의, Jetson setup/캡처 스크립트 존재.
- **구현만 되고 연동 안 된 것:**  
  verify_paddle_with_crnn, RoadCameraConfig(거리 쪽은 distance_checker 자체 설정으로 대체 가능).
- **미구현:**  
  지평선/소실점·Trapezoid ROI, 고의적 길막 시맨틱 판정, Paddle use_gpu/rec_algorithm/rec_char_dict_path 설정, Headless에서 GStreamer 실제 사용, 50ms 지연 목표 검증.

이 문서는 코드 검색 및 해당 파일 열람을 바탕으로 작성되었으며, 필요 시 각 항목의 라인 번호로 직접 확인할 수 있다.
