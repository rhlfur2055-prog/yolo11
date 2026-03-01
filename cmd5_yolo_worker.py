# -*- coding: utf-8 -*-
"""
CMD 5: YOLO 탐지 워커 프로세스
plate_engine_pro.py process_frame()에서 YOLO 탐지 부분만 추출하여
별도 프로세스에서 실행. multiprocessing.Queue로 IPC 통신.

추출 원본: plate_engine_pro.py L1431-1548
"""
import os
import time
import traceback
import numpy as np
from pathlib import Path
from multiprocessing import Queue

from pipeline_common import bbox_iou, DETECT_CONFIG, CMD_STOP


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 우선순위 (plate_engine_pro.py L9-17 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_MODEL_PRIORITY = [
    "yolo11x_plate.pt",    # ★ YOLOv11x fine-tuned (mAP@50=98.4%) - 최우선
    "yolo11n_plate.pt",    # YOLOv11n 경량 번호판 전용
    "yolo26n.pt",          # YOLO26n - 최신 경량
    "yolo26s.pt",          # YOLO26s - 소형
    "yolo26.pt",           # 기존 프로젝트 모델
    "yolo11n.pt",          # COCO fallback
    "yolov8n.pt",          # 최후 fallback
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 경로 해석 (plate_engine_pro.py L1093-1110 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _resolve_plate_model():
    """프로젝트(스크립트) 폴더 기준으로 번호판용 YOLO 모델 경로를 찾는다."""
    script_dir = Path(__file__).resolve().parent
    cfg = DETECT_CONFIG
    candidates = [
        script_dir / "runs" / "detect" / "plate_korean_3k_v2" / "weights" / "best.pt",
        script_dir / "runs" / "detect" / "plate_korean_3k3" / "weights" / "best.pt",
        script_dir / "best.pt",
        script_dir / "runs" / "detect" / "highway_plate" / "weights" / "best.pt",
        script_dir / cfg["YOLO_MODEL"],
        script_dir / cfg["YOLO_FALLBACK"],
    ]
    for m in _MODEL_PRIORITY:
        candidates.append(script_dir / m)
        candidates.append(Path(m))  # CWD
    for p in candidates:
        if p.exists():
            return p
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 자동 로드 (plate_engine_pro.py L19-28 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _load_best_model():
    """우선순위에 따라 가장 좋은 모델 자동 로드"""
    from ultralytics import YOLO
    for m in _MODEL_PRIORITY:
        if os.path.exists(m):
            print(f"[CMD5-YOLO] 모델 로드: {m}")
            return YOLO(m)
    print("[CMD5-YOLO] yolo26n.pt 자동 다운로드 중...")
    return YOLO("yolo26n.pt")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 해상도별 imgsz 결정 (plate_engine_pro.py L1443-1452 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _calc_imgsz(fw):
    """프레임 너비에 따라 YOLO 추론 해상도 결정"""
    if fw >= 3840:      # 4K
        return 1920
    elif fw >= 1920:    # FHD
        return 1280
    elif fw >= 1280:    # HD
        return 960
    else:
        return 640


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 수동 NMS (plate_engine_pro.py L1463-1490 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _manual_nms(raw_boxes, is_plate_model, cfg):
    """겹치는 bbox 제거 (수동 NMS): IoU > 0.65인 겹침에서 낮은 conf 제거
    YOLO NMS-free 모델이 중복 bbox를 출력하는 경우 대비

    Args:
        raw_boxes: YOLO detections[0].boxes 리스트
        is_plate_model: 번호판 전용 모델 여부 (ROI 필터 적용 결정)
        cfg: DETECT_CONFIG dict
    Returns:
        필터된 (bbox, conf, det) 튜플 리스트
    """
    _raw = []
    for det in raw_boxes:
        _rb = list(map(int, det.xyxy[0].tolist()))
        _rc = float(det.conf[0])
        # ★ ROI 필터 적용 (번호판 모델에서 영상 모드용)
        cx = (_rb[0] + _rb[2]) / 2
        cy = (_rb[1] + _rb[3]) / 2
        if not (cfg["ROI_X1"] <= cx <= cfg["ROI_X2"] and cfg["ROI_Y1"] <= cy <= cfg["ROI_Y2"]):
            continue
        # 번호판 비율 필터 (w/h: 0.8~7.0 — 1줄+2줄 모두 허용)
        _bw = _rb[2] - _rb[0]
        _bh = _rb[3] - _rb[1]
        if _bh <= 0 or not (0.8 <= _bw / _bh <= 7.0):
            continue
        _raw.append((_rb, _rc, det))

    # bbox 면적 큰 순 정렬
    _raw.sort(key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]), reverse=True)

    # IoU > 0.65 억제
    _keep = []
    for _rb, _rc, _det in _raw:
        _suppressed = False
        for _kb, _kc, _ in _keep:
            if bbox_iou(_rb, _kb) > cfg["NMS_IOU_THRESHOLD"]:
                _suppressed = True
                break
        if not _suppressed:
            _keep.append((_rb, _rc, _det))

    return _keep


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 탐지 필터 (plate_engine_pro.py L1494-1536 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _filter_detection(det, sx, sy, ch_full, is_plate_model, cfg):
    """개별 탐지에 크기/비율/위치/페널티 필터 적용

    Returns:
        None이면 필터됨, 아니면 dict {bbox, conf, bbox_conf_penalty, is_small_plate, aspect_type}
    """
    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
    conf = float(det.conf[0])

    # ★ COCO 모델일 때 차량 클래스만 허용
    if not is_plate_model:
        cls_id = int(det.cls[0]) if hasattr(det, 'cls') and det.cls is not None else -1
        # COCO 차량 클래스: 2=car, 3=motorcycle, 5=bus, 7=truck
        if cls_id not in {2, 3, 5, 7}:
            return None

    # 좌표 스케일링 (추론 해상도 → 원본 해상도)
    ox1, oy1 = int(x1 * sx), int(y1 * sy)
    ox2, oy2 = int(x2 * sx), int(y2 * sy)

    # ★ 최소 크기 필터
    det_w = ox2 - ox1
    det_h = oy2 - oy1
    if det_w < cfg["MIN_BBOX_WIDTH"] or det_h < cfg["MIN_BBOX_HEIGHT"]:
        return None

    # ★ bbox 가로세로 비율 검증 (엠블럼/그릴/간판 제거)
    _aspect = det_w / max(det_h, 1)
    _is_1line = 2.0 <= _aspect <= 5.5   # 1줄 번호판
    _is_2line = 0.8 <= _aspect < 2.0    # 2줄 번호판
    if not (_is_1line or _is_2line):
        return None

    # ★ bbox 위치 검증 (이미지 상단 10% / 하단 5% → 간판/노면)
    if oy1 < ch_full * 0.10:
        return None
    if oy2 > ch_full * 0.95:
        return None

    # ★ bbox 크기 기반 신뢰도 페널티
    _bbox_conf_penalty = 1.0
    _is_small_plate = False
    if det_w < 70:
        _bbox_conf_penalty = 0.60
        _is_small_plate = True
    elif det_w < 100:
        _bbox_conf_penalty = 0.85
    elif det_w < 120:
        _bbox_conf_penalty = 0.95

    aspect_type = "1line" if _is_1line else "2line"

    return {
        "bbox": [ox1, oy1, ox2, oy2],
        "conf": conf,
        "bbox_conf_penalty": _bbox_conf_penalty,
        "is_small_plate": _is_small_plate,
        "aspect_type": aspect_type,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROI 크롭 (plate_engine_pro.py L1538-1548 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _crop_roi(crop_src, bbox, cfg):
    """번호판 영역 ROI 크롭 (35%/40% 마진)

    Args:
        crop_src: 원본 해상도 프레임
        bbox: [ox1, oy1, ox2, oy2]
        cfg: DETECT_CONFIG dict
    Returns:
        roi ndarray 또는 None (빈 크롭)
    """
    ox1, oy1, ox2, oy2 = bbox
    ch_full, cw_full = crop_src.shape[:2]

    margin_x = int((ox2 - ox1) * cfg["ROI_MARGIN_X"])
    margin_y = int((oy2 - oy1) * cfg["ROI_MARGIN_Y"])
    rx1 = max(0, ox1 - margin_x)
    ry1 = max(0, oy1 - margin_y)
    rx2 = min(cw_full, ox2 + margin_x)
    ry2 = min(ch_full, oy2 + margin_y)
    roi = crop_src[ry1:ry2, rx1:rx2]

    if roi.size == 0:
        return None
    return roi


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 워커 메인 루프 (신규 코드)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def yolo_worker_loop(frame_queue: Queue, result_queue: Queue, cmd_queue: Queue):
    """YOLO 워커 프로세스 메인 루프

    frame_queue에서 프레임을 꺼내 YOLO 추론 → NMS → 필터 → ROI 크롭 후
    result_queue에 결과를 넣는다.

    cmd_queue에서 CMD_STOP 수신 시 정리 후 종료.
    """
    print("[CMD5-YOLO] 워커 프로세스 시작")
    cfg = DETECT_CONFIG

    # 모델 로드 (자식 프로세스에서 1회)
    # 기존 엔진과 동일: _resolve_plate_model() 우선, 없으면 _load_best_model() 폴백
    from ultralytics import YOLO
    model_path = _resolve_plate_model()
    if model_path is not None:
        print(f"[CMD5-YOLO] 모델 로드: {model_path}")
        model = YOLO(str(model_path))
    else:
        model = _load_best_model()

    # 번호판 전용 모델 여부 판별 (COCO 모델이면 차량 클래스 필터 적용)
    _names = model.names or {}
    _name_vals = [str(v).lower() for v in _names.values()]
    is_plate_model = any(
        kw in n for n in _name_vals
        for kw in ("plate", "license", "번호판")
    ) or (model_path is not None and "plate" in str(model_path).lower())
    _mtype = "번호판 전용" if is_plate_model else "범용(COCO)"
    print(f"[CMD5-YOLO] 모델 유형: {_mtype}")

    while True:
        # 제어 명령 확인 (논블로킹)
        try:
            cmd = cmd_queue.get_nowait()
            if cmd == CMD_STOP:
                print("[CMD5-YOLO] STOP 수신 → 종료")
                break
        except Exception:
            pass

        # 프레임 가져오기 (블로킹, 타임아웃 0.1초)
        try:
            packet = frame_queue.get(timeout=0.1)
        except Exception:
            continue

        try:
            frame = packet["frame"]
            full_frame = packet.get("full_frame")
            frame_id = packet["frame_id"]

            crop_src = full_frame if full_frame is not None else frame
            ch_full, cw_full = crop_src.shape[:2]
            ch_det, cw_det = frame.shape[:2]
            sx = cw_full / cw_det
            sy = ch_full / ch_det

            # YOLO 추론
            _fh, _fw = frame.shape[:2]
            _imgsz = _calc_imgsz(_fw)

            t0 = time.perf_counter()
            detections = model(frame, conf=cfg["DETECT_CONF"], imgsz=_imgsz, verbose=False, max_det=3)
            inference_ms = (time.perf_counter() - t0) * 1000

            # 디버그: 탐지 개수 (60프레임마다)
            _raw_count = len(detections[0].boxes) if detections and detections[0].boxes is not None else 0

            # 수동 NMS
            _keep_dets = _manual_nms(detections[0].boxes, is_plate_model, cfg)

            # NMS 후 개수 (디버그용)

            # 필터 + ROI 크롭
            det_results = []
            for _, _, det in _keep_dets:
                info = _filter_detection(det, sx, sy, ch_full, is_plate_model, cfg)
                if info is None:
                    continue

                roi = _crop_roi(crop_src, info["bbox"], cfg)
                if roi is None:
                    continue

                det_results.append({
                    "bbox": info["bbox"],
                    "conf": info["conf"],
                    "roi": roi,
                    "bbox_conf_penalty": info["bbox_conf_penalty"],
                    "is_small_plate": info["is_small_plate"],
                    "aspect_type": info["aspect_type"],
                })

            # 주기적 로그 (60프레임마다)
            if frame_id % 60 == 0:
                print(f"[CMD5] frame={frame_id} raw={_raw_count} "
                      f"nms={len(_keep_dets)} final={len(det_results)} "
                      f"{inference_ms:.0f}ms", flush=True)

            # 결과 전송
            result = {
                "frame_id": frame_id,
                "detections": det_results,
                "inference_ms": inference_ms,
                "timestamp": time.time(),
            }
            result_queue.put(result)

        except Exception as e:
            print(f"[CMD5-YOLO] 추론 오류: {e}")
            traceback.print_exc()
            # 오류 발생해도 빈 결과 전송 (메인 프로세스 블로킹 방지)
            result_queue.put({
                "frame_id": packet.get("frame_id", -1),
                "detections": [],
                "inference_ms": 0.0,
                "timestamp": time.time(),
            })

    print("[CMD5-YOLO] 워커 프로세스 종료")
