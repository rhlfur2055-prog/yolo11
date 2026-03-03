# -*- coding: utf-8 -*-
"""
CMD 5: YOLO 탐지 워커 프로세스
plate_engine_pro.py process_frame()에서 YOLO 탐지 부분만 추출하여
별도 프로세스에서 실행. multiprocessing.Queue로 IPC 통신.

추출 원본: plate_engine_pro.py L1431-1548
"""
import os
import re
import queue
import time
import traceback
import numpy as np
from pathlib import Path
from multiprocessing import Queue

from pipeline_common import bbox_iou, DETECT_CONFIG, CMD_STOP, CMD_YOLO_READY

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 번호판 형식 정규식 패턴 (한국 번호판)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_PLATE_PATTERNS = [
    re.compile(r'^\d{2}[가-힣]\d{4}$'),           # 12가3456 (신형 2자리)
    re.compile(r'^\d{3}[가-힣]\d{4}$'),           # 123가4567 (신형 3자리)
    re.compile(r'^[가-힣]{2}\d{2}[가-힣]\d{4}$'),  # 서울12가3456 (구형)
    re.compile(r'^[가-힣]{2}\d{3}[가-힣]\d{4}$'),  # 서울123가4567
    re.compile(r'^[가-힣]{3}\d{2}[가-힣]\d{4}$'),  # 충남12가3456
]


def _is_valid_plate_format(text):
    """한국 번호판 형식 패턴 매칭"""
    return any(p.match(text) for p in _PLATE_PATTERNS)


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
def _manual_nms(raw_boxes, is_plate_model, cfg, frame_shape=None):
    """겹치는 bbox 제거 (수동 NMS): IoU > 0.65인 겹침에서 낮은 conf 제거
    YOLO NMS-free 모델이 중복 bbox를 출력하는 경우 대비

    Args:
        raw_boxes: YOLO detections[0].boxes 리스트
        is_plate_model: 번호판 전용 모델 여부 (ROI 필터 적용 결정)
        cfg: DETECT_CONFIG dict
        frame_shape: (H, W, C) — ratio→pixel ROI 변환용
    Returns:
        필터된 (bbox, conf, det) 튜플 리스트
    """
    # ★ ROI: ratio(0~1) → pixel 변환
    if frame_shape is not None:
        _fh, _fw = frame_shape[:2]
        _roi_x1 = int(cfg["ROI_X1"] * _fw)
        _roi_x2 = int(cfg["ROI_X2"] * _fw)
        _roi_y1 = int(cfg["ROI_Y1"] * _fh)
        _roi_y2 = int(cfg["ROI_Y2"] * _fh)
    else:
        _roi_x1, _roi_x2 = int(cfg["ROI_X1"]), int(cfg["ROI_X2"])
        _roi_y1, _roi_y2 = int(cfg["ROI_Y1"]), int(cfg["ROI_Y2"])

    _raw = []
    for det in raw_boxes:
        _rb = list(map(int, det.xyxy[0].tolist()))
        _rc = float(det.conf[0])
        # ★ ROI 필터 적용 (ratio→pixel 변환 완료)
        cx = (_rb[0] + _rb[2]) / 2
        cy = (_rb[1] + _rb[3]) / 2
        if not (_roi_x1 <= cx <= _roi_x2 and _roi_y1 <= cy <= _roi_y2):
            continue
        # 번호판 비율 필터 (w/h: 0.8~7.0 — 1줄+2줄 모두 허용)
        _bw = _rb[2] - _rb[0]
        _bh = _rb[3] - _rb[1]
        if _bh <= 0 or not (0.8 <= _bw / _bh <= 7.0):
            continue
        _raw.append((_rb, _rc, det))

    # ★ 중앙거리 우선 정렬 (NMS 전에 적용 — 중앙 bbox가 NMS에서 먼저 선택됨)
    _max_det = cfg.get("MAX_DET_DISPLAY", 99)
    if frame_shape is not None and _max_det <= 2:
        _fw = frame_shape[1]
        _center_x = _fw / 2
        # 중앙 가까운 순 정렬 → NMS에서 중앙 bbox 우선 유지
        _raw.sort(key=lambda t: abs((t[0][0] + t[0][2]) / 2 - _center_x))
    else:
        # bbox 면적 큰 순 정렬 (기존 동작)
        _raw.sort(key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]), reverse=True)

    # IoU > 0.65 억제 (중앙 bbox가 먼저 _keep에 들어감)
    _keep = []
    for _rb, _rc, _det in _raw:
        _suppressed = False
        for _kb, _kc, _ in _keep:
            if bbox_iou(_rb, _kb) > cfg["NMS_IOU_THRESHOLD"]:
                _suppressed = True
                break
        if not _suppressed:
            _keep.append((_rb, _rc, _det))

    # MAX_DET_DISPLAY 제한
    if len(_keep) > _max_det:
        _keep = _keep[:_max_det]

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
# Fast OCR용 ROI 업스케일 (cmd6 _upscale_roi 동일 로직)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _parse_paddle_result(paddle_result, validator):
    """PaddleOCR 결과 파싱 → (text, conf) 또는 ("", 0.0)"""
    if not paddle_result or not paddle_result[0]:
        return "", 0.0
    lines = sorted(paddle_result[0], key=lambda l: l[0][0][1])
    texts = [l[1][0] for l in lines]
    confs = [l[1][1] for l in lines]
    if not texts:
        return "", 0.0
    raw_text = "".join(texts)
    raw_conf = float(np.mean(confs))
    cleaned = validator.clean_ocr_text(raw_text)
    if not validator.is_valid_length(cleaned):
        return "", 0.0
    is_valid, final = validator.validate(cleaned)
    if is_valid:
        return final, raw_conf
    return "", 0.0


def _upscale_roi_fast(roi, cv2=None):
    """ROI를 500px 목표로 업스케일 + 선명화 + 흰색 패딩
    cv2는 워커 프로세스 내부에서 전달 (모듈 레벨 import 시 DLL 충돌 방지)
    """
    if cv2 is None:
        import cv2
    roi_h, roi_w = roi.shape[:2]
    target_w = 500
    if roi_w < target_w:
        scale = target_w / roi_w
        if roi_w < 60:
            scale = max(scale, 9.0)
        elif roi_w < 120:
            scale = max(scale, 4.0)
    else:
        scale = 1.0

    if scale > 1.0:
        _interp = cv2.INTER_LANCZOS4 if roi_w < 80 else cv2.INTER_CUBIC
        roi_up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=_interp)
        if roi_w < 80:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        else:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        roi_up = cv2.filter2D(roi_up, -1, kernel)
    else:
        roi_up = roi

    pad = max(10, int(roi_up.shape[0] * 0.15))
    roi_up = cv2.copyMakeBorder(
        roi_up, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    return roi_up


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fast OCR 엔진 초기화 (cmd6 _init_ocr_engines PaddleOCR 부분 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _init_fast_ocr():
    """Fast OCR용 PaddleOCR 엔진 초기화 (프로세스 시작 시 1회)

    Returns:
        PaddleOCR 인스턴스 또는 None (초기화 실패 시)
    """
    try:
        from paddleocr import PaddleOCR
    except Exception:
        print("[CMD5] PaddleOCR 미설치 — Fast OCR 비활성")
        return None

    paddle_kwargs = dict(lang="korean", use_angle_cls=True, show_log=False, use_gpu=False)
    # Windows 한글 경로 우회: 영문 경로에 모델이 있으면 직접 지정
    _paddle_model_root = None
    for _pdir in [
        Path("C:/paddle_models/.paddleocr/whl"),
        Path("C:/tools/paddleocr_models"),
    ]:
        if _pdir.exists():
            _paddle_model_root = _pdir
            break
    if _paddle_model_root is not None:
        _det = _paddle_model_root / "det/ml/Multilingual_PP-OCRv3_det_infer"
        _rec = _paddle_model_root / "rec/korean/korean_PP-OCRv4_rec_infer"
        _cls = _paddle_model_root / "cls/ch_ppocr_mobile_v2.0_cls_infer"
        if _det.exists():
            paddle_kwargs["det_model_dir"] = str(_det)
        if _rec.exists():
            paddle_kwargs["rec_model_dir"] = str(_rec)
        if _cls.exists():
            paddle_kwargs["cls_model_dir"] = str(_cls)
    try:
        engine = PaddleOCR(**paddle_kwargs)
        return engine
    except TypeError:
        paddle_kwargs.pop("show_log", None)
        try:
            engine = PaddleOCR(**paddle_kwargs)
            return engine
        except Exception as e:
            print(f"[CMD5] PaddleOCR 초기화 실패: {e}")
            return None
    except Exception as e:
        print(f"[CMD5] PaddleOCR 초기화 실패: {e}")
        return None


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

    # ★ YOLO warm-up: 첫 추론은 GPU/CPU 초기화로 느림 → dummy로 선행
    _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ready_t0 = time.perf_counter()
    model(_dummy, conf=0.25, imgsz=640, verbose=False)
    _warmup_ms = (time.perf_counter() - _ready_t0) * 1000
    print(f"[CMD5-YOLO] warm-up 완료: {_warmup_ms:.0f}ms")

    # ★ ready 신호 전송 → pipeline_stage2가 영상 재생 시작
    result_queue.put({"cmd": CMD_YOLO_READY, "warmup_ms": _warmup_ms})
    print(f"[CMD5-YOLO] ready 신호 전송 완료")

    # ★ Fast OCR 초기화: PaddleOCR + PlateValidator (프로세스 내 1회)
    # cv2/PaddleOCR/PlateValidator 모두 워커 프로세스 내부에서만 import
    # (모듈 레벨에서 import하면 DLL 검색 경로 오염 → torch shm.dll 로딩 실패)
    import cv2 as _cv2
    from plate_engine_pro import PlateValidator as _PlateValidator
    fast_ocr_engine = _init_fast_ocr()
    fast_validator = _PlateValidator() if fast_ocr_engine is not None else None
    if fast_ocr_engine is not None:
        _focr_t0 = time.perf_counter()
        _dummy_ocr = np.ones((100, 300, 3), dtype=np.uint8) * 255
        try:
            fast_ocr_engine.ocr(_dummy_ocr, cls=True)
        except Exception:
            pass
        _focr_ms = (time.perf_counter() - _focr_t0) * 1000
        print(f"[CMD5] Fast OCR(PaddleOCR) 준비 완료 — 워밍업 {_focr_ms:.0f}ms")
    else:
        print("[CMD5] Fast OCR 비활성 — PaddleOCR 없음")

    while True:
        # 제어 명령 확인 (논블로킹)
        try:
            cmd = cmd_queue.get_nowait()
            if cmd == CMD_STOP:
                print("[CMD5-YOLO] STOP 수신 → 종료")
                break
        except queue.Empty:
            pass

        # 프레임 가져오기 (블로킹, 타임아웃 0.1초)
        try:
            packet = frame_queue.get(timeout=0.1)
        except queue.Empty:
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
            detections = model(frame, conf=cfg["DETECT_CONF"], imgsz=_imgsz, verbose=False, max_det=cfg["MAX_DET"])
            inference_ms = (time.perf_counter() - t0) * 1000

            # 디버그: 탐지 개수 (60프레임마다)
            _raw_count = len(detections[0].boxes) if detections and detections[0].boxes is not None else 0

            # 수동 NMS (ratio→pixel ROI 변환을 위해 frame_shape 전달)
            _keep_dets = _manual_nms(detections[0].boxes, is_plate_model, cfg, frame_shape=frame.shape)

            # NMS 후 개수 (디버그용)

            # 필터 + ROI 크롭 + Fast OCR
            det_results = []
            for _, _, det in _keep_dets:
                info = _filter_detection(det, sx, sy, ch_full, is_plate_model, cfg)
                if info is None:
                    continue

                roi = _crop_roi(crop_src, info["bbox"], cfg)
                if roi is None:
                    continue

                # ★ Fast OCR: PaddleOCR 2회 (original + CLAHE+Unsharp) 합의 체크
                # + Laplacian Variance 블러 프레임 스킵
                fast_text = ""
                fast_conf = 0.0
                fast_ms = 0.0
                _roi_h_f, _roi_w_f = roi.shape[:2]
                if fast_ocr_engine is not None and _roi_w_f >= 40 and _roi_h_f >= 15:
                    _ft0 = time.perf_counter()
                    try:
                        # ★ 블러 프레임 스킵: Laplacian Variance < 50이면 너무 흐림
                        _blur_gray = _cv2.cvtColor(roi, _cv2.COLOR_BGR2GRAY)
                        _lap_var = _cv2.Laplacian(_blur_gray, _cv2.CV_64F).var()
                        if _lap_var < 50.0:
                            # 흐릿한 ROI -- OCR 스킵 (오인식 방지)
                            fast_ms = (time.perf_counter() - _ft0) * 1000
                        else:
                            _up_roi = _upscale_roi_fast(roi, cv2=_cv2)

                            # 1차: original
                            _result1 = fast_ocr_engine.ocr(_up_roi, cls=True)
                            _text1, _conf1 = _parse_paddle_result(_result1, fast_validator)

                            # ★ 1차 성공 시 2차 스킵 (속도 2배 향상)
                            if _text1:
                                fast_text = _text1
                                fast_conf = _conf1
                            else:
                                # 2차: CLAHE + Unsharp Masking (1차 실패 시에만)
                                _gray = _cv2.cvtColor(_up_roi, _cv2.COLOR_BGR2GRAY)
                                _clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                _clahe_img = _clahe.apply(_gray)
                                _blur = _cv2.GaussianBlur(_clahe_img, (0, 0), 3)
                                _sharp = _cv2.addWeighted(_clahe_img, 1.5, _blur, -0.5, 0)
                                _sharp_bgr = _cv2.cvtColor(_sharp, _cv2.COLOR_GRAY2BGR)
                                _result2 = fast_ocr_engine.ocr(_sharp_bgr, cls=True)
                                _text2, _conf2 = _parse_paddle_result(_result2, fast_validator)
                                if _text2:
                                    fast_text = _text2
                                    fast_conf = _conf2
                            fast_ms = (time.perf_counter() - _ft0) * 1000
                    except Exception:
                        pass
                    if fast_ms == 0.0:
                        fast_ms = (time.perf_counter() - _ft0) * 1000

                if fast_text:
                    print(f"[CMD5] frame={frame_id} fast_ocr='{fast_text}' "
                          f"conf={fast_conf:.2f} time={fast_ms:.1f}ms", flush=True)

                det_results.append({
                    "bbox": info["bbox"],
                    "conf": info["conf"],
                    "roi": roi,
                    "bbox_conf_penalty": info["bbox_conf_penalty"],
                    "is_small_plate": info["is_small_plate"],
                    "aspect_type": info["aspect_type"],
                    "fast_ocr_text": fast_text,
                    "fast_ocr_conf": fast_conf,
                    "fast_ocr_ms": fast_ms,
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
