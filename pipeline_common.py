# -*- coding: utf-8 -*-
"""
파이프라인 공통 모듈: 타입 정의, 유틸리티, 설정값
CMD 5 (YOLO 워커) ↔ CMD 12 (메인 프로세스) 간 공유
"""
import numpy as np
from typing import TypedDict, List


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IPC 큐 데이터 타입 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FramePacket(TypedDict):
    """frame_queue 아이템 (CMD 12 → CMD 5)"""
    frame: np.ndarray          # BGR 프레임 (YOLO 추론용, 축소 가능)
    full_frame: np.ndarray     # 원본 해상도 (ROI 크롭용, None이면 frame 사용)
    frame_id: int              # 프레임 번호 (순서 추적)
    camera_id: str             # "CAM01"


class DetectionItem(TypedDict):
    """개별 탐지 결과"""
    bbox: List[int]            # [x1, y1, x2, y2] full_frame 좌표계
    conf: float                # YOLO 신뢰도
    roi: np.ndarray            # 크롭된 ROI 이미지 (OCR 입력용)
    bbox_conf_penalty: float   # 크기 기반 페널티 (0.60/0.85/0.95/1.0)
    is_small_plate: bool       # det_w < 70px 여부
    aspect_type: str           # "1line" 또는 "2line"


class DetectionResult(TypedDict):
    """result_queue 아이템 (CMD 5 → CMD 12)"""
    frame_id: int              # 대응하는 프레임 번호
    detections: List[DetectionItem]
    inference_ms: float        # YOLO 추론 시간 (성능 모니터링)
    timestamp: float           # time.time()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bbox IoU 계산 (독립 함수)
# plate_engine_pro.py L1143-1151, L2457-2466에서 추출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def bbox_iou(a, b):
    """두 bbox [x1,y1,x2,y2] 간 IoU 계산"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 탐지 관련 설정값 (PlateEngineConfig에서 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETECT_CONFIG = {
    "DETECT_CONF": 0.25,         # YOLO 감지 임계값
    "ROI_X1": 300,               # ROI 필터 범위
    "ROI_X2": 800,
    "ROI_Y1": 150,
    "ROI_Y2": 700,
    "MIN_BBOX_WIDTH": 50,        # 최소 bbox 가로 px
    "MIN_BBOX_HEIGHT": 15,       # 최소 bbox 세로 px
    "TRACKER_IOU_THRESHOLD": 0.30,
    "NMS_IOU_THRESHOLD": 0.65,   # 수동 NMS IoU 임계값
    "ROI_MARGIN_X": 0.35,        # ROI 크롭 좌우 마진
    "ROI_MARGIN_Y": 0.40,        # ROI 크롭 상하 마진
    "YOLO_MODEL": "yolo11x_plate.pt",
    "YOLO_FALLBACK": "yolo26.pt",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OCR 파이프라인 IPC 타입 (CMD 12 ↔ CMD 6)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OcrRequest(TypedDict):
    """ocr_queue 아이템 (CMD 12 → CMD 6)"""
    frame_id: int                  # 프레임 번호
    det_index: int                 # 탐지 인덱스 (0, 1, 2...)
    roi: np.ndarray                # 원본 ROI (~100px, 업스케일 전)
    bbox: List[int]                # [ox1, oy1, ox2, oy2] full_frame 좌표
    roi_crop_offset: List[int]     # [rx1, ry1] 한글 크롭 좌표 계산용
    det_w: int                     # 탐지 bbox 너비
    det_h: int                     # 탐지 bbox 높이
    bbox_conf_penalty: float       # 크기 페널티 (0.60/0.85/0.95/1.0)
    is_small_plate: bool
    aspect_type: str               # "1line" / "2line"
    timestamp: float


class OcrResult(TypedDict):
    """ocr_result_queue 아이템 (CMD 6 → CMD 12)"""
    frame_id: int
    det_index: int
    best_text: str                 # 최종 인식 텍스트 ("" = 실패)
    best_conf: float               # 최종 신뢰도
    bbox: List[int]                # passthrough (트래커 매칭용)
    is_small_plate: bool
    det_w: int
    is_green_plate: bool
    plate_type: str                # "자가용", "영업용" 등
    vehicle_type: str
    plate_lines: int               # 1 or 2
    plate_color: str               # "흰색바탕_검은글씨" 등
    is_valid_format: bool
    ocr_ms: float                  # OCR 처리 시간 (ms)
    timestamp: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 제어 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CMD_STOP = "STOP"
CMD_OCR_STOP = "OCR_STOP"
