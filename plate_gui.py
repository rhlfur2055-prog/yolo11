"""
plate_gui.py - 실시간 번호판 인식 GUI v3.0 (2단계 파이프라인)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tkinter + OpenCV 하이브리드 데스크톱 애플리케이션.
실행 즉시 영상을 자동 재생하면서 번호판을 실시간 탐지.

★ v3.0 핵심: 1모델 2단계 파이프라인
  Phase 1: YOLO 탐지 즉시 → 빨간 박스 + "탐지중..." (~50ms)
  Phase 2: OCR 완료 후   → 초록 박스 + "서울바9203 (72%)"

사용법:
    python plate_gui.py                         # 기본 영상 자동 실행
    python plate_gui.py video.mp4               # 지정 영상 자동 실행
"""

from __future__ import annotations

import os
import sys
import time
import queue
import argparse
import threading

# Windows 콘솔 한글 깨짐 방지
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import Optional

import cv2
import numpy as np

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageDraw, ImageFont, ImageTk

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VIDEO_DISPLAY_W = 960
VIDEO_DISPLAY_H = 540
SIDE_PANEL_W = 300
REFRESH_MS = 33  # ~30 FPS display
DETECTION_SKIP = 1  # 매 프레임 탐지 (즉시 인식)

FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
FONT_PATH_FALLBACK = "C:/Windows/Fonts/malgun.ttf"

# 다크 테마 색상
C_BG = "#0f1117"
C_PANEL = "#1a1d27"
C_SURFACE = "#252836"
C_BORDER = "#2d3148"
C_TEXT = "#e0e0e0"
C_DIM = "#888888"
C_ACCENT = "#3d5afe"
C_GREEN = "#00e676"
C_ORANGE = "#ff9800"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bbox 검증 — YOLO 오탐지 필터링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_bbox(bbox: list, frame_shape: tuple, confidence: float) -> bool:
    """YOLO bbox가 번호판으로 유효한지 검증.
    엠블럼, 그릴, 프레임 가장자리 등 오탐지를 필터링한다."""
    if len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox[:4]
    w = x2 - x1
    h = y2 - y1

    # 가로세로 비율 검증 (다양한 번호판 형태 대응: 0.5~6.0)
    aspect_ratio = w / max(h, 1)
    if not (0.5 <= aspect_ratio <= 6.0):
        return False

    # 최소 크기
    if w < 45 or h < 15:
        return False

    # 위치 필터 — 상단 5%, 하단 5% 제외 (CCTV 각도 대응)
    frame_h = frame_shape[0]
    if y1 < frame_h * 0.05 or y2 > frame_h * 0.95:
        return False

    # bbox 크기별 신뢰도 차등 적용
    # ★ PlateEnginePro의 앙상블 투표 신뢰도는 0.5~0.85 범위가 정상
    #   (다중 OCR 엔진 합산 → 단일 OCR보다 낮은 개별 점수)
    #   오탐지 필터는 YOLO confidence 기반이므로 임계값을 엔진 스케일에 맞춤
    if w >= 100:
        min_conf = 0.45
    elif w >= 60:
        min_conf = 0.50
    else:
        min_conf = 0.60
    if confidence < min_conf:
        return False

    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PlateTracker — IoU 기반 차량 추적 (Ghost Detection 방지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PlateTracker:
    """IoU 기반으로 동일 차량을 추적하고, 새 차량 진입 시 이전 결과를 초기화한다.

    동작 원리:
      1. 현재 프레임 detection과 기존 track의 IoU 계산
      2. IoU >= threshold → 같은 차량 → OCR 결과 유지/업데이트
      3. IoU < threshold → 새 차량 → 이전 OCR 결과 완전 초기화
      4. TTL 초과 track → 삭제 (화면에서 사라진 차량)
    """

    # 트랙에 OCR 결과가 있을 때, 마지막 OCR 이후 이 프레임 수 이상 지나면
    # 다른 차량이 같은 위치에 들어온 것으로 판단하고 OCR 결과를 지운다.
    # ★ OCR 처리 500ms+ → 30fps에서 15~20프레임 간격 발생하므로 여유 있게 설정
    STALE_FRAME_GAP = 20

    # bbox 면적 비율이 이 범위를 벗어나면 다른 차량으로 판단
    AREA_RATIO_MIN = 0.5
    AREA_RATIO_MAX = 2.0

    # bbox 중심이 대각선 길이의 이 비율 이상 이동하면 다른 차량으로 판단
    CENTER_JUMP_RATIO = 0.5

    # OCR 결과 확인 횟수 (PlateEnginePro가 이미 consecutive=3을 요구하므로
    # GUI에서는 1회 수신 즉시 표시 — 이중 지연 방지)
    CONSECUTIVE_REQUIRED = 1

    # ── 안정화 파라미터 ──
    # bbox EMA 평활화 계수 (0=이전 유지, 1=새 값 즉시 적용)
    BBOX_SMOOTH_ALPHA = 0.8
    # OCR 확인된 트랙의 표시 유지 프레임 수 (OCR 미수신 시에도 초록 박스 유지)
    # ★ 실시간 카메라: 너무 크면 이전 차량 잔존, 너무 작으면 깜빡임
    DISPLAY_HOLD_FRAMES = 8
    # 프레임 갭 허용치 (이 이내 미감지는 차량 교체로 보지 않음)
    GAP_TOLERANCE = 5

    def __init__(self, iou_threshold: float = 0.35, max_ttl: int = 20):
        self.iou_threshold = iou_threshold
        self.max_ttl = max_ttl
        self.tracks: dict[int, dict] = {}  # track_id → track 정보
        self._next_id = 0

    @staticmethod
    def calculate_iou(box1: list, box2: list) -> float:
        """두 bbox [x1,y1,x2,y2]의 IoU를 계산한다."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def update(self, detections: list[dict], frame_idx: int) -> list[dict]:
        """현재 프레임의 detection 리스트로 tracker를 업데이트하고,
        추적 결과가 반영된 detection 리스트를 반환한다.

        - 같은 차량: 이전 OCR 결과를 유지 (새 OCR가 더 좋으면 교체)
        - 새 차량: 깨끗한 상태로 시작 (Ghost 결과 제거)
        """
        # 매칭되지 않은 detection / track 추적용
        matched_track_ids = set()
        matched_det_indices = set()
        output = []

        # 1단계: 각 detection에 대해 가장 높은 IoU를 가진 기존 track 찾기
        det_track_pairs = []
        for d_idx, det in enumerate(detections):
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            best_iou = 0.0
            best_tid = -1
            for tid, track in self.tracks.items():
                iou = self.calculate_iou(bbox, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            det_track_pairs.append((d_idx, best_tid, best_iou))

        # IoU 높은 순으로 정렬하여 greedy 매칭
        det_track_pairs.sort(key=lambda x: x[2], reverse=True)

        for d_idx, best_tid, best_iou in det_track_pairs:
            det = detections[d_idx]
            bbox = det.get("bbox", [])

            if best_iou >= self.iou_threshold and best_tid >= 0 and best_tid not in matched_track_ids:
                # ── IoU 매칭됨: 같은 위치의 차량 ──
                track = self.tracks[best_tid]

                # ★ 프레임 갭 체크: GAP_TOLERANCE 프레임 이상 미감지 후 재매칭 → 다른 차량 가능성
                #   OCR 처리 지연(500ms+)으로 detection 간격이 벌어질 수 있으므로 여유 부여
                last_matched = track.get("last_matched_frame", track["frame_idx"])
                gap = frame_idx - last_matched
                if gap > self.GAP_TOLERANCE and track.get("plate_text"):
                    track["plate_text"] = ""
                    track["confidence"] = 0
                    track["ocr_count"] = 0
                    track["det_data"] = {}
                    track["last_ocr_frame"] = 0  # display hold 즉시 해제

                # ★ bbox 면적 비율 체크: 크기가 급변하면 다른 차량
                old_bbox = track["bbox"]
                old_area = max(1, (old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1]))
                new_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                area_ratio = new_area / old_area
                if not (self.AREA_RATIO_MIN <= area_ratio <= self.AREA_RATIO_MAX):
                    track["plate_text"] = ""
                    track["confidence"] = 0
                    track["ocr_count"] = 0
                    track["det_data"] = {}
                    track["last_ocr_frame"] = 0  # display hold 즉시 해제

                # ★ bbox 중심 이동 거리 체크: 중심이 크게 점프하면 다른 차량
                old_cx = (old_bbox[0] + old_bbox[2]) / 2
                old_cy = (old_bbox[1] + old_bbox[3]) / 2
                new_cx = (bbox[0] + bbox[2]) / 2
                new_cy = (bbox[1] + bbox[3]) / 2
                old_diag = max(1, ((old_bbox[2] - old_bbox[0])**2 + (old_bbox[3] - old_bbox[1])**2) ** 0.5)
                center_dist = ((new_cx - old_cx)**2 + (new_cy - old_cy)**2) ** 0.5
                if center_dist > old_diag * self.CENTER_JUMP_RATIO and track.get("plate_text"):
                    track["plate_text"] = ""
                    track["confidence"] = 0
                    track["ocr_count"] = 0
                    track["det_data"] = {}
                    track["last_ocr_frame"] = 0  # display hold 즉시 해제

                det_text = det.get("text", "")
                det_conf = det.get("ocr_confidence", det.get("confidence", 0))
                det_valid = det.get("is_valid_plate", False)

                # ★ 마지막 OCR 이후 프레임 간격이 크면 이전 OCR 결과 폐기
                last_ocr_frame = track.get("last_ocr_frame", track["frame_idx"])
                ocr_gap = frame_idx - last_ocr_frame
                if ocr_gap > self.STALE_FRAME_GAP and track.get("plate_text"):
                    track["plate_text"] = ""
                    track["confidence"] = 0
                    track["ocr_count"] = 0
                    track["det_data"] = {}
                    track["last_ocr_frame"] = 0  # display hold 즉시 해제

                # ★ bbox EMA 평활화: 좌표 떨림 방지
                alpha = self.BBOX_SMOOTH_ALPHA
                old_b = track["bbox"]
                track["bbox"] = [
                    old_b[0] * (1 - alpha) + bbox[0] * alpha,
                    old_b[1] * (1 - alpha) + bbox[1] * alpha,
                    old_b[2] * (1 - alpha) + bbox[2] * alpha,
                    old_b[3] * (1 - alpha) + bbox[3] * alpha,
                ]
                track["ttl"] = self.max_ttl
                track["frame_idx"] = frame_idx
                track["last_matched_frame"] = frame_idx

                if det_valid and det_text:
                    # 새 OCR 결과가 있으면: 신뢰도 비교 후 교체
                    track["last_ocr_frame"] = frame_idx
                    # 같은 번호면 연속 카운트 증가, 다른 번호면 리셋
                    if det_text == track.get("plate_text", ""):
                        track["ocr_count"] = track.get("ocr_count", 0) + 1
                    else:
                        # ★ 유사 텍스트면 OCR 노이즈로 간주 (리셋 방지)
                        _old_text = track.get("plate_text", "")
                        if _old_text and self._text_similar_quick(_old_text, det_text):
                            track["ocr_count"] = track.get("ocr_count", 0) + 1
                        else:
                            track["ocr_count"] = 1
                    if det_conf >= track.get("confidence", 0):
                        track["plate_text"] = det_text
                        track["confidence"] = det_conf
                        track["det_data"] = det
                    # 연속 감지 횟수 미달이면 Phase 1로 표시
                    if track.get("ocr_count", 0) >= self.CONSECUTIVE_REQUIRED:
                        result_det = dict(track.get("det_data", det))
                        result_det["bbox"] = track["bbox"]  # 평활화된 bbox 사용
                    else:
                        result_det = det
                        result_det["is_valid_plate"] = False
                else:
                    # Phase 1 (OCR 미완료): 이전 확인 결과를 DISPLAY_HOLD_FRAMES 동안 유지
                    # → OCR 주기 사이에도 초록 박스 유지하여 떨림 방지
                    _last_ocr = track.get("last_ocr_frame", 0)
                    _hold_ok = (frame_idx - _last_ocr) <= self.DISPLAY_HOLD_FRAMES
                    if track.get("plate_text") and track.get("ocr_count", 0) >= self.CONSECUTIVE_REQUIRED and _hold_ok:
                        result_det = dict(track.get("det_data", {}))
                        result_det["bbox"] = track["bbox"]  # 평활화된 bbox 사용
                    else:
                        result_det = det

                output.append(result_det)
                matched_track_ids.add(best_tid)
                matched_det_indices.add(d_idx)
            elif d_idx not in matched_det_indices:
                # ── 새 차량: 깨끗한 상태로 track 생성 ──
                new_id = self._next_id
                self._next_id += 1
                det_text = det.get("text", "")
                det_conf = det.get("ocr_confidence", det.get("confidence", 0))
                has_ocr = det.get("is_valid_plate", False) and det_text
                self.tracks[new_id] = {
                    "bbox": bbox,
                    "plate_text": det_text if has_ocr else "",
                    "confidence": det_conf if has_ocr else 0,
                    "ttl": self.max_ttl,
                    "frame_idx": frame_idx,
                    "last_matched_frame": frame_idx,
                    "last_ocr_frame": frame_idx if has_ocr else 0,
                    "ocr_count": 1 if has_ocr else 0,
                    "det_data": det if has_ocr else {},
                }
                # 새 차량 첫 감지: consecutive 미달이면 Phase 1로 표시
                new_det = dict(det)
                if has_ocr and self.CONSECUTIVE_REQUIRED <= 1:
                    pass  # 그대로 출력
                elif has_ocr:
                    new_det["is_valid_plate"] = False
                output.append(new_det)
                matched_det_indices.add(d_idx)

        # 2단계: TTL 감소 + 만료된 track 삭제
        expired = []
        for tid, track in self.tracks.items():
            if tid not in matched_track_ids:
                track["ttl"] -= 1
                if track["ttl"] <= 0:
                    expired.append(tid)
        for tid in expired:
            del self.tracks[tid]

        # 3단계: 출력 중복 bbox NMS — 같은 영역에 Phase 1 + Phase 2가 동시에 있으면 Phase 2만 유지
        output = self._nms_output(output)

        return output

    @staticmethod
    def _text_similar_quick(t1: str, t2: str) -> bool:
        """빠른 유사도 판별: 2글자 이하 차이면 같은 번호판으로 간주.
        engine OCR 노이즈(한글 오인식, 숫자 1↔7 등)에 대한 내성 확보."""
        t1 = t1.replace(" ", "")
        t2 = t2.replace(" ", "")
        if t1 == t2:
            return True
        if not t1 or not t2:
            return False
        if abs(len(t1) - len(t2)) > 2:
            return False
        # 같은 길이: hamming 비교
        if len(t1) == len(t2):
            diff = sum(1 for a, b in zip(t1, t2) if a != b)
            return diff <= 2
        # 길이 차이 1~2: 짧은 쪽이 긴 쪽에 포함되면 유사
        short, long = (t1, t2) if len(t1) < len(t2) else (t2, t1)
        if short in long:
            return True
        # 숫자 부분만 비교 (한글 오인식 흡수)
        d1 = "".join(c for c in t1 if c.isdigit())
        d2 = "".join(c for c in t2 if c.isdigit())
        if d1 and d2 and d1 == d2:
            return True
        return False

    def _nms_output(self, output: list[dict]) -> list[dict]:
        """출력 리스트에서 겹치는 bbox 중복 제거.
        같은 영역에 Phase 2 (is_valid_plate=True)와 Phase 1이 동시에 있으면
        Phase 2만 유지하여 깜빡임 방지."""
        if len(output) <= 1:
            return output
        # Phase 2 (확정) 우선 정렬
        output.sort(key=lambda d: (d.get("is_valid_plate", False), d.get("ocr_confidence", 0)), reverse=True)
        keep = []
        suppressed = set()
        for i, det_i in enumerate(output):
            if i in suppressed:
                continue
            keep.append(det_i)
            bbox_i = det_i.get("bbox", [])
            if len(bbox_i) < 4:
                continue
            for j in range(i + 1, len(output)):
                if j in suppressed:
                    continue
                bbox_j = output[j].get("bbox", [])
                if len(bbox_j) < 4:
                    continue
                iou = self.calculate_iou(bbox_i, bbox_j)
                if iou >= 0.3:
                    # 겹치는 bbox → 낮은 우선순위(Phase 1) 억제
                    suppressed.add(j)
        return keep

    def reset(self) -> None:
        """모든 track을 초기화한다 (영상 변경 시)."""
        self.tracks.clear()
        self._next_id = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한글 텍스트 오버레이 (PIL 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        for fp in [FONT_PATH, FONT_PATH_FALLBACK]:
            try:
                _font_cache[size] = ImageFont.truetype(fp, size)
                break
            except OSError:
                continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_korean_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_size: int = 22,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """BGR 프레임 위에 한글 텍스트를 렌더링한다. frame을 복사해 반환."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    x, y = pos
    # 검은 외곽선
    rgb_color = (color[2], color[1], color[0])  # BGR -> RGB
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (0, 2), (-2, 0), (2, 0)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VideoReader 스레드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VideoReader(threading.Thread):
    """영상 프레임을 읽어 display_queue / detection_queue에 전달."""

    def __init__(
        self,
        video_path: str,
        display_queue: queue.Queue,
        detection_queue: queue.Queue,
        playing: threading.Event,
    ):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.display_queue = display_queue
        self.detection_queue = detection_queue
        self.playing = playing
        self.stop_flag = threading.Event()
        self.frame_idx = 0
        self._seek_to: Optional[int] = None
        self.lock = threading.Lock()

    def seek(self, frame_idx: int) -> None:
        with self.lock:
            self._seek_to = max(0, min(frame_idx, self.total_frames - 1))

    def stop(self) -> None:
        self.stop_flag.set()

    def run(self) -> None:
        interval = 1.0 / self.fps

        while not self.stop_flag.is_set():
            # pause 대기
            if not self.playing.wait(timeout=0.1):
                continue

            # seek 처리
            with self.lock:
                if self._seek_to is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to)
                    self.frame_idx = self._seek_to
                    self._seek_to = None

            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                # 영상 끝 → 처음으로 되감기 (반복 재생)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                continue

            self.frame_idx += 1
            ts = self.frame_idx / self.fps
            data = (self.frame_idx, frame, ts)

            # display queue (항상)
            try:
                self.display_queue.put_nowait(data)
            except queue.Full:
                try:
                    self.display_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.display_queue.put_nowait(data)
                except queue.Full:
                    pass

            # detection queue (N프레임마다)
            if self.frame_idx % DETECTION_SKIP == 0:
                try:
                    self.detection_queue.put_nowait(data)
                except queue.Full:
                    pass

            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.cap.release()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DetectionWorker 스레드 — 1모델 2단계 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pro_engine_results_to_gui(pro_results: list, frame: np.ndarray) -> list:
    """Pro/Fast/Unified 결과를 GUI 형식으로 변환."""
    out = []
    for r in pro_results:
        x1, y1, x2, y2 = r.get("bbox", [0, 0, 0, 0])
        plate_img = None
        if frame.size > 0 and len(r.get("bbox", [])) == 4:
            h, w = frame.shape[:2]
            x1_, y1_ = max(0, x1), max(0, y1)
            x2_, y2_ = min(w, x2), min(h, y2)
            if x2_ > x1_ and y2_ > y1_:
                plate_img = frame[y1_:y2_, x1_:x2_].copy()
        out.append({
            "text": r.get("plate", ""),
            "ocr_confidence": r.get("confidence", 0),
            "bbox": r.get("bbox", []),
            "is_valid_plate": bool(r.get("plate", "")),
            "plate_image": plate_img,
            "pattern_score": r.get("confidence", 0),
            # ── 강화 필드 (PlateEnginePro 출력 → GUI 전달) ──
            "confidence_level": r.get("confidence_level", ""),
            "plate_type": r.get("plate_type", ""),
            "vehicle_type": r.get("vehicle_type", ""),
            "frame_count": r.get("frame_count", 1),
        })
    return out


class DetectionWorker(threading.Thread):
    """PlateRecognizer 또는 PlateEnginePro로 번호판 인식.
    ★ 2단계: Phase1=YOLO bbox 즉시 전송 → Phase2=OCR 결과 전송."""

    def __init__(
        self,
        detection_queue: queue.Queue,
        results_queue: queue.Queue,
        recognizer_kwargs: dict,
    ):
        super().__init__(daemon=True)
        self.detection_queue = detection_queue
        self.results_queue = results_queue
        self.recognizer_kwargs = recognizer_kwargs
        self.stop_flag = threading.Event()
        self.ready = threading.Event()
        self.loading_msg = ""
        self.use_pro_engine = recognizer_kwargs.pop("use_pro_engine", False)
        self.engine_mode = recognizer_kwargs.pop("engine_mode", "pro")
        self.use_multiframe = recognizer_kwargs.pop("use_multiframe", False)
        self.recognizer = None
        self.pro_engine = None
        self.fast_engine = None

    def run(self) -> None:
        if self.use_pro_engine:
            self.loading_msg = "Loading Pro/Fast engines..."
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from plate_engine_pro import (
                    PlateEnginePro,
                    PlateEngineFast,
                    process_frame_unified,
                )
                self.pro_engine = PlateEnginePro()
                self.fast_engine = PlateEngineFast()
                self._process_frame_unified = process_frame_unified
            except Exception as e:
                self.loading_msg = f"Pro engine error: {e}"
                return
        else:
            self.loading_msg = "Loading plate model..."
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from plate_recognition_4k import PlateRecognizer
                self.recognizer = PlateRecognizer(**self.recognizer_kwargs)
            except Exception as e:
                self.loading_msg = f"Model load error: {e}"
                return
        self.loading_msg = ""
        self.ready.set()

        while not self.stop_flag.is_set():
            try:
                frame_idx, frame, ts = self.detection_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # ══════════════════════════════════════════════════
            # Phase 1: YOLO 탐지만 → 즉시 빨간 박스 표시 (~50ms)
            # ══════════════════════════════════════════════════
            if self.pro_engine is not None and hasattr(self.pro_engine, 'detect_only'):
                try:
                    yolo_only = self.pro_engine.detect_only(frame)
                    if yolo_only:
                        bbox_gui = []
                        for r in yolo_only:
                            x1, y1, x2, y2 = r.get("bbox", [0, 0, 0, 0])
                            bbox_gui.append({
                                "text": "",  # OCR 미완료 → 텍스트 없음
                                "ocr_confidence": r.get("confidence", 0),
                                "bbox": r.get("bbox", []),
                                "is_valid_plate": False,  # Phase1 = 빨간/주황 박스
                                "phase": 1,  # 1단계 표시
                            })
                        bbox_data = {
                            "frame_idx": frame_idx,
                            "timestamp": ts,
                            "results": bbox_gui,
                            "process_ms": 0,  # Phase1은 FPS 계산에서 제외
                        }
                        try:
                            self.results_queue.put_nowait(bbox_data)
                        except queue.Full:
                            pass
                except Exception:
                    pass

            # ══════════════════════════════════════════════════
            # Phase 2: 전체 OCR 처리 → 초록 박스 + 번호판 텍스트
            # ══════════════════════════════════════════════════
            t0 = time.time()
            if self.pro_engine is not None:
                if self.engine_mode in ("fast", "auto") and getattr(self.fast_engine, "available", False):
                    results_list, ms_pro, ms_fast = self._process_frame_unified(
                        frame, "CAM01",
                        engine_pro=self.pro_engine,
                        engine_fast=self.fast_engine,
                        engine_mode=self.engine_mode,
                        use_multiframe=self.use_multiframe,
                    )
                    results = _pro_engine_results_to_gui(results_list, frame)
                    elapsed_ms = max(ms_pro, ms_fast)
                    data = {
                        "frame_idx": frame_idx,
                        "timestamp": ts,
                        "results": results,
                        "process_ms": elapsed_ms,
                        "process_ms_pro": ms_pro,
                        "process_ms_fast": ms_fast,
                    }
                else:
                    pro_results = self.pro_engine.process_frame(
                        frame, "CAM01", use_multiframe=self.use_multiframe
                    )
                    results = _pro_engine_results_to_gui(pro_results, frame)
                    elapsed_ms = (time.time() - t0) * 1000
                    data = {
                        "frame_idx": frame_idx,
                        "timestamp": ts,
                        "results": results,
                        "process_ms": elapsed_ms,
                        "process_ms_pro": elapsed_ms,
                        "process_ms_fast": 0.0,
                    }
            else:
                results = self.recognizer.process_frame(frame, frame_idx)
                elapsed_ms = (time.time() - t0) * 1000
                data = {
                    "frame_idx": frame_idx,
                    "timestamp": ts,
                    "results": results,
                    "process_ms": elapsed_ms,
                }

            try:
                self.results_queue.put_nowait(data)
            except queue.Full:
                try:
                    self.results_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.results_queue.put_nowait(data)
                except queue.Full:
                    pass

    def stop(self) -> None:
        self.stop_flag.set()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 GUI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PlateGUIApp(tk.Tk):
    def __init__(self, cli_args: argparse.Namespace):
        super().__init__()
        self.cli_args = cli_args
        self.title("YOLO26 번호판 인식")
        self.configure(bg=C_BG)
        self.geometry(f"{VIDEO_DISPLAY_W}x{VIDEO_DISPLAY_H + 320}")  # 960 x 860
        self.minsize(900, 600)

        # 스레드 / 상태
        self.video_reader: Optional[VideoReader] = None
        self.detection_worker: Optional[DetectionWorker] = None
        self.display_queue: Optional[queue.Queue] = None
        self.detection_queue: Optional[queue.Queue] = None
        self.results_queue: Optional[queue.Queue] = None
        self.playing_event = threading.Event()

        # 오버레이 상태
        self._latest_detections: list[dict] = []
        self._detection_lock = threading.Lock()
        self._detection_history: list[dict] = []
        self._plate_tracker = PlateTracker(iou_threshold=0.35, max_ttl=15)
        self._process_ms = 0.0
        self._process_ms_pro = 0.0
        self._process_ms_fast = 0.0
        self._det_fps_samples: list[float] = []
        self._video_w = 0
        self._video_h = 0

        # 로깅 + 자동 저장
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        self._results_dir = os.path.join(self._script_dir, "plate_results_v3")
        os.makedirs(self._results_dir, exist_ok=True)
        self._log_path = os.path.join(self._script_dir, "gui_log.txt")
        self._saved_count = 0
        self._log("=== PlateGUI v3.0 시작 ===")

        # UI 빌드
        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # 디스플레이 루프 시작
        self.after(REFRESH_MS, self._refresh_display)

    # ─── 로깅 ───

    def _log(self, msg: str) -> None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    # ─── UI 빌드 ───

    def _build_ui(self) -> None:
        main = tk.Frame(self, bg=C_BG)
        main.pack(fill=tk.BOTH, expand=True)

        # ── 상단: 영상 재생 영역 ──
        video_frame = tk.Frame(main, bg=C_BG)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = tk.Label(video_frame, bg="#000000", anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ── 하단: Detection Log ──
        log_frame = tk.Frame(main, bg=C_PANEL, height=200)
        log_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        log_frame.pack_propagate(False)

        tk.Label(log_frame, text="Detection Log  —  인식된 번호판 (동영상 재생 시 즉시 표시)",
                 bg=C_PANEL, fg=C_DIM, font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=8, pady=(8, 4))

        header = tk.Frame(log_frame, bg=C_SURFACE)
        header.pack(fill=tk.X, padx=8, pady=0)
        tk.Label(header, text="시간", width=14, anchor=tk.W, bg=C_SURFACE, fg=C_DIM, font=("Consolas", 9)).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Label(header, text="번호판", width=18, anchor=tk.W, bg=C_SURFACE, fg=C_DIM, font=("Consolas", 9)).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Label(header, text="신뢰도", width=8, anchor=tk.W, bg=C_SURFACE, fg=C_DIM, font=("Consolas", 9)).pack(side=tk.LEFT, padx=4, pady=4)

        self.history_list = tk.Listbox(
            log_frame, bg=C_BG, fg=C_TEXT, selectbackground=C_ACCENT,
            font=("Consolas", 10), borderwidth=0, highlightthickness=0,
            activestyle="none", height=8,
        )
        self.history_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.plate_text_var = tk.StringVar(value="---")
        self.conf_var = tk.StringVar(value="")
        self.engine_choice_var = tk.StringVar(value="Pro 엔진")
        self.multiframe_var = tk.BooleanVar(value=False)
        self.api_server_process = None

        # ── 맨 하단: 컨트롤 바 ──
        bar = tk.Frame(self, bg=C_SURFACE, height=44)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        btn_frame = tk.Frame(bar, bg=C_SURFACE)
        btn_frame.pack(side=tk.LEFT, padx=6, pady=6)

        tk.Button(btn_frame, text="\u25b6 재생", command=self._on_play, bg=C_GREEN, fg="#000",
                  font=("Segoe UI", 9), relief=tk.FLAT, padx=10, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="\u23f8 일시정지", command=self._on_pause, bg=C_ORANGE, fg="#000",
                  font=("Segoe UI", 9), relief=tk.FLAT, padx=10, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="\U0001f4c2 파일 열기", command=self._on_file_open, bg=C_ACCENT, fg="white",
                  font=("Segoe UI", 9), relief=tk.FLAT, padx=10, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="\U0001f4be 저장", command=self._on_save_log, bg=C_PANEL, fg=C_TEXT,
                  font=("Segoe UI", 9), relief=tk.FLAT, padx=10, pady=4).pack(side=tk.LEFT, padx=2)

        self.api_btn = tk.Button(btn_frame, text="API 서버 (8765)", command=self._on_api_server_click,
                                  bg=C_BORDER, fg=C_TEXT, font=("Segoe UI", 9), relief=tk.FLAT, padx=8, pady=4)
        self.api_btn.pack(side=tk.LEFT, padx=2)

        self.stats_var = tk.StringVar(value="동영상 파일을 열어주세요.")
        tk.Label(bar, textvariable=self.stats_var, bg=C_SURFACE, fg=C_DIM,
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=12, pady=4, fill=tk.X, expand=True)

    def _bind_keys(self) -> None:
        self.bind("<q>", lambda e: self._on_close())
        self.bind("<Q>", lambda e: self._on_close())
        self.bind("<Escape>", lambda e: self._on_close())

    def _on_play(self) -> None:
        self.playing_event.set()
        if self.video_reader:
            self.stats_var.set("재생 중")

    def _on_pause(self) -> None:
        self.playing_event.clear()
        self.stats_var.set("일시정지")

    def _on_file_open(self) -> None:
        path = filedialog.askopenfilename(
            title="동영상 선택",
            filetypes=[("동영상", "*.mp4 *.avi *.mov *.mkv"), ("모든 파일", "*.*")],
        )
        if path:
            self._open_video(path)

    def _on_save_log(self) -> None:
        if not self._detection_history:
            messagebox.showinfo("저장", "저장할 인식 기록이 없습니다.")
            return
        path = filedialog.asksaveasfilename(
            title="인식 로그 저장", defaultextension=".json", initialdir=self._results_dir,
            filetypes=[("JSON", "*.json"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        try:
            import json
            out = []
            for h in self._detection_history:
                rec = {k: v for k, v in h.items()
                       if k not in ("plate_image", "preprocessed_image", "plate_img", "preprocessed")
                       and not isinstance(v, np.ndarray)}
                out.append(rec)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("저장 완료", f"{len(out)}건 저장: {path}")
        except Exception as e:
            messagebox.showerror("저장 오류", str(e))

    def _on_api_server_click(self) -> None:
        if self.api_server_process is not None:
            try:
                self.api_server_process.terminate()
                self.api_server_process = None
            except Exception:
                pass
            self.api_btn.config(text="API 서버 시작 (8765)")
            return
        try:
            import subprocess
            self.api_server_process = subprocess.Popen(
                [sys.executable, os.path.join(self._script_dir, "api_server.py"), "--port", "8765"],
                cwd=self._script_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.api_btn.config(text="API 서버 중지 (8765)")
        except Exception as e:
            self._log(f"API 서버 시작 실패: {e}")

    # ─── 영상 열기 (자동 시작) ───

    def _open_video(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            self.stats_var.set(f"File not found: {path}")
            return

        self._stop_threads()
        self._latest_detections = []
        self._detection_history = []
        self._plate_tracker.reset()
        # ★ 엔진 내부 트래커도 리셋 (이전 영상 트랙 잔존 방지)
        if self.detection_worker and hasattr(self.detection_worker, 'pro_engine') and self.detection_worker.pro_engine:
            self.detection_worker.pro_engine.reset_state()
        self.history_list.delete(0, tk.END)
        self.plate_text_var.set("---")
        self.conf_var.set("")

        # 큐 생성
        self.display_queue = queue.Queue(maxsize=3)
        self.detection_queue = queue.Queue(maxsize=2)
        self.results_queue = queue.Queue(maxsize=10)

        # Detection worker
        self.stats_var.set("Loading model...")
        self.update()

        choice = self.engine_choice_var.get()
        engine_mode = "pro" if "Pro" in choice else ("fast" if "Fast" in choice else "auto")
        use_pro = getattr(self.cli_args, "pro_engine", True)
        kwargs = {
            "model_size": self.cli_args.model_size,
            "confidence_threshold": self.cli_args.confidence,
            "use_sahi": False,
            "frame_skip": 1,
            "burst_frames": 1,
            "use_pro_engine": use_pro,
            "engine_mode": engine_mode,
            "use_multiframe": self.multiframe_var.get(),
        }
        self.detection_worker = DetectionWorker(
            self.detection_queue, self.results_queue, recognizer_kwargs=kwargs,
        )
        self.detection_worker.start()

        # Video reader
        self.playing_event = threading.Event()
        self.video_reader = VideoReader(
            path, self.display_queue, self.detection_queue, self.playing_event,
        )
        self._video_w = self.video_reader.width
        self._video_h = self.video_reader.height
        self.video_reader.start()

        self.title(f"YOLO26 번호판 인식 - {os.path.basename(path)} ({self._video_w}x{self._video_h})")

        # 엔진 로딩 대기
        self.stats_var.set("엔진 로딩 중... 잠시 기다려주세요")
        self.update()
        if not self.detection_worker.ready.wait(timeout=60):
            self.stats_var.set("엔진 로딩 타임아웃 - 재생 시작")
        else:
            self.stats_var.set("엔진 로딩 완료! 재생 시작")
        self.update()

        self.playing_event.set()
        self.stats_var.set(f"Auto-playing {self._video_w}x{self._video_h} @ {self.video_reader.fps:.0f}fps")

    # ─── 디스플레이 루프 (30 FPS) ───

    def _refresh_display(self) -> None:
        # 1) 인식 결과 수신 (validate_bbox + PlateTracker 적용)
        while self.results_queue is not None:
            try:
                data = self.results_queue.get_nowait()
                with self._detection_lock:
                    raw_results = data["results"]
                    frame_idx = data.get("frame_idx", 0)

                    # ── bbox 검증: 오탐지 필터링 ──
                    validated = []
                    for det in raw_results:
                        bbox = det.get("bbox", [])
                        conf = det.get("ocr_confidence", det.get("confidence", 0))
                        frame_shape = (self._video_h, self._video_w)
                        if validate_bbox(bbox, frame_shape, conf):
                            validated.append(det)

                    # ── PlateTracker: Ghost Detection 방지 ──
                    tracked = self._plate_tracker.update(validated, frame_idx)

                    # ★ 엔진/트래커 결과를 그대로 반영 (사라진 차 즉시 제거)
                    self._latest_detections = tracked

                    if data["process_ms"] > 0:
                        self._process_ms = data["process_ms"]
                        self._process_ms_pro = data.get("process_ms_pro", 0)
                        self._process_ms_fast = data.get("process_ms_fast", 0)
                        self._det_fps_samples.append(1000.0 / data["process_ms"])
                        if len(self._det_fps_samples) > 10:
                            self._det_fps_samples.pop(0)
                    ts = data.get("timestamp", 0)
                    for det in tracked:
                        text = det.get("text", "")
                        conf = det.get("ocr_confidence", det.get("confidence", 0))
                        is_valid = det.get("is_valid_plate", False)
                        # ★ 로그 표시 조건: 화면 표시(Phase 2)와 동일 — is_valid + text
                        if is_valid and text and len(text) >= 2:
                            self._add_to_history(det, ts)
            except (queue.Empty, AttributeError):
                break

        # 2) 비디오 프레임 수신 (최신 것만)
        frame_data = None
        while self.display_queue is not None:
            try:
                frame_data = self.display_queue.get_nowait()
            except (queue.Empty, AttributeError):
                break

        if frame_data is not None:
            frame_idx, frame, ts = frame_data

            with self._detection_lock:
                annotated = self._draw_overlay(frame, self._latest_detections)

            # 리사이즈 (종횡비 유지)
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            if label_w < 100:
                label_w, label_h = VIDEO_DISPLAY_W, VIDEO_DISPLAY_H
            scale = min(label_w / self._video_w, label_h / self._video_h) if self._video_w > 0 else 1.0
            disp_w = int(self._video_w * scale)
            disp_h = int(self._video_h * scale)
            if disp_w > 0 and disp_h > 0:
                display = cv2.resize(annotated, (disp_w, disp_h))
            else:
                display = annotated

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            self.video_label.configure(image=tk_img)
            self.video_label.image = tk_img

            self._update_stats(frame_idx, ts)

        # 모델 로딩 상태 체크
        if (self.detection_worker is not None
                and not self.detection_worker.ready.is_set()
                and self.detection_worker.loading_msg):
            self.stats_var.set(self.detection_worker.loading_msg)

        self.after(REFRESH_MS, self._refresh_display)

    # ─── 오버레이 그리기 (2단계) ───

    # 오버레이 최대 표시 개수 (화면 정리용)
    MAX_PHASE2_DISPLAY = 4   # OCR 확정 박스 최대 표시 수
    MAX_PHASE1_DISPLAY = 0   # 탐지중 박스 표시 안 함 (OCR 확정만 표시)

    def _draw_overlay(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """프레임 위에 OCR 확정 결과만 오버레이:
        ★ Phase 1 (탐지중) 미표시 — 확정된 번호판만 깔끔하게 표시
        ★ 라벨: '서울바9203 (72%)' — 등급 태그 제거
        ★ 최대 4개, 겹침 방지
        """
        result = frame.copy()

        # ── OCR 확정만 필터 ──
        phase2_dets = []
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            text = det.get("text") or det.get("plate", "")
            is_valid = det.get("is_valid_plate", False)
            if is_valid and text:
                phase2_dets.append(det)

        if not phase2_dets:
            return result

        # ── 신뢰도 내림차순 정렬 후 상위만 표시 ──
        phase2_dets.sort(key=lambda d: d.get("ocr_confidence", d.get("confidence", 0)), reverse=True)
        phase2_dets = phase2_dets[:self.MAX_PHASE2_DISPLAY]

        # 라벨 겹침 방지용 점유 영역 목록 [(y_top, y_bottom, x_left, x_right)]
        _label_rects = []

        def _find_label_y(x1, base_y, font_sz, label_w):
            """라벨 Y 위치를 기존 라벨과 겹치지 않게 조정"""
            y_top = base_y
            y_bot = base_y + font_sz + 6
            for _attempt in range(8):
                overlap = False
                for (ry1, ry2, rx1, rx2) in _label_rects:
                    if x1 < rx2 and (x1 + label_w) > rx1 and y_top < ry2 and y_bot > ry1:
                        overlap = True
                        y_top = ry2 + 4
                        y_bot = y_top + font_sz + 6
                        break
                if not overlap:
                    break
            _label_rects.append((y_top, y_bot, x1, x1 + label_w))
            return y_top

        for det in phase2_dets:
            bbox = det.get("bbox", [])
            x1, y1, x2, y2 = [int(v) for v in bbox]
            text = det.get("text") or det.get("plate", "")
            conf = det.get("ocr_confidence", det.get("confidence", 0))

            _clevel = det.get("confidence_level", "")
            if _clevel.startswith("HIGH"):
                color = (0, 230, 70)    # 초록 (HIGH)
            elif _clevel.startswith("MEDIUM"):
                color = (0, 200, 255)   # 주황/노랑 (MEDIUM)
            else:
                color = (0, 230, 70)    # 기본 초록
            thickness = 2
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # ★ 라벨: 번호판 + 신뢰도만 (등급 태그 제거)
            label = f"{text} ({conf:.0%})"
            font_sz = max(18, min(28, (x2 - x1) // 4))
            # 한글 포함 라벨 너비: 한글=폰트크기, ASCII=폰트크기*0.6
            est_label_w = sum(font_sz if ord(c) > 127 else int(font_sz * 0.6) for c in label)
            label_y = _find_label_y(x1, max(0, y1 - font_sz - 6), font_sz, est_label_w)
            result = draw_korean_text(result, label, (x1, label_y),
                                      font_size=font_sz, color=color)

            # 신뢰도 바 (얇게)
            if conf > 0:
                bar_w = int((x2 - x1) * conf)
                cv2.rectangle(result, (x1, y2 + 1), (x1 + bar_w, y2 + 4), color, -1)

        return result

    # ─── Detection Log ───

    def _add_to_history(self, det: dict, timestamp: float = 0) -> None:
        text = (det.get("text") or det.get("plate", "")).strip()
        if not text:
            return
        conf = det.get("ocr_confidence", det.get("confidence", 0))
        det_norm = dict(det)
        det_norm["text"] = text
        det_norm["ocr_confidence"] = conf

        for existing in self._detection_history:
            if self._text_similar(existing.get("text", ""), text):
                if conf > existing.get("ocr_confidence", 0):
                    # ★ is_valid_plate=True 보존 (False로 덮어쓰기 방지)
                    was_valid = existing.get("is_valid_plate", False)
                    existing.update(det_norm)
                    if was_valid:
                        existing["is_valid_plate"] = True
                existing["count"] = existing.get("count", 1) + 1
                existing["timestamp"] = timestamp
                self._refresh_history_list()
                return

        entry = dict(det_norm)
        entry["count"] = 1
        entry["timestamp"] = timestamp
        self._detection_history.insert(0, entry)
        self._refresh_history_list()
        self._auto_save_plate(entry)
        self.plate_text_var.set(text)
        self.conf_var.set(f"Conf: {conf:.1%}")

    def _auto_save_plate(self, det: dict) -> None:
        try:
            import json
            idx = self._saved_count
            plate_img = det.get("plate_image")
            if plate_img is not None:
                cv2.imwrite(os.path.join(self._results_dir, f"plate_{idx:04d}.png"), plate_img)
            pre_img = det.get("preprocessed_image")
            if pre_img is not None:
                cv2.imwrite(os.path.join(self._results_dir, f"plate_{idx:04d}_preprocessed.png"), pre_img)

            _EXCLUDE = {"plate_image", "preprocessed_image", "plate_img", "preprocessed", "phase"}
            det_record = {k: v for k, v in det.items() if k not in _EXCLUDE and not isinstance(v, np.ndarray)}
            det_record["index"] = idx

            results_path = os.path.join(self._results_dir, "results.json")
            existing = []
            if os.path.isfile(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(det_record)

            from plate_recognition_4k import NumpyEncoder
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            self._saved_count += 1
            self._log(f"저장 #{idx}: {det.get('text', '?')} conf={det.get('ocr_confidence', 0):.3f}")
        except Exception as e:
            self._log(f"저장 오류: {e}")

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return PlateGUIApp._levenshtein(s2, s1)
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if c1 == c2 else 1)))
            prev = curr
        return prev[-1]

    def _text_similar(self, t1: str, t2: str) -> bool:
        """같은 차량의 유사 인식 결과인지 판단 (levenshtein ≤ 2)."""
        t1 = t1.replace(" ", "")
        t2 = t2.replace(" ", "")
        if t1 == t2:
            return True
        if not t1 or not t2:
            return False
        if abs(len(t1) - len(t2)) > 2:
            return False
        return self._levenshtein(t1, t2) <= 2

    def _refresh_history_list(self) -> None:
        self.history_list.delete(0, tk.END)
        for h in self._detection_history[:50]:
            ts = h.get("timestamp", 0)
            time_str = f"{int(ts) // 60:02d}:{ts % 60:04.1f}"
            plate = (h.get("text") or h.get("plate", "")).strip()
            conf = h.get("ocr_confidence", h.get("confidence", 0))
            clevel = h.get("confidence_level", "")
            vtype = h.get("vehicle_type", "")
            if clevel.startswith("HIGH"):
                mark = "\U0001f7e2"   # Green circle
            elif clevel.startswith("MEDIUM"):
                mark = "\U0001f7e1"   # Yellow circle
            elif conf >= 0.90:
                mark = "\U0001f7e2"
            elif conf >= 0.70:
                mark = "\U0001f7e1"
            else:
                mark = "\U0001f534"   # Red circle
            # 차량 유형 약어
            _vt = ""
            if vtype == "영업용":
                _vt = " [영]"
            elif vtype == "렌터카":
                _vt = " [렌]"
            line = f"{mark} {time_str:<8} {plate:<14} {conf:.0%}{_vt}"
            self.history_list.insert(tk.END, line)

    # ─── 상태 표시 ───

    def _update_stats(self, frame_idx: int, ts: float) -> None:
        if self.video_reader is None:
            return
        total = self.video_reader.total_frames
        total_sec = total / self.video_reader.fps if self.video_reader.fps > 0 else 0
        det_fps = sum(self._det_fps_samples) / len(self._det_fps_samples) if self._det_fps_samples else 0.0
        n_det = len(self._latest_detections)
        mins, secs = divmod(int(ts), 60)
        t_mins, t_secs = divmod(int(total_sec), 60)

        speed_info = f"Proc:{self._process_ms:.0f}ms"
        if self._process_ms_pro or self._process_ms_fast:
            speed_info = f"Pro:{self._process_ms_pro:.0f}ms / Fast:{self._process_ms_fast:.0f}ms"

        status = (
            f"{mins:02d}:{secs:02d}/{t_mins:02d}:{t_secs:02d}  "
            f"F:{frame_idx}/{total}  Det:{n_det}  "
            f"DetFPS:{det_fps:.1f}  {speed_info}  "
            f"Log:{len(self._detection_history)}"
        )
        self.stats_var.set(status)

        if frame_idx % 5 == 0 and det_fps > 0:
            print(f"[F{frame_idx}] DetFPS:{det_fps:.2f} Proc:{self._process_ms:.0f}ms "
                  f"Det:{n_det} Log:{len(self._detection_history)}", flush=True)

    # ─── 종료 ───

    def _stop_threads(self) -> None:
        if self.video_reader is not None:
            self.video_reader.stop()
            self.playing_event.set()
            self.video_reader.join(timeout=2)
            self.video_reader = None
        if self.detection_worker is not None:
            self.detection_worker.stop()
            self.detection_worker.join(timeout=3)
            self.detection_worker = None

    def _on_close(self) -> None:
        self._log(f"=== PlateGUI 종료 === (저장: {self._saved_count}건, 기록: {len(self._detection_history)}건)")
        if getattr(self, "api_server_process", None) is not None:
            try:
                self.api_server_process.terminate()
                self.api_server_process = None
            except Exception:
                pass
        self._stop_threads()
        self.destroy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 엔트리포인트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_VIDEO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "temp_youtube", "plate_hd.mp4",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time Plate Recognition GUI v3.0")
    parser.add_argument("video", nargs="?", default=None, help="Video file path")
    parser.add_argument("--youtube", metavar="URL", default=None,
                        help="YouTube URL (다운로드 후 재생, yt-dlp 필요)")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"],
                        help="Plate model size (default: n)")
    parser.add_argument("--confidence", type=float, default=0.15,
                        help="Min detection confidence (default: 0.15)")
    parser.add_argument("--pro-engine", action="store_true", default=True,
                        help="Use PlateEnginePro (기본값)")
    parser.add_argument("--no-pro-engine", action="store_false", dest="pro_engine",
                        help="Use 4k recognizer instead of Pro")
    args = parser.parse_args()

    video_path = None
    if args.youtube:
        try:
            from youtube_helper import download_youtube, check_ytdlp
        except ImportError:
            print("[오류] youtube_helper.py를 찾을 수 없습니다.", flush=True)
            sys.exit(1)
        if not check_ytdlp():
            print("[오류] yt-dlp가 설치되지 않았습니다. 설치: pip install yt-dlp", flush=True)
            sys.exit(1)
        try:
            video_path = download_youtube(args.youtube)
            print(f"[YouTube] 로컬 파일로 재생: {video_path}", flush=True)
        except Exception as e:
            print(f"[오류] YouTube 다운로드 실패: {e}", flush=True)
            sys.exit(1)
    elif args.video:
        video_path = args.video
    else:
        video_path = DEFAULT_VIDEO

    app = PlateGUIApp(args)
    app.after(200, lambda: app._open_video(video_path))
    app.mainloop()


if __name__ == "__main__":
    main()
