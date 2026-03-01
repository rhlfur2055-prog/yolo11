# -*- coding: utf-8 -*-
"""
CMD 12: 메인 프로세스 - 영상 읽기 + YOLO 결과 + OCR 결과 + 표시
2단계 파이프라인: YOLO(CMD5) + OCR(CMD6)를 별도 프로세스로 분리

사용법:
  python pipeline_stage2.py                          # 기본 영상
  python pipeline_stage2.py --input movie/hiway.mp4  # 동영상 파일
  python pipeline_stage2.py --input 0                # 웹캠
"""
import argparse
import os
import time
import sys
import cv2
import numpy as np
from multiprocessing import Process, Queue, freeze_support
from queue import Full, Empty
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

from pipeline_common import (
    CMD_STOP, CMD_OCR_STOP, DETECT_CONFIG,
)
from cmd5_yolo_worker import yolo_worker_loop
from cmd6_ocr_worker import ocr_worker_loop

# plate_engine_pro.py에서 트래커 import
from plate_engine_pro import PlateTracker, PlateValidator, PlateDatabase

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한글 텍스트 렌더링 (plate_gui.py에서 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
FONT_PATH_FALLBACK = "C:/Windows/Fonts/malgun.ttf"
_font_cache = {}


def _get_font(size):
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


def draw_korean_text(frame, text, pos, font_size=22, color=(0, 255, 0)):
    """BGR 프레임 위에 한글 텍스트를 렌더링한다. frame을 in-place 수정."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    x, y = pos
    rgb_color = (color[2], color[1], color[0])  # BGR -> RGB
    # 검은 외곽선
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (0, 2), (-2, 0), (2, 0)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=rgb_color)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(frame, result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. OCR 스킵 판단 (plate_engine_pro.py L1550-1596 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _should_skip_ocr(tracker, bbox, consecutive_required):
    """트래커 기반 OCR 스킵 판단

    Args:
        tracker: PlateTracker 인스턴스
        bbox: [ox1, oy1, ox2, oy2]
        consecutive_required: 표시에 필요한 연속 프레임 수

    Returns:
        skip: bool (True → OCR 없이 캐시 사용)
        matched_trk: 매칭된 트랙 dict
        is_new_track: bool
        cached_result: dict 또는 None (스킵 시 표시용)
    """
    matched_trk, is_new_track = tracker.match(bbox)
    skip_ocr = False

    if not is_new_track and matched_trk["texts"]:
        _pre_gap = matched_trk.get("_pre_gap", 999)
        _pre_top_text = max(matched_trk["texts"], key=matched_trk["texts"].get)
        _pre_top_votes = matched_trk["texts"][_pre_top_text]
        _pre_detect_cnt = matched_trk.get("_detect_count", 0)
        # 직전 프레임 연속(gap≤1) & 3프레임+ 감지 & 3+표 → OCR 스킵
        if _pre_gap <= 1 and _pre_detect_cnt >= 3 and _pre_top_votes >= 3:
            skip_ocr = True

    cached_result = None
    if skip_ocr:
        _pre_top_text = max(matched_trk["texts"], key=matched_trk["texts"].get)
        _pre_conf = matched_trk.get("best_conf", 0.5)
        _pre_detect_cnt = matched_trk.get("_detect_count", 0)
        _show = (matched_trk["consecutive"] >= consecutive_required
                 or _pre_detect_cnt >= consecutive_required)
        if _show:
            _adj_conf = min(_pre_conf + 0.10, 1.0)
            ox1, oy1, ox2, oy2 = bbox
            _bbox_w = ox2 - ox1
            _bbox_h = oy2 - oy1
            cached_result = {
                "plate": _pre_top_text,
                "confidence": _pre_conf,
                "bbox": bbox,
                "is_alert": False,
                "alert_info": None,
                "plate_number": _pre_top_text,
                "confidence_level": "(추적)",
                "plate_type": "",
                "vehicle_type": "",
                "plate_lines": 1 if _bbox_w / max(_bbox_h, 1) > 2.5 else 2,
                "plate_color": "흰색바탕_검은글씨",
                "bbox_area": _bbox_w * _bbox_h,
                "frame_count": _pre_top_votes,
                "is_valid_format": True,
                "rejection_reason": None,
            }

    return skip_ocr, matched_trk, is_new_track, cached_result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. 트래커 업데이트 (plate_engine_pro.py L2292-2444 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _update_tracker_with_ocr(tracker, ocr_result, matched_trk, is_new_track,
                             consecutive_required, validator, db):
    """OCR 결과로 트래커 업데이트 → 표시용 result dict 생성

    Args:
        tracker: PlateTracker
        ocr_result: OcrResult dict
        matched_trk: 매칭된 트랙 dict
        is_new_track: bool
        consecutive_required: int
        validator: PlateValidator
        db: PlateDatabase

    Returns:
        result dict 또는 None (표시 불필요)
    """
    best_text = ocr_result["best_text"]
    best_conf = ocr_result["best_conf"]
    bbox = ocr_result["bbox"]
    is_small_plate = ocr_result["is_small_plate"]
    det_w = ocr_result["det_w"]

    if not best_text:
        return None

    # 텍스트 기반 트랙 병합 (L2298-2324)
    if is_new_track and best_text:
        _new_cx = (bbox[0] + bbox[2]) / 2
        _new_cy = (bbox[1] + bbox[3]) / 2
        for _exist_trk in tracker.tracks:
            if _exist_trk is matched_trk:
                continue
            if _exist_trk["texts"]:
                _exist_top = max(_exist_trk["texts"], key=_exist_trk["texts"].get)
                if _exist_top == best_text:
                    _e_bbox = _exist_trk["bbox"]
                    _exist_cx = (_e_bbox[0] + _e_bbox[2]) / 2
                    _exist_cy = (_e_bbox[1] + _e_bbox[3]) / 2
                    _cdist = (((_new_cx - _exist_cx) ** 2) + ((_new_cy - _exist_cy) ** 2)) ** 0.5
                    if _cdist >= 200:
                        continue
                    matched_trk["texts"] = _exist_trk["texts"]
                    matched_trk["best_conf"] = max(matched_trk["best_conf"], _exist_trk["best_conf"])
                    matched_trk["_detect_count"] = _exist_trk.get("_detect_count", 0) + 1
                    matched_trk["consecutive"] = _exist_trk["consecutive"] + 1
                    matched_trk["recorded"] = _exist_trk["recorded"]
                    _exist_trk["texts"] = defaultdict(int)
                    _exist_trk["last_frame"] = 0
                    is_new_track = False
                    break

    # 투표 decay (L2326-2338)
    if matched_trk["texts"] and not is_new_track:
        _cur_top = max(matched_trk["texts"], key=matched_trk["texts"].get)
        if best_text != _cur_top:
            _cur_top_from_small = matched_trk.get("_last_small_plate", False)
            if is_small_plate and _cur_top_from_small:
                pass
            else:
                matched_trk["texts"].clear()

    # 투표 가중치 (L2340-2350)
    if is_small_plate:
        _vote_weight = 1
    elif det_w >= 120:
        _vote_weight = 3
    elif det_w >= 100:
        _vote_weight = 2
    else:
        _vote_weight = 1
    matched_trk["texts"][best_text] += _vote_weight
    matched_trk["_last_small_plate"] = is_small_plate

    top_text = max(matched_trk["texts"], key=matched_trk["texts"].get)
    top_conf = max(best_conf, matched_trk.get("best_conf", 0))
    matched_trk["best_conf"] = top_conf

    # 표시 판단 (L2364-2369)
    _detect_count = matched_trk.get("_detect_count", matched_trk["consecutive"])
    _show = (matched_trk["consecutive"] >= consecutive_required
             or _detect_count >= consecutive_required)
    if not _show:
        return None

    # DB 기록 (L2370-2380)
    is_alert, alert_info = 0, None
    if not matched_trk["recorded"]:
        matched_trk["recorded"] = True
        try:
            is_alert, alert_info = db.record_plate(top_text, top_conf, "CAM01")
        except Exception:
            pass

    # 프레임 보너스 (L2384-2395)
    _frame_count = matched_trk["texts"].get(top_text, 1)
    if _frame_count >= 5:
        _frame_bonus = 0.10
    elif _frame_count >= 3:
        _frame_bonus = 0.05
    elif _frame_count >= 2:
        _frame_bonus = 0.02
    else:
        _frame_bonus = 0
    _adj_conf = min(top_conf + _frame_bonus, 1.0)
    _conf_level = validator.get_confidence_level(_adj_conf)
    if is_small_plate:
        _conf_level += "(원거리)"

    ox1, oy1, ox2, oy2 = bbox
    _bbox_w = ox2 - ox1
    _bbox_h = oy2 - oy1

    return {
        "plate": top_text,
        "confidence": top_conf,
        "bbox": bbox,
        "is_alert": bool(is_alert),
        "alert_info": alert_info,
        "plate_number": top_text,
        "confidence_level": _conf_level,
        "plate_type": ocr_result.get("plate_type", ""),
        "vehicle_type": ocr_result.get("vehicle_type", ""),
        "plate_lines": ocr_result.get("plate_lines", 1),
        "plate_color": ocr_result.get("plate_color", "흰색바탕_검은글씨"),
        "bbox_area": _bbox_w * _bbox_h,
        "frame_count": _frame_count,
        "is_valid_format": True,
        "rejection_reason": None,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. 결과 그리기 (확장)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_results(frame, results, inference_ms, display_fps, ocr_count=0):
    """bbox + 번호판 텍스트 + 신뢰도 표시 (한글 1회 PIL 변환)

    Args:
        frame: 표시할 프레임 (in-place 수정)
        results: 표시용 result dict 리스트
        inference_ms: YOLO 추론 시간
        display_fps: 표시 FPS
        ocr_count: 누적 OCR 처리 수
    """
    # 상단 정보 텍스트 (영문만 → cv2.putText OK)
    info_text = f"FPS: {display_fps:.1f} | YOLO: {inference_ms:.0f}ms | Det: {len(results)} | OCR: {ocr_count}"
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # bbox는 cv2로 먼저 그리고, 한글 텍스트는 모아서 1회 PIL 변환
    korean_labels = []  # (label, x, y, font_sz, color_bgr)

    for i, res in enumerate(results):
        bbox = res["bbox"]
        text = res.get("plate", "")
        conf = res.get("confidence", 0)
        x1, y1, x2, y2 = bbox

        if text:
            color = (0, 255, 0)  # 초록: 인식 성공
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{text} {conf:.2f}"
            font_sz = max(18, min(28, (x2 - x1) // 4))
            label_y = max(0, y1 - font_sz - 6)
            korean_labels.append((label, x1 + 2, label_y, font_sz, color))
        else:
            color = (0, 165, 255)  # 주황: 탐지만 (OCR 미완)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # 한글 텍스트가 있으면 1회 PIL 변환으로 모두 그리기
    if korean_labels:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for label, x, y, font_sz, color_bgr in korean_labels:
            font = _get_font(font_sz)
            rgb_color = (color_bgr[2], color_bgr[1], color_bgr[0])
            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                draw.text((x + dx, y + dy), label, font=font, fill=(0, 0, 0))
            draw.text((x, y), label, font=font, fill=rgb_color)
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        np.copyto(frame, result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 영상 열기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def open_video(source):
    """VideoCapture 열기"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[CMD12] 영상 열기 실패: {source}")
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[CMD12] 영상: {source} ({w}x{h}, {fps:.1f}fps, {total}프레임)")
    return cap


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D. 메인 표시 루프 (비동기 OCR)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def display_loop(cap, frame_queue, detect_queue, ocr_queue,
                 ocr_result_queue, cmd_queue_yolo, cmd_queue_ocr):
    """메인 루프: 프레임 → YOLO → OCR 스킵 판단 → OCR 요청/결과 → 표시"""
    cfg = DETECT_CONFIG

    # 트래커 + 검증기 + DB
    tracker = PlateTracker(
        iou_threshold=cfg["TRACKER_IOU_THRESHOLD"],
        ttl_frames=15,
    )
    validator = PlateValidator()
    db = PlateDatabase()

    _env = os.environ.get("PLATE_CONSECUTIVE_FRAMES")
    consecutive_required = int(_env) if (_env and _env.isdigit()) else 1

    frame_id = 0
    last_det_result = None
    fps_counter = 0
    fps_time = time.time()
    display_fps = 0.0
    total_ocr_count = 0

    # _pending_ocr: (frame_id, det_index) → (matched_trk, is_new_track, det_info)
    _pending_ocr = {}
    # 보류 결과 최대 보관 시간 (초) — OCR 엔진 초기 로딩 + 처리 시간 고려
    _PENDING_TIMEOUT = 60.0

    # 현재 표시할 결과 목록 (각 항목에 "_last_seen" 프레임 번호 저장)
    display_results = []
    _DISPLAY_TTL = 45  # 45프레임(~1.5초) 미감지 시 표시 제거

    # 영상 정보
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / video_fps  # 프레임 간격 (초)
    _last_frame_time = time.time()

    # cv2 창 초기화 (WINDOW_AUTOSIZE: 프레임 크기에 맞춤)
    _win_name = "Pipeline Stage 2 - YOLO + OCR"
    cv2.namedWindow(_win_name, cv2.WINDOW_AUTOSIZE)

    print("[CMD12] 표시 루프 시작 (q: 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CMD12] 영상 끝")
            break

        frame_id += 1

        # 1. 프레임 → YOLO 워커 전송
        packet = {
            "frame": frame,
            "full_frame": None,
            "frame_id": frame_id,
            "camera_id": "CAM01",
        }
        try:
            frame_queue.put_nowait(packet)
        except Full:
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
            try:
                frame_queue.put_nowait(packet)
            except Full:
                pass

        # 2. YOLO 결과 수신 (논블로킹)
        _new_det_received = False
        try:
            det_result = detect_queue.get_nowait()
            last_det_result = det_result
            _new_det_received = True
        except Empty:
            pass

        # 3. OCR 결과 수신 (논블로킹, 복수 건)
        while True:
            try:
                ocr_result = ocr_result_queue.get_nowait()
                total_ocr_count += 1
                key = (ocr_result["frame_id"], ocr_result["det_index"])
                trk_state = _pending_ocr.pop(key, None)
                if trk_state is not None and ocr_result["best_text"]:
                    matched_trk, is_new, _ = trk_state
                    # 트래커 업데이트
                    result = _update_tracker_with_ocr(
                        tracker, ocr_result, matched_trk, is_new,
                        consecutive_required, validator, db
                    )
                    if result:
                        result["_last_seen"] = frame_id
                        print(f"[CMD12] OCR: '{ocr_result['best_text']}' "
                              f"conf={ocr_result['best_conf']:.2f}", flush=True)
                        # 기존 표시 결과에서 같은 번호판 교체
                        _replaced = False
                        for idx, existing in enumerate(display_results):
                            if existing.get("plate") == result["plate"]:
                                display_results[idx] = result
                                _replaced = True
                                break
                        if not _replaced:
                            display_results.append(result)
            except Empty:
                break

        # 4. YOLO 탐지 처리 (새 결과가 도착할 때마다)
        # ★ YOLO 결과 도착 시 즉시 처리 (frame_id 정확 매칭 불필요)
        # YOLO는 비동기이므로 결과의 frame_id가 현재 표시 frame_id와 다를 수 있음
        if _new_det_received and last_det_result is not None:
            _det_frame_id = last_det_result["frame_id"]  # YOLO 결과의 원본 프레임 번호
            new_display = []
            tracker.begin_frame()

            for det_idx, det in enumerate(last_det_result["detections"]):
                bbox = det["bbox"]
                skip, matched_trk, is_new, cached = _should_skip_ocr(
                    tracker, bbox, consecutive_required
                )
                if skip and cached:
                    cached["_last_seen"] = frame_id
                    new_display.append(cached)
                elif not skip:
                    # ROI가 있으면 OCR 요청 전송
                    roi = det.get("roi")
                    if roi is not None and roi.size > 0:
                        ox1, oy1, ox2, oy2 = bbox
                        ch_full, cw_full = frame.shape[:2]
                        margin_x = int((ox2 - ox1) * cfg["ROI_MARGIN_X"])
                        margin_y = int((oy2 - oy1) * cfg["ROI_MARGIN_Y"])
                        rx1 = max(0, ox1 - margin_x)
                        ry1 = max(0, oy1 - margin_y)

                        ocr_request = {
                            "frame_id": _det_frame_id,
                            "det_index": det_idx,
                            "roi": roi,
                            "bbox": bbox,
                            "roi_crop_offset": [rx1, ry1],
                            "det_w": ox2 - ox1,
                            "det_h": oy2 - oy1,
                            "bbox_conf_penalty": det["bbox_conf_penalty"],
                            "is_small_plate": det["is_small_plate"],
                            "aspect_type": det["aspect_type"],
                            "timestamp": time.time(),
                        }

                        # ocr_queue: Full이면 오래된 요청 버림
                        try:
                            ocr_queue.put_nowait(ocr_request)
                        except Full:
                            try:
                                ocr_queue.get_nowait()
                            except Empty:
                                pass
                            try:
                                ocr_queue.put_nowait(ocr_request)
                            except Full:
                                pass

                        # 트래커 상태 보관 (OCR request의 frame_id와 동일한 키 사용)
                        _pending_ocr[(_det_frame_id, det_idx)] = (
                            matched_trk, is_new, {"timestamp": time.time()}
                        )

                    # 아직 OCR 결과 없는 탐지는 bbox만 표시
                    new_display.append({
                        "plate": "",
                        "confidence": 0,
                        "bbox": bbox,
                        "is_alert": False,
                        "alert_info": None,
                        "plate_number": "",
                        "confidence_level": "",
                        "plate_type": "",
                        "vehicle_type": "",
                        "plate_lines": 1,
                        "plate_color": "",
                        "bbox_area": 0,
                        "frame_count": 0,
                        "is_valid_format": False,
                        "rejection_reason": None,
                    })

            tracker.end_frame()

            # 기존 OCR 결과 유지 (TTL 이내 + 같은 번호판이 new_display에 없는 것만)
            for prev in display_results:
                if prev.get("plate"):
                    _age = frame_id - prev.get("_last_seen", 0)
                    if _age > _DISPLAY_TTL:
                        continue  # TTL 만료 → 제거
                    found = False
                    for nd in new_display:
                        if nd.get("plate") == prev["plate"]:
                            found = True
                            break
                    if not found:
                        new_display.append(prev)

            display_results = new_display

            # ★ 디버그 (30프레임마다 또는 상태 변화 시)
            if frame_id % 30 == 0:
                _plates = sum(1 for d in display_results if d.get("plate"))
                _pending = sum(1 for d in display_results if not d.get("plate") and d.get("bbox"))
                print(f"[CMD12] frame={frame_id} yolo={len(last_det_result['detections'])} "
                      f"plates={_plates} pending={_pending} "
                      f"ocr={total_ocr_count}", flush=True)

        # _pending_ocr 타임아웃 정리
        now = time.time()
        expired = [k for k, (_, _, det) in _pending_ocr.items()
                   if now - det.get("timestamp", now) > _PENDING_TIMEOUT]
        for k in expired:
            del _pending_ocr[k]

        # 5. 표시
        display_frame = frame.copy()

        # FPS 계산
        fps_counter += 1
        if now - fps_time >= 1.0:
            display_fps = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        inference_ms = last_det_result["inference_ms"] if last_det_result else 0.0
        draw_results(display_frame, display_results, inference_ms, display_fps, total_ocr_count)

        cv2.imshow(_win_name, display_frame)

        # 프레임 레이트 제한: 원본 FPS에 맞춰 대기
        _elapsed = time.time() - _last_frame_time
        _wait_ms = max(1, int((frame_interval - _elapsed) * 1000))
        _last_frame_time = time.time()
        key = cv2.waitKey(_wait_ms) & 0xFF
        if key == ord('q') or key == 27:
            print("[CMD12] 사용자 종료 요청")
            break

    cv2.destroyAllWindows()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E. 메인 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    """argparse + CMD5/CMD6 프로세스 관리"""
    parser = argparse.ArgumentParser(description="2단계 파이프라인: YOLO + OCR 분리")
    parser.add_argument("--input", type=str, default="movie/hiway.mp4",
                        help="입력 소스 (파일 경로 또는 웹캠 인덱스)")
    args = parser.parse_args()

    source = args.input
    if source.isdigit():
        source = int(source)

    cap = open_video(source)
    if cap is None:
        sys.exit(1)

    # IPC 큐 생성
    frame_queue = Queue(maxsize=2)         # CMD 12 → CMD 5
    detect_queue = Queue(maxsize=10)       # CMD 5 → CMD 12
    ocr_queue = Queue(maxsize=3)           # CMD 12 → CMD 6
    ocr_result_queue = Queue(maxsize=20)   # CMD 6 → CMD 12
    cmd_queue_yolo = Queue(maxsize=5)      # CMD 12 → CMD 5 제어
    cmd_queue_ocr = Queue(maxsize=5)       # CMD 12 → CMD 6 제어

    # CMD 5: YOLO 워커
    yolo_worker = Process(
        target=yolo_worker_loop,
        args=(frame_queue, detect_queue, cmd_queue_yolo),
        daemon=True,
        name="CMD5-YOLO-Worker",
    )
    # CMD 6: OCR 워커
    ocr_worker = Process(
        target=ocr_worker_loop,
        args=(ocr_queue, ocr_result_queue, cmd_queue_ocr),
        daemon=True,
        name="CMD6-OCR-Worker",
    )

    print("[CMD12] YOLO 워커 시작...")
    yolo_worker.start()
    print("[CMD12] OCR 워커 시작...")
    ocr_worker.start()

    try:
        display_loop(cap, frame_queue, detect_queue, ocr_queue,
                     ocr_result_queue, cmd_queue_yolo, cmd_queue_ocr)
    finally:
        print("[CMD12] 워커 프로세스 종료 중...")
        # YOLO 워커 종료
        try:
            cmd_queue_yolo.put_nowait(CMD_STOP)
        except Full:
            pass
        # OCR 워커 종료
        try:
            cmd_queue_ocr.put_nowait(CMD_OCR_STOP)
        except Full:
            pass

        cap.release()

        yolo_worker.join(timeout=5)
        if yolo_worker.is_alive():
            print("[CMD12] YOLO 워커 강제 종료")
            yolo_worker.terminate()
            yolo_worker.join(timeout=2)

        ocr_worker.join(timeout=5)
        if ocr_worker.is_alive():
            print("[CMD12] OCR 워커 강제 종료")
            ocr_worker.terminate()
            ocr_worker.join(timeout=2)

        print("[CMD12] 파이프라인 종료 완료")


if __name__ == "__main__":
    freeze_support()
    main()
