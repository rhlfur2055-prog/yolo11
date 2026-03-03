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
import re
import time
import sys
import cv2
import numpy as np
from multiprocessing import Process, Queue, freeze_support
from queue import Full, Empty
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

from pipeline_common import (
    CMD_STOP, CMD_OCR_STOP, CMD_YOLO_READY, CMD_FLUSH, DETECT_CONFIG, bbox_iou,
)
from cmd5_yolo_worker import yolo_worker_loop, _is_valid_plate_format
from cmd6_ocr_worker import ocr_worker_loop
from config import PathConfig

# plate_engine_pro.py에서 트래커 import
from plate_engine_pro import PlateTracker, PlateValidator, PlateDatabase

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한글 텍스트 렌더링 (plate_gui.py에서 추출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FONT_PATH = PathConfig.font_path(bold=True)
FONT_PATH_FALLBACK = PathConfig.font_path(bold=False)
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
def _display_nms(results, iou_threshold=0.5):
    """display_results에서 IoU 높은 중복 제거 — conf 높은 것만 유지"""
    if not results:
        return results
    sorted_results = sorted(results, key=lambda r: r.get("confidence", 0), reverse=True)
    keep = []
    for r in sorted_results:
        bbox = r.get("bbox", [0, 0, 0, 0])
        is_dup = False
        for k in keep:
            if bbox_iou(bbox, k.get("bbox", [0, 0, 0, 0])) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(r)
    return keep


def draw_results(frame, results, inference_ms, display_fps, ocr_count=0):
    """bbox + 번호판 텍스트 + 신뢰도 표시 (NMS + 크기필터 + 겹침방지)"""
    # ★ NMS 적용 — 중복 bbox 제거
    results = _display_nms(results)

    info_text = f"FPS: {display_fps:.1f} | YOLO: {inference_ms:.0f}ms | Det: {len(results)} | OCR: {ocr_count}"
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    korean_labels = []
    _drawn_label_ys = []  # ★ 겹침 방지용 y좌표 기록

    for i, res in enumerate(results):
        bbox = res["bbox"]
        text = res.get("plate", "")
        conf = res.get("confidence", 0)
        x1, y1, x2, y2 = bbox

        # ★ 원거리 차량 스킵 (너무 작은 bbox는 그리지 않음)
        _bw = x2 - x1
        _bh = y2 - y1
        if _bw < 80 or _bh < 25:
            continue

        if text:
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{text} ({conf*100:.0f}%)"
            font_sz = max(18, min(28, _bw // 4))

            # ★ 텍스트 겹침 방지 — 이미 그린 라벨과 겹치면 위로 밀기
            label_y = max(0, y1 - font_sz - 6)
            for dy in _drawn_label_ys:
                if abs(label_y - dy) < 35:
                    label_y = dy - 35
            label_y = max(0, label_y)
            _drawn_label_ys.append(label_y)

            korean_labels.append((label, x1 + 2, label_y, font_sz, color))
        else:
            color = (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    if korean_labels:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for label, x, y, font_sz, color_bgr in korean_labels:
            font = _get_font(font_sz)
            rgb_color = (color_bgr[2], color_bgr[1], color_bgr[0])
            # ★ 배경 박스 (가독성 향상)
            _bbox = draw.textbbox((x, y), label, font=font)
            draw.rectangle([_bbox[0]-2, _bbox[1]-2, _bbox[2]+2, _bbox[3]+2], fill=(0, 0, 0))
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
                 ocr_result_queue, cmd_queue_yolo, cmd_queue_ocr,
                 headless=False):
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

    # _pending_ocr: (frame_id, det_index) → (tracker_id, matched_trk, is_new_track, timestamp)
    _pending_ocr = {}
    # tracker_id → ocr_key 역매핑 (동일 트래커 중복 OCR 요청 방지)
    _pending_tracker_ids = {}
    # 트랙 고유 ID 카운터 (단조 증가)
    _next_track_id = 0
    # 보류 결과 최대 보관 시간 (초) — dedup으로 누적 감소하여 단축
    _PENDING_TIMEOUT = 15.0

    # 현재 표시할 결과 목록 (각 항목에 "_last_seen" 프레임 번호 저장)
    display_results = []
    _DISPLAY_TTL = 12  # 12프레임(~0.4초) — Heavy OCR 도착 여유 확보

    # ★ 연속 Det=0 카운터 (잔상 즉시 클리어용)
    _zero_det_streak = 0

    # ★ 트래커별 번호판 투표 카운터 (오인식 방지)
    _plate_votes = {}  # {tid: {text: count}}

    # ★ 스크린샷 저장 설정
    _screenshot_dir = "test_results/screenshots"
    os.makedirs(_screenshot_dir, exist_ok=True)
    _prev_det_count = 0          # 이전 프레임 탐지 수 (Det=0 직전 감지용)
    _ss_pending_fast = []        # Fast OCR 성공 시 지연 저장 목록: [(frame_id, text)]
    _ss_saved_count = 0          # 총 저장 수 (로그용)

    # 영상 정보
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / video_fps  # 프레임 간격 (초)
    _last_frame_time = time.time()

    # cv2 창 초기화 (headless 모드에서는 생략)
    _win_name = "Pipeline Stage 2 - YOLO + OCR"
    if not headless:
        cv2.namedWindow(_win_name, cv2.WINDOW_AUTOSIZE)

    print(f"[CMD12] 표시 루프 시작 ({'headless' if headless else 'q: 종료'})")

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
        if headless:
            # ★ Headless: 블로킹 put — YOLO 처리 속도에 맞춤 (프레임 드롭 최소화)
            try:
                frame_queue.put(packet, timeout=5.0)
            except Full:
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
        else:
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
            # ━━ CMD_FLUSH 처리: 차량 소멸 시 큐 초기화 ━━
            if isinstance(det_result, dict) and det_result.get("cmd") == CMD_FLUSH:
                _flush_fid = det_result.get("frame_id", "?")
                # ocr_queue 비우기
                while not ocr_queue.empty():
                    try:
                        ocr_queue.get_nowait()
                    except Empty:
                        break
                # cmd6에도 FLUSH 전달
                try:
                    cmd_queue_ocr.put_nowait(CMD_FLUSH)
                except Full:
                    pass
                # 대기 중 OCR 추적 초기화
                _pending_ocr.clear()
                _pending_tracker_ids.clear()
                print(f"[CMD12] FLUSH 수신 (frame={_flush_fid}) → ocr_queue/pending 초기화")
                det_result = None
            else:
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
                if trk_state is None:
                    continue  # 이미 드레인/타임아웃으로 폐기된 요청
                tid, matched_trk, is_new, _ts = trk_state
                # 역매핑 정리
                _pending_tracker_ids.pop(tid, None)

                if not ocr_result["best_text"]:
                    continue

                # ★ 트래커 생존 확인 — 사망 시 트랙 부활 안 함 (좀비 방지)
                # 단, 결과 자체는 TTL 기한부로 표시 (OCR 지연 대응)
                _tracker_alive = matched_trk in tracker.tracks
                print(f"[DEBUG-OCR-RECV] frame={frame_id} text={ocr_result['best_text']} "
                      f"tracker_alive={_tracker_alive} tid={tid}", flush=True)
                if not _tracker_alive:
                    # ★ 트래커 사망 — display에 추가하지 않음 (잔상 방지)
                    # 기존 display에 같은 tid가 있으면 교체만 허용 (이미 표시 중인 건)
                    if ocr_result["best_text"]:
                        print(f"[CMD12] OCR(dead): '{ocr_result['best_text']}' "
                              f"conf={ocr_result['best_conf']:.2f} tid={tid}", flush=True)
                        # 기존 display에 같은 tid가 있으면 Heavy 결과로 교체 (정확도 향상)
                        for idx, existing in enumerate(display_results):
                            if existing.get("_tracker_id") == tid:
                                existing["plate"] = ocr_result["best_text"]
                                existing["confidence"] = ocr_result["best_conf"]
                                existing["plate_number"] = ocr_result["best_text"]
                                existing.pop("is_fast_ocr", None)
                                break
                    continue

                # 트래커 업데이트
                result = _update_tracker_with_ocr(
                    tracker, ocr_result, matched_trk, is_new,
                    consecutive_required, validator, db
                )
                if result:
                    result["_last_seen"] = frame_id
                    result["_tracker_id"] = tid

                    # ★ C-2: Heavy OCR 결과를 투표 카운터에 가중 반영 (+5)
                    _heavy_text = ocr_result["best_text"]
                    if tid not in _plate_votes:
                        _plate_votes[tid] = {}
                    _plate_votes[tid][_heavy_text] = _plate_votes[tid].get(_heavy_text, 0) + 5

                    # ★ 기존 표시 결과에서 같은 tracker_id 또는 같은 번호판 교체
                    _replaced = False
                    for idx, existing in enumerate(display_results):
                        if existing.get("_tracker_id") == tid or existing.get("plate") == result["plate"]:
                            # fast_ocr → heavy 교체 로그
                            if existing.get("is_fast_ocr"):
                                print(f"[CMD12] Heavy→Fast교체: '{existing.get('plate')}' → "
                                      f"'{result['plate']}' conf={ocr_result['best_conf']:.2f} "
                                      f"tid={tid}", flush=True)
                            else:
                                print(f"[CMD12] OCR: '{ocr_result['best_text']}' "
                                      f"conf={ocr_result['best_conf']:.2f} tid={tid}", flush=True)
                            display_results[idx] = result
                            _replaced = True
                            break
                    if not _replaced:
                        print(f"[CMD12] OCR: '{ocr_result['best_text']}' "
                              f"conf={ocr_result['best_conf']:.2f} tid={tid}", flush=True)
                        display_results.append(result)
            except Empty:
                break

        # 4. YOLO 탐지 처리 (새 결과가 도착할 때마다)
        if _new_det_received and last_det_result is not None:
            _det_frame_id = last_det_result["frame_id"]

            # ★ 연속 Det=0 카운터 관리 → 잔상 즉시 클리어
            if len(last_det_result["detections"]) == 0:
                _zero_det_streak += 1
                if _zero_det_streak >= 3:
                    display_results.clear()
                    _plate_votes.clear()  # 투표 카운터도 리셋
            else:
                _zero_det_streak = 0

            new_display = []
            tracker.begin_frame()

            for det_idx, det in enumerate(last_det_result["detections"]):
                bbox = det["bbox"]
                fast_text = det.get("fast_ocr_text", "")
                fast_conf = det.get("fast_ocr_conf", 0.0)

                skip, matched_trk, is_new, cached = _should_skip_ocr(
                    tracker, bbox, consecutive_required
                )

                # 모든 트랙에 고유 _pipeline_id 할당
                if "_pipeline_id" not in matched_trk:
                    matched_trk["_pipeline_id"] = _next_track_id
                    _next_track_id += 1
                tid = matched_trk["_pipeline_id"]

                if skip and cached:
                    # ★ bbox IoU 체크 — 같은 차인지 확인
                    _cached_bbox = cached.get("bbox", [0, 0, 0, 0])
                    _cache_iou = bbox_iou(_cached_bbox, bbox)
                    if _cache_iou < 0.3:
                        # 다른 차 → 캐시 무효화, 새 OCR 요청으로 전환
                        skip = False
                        cached = None
                    else:
                        # ★ 같은 차 — 캐시에 fast_ocr가 더 높은 conf면 교체
                        if fast_text and fast_conf > cached.get("confidence", 0):
                            cached["plate"] = fast_text
                            cached["confidence"] = fast_conf
                            cached["plate_number"] = fast_text
                            cached["is_fast_ocr"] = True
                        cached["_last_seen"] = frame_id
                        cached["_tracker_id"] = tid
                        new_display.append(cached)
                if not skip:
                    # ROI가 있으면 Heavy OCR 요청 전송 (기존 로직 유지)
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
                            "fast_ocr_text": fast_text,
                            "fast_ocr_conf": fast_conf,
                        }

                        # ★ 이미 이 tracker에 대한 OCR 대기 중이면 스킵
                        if tid in _pending_tracker_ids:
                            pass
                        else:
                            try:
                                ocr_queue.put_nowait(ocr_request)
                            except Full:
                                while True:
                                    try:
                                        _old = ocr_queue.get_nowait()
                                        _old_key = (_old["frame_id"], _old["det_index"])
                                        _old_state = _pending_ocr.pop(_old_key, None)
                                        if _old_state is not None:
                                            _pending_tracker_ids.pop(_old_state[0], None)
                                    except Empty:
                                        break
                                try:
                                    ocr_queue.put_nowait(ocr_request)
                                except Full:
                                    pass

                            ocr_key = (_det_frame_id, det_idx)
                            _pending_ocr[ocr_key] = (
                                tid, matched_trk, is_new, time.time()
                            )
                            _pending_tracker_ids[tid] = ocr_key

                    # ★ Fast OCR 결과가 있으면 투표 후 즉시 display에 추가
                    if fast_text and fast_conf > 0.3:
                        # ★ 트래커별 투표: 동일 tid에서 여러 프레임 결과 누적
                        # 정규식(한국 번호판 패턴) 완벽 일치 시 가중치 2배
                        if tid not in _plate_votes:
                            _plate_votes[tid] = {}
                        _vote_w = 2 if _is_valid_plate_format(fast_text) else 1
                        _plate_votes[tid][fast_text] = _plate_votes[tid].get(fast_text, 0) + _vote_w

                        # 최다 투표 텍스트를 display에 사용
                        _best_voted = max(_plate_votes[tid], key=_plate_votes[tid].get)
                        _best_voted_count = _plate_votes[tid][_best_voted]
                        # 투표 2회 이상이면 최다 텍스트, 아니면 현재 결과 사용
                        _display_text = _best_voted if _best_voted_count >= 2 else fast_text
                        _display_conf = fast_conf

                        _bw = bbox[2] - bbox[0]
                        _bh = bbox[3] - bbox[1]
                        fast_result = {
                            "plate": _display_text,
                            "confidence": _display_conf,
                            "bbox": bbox,
                            "is_alert": False,
                            "alert_info": None,
                            "plate_number": _display_text,
                            "confidence_level": "",
                            "plate_type": "",
                            "vehicle_type": "",
                            "plate_lines": 1,
                            "plate_color": "",
                            "bbox_area": _bw * _bh,
                            "frame_count": _best_voted_count,
                            "is_valid_format": True,
                            "rejection_reason": None,
                            "_last_seen": frame_id,
                            "_tracker_id": tid,
                            "is_fast_ocr": True,
                        }
                        new_display.append(fast_result)

                        # ★ 트래커에도 fast 결과 저장 (다음 프레임 skip 판단용)
                        matched_trk["_best_text"] = _display_text
                        matched_trk["_best_conf"] = _display_conf

                        print(f"[CMD12] Fast OCR 즉시표시: '{_display_text}' "
                              f"conf={_display_conf:.2f} tid={tid} "
                              f"votes={_plate_votes[tid]}", flush=True)

                        # ★ 스크린샷: Fast OCR 성공 시 지연 저장 예약
                        _ss_pending_fast.append((frame_id, _display_text))
                    else:
                        # Fast OCR 없음 — bbox만 표시 (Heavy OCR 대기)
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
                            "_tracker_id": tid,
                        })

            tracker.end_frame()

            # ★ 기존 OCR 결과 유지 — TTL 이내 + 고스트 체크
            for prev in display_results:
                if not prev.get("plate"):
                    continue
                _prev_tid = prev.get("_tracker_id")
                _age = frame_id - prev.get("_last_seen", 0)

                # TTL 만료 → 즉시 제거
                if _age > _DISPLAY_TTL:
                    continue

                # 같은 tracker_id/텍스트가 이미 new_display에 있으면 스킵
                found = False
                for nd in new_display:
                    if _prev_tid is not None and nd.get("_tracker_id") == _prev_tid:
                        found = True
                        break
                    if nd.get("plate") == prev["plate"]:
                        found = True
                        break
                if not found:
                    # 고스트 체크: 같은 위치에 다른 텍스트의 새 탐지 → 제거
                    prev_bbox = prev.get("bbox", [0, 0, 0, 0])
                    _is_ghost = False
                    for nd in new_display:
                        if nd.get("plate") and nd.get("plate") != prev["plate"]:
                            if bbox_iou(prev_bbox, nd.get("bbox", [0,0,0,0])) > 0.20:
                                _is_ghost = True
                                break
                    if not _is_ghost:
                        new_display.append(prev)

            display_results = new_display

            # ★ display_results 갱신 로그
            _plate_texts = [p.get('plate', '') for p in display_results if p.get('plate')]
            if _plate_texts:
                print(f"[DEBUG-DISPLAY] plates={len(_plate_texts)} texts={_plate_texts}", flush=True)

            # ★ 디버그 (30프레임마다 또는 상태 변화 시)
            if frame_id % 30 == 0:
                _plates = sum(1 for d in display_results if d.get("plate"))
                _pending = sum(1 for d in display_results if not d.get("plate") and d.get("bbox"))
                print(f"[CMD12] frame={frame_id} yolo={len(last_det_result['detections'])} "
                      f"plates={_plates} pending={_pending} "
                      f"ocr={total_ocr_count}", flush=True)

        # ★ YOLO 결과 없는 프레임에서도 TTL 기반 잔상 제거
        if not _new_det_received and display_results:
            display_results = [
                d for d in display_results
                if d.get("plate") and (frame_id - d.get("_last_seen", 0)) <= _DISPLAY_TTL
            ]

        # _pending_ocr 타임아웃 정리 (4-tuple: tid, matched_trk, is_new, timestamp)
        now = time.time()
        expired = [k for k, (tid, _trk, _new, ts) in _pending_ocr.items()
                   if now - ts > _PENDING_TIMEOUT]
        for k in expired:
            _exp_state = _pending_ocr.pop(k, None)
            if _exp_state is not None:
                _pending_tracker_ids.pop(_exp_state[0], None)

        # 5. 표시 (headless 모드에서는 렌더링/대기 생략 → Max FPS)
        if not headless:
            display_frame = frame.copy()

            # FPS 계산
            fps_counter += 1
            if now - fps_time >= 1.0:
                display_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            inference_ms = last_det_result["inference_ms"] if last_det_result else 0.0
            draw_results(display_frame, display_results, inference_ms, display_fps, total_ocr_count)

            # ★ 스크린샷 저장 (draw_results 이후 — bbox+텍스트가 그려진 상태)
            # 1. Fast OCR 성공 시
            if _ss_pending_fast:
                for _ss_fid, _ss_text in _ss_pending_fast:
                    _ss_safe_text = re.sub(r'[\\/:*?"<>|]', '_', _ss_text)
                    _ss_path = f"{_screenshot_dir}/frame_{_ss_fid:05d}_fast_{_ss_safe_text}.jpg"
                    cv2.imwrite(_ss_path, display_frame)
                    _ss_saved_count += 1
                _ss_pending_fast.clear()

            # 2. Det=0 직전 (이전에 탐지 있었는데 현재 0)
            _cur_det_count = len(last_det_result["detections"]) if last_det_result else 0
            if _new_det_received and _cur_det_count == 0 and _prev_det_count > 0:
                _ss_path = f"{_screenshot_dir}/frame_{frame_id:05d}_leaving.jpg"
                cv2.imwrite(_ss_path, display_frame)
                _ss_saved_count += 1
            _prev_det_count = _cur_det_count if _new_det_received else _prev_det_count

            # 3. 30프레임마다 정기 저장
            if frame_id % 30 == 0:
                _ss_path = f"{_screenshot_dir}/frame_{frame_id:05d}_periodic.jpg"
                cv2.imwrite(_ss_path, display_frame)
                _ss_saved_count += 1

            cv2.imshow(_win_name, display_frame)

            # 프레임 레이트 제한: 원본 FPS에 맞춰 대기
            _elapsed = time.time() - _last_frame_time
            _wait_ms = max(1, int((frame_interval - _elapsed) * 1000))
            _last_frame_time = time.time()
            key = cv2.waitKey(_wait_ms) & 0xFF
            if key == ord('q') or key == 27:
                print("[CMD12] 사용자 종료 요청")
                break
        else:
            # ★ Headless: 렌더링/대기 생략 — 미사용 리스트만 클리어
            _ss_pending_fast.clear()

    # ★ Headless: 잔여 YOLO/OCR 결과 드레인 (파이프라인 처리 완료 대기)
    if headless:
        _drain_deadline = time.time() + 15
        while time.time() < _drain_deadline:
            _drain_activity = False

            # YOLO 잔여 결과 수신 + Fast OCR 로그
            try:
                _dr_det = detect_queue.get(timeout=0.5)
                if isinstance(_dr_det, dict) and "detections" in _dr_det:
                    _drain_activity = True
                    for _dd in _dr_det.get("detections", []):
                        _ft = _dd.get("fast_ocr_text", "")
                        _fc = _dd.get("fast_ocr_conf", 0.0)
                        if _ft and _fc > 0.3:
                            _drain_tid = _next_track_id
                            _next_track_id += 1
                            print(f"[CMD12] Fast OCR 즉시표시: '{_ft}' "
                                  f"conf={_fc:.2f} tid={_drain_tid} "
                                  f"votes={{'{_ft}': 1}}", flush=True)
            except Empty:
                pass

            # OCR 잔여 결과 수신 + alive/dead 로그
            while True:
                try:
                    _dr_ocr = ocr_result_queue.get_nowait()
                    total_ocr_count += 1
                    _drain_activity = True
                    if _dr_ocr.get("best_text"):
                        _dk = (_dr_ocr["frame_id"], _dr_ocr["det_index"])
                        _ds = _pending_ocr.pop(_dk, None)
                        if _ds:
                            _dt, _dm, _, _ = _ds
                            _pending_tracker_ids.pop(_dt, None)
                            _da = _dm in tracker.tracks
                            if _da:
                                print(f"[CMD12] OCR: '{_dr_ocr['best_text']}' "
                                      f"conf={_dr_ocr['best_conf']:.2f} tid={_dt}", flush=True)
                            else:
                                print(f"[CMD12] OCR(dead): '{_dr_ocr['best_text']}' "
                                      f"conf={_dr_ocr['best_conf']:.2f} tid={_dt}", flush=True)
                except Empty:
                    break

            if not _drain_activity and not _pending_ocr:
                break

    if _ss_saved_count > 0:
        print(f"[CMD12] 스크린샷 {_ss_saved_count}장 저장 → {_screenshot_dir}/", flush=True)
    if not headless:
        cv2.destroyAllWindows()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E. 메인 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    """argparse + CMD5/CMD6 프로세스 관리"""
    parser = argparse.ArgumentParser(description="2단계 파이프라인: YOLO + OCR 분리")
    parser.add_argument("--input", type=str, default="movie/hiway.mp4",
                        help="입력 소스 (파일 경로 또는 웹캠 인덱스)")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="헤드리스 모드 (cv2 표시 없이 최대 속도 실행)")
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

    # ★ YOLO 모델 로딩 + warm-up 완료 대기 (영상 재생 전 필수)
    print("[CMD12] YOLO ready 대기 중...")
    _yolo_ready = False
    _ready_deadline = time.time() + 30  # 최대 30초 대기
    while not _yolo_ready and time.time() < _ready_deadline:
        try:
            _sig = detect_queue.get(timeout=1.0)
            if isinstance(_sig, dict) and _sig.get("cmd") == CMD_YOLO_READY:
                _yolo_ready = True
                _wms = _sig.get("warmup_ms", 0)
                print(f"[CMD12] YOLO ready 수신 (warm-up: {_wms:.0f}ms)")
            else:
                # ready가 아닌 일반 결과 → 다시 큐에 넣기
                try:
                    detect_queue.put_nowait(_sig)
                except Full:
                    pass
        except Empty:
            pass
    if not _yolo_ready:
        print("[CMD12] YOLO ready 타임아웃 — 강제 시작")

    try:
        display_loop(cap, frame_queue, detect_queue, ocr_queue,
                     ocr_result_queue, cmd_queue_yolo, cmd_queue_ocr,
                     headless=args.headless)
    finally:
        print("[CMD12] 워커 프로세스 종료 중...")

        # 1. 종료 신호 전송 (큐가 가득 차면 비우고 재시도)
        for _cq, _cmd in [(cmd_queue_yolo, CMD_STOP), (cmd_queue_ocr, CMD_OCR_STOP)]:
            for _ in range(3):
                try:
                    _cq.put_nowait(_cmd)
                    break
                except Full:
                    try:
                        _cq.get_nowait()
                    except Empty:
                        pass

        # 2. 데이터 큐 드레인 (워커가 put에서 블로킹되지 않도록)
        for _dq in [frame_queue, detect_queue, ocr_queue, ocr_result_queue]:
            try:
                while True:
                    _dq.get_nowait()
            except Empty:
                pass

        cap.release()

        # 3. 워커 join (타임아웃 후 강제 종료)
        for _w, _name in [(yolo_worker, "YOLO"), (ocr_worker, "OCR")]:
            _w.join(timeout=5)
            if _w.is_alive():
                print(f"[CMD12] {_name} 워커 강제 종료")
                _w.terminate()
                _w.join(timeout=3)

        # 4. 큐 feeder thread 데드락 방지
        for _q in [frame_queue, detect_queue, ocr_queue, ocr_result_queue,
                   cmd_queue_yolo, cmd_queue_ocr]:
            try:
                _q.cancel_join_thread()
            except Exception:
                pass

        print("[CMD12] 파이프라인 종료 완료")


if __name__ == "__main__":
    freeze_support()
    main()
