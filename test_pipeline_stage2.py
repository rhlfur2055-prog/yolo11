# -*- coding: utf-8 -*-
"""
파이프라인 2단계 테스트: OCR 파이프라인 등가 검증
- [A] 12장 정적 이미지: cmd6_ocr_worker의 _process_single_ocr() vs plate_engine_pro.py
- [B] 30프레임 영상: cmd6_ocr_worker OCR 직접 호출 비교
- GUI 없이 헤드리스 실행
"""
import os
import sys
import time
import re
import cv2
import numpy as np
from multiprocessing import Process, Queue, freeze_support
from queue import Full, Empty

# 파이프라인 모듈
from pipeline_common import CMD_STOP, CMD_OCR_STOP, DETECT_CONFIG
from cmd5_yolo_worker import (
    yolo_worker_loop, _resolve_plate_model, _load_best_model,
    _calc_imgsz, _manual_nms, _filter_detection, _crop_roi,
)
from cmd6_ocr_worker import (
    _init_ocr_engines, _process_single_ocr,
)

# 기존 엔진 (비교용)
from plate_engine_pro import PlateEnginePro

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "22")
VIDEO_PATH = "movie/hiway.mp4"
MAX_FRAMES = 30
START_FRAME = 300
RESULT_FILE = "pipeline_stage2_test_result.txt"

# 테스트 케이스 (test_ocr_accuracy.py와 동일)
TEST_CASES = [
    ("경기76바7789.png",        "경기76바7789"),
    ("서울바9203.png",          "서울70바9203"),
    ("트럭 경기91바6286.png",   "경기91바6286"),
    ("01나8060.png",            "01나8060"),
    ("02누2754.png",            "02누2754"),
    ("14니3234.png",            "14나3234"),
    ("36다7117.png",            "36다7117"),
    ("48보7062.png",            "48보7062"),
    ("55저9392.png",            "55저9392"),
    ("58두9599.png",            "58두9599"),
    ("70버6393.png",            "70버6393"),
    ("80부5915.png",            "80부5915"),
]


def normalize_plate(text):
    """비교용 정규화"""
    if not text:
        return ""
    text = re.sub(r"[\s\-\.\(\)（）]", "", text)
    text = text.replace("(F)", "")
    return text.strip().upper()


def test_static_images(output_lines):
    """[A] 12장 정적 이미지: 파이프라인 OCR vs 기존 엔진
    전략: 기존 엔진으로 YOLO 탐지 + ROI 추출 → CMD6 OCR 파이프라인에 전달
    (정적 이미지의 ROI 필터(300-800px)는 영상 전용이므로 기존 엔진의 탐지를 공유)
    """
    output_lines.append("=" * 70)
    output_lines.append("  [A] 정적 이미지 OCR 등가 테스트 (12장)")
    output_lines.append("    (YOLO 탐지=기존 엔진 공유, OCR만 CMD6 비교)")
    output_lines.append("=" * 70)
    output_lines.append("")

    # 기존 엔진 초기화 (YOLO 탐지 + OCR 동시 수행)
    output_lines.append("[기존 엔진] PlateEnginePro 로딩...")
    engine = PlateEnginePro()
    engine.consecutive_required = 1

    # cmd6 OCR 엔진 초기화
    output_lines.append("[CMD6 OCR] 엔진 로딩...")
    ocr_engines, crnn_model, hangul_clf, kr_allowlist = _init_ocr_engines()
    from plate_engine_pro import ImagePreprocessor, PlateValidator, PlateEngineConfig
    preprocessor = ImagePreprocessor()
    validator = PlateValidator()
    cfg = DETECT_CONFIG

    output_lines.append("")
    output_lines.append(f"{'#':>2}  {'파일명':<28} {'정답':<14} {'기존엔진':<20} {'CMD6-OCR':<20} {'일치'}")
    output_lines.append("-" * 100)

    engine_correct = 0
    cmd6_correct = 0
    match_count = 0
    total = len(TEST_CASES)

    for idx, (filename, gt) in enumerate(TEST_CASES, 1):
        filepath = os.path.join(IMG_DIR, filename)
        if not os.path.exists(filepath):
            output_lines.append(f"{idx:>2}  {filename:<28} {gt:<14} {'파일없음':<20} {'파일없음':<20} -")
            continue

        img = cv2.imread(filepath)
        if img is None:
            output_lines.append(f"{idx:>2}  {filename:<28} {gt:<14} {'로드실패':<20} {'로드실패':<20} -")
            continue

        # --- 기존 엔진: YOLO 탐지 + OCR (동시) ---
        engine.reset_state()
        engine_dets = engine.process_frame(img)
        if engine_dets:
            best_eng = max(engine_dets, key=lambda d: d.get("confidence", 0))
            eng_text = best_eng.get("plate", "")
            eng_conf = best_eng.get("confidence", 0)
            eng_bbox = best_eng.get("bbox", [0, 0, 0, 0])
        else:
            eng_text, eng_conf, eng_bbox = "", 0.0, [0, 0, 0, 0]

        # --- CMD6 OCR: 기존 엔진의 YOLO bbox를 재사용하여 ROI 추출 ---
        cmd6_text, cmd6_conf = "", 0.0
        if eng_bbox and eng_bbox != [0, 0, 0, 0]:
            # 기존 엔진과 동일한 방식으로 ROI 크롭
            ox1, oy1, ox2, oy2 = eng_bbox
            ch_full, cw_full = img.shape[:2]
            margin_x = int((ox2 - ox1) * cfg["ROI_MARGIN_X"])
            margin_y = int((oy2 - oy1) * cfg["ROI_MARGIN_Y"])
            rx1 = max(0, ox1 - margin_x)
            ry1 = max(0, oy1 - margin_y)
            rx2 = min(cw_full, ox2 + margin_x)
            ry2 = min(ch_full, oy2 + margin_y)
            roi = img[ry1:ry2, rx1:rx2]

            if roi.size > 0:
                det_w = ox2 - ox1
                det_h = oy2 - oy1
                # bbox 크기 기반 페널티 (cmd5 _filter_detection과 동일)
                if det_w < 70:
                    bbox_penalty = 0.60
                    is_small = True
                elif det_w < 100:
                    bbox_penalty = 0.85
                    is_small = False
                elif det_w < 120:
                    bbox_penalty = 0.95
                    is_small = False
                else:
                    bbox_penalty = 1.0
                    is_small = False
                aspect = det_w / max(det_h, 1)
                aspect_type = "1line" if aspect > 2.0 else "2line"

                request = {
                    "frame_id": idx,
                    "det_index": 0,
                    "roi": roi,
                    "bbox": eng_bbox,
                    "roi_crop_offset": [rx1, ry1],
                    "det_w": det_w,
                    "det_h": det_h,
                    "bbox_conf_penalty": bbox_penalty,
                    "is_small_plate": is_small,
                    "aspect_type": aspect_type,
                    "timestamp": time.time(),
                }
                result = _process_single_ocr(
                    request, ocr_engines, preprocessor, validator,
                    crnn_model, hangul_clf, kr_allowlist
                )
                if result["best_text"]:
                    cmd6_text = result["best_text"]
                    cmd6_conf = result["best_conf"]

        # 비교
        eng_ok = normalize_plate(eng_text) == normalize_plate(gt)
        cmd6_ok = normalize_plate(cmd6_text) == normalize_plate(gt)
        texts_match = normalize_plate(eng_text) == normalize_plate(cmd6_text)

        if eng_ok:
            engine_correct += 1
        if cmd6_ok:
            cmd6_correct += 1
        if texts_match:
            match_count += 1

        eng_str = f"{eng_text}({eng_conf:.0%})" if eng_text else "(미인식)"
        cmd6_str = f"{cmd6_text}({cmd6_conf:.0%})" if cmd6_text else "(미인식)"
        match_mark = "O" if texts_match else "X"

        display_fname = filename if len(filename) <= 26 else filename[:23] + "..."
        output_lines.append(f"{idx:>2}  {display_fname:<28} {gt:<14} {eng_str:<20} {cmd6_str:<20} {match_mark}")

    output_lines.append("-" * 100)
    output_lines.append(f"기존 엔진 정확도: {engine_correct}/{total}")
    output_lines.append(f"CMD6 OCR 정확도: {cmd6_correct}/{total}")
    output_lines.append(f"텍스트 일치율: {match_count}/{total}")
    output_lines.append("")

    return engine_correct, cmd6_correct, match_count


def test_video_frames(output_lines):
    """[B] 30프레임 영상: CMD6 OCR 직접 호출 vs 기존 엔진"""
    output_lines.append("=" * 70)
    output_lines.append("  [B] 영상 30프레임 OCR 비교 테스트")
    output_lines.append("=" * 70)
    output_lines.append("")

    if not os.path.exists(VIDEO_PATH):
        output_lines.append(f"[건너뜀] 영상 파일 없음: {VIDEO_PATH}")
        return 0, 0

    # 기존 엔진
    engine = PlateEnginePro()
    engine.consecutive_required = 1

    # CMD6 OCR
    ocr_engines, crnn_model, hangul_clf, kr_allowlist = _init_ocr_engines()
    from plate_engine_pro import ImagePreprocessor, PlateValidator
    preprocessor = ImagePreprocessor()
    validator = PlateValidator()

    # YOLO 모델
    from ultralytics import YOLO
    model_path = _resolve_plate_model()
    yolo_model = YOLO(str(model_path)) if model_path else _load_best_model()
    _names = yolo_model.names or {}
    _name_vals = [str(v).lower() for v in _names.values()]
    is_plate_model = any(
        kw in n for n in _name_vals
        for kw in ("plate", "license", "번호판")
    ) or (model_path is not None and "plate" in str(model_path).lower())
    cfg = DETECT_CONFIG

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        output_lines.append(f"[오류] 영상 열기 실패: {VIDEO_PATH}")
        return 0, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    eng_total_text = 0
    cmd6_total_text = 0
    match_count = 0
    frames_compared = 0

    output_lines.append(f"{'프레임':>6}  {'기존엔진':<20} {'CMD6-OCR':<20} {'일치'}")
    output_lines.append("-" * 60)

    for i in range(MAX_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        fid = START_FRAME + i + 1

        # --- 기존 엔진 ---
        engine.reset_state()
        engine_dets = engine.process_frame(frame)
        eng_text = ""
        if engine_dets:
            best_eng = max(engine_dets, key=lambda d: d.get("confidence", 0))
            eng_text = best_eng.get("plate", "")

        # --- CMD6 OCR ---
        ch_full, cw_full = frame.shape[:2]
        _imgsz = _calc_imgsz(cw_full)
        detections = yolo_model(frame, conf=cfg["DETECT_CONF"], imgsz=_imgsz, verbose=False, max_det=3)
        _keep = _manual_nms(detections[0].boxes, is_plate_model, cfg)

        cmd6_text = ""
        cmd6_conf = 0.0
        for _, _, det in _keep:
            info = _filter_detection(det, 1.0, 1.0, ch_full, is_plate_model, cfg)
            if info is None:
                continue
            roi = _crop_roi(frame, info["bbox"], cfg)
            if roi is None:
                continue

            ox1, oy1, ox2, oy2 = info["bbox"]
            margin_x = int((ox2 - ox1) * cfg["ROI_MARGIN_X"])
            margin_y = int((oy2 - oy1) * cfg["ROI_MARGIN_Y"])
            rx1 = max(0, ox1 - margin_x)
            ry1 = max(0, oy1 - margin_y)

            request = {
                "frame_id": fid,
                "det_index": 0,
                "roi": roi,
                "bbox": info["bbox"],
                "roi_crop_offset": [rx1, ry1],
                "det_w": ox2 - ox1,
                "det_h": oy2 - oy1,
                "bbox_conf_penalty": info["bbox_conf_penalty"],
                "is_small_plate": info["is_small_plate"],
                "aspect_type": info["aspect_type"],
                "timestamp": time.time(),
            }
            result = _process_single_ocr(
                request, ocr_engines, preprocessor, validator,
                crnn_model, hangul_clf, kr_allowlist
            )
            if result["best_text"] and result["best_conf"] > cmd6_conf:
                cmd6_text = result["best_text"]
                cmd6_conf = result["best_conf"]

        # 비교
        if eng_text or cmd6_text:
            frames_compared += 1
            if eng_text:
                eng_total_text += 1
            if cmd6_text:
                cmd6_total_text += 1
            texts_match = normalize_plate(eng_text) == normalize_plate(cmd6_text)
            if texts_match:
                match_count += 1
            match_mark = "O" if texts_match else "X"
            eng_str = eng_text if eng_text else "(없음)"
            cmd6_str = cmd6_text if cmd6_text else "(없음)"
            output_lines.append(f"{fid:>6}  {eng_str:<20} {cmd6_str:<20} {match_mark}")

    cap.release()

    output_lines.append("-" * 60)
    output_lines.append(f"기존 엔진 인식 프레임: {eng_total_text}/{MAX_FRAMES}")
    output_lines.append(f"CMD6 OCR 인식 프레임: {cmd6_total_text}/{MAX_FRAMES}")
    output_lines.append(f"텍스트 일치: {match_count}/{max(frames_compared, 1)} "
                        f"({match_count/max(frames_compared, 1)*100:.0f}%)")
    output_lines.append("")

    return eng_total_text, cmd6_total_text


def main():
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("  파이프라인 2단계 OCR 등가 테스트")
    output_lines.append("=" * 70)
    output_lines.append("")

    # [A] 정적 이미지 12장
    t0 = time.time()
    eng_correct, cmd6_correct, match_count = test_static_images(output_lines)
    static_elapsed = time.time() - t0

    # [B] 영상 30프레임
    t0 = time.time()
    eng_video, cmd6_video = test_video_frames(output_lines)
    video_elapsed = time.time() - t0

    # 요약
    output_lines.append("=" * 70)
    output_lines.append("  [C] 종합 결과")
    output_lines.append("=" * 70)
    output_lines.append(f"  정적 12장 — 기존: {eng_correct}/12, CMD6: {cmd6_correct}/12, 일치: {match_count}/12")
    output_lines.append(f"  영상 30프레임 — 기존: {eng_video}, CMD6: {cmd6_video}")
    output_lines.append(f"  정적 테스트 소요: {static_elapsed:.1f}초")
    output_lines.append(f"  영상 테스트 소요: {video_elapsed:.1f}초")

    # 판정
    if cmd6_correct >= eng_correct:
        output_lines.append("  => CMD6 OCR 정확도 기존 엔진 이상")
    else:
        output_lines.append(f"  => CMD6 OCR 정확도 {eng_correct - cmd6_correct}건 열세 (확인 필요)")

    if match_count >= 10:
        output_lines.append("  => 텍스트 일치율 양호 (10+/12)")
    else:
        output_lines.append(f"  => 텍스트 일치율 미흡 ({match_count}/12)")

    output_lines.append("")

    full_text = "\n".join(output_lines)
    print(full_text)

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"\n결과 저장: {RESULT_FILE}")


if __name__ == "__main__":
    freeze_support()
    main()
