# -*- coding: utf-8 -*-
"""
파이프라인 1단계 테스트: YOLO 탐지 비교 (파이프라인 vs 기존 엔진)
- movie/hiway.mp4에서 처음 30프레임 처리
- 각 프레임마다 bbox 정보 출력
- ROI 크롭을 crops_test/ 폴더에 저장
- 기존 plate_engine_pro.py와 탐지 수 비교
- GUI 없이 헤드리스 실행
"""
import os
import sys
import time
import cv2
import numpy as np
from multiprocessing import Process, Queue, freeze_support
from queue import Full, Empty

# 파이프라인 모듈
from pipeline_common import CMD_STOP, DETECT_CONFIG
from cmd5_yolo_worker import yolo_worker_loop

VIDEO_PATH = "movie/hiway.mp4"
MAX_FRAMES = 30
START_FRAME = 300          # 번호판이 보이는 구간부터 시작 (처음 ~10초 스킵)
CROPS_DIR = "crops_test"
RESULT_FILE = "pipeline_test_result.txt"
TIMEOUT_SEC = 120


def test_pipeline(output_lines):
    """파이프라인 방식으로 30프레임 처리"""
    output_lines.append("=" * 70)
    output_lines.append("  [A] 파이프라인 YOLO 탐지 테스트 (multiprocessing)")
    output_lines.append("=" * 70)

    # crops_test 폴더 생성
    os.makedirs(CROPS_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        output_lines.append(f"[오류] 영상 열기 실패: {VIDEO_PATH}")
        return 0, 0.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    output_lines.append(f"영상: {VIDEO_PATH} ({w}x{h}, {fps:.1f}fps)")
    output_lines.append(f"시작 프레임: {START_FRAME}")
    output_lines.append("")

    # 시작 프레임까지 스킵
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    # IPC 큐 생성
    frame_queue = Queue(maxsize=5)
    result_queue = Queue(maxsize=30)
    cmd_queue = Queue(maxsize=5)

    # 워커 프로세스 시작
    worker = Process(
        target=yolo_worker_loop,
        args=(frame_queue, result_queue, cmd_queue),
        daemon=True,
    )
    worker.start()

    # 30프레임 전송
    frames_sent = 0
    frames_data = {}  # frame_id → frame (크롭 저장용)
    for i in range(MAX_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        frames_sent += 1
        fid = START_FRAME + i + 1
        frames_data[fid] = frame
        packet = {
            "frame": frame,
            "full_frame": None,
            "frame_id": fid,
            "camera_id": "CAM01",
        }
        frame_queue.put(packet, timeout=10)

    # 결과 수집 (타임아웃 내) — STOP 전에 모든 결과 수신
    results = []
    t_start = time.time()
    while len(results) < frames_sent and (time.time() - t_start) < TIMEOUT_SEC:
        try:
            r = result_queue.get(timeout=5)
            results.append(r)
        except Empty:
            if not worker.is_alive():
                break

    # 모든 결과 수집 후 STOP 전송
    try:
        cmd_queue.put_nowait(CMD_STOP)
    except Full:
        pass

    worker.join(timeout=5)
    if worker.is_alive():
        worker.terminate()
    cap.release()

    # 결과 출력
    total_dets = 0
    total_inference_ms = 0.0
    crop_count = 0

    output_lines.append(f"{'#':>3}  {'프레임':>6}  {'탐지수':>5}  {'추론(ms)':>9}  bbox 상세")
    output_lines.append("-" * 70)

    # frame_id 순서로 정렬
    results.sort(key=lambda x: x["frame_id"])

    for r in results:
        fid = r["frame_id"]
        dets = r["detections"]
        infer = r["inference_ms"]
        total_dets += len(dets)
        total_inference_ms += infer

        det_strs = []
        for j, d in enumerate(dets):
            x1, y1, x2, y2 = d["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            det_strs.append(f"[{x1},{y1},{x2},{y2}] {bw}x{bh} conf={d['conf']:.2f} {d['aspect_type']}")

            # ROI 크롭 저장
            roi = d.get("roi")
            if roi is not None and roi.size > 0:
                crop_path = os.path.join(CROPS_DIR, f"frame{fid:03d}_det{j}.png")
                cv2.imwrite(crop_path, roi)
                crop_count += 1

        if dets:
            output_lines.append(f"{fid:3d}  frame{fid:03d}  {len(dets):5d}  {infer:9.1f}  {det_strs[0]}")
            for ds in det_strs[1:]:
                output_lines.append(f"{'':3}  {'':6}  {'':5}  {'':9}  {ds}")
        else:
            output_lines.append(f"{fid:3d}  frame{fid:03d}  {0:5d}  {infer:9.1f}  (탐지 없음)")

    output_lines.append("-" * 70)
    avg_dets = total_dets / max(len(results), 1)
    avg_infer = total_inference_ms / max(len(results), 1)
    elapsed = time.time() - t_start
    pipeline_fps = len(results) / max(elapsed, 0.001)

    output_lines.append(f"총 프레임: {len(results)}/{frames_sent}")
    output_lines.append(f"총 탐지 수: {total_dets}")
    output_lines.append(f"평균 탐지/프레임: {avg_dets:.2f}")
    output_lines.append(f"평균 추론 시간: {avg_infer:.1f}ms")
    output_lines.append(f"처리 FPS: {pipeline_fps:.1f}")
    output_lines.append(f"크롭 저장: {crop_count}장 → {CROPS_DIR}/")
    output_lines.append("")

    return total_dets, avg_infer


def test_engine_yolo_only(output_lines):
    """기존 엔진과 동일한 모델로 YOLO 탐지만 실행 (OCR 제외, 순수 탐지 수 비교)
    cmd5_yolo_worker.py의 함수를 직접 호출 (동일 코드이므로 공정 비교)
    """
    output_lines.append("=" * 70)
    output_lines.append("  [B] 기존 엔진 동일 모델 YOLO-only 탐지 테스트 (싱글 프로세스)")
    output_lines.append("=" * 70)

    from cmd5_yolo_worker import (
        _resolve_plate_model, _load_best_model, _calc_imgsz,
        _manual_nms, _filter_detection, _crop_roi,
    )
    from ultralytics import YOLO

    # 기존 엔진과 동일한 모델 로드
    model_path = _resolve_plate_model()
    if model_path is not None:
        model = YOLO(str(model_path))
        output_lines.append(f"모델: {model_path}")
    else:
        model = _load_best_model()
        output_lines.append("모델: _load_best_model() 폴백")

    _names = model.names or {}
    _name_vals = [str(v).lower() for v in _names.values()]
    is_plate_model = any(
        kw in n for n in _name_vals
        for kw in ("plate", "license", "번호판")
    ) or (model_path is not None and "plate" in str(model_path).lower())

    cfg = DETECT_CONFIG

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        output_lines.append(f"[오류] 영상 열기 실패: {VIDEO_PATH}")
        return 0, 0.0

    # 동일한 시작 프레임으로 스킵
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    total_dets = 0
    total_time = 0.0

    output_lines.append(f"{'#':>3}  {'프레임':>6}  {'탐지수':>5}  {'시간(ms)':>9}  bbox 상세")
    output_lines.append("-" * 70)

    for i in range(MAX_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        fid = START_FRAME + i + 1
        ch_full, cw_full = frame.shape[:2]
        sx, sy = 1.0, 1.0  # 동일 해상도

        _fh, _fw = frame.shape[:2]
        _imgsz = _calc_imgsz(_fw)

        t0 = time.perf_counter()
        detections = model(frame, conf=cfg["DETECT_CONF"], imgsz=_imgsz, verbose=False, max_det=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_time += elapsed_ms

        _keep_dets = _manual_nms(detections[0].boxes, is_plate_model, cfg)

        det_results = []
        for _, _, det in _keep_dets:
            info = _filter_detection(det, sx, sy, ch_full, is_plate_model, cfg)
            if info is None:
                continue
            roi = _crop_roi(frame, info["bbox"], cfg)
            if roi is None:
                continue
            det_results.append(info)

        n_det = len(det_results)
        total_dets += n_det

        det_strs = []
        for d in det_results:
            x1, y1, x2, y2 = d["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            det_strs.append(f"[{x1},{y1},{x2},{y2}] {bw}x{bh} conf={d['conf']:.2f} {d['aspect_type']}")

        if det_results:
            output_lines.append(f"{fid:3d}  frame{fid:03d}  {n_det:5d}  {elapsed_ms:9.1f}  {det_strs[0]}")
            for ds in det_strs[1:]:
                output_lines.append(f"{'':3}  {'':6}  {'':5}  {'':9}  {ds}")
        else:
            output_lines.append(f"{fid:3d}  frame{fid:03d}  {0:5d}  {elapsed_ms:9.1f}  (탐지 없음)")

    cap.release()

    avg_dets = total_dets / MAX_FRAMES
    avg_time = total_time / MAX_FRAMES

    output_lines.append("-" * 70)
    output_lines.append(f"총 탐지 수: {total_dets}")
    output_lines.append(f"평균 탐지/프레임: {avg_dets:.2f}")
    output_lines.append(f"평균 추론 시간: {avg_time:.1f}ms (YOLO only)")
    output_lines.append("")

    return total_dets, avg_time


def main():
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("  파이프라인 1단계 YOLO 탐지 비교 테스트")
    output_lines.append(f"  영상: {VIDEO_PATH} | 프레임: {MAX_FRAMES}")
    output_lines.append("=" * 70)
    output_lines.append("")

    # [A] 파이프라인 테스트
    t0 = time.time()
    pipeline_dets, pipeline_avg_ms = test_pipeline(output_lines)
    pipeline_elapsed = time.time() - t0

    # [B] 기존 엔진 동일 모델 YOLO-only 테스트
    t0 = time.time()
    engine_dets, engine_avg_ms = test_engine_yolo_only(output_lines)
    engine_elapsed = time.time() - t0

    # 비교
    output_lines.append("=" * 70)
    output_lines.append("  [C] 비교 결과")
    output_lines.append("=" * 70)
    output_lines.append(f"  파이프라인 총 탐지: {pipeline_dets}")
    output_lines.append(f"  기존 엔진 총 탐지: {engine_dets}")

    if engine_dets > 0:
        diff_pct = abs(pipeline_dets - engine_dets) / engine_dets * 100
        output_lines.append(f"  차이: {diff_pct:.1f}%")
        if diff_pct > 20:
            output_lines.append(f"  ❌ 차이 20% 초과 → 문제 있음!")
        else:
            output_lines.append(f"  ✅ 차이 20% 이내 → 정상")
    else:
        output_lines.append("  (기존 엔진 탐지 0 → 비교 불가)")

    output_lines.append("")
    output_lines.append(f"  파이프라인 소요: {pipeline_elapsed:.1f}초")
    output_lines.append(f"  기존 엔진 소요: {engine_elapsed:.1f}초")

    # crops_test 파일 수
    if os.path.exists(CROPS_DIR):
        crops = [f for f in os.listdir(CROPS_DIR) if f.endswith(".png")]
        output_lines.append(f"  크롭 이미지: {len(crops)}장 ({CROPS_DIR}/)")
    output_lines.append("")

    # 출력
    full_text = "\n".join(output_lines)
    print(full_text)

    # 파일 저장
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"\n결과 저장: {RESULT_FILE}")


if __name__ == "__main__":
    freeze_support()
    main()
