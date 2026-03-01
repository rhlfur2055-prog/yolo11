# -*- coding: utf-8 -*-
"""
CMD 12: 메인 프로세스 - 영상 읽기 + YOLO 결과 표시
1단계 파이프라인: YOLO 탐지만 별도 프로세스로 분리

사용법:
  python pipeline_stage1.py                          # 웹캠 (기본)
  python pipeline_stage1.py --input movie/hiway.mp4  # 동영상 파일
  python pipeline_stage1.py --input 0                # 웹캠 명시
"""
import argparse
import time
import sys
import cv2
import numpy as np
from multiprocessing import Process, Queue, freeze_support
from queue import Full, Empty

from pipeline_common import CMD_STOP
from cmd5_yolo_worker import yolo_worker_loop


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 영상 열기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def open_video(source):
    """VideoCapture 열기 + 해상도 출력

    Args:
        source: 파일 경로 또는 웹캠 인덱스 (int)
    Returns:
        cv2.VideoCapture 객체 (열기 실패 시 None)
    """
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
# 결과 그리기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_results(frame, detections, inference_ms, display_fps):
    """bbox + ROI 미리보기 + 정보 오버레이 그리기

    Args:
        frame: 표시할 프레임 (직접 수정됨)
        detections: DetectionItem 리스트
        inference_ms: YOLO 추론 시간
        display_fps: 표시 FPS
    """
    # FPS + 추론시간 표시
    info_text = f"FPS: {display_fps:.1f} | YOLO: {inference_ms:.0f}ms | Det: {len(detections)}"
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 각 탐지에 대해 bbox 그리기
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        conf = det["conf"]
        aspect_type = det["aspect_type"]
        penalty = det["bbox_conf_penalty"]

        x1, y1, x2, y2 = bbox

        # 페널티에 따라 색상 결정
        if penalty >= 1.0:
            color = (0, 255, 0)    # 초록: 정상
        elif penalty >= 0.85:
            color = (0, 255, 255)  # 노랑: 중소형
        else:
            color = (0, 165, 255)  # 주황: 소형

        # bbox 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 라벨: conf + aspect_type
        label = f"{conf:.2f} {aspect_type}"
        label_y = max(y1 - 8, 20)
        cv2.putText(frame, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ROI 미리보기 (좌측 상단에 작은 썸네일)
        roi = det.get("roi")
        if roi is not None and roi.size > 0:
            thumb_h = 60
            roi_h, roi_w = roi.shape[:2]
            if roi_h > 0 and roi_w > 0:
                thumb_w = int(roi_w * thumb_h / roi_h)
                thumb_w = max(thumb_w, 10)
                thumb = cv2.resize(roi, (thumb_w, thumb_h))
                # 프레임 우측 상단에 ROI 썸네일 표시
                fh, fw = frame.shape[:2]
                tx1 = fw - thumb_w - 10
                ty1 = 50 + i * (thumb_h + 10)
                tx2 = tx1 + thumb_w
                ty2 = ty1 + thumb_h
                if tx1 >= 0 and ty2 < fh:
                    frame[ty1:ty2, tx1:tx2] = thumb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 표시 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def display_loop(cap, frame_queue, result_queue, cmd_queue):
    """메인 루프: frame 읽기 → frame_queue.put → result_queue.get → cv2.imshow

    Args:
        cap: cv2.VideoCapture 객체
        frame_queue: CMD 12 → CMD 5
        result_queue: CMD 5 → CMD 12
        cmd_queue: 제어 명령
    """
    frame_id = 0
    last_result = None
    fps_counter = 0
    fps_time = time.time()
    display_fps = 0.0
    total_detections = 0       # 총 탐지 수 카운터
    log_interval = 30          # N프레임마다 상태 로그

    # 원본 FPS에 맞춘 재생 속도 제어
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / video_fps

    print("[CMD12] 표시 루프 시작 (q: 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CMD12] 영상 끝")
            break

        frame_id += 1

        # frame_queue에 프레임 전송 (오버플로 시 오래된 프레임 버림)
        packet = {
            "frame": frame,
            "full_frame": None,  # 동일 해상도이므로 None
            "frame_id": frame_id,
            "camera_id": "CAM01",
        }
        try:
            frame_queue.put_nowait(packet)
        except Full:
            # 큐가 가득 차면 오래된 프레임 버리고 최신 넣기
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
            try:
                frame_queue.put_nowait(packet)
            except Full:
                pass

        # result_queue에서 결과 수신 (논블로킹)
        try:
            result = result_queue.get_nowait()
            last_result = result
        except Empty:
            pass

        # 결과 그리기 (이전 결과 재사용)
        display_frame = frame.copy()

        # FPS 계산
        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            display_fps = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        if last_result is not None:
            draw_results(display_frame, last_result["detections"],
                         last_result["inference_ms"], display_fps)
        else:
            # 아직 결과 없음 → FPS만 표시
            cv2.putText(display_frame, f"FPS: {display_fps:.1f} | 대기중...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Pipeline Stage 1 - YOLO Detection", display_frame)

        # 'q' 키로 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 또는 ESC
            print("[CMD12] 사용자 종료 요청")
            break

    cv2.destroyAllWindows()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    """argparse + 프로세스 생성/시작/종료"""
    parser = argparse.ArgumentParser(description="1단계 파이프라인: YOLO 탐지 분리")
    parser.add_argument("--input", type=str, default="movie/hiway.mp4",
                        help="입력 소스 (파일 경로 또는 웹캠 인덱스, 기본: movie/hiway.mp4)")
    args = parser.parse_args()

    # 입력 소스: 숫자면 웹캠 인덱스
    source = args.input
    if source.isdigit():
        source = int(source)

    # 영상 열기
    cap = open_video(source)
    if cap is None:
        sys.exit(1)

    # IPC 큐 생성
    frame_queue = Queue(maxsize=2)    # 최신 2장만 유지
    result_queue = Queue(maxsize=10)  # 탐지 결과 버퍼
    cmd_queue = Queue(maxsize=5)      # 제어 명령

    # YOLO 워커 프로세스 생성 + 시작
    worker = Process(
        target=yolo_worker_loop,
        args=(frame_queue, result_queue, cmd_queue),
        daemon=True,
        name="CMD5-YOLO-Worker",
    )
    print("[CMD12] YOLO 워커 프로세스 시작...")
    worker.start()

    try:
        # 메인 표시 루프 (메인 스레드에서 실행 — cv2.imshow 필수)
        display_loop(cap, frame_queue, result_queue, cmd_queue)
    finally:
        # 정리: 워커에 STOP 전송 + join
        print("[CMD12] 워커 프로세스 종료 중...")
        try:
            cmd_queue.put_nowait(CMD_STOP)
        except Full:
            pass

        cap.release()

        worker.join(timeout=5)
        if worker.is_alive():
            print("[CMD12] 워커 강제 종료")
            worker.terminate()
            worker.join(timeout=2)

        print("[CMD12] 파이프라인 종료 완료")


if __name__ == "__main__":
    freeze_support()
    main()
