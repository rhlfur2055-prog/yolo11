# -*- coding: utf-8 -*-
"""
realtime_plate.py — 실시간 번호판 인식
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
웹캠 또는 영상/RTSP 스트림에서 실시간으로 번호판을 탐지·인식합니다.

사용법:
    python realtime_plate.py                    # 웹캠(0) 실시간
    python realtime_plate.py --source 0          # 웹캠
    python realtime_plate.py --source video.mp4 # 영상 파일
    python realtime_plate.py --source rtsp://... # RTSP 스트림
    python realtime_plate.py --4k               # 고정밀 엔진(plate_recognition_4k, 느림)

종료: 화면 포커스 상태에서 'q' 키
"""

import sys
import time
import argparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np


def _parse_source(source: str):
    """웹캠은 정수 0으로, 그 외는 문자열 경로/URL."""
    s = (source or "0").strip()
    if s in ("0", "1", "2"):
        return int(s)
    return s


def run_realtime_pro(source, camera_id: str, save: bool, skip: int):
    """PlateEnginePro 기반 실시간 인식 (가벼움, 권장)."""
    from plate_engine_pro import PlateEnginePro, PlateEngineConfig

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[오류] 소스 열기 실패: {source}")
        return
    fps_target = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("[모델] PlateEnginePro 로딩...")
    engine = PlateEnginePro()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[오류] 소스 재오픈 실패: {source}")
        return

    writer = None
    if save:
        out_path = f"realtime_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_target, (w, h))
        print(f"[저장] {out_path}")

    frame_idx = 0
    total_plates = 0
    t0 = time.perf_counter()
    window_name = "실시간 번호판 인식 (Pro) [q: 종료]"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # N프레임마다만 인식 (속도 조절)
        if frame_idx % (skip + 1) == 0:
            results = engine.process_frame(frame, camera_id=camera_id)
        else:
            results = []

        total_plates += len(results)

        # 오버레이
        for r in results:
            x1, y1, x2, y2 = r.get("bbox", [0, 0, 0, 0])
            if len(r.get("bbox", [])) != 4:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (0, 255, 0)  # BGR 초록
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{r.get('plate', '')} {r.get('confidence', 0):.0%}"
            cv2.putText(
                frame, label, (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # FPS / 통계
        elapsed = time.perf_counter() - t0
        current_fps = frame_idx / elapsed if elapsed > 0 else 0
        info = f"FPS: {current_fps:.1f}  |  인식: {total_plates}건  |  [q] 종료"
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)

        if writer:
            writer.write(frame)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n[종료] {frame_idx}프레임, {total_plates}건 인식, 평균 FPS: {frame_idx/elapsed:.1f}")


def run_realtime_4k(source, skip: int):
    """plate_recognition_4k 기반 실시간 인식 (고정밀, 상대적으로 느림)."""
    from plate_recognition_4k import PlateRecognizer

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[오류] 소스 열기 실패: {source}")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("[모델] PlateRecognizer(4k) 로딩...")
    recognizer = PlateRecognizer()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[오류] 소스 재오픈 실패: {source}")
        return

    frame_idx = 0
    total_plates = 0
    t0 = time.perf_counter()
    window_name = "실시간 번호판 인식 (4k) [q: 종료]"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = []
        if frame_idx % (skip + 1) == 0:
            try:
                results = recognizer.process_frame(frame, frame_idx)
            except Exception:
                results = []

        total_plates += len(results)

        for r in results:
            bbox = r.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = (0, 255, 0) if r.get("is_valid_plate", False) else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{r.get('text', '')}"
            cv2.putText(
                frame, label, (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        elapsed = time.perf_counter() - t0
        current_fps = frame_idx / elapsed if elapsed > 0 else 0
        info = f"FPS: {current_fps:.1f}  |  인식: {total_plates}건  |  [q] 종료"
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[종료] {frame_idx}프레임, {total_plates}건 인식, 평균 FPS: {frame_idx/elapsed:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="실시간 번호판 인식 (웹캠/영상/RTSP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", "-s", default="0",
        help="영상 소스: 0(웹캠), 1(2번 카메라), 파일경로, RTSP URL (기본: 0)",
    )
    parser.add_argument(
        "--camera", default="CAM01",
        help="카메라 ID (Pro 엔진 DB 기록용, 기본: CAM01)",
    )
    parser.add_argument(
        "--4k", dest="use_4k", action="store_true",
        help="plate_recognition_4k 엔진 사용 (고정밀, 느림)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="실시간 녹화 저장 안 함 (Pro 모드)",
    )
    parser.add_argument(
        "--skip", type=int, default=0,
        help="인식 스킵 간격 (0=매프레임, 1=1프레임마다, 2=2프레임마다 … 기본: 0)",
    )
    args = parser.parse_args()

    source = _parse_source(args.source)
    print()
    print("=" * 50)
    print("  실시간 번호판 인식")
    print("=" * 50)
    print(f"  소스: {source}")
    print(f"  엔진: {'plate_recognition_4k' if args.use_4k else 'PlateEnginePro'}")
    print("  종료: 창 포커스 상태에서 [q] 키")
    print("=" * 50)
    print()

    if args.use_4k:
        run_realtime_4k(source, args.skip)
    else:
        run_realtime_pro(source, args.camera, save=not args.no_save, skip=args.skip)


if __name__ == "__main__":
    main()
