"""
run_safeplate.py — SafePlate 4K 통합 실행 스크립트 (계단 B-4)

역할: ShockSimulator + DepartureDetector + EvidencePackage를 EventBus로 연결하고
      SimulationFramework 위에서 실행

비유: 지휘관. 충격 감지→이탈 추적→증거 저장 3단 파이프라인의 모든 부품을 조립하고 시동을 건다

실행 방법:
    python -m simulation.safeplate.run_safeplate --video movie/emergency_test.mp4 --threshold 2 --scale 1.5
    python -m simulation.safeplate.run_safeplate --video movie/hiway.mp4
    python -m simulation.safeplate.run_safeplate --video movie/emergency_test.mp4 --headless --shock-at 150

옵션:
    --video <경로>       입력 영상 파일 경로 (필수)
    --scale <배율>       화면 표시 배율 (기본 1.0)
    --threshold <초>     충격 후 이탈 판정 타임아웃 (기본 3.0초)
    --pre-buffer <초>    충격 전 버퍼 시간 (기본 10.0초)
    --post-buffer <초>   충격 후 버퍼 시간 (기본 15.0초)
    --output <폴더>      증거 저장 폴더 (기본 ./evidence_output)
    --headless           GUI 없이 실행 (자동 테스트용)
    --shock-at <프레임>  headless 모드에서 자동 충격 트리거 프레임 (기본 150)
    --shock-end <프레임> headless 모드에서 자동 충격 종료 프레임 (기본 shock-at+500)
    --max-frames <수>    최대 처리 프레임 수 (headless 모드, 기본 영상 끝까지)
    --conf <값>          YOLO confidence threshold (기본 엔진 기본값)

기존 코드 수정: 없음
Mock 데이터: 없음
"""

import sys
import os
import argparse
import time

import cv2
import numpy as np

# 프로젝트 루트를 path에 추가 (simulation/ 하위에서 import 가능하도록)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from simulation.simulation_framework import SimulationFramework, EventBus
from simulation.safeplate.shock_simulator import ShockSimulator
from simulation.safeplate.departure_detector import DepartureDetector
from simulation.safeplate.evidence_package import EvidencePackage
# NightEnhancer는 --night 또는 --night-auto 사용 시에만 지연 import (메모리 절약)
NightEnhancer = None


def parse_args() -> argparse.Namespace:
    """커맨드라인 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="SafePlate 4K — 물피도주 의심 차량 자동 채증 시뮬레이션",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--video", required=True,
                        help="입력 영상 파일 경로")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="화면 표시 배율 (기본 1.0)")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="이탈 판정 타임아웃 (초, 기본 3.0)")
    parser.add_argument("--pre-buffer", type=float, default=10.0,
                        help="충격 전 버퍼 시간 (초, 기본 10.0)")
    parser.add_argument("--post-buffer", type=float, default=15.0,
                        help="충격 후 버퍼 시간 (초, 기본 15.0)")
    parser.add_argument("--output", default="./evidence_output",
                        help="증거 저장 폴더 (기본 ./evidence_output)")
    parser.add_argument("--headless", action="store_true",
                        help="GUI 없이 실행 (자동 테스트용)")
    parser.add_argument("--shock-at", type=int, default=150,
                        help="headless 모드에서 자동 충격 트리거 프레임 (기본 150)")
    parser.add_argument("--shock-end", type=int, default=None,
                        help="headless 모드에서 자동 충격 종료 프레임 (기본 shock-at+500)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="최대 처리 프레임 수 (headless)")
    parser.add_argument("--conf", type=float, default=None,
                        help="YOLO confidence threshold")
    parser.add_argument("--night", action="store_true",
                        help="야간 모드 강제 활성화")
    parser.add_argument("--night-auto", action="store_true", default=False,
                        help="야간 모드 자동 감지 활성화")
    parser.add_argument("--night-threshold", type=float, default=80.0,
                        help="야간 판정 밝기 임계값 (기본 80.0)")
    parser.add_argument("--night-gamma", type=float, default=1.8,
                        help="야간 감마 보정 값 (기본 1.8)")

    return parser.parse_args()


def run_gui_mode(args: argparse.Namespace) -> None:
    """GUI 모드 실행 — 대화형 S/E 키 입력으로 충격 감지/종료"""

    print("\n" + "=" * 60)
    print("  SafePlate 4K — 물피도주 의심 차량 자동 채증")
    print("=" * 60)

    # ── 1. SimulationFramework 초기화 ──
    sim = SimulationFramework(
        video_path=args.video,
        window_name="SafePlate 4K",
        display_scale=args.scale,
    )

    # YOLO confidence 설정
    if args.conf is not None:
        sim.engine.detect_conf = args.conf
        print(f"  YOLO conf: {args.conf}")

    # ── 2. SafePlate 모듈 초기화 + EventBus 연결 ──
    shock = ShockSimulator(
        event_bus=sim.event_bus,
        fps=sim.fps,
        pre_buffer_sec=args.pre_buffer,
        post_buffer_sec=args.post_buffer,
    )

    departure = DepartureDetector(
        event_bus=sim.event_bus,
        frame_width=sim.frame_width,
        frame_height=sim.frame_height,
        vanish_timeout_sec=args.threshold,
    )

    evidence = EvidencePackage(
        event_bus=sim.event_bus,
        shock_simulator=shock,
        output_dir=args.output,
        fps=sim.fps,
        frame_width=sim.frame_width,
        frame_height=sim.frame_height,
    )

    # ── 야간 모드 초기화 (선택적) ──
    night = None
    if args.night or args.night_auto:
        from simulation.safeplate.night_enhancer import NightEnhancer as _NE
        night_config = {
            "brightness_threshold": args.night_threshold,
            "gamma": args.night_gamma,
            "auto_detect": True,
        }
        night = _NE(event_bus=sim.event_bus, config=night_config)
        if args.night:
            night.force_night_mode(True)

    # ── 3. 오버레이 콜백 등록 (순서: 충격→이탈→증거→야간) ──
    sim.overlay_callbacks.append(shock.draw_shock_overlay)
    sim.overlay_callbacks.append(departure.draw_departure_overlay)
    sim.overlay_callbacks.append(evidence.draw_export_overlay)
    if night:
        sim.overlay_callbacks.append(night.draw_night_overlay)

    # ── 4. 번호판 인식 로그 ──
    sim.event_bus.subscribe(
        "plate_recognized",
        lambda data: print(
            f"  [번호판] {data['plate']} "
            f"(신뢰도: {data['confidence']:.0%}) "
            f"@ {data['timestamp']:.1f}s"
        )
    )

    # ── 5. 증거 내보내기 완료 로그 ──
    sim.event_bus.subscribe(
        "EVIDENCE_EXPORTED",
        lambda data: print(
            f"\n  증거 패키지 저장 완료: {data['plate']} → {data['folder']}"
        )
    )

    print("\n  조작 방법:")
    print("    [S] 충격 감지 시작 → 이탈 차량 추적")
    print("    [E] 충격 감지 종료")
    print("    [N] 야간 모드 토글 (자동→수동ON→수동OFF→자동)")
    print("    [SPACE] 일시정지/재개")
    print("    [Q] 종료")
    print("=" * 60 + "\n")

    # ── 6. 시뮬레이션 실행 ──
    sim.run()

    # ── 7. 결과 요약 ──
    _print_summary(departure, evidence)


def run_headless_mode(args: argparse.Namespace) -> None:
    """Headless 모드 실행 — GUI 없이 자동 충격→이탈→증거 파이프라인

    동작 흐름:
        1. 영상 프레임을 순차 처리
        2. shock-at 프레임에서 자동 충격 트리거
        3. shock-end 프레임에서 자동 충격 종료
        4. YOLO 감지 + OCR → 이탈 감지 → 증거 저장
        5. 결과 요약 출력
    """

    print("\n" + "=" * 60)
    print("  SafePlate 4K — HEADLESS 자동 테스트")
    print("=" * 60)

    # ── 1. 영상 열기 ──
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] 영상 파일을 열 수 없습니다: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  영상: {args.video}")
    print(f"  해상도: {frame_width}x{frame_height} @ {fps:.1f}fps")
    print(f"  총 프레임: {total_frames}")

    shock_at = args.shock_at
    shock_end = args.shock_end if args.shock_end else shock_at + 500
    max_frames = args.max_frames if args.max_frames else total_frames

    print(f"  충격 트리거: 프레임 #{shock_at}")
    print(f"  충격 종료:   프레임 #{shock_end}")
    print(f"  최대 프레임: {max_frames}")

    # ── 2. EventBus + YOLO 엔진 ──
    event_bus = EventBus()

    # YOLO 엔진 import
    from plate_engine_pro import PlateEnginePro
    engine = PlateEnginePro()

    # YOLO confidence 설정
    if args.conf is not None:
        engine.detect_conf = args.conf
        print(f"  YOLO conf: {args.conf}")

    # ── 3. SafePlate 모듈 연결 ──
    shock = ShockSimulator(
        event_bus=event_bus,
        fps=fps,
        pre_buffer_sec=args.pre_buffer,
        post_buffer_sec=args.post_buffer,
    )

    departure = DepartureDetector(
        event_bus=event_bus,
        frame_width=frame_width,
        frame_height=frame_height,
        vanish_timeout_sec=args.threshold,
    )

    evidence = EvidencePackage(
        event_bus=event_bus,
        shock_simulator=shock,
        output_dir=args.output,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    # ── 야간 모드 (선택적) ──
    night = None
    if args.night or args.night_auto:
        from simulation.safeplate.night_enhancer import NightEnhancer as _NE
        night_config = {
            "brightness_threshold": args.night_threshold,
            "gamma": args.night_gamma,
            "auto_detect": not args.night,
        }
        night = _NE(event_bus=event_bus, config=night_config)
        if args.night:
            night.force_night_mode(True)

    # ── 4. 이벤트 로거 ──
    detected_plates = []

    def _on_plate(data):
        plate = data.get("plate", "")
        conf = data.get("confidence", 0.0)
        ts = data.get("timestamp", 0.0)
        detected_plates.append(data)
        print(f"  [번호판] {plate} (신뢰도: {conf:.0%}) @ {ts:.1f}s")

    event_bus.subscribe("plate_recognized", _on_plate)

    export_results = []

    def _on_export(data):
        export_results.append(data)
        print(f"\n  증거 패키지 저장: {data['plate']} → {data['folder']}")

    event_bus.subscribe("EVIDENCE_EXPORTED", _on_export)

    print("\n" + "-" * 60)
    print("  프레임 처리 시작...")
    print("-" * 60)

    # ── 5. 메인 프레임 루프 (GUI 없음) ──
    start_time = time.time()
    frame_idx = 0
    shock_triggered = False
    shock_ended = False

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # 프레임 이벤트 발행
        event_bus.publish("frame_read", {
            "frame": frame,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
        })

        # 야간 모드 보정 적용
        process_frame = frame
        if night and night.is_night_mode:
            process_frame = night.enhance_frame(frame)

        # YOLO 감지
        detections = engine.process_frame(process_frame)

        # 감지 결과 이벤트 발행
        event_bus.publish("detection_result", {
            "detections": detections,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
        })

        # 개별 번호판 인식 이벤트
        for det in detections:
            plate_text = det.get("plate", "")
            if plate_text:
                event_bus.publish("plate_recognized", {
                    "plate": plate_text,
                    "confidence": det.get("confidence", 0.0),
                    "bbox": det.get("bbox", []),
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                })

        # ── 자동 충격 트리거 ──
        if frame_idx == shock_at and not shock_triggered:
            print(f"\n  >>> 자동 충격 트리거 (프레임 #{frame_idx}) <<<")
            shock.trigger_shock(frame_idx=frame_idx, timestamp=timestamp)
            shock_triggered = True

        # ── 자동 충격 종료 ──
        if frame_idx == shock_end and not shock_ended and shock_triggered:
            print(f"\n  >>> 자동 충격 종료 (프레임 #{frame_idx}) <<<")
            shock.trigger_shock_end(frame_idx=frame_idx, timestamp=timestamp)
            shock_ended = True

        # 진행률 표시 (100프레임마다)
        if frame_idx > 0 and frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            pct = frame_idx / max_frames * 100
            print(f"  ... 프레임 {frame_idx}/{max_frames} ({pct:.0f}%) — {fps_actual:.1f} fps")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - start_time

    print(f"\n  처리 완료: {frame_idx} 프레임 / {elapsed:.1f}초 ({frame_idx/elapsed:.1f} fps)")

    # 야간 모드 통계
    if night:
        night_stats = night.get_stats()
        if night_stats["enhanced_frames"] > 0:
            ratio = night_stats["enhancement_ratio"]
            print(f"  야간 보정: {night_stats['enhanced_frames']}/{night_stats['total_frames']} 프레임 ({ratio:.0%})")
            print(f"  평균 밝기: {night_stats['avg_brightness']:.1f}")

    # ── 6. 결과 요약 ──
    _print_summary(departure, evidence)

    # ── 7. 테스트 결과 반환 ──
    if export_results:
        print(f"\n  [PASS] 증거 패키지 {len(export_results)}건 생성 완료")
    else:
        print(f"\n  [INFO] 이탈 차량 감지 없음 — 증거 패키지 미생성")

    # 이탈 차량이 있는데 증거가 없으면 경고
    if departure.get_departed_count() > 0 and not export_results:
        print(f"  [WARN] 이탈 차량 {departure.get_departed_count()}대 감지되었으나 증거 패키지 미생성")
        print(f"         (충격 전후 프레임 버퍼가 비어 있을 수 있음)")


def _print_summary(departure: DepartureDetector, evidence: EvidencePackage) -> None:
    """실행 결과 요약 출력"""
    print("\n" + "=" * 60)
    print("  SafePlate 4K — 실행 결과 요약")
    print("=" * 60)

    # 추적 결과
    tracking_count = departure.get_tracking_count()
    departed_count = departure.get_departed_count()

    print(f"\n  차량 추적:")
    print(f"    현재 추적 중: {tracking_count}대")
    print(f"    이탈 감지:    {departed_count}대")

    if departed_count > 0:
        print(f"\n  이탈 차량 목록:")
        for i, dep in enumerate(departure.get_departed_vehicles(), 1):
            plate = dep.get("plate", "unknown")
            direction = dep.get("departure_direction", "unknown")
            conf = dep.get("confidence", 0.0)
            det_count = dep.get("detection_count", 0)

            dir_map = {
                "left": "좌측 이탈",
                "right": "우측 이탈",
                "top": "상방 이탈",
                "bottom": "하방 이탈",
                "vanished": "화면 소멸",
            }
            dir_label = dir_map.get(direction, direction)

            print(f"    [{i}] {plate}")
            print(f"        방향: {dir_label} | 신뢰도: {conf:.0%} | 감지: {det_count}회")

    # 증거 패키지 결과
    export_count = evidence.get_export_count()
    print(f"\n  증거 패키지:")
    print(f"    생성 건수: {export_count}건")

    if export_count > 0:
        for i, exp in enumerate(evidence.get_export_history(), 1):
            plate = exp.get("plate", "unknown")
            folder = exp.get("folder", "")
            frames = exp.get("total_frames", 0)
            print(f"    [{i}] {plate} — {frames} 프레임")
            print(f"        {folder}")

    print("\n" + "=" * 60)


# ═══════════════════════════════════════════════
# CLI 진입점
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()

    # 영상 파일 존재 확인
    if not os.path.exists(args.video):
        print(f"[ERROR] 영상 파일 없음: {args.video}")
        sys.exit(1)

    if args.headless:
        run_headless_mode(args)
    else:
        run_gui_mode(args)
