"""
run_goldentime.py — 골든타임 2.0 통합 실행 스크립트 (계단 6)

역할: 계단 1~5의 모든 모듈을 하나로 연결하는 메인 실행 파일.
      각 모듈은 이미 만들어져 있으므로 import해서 조합만 함.

비유: 오케스트라 지휘자. 각 악기(모듈)는 이미 준비되어 있고,
     지휘자는 연주 순서(이벤트 흐름)와 타이밍을 맞추는 역할.

전체 파이프라인:
    1. 영상 재생 + YOLO 번호판 감지      (simulation_framework — 계단 1)
    2. 'S' 키로 사이렌 트리거              (siren_trigger — 계단 2)
    3. 15초 이상 감지 번호판 채증          (plate_evidence — 계단 3)
    4. bbox 기반 거리 판정                 (distance_checker — 계단 4)
    5. 증거 패키지 자동 저장               (evidence_export — 계단 5)

실행:
    python run_goldentime.py --video movie/hiway.mp4
    python run_goldentime.py --video movie/hiway.mp4 --scale 0.5
    python run_goldentime.py --video movie/hiway.mp4 --threshold 5
    python run_goldentime.py --video movie/hiway.mp4 --output ./my_evidence

검증:
    python test_ocr_accuracy.py  # 12/12 통과 필수

기존 코드 수정: 없음 — 모든 import는 기존 파일 그대로 사용
"""

import sys
import os
import argparse
from datetime import datetime

# ──────────────────────────────────────────────
# 모듈 import (계단 1~5, 전부 기존 코드 — 새로 만드는 것 없음)
# ──────────────────────────────────────────────

from simulation.simulation_framework import SimulationFramework
from simulation.goldentime.siren_trigger import SirenTrigger
from simulation.goldentime.plate_evidence import PlateEvidence
from simulation.goldentime.distance_checker import DistanceChecker
from simulation.goldentime.evidence_export import EvidenceExport


def parse_args():
    """커맨드라인 인자 파싱

    Returns:
        argparse.Namespace — 파싱된 인자들
    """
    parser = argparse.ArgumentParser(
        description="골든타임 2.0 — 긴급차량 통행방해 증거 확보 시뮬레이션",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
조작법:
  S     사이렌 감지 시작 (추적 + 거리 판정 시작)
  E     사이렌 종료
  SPACE 일시정지 / 재개
  q     종료

파이프라인:
  S키 → 사이렌 감지 → 번호판 추적 시작
  → 15초 연속 감지 → 채증 시작 (전 30초 + 후 30초)
  → 거리 미확보 판정 → 증거 패키지 자동 저장
  → evidence_output/ 폴더에 영상 + JSON + 스크린샷 + 리포트

예시:
  python run_goldentime.py --video movie/hiway.mp4
  python run_goldentime.py --video movie/hiway.mp4 --scale 0.7
  python run_goldentime.py --video movie/hiway.mp4 --threshold 5 --violation-sec 3
        """
    )

    parser.add_argument(
        "--video", required=True,
        help="입력 영상 파일 경로 (예: movie/hiway.mp4)"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="화면 표시 배율 (기본: 1.0)"
    )
    parser.add_argument(
        "--output", default="./evidence_output",
        help="증거 패키지 저장 경로 (기본: ./evidence_output)"
    )
    parser.add_argument(
        "--threshold", type=float, default=15.0,
        help="채증 기준 연속 감지 시간 (초, 기본: 15)"
    )
    parser.add_argument(
        "--violation-sec", type=float, default=5.0,
        help="거리 미확보 판정 기준 시간 (초, 기본: 5)"
    )
    parser.add_argument(
        "--close-ratio", type=float, default=0.0008,
        help="가까움 판정 bbox 비율 임계값 (기본: 0.0008)"
    )
    parser.add_argument(
        "--pre-buffer", type=float, default=30.0,
        help="채증 전 버퍼 시간 (초, 기본: 30)"
    )
    parser.add_argument(
        "--post-buffer", type=float, default=30.0,
        help="채증 후 녹화 시간 (초, 기본: 30)"
    )

    return parser.parse_args()


def print_banner():
    """시작 배너 출력"""
    print()
    print("=" * 62)
    print("  🚑 골든타임 2.0 — 긴급차량 통행방해 증거 확보 시뮬레이션")
    print("=" * 62)
    print()


def print_config(args, sim):
    """설정 요약 출력"""
    print("-" * 62)
    print("  설정")
    print("-" * 62)
    print(f"  영상:         {args.video}")
    print(f"  해상도:       {sim.frame_width}x{sim.frame_height}")
    print(f"  FPS:          {sim.fps:.1f}")
    print(f"  화면 배율:    {args.scale}")
    print(f"  채증 기준:    {args.threshold}초 연속 감지")
    print(f"  거리 기준:    가까움 {args.violation_sec}초 이상 (비율 >= {args.close_ratio * 100:.2f}%)")
    print(f"  버퍼:         전 {args.pre_buffer}초 + 후 {args.post_buffer}초 = {args.pre_buffer + args.post_buffer:.0f}초")
    print(f"  증거 저장:    {os.path.abspath(args.output)}")
    print("-" * 62)


def print_controls():
    """조작법 출력"""
    print()
    print("-" * 62)
    print("  조작법")
    print("-" * 62)
    print("  S     사이렌 감지 시작")
    print("  E     사이렌 종료")
    print("  SPACE 일시정지 / 재개")
    print("  q     종료")
    print("-" * 62)
    print()


def main():
    """메인 실행 함수

    모든 모듈을 생성하고 이벤트 버스로 연결.
    각 모듈은 이미 존재하는 파일에서 import — 새 코드 없음.

    시간복잡도: O(N * D), N = 총 프레임 수, D = 프레임당 처리 시간
    """
    args = parse_args()

    # ── 배너 ──
    print_banner()

    # ── 1. 시뮬레이션 프레임워크 (계단 1) ──
    try:
        sim = SimulationFramework(
            video_path=args.video,
            window_name="GoldenTime 2.0 Simulation",
            display_scale=args.scale,
        )
    except FileNotFoundError as e:
        print(f"\n  ❌ 에러: {e}")
        print(f"     영상 파일 경로를 확인해주세요.")
        sys.exit(1)

    # ── 설정 출력 ──
    print_config(args, sim)

    # ── 2. 사이렌 트리거 (계단 2) ──
    siren = SirenTrigger(sim.event_bus)

    # ── 3. 번호판 채증 (계단 3) ──
    evidence_config = {
        "continuous_threshold_sec": args.threshold,
        "pre_buffer_sec": args.pre_buffer,
        "post_buffer_sec": args.post_buffer,
        "gap_tolerance_sec": 2.0,
        "min_confidence": 0.5,
    }
    evidence = PlateEvidence(sim.event_bus, fps=sim.fps, config=evidence_config)

    # ── 4. 거리 판정 (계단 4) ──
    distance_config = {
        "close_ratio_threshold": args.close_ratio,
        "violation_duration_sec": args.violation_sec,
        "gap_tolerance_sec": 2.0,
        "distance_levels": [
            (0.005,  "~1m",  (0, 0, 255)),
            (0.002,  "~3m",  (0, 100, 255)),
            (0.0008, "~5m",  (0, 200, 255)),
            (0.0004, "~10m", (0, 255, 200)),
            (0.0,    ">10m", (0, 255, 0)),
        ],
    }
    distance = DistanceChecker(
        sim.event_bus,
        frame_width=sim.frame_width,
        frame_height=sim.frame_height,
        config=distance_config,
    )

    # ── 5. 증거 내보내기 (계단 5) ──
    exporter = EvidenceExport(
        sim.event_bus,
        output_dir=args.output,
        fps=sim.fps,
        frame_width=sim.frame_width,
        frame_height=sim.frame_height,
    )

    # ── 오버레이 등록 (화면 그리기 순서) ──
    sim.overlay_callbacks.append(siren.draw_siren_overlay)       # 상단 사이렌 경고
    sim.overlay_callbacks.append(evidence.draw_evidence_overlay)  # bbox + 추적 패널
    sim.overlay_callbacks.append(distance.draw_distance_overlay)  # 거리 패널
    sim.overlay_callbacks.append(exporter.draw_export_overlay)    # 저장 상태

    # ── 디버그 콜백 (콘솔 출력) ──
    sim.event_bus.subscribe(
        "plate_recognized",
        lambda data: print(
            f"  [번호판] {data['plate']} "
            f"(신뢰도: {data['confidence']:.0%}) "
            f"@ {data['timestamp']:.1f}s"
        )
    )

    # ── 종료 시 요약 출력 콜백 ──
    def on_export(data):
        print(f"\n  {'='*50}")
        print(f"  📦 증거 패키지 저장됨: {data.get('folder', '')}")
        print(f"  {'='*50}")

    sim.event_bus.subscribe("EVIDENCE_EXPORTED", on_export)

    # ── 조작법 출력 + 실행 ──
    print_controls()

    start_time = datetime.now()
    print(f"  ▶ 시뮬레이션 시작: {start_time.strftime('%H:%M:%S')}")
    print()

    sim.run()

    # ── 종료 요약 ──
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print()
    print("=" * 62)
    print("  시뮬레이션 종료 요약")
    print("=" * 62)
    print(f"  실행 시간:    {elapsed:.1f}초")
    print(f"  처리 프레임:  {sim.frame_idx}")
    print(f"  채증 대상:    {len(evidence.evidence_targets)}건")
    print(f"  거리 위반:    {len(distance.violations)}건")
    print(f"  증거 저장:    {exporter.export_count}건")

    if exporter.export_history:
        print()
        print("  저장된 증거 패키지:")
        for i, exp in enumerate(exporter.export_history, 1):
            viol = "⚠️ 위반" if exp.get("has_violation") else "✅ 정상"
            print(f"    [{i}] {exp['plate']} — {viol} — {exp['folder']}")

    if distance.violations:
        print()
        print("  거리 미확보 위반 차량:")
        for plate, rec in distance.violations.items():
            print(f"    ⚠️ {plate}: 가까움 {rec.close_duration:.1f}초 ({rec.last_distance_label})")

    print()
    print(f"  증거 폴더:    {os.path.abspath(args.output)}")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
