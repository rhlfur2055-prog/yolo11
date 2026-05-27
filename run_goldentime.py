"""run_goldentime.py — 골든타임 2.0 통합 실행 스크립트 (계단 6)

역할:
    계단 1~5의 모든 모듈을 import해 이벤트 버스로 묶는 오케스트레이션 엔트리포인트.
    각 도메인 모듈(시뮬레이션/사이렌/채증/거리/내보내기)은 ``simulation/`` 패키지에
    이미 존재하며, 이 파일은 그 조합과 타이밍만 책임진다.

파이프라인:
    1. 영상 재생 + YOLO 번호판 감지   (simulation_framework — 계단 1)
    2. 'S' 키로 사이렌 트리거          (siren_trigger        — 계단 2)
    3. 일정 시간 연속 감지 시 채증     (plate_evidence       — 계단 3)
    4. bbox 비율 기반 거리 판정        (distance_checker     — 계단 4)
    5. 증거 패키지 자동 저장           (evidence_export      — 계단 5)

실행:
    python run_goldentime.py --video movie/hiway.mp4
    python run_goldentime.py --video movie/hiway.mp4 --scale 0.5
    python run_goldentime.py --video movie/hiway.mp4 --threshold 5
    python run_goldentime.py --video movie/hiway.mp4 --output ./my_evidence

회귀 테스트:
    python test_ocr_accuracy.py
"""

from __future__ import annotations

# ── stdlib ─────────────────────────────────────────────────
import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# ── local (simulation 패키지는 read-only 의존) ──────────────
from simulation.simulation_framework import SimulationFramework
from simulation.goldentime.siren_trigger import SirenTrigger
from simulation.goldentime.plate_evidence import PlateEvidence
from simulation.goldentime.distance_checker import DistanceChecker
from simulation.goldentime.evidence_export import EvidenceExport


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 매직 넘버 상수화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WINDOW_NAME: str = "GoldenTime 2.0 Simulation"
DEFAULT_OUTPUT_DIR: str = "./evidence_output"
DEFAULT_DISPLAY_SCALE: float = 1.0

# 채증/거리 판정 시간(초)
DEFAULT_THRESHOLD_SEC: float = 15.0
DEFAULT_VIOLATION_SEC: float = 5.0
DEFAULT_PRE_BUFFER_SEC: float = 30.0
DEFAULT_POST_BUFFER_SEC: float = 30.0

# bbox 면적 비율 임계값 (frame 대비)
DEFAULT_CLOSE_RATIO: float = 0.0008

# 끊긴 감지 허용 간격 + OCR 신뢰도 하한
GAP_TOLERANCE_SEC: float = 2.0
MIN_OCR_CONFIDENCE: float = 0.5

# 거리 단계 표 — (면적비 하한, 라벨, BGR 색상)
DISTANCE_LEVELS: list[tuple[float, str, tuple[int, int, int]]] = [
    (0.0050, "~1m",  (0, 0, 255)),
    (0.0020, "~3m",  (0, 100, 255)),
    (0.0008, "~5m",  (0, 200, 255)),
    (0.0004, "~10m", (0, 255, 200)),
    (0.0000, ">10m", (0, 255, 0)),
]

# 콘솔 출력 폭
BANNER_WIDTH: int = 62
SUB_WIDTH: int = 50

# 종료 코드
EXIT_OK: int = 0
EXIT_VIDEO_NOT_FOUND: int = 1


logger = logging.getLogger("goldentime")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모듈 묶음 (5개 단계 인스턴스를 한 그룹으로)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class GoldenTimeModules:
    """골든타임 시뮬레이션을 구성하는 5개 단계 인스턴스."""

    sim: SimulationFramework
    siren: SirenTrigger
    evidence: PlateEvidence
    distance: DistanceChecker
    exporter: EvidenceExport


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI 파싱
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """커맨드라인 인자 파싱."""
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
        """,
    )
    parser.add_argument(
        "--video", required=True,
        help="입력 영상 파일 경로 (예: movie/hiway.mp4)",
    )
    parser.add_argument(
        "--scale", type=float, default=DEFAULT_DISPLAY_SCALE,
        help=f"화면 표시 배율 (기본: {DEFAULT_DISPLAY_SCALE})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR,
        help=f"증거 패키지 저장 경로 (기본: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD_SEC,
        help=f"채증 기준 연속 감지 시간 (초, 기본: {DEFAULT_THRESHOLD_SEC:g})",
    )
    parser.add_argument(
        "--violation-sec", type=float, default=DEFAULT_VIOLATION_SEC,
        help=f"거리 미확보 판정 기준 시간 (초, 기본: {DEFAULT_VIOLATION_SEC:g})",
    )
    parser.add_argument(
        "--close-ratio", type=float, default=DEFAULT_CLOSE_RATIO,
        help=f"가까움 판정 bbox 비율 임계값 (기본: {DEFAULT_CLOSE_RATIO})",
    )
    parser.add_argument(
        "--pre-buffer", type=float, default=DEFAULT_PRE_BUFFER_SEC,
        help=f"채증 전 버퍼 시간 (초, 기본: {DEFAULT_PRE_BUFFER_SEC:g})",
    )
    parser.add_argument(
        "--post-buffer", type=float, default=DEFAULT_POST_BUFFER_SEC,
        help=f"채증 후 녹화 시간 (초, 기본: {DEFAULT_POST_BUFFER_SEC:g})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="번호판 인식 디버그 로그 출력",
    )
    return parser.parse_args(argv)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 출력 유틸 (사용자 대상 CLI — print 유지, 로그/디버그는 logger 사용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_banner() -> None:
    """시작 배너 출력."""
    print()
    print("=" * BANNER_WIDTH)
    print("  🚑 골든타임 2.0 — 긴급차량 통행방해 증거 확보 시뮬레이션")
    print("=" * BANNER_WIDTH)
    print()


def print_config(args: argparse.Namespace, sim: SimulationFramework) -> None:
    """실행 설정 요약 출력."""
    print("-" * BANNER_WIDTH)
    print("  설정")
    print("-" * BANNER_WIDTH)
    print(f"  영상:         {args.video}")
    print(f"  해상도:       {sim.frame_width}x{sim.frame_height}")
    print(f"  FPS:          {sim.fps:.1f}")
    print(f"  화면 배율:    {args.scale}")
    print(f"  채증 기준:    {args.threshold}초 연속 감지")
    print(
        f"  거리 기준:    가까움 {args.violation_sec}초 이상 "
        f"(비율 >= {args.close_ratio * 100:.2f}%)"
    )
    print(
        f"  버퍼:         전 {args.pre_buffer}초 + 후 {args.post_buffer}초 "
        f"= {args.pre_buffer + args.post_buffer:.0f}초"
    )
    print(f"  증거 저장:    {os.path.abspath(args.output)}")
    print("-" * BANNER_WIDTH)


def print_controls() -> None:
    """조작법 출력."""
    print()
    print("-" * BANNER_WIDTH)
    print("  조작법")
    print("-" * BANNER_WIDTH)
    print("  S     사이렌 감지 시작")
    print("  E     사이렌 종료")
    print("  SPACE 일시정지 / 재개")
    print("  q     종료")
    print("-" * BANNER_WIDTH)
    print()


def print_summary(
    args: argparse.Namespace,
    modules: GoldenTimeModules,
    elapsed_sec: float,
) -> None:
    """시뮬레이션 종료 요약 + 위반 차량 리스트 출력."""
    sim = modules.sim
    evidence = modules.evidence
    distance = modules.distance
    exporter = modules.exporter

    print()
    print("=" * BANNER_WIDTH)
    print("  시뮬레이션 종료 요약")
    print("=" * BANNER_WIDTH)
    print(f"  실행 시간:    {elapsed_sec:.1f}초")
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
            print(
                f"    ⚠️ {plate}: 가까움 {rec.close_duration:.1f}초 "
                f"({rec.last_distance_label})"
            )

    print()
    print(f"  증거 폴더:    {os.path.abspath(args.output)}")
    print("=" * BANNER_WIDTH)
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모듈 빌드 + 콜백 와이어링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_modules(args: argparse.Namespace) -> GoldenTimeModules:
    """5개 단계 모듈을 인자값으로 인스턴스화.

    Raises:
        FileNotFoundError: ``args.video`` 경로가 존재하지 않을 때.
    """
    sim = SimulationFramework(
        video_path=args.video,
        window_name=WINDOW_NAME,
        display_scale=args.scale,
    )
    siren = SirenTrigger(sim.event_bus)
    evidence = PlateEvidence(
        sim.event_bus,
        fps=sim.fps,
        config={
            "continuous_threshold_sec": args.threshold,
            "pre_buffer_sec": args.pre_buffer,
            "post_buffer_sec": args.post_buffer,
            "gap_tolerance_sec": GAP_TOLERANCE_SEC,
            "min_confidence": MIN_OCR_CONFIDENCE,
        },
    )
    distance = DistanceChecker(
        sim.event_bus,
        frame_width=sim.frame_width,
        frame_height=sim.frame_height,
        config={
            "close_ratio_threshold": args.close_ratio,
            "violation_duration_sec": args.violation_sec,
            "gap_tolerance_sec": GAP_TOLERANCE_SEC,
            "distance_levels": DISTANCE_LEVELS,
        },
    )
    exporter = EvidenceExport(
        sim.event_bus,
        output_dir=args.output,
        fps=sim.fps,
        frame_width=sim.frame_width,
        frame_height=sim.frame_height,
    )
    return GoldenTimeModules(sim, siren, evidence, distance, exporter)


def _on_plate_recognized(data: dict[str, Any]) -> None:
    """번호판 인식 이벤트 — 디버그 로그(verbose 모드)."""
    logger.info(
        "  [번호판] %s (신뢰도: %.0f%%) @ %.1fs",
        data["plate"],
        data["confidence"] * 100,
        data["timestamp"],
    )


def _on_evidence_exported(data: dict[str, Any]) -> None:
    """증거 패키지 저장 완료 알림 — 사용자 가시 출력 유지."""
    folder = data.get("folder", "")
    print()
    print("  " + "=" * SUB_WIDTH)
    print(f"  📦 증거 패키지 저장됨: {folder}")
    print("  " + "=" * SUB_WIDTH)


def register_callbacks(modules: GoldenTimeModules) -> None:
    """오버레이 렌더링 + 이벤트 구독을 등록."""
    sim = modules.sim
    sim.overlay_callbacks.append(modules.siren.draw_siren_overlay)
    sim.overlay_callbacks.append(modules.evidence.draw_evidence_overlay)
    sim.overlay_callbacks.append(modules.distance.draw_distance_overlay)
    sim.overlay_callbacks.append(modules.exporter.draw_export_overlay)

    sim.event_bus.subscribe("plate_recognized", _on_plate_recognized)
    sim.event_bus.subscribe("EVIDENCE_EXPORTED", _on_evidence_exported)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 엔트리포인트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _configure_logging(verbose: bool) -> None:
    """logger 포맷/레벨을 verbose 플래그에 맞춰 설정."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")


def main(argv: list[str] | None = None) -> int:
    """엔트리포인트 — 종료 코드 반환."""
    args = parse_args(argv)
    _configure_logging(args.verbose)

    print_banner()

    try:
        modules = build_modules(args)
    except FileNotFoundError as exc:
        print(f"\n  ❌ 에러: {exc}")
        print("     영상 파일 경로를 확인해주세요.")
        return EXIT_VIDEO_NOT_FOUND

    print_config(args, modules.sim)
    register_callbacks(modules)
    print_controls()

    start_time = datetime.now()
    print(f"  ▶ 시뮬레이션 시작: {start_time.strftime('%H:%M:%S')}")
    print()

    modules.sim.run()

    elapsed_sec = (datetime.now() - start_time).total_seconds()
    print_summary(args, modules, elapsed_sec)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
