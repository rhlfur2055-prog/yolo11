#!/usr/bin/env python3
"""OCR 정확도 회귀 테스트 (PlateEnginePro 12장 베이스라인).

22/ 폴더의 한국 번호판 12장에 대해 ``PlateEnginePro.process_frame`` 을 실행하고
파일명 기반 Ground Truth 와 결과를 비교한다.

산출물
------
- stdout: 표 + 요약 + (옵션) 실패 케이스 상세
- JSON 리포트: ``test_results/{timestamp}.json``
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# 상수
BASE_DIR: Path = Path(__file__).resolve().parent
IMG_DIR: Path = BASE_DIR / "22"
DEFAULT_OUTPUT_DIR: Path = BASE_DIR / "test_results"
ACCURACY_PASS_THRESHOLD: float = 0.90
TABLE_WIDTH: int = 78

_NORMALIZE_STRIP_RE = re.compile(r"[\s\-\.\(\)（）]")
_FALLBACK_TAG = "(F)"
_HANGUL_RE = re.compile(r"[가-힣]")
_PLATE_REGIONAL_RE = re.compile(r"^[가-힣]{2,3}\d{2}[가-힣]\d{4}$")
_GT_TOKEN_RE = re.compile(r"[가-힣0-9]+")

# 12장 회귀 케이스 (파일명만 명시 — GT 는 함수로 추출).
TEST_FILENAMES: list[str] = [
    "경기76바7789.png",
    "서울70바9203.png",
    "트럭 경기91바6286.png",
    "01나8060.png",
    "02누2754.png",
    "14나3234.png",
    "36다7117.png",
    "48보7062.png",
    "55저9392.png",
    "58두9599.png",
    "70버6393.png",
    "80부5915.png",
]

logger = logging.getLogger("test_ocr_accuracy")

sys.path.insert(0, str(BASE_DIR))


# 데이터 모델
@dataclass
class PerImageResult:
    file: str
    gt: str
    ocr: str
    conf: float
    time_ms: int
    passed: bool


@dataclass
class FailureCase:
    file: str
    gt: str
    ocr: str
    hypothesis_stage: str


@dataclass
class TestReport:
    timestamp: str
    accuracy: float
    total: int
    passed: int
    failed: int
    avg_time_ms: int
    per_image: list[PerImageResult] = field(default_factory=list)
    failure_cases: list[FailureCase] = field(default_factory=list)


# 헬퍼
def extract_ground_truth_from_filename(filename: str) -> str:
    """파일명에서 번호판 GT 추출.

    예: "트럭 경기91바6286.png" → "경기91바6286"
    """
    stem = Path(filename).stem
    tokens = stem.split()
    candidate = tokens[-1] if tokens else stem
    match = _GT_TOKEN_RE.search(candidate)
    return match.group(0) if match else candidate


def normalize_plate(text: str) -> str:
    """비교용 정규화: 공백/특수문자 제거, 대문자, COCO 폴백 마크 제거."""
    if not text:
        return ""
    return _NORMALIZE_STRIP_RE.sub("", text).replace(_FALLBACK_TAG, "").strip().upper()


def plates_match(ocr_text: str, gt_text: str) -> bool:
    return normalize_plate(ocr_text) == normalize_plate(gt_text)


def classify_failure(ocr_text: str, gt_text: str) -> str:
    """7단계 파이프라인 중 어느 단계가 실패의 원인인지 1차 추정."""
    if not ocr_text:
        return "단계 4 PaddleOCR 한글 누락 + 단계 4 CRNN 폴백 미트리거"

    ocr_norm = normalize_plate(ocr_text)
    gt_norm = normalize_plate(gt_text)

    if _PLATE_REGIONAL_RE.match(gt_norm) and not _PLATE_REGIONAL_RE.match(ocr_norm):
        return "단계 4 지역명 오인식"

    if len(ocr_norm) != len(gt_norm):
        return "단계 4 OCR 결과 패턴 불일치"

    diff_positions = [i for i, (a, b) in enumerate(zip(ocr_norm, gt_norm)) if a != b]
    if len(diff_positions) == 1:
        i = diff_positions[0]
        if _HANGUL_RE.match(ocr_norm[i]) or _HANGUL_RE.match(gt_norm[i]):
            return "단계 4 한글 오인식 (CRNN 교차검증 한계)"
        return "단계 4 숫자 오인식"

    return "단계 5 투표/검증 미수렴"


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


def load_image(filepath: Path) -> np.ndarray | None:
    """한글 파일명 호환 cv2.imread."""
    img = cv2.imread(str(filepath))
    if img is not None:
        return img
    data = np.fromfile(str(filepath), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@contextlib.contextmanager
def maybe_silence(silence: bool):
    """verbose=False 일 때 엔진의 시끄러운 stdout 만 음소거."""
    if not silence:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def run_single(engine, filepath: Path) -> tuple[str, float, int]:
    """이미지 1장 처리 → (ocr_text, conf, elapsed_ms)."""
    img = load_image(filepath)
    if img is None:
        return "", 0.0, 0

    engine.reset_state()
    started_at = time.time()
    detections = engine.process_frame(img)
    elapsed_ms = int((time.time() - started_at) * 1000)

    if not detections:
        return "", 0.0, elapsed_ms

    best = max(detections, key=lambda d: d.get("confidence", 0))
    return best.get("plate", ""), float(best.get("confidence", 0)), elapsed_ms


def build_test_cases() -> list[tuple[str, str]]:
    return [(fname, extract_ground_truth_from_filename(fname)) for fname in TEST_FILENAMES]


# 출력
def render_table(results: Iterable[PerImageResult]) -> None:
    print("-" * TABLE_WIDTH)
    print(f"{'#':>2}  {'파일명':<28} {'정답':<14} {'OCR결과':<22} {'시간':>7}  판정")
    print("-" * TABLE_WIDTH)
    for idx, r in enumerate(results, 1):
        fname = r.file if len(r.file) <= 26 else r.file[:23] + "..."
        ocr_disp = f"{r.ocr} ({r.conf:.0%})" if r.ocr else "(미인식)"
        mark = "✅ 정확" if r.passed else "❌ 오인식"
        print(f"{idx:>2}  {fname:<28} {r.gt:<14} {ocr_disp:<22} {r.time_ms:>5}ms  {mark}")
    print("-" * TABLE_WIDTH)


def render_failures(failures: list[FailureCase]) -> None:
    if not failures:
        return
    print()
    print("─── 실패 케이스 상세 ───")
    for f in failures:
        ocr_disp = f.ocr or "(미인식)"
        print(f"❌ {f.file}")
        print(f"   GT       : {f.gt}")
        print(f"   OCR      : {ocr_disp}")
        print(f"   추정 단계: {f.hypothesis_stage}")


def save_report(report: TestReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.timestamp}.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(asdict(report), fp, ensure_ascii=False, indent=2)
    return path


# 파이프라인
def load_engine(verbose: bool):
    """PlateEnginePro 인스턴스 + consecutive_required=1 보정."""
    print("[init] PlateEnginePro 로딩...")
    started_at = time.time()
    from plate_engine_pro import PlateEnginePro  # 지연 import (logging 설정 이후)

    with maybe_silence(not verbose):
        engine = PlateEnginePro()
    engine.consecutive_required = 1
    print(f"[init] 완료 ({time.time() - started_at:.1f}s)")
    return engine


def evaluate(engine, cases: list[tuple[str, str]], verbose: bool) -> TestReport:
    report = TestReport(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        accuracy=0.0,
        total=len(cases),
        passed=0,
        failed=0,
        avg_time_ms=0,
    )

    for filename, gt in cases:
        filepath = IMG_DIR / filename
        if not filepath.exists():
            report.per_image.append(
                PerImageResult(file=filename, gt=gt, ocr="", conf=0.0, time_ms=0, passed=False)
            )
            report.failure_cases.append(
                FailureCase(file=filename, gt=gt, ocr="", hypothesis_stage="입력 파일 누락")
            )
            report.failed += 1
            continue

        with maybe_silence(not verbose):
            ocr_text, conf, time_ms = run_single(engine, filepath)

        passed = plates_match(ocr_text, gt)
        report.per_image.append(
            PerImageResult(file=filename, gt=gt, ocr=ocr_text, conf=conf,
                           time_ms=time_ms, passed=passed)
        )
        if passed:
            report.passed += 1
        else:
            report.failed += 1
            report.failure_cases.append(
                FailureCase(file=filename, gt=gt, ocr=ocr_text,
                            hypothesis_stage=classify_failure(ocr_text, gt))
            )

    times = [r.time_ms for r in report.per_image if r.time_ms > 0]
    report.avg_time_ms = int(sum(times) / len(times)) if times else 0
    report.accuracy = report.passed / report.total if report.total else 0.0
    return report


# CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR 정확도 회귀 테스트 (PlateEnginePro 12장 베이스라인)"
    )
    parser.add_argument("--verbose", action="store_true",
                        help="엔진 디버그 로그를 그대로 출력")
    parser.add_argument("--save-failures", action="store_true",
                        help="실패 케이스 상세를 stdout 에 추가 출력")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"JSON 리포트 저장 디렉터리 (기본: {DEFAULT_OUTPUT_DIR.name}/)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    print("=" * TABLE_WIDTH)
    print("  번호판 OCR 정확도 회귀 테스트")
    print("=" * TABLE_WIDTH)

    engine = load_engine(args.verbose)
    cases = build_test_cases()
    report = evaluate(engine, cases, args.verbose)

    render_table(report.per_image)
    print()
    print(f"  총 {report.total}장  |  ✅ {report.passed}  |  ❌ {report.failed}  "
          f"|  평균 {report.avg_time_ms}ms")
    print(f"  정확도: {report.passed}/{report.total} = {report.accuracy * 100:.1f}%")

    if args.save_failures:
        render_failures(report.failure_cases)

    json_path = save_report(report, args.output_dir)
    try:
        rel_path = json_path.relative_to(BASE_DIR)
    except ValueError:
        rel_path = json_path
    print(f"\n📄 JSON 리포트: {rel_path}")
    print("=" * TABLE_WIDTH)

    return 0 if report.accuracy >= ACCURACY_PASS_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
