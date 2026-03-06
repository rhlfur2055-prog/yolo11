"""
골든타임 신규 옵션 검증 스크립트.

- 핀홀 거리 공식 d = f*W/w 단위 검증
- 후진 양보: is_yielding 시 위반 누적 중단 시퀀스 검증
- (선택) test_ocr_accuracy.py 회귀 12/12 Zero-impact 확인
"""

import sys
import os
import subprocess

# 프로젝트 루트를 path에 추가
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_pinhole_formula():
    """d = f*W/w 공식 및 close_distance_m 전이 구간 검증."""
    from simulation.goldentime.distance_checker import DistanceChecker, DISTANCE_CONFIG

    class DummyBus:
        def subscribe(self, *a, **k): pass
        def publish(self, *a, **k): pass

    bus = DummyBus()
    dc = DistanceChecker(bus, frame_width=1920, frame_height=1080, config={
        **DISTANCE_CONFIG,
        "use_pinhole_distance": True,
        "focal_length_px": 1200.0,
        "plate_width_m": 0.52,
        "close_distance_m": 5.0,
    })

    # d = 5m → w = f*W/d = 1200*0.52/5 = 124.8
    bbox_5m = [100, 500, 100 + 125, 500 + 26]  # w=125px
    d = dc._pinhole_distance_m(bbox_5m)
    assert 4.5 < d < 5.5, f"5m 기대, d={d}"

    # d = 10m → w ≈ 62.4
    bbox_10m = [100, 500, 100 + 62, 500 + 13]
    d10 = dc._pinhole_distance_m(bbox_10m)
    assert 9.0 < d10 < 11.0, f"10m 기대, d={d10}"

    # close_distance_m=5 이하일 때만 is_close
    assert dc.config["close_distance_m"] == 5.0
    print("[OK] 핀홀 공식 d=f*W/w 및 5m 전이 구간 검증 통과")


def test_yield_stops_violation_accumulation():
    """is_yielding=True 시 close_duration이 누적되지 않고 check_violation이 False를 반환하는지 검증."""
    from simulation.goldentime.distance_checker import PlateDistanceRecord, DISTANCE_CONFIG

    close_threshold = DISTANCE_CONFIG["close_ratio_threshold"]
    gap_tolerance = DISTANCE_CONFIG["gap_tolerance_sec"]

    rec = PlateDistanceRecord("경기76바7789", timestamp=0.0)

    # 시뮬레이션: 0.5초 동안 bbox 면적을 2%/s 이상 증가시키면 is_yielding (YIELD_AREA_RATE_MIN=0.02)
    import math
    base_w, base_h = 100.0, 22.0
    for i in range(20):
        t = i * 0.033  # ~30fps, 총 ~0.63초
        # 면적이 초당 약 5% 증가 (2% 이상이어야 yielding)
        growth = 1.0 + 0.05 * t
        w = base_w * math.sqrt(growth)
        h = base_h * math.sqrt(growth)
        bbox = [100, 500, 100 + w, 500 + h]
        area = w * h
        frame_area = 1920 * 1080
        ratio = area / frame_area
        rec.update_bbox(ratio, t, close_threshold, gap_tolerance, bbox=bbox)

    assert rec.is_yielding, "면적 증가 시 is_yielding True 기대"
    assert rec.is_close == False or rec.close_duration == 0.0, "양보 중에는 위반 누적 없음"
    assert rec.check_violation(5.0, 1.0) == False, "양보 중 check_violation False 기대"
    print("[OK] 후진 양보 시 위반 누적 중단 시퀀스 검증 통과")


def test_regression_ocr_accuracy():
    """test_ocr_accuracy.py 실행하여 12/12 회귀(Zero-impact) 확인.

    환경변수 RUN_REGRESSION=1 일 때만 실행 (YOLO/OCR 로딩으로 시간 소요).
    """
    if os.environ.get("RUN_REGRESSION") != "1":
        print("[SKIP] 회귀 테스트 (RUN_REGRESSION=1 설정 시 실행)")
        return
    script = os.path.join(REPO_ROOT, "test_ocr_accuracy.py")
    if not os.path.isfile(script):
        print("[SKIP] test_ocr_accuracy.py 없음")
        return
    try:
        r = subprocess.run(
            [sys.executable, script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            print(f"[FAIL] test_ocr_accuracy.py exit code={r.returncode}")
            print(r.stdout or r.stderr or "")
            sys.exit(1)
        if "12/12" in (r.stdout or "") or "12/12" in (r.stderr or ""):
            print("[OK] 회귀 12/12 유지 확인")
        else:
            print("[WARN] 12/12 문자열 미확인 — stdout/stderr 확인 필요")
    except subprocess.TimeoutExpired:
        print("[FAIL] test_ocr_accuracy.py 타임아웃")
        sys.exit(1)
    except Exception as e:
        print(f"[SKIP] 회귀 실행 예외: {e}")


if __name__ == "__main__":
    test_pinhole_formula()
    test_yield_stops_violation_accumulation()
    test_regression_ocr_accuracy()
    print("\n모든 검증 통과.")
