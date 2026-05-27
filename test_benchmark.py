#!/usr/bin/env python3
"""
성능 벤치마크 테스트 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
측정 항목:
  1. FPS (목표: >= 15fps) — 연속 프레임 처리 속도
  2. OCR 1회 처리 시간 (목표: <= 1500ms) — 단일 이미지 process_frame 소요
  3. 메모리 사용량 (목표: <= 4GB) — 엔진 로드 후 / 영상 처리 중 RSS

사용법:
  py -3.10 test_benchmark.py
"""

import os
import sys
import time
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 메모리 측정 유틸
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import cv2

from plate_engine_pro import PlateEnginePro

# 설정
IMG_DIR = os.path.join(BASE_DIR, "22")
VIDEO_PATH = os.path.join(BASE_DIR, "movie", "hiway.mp4")

# 테스트 이미지 (정확도 테스트와 동일)
TEST_IMAGES = [
    "경기76바7789.png",
    "서울바9203.png",
    "트럭 경기91바6286.png",
    "01나8060.png",
    "02누2754.png",
    "14니3234.png",
    "36다7117.png",
    "48보7062.png",
    "55저9392.png",
    "58두9599.png",
    "70버6393.png",
    "80부5915.png",
]

# 목표 기준
TARGET_FPS = 15.0
TARGET_OCR_MS = 1500.0
TARGET_MEM_GB = 4.0

# FPS 측정 프레임 수
FPS_FRAME_COUNT = 100


def get_memory_mb():
    """현재 프로세스 RSS 메모리 (MB) 반환"""
    if HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    else:
        # psutil 없으면 -1 반환 (측정 불가)
        return -1.0


def load_test_images():
    """22/ 폴더에서 테스트 이미지 로드"""
    images = []
    for fname in TEST_IMAGES:
        fpath = os.path.join(IMG_DIR, fname)
        img = cv2.imread(fpath)
        if img is not None:
            images.append((fname, img))
        else:
            print(f"  [경고] 이미지 로드 실패: {fname}")
    return images


# 벤치마크 1: OCR 1회 처리 시간
def bench_ocr_latency(engine, images):
    """
    각 테스트 이미지에 대해 process_frame 1회 호출 시간 측정.
    - 매 이미지마다 reset_state() 호출 (독립 측정)
    - 결과: 개별 시간 + 평균/최대/최소
    """
    print(f"\n{'='*60}")
    print(f"  벤치마크 1: OCR 1회 처리 시간 (목표: <= {TARGET_OCR_MS:.0f}ms)")
    print(f"{'='*60}")

    engine.consecutive_required = 1
    latencies = []

    print(f"\n  {'#':>2}  {'이미지':<28} {'시간(ms)':>10}  {'판정'}")
    print(f"  {'-'*55}")

    for idx, (fname, img) in enumerate(images, 1):
        engine.reset_state()
        gc.collect()

        t0 = time.perf_counter()
        _ = engine.process_frame(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        latencies.append(elapsed_ms)
        status = "PASS" if elapsed_ms <= TARGET_OCR_MS else "FAIL"
        display_name = fname if len(fname) <= 26 else fname[:23] + "..."
        print(f"  {idx:>2}  {display_name:<28} {elapsed_ms:>8.1f}ms  {status}")

    avg_ms = sum(latencies) / len(latencies) if latencies else 0
    min_ms = min(latencies) if latencies else 0
    max_ms = max(latencies) if latencies else 0

    print(f"  {'-'*55}")
    print(f"  평균: {avg_ms:.1f}ms | 최소: {min_ms:.1f}ms | 최대: {max_ms:.1f}ms")

    passed = avg_ms <= TARGET_OCR_MS
    over_count = sum(1 for t in latencies if t > TARGET_OCR_MS)
    print(f"  기준 초과: {over_count}/{len(latencies)}장")
    print(f"\n  결과: {'PASS' if passed else 'FAIL'} — 평균 {avg_ms:.1f}ms (목표 <= {TARGET_OCR_MS:.0f}ms)")

    return passed, {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "over_count": over_count,
        "total": len(latencies),
        "latencies": latencies,
    }


# 벤치마크 2: FPS (연속 프레임 처리 속도)
def bench_fps(engine, images):
    """
    연속 프레임 처리 FPS 측정.
    - 영상 파일 있으면 실제 영상 사용, 없으면 테스트 이미지 순환
    - FPS_FRAME_COUNT 프레임 처리 후 총 시간으로 FPS 계산
    """
    print(f"\n{'='*60}")
    print(f"  벤치마크 2: FPS 측정 (목표: >= {TARGET_FPS:.0f}fps, {FPS_FRAME_COUNT}프레임)")
    print(f"{'='*60}")

    engine.reset_state()
    engine.consecutive_required = 3  # 실제 운영 설정 시뮬레이션

    frames = []
    source_name = ""

    # 영상 파일 우선 사용
    if os.path.exists(VIDEO_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        if cap.isOpened():
            source_name = os.path.basename(VIDEO_PATH)
            print(f"\n  소스: {source_name}")
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  영상 FPS: {video_fps:.1f}")

            # 프레임 미리 로드 (I/O 시간 제외)
            count = 0
            while count < FPS_FRAME_COUNT:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 영상 끝나면 처음부터
                    ret, frame = cap.read()
                    if not ret:
                        break
                frames.append(frame)
                count += 1
            cap.release()

    # 영상 없으면 테스트 이미지 순환
    if not frames:
        source_name = "테스트 이미지 순환"
        print(f"\n  소스: {source_name} (영상 없음)")
        for i in range(FPS_FRAME_COUNT):
            _, img = images[i % len(images)]
            frames.append(img)

    print(f"  총 프레임: {len(frames)}개")
    print(f"\n  처리 중...")

    # 워밍업 (첫 프레임은 캐시/JIT 효과로 느림)
    engine.reset_state()
    _ = engine.process_frame(frames[0])
    engine.reset_state()

    # 실제 측정
    frame_times = []
    t_total_start = time.perf_counter()

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        _ = engine.process_frame(frame)
        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)

        # 진행률 표시 (25% 단위)
        progress = (i + 1) / len(frames)
        if (i + 1) % (len(frames) // 4) == 0 or i == len(frames) - 1:
            current_fps = (i + 1) / (time.perf_counter() - t_total_start)
            print(f"    [{progress:>6.1%}] {i+1}/{len(frames)} 프레임 — 현재 FPS: {current_fps:.2f}")

    t_total = time.perf_counter() - t_total_start
    overall_fps = len(frames) / t_total

    # 프레임별 통계
    avg_frame_ms = (sum(frame_times) / len(frame_times)) * 1000
    min_frame_ms = min(frame_times) * 1000
    max_frame_ms = max(frame_times) * 1000

    # 상위 10% 제외한 평균 (outlier 제거)
    sorted_times = sorted(frame_times)
    p90_idx = int(len(sorted_times) * 0.9)
    p90_fps = p90_idx / sum(sorted_times[:p90_idx]) if p90_idx > 0 else 0

    print(f"\n  총 소요: {t_total:.2f}초")
    print(f"  전체 FPS: {overall_fps:.2f}")
    print(f"  P90 FPS (상위 10% 제외): {p90_fps:.2f}")
    print(f"  프레임 처리 시간 — 평균: {avg_frame_ms:.1f}ms | 최소: {min_frame_ms:.1f}ms | 최대: {max_frame_ms:.1f}ms")

    passed = overall_fps >= TARGET_FPS
    print(f"\n  결과: {'PASS' if passed else 'FAIL'} — {overall_fps:.2f} fps (목표 >= {TARGET_FPS:.0f}fps)")

    return passed, {
        "overall_fps": overall_fps,
        "p90_fps": p90_fps,
        "avg_frame_ms": avg_frame_ms,
        "min_frame_ms": min_frame_ms,
        "max_frame_ms": max_frame_ms,
        "total_seconds": t_total,
        "frame_count": len(frames),
    }


# 벤치마크 3: 메모리 사용량
def bench_memory(engine, images):
    """
    메모리 사용량 측정.
    - 엔진 로드 후 기본 메모리
    - 12장 이미지 처리 후 메모리
    - 피크 메모리 (연속 처리 중)
    """
    print(f"\n{'='*60}")
    print(f"  벤치마크 3: 메모리 사용량 (목표: <= {TARGET_MEM_GB:.0f}GB)")
    print(f"{'='*60}")

    if not HAS_PSUTIL:
        print("\n  [경고] psutil 미설치 — 메모리 측정 불가")
        print("  설치: pip install psutil")
        print(f"\n  결과: SKIP — psutil 필요")
        return None, {"status": "SKIP", "reason": "psutil 미설치"}

    gc.collect()
    mem_baseline = get_memory_mb()
    print(f"\n  엔진 로드 후 기본 메모리: {mem_baseline:.1f} MB ({mem_baseline/1024:.2f} GB)")

    # 12장 이미지 순차 처리하면서 메모리 추적
    engine.consecutive_required = 1
    engine.reset_state()

    mem_readings = [mem_baseline]

    for idx, (fname, img) in enumerate(images, 1):
        engine.reset_state()
        _ = engine.process_frame(img)
        mem_now = get_memory_mb()
        mem_readings.append(mem_now)

    mem_after_images = get_memory_mb()
    print(f"  12장 처리 후 메모리: {mem_after_images:.1f} MB ({mem_after_images/1024:.2f} GB)")

    # 연속 프레임 처리 (메모리 누수 확인)
    engine.reset_state()
    engine.consecutive_required = 3

    # 첫 번째 이미지로 50프레임 연속 처리
    _, test_img = images[0]
    for _ in range(50):
        _ = engine.process_frame(test_img)
    mem_after_continuous = get_memory_mb()
    print(f"  50프레임 연속 처리 후: {mem_after_continuous:.1f} MB ({mem_after_continuous/1024:.2f} GB)")

    peak_mb = max(mem_readings + [mem_after_images, mem_after_continuous])
    peak_gb = peak_mb / 1024
    mem_growth = mem_after_continuous - mem_baseline

    print(f"\n  피크 메모리: {peak_mb:.1f} MB ({peak_gb:.2f} GB)")
    print(f"  메모리 증가분: {mem_growth:+.1f} MB (기준 대비)")

    # 메모리 누수 경고 (50프레임 처리 후 100MB 이상 증가 시)
    if mem_after_continuous - mem_after_images > 100:
        print(f"  [경고] 연속 처리 중 메모리 증가 감지: +{mem_after_continuous - mem_after_images:.1f} MB")

    passed = peak_gb <= TARGET_MEM_GB
    print(f"\n  결과: {'PASS' if passed else 'FAIL'} — 피크 {peak_gb:.2f} GB (목표 <= {TARGET_MEM_GB:.0f}GB)")

    return passed, {
        "baseline_mb": mem_baseline,
        "after_images_mb": mem_after_images,
        "after_continuous_mb": mem_after_continuous,
        "peak_mb": peak_mb,
        "peak_gb": peak_gb,
        "growth_mb": mem_growth,
    }


# 메인
def main():
    print("=" * 60)
    print("  YOLO26 성능 벤치마크")
    print("=" * 60)

    # 환경 정보
    print(f"\n  Python: {sys.version.split()[0]}")
    print(f"  OpenCV: {cv2.__version__}")
    if HAS_PSUTIL:
        mem_total = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        print(f"  시스템 메모리: {mem_total:.1f} GB")
        print(f"  CPU 코어: {cpu_count}개")

    try:
        import torch
        gpu = torch.cuda.is_available()
        print(f"  CUDA: {'사용 가능 — ' + torch.cuda.get_device_name(0) if gpu else '사용 불가 (CPU 모드)'}")
    except ImportError:
        print(f"  PyTorch: 미설치")

    # 엔진 초기화
    print(f"\n[초기화] PlateEnginePro 로딩...")
    t0 = time.perf_counter()
    engine = PlateEnginePro()
    init_time = time.perf_counter() - t0
    print(f"[초기화] 완료 ({init_time:.1f}초)")

    # 테스트 이미지 로드
    print(f"\n[이미지] 테스트 이미지 로드 중...")
    images = load_test_images()
    print(f"[이미지] {len(images)}장 로드 완료")

    if not images:
        print("\n[오류] 테스트 이미지 없음 — 22/ 폴더를 확인하세요")
        return 1

    # 벤치마크 실행
    results = {}

    # 1. OCR 레이턴시
    passed_ocr, ocr_data = bench_ocr_latency(engine, images)
    results["ocr_latency"] = {"passed": passed_ocr, "data": ocr_data}

    # 2. FPS
    passed_fps, fps_data = bench_fps(engine, images)
    results["fps"] = {"passed": passed_fps, "data": fps_data}

    # 3. 메모리
    passed_mem, mem_data = bench_memory(engine, images)
    results["memory"] = {"passed": passed_mem, "data": mem_data}

    # 최종 결과 요약
    print(f"\n{'='*60}")
    print(f"  벤치마크 최종 결과")
    print(f"{'='*60}")

    print(f"\n  {'항목':<24} {'측정값':>12} {'목표':>12} {'판정':>6}")
    print(f"  {'-'*58}")

    # OCR 레이턴시
    ocr_val = f"{ocr_data['avg_ms']:.0f}ms"
    ocr_target = f"<= {TARGET_OCR_MS:.0f}ms"
    ocr_status = "PASS" if passed_ocr else "FAIL"
    print(f"  {'OCR 평균 처리시간':<24} {ocr_val:>12} {ocr_target:>12} {ocr_status:>6}")

    # FPS
    fps_val = f"{fps_data['overall_fps']:.2f}"
    fps_target = f">= {TARGET_FPS:.0f}"
    fps_status = "PASS" if passed_fps else "FAIL"
    print(f"  {'FPS (연속 프레임)':<24} {fps_val:>12} {fps_target:>12} {fps_status:>6}")

    # 메모리
    if passed_mem is not None:
        mem_val = f"{mem_data['peak_gb']:.2f}GB"
        mem_target = f"<= {TARGET_MEM_GB:.0f}GB"
        mem_status = "PASS" if passed_mem else "FAIL"
    else:
        mem_val = "N/A"
        mem_target = f"<= {TARGET_MEM_GB:.0f}GB"
        mem_status = "SKIP"
    print(f"  {'피크 메모리':<24} {mem_val:>12} {mem_target:>12} {mem_status:>6}")

    print(f"  {'-'*58}")

    # 엔진 초기화 시간 (참고)
    print(f"\n  참고: 엔진 초기화 시간: {init_time:.1f}초")

    # 최종 판정
    all_results = [passed_ocr, passed_fps]
    if passed_mem is not None:
        all_results.append(passed_mem)

    total_pass = sum(1 for r in all_results if r)
    total = len(all_results)

    all_passed = all(all_results)
    print(f"\n  통과: {total_pass}/{total}")
    print(f"  최종 판정: {'PASS' if all_passed else 'FAIL'}")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
