# -*- coding: utf-8 -*-
"""
YOLO26 자동 평가 스크립트
pipeline_stage2.py를 subprocess로 실행 -> 로그 파싱 -> PASS/FAIL 자동 판정

사용법:
  python auto_eval.py
  python auto_eval.py --input movie/hiway.mp4
  python auto_eval.py --timeout 600
"""
import argparse
import os
import re
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path


# ====================================================
# PASS/FAIL 판정 기준
# ====================================================
PASS_CRITERIA = {
    "alive_ratio": 75.0,        # alive 비율 75% 이상 (현실적 목표)
    "fast_ocr_avg_ms": 400.0,   # Fast OCR 평균 400ms 이내
    "ghost_frames_max": 15,     # 잔상(dead OCR) 15건 이하
    "min_unique_plates": 10,    # 고유 번호판 10개 이상
}

# 경로 설정
PYTHON_EXE = r"C:\Users\jomg2\AppData\Local\Programs\Python\Python310\python.exe"
WORK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = WORK_DIR / "test_results"
EVAL_REPORT = RESULTS_DIR / "eval_report.txt"
EVAL_HISTORY = RESULTS_DIR / "eval_history.txt"
FULL_LOG = RESULTS_DIR / "full_log.txt"
DEFAULT_TIMEOUT = 600  # 10분


# ====================================================
# 1. 파이프라인 실행
# ====================================================
def run_pipeline(input_source, timeout):
    """pipeline_stage2.py를 subprocess로 실행, stdout+stderr 캡처"""
    cmd = [PYTHON_EXE, "-u", "pipeline_stage2.py", "--input", input_source, "--headless"]
    # -u: unbuffered stdout (실시간 로그 수집)
    print(f"[auto_eval] pipeline 실행: {' '.join(cmd)}")
    print(f"[auto_eval] 작업 디렉토리: {WORK_DIR}")
    print(f"[auto_eval] timeout: {timeout}s")
    print()

    # PYTHONIOENCODING=utf-8 환경변수 전달 (자식 프로세스 인코딩 강제)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # ★ 파일 리다이렉트 방식: 자식 프로세스(CMD5/CMD6)의 stdout/stderr를
    # 파이프 대신 파일로 보냄 → 파이프 블로킹 해결
    RESULTS_DIR.mkdir(exist_ok=True)
    _log_out = RESULTS_DIR / "_pipeline_stdout.tmp"
    _log_err = RESULTS_DIR / "_pipeline_stderr.tmp"

    t0 = time.time()
    output = ""
    try:
        with open(_log_out, "w", encoding="utf-8", errors="replace") as f_out, \
             open(_log_err, "w", encoding="utf-8", errors="replace") as f_err:
            proc = subprocess.Popen(
                cmd,
                cwd=str(WORK_DIR),
                stdout=f_out,
                stderr=f_err,
                env=env,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # ★ Windows: 프로세스 트리 전체 종료 (CMD5/CMD6 고아 프로세스 방지)
                if sys.platform == "win32":
                    os.system(f"taskkill /T /F /PID {proc.pid} >nul 2>&1")
                else:
                    proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass  # kill 후에도 안 죽으면 무시 (OS가 정리)
                print(f"[auto_eval] timeout ({timeout}s) -- 부분 결과로 평가 진행")
        elapsed = time.time() - t0

        # 파일에서 로그 읽기
        _stdout_text = _log_out.read_text(encoding="utf-8", errors="replace") if _log_out.exists() else ""
        _stderr_text = _log_err.read_text(encoding="utf-8", errors="replace") if _log_err.exists() else ""
        output = _stdout_text + "\n" + _stderr_text
        print(f"[auto_eval] pipeline 종료 (exit={proc.returncode}, {elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        # 예외 발생해도 파일에서 부분 로그 수집 시도
        try:
            _stdout_text = _log_out.read_text(encoding="utf-8", errors="replace") if _log_out.exists() else ""
            _stderr_text = _log_err.read_text(encoding="utf-8", errors="replace") if _log_err.exists() else ""
            output = _stdout_text + "\n" + _stderr_text
        except Exception:
            pass
        print(f"[auto_eval] 실행 오류: {e}")
    finally:
        # 임시 파일 정리
        for _tmp in [_log_out, _log_err]:
            try:
                _tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # 전체 로그 파일 저장
    if output:
        try:
            FULL_LOG.write_text(output, encoding="utf-8")
            print(f"[auto_eval] 전체 로그 저장: {FULL_LOG}")
        except Exception:
            pass

    return output, elapsed


# ====================================================
# 2. 로그 파싱
# ====================================================
def parse_logs(output):
    """파이프라인 stdout/stderr 로그를 파싱하여 평가 메트릭 추출"""

    # 정규식 패턴
    # [CMD5] frame=133 fast_ocr='45소8019' conf=1.00 time=251.2ms
    re_fast_cmd5 = re.compile(
        r"\[CMD5\] frame=(\d+) fast_ocr='([^']+)' conf=([\d.]+) time=([\d.]+)ms"
    )
    # [CMD12] Fast OCR 즉시표시: '45소8019' conf=1.00 tid=5
    re_fast_display = re.compile(
        r"\[CMD12\] Fast OCR .*: '([^']+)' conf=([\d.]+) tid=(\d+)"
    )
    # [CMD12] OCR(dead): '80부5915' conf=0.94 tid=93
    re_ocr_dead = re.compile(
        r"\[CMD12\] OCR\(dead\): '([^']+)' conf=([\d.]+) tid=(\d+)"
    )
    # [CMD12] Heavy->Fast교체: 'A' -> 'B' conf=0.92 tid=5
    re_heavy_replace = re.compile(
        r"\[CMD12\] Heavy.*: '([^']+)' .* '([^']+)'"
    )
    # [CMD12] OCR: '14나5915' conf=0.80 tid=89  (alive Heavy OCR)
    re_ocr_alive = re.compile(
        r"\[CMD12\] OCR: '([^']+)' conf=([\d.]+) tid=(\d+)"
    )
    # [CMD6] #1544/0 -> 14나5915 (conf=0.80, 389ms)
    re_cmd6 = re.compile(
        r"\[CMD6\] #(\d+)/\d+ .* (\S+) \(conf=([\d.]+), (\d+)ms\)"
    )
    # [CMD5] frame=1800 raw=0 nms=0 final=0 62ms
    re_yolo_periodic = re.compile(
        r"\[CMD5\] frame=(\d+) raw=\d+ nms=\d+ final=(\d+)"
    )
    # [CMD12] 영상 끝
    re_video_end = re.compile(r"\[CMD12\].*끝")

    # 수집 변수
    fast_ocr_times = []
    fast_display_tids = set()
    dead_tids = set()
    heavy_alive_tids = set()
    heavy_replace_count = 0
    unique_plates = set()
    heavy_ocr_ms_list = []
    max_frame = 0
    yolo_det_count = 0
    video_ended = False

    for line in output.splitlines():
        # Fast OCR (CMD5에서 인식 성공)
        m = re_fast_cmd5.search(line)
        if m:
            frame = int(m.group(1))
            text = m.group(2)
            t_ms = float(m.group(4))
            fast_ocr_times.append(t_ms)
            unique_plates.add(text)
            max_frame = max(max_frame, frame)
            continue

        # Fast OCR 즉시표시 (CMD12 display에 반영됨 = alive)
        m = re_fast_display.search(line)
        if m:
            text = m.group(1)
            tid = m.group(3)
            fast_display_tids.add(tid)
            unique_plates.add(text)
            continue

        # OCR(dead)
        m = re_ocr_dead.search(line)
        if m:
            text = m.group(1)
            tid = m.group(3)
            dead_tids.add(tid)
            unique_plates.add(text)
            continue

        # Heavy->Fast 교체
        m = re_heavy_replace.search(line)
        if m:
            heavy_replace_count += 1
            unique_plates.add(m.group(2))
            continue

        # Heavy OCR alive
        m = re_ocr_alive.search(line)
        if m:
            text = m.group(1)
            tid = m.group(3)
            heavy_alive_tids.add(tid)
            unique_plates.add(text)
            continue

        # CMD6 Heavy OCR 결과 (처리 시간 수집)
        m = re_cmd6.search(line)
        if m:
            text = m.group(2)
            ocr_ms = int(m.group(4))
            heavy_ocr_ms_list.append(ocr_ms)
            unique_plates.add(text)
            continue

        # YOLO 주기적 로그 (탐지 프레임 카운트)
        m = re_yolo_periodic.search(line)
        if m:
            frame = int(m.group(1))
            final = int(m.group(2))
            max_frame = max(max_frame, frame)
            if final > 0:
                yolo_det_count += 1
            continue

        # 영상 끝
        if re_video_end.search(line):
            video_ended = True

    # dead_tids에서 alive에도 있는 것 제거 (alive 판정 우선)
    pure_dead_tids = dead_tids - fast_display_tids - heavy_alive_tids

    return {
        "total_frames": max_frame,
        "video_ended": video_ended,
        "yolo_det_periodic": yolo_det_count,
        "fast_ocr_count": len(fast_ocr_times),
        "fast_ocr_times": fast_ocr_times,
        "fast_display_count": len(fast_display_tids),
        "heavy_ocr_alive": len(heavy_alive_tids),
        "heavy_ocr_dead": len(pure_dead_tids),
        "heavy_replace_count": heavy_replace_count,
        "heavy_ocr_ms_list": heavy_ocr_ms_list,
        "unique_plates": unique_plates,
    }


# ====================================================
# 3. 평가 결과 계산
# ====================================================
def compute_results(parsed):
    """파싱된 메트릭에서 평가 지표 계산"""
    fast_count = parsed["fast_display_count"]
    dead_count = parsed["heavy_ocr_dead"]
    total_events = fast_count + dead_count

    # alive 비율: Fast OCR 즉시표시 / (Fast + dead)
    alive_ratio = (fast_count / total_events * 100) if total_events > 0 else 0.0

    # Fast OCR 평균 속도
    times = parsed["fast_ocr_times"]
    fast_avg_ms = sum(times) / len(times) if times else 0.0

    # Heavy OCR 평균 속도
    heavy_times = parsed["heavy_ocr_ms_list"]
    heavy_avg_ms = sum(heavy_times) / len(heavy_times) if heavy_times else 0.0

    return {
        "total_frames": parsed["total_frames"],
        "fast_ocr_count": parsed["fast_ocr_count"],
        "fast_display_count": fast_count,
        "fast_ocr_avg_ms": round(fast_avg_ms, 1),
        "heavy_ocr_alive": parsed["heavy_ocr_alive"],
        "heavy_ocr_dead": dead_count,
        "heavy_ocr_avg_ms": round(heavy_avg_ms, 0),
        "heavy_replace_count": parsed["heavy_replace_count"],
        "alive_ratio": round(alive_ratio, 1),
        "ghost_frames": dead_count,
        "unique_plates": len(parsed["unique_plates"]),
        "unique_plates_list": sorted(parsed["unique_plates"]),
    }


# ====================================================
# 4. PASS/FAIL 판정
# ====================================================
def judge(results):
    """PASS_CRITERIA 기준으로 각 항목 판정"""
    c = PASS_CRITERIA
    items = []

    items.append({
        "name": "alive 비율",
        "value": f"{results['alive_ratio']}%",
        "criterion": f"{c['alive_ratio']}%+",
        "passed": results["alive_ratio"] >= c["alive_ratio"],
    })

    items.append({
        "name": "Fast OCR 속도",
        "value": f"{results['fast_ocr_avg_ms']:.0f}ms",
        "criterion": f"{c['fast_ocr_avg_ms']:.0f}ms 이내",
        "passed": results["fast_ocr_avg_ms"] <= c["fast_ocr_avg_ms"],
    })

    items.append({
        "name": "잔상(dead OCR)",
        "value": f"{results['ghost_frames']}건",
        "criterion": f"{c['ghost_frames_max']}건 이하",
        "passed": results["ghost_frames"] <= c["ghost_frames_max"],
    })

    items.append({
        "name": "고유 번호판",
        "value": f"{results['unique_plates']}개",
        "criterion": f"{c['min_unique_plates']}개+",
        "passed": results["unique_plates"] >= c["min_unique_plates"],
    })

    return items


# ====================================================
# 5. 출력 + 저장
# ====================================================
def print_report(results, judgements, elapsed, input_source):
    """터미널 출력 (cp949 안전 -- 특수문자 미사용)"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ANSI 색상 (Windows 10+ 지원)
    G = "\033[92m"   # 초록
    R = "\033[91m"   # 빨강
    B = "\033[1m"    # 굵게
    E = "\033[0m"    # 리셋

    # Windows 터미널 ANSI 활성화
    if sys.platform == "win32":
        os.system("")

    print()
    print(f"{B}===== YOLO26 자동 평가 ====={E}")
    print(f"시각: {now}")
    print(f"영상: {input_source}")
    print(f"실행 시간: {elapsed:.1f}초")
    print(f"총 프레임: {results['total_frames']}")
    print(f"Fast OCR: {results['fast_display_count']}건 (평균 {results['fast_ocr_avg_ms']:.0f}ms)")
    print(f"Heavy OCR: alive {results['heavy_ocr_alive']}건 "
          f"| dead {results['heavy_ocr_dead']}건 "
          f"(평균 {results['heavy_ocr_avg_ms']:.0f}ms)")
    if results["heavy_replace_count"] > 0:
        print(f"Heavy->Fast 교체: {results['heavy_replace_count']}건")
    print()

    all_passed = True
    fail_count = 0
    for j in judgements:
        if j["passed"]:
            tag = f"{G}[PASS]{E}"
            suffix = ""
        else:
            tag = f"{R}[FAIL]{E}"
            suffix = f" {R}<- 미달{E}"
            all_passed = False
            fail_count += 1
        print(f"  {tag} {j['name']}: {B}{j['value']}{E} (기준: {j['criterion']}{suffix})")

    print()
    if all_passed:
        print(f"  {G}{B}[ALL PASS]{E}")
    else:
        print(f"  {R}{B}[FAIL] {fail_count}건 미달{E}")
    print(f"{B}============================{E}")

    # 인식된 번호판 목록
    if results["unique_plates_list"]:
        print(f"\n인식된 번호판 ({results['unique_plates']}개):")
        for p in results["unique_plates_list"]:
            print(f"  {p}")
    print()

    return all_passed


def save_report(results, judgements, elapsed, input_source, all_passed):
    """test_results/eval_report.txt 저장"""
    RESULTS_DIR.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "========== 파이프라인 성적표 ==========",
        f"시각: {now}",
        f"영상: {input_source}",
        f"실행 시간: {elapsed:.1f}초",
        f"총 프레임: {results['total_frames']}",
        f"Fast OCR 성공: {results['fast_display_count']}건 (평균 {results['fast_ocr_avg_ms']:.0f}ms)",
        f"Heavy OCR alive: {results['heavy_ocr_alive']}건 (평균 {results['heavy_ocr_avg_ms']:.0f}ms)",
        f"Heavy OCR dead: {results['heavy_ocr_dead']}건",
        f"alive 비율: {results['alive_ratio']}% ({results['fast_display_count']}/{results['fast_display_count'] + results['heavy_ocr_dead']})",
        f"dead 비율: {round(100 - results['alive_ratio'], 1)}% ({results['heavy_ocr_dead']}/{results['fast_display_count'] + results['heavy_ocr_dead']})",
        f"잔상(dead OCR): {results['ghost_frames']}건",
        f"인식된 고유 번호판: {results['unique_plates']}개",
        "",
    ]

    for j in judgements:
        tag = "[PASS]" if j["passed"] else "[FAIL]"
        lines.append(f"{tag} {j['name']}: {j['value']} (기준: {j['criterion']})")

    lines.append("")
    if all_passed:
        lines.append("[ALL PASS]")
    else:
        fail_count = sum(1 for j in judgements if not j["passed"])
        lines.append(f"[FAIL] {fail_count}건 미달")
    lines.append("==========================================")

    # 번호판 목록
    if results["unique_plates_list"]:
        lines.append("")
        lines.append(f"인식된 번호판 ({results['unique_plates']}개):")
        for p in results["unique_plates_list"]:
            lines.append(f"  {p}")

    EVAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[auto_eval] 리포트 저장: {EVAL_REPORT}")


def append_history(results, all_passed):
    """test_results/eval_history.txt 누적 기록 (시간순)"""
    RESULTS_DIR.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict = "PASS" if all_passed else "FAIL"

    line = (
        f"{now} | {verdict} | "
        f"alive={results['alive_ratio']}% "
        f"fast={results['fast_ocr_avg_ms']:.0f}ms "
        f"ghost={results['ghost_frames']} "
        f"plates={results['unique_plates']}\n"
    )

    with open(EVAL_HISTORY, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[auto_eval] 이력 추가: {EVAL_HISTORY}")


# ====================================================
# 메인
# ====================================================
def main():
    # stdout utf-8 강제 (Windows cp949 충돌 방지)
    if sys.stdout.encoding != "utf-8":
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", buffering=1)
    if sys.stderr.encoding != "utf-8":
        sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", errors="replace", buffering=1)

    parser = argparse.ArgumentParser(description="YOLO26 자동 평가")
    parser.add_argument("--input", default="movie/hiway.mp4",
                        help="입력 영상 경로 (기본: movie/hiway.mp4)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"실행 타임아웃 초 (기본: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()

    # 영상 파일 존재 확인
    video_path = WORK_DIR / args.input
    if not video_path.exists() and not args.input.isdigit():
        print(f"[auto_eval] 영상 파일 없음: {video_path}")
        sys.exit(1)

    # 1. 파이프라인 실행
    output, elapsed = run_pipeline(args.input, args.timeout)
    if not output.strip():
        print("[auto_eval] 파이프라인 출력 없음 -- 중단")
        sys.exit(1)

    # 2. 로그 파싱
    parsed = parse_logs(output)

    if not parsed["video_ended"]:
        print("[auto_eval] 영상이 끝까지 재생되지 않음 (타임아웃 또는 오류)")
        # 타임아웃이어도 부분 결과로 평가 진행

    # 3. 결과 계산
    results = compute_results(parsed)

    # 4. PASS/FAIL 판정
    judgements = judge(results)

    # 5. 출력 + 저장
    all_passed = print_report(results, judgements, elapsed, args.input)
    save_report(results, judgements, elapsed, args.input, all_passed)
    append_history(results, all_passed)

    # exit code: 0=PASS, 1=FAIL
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
