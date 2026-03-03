# -*- coding: utf-8 -*-
"""
hiway.mp4 번호판 인식 검증 테스트
- pipeline_stage2.py의 실제 인식 결과를 기대값과 비교
- 사용법: python test_recognition.py
"""
import subprocess
import sys
import os
import re

# ── 기대 번호판 목록 (hiway.mp4에서 육안 확인된 것들) ──
EXPECTED_PLATES = {
    "경기91바6286",
    "서울70바9203",
    "01나8060",
    "48보7062",
    "58두9599",
    "경기76바7789",
    "70버6393",
    "55저9392",
    "80부5915",
}

# ── 최소 인식률 기준 ──
MIN_RECALL = 0.70


def run_pipeline():
    """pipeline_stage2.py 실행 후 stdout에서 인식된 번호판 추출"""
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    cmd = [
        sys.executable, "-u", "pipeline_stage2.py",
        "--input", "movie/hiway.mp4"
    ]
    print(f"[TEST] 실행: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, cwd=r"C:\tool\yolo26-main"
    )
    output_lines = []
    try:
        for raw in proc.stdout:
            line = raw.decode('utf-8', errors='ignore')
            output_lines.append(line)
    except Exception:
        pass
    proc.wait(timeout=30)
    return ''.join(output_lines)


def extract_plates(output: str) -> dict:
    """로그에서 인식된 번호판과 confidence 추출"""
    pattern = r'([가-힣]*\d{1,3}[가-힣]\d{4}).*?conf=(\d+\.\d+)'
    found = {}
    for match in re.finditer(pattern, output):
        plate, conf = match.group(1), float(match.group(2))
        if plate not in found or conf > found[plate]:
            found[plate] = conf
    return found


def test():
    print("=" * 60)
    print("  hiway.mp4 번호판 인식 검증 테스트")
    print("=" * 60)

    # 1) 파이프라인 실행
    output = run_pipeline()

    # 2) 인식 결과 추출
    recognized = extract_plates(output)
    print(f"\n[결과] 인식된 번호판 {len(recognized)}개:")
    for plate, conf in sorted(recognized.items()):
        mark = "O" if plate in EXPECTED_PLATES else "? (기대목록에 없음)"
        print(f"  {plate}  conf={conf:.2f}  {mark}")

    # 3) 기대 번호판 매칭
    matched = EXPECTED_PLATES & set(recognized.keys())
    missed = EXPECTED_PLATES - set(recognized.keys())
    recall = len(matched) / len(EXPECTED_PLATES) if EXPECTED_PLATES else 0

    print(f"\n[매칭] {len(matched)}/{len(EXPECTED_PLATES)} ({recall:.0%})")
    if missed:
        print(f"[미인식] {missed}")

    # 4) PASS/FAIL 판정
    print("\n" + "=" * 60)
    if recall >= MIN_RECALL:
        print(f"  PASS  (인식률 {recall:.0%} >= 기준 {MIN_RECALL:.0%})")
    else:
        print(f"  FAIL  (인식률 {recall:.0%} < 기준 {MIN_RECALL:.0%})")
    print("=" * 60)

    return recall >= MIN_RECALL


if __name__ == "__main__":
    success = test()
    sys.exit(0 if success else 1)
