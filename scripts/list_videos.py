# -*- coding: utf-8 -*-
"""
C:/tools/yolo26 전체 재귀 탐색 → .mp4 .avi 목록 + 크기(MB)
- 100MB 이하만 다운로드 대상
- 제외: highway_kr.mp4, d09a4761.mp4, d819ca75.mp4
"""
import os
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

EXCLUDE = {"highway_kr.mp4", "d09a4761.mp4", "d819ca75.mp4"}
MAX_MB = 100
ROOT = Path(r"C:/tools/yolo26")

# AI Hub 3개 데이터셋 관련 키워드 (우선순위 판단용)
KEYWORDS_CCTV_OVERHEAD = ["cctv", "부감", "overhead", "topdown", "주차", "parking", "교차로", "intersection"]
KEYWORDS_PLATE = ["번호판", "plate", "차종", "색상", "vehicle"]
KEYWORDS_TRAFFIC = ["교통", "traffic"]

def main():
    videos = []
    for ext in ("*.mp4", "*.avi"):
        for f in ROOT.rglob(ext):
            if f.name in EXCLUDE:
                continue
            try:
                size_mb = f.stat().st_size / (1024 * 1024)
            except OSError:
                size_mb = 0
            videos.append({
                "path": str(f),
                "name": f.name,
                "size_mb": round(size_mb, 2),
                "under_100mb": size_mb <= MAX_MB,
            })

    # 100MB 이하만
    under = [v for v in videos if v["under_100mb"]]
    over = [v for v in videos if not v["under_100mb"]]

    def priority(v):
        name_lower = v["name"].lower()
        path_lower = v["path"].lower()
        score = 0
        for k in KEYWORDS_CCTV_OVERHEAD:
            if k in name_lower or k in path_lower:
                score += 3
                break
        for k in KEYWORDS_PLATE:
            if k in name_lower or k in path_lower:
                score += 2
                break
        for k in KEYWORDS_TRAFFIC:
            if k in name_lower or k in path_lower:
                score += 1
                break
        return (-score, v["size_mb"])

    under.sort(key=priority)

    print("=" * 60)
    print("[STEP 1] 영상 탐색 결과 (C:/tools/yolo26 재귀)")
    print("=" * 60)
    print(f"\n전체 .mp4/.avi: {len(videos)}개")
    print(f"100MB 이하 (다운로드 대상): {len(under)}개")
    print(f"100MB 초과 (다운로드 금지): {len(over)}개")
    print(f"제외 파일: {EXCLUDE}")

    print("\n--- 100MB 이하 영상 (우선순위: CCTV부감 > 번호판/차종 > 교통) ---")
    for v in under[:50]:
        print(f"  {v['size_mb']:.2f} MB  {v['path']}")
    if len(under) > 50:
        print(f"  ... 외 {len(under)-50}개")

    print("\n--- 100MB 초과 (다운로드 금지) ---")
    for v in over[:20]:
        print(f"  {v['size_mb']:.2f} MB  {v['path']}")
    if len(over) > 20:
        print(f"  ... 외 {len(over)-20}개")

    # 리테스트용 추천 (0.1MB 이상 + 가장 큰 100MB 이하)
    usable = [v for v in under if v["size_mb"] >= 0.1]
    usable.sort(key=lambda v: -v["size_mb"])
    first = usable[0] if usable else None
    if first:
        print(f"\n--- GUI 리테스트 추천 ({first['size_mb']:.2f}MB) ---")
        print(f"  python plate_gui.py \"{first['path']}\"")
        print(f"  python scripts/run_retest_pro.py \"{first['path']}\"")
    else:
        print("\n--- 추천 영상 없음 (0.1MB 이상 영상 필요) ---")
    return under


if __name__ == "__main__":
    list_100mb = main()
