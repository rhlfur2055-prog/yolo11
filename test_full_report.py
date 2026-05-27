#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""전수 테스트 리포트 — hiway.mp4 심층 분석"""

import sys, io, os, time, cv2, re
import numpy as np
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PLATE_CONSECUTIVE_FRAMES"] = "2"

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from plate_engine_pro import PlateEnginePro

def main():
    video_path = ROOT / "movie" / "hiway.mp4"
    if not video_path.is_file():
        print(f"[에러] 영상 없음: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 70)
    print("  hiway.mp4 심층 분석 리포트")
    print("=" * 70)
    print(f"  해상도: {width}x{height} @ {fps_vid:.0f}fps")
    print(f"  총 프레임: {total_frames} ({total_frames/fps_vid:.0f}초)")

    engine = PlateEnginePro()

    SAMPLE = 3
    processed = 0
    frame_idx = 0
    all_plates = {}
    frame_times = []
    zone_stats = {"near": 0, "mid": 0, "far": 0}
    detect_frames = 0

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SAMPLE != 0:
            continue
        processed += 1

        t1 = time.perf_counter()
        dets = engine.process_frame(frame)
        dt = time.perf_counter() - t1
        frame_times.append(dt)

        if dets:
            detect_frames += 1
            for d in dets:
                plate = d.get("plate", "")
                conf = d.get("confidence", 0)
                bbox = d.get("bbox", [0, 0, 0, 0])
                if plate:
                    if plate not in all_plates:
                        all_plates[plate] = {
                            "count": 0,
                            "first_frame": frame_idx,
                            "last_frame": frame_idx,
                            "confs": [],
                            "areas": [],
                        }
                    all_plates[plate]["count"] += 1
                    all_plates[plate]["last_frame"] = frame_idx
                    all_plates[plate]["confs"].append(conf)
                    if len(bbox) == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        area_ratio = (bw * bh) / (width * height) * 100
                        all_plates[plate]["areas"].append(area_ratio)
                        if area_ratio > 5:
                            zone_stats["near"] += 1
                        elif area_ratio > 1:
                            zone_stats["mid"] += 1
                        else:
                            zone_stats["far"] += 1

        if processed % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  [{processed:4d}] F:{frame_idx} Det:{detect_frames} "
                  f"Plates:{len(all_plates)} ({processed/elapsed:.1f} proc/s)",
                  flush=True)

    cap.release()
    elapsed_total = time.time() - t_start

    # 결과 출력
    print()
    print("-" * 70)
    print(f"  처리: {processed}프레임 / 총 {frame_idx}프레임 ({SAMPLE}프레임 간격)")
    print(f"  시간: {elapsed_total:.0f}초 ({processed/elapsed_total:.1f} frames/sec)")

    if frame_times:
        avg_ms = sum(frame_times) / len(frame_times) * 1000
        p50 = sorted(frame_times)[len(frame_times) // 2] * 1000
        p95 = sorted(frame_times)[int(len(frame_times) * 0.95)] * 1000
        print(f"  지연: 평균 {avg_ms:.0f}ms, P50 {p50:.0f}ms, P95 {p95:.0f}ms")

    print(f"  감지율: {detect_frames}/{processed} "
          f"({detect_frames / max(processed, 1) * 100:.1f}%)")
    print(f"  고유 번호판: {len(all_plates)}개")

    # 거리별 분포
    print()
    print("  ── 거리별 감지 분포 ──")
    total_zone = sum(zone_stats.values()) or 1
    for z in ["near", "mid", "far"]:
        cnt = zone_stats[z]
        label = {"near": "근거리(>5%)", "mid": "중거리(1-5%)", "far": "원거리(<1%)"}[z]
        bar = "#" * int(cnt / total_zone * 30)
        print(f"    {label:15s} {cnt:3d}회 ({cnt / total_zone * 100:4.1f}%) {bar}")

    # 번호판 상세
    print()
    print("  ── 인식 번호판 상세 ──")
    print(f"    {'번호판':15s} {'횟수':>4s}  {'conf':>5s}  {'면적':>6s}  {'거리':>4s}  프레임범위")
    print("    " + "-" * 60)
    for plate, info in sorted(all_plates.items(), key=lambda x: -x[1]["count"]):
        avg_conf = sum(info["confs"]) / len(info["confs"])
        avg_area = sum(info["areas"]) / len(info["areas"]) if info["areas"] else 0
        dist = "근" if avg_area > 5 else ("중" if avg_area > 1 else "원")
        f_range = f"F{info['first_frame']}-{info['last_frame']}"
        print(f"    {plate:15s} {info['count']:3d}회  {avg_conf:4.0%}  {avg_area:5.1f}%  {dist:>3s}  {f_range}")

    # GT 매칭
    gt_plates = [
        "58두9599", "45소8019", "35무5546",
        "서울70바9203", "48보7062", "경기91바6286", "01나8060",
    ]
    print()
    print("  ── GT 매칭 (알려진 7대) ──")
    matched = 0
    for gt in gt_plates:
        found = gt in all_plates
        if found:
            matched += 1
            info = all_plates[gt]
            avg_c = sum(info["confs"]) / len(info["confs"])
            print(f"    {gt:15s} {info['count']}회 감지 (conf:{avg_c:.0%})")
        else:
            digits = re.sub(r"[^0-9]", "", gt)
            partial = [p for p in all_plates if re.sub(r"[^0-9]", "", p) == digits]
            if partial:
                print(f"    {gt:15s} 변이감지: {', '.join(partial)}")
            else:
                print(f"    {gt:15s} 미감지")

    print(f"\n  GT 매칭률: {matched}/{len(gt_plates)} ({matched / len(gt_plates) * 100:.0f}%)")

    # 취약점 분석
    print()
    print("  ── 취약점 분석 ──")
    miss = [gt for gt in gt_plates if gt not in all_plates]
    if miss:
        print(f"    미감지 차량: {', '.join(miss)}")
        print(f"    원인: 노출 시간 짧아 consecutive=2 미달 또는 OCR conf<0.70")

    if zone_stats["far"] == 0:
        print("    원거리 감지 0회 — 소형 번호판(bbox<1%) 인식 불가")

    low_conf = [(p, sum(i["confs"])/len(i["confs"]))
                for p, i in all_plates.items()
                if sum(i["confs"])/len(i["confs"]) < 0.85]
    if low_conf:
        print(f"    저신뢰 번호판: {', '.join(f'{p}({c:.0%})' for p,c in low_conf)}")

    print()
    print("=" * 70)
    print(f"  종합: 고유 {len(all_plates)}개 번호판, GT {matched}/{len(gt_plates)} 매칭")
    print("=" * 70)


if __name__ == "__main__":
    main()
