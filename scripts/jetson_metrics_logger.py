# -*- coding: utf-8 -*-
"""
Jetson 실차 테스트 메트릭 수집: CSV/JSON 로그 (violation latency, FP rate, OCR hit rate, blur, power).

- 세션 요약 CSV: logs/jetson_metrics_*.csv
- 이벤트 단위 JSON: logs/events_*.json (호출 측에서 기록 시 사용)
- tegrastats 파싱: power_watt, memory_mb (선택)

사용법:
  # 다른 스크립트에서 임포트해 이벤트 기록
  from scripts.jetson_metrics_logger import MetricsLogger
  logger = MetricsLogger(logs_dir="logs", session_id="sess_001")
  logger.log_violation(latency_sec=2.3, plate="경기76바7789", type="DISTANCE_VIOLATION")
  logger.write_summary(fp_rate=0.0, ocr_hit_rate=0.92)

  # tegrastats 로그 파싱 (평균 전력/메모리)
  python scripts/jetson_metrics_logger.py --parse-tegra logs/tegrastats_20260307_091500.log
"""
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"


class MetricsLogger:
    """실차 테스트 메트릭 CSV/JSON 기록."""

    def __init__(self, logs_dir=None, session_id=None):
        self.logs_dir = Path(logs_dir or LOGS_DIR)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.violations = []
        self.start_time = datetime.now(timezone.utc).isoformat()

    def log_violation(self, latency_sec, plate, event_type="DISTANCE_VIOLATION", event_time=None):
        event_time = event_time or datetime.now(timezone.utc).isoformat()
        self.violations.append({
            "event_time": event_time,
            "latency_sec": latency_sec,
            "plate": plate,
            "type": event_type,
        })

    def set_end_time(self, end_time=None):
        self.end_time = (end_time or datetime.now(timezone.utc)).isoformat()

    def write_summary(
        self,
        total_frames,
        fp_rate=0.0,
        ocr_hit_rate=None,
        blur_affected_frames=0,
        evidence_count=0,
        fps_avg=0.0,
        memory_mb=None,
        power_watt=None,
    ):
        self.set_end_time()
        payload = {
            "session_id": self.session_id,
            "start": self.start_time,
            "end": self.end_time,
            "violations": self.violations,
            "metrics": {
                "total_frames": total_frames,
                "fp_rate": fp_rate,
                "ocr_hit_rate": ocr_hit_rate,
                "blur_affected_frames": blur_affected_frames,
                "evidence_count": evidence_count,
                "fps_avg": fps_avg,
                "memory_mb": memory_mb,
                "power_watt": power_watt,
            },
        }
        out = self.logs_dir / f"events_{self.session_id.replace('sess_', '')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out

    def append_csv_row(
        self,
        timestamp_utc,
        violation_latency_sec,
        fp_count,
        total_frames,
        fps_avg,
        memory_mb,
        power_watt,
        blur_affected_frames,
        evidence_count,
    ):
        csv_path = self.logs_dir / f"jetson_metrics_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        header = "timestamp_utc,session_id,violation_latency_sec,fp_count,total_frames,fps_avg,memory_mb,power_watt,blur_affected_frames,evidence_count"
        row = f"{timestamp_utc},{self.session_id},{violation_latency_sec},{fp_count},{total_frames},{fps_avg},{memory_mb},{power_watt},{blur_affected_frames},{evidence_count}"
        write_header = not csv_path.exists()
        with open(csv_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(header + "\n")
            f.write(row + "\n")
        return csv_path


def parse_tegrastats_log(log_path):
    """tegrastats 로그에서 평균 power(W), memory(MB) 추출."""
    power_sum, power_n = 0.0, 0
    mem_sum, mem_n = 0.0, 0
    # 예: RAM 2345/7772MB ... GPU 12%
    re_power = re.compile(r"POM_5V_IN\s+(\d+)/\d+")
    re_ram = re.compile(r"RAM\s+(\d+)/\d+MB")
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re_ram.search(line)
            if m:
                mem_sum += int(m.group(1))
                mem_n += 1
            # 전력은 Jetson 모델별 포맷 상이; 필요 시 보강
            if "POM_5V" in line:
                m = re_power.search(line)
                if m:
                    power_sum += int(m.group(1)) / 1000.0  # mA -> A 가정 후 W 근사
                    power_n += 1
    return {
        "memory_mb_avg": round(mem_sum / mem_n, 1) if mem_n else None,
        "power_watt_avg": round(power_sum / power_n, 2) if power_n else None,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--parse-tegra", type=str, help="tegrastats 로그 파일 경로")
    args = p.parse_args()
    if args.parse_tegra:
        out = parse_tegrastats_log(args.parse_tegra)
        print(json.dumps(out, indent=2))
    else:
        print("Usage: --parse-tegra <path> to parse tegrastats log")
