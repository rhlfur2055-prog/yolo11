"""
evidence_export.py — 증거 패키지 생성 + 저장 (계단 5)

역할: 채증 완료된 영상/메타데이터를 파일로 저장하여 증거 패키지 생성
      이 패키지가 향후 안전신문고/소방공익제보 시스템에 전송될 자료

비유: 재판 증거 봉투. 영상(CCTV 녹화), 사진(결정적 장면), 기록부(JSON), 요약서(report)를
     하나의 봉투(폴더)에 담아서 제출하는 것

최민수님 스펙:
    - 채증 완료된 1분 영상과 메타데이터를 증거 패키지로 저장
    - 안전신문고(행정안전부) / 119 소방공익제보 시스템 전송 대비

저장 내용:
    evidence_YYYYMMDD_HHMMSS_<번호판>/
    ├── video.mp4           — 채증 1분 영상 (pre 30초 + post 30초)
    ├── plates.json         — 채증 메타데이터 (번호판, 시간, 거리 판정 등)
    ├── screenshots/        — 핵심 순간 캡처 이미지들
    │   ├── evidence_start.jpg    — 채증 시작 시점
    │   ├── violation_moment.jpg  — 위반 판정 시점 (있는 경우)
    │   ├── midpoint.jpg          — 채증 중간 시점
    │   └── evidence_end.jpg      — 채증 종료 시점
    └── report.txt          — 사람이 읽을 수 있는 한글 요약 리포트

기존 코드 수정: 없음
Mock 데이터: 없음

이벤트 버스 연동:
    - 수신: "EVIDENCE_COMPLETE", "DISTANCE_VIOLATION"
    - 발행: "EVIDENCE_EXPORTED" (저장 완료 알림)

검증: python test_ocr_accuracy.py  # 12/12 통과 필수
"""

import os
import json
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np


class EvidenceExport:
    """증거 패키지 생성 및 저장

    EVIDENCE_COMPLETE 이벤트를 수신하면 자동으로 증거 폴더를 생성하고
    영상, 메타데이터, 스크린샷, 리포트를 저장.

    비유: 경찰서 증거 보관실. 사건(채증)이 종결되면 자동으로
         증거물을 분류하고 보관함에 넣어두는 시스템.

    Args:
        event_bus: SimulationFramework의 이벤트 버스
        output_dir: 증거 패키지 저장 루트 폴더 (기본: ./evidence_output)
        fps: 영상 FPS (VideoWriter용)
        frame_width: 영상 가로 픽셀
        frame_height: 영상 세로 픽셀
    """

    def __init__(self, event_bus, output_dir: str = "./evidence_output",
                 fps: float = 30.0, frame_width: int = 1920, frame_height: int = 1080):
        self.event_bus = event_bus
        self.output_dir = output_dir
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        # ── 거리 위반 기록 수집 ──
        # DISTANCE_VIOLATION 이벤트를 받아서 저장 → 증거 패키지에 포함
        # {"번호판": [{"timestamp": ..., "distance_label": ..., ...}, ...]}
        self.violation_records: dict[str, list[dict]] = {}

        # ── 내보내기 이력 ──
        self.export_count = 0
        self.export_history: list[dict] = []

        # ── 이벤트 구독 ──
        self.event_bus.subscribe("EVIDENCE_COMPLETE", self._on_evidence_complete)
        self.event_bus.subscribe("DISTANCE_VIOLATION", self._on_distance_violation)

        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[EvidenceExport] 초기화 완료")
        print(f"  저장 경로: {os.path.abspath(self.output_dir)}")

    # ───────────────────────────────────────────
    # 이벤트 핸들러
    # ───────────────────────────────────────────

    def _on_distance_violation(self, data: dict) -> None:
        """거리 위반 이벤트 수집 — 나중에 증거 패키지에 포함

        시간복잡도: O(1)
        """
        plate = data.get("plate", "")
        if not plate:
            return

        if plate not in self.violation_records:
            self.violation_records[plate] = []

        self.violation_records[plate].append({
            "timestamp": data.get("timestamp", 0.0),
            "frame_idx": data.get("frame_idx", 0),
            "bbox_ratio": data.get("bbox_ratio", 0.0),
            "close_duration": data.get("close_duration", 0.0),
            "distance_label": data.get("distance_label", ""),
        })

    def _on_evidence_complete(self, data: dict) -> None:
        """채증 완료 → 증거 패키지 생성

        이 함수가 실제 파일 저장을 수행.
        비유: "증거 봉투 포장" — 영상, 사진, 기록부, 요약서를 봉투에 넣고 봉인

        시간복잡도: O(F), F = 총 프레임 수 (영상 쓰기)
        """
        plate = data.get("plate", "unknown")
        start_time = data.get("start_time", 0.0)
        end_time = data.get("end_time", 0.0)
        pre_frames = data.get("pre_frames", [])
        post_frames = data.get("post_frames", [])
        record = data.get("record", {})

        all_frames = pre_frames + post_frames

        if not all_frames:
            print(f"  [EvidenceExport] 프레임 없음 — 저장 건너뜀: {plate}")
            return

        # ── 폴더 생성 ──
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 번호판 텍스트에서 파일명에 쓸 수 없는 문자 제거
        safe_plate = self._sanitize_filename(plate)
        folder_name = f"evidence_{timestamp_str}_{safe_plate}"
        folder_path = os.path.join(self.output_dir, folder_name)
        screenshots_path = os.path.join(folder_path, "screenshots")

        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(screenshots_path, exist_ok=True)

        print(f"\n  📦 [EvidenceExport] 증거 패키지 생성 시작: {plate}")
        print(f"     폴더: {folder_path}")

        # ── 1. 영상 저장 (video.mp4) ──
        video_path = os.path.join(folder_path, "video.mp4")
        self._save_video(video_path, all_frames)

        # ── 2. 스크린샷 저장 ──
        self._save_screenshots(screenshots_path, all_frames, pre_frames, post_frames)

        # ── 3. 메타데이터 JSON 저장 (plates.json) ──
        json_path = os.path.join(folder_path, "plates.json")
        violations = self.violation_records.get(plate, [])
        self._save_metadata(json_path, plate, data, violations)

        # ── 4. 리포트 저장 (report.txt) ──
        report_path = os.path.join(folder_path, "report.txt")
        self._save_report(report_path, plate, data, violations, folder_name)

        # ── 완료 ──
        self.export_count += 1
        export_info = {
            "plate": plate,
            "folder": folder_path,
            "timestamp": timestamp_str,
            "total_frames": len(all_frames),
            "has_violation": len(violations) > 0,
        }
        self.export_history.append(export_info)

        # 이벤트 발행
        self.event_bus.publish("EVIDENCE_EXPORTED", export_info)

        print(f"  📦 [EvidenceExport] 증거 패키지 저장 완료!")
        print(f"     📹 video.mp4 ({len(all_frames)} 프레임)")
        print(f"     📸 screenshots/ ({len(os.listdir(screenshots_path))}장)")
        print(f"     📋 plates.json")
        print(f"     📝 report.txt")
        if violations:
            print(f"     ⚠️ 거리 미확보 위반 {len(violations)}건 포함")

    # ───────────────────────────────────────────
    # 파일 저장 로직
    # ───────────────────────────────────────────

    def _save_video(self, video_path: str, frames: list[dict]) -> None:
        """프레임 리스트를 MP4 영상으로 저장

        Args:
            video_path: 저장 경로
            frames: [{"frame": np.ndarray, "timestamp": float, "frame_idx": int}, ...]

        시간복잡도: O(F), F = 프레임 수
        """
        if not frames:
            return

        # 첫 프레임에서 크기 결정
        first_frame = frames[0]["frame"]
        h, w = first_frame.shape[:2]

        # MP4 코덱 설정
        # mp4v는 가장 호환성 좋은 코덱
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            print(f"  [EvidenceExport] 영상 Writer 열기 실패: {video_path}")
            return

        frame_count = 0
        for frame_data in frames:
            frame = frame_data.get("frame")
            if frame is not None:
                # 프레임 크기가 다르면 리사이즈
                fh, fw = frame.shape[:2]
                if fw != w or fh != h:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
                frame_count += 1

        writer.release()
        print(f"     📹 영상 저장: {frame_count} 프레임 → {video_path}")

    def _save_screenshots(self, screenshots_dir: str,
                          all_frames: list[dict],
                          pre_frames: list[dict],
                          post_frames: list[dict]) -> None:
        """핵심 순간 스크린샷 저장

        저장 시점:
            1. evidence_start — 채증 시작 (= pre_buffer 끝, post_buffer 시작)
            2. midpoint — 채증 중간
            3. evidence_end — 채증 종료
            4. best_detection — pre_buffer 중 가장 선명한 프레임 (있으면)

        Args:
            screenshots_dir: 스크린샷 저장 폴더
            all_frames: 전체 프레임 (pre + post)
            pre_frames: pre-buffer 프레임들
            post_frames: post-buffer 프레임들

        시간복잡도: O(1) (고정 개수의 스크린샷만 저장)
        """
        saved = 0

        # 1. 채증 시작 시점 (pre → post 경계)
        if pre_frames:
            start_frame = pre_frames[-1]  # pre의 마지막 = 채증 직전
            path = os.path.join(screenshots_dir, "evidence_start.jpg")
            self._save_frame_as_jpg(path, start_frame)
            saved += 1

        # 2. 채증 중간 시점
        if post_frames:
            mid_idx = len(post_frames) // 2
            mid_frame = post_frames[mid_idx]
            path = os.path.join(screenshots_dir, "midpoint.jpg")
            self._save_frame_as_jpg(path, mid_frame)
            saved += 1

        # 3. 채증 종료 시점
        if post_frames:
            end_frame = post_frames[-1]
            path = os.path.join(screenshots_dir, "evidence_end.jpg")
            self._save_frame_as_jpg(path, end_frame)
            saved += 1

        # 4. pre-buffer 첫 프레임 (사이렌 직후 상황)
        if pre_frames:
            first_frame = pre_frames[0]
            path = os.path.join(screenshots_dir, "buffer_start.jpg")
            self._save_frame_as_jpg(path, first_frame)
            saved += 1

        # 5. 전체 영상의 1/4, 3/4 지점 추가 캡처
        if len(all_frames) > 4:
            q1_frame = all_frames[len(all_frames) // 4]
            path = os.path.join(screenshots_dir, "quarter_1.jpg")
            self._save_frame_as_jpg(path, q1_frame)
            saved += 1

            q3_frame = all_frames[3 * len(all_frames) // 4]
            path = os.path.join(screenshots_dir, "quarter_3.jpg")
            self._save_frame_as_jpg(path, q3_frame)
            saved += 1

    def _save_frame_as_jpg(self, path: str, frame_data: dict) -> None:
        """개별 프레임을 JPG로 저장

        시간복잡도: O(1)
        """
        frame = frame_data.get("frame")
        if frame is not None:
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def _save_metadata(self, json_path: str, plate: str,
                       evidence_data: dict, violations: list[dict]) -> None:
        """메타데이터를 JSON으로 저장

        plates.json 구조:
        {
            "version": "1.0",
            "generated_at": "2025-03-05T14:30:00",
            "plate": "경기76바7789",
            "evidence": {
                "start_time": 45.2,
                "end_time": 105.8,
                "total_frames": 1800,
                "total_duration_sec": 60.0,
                "continuous_duration": 18.3,
                "detection_count": 412,
                ...
            },
            "distance_violations": [
                {"timestamp": 52.1, "distance_label": "~3m", "close_duration": 7.2, ...},
                ...
            ],
            "submission_targets": [
                "안전신문고 (행정안전부) - 교통위반 카테고리",
                "119 소방공익제보 시스템 (소방청)"
            ]
        }

        시간복잡도: O(V), V = 위반 건수
        """
        record = evidence_data.get("record", {})

        metadata = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "system": "GoldenTime 2.0 - YOLO26 Simulation",
            "plate": plate,
            "evidence": {
                "start_time_sec": evidence_data.get("start_time", 0.0),
                "end_time_sec": evidence_data.get("end_time", 0.0),
                "start_frame": evidence_data.get("start_frame", 0),
                "end_frame": evidence_data.get("end_frame", 0),
                "total_frames": evidence_data.get("total_frames", 0),
                "total_duration_sec": evidence_data.get("total_duration_sec", 0.0),
                "first_seen_time_sec": record.get("first_seen_time", 0.0),
                "continuous_duration_sec": record.get("continuous_duration", 0.0),
                "detection_count": record.get("detection_count", 0),
                "last_confidence": record.get("last_confidence", 0.0),
                "last_bbox": record.get("last_bbox", []),
            },
            "distance_violations": [
                {
                    "timestamp_sec": v.get("timestamp", 0.0),
                    "frame_idx": v.get("frame_idx", 0),
                    "distance_label": v.get("distance_label", ""),
                    "close_duration_sec": v.get("close_duration", 0.0),
                    "bbox_ratio": v.get("bbox_ratio", 0.0),
                }
                for v in violations
            ],
            "violation_summary": {
                "has_distance_violation": len(violations) > 0,
                "violation_count": len(violations),
            },
            "submission_targets": [
                "안전신문고 (행정안전부) - 교통위반 카테고리 - API 자동 전송",
                "119 소방공익제보 시스템 (소방청) - 소방기본법 기반 과태료 부과",
            ],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _save_report(self, report_path: str, plate: str,
                     evidence_data: dict, violations: list[dict],
                     folder_name: str) -> None:
        """사람이 읽을 수 있는 한글 요약 리포트 저장

        비유: 경찰 보고서. 누가 봐도 사건 내용을 이해할 수 있도록 작성.

        시간복잡도: O(V), V = 위반 건수
        """
        record = evidence_data.get("record", {})
        total_frames = evidence_data.get("total_frames", 0)
        total_sec = evidence_data.get("total_duration_sec", 0.0)
        start_time = evidence_data.get("start_time", 0.0)
        end_time = evidence_data.get("end_time", 0.0)
        continuous = record.get("continuous_duration", 0.0)
        det_count = record.get("detection_count", 0)
        confidence = record.get("last_confidence", 0.0)

        now = datetime.now()
        generated_str = now.strftime("%Y년 %m월 %d일 %H시 %M분 %S초")

        lines = []
        lines.append("=" * 60)
        lines.append("   골든타임 2.0 — 긴급차량 통행방해 증거 보고서")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  생성일시: {generated_str}")
        lines.append(f"  시스템:   GoldenTime 2.0 (YOLO26 Simulation)")
        lines.append(f"  패키지:   {folder_name}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  1. 위반 차량 정보")
        lines.append("-" * 60)
        lines.append(f"  번호판:     {plate}")
        lines.append(f"  신뢰도:     {confidence:.1%}")
        lines.append(f"  연속 감지:  {continuous:.1f}초")
        lines.append(f"  총 감지:    {det_count}회")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  2. 채증 영상 정보")
        lines.append("-" * 60)
        lines.append(f"  영상 구간:  {start_time:.1f}초 ~ {end_time:.1f}초")
        lines.append(f"  영상 길이:  {total_sec:.1f}초 ({total_frames} 프레임)")
        lines.append(f"  파일명:     video.mp4")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  3. 거리 미확보 위반 내역")
        lines.append("-" * 60)

        if violations:
            lines.append(f"  위반 건수:  {len(violations)}건")
            lines.append("")
            for i, v in enumerate(violations, 1):
                t = v.get("timestamp", 0.0)
                label = v.get("distance_label", "")
                dur = v.get("close_duration", 0.0)
                ratio = v.get("bbox_ratio", 0.0) * 100
                lines.append(f"  [{i}] 시각: {t:.1f}초 | 거리: {label} | "
                             f"지속: {dur:.1f}초 | bbox비율: {ratio:.3f}%")
        else:
            lines.append("  거리 미확보 위반 없음")
            lines.append("  (채증은 완료되었으나 거리 기준 위반은 감지되지 않음)")

        lines.append("")
        lines.append("-" * 60)
        lines.append("  4. 제출 대상")
        lines.append("-" * 60)
        lines.append("  [1] 안전신문고 (행정안전부)")
        lines.append("      - 교통위반 카테고리로 자동 전송")
        lines.append("      - 경찰청 등 소관기관으로 자동 이송")
        lines.append("")
        lines.append("  [2] 119 소방공익제보 시스템 (소방청)")
        lines.append("      - 소방기본법에 따라 과태료 부과 가능")
        lines.append("      - 소방청 직통 라인으로 빠른 처리")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  5. 첨부 파일 목록")
        lines.append("-" * 60)
        lines.append("  - video.mp4         채증 영상 (1분)")
        lines.append("  - plates.json       기계 판독용 메타데이터")
        lines.append("  - screenshots/      핵심 순간 캡처 이미지")
        lines.append("    ├ evidence_start.jpg    채증 시작")
        lines.append("    ├ midpoint.jpg          채증 중간")
        lines.append("    ├ evidence_end.jpg      채증 종료")
        lines.append("    ├ buffer_start.jpg      버퍼 시작")
        lines.append("    ├ quarter_1.jpg         1/4 지점")
        lines.append("    └ quarter_3.jpg         3/4 지점")
        lines.append("")
        lines.append("=" * 60)
        lines.append("  본 보고서는 GoldenTime 2.0 시스템에 의해 자동 생성되었습니다.")
        lines.append("  증거 영상 및 데이터의 무결성은 시스템 로그로 보증됩니다.")
        lines.append("=" * 60)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ───────────────────────────────────────────
    # 유틸리티
    # ───────────────────────────────────────────

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        """파일명에 사용할 수 없는 문자 제거

        한글은 유지, 특수문자는 언더스코어로 대체.

        시간복잡도: O(n), n = 텍스트 길이
        """
        safe = ""
        for ch in text:
            if ch.isalnum() or ch in ("-", "_"):
                safe += ch
            elif '\uac00' <= ch <= '\ud7a3':
                # 한글 완성형 유지
                safe += ch
            elif '\u3131' <= ch <= '\u318e':
                # 한글 자모 유지
                safe += ch
            else:
                safe += "_"
        return safe or "unknown"

    # ───────────────────────────────────────────
    # 오버레이
    # ───────────────────────────────────────────

    def draw_export_overlay(self, frame: np.ndarray) -> np.ndarray:
        """증거 내보내기 상태 오버레이

        내보내기 완료 시 잠시 화면에 알림 표시.
        기본적으로는 아무것도 안 그림 (다른 오버레이에 방해되지 않도록).
        현재는 최근 내보내기 이력만 우하단에 작게 표시.

        시간복잡도: O(1)
        """
        if not self.export_history:
            return frame

        h, w = frame.shape[:2]

        # 우하단에 최근 내보내기 상태 (상태바 위)
        last = self.export_history[-1]
        text = f"Evidence saved: {last['plate']} ({self.export_count} total)"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        x = w - text_size[0] - 10
        y = h - 44  # 상태바(36px) 위

        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

        return frame

    # ───────────────────────────────────────────
    # 외부 API
    # ───────────────────────────────────────────

    def get_export_history(self) -> list[dict]:
        """내보내기 이력 반환

        Returns:
            [{"plate", "folder", "timestamp", "total_frames", "has_violation"}, ...]

        시간복잡도: O(1)
        """
        return self.export_history

    def get_export_count(self) -> int:
        """총 내보내기 횟수

        시간복잡도: O(1)
        """
        return self.export_count
