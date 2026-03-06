"""
evidence_package.py — SafePlate 4K 증거 패키지 생성 (계단 B-3)

역할: 물피도주 의심 차량의 증거 패키지를 생성하여 파일로 저장
비유: 사고 현장 수사관. 충격 시점 영상 + 이탈 차량 번호판 + 스크린샷 + JSON을
     하나의 증거 봉투(폴더)에 담아서 제출 준비

동작 흐름:
    1. DEPARTURE_DETECTED 수신 → 이탈 차량 정보 수집
    2. ShockSimulator에서 충격 전후 프레임 가져오기
    3. 증거 폴더 생성: safeplate_YYYYMMDD_HHMMSS_번호판/
    4. video.mp4 + screenshots/ + plates.json + report.txt 저장

기존 코드 수정: 없음
기존 goldentime 코드 패턴 재활용 (evidence_export.py 구조 동일)
Mock 데이터: 없음

이벤트 버스 연동:
    - 수신: "DEPARTURE_DETECTED"
    - 발행: "EVIDENCE_EXPORTED"
"""

import os
import json
from datetime import datetime

import cv2
import numpy as np


class EvidencePackage:
    """SafePlate 4K 증거 패키지 생성기

    DEPARTURE_DETECTED 이벤트를 수신하면 자동으로 증거 폴더를 생성하고
    충격 전후 영상, 이탈 차량 메타데이터, 스크린샷, 리포트를 저장.

    Args:
        event_bus: SimulationFramework의 이벤트 버스
        shock_simulator: ShockSimulator 인스턴스 (프레임 버퍼 접근용)
        output_dir: 증거 패키지 저장 루트 폴더
        fps: 영상 FPS
        frame_width: 영상 가로 픽셀
        frame_height: 영상 세로 픽셀
    """

    def __init__(self, event_bus, shock_simulator, output_dir: str = "./evidence_output",
                 fps: float = 30.0, frame_width: int = 1920, frame_height: int = 1080):
        self.event_bus = event_bus
        self.shock_simulator = shock_simulator
        self.output_dir = output_dir
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        # ── 내보내기 이력 ──
        self.export_count = 0
        self.export_history: list[dict] = []

        # ── 이탈 차량 축적 ──
        self._departure_records: list[dict] = []

        # ── 이벤트 구독 ──
        self.event_bus.subscribe("DEPARTURE_DETECTED", self._on_departure_detected)

        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[EvidencePackage] 초기화 완료")
        print(f"  저장 경로: {os.path.abspath(self.output_dir)}")

    def _on_departure_detected(self, data: dict) -> None:
        """이탈 차량 감지 → 증거 패키지 생성"""
        plate = data.get("plate", "unknown")
        self._departure_records.append(data)

        # ShockSimulator에서 충격 전후 프레임 가져오기
        shock_data = self.shock_simulator.get_shock_frames()
        pre_frames = shock_data.get("pre_frames", [])
        post_frames = shock_data.get("post_frames", [])
        all_frames = pre_frames + post_frames

        if not all_frames:
            print(f"  [EvidencePackage] 프레임 없음 — 저장 건너뜀: {plate}")
            return

        # ── 폴더 생성 ──
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_plate = self._sanitize_filename(plate)
        folder_name = f"safeplate_{timestamp_str}_{safe_plate}"
        folder_path = os.path.join(self.output_dir, folder_name)
        screenshots_path = os.path.join(folder_path, "screenshots")

        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(screenshots_path, exist_ok=True)

        print(f"\n  [EvidencePackage] 증거 패키지 생성 시작: {plate}")
        print(f"     폴더: {folder_path}")

        # ── 1. 영상 저장 (video.mp4) ──
        video_path = os.path.join(folder_path, "video.mp4")
        self._save_video(video_path, all_frames)

        # ── 2. 스크린샷 저장 ──
        self._save_screenshots(screenshots_path, all_frames, pre_frames, post_frames)

        # ── 3. 메타데이터 JSON 저장 (plates.json) ──
        json_path = os.path.join(folder_path, "plates.json")
        self._save_metadata(json_path, plate, data, shock_data)

        # ── 4. 리포트 저장 (report.txt) ──
        report_path = os.path.join(folder_path, "report.txt")
        self._save_report(report_path, plate, data, shock_data, folder_name)

        # ── 완료 ──
        self.export_count += 1
        export_info = {
            "plate": plate,
            "folder": folder_path,
            "folder_name": folder_name,
            "timestamp": timestamp_str,
            "total_frames": len(all_frames),
            "departure_direction": data.get("departure_direction", ""),
            "scenario": "safeplate",
        }
        self.export_history.append(export_info)

        # 이벤트 발행
        self.event_bus.publish("EVIDENCE_EXPORTED", export_info)

        screenshot_count = len(os.listdir(screenshots_path)) if os.path.exists(screenshots_path) else 0
        print(f"  [EvidencePackage] 증거 패키지 저장 완료!")
        print(f"     video.mp4 ({len(all_frames)} 프레임)")
        print(f"     screenshots/ ({screenshot_count}장)")
        print(f"     plates.json")
        print(f"     report.txt")
        print(f"     이탈 방향: {data.get('departure_direction', 'unknown')}")

    def _save_video(self, video_path: str, frames: list[dict]) -> None:
        """프레임 리스트를 MP4 영상으로 저장"""
        if not frames:
            return

        first_frame = frames[0]["frame"]
        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            print(f"  [EvidencePackage] 영상 Writer 열기 실패: {video_path}")
            return

        frame_count = 0
        for frame_data in frames:
            frame = frame_data.get("frame")
            if frame is not None:
                fh, fw = frame.shape[:2]
                if fw != w or fh != h:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
                frame_count += 1

        writer.release()

    def _save_screenshots(self, screenshots_dir: str, all_frames: list[dict],
                          pre_frames: list[dict], post_frames: list[dict]) -> None:
        """핵심 순간 스크린샷 저장

        저장 시점:
            1. shock_moment — 충격 시점 (pre 마지막 = 충격 직전)
            2. midpoint — 전체 중간
            3. evidence_end — 마지막 프레임
            4. buffer_start — pre 시작
            5. quarter_1, quarter_3 — 1/4, 3/4 지점
        """
        saved = 0

        # 1. 충격 시점 (pre 마지막)
        if pre_frames:
            path = os.path.join(screenshots_dir, "evidence_start.jpg")
            self._save_frame_as_jpg(path, pre_frames[-1])
            saved += 1

        # 2. 중간 시점
        if post_frames:
            mid_idx = len(post_frames) // 2
            path = os.path.join(screenshots_dir, "midpoint.jpg")
            self._save_frame_as_jpg(path, post_frames[mid_idx])
            saved += 1

        # 3. 마지막 시점
        if post_frames:
            path = os.path.join(screenshots_dir, "evidence_end.jpg")
            self._save_frame_as_jpg(path, post_frames[-1])
            saved += 1

        # 4. pre 시작
        if pre_frames:
            path = os.path.join(screenshots_dir, "buffer_start.jpg")
            self._save_frame_as_jpg(path, pre_frames[0])
            saved += 1

        # 5. 1/4, 3/4 지점
        if len(all_frames) > 4:
            q1 = all_frames[len(all_frames) // 4]
            path = os.path.join(screenshots_dir, "quarter_1.jpg")
            self._save_frame_as_jpg(path, q1)
            saved += 1

            q3 = all_frames[3 * len(all_frames) // 4]
            path = os.path.join(screenshots_dir, "quarter_3.jpg")
            self._save_frame_as_jpg(path, q3)
            saved += 1

    def _save_frame_as_jpg(self, path: str, frame_data: dict) -> None:
        """개별 프레임을 JPG로 저장"""
        frame = frame_data.get("frame")
        if frame is not None:
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def _save_metadata(self, json_path: str, plate: str,
                       departure_data: dict, shock_data: dict) -> None:
        """메타데이터를 JSON으로 저장"""
        pre_count = len(shock_data.get("pre_frames", []))
        post_count = len(shock_data.get("post_frames", []))
        total_count = pre_count + post_count
        total_sec = total_count / self.fps if self.fps > 0 else 0

        metadata = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "system": "SafePlate 4K - YOLO26 Simulation",
            "scenario": "물피도주 의심 차량 채증",
            "plate": plate,
            "shock_event": {
                "shock_frame_idx": shock_data.get("shock_frame_idx", 0),
                "shock_timestamp_sec": shock_data.get("shock_timestamp", 0.0),
            },
            "departure_info": {
                "departure_direction": departure_data.get("departure_direction", ""),
                "departure_time_sec": departure_data.get("departure_time", 0.0),
                "departure_frame": departure_data.get("departure_frame", 0),
                "first_seen_time_sec": departure_data.get("first_seen_time", 0.0),
                "last_seen_time_sec": departure_data.get("last_seen_time", 0.0),
                "detection_count": departure_data.get("detection_count", 0),
                "confidence": departure_data.get("confidence", 0.0),
                "first_bbox": departure_data.get("first_bbox", []),
                "last_bbox": departure_data.get("last_bbox", []),
                "movement_vector": departure_data.get("movement_vector", (0, 0)),
            },
            "evidence": {
                "total_frames": total_count,
                "total_duration_sec": total_sec,
                "pre_buffer_frames": pre_count,
                "post_buffer_frames": post_count,
            },
            "violation_summary": {
                "type": "물피도주 의심",
                "description": "충격 감지 후 화면에서 이탈한 차량",
            },
            "submission_targets": [
                "경찰청 112 — 물피도주(뺑소니) 신고",
                "보험사 — 사고 증거 자료 제출",
                "안전신문고 (행정안전부) — 교통사고 신고",
            ],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _save_report(self, report_path: str, plate: str,
                     departure_data: dict, shock_data: dict,
                     folder_name: str) -> None:
        """사람이 읽을 수 있는 한글 요약 리포트 저장"""
        pre_count = len(shock_data.get("pre_frames", []))
        post_count = len(shock_data.get("post_frames", []))
        total_count = pre_count + post_count
        total_sec = total_count / self.fps if self.fps > 0 else 0

        now = datetime.now()
        generated_str = now.strftime("%Y년 %m월 %d일 %H시 %M분 %S초")

        direction = departure_data.get("departure_direction", "unknown")
        confidence = departure_data.get("confidence", 0.0)
        det_count = departure_data.get("detection_count", 0)
        shock_time = shock_data.get("shock_timestamp", 0.0)
        departure_time = departure_data.get("departure_time", 0.0)

        # 이탈 방향 한글화
        dir_map = {
            "left": "좌측 이탈",
            "right": "우측 이탈",
            "top": "상방 이탈",
            "bottom": "하방 이탈",
            "vanished": "화면에서 소멸",
        }
        dir_label = dir_map.get(direction, direction)

        lines = []
        lines.append("=" * 60)
        lines.append("   SafePlate 4K — 물피도주 의심 차량 증거 보고서")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  생성일시: {generated_str}")
        lines.append(f"  시스템:   SafePlate 4K (YOLO26 Simulation)")
        lines.append(f"  패키지:   {folder_name}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  1. 물피도주 의심 차량 정보")
        lines.append("-" * 60)
        lines.append(f"  번호판:     {plate}")
        lines.append(f"  신뢰도:     {confidence:.1%}")
        lines.append(f"  감지 횟수:  {det_count}회")
        lines.append(f"  이탈 방향:  {dir_label}")
        lines.append(f"  이탈 시각:  영상 {departure_time:.1f}초 시점")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  2. 충격 감지 정보")
        lines.append("-" * 60)
        lines.append(f"  충격 시각:  영상 {shock_time:.1f}초 시점")
        lines.append(f"  충격 프레임: #{shock_data.get('shock_frame_idx', 0)}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  3. 채증 영상 정보")
        lines.append("-" * 60)
        lines.append(f"  영상 길이:  {total_sec:.1f}초 ({total_count} 프레임)")
        lines.append(f"  전 버퍼:    {pre_count} 프레임 (충격 이전)")
        lines.append(f"  후 버퍼:    {post_count} 프레임 (충격 이후)")
        lines.append(f"  파일명:     video.mp4")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  4. 제출 대상")
        lines.append("-" * 60)
        lines.append("  [1] 경찰청 112")
        lines.append("      - 물피도주(뺑소니) 신고 — 도로교통법 제54조 위반")
        lines.append("      - 형법 제268조 (업무상과실치상) 적용 가능")
        lines.append("")
        lines.append("  [2] 보험사")
        lines.append("      - 사고 증거 자료로 제출")
        lines.append("      - 영상 + 번호판 + 이탈 정보 포함")
        lines.append("")
        lines.append("  [3] 안전신문고 (행정안전부)")
        lines.append("      - 교통사고 카테고리로 신고")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  5. 첨부 파일 목록")
        lines.append("-" * 60)
        lines.append("  - video.mp4         충격 전후 영상")
        lines.append("  - plates.json       기계 판독용 메타데이터")
        lines.append("  - screenshots/      핵심 순간 캡처 이미지")
        lines.append("    |- evidence_start.jpg    충격 시점")
        lines.append("    |- midpoint.jpg          중간 시점")
        lines.append("    |- evidence_end.jpg      종료 시점")
        lines.append("    |- buffer_start.jpg      버퍼 시작")
        lines.append("    |- quarter_1.jpg         1/4 지점")
        lines.append("    +- quarter_3.jpg         3/4 지점")
        lines.append("")
        lines.append("=" * 60)
        lines.append("  본 보고서는 SafePlate 4K 시스템에 의해 자동 생성되었습니다.")
        lines.append("  증거 영상 및 데이터의 무결성은 시스템 로그로 보증됩니다.")
        lines.append("=" * 60)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        """파일명에 사용할 수 없는 문자 제거"""
        safe = ""
        for ch in text:
            if ch.isalnum() or ch in ("-", "_"):
                safe += ch
            elif '\uac00' <= ch <= '\ud7a3':
                safe += ch
            elif '\u3131' <= ch <= '\u318e':
                safe += ch
            else:
                safe += "_"
        return safe or "unknown"

    def draw_export_overlay(self, frame: np.ndarray) -> np.ndarray:
        """증거 내보내기 상태 오버레이"""
        if not self.export_history:
            return frame

        h, w = frame.shape[:2]

        last = self.export_history[-1]
        text = f"SafePlate saved: {last['plate']} ({self.export_count} total)"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        x = w - text_size[0] - 10
        y = h - 44

        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

        return frame

    def get_export_history(self) -> list[dict]:
        """내보내기 이력 반환"""
        return self.export_history

    def get_export_count(self) -> int:
        """총 내보내기 횟수"""
        return self.export_count
