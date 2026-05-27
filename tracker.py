"""tracker.py — IoU 기반 번호판 추적 도메인 모듈.

PlateTracker를 plate_gui.py에서 분리하여 단일 책임 원칙(SRP)을 적용한다.
GUI 의존성이 전혀 없으므로 헤드리스 테스트(Pytest 등)에서 단독 검증 가능.

- 동작: IoU + bbox 면적/중심 검증으로 동일 차량을 추적, 새 차량 진입 시
  이전 OCR 결과(Ghost)를 즉시 무효화한다.
- 외부 인터페이스(__init__, update, reset, calculate_iou)는 plate_gui.py
  기존 코드와 100% 호환되도록 시그니처를 보존했다.
"""

from __future__ import annotations

from typing import Final


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 매직 넘버 → Final 상수 (PEP 591 스타일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 트랙에 OCR 결과가 있을 때, 마지막 OCR 이후 이 프레임 수 이상 지나면
# 다른 차량이 같은 위치에 들어온 것으로 판단하고 OCR 결과를 지운다.
# ★ OCR 처리 1-5초 → 30fps에서 30-150프레임 간격 발생하므로 여유 있게 설정
STALE_FRAME_GAP: Final[int] = 15  # 0.5초 (30fps × 0.5) — 즉시 잔상 제거

# bbox 면적 비율이 이 범위를 벗어나면 다른 차량으로 판단
AREA_RATIO_MIN: Final[float] = 0.5
AREA_RATIO_MAX: Final[float] = 2.0

# bbox 중심이 대각선 길이의 이 비율 이상 이동하면 다른 차량으로 판단
CENTER_JUMP_RATIO: Final[float] = 0.5

# OCR 결과 확인 횟수 (PlateEnginePro가 이미 consecutive=3을 요구하므로
# GUI에서는 1회 수신 즉시 표시 — 이중 지연 방지)
CONSECUTIVE_REQUIRED: Final[int] = 1

# bbox EMA 평활화 계수 (0=이전 유지, 1=새 값 즉시 적용)
BBOX_SMOOTH_ALPHA: Final[float] = 0.8

# OCR 확인된 트랙의 표시 유지 프레임 수
# ★ OCR 4-8초 지연 → 그 사이 결과 유지 필요 (30fps 기준 120-240프레임)
DISPLAY_HOLD_FRAMES: Final[int] = 30  # 1초 유지 (30fps × 1) — 잔상 최소화

# 프레임 갭 허용치 (이 이내 미감지는 차량 교체로 보지 않음)
GAP_TOLERANCE: Final[int] = 2  # 즉시 차량 교체 감지 (5→2 고스트 방지)

# 출력 NMS 임계값 (Phase 2 우선, 같은 영역 Phase 1 억제)
_NMS_IOU_THRESHOLD: Final[float] = 0.3

# 뒷자리 매칭 임계 (Ghost 감지: 뒷4자리 중 일치 자릿수가 이 미만이면 별차량)
_TAIL_MATCH_MIN: Final[int] = 2

# 유사 텍스트 판별 한계치 (Hamming 차이 허용 자릿수)
_SIMILAR_HAMMING_MAX: Final[int] = 2


class PlateTracker:
    """IoU 기반으로 동일 차량을 추적하고, 새 차량 진입 시 이전 결과를 초기화한다.

    동작 원리:
      1. 현재 프레임 detection과 기존 track의 IoU 계산
      2. IoU >= threshold → 같은 차량 → OCR 결과 유지/업데이트
      3. IoU < threshold → 새 차량 → 이전 OCR 결과 완전 초기화
      4. TTL 초과 track → 삭제 (화면에서 사라진 차량)
    """

    # ── 클래스 상수(공개 API 호환 보존) ─────────────────────
    # 외부 코드/테스트가 `PlateTracker.STALE_FRAME_GAP` 등 클래스 속성에
    # 접근할 수 있으므로 모듈 상수와 동일 값을 클래스에서도 노출한다.
    STALE_FRAME_GAP = STALE_FRAME_GAP
    AREA_RATIO_MIN = AREA_RATIO_MIN
    AREA_RATIO_MAX = AREA_RATIO_MAX
    CENTER_JUMP_RATIO = CENTER_JUMP_RATIO
    CONSECUTIVE_REQUIRED = CONSECUTIVE_REQUIRED
    BBOX_SMOOTH_ALPHA = BBOX_SMOOTH_ALPHA
    DISPLAY_HOLD_FRAMES = DISPLAY_HOLD_FRAMES
    GAP_TOLERANCE = GAP_TOLERANCE

    def __init__(self, iou_threshold: float = 0.35, max_ttl: int = 8) -> None:
        """IoU 임계값과 트랙 TTL 상한으로 추적기를 초기화한다."""
        self.iou_threshold: float = iou_threshold
        self.max_ttl: int = max_ttl
        self.tracks: dict[int, dict] = {}  # track_id → track 정보
        self._next_id: int = 0

    @staticmethod
    def calculate_iou(box1: list, box2: list) -> float:
        """두 bbox [x1,y1,x2,y2]의 IoU를 계산한다."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        # Guard Clause — 분모 0 방어
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def update(self, detections: list[dict], frame_idx: int) -> list[dict]:
        """현재 프레임의 detection 리스트로 tracker를 업데이트하고,
        추적 결과가 반영된 detection 리스트를 반환한다.

        - 같은 차량: 이전 OCR 결과를 유지 (새 OCR가 더 좋으면 교체)
        - 새 차량: 깨끗한 상태로 시작 (Ghost 결과 제거)
        """
        # 매칭되지 않은 detection / track 추적용
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        output: list[dict] = []

        # 1단계: 각 detection에 대해 가장 높은 IoU를 가진 기존 track 찾기
        det_track_pairs: list[tuple[int, int, float]] = []
        for d_idx, det in enumerate(detections):
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            best_iou = 0.0
            best_tid = -1
            for tid, track in self.tracks.items():
                iou = self.calculate_iou(bbox, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            det_track_pairs.append((d_idx, best_tid, best_iou))

        # IoU 높은 순으로 정렬하여 greedy 매칭
        det_track_pairs.sort(key=lambda x: x[2], reverse=True)

        for d_idx, best_tid, best_iou in det_track_pairs:
            det = detections[d_idx]
            bbox = det.get("bbox", [])

            if best_iou >= self.iou_threshold and best_tid >= 0 and best_tid not in matched_track_ids:
                # ── IoU 매칭됨: 같은 위치의 차량 ──
                track = self.tracks[best_tid]

                # ★ 프레임 갭 체크: GAP_TOLERANCE 프레임 이상 미감지 후 재매칭 → 다른 차량 가능성
                #   OCR 처리 지연(500ms+)으로 detection 간격이 벌어질 수 있으므로 여유 부여
                last_matched = track.get("last_matched_frame", track["frame_idx"])
                gap = frame_idx - last_matched
                if gap > self.GAP_TOLERANCE and track.get("plate_text"):
                    track["plate_text"] = ""
                    track["confidence"] = 0
                    track["ocr_count"] = 0
                    track["det_data"] = {}
                    track["last_ocr_frame"] = 0  # display hold 즉시 해제

                # ★ bbox 면적 비율 체크: 크기가 급변하면 다른 차량
                old_bbox = track["bbox"]
                old_area = max(1, (old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1]))
                new_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                area_ratio = new_area / old_area
                if not (self.AREA_RATIO_MIN <= area_ratio <= self.AREA_RATIO_MAX):
                    track["plate_text"] = ""
                    track["confidence"] = 0
                    track["ocr_count"] = 0
                    track["det_data"] = {}
                    track["last_ocr_frame"] = 0  # display hold 즉시 해제

                # ★ bbox 중심 이동 거리 체크: 중심이 크게 점프하면 다른 차량
                old_cx = (old_bbox[0] + old_bbox[2]) / 2
                old_cy = (old_bbox[1] + old_bbox[3]) / 2
                new_cx = (bbox[0] + bbox[2]) / 2
                new_cy = (bbox[1] + bbox[3]) / 2
                old_diag = max(1, ((old_bbox[2] - old_bbox[0])**2 + (old_bbox[3] - old_bbox[1])**2) ** 0.5)
                center_dist = ((new_cx - old_cx)**2 + (new_cy - old_cy)**2) ** 0.5
                if center_dist > old_diag * self.CENTER_JUMP_RATIO and track.get("plate_text"):
                    track["plate_text"] = ""
                    track["confidence"] = 0
                    track["ocr_count"] = 0
                    track["det_data"] = {}
                    track["last_ocr_frame"] = 0  # display hold 즉시 해제

                det_text = det.get("text", "")
                det_conf = det.get("ocr_confidence", det.get("confidence", 0))
                det_valid = det.get("is_valid_plate", False)

                # ★ 마지막 OCR 이후 프레임 간격이 크면 이전 OCR 결과 폐기
                last_ocr_frame = track.get("last_ocr_frame", track["frame_idx"])
                ocr_gap = frame_idx - last_ocr_frame
                if ocr_gap > self.STALE_FRAME_GAP and track.get("plate_text"):
                    track["plate_text"] = ""
                    track["confidence"] = 0
                    track["ocr_count"] = 0
                    track["det_data"] = {}
                    track["last_ocr_frame"] = 0  # display hold 즉시 해제

                # ★ bbox EMA 평활화: 좌표 떨림 방지
                alpha = self.BBOX_SMOOTH_ALPHA
                old_b = track["bbox"]
                track["bbox"] = [
                    old_b[0] * (1 - alpha) + bbox[0] * alpha,
                    old_b[1] * (1 - alpha) + bbox[1] * alpha,
                    old_b[2] * (1 - alpha) + bbox[2] * alpha,
                    old_b[3] * (1 - alpha) + bbox[3] * alpha,
                ]
                track["ttl"] = self.max_ttl
                track["frame_idx"] = frame_idx
                track["last_matched_frame"] = frame_idx

                if det_valid and det_text:
                    # 새 OCR 결과가 있으면: 신뢰도 비교 후 교체
                    track["last_ocr_frame"] = frame_idx

                    # ★ Ghost 완전 제거: 뒷4자리 비교로 다른 차량 즉시 감지
                    #   캐시 뒷4자리 vs 엔진 뒷4자리 → 일치 2자리 미만이면 캐시 삭제
                    _cached_text = track.get("plate_text", "")
                    if _cached_text and len(_cached_text) >= 4 and len(det_text) >= 4:
                        _cached_tail = _cached_text[-4:]
                        _new_tail = det_text[-4:]
                        _match_count = sum(1 for a, b in zip(_cached_tail, _new_tail) if a == b)
                        if _match_count < _TAIL_MATCH_MIN:
                            track["plate_text"] = ""
                            track["confidence"] = 0
                            track["ocr_count"] = 0
                            track["det_data"] = {}

                    # 같은 번호면 연속 카운트 증가, 다른 번호면 리셋
                    if det_text == track.get("plate_text", ""):
                        track["ocr_count"] = track.get("ocr_count", 0) + 1
                    else:
                        # ★ 유사 텍스트면 OCR 노이즈로 간주 (리셋 방지)
                        _old_text = track.get("plate_text", "")
                        if _old_text and self._text_similar_quick(_old_text, det_text):
                            track["ocr_count"] = track.get("ocr_count", 0) + 1
                        else:
                            track["ocr_count"] = 1
                    if det_conf >= track.get("confidence", 0):
                        track["plate_text"] = det_text
                        track["confidence"] = det_conf
                        track["det_data"] = det
                    # 연속 감지 횟수 미달이면 Phase 1로 표시
                    if track.get("ocr_count", 0) >= self.CONSECUTIVE_REQUIRED:
                        result_det = dict(track.get("det_data", det))
                        result_det["bbox"] = track["bbox"]  # 평활화된 bbox 사용
                    else:
                        result_det = det
                        result_det["is_valid_plate"] = False
                else:
                    # Phase 1 (OCR 미완료): 이전 확인 결과를 DISPLAY_HOLD_FRAMES 동안 유지
                    # → OCR 주기 사이에도 초록 박스 유지하여 떨림 방지
                    _last_ocr = track.get("last_ocr_frame", 0)
                    _hold_ok = (frame_idx - _last_ocr) <= self.DISPLAY_HOLD_FRAMES
                    if track.get("plate_text") and track.get("ocr_count", 0) >= self.CONSECUTIVE_REQUIRED and _hold_ok:
                        result_det = dict(track.get("det_data", {}))
                        result_det["bbox"] = track["bbox"]  # 평활화된 bbox 사용
                    else:
                        result_det = det

                output.append(result_det)
                matched_track_ids.add(best_tid)
                matched_det_indices.add(d_idx)
            elif d_idx not in matched_det_indices:
                # ── 새 차량: 깨끗한 상태로 track 생성 ──
                new_id = self._next_id
                self._next_id += 1
                det_text = det.get("text", "")
                det_conf = det.get("ocr_confidence", det.get("confidence", 0))
                has_ocr = det.get("is_valid_plate", False) and det_text
                self.tracks[new_id] = {
                    "bbox": bbox,
                    "plate_text": det_text if has_ocr else "",
                    "confidence": det_conf if has_ocr else 0,
                    "ttl": self.max_ttl,
                    "frame_idx": frame_idx,
                    "last_matched_frame": frame_idx,
                    "last_ocr_frame": frame_idx if has_ocr else 0,
                    "ocr_count": 1 if has_ocr else 0,
                    "det_data": det if has_ocr else {},
                }
                # 새 차량 첫 감지: consecutive 미달이면 Phase 1로 표시
                new_det = dict(det)
                if has_ocr and self.CONSECUTIVE_REQUIRED <= 1:
                    pass  # 그대로 출력
                elif has_ocr:
                    new_det["is_valid_plate"] = False
                output.append(new_det)
                matched_det_indices.add(d_idx)

        # 2단계: TTL 감소 + 만료된 track 삭제
        expired: list[int] = []
        for tid, track in self.tracks.items():
            if tid not in matched_track_ids:
                track["ttl"] -= 1
                if track["ttl"] <= 0:
                    expired.append(tid)
        for tid in expired:
            del self.tracks[tid]

        # 3단계: 출력 중복 bbox NMS — 같은 영역에 Phase 1 + Phase 2가 동시에 있으면 Phase 2만 유지
        output = self._nms_output(output)

        return output

    @staticmethod
    def _text_similar_quick(t1: str, t2: str) -> bool:
        """빠른 유사도 판별: 2글자 이하 차이면 같은 번호판으로 간주.
        engine OCR 노이즈(한글 오인식, 숫자 1↔7 등)에 대한 내성 확보."""
        t1 = t1.replace(" ", "")
        t2 = t2.replace(" ", "")
        if t1 == t2:
            return True
        if not t1 or not t2:
            return False
        if abs(len(t1) - len(t2)) > _SIMILAR_HAMMING_MAX:
            return False
        # 같은 길이: hamming 비교
        if len(t1) == len(t2):
            diff = sum(1 for a, b in zip(t1, t2) if a != b)
            return diff <= _SIMILAR_HAMMING_MAX
        # 길이 차이 1~2: 짧은 쪽이 긴 쪽에 포함되면 유사
        short, long = (t1, t2) if len(t1) < len(t2) else (t2, t1)
        if short in long:
            return True
        # 숫자 부분만 비교 (한글 오인식 흡수)
        d1 = "".join(c for c in t1 if c.isdigit())
        d2 = "".join(c for c in t2 if c.isdigit())
        if d1 and d2 and d1 == d2:
            return True
        return False

    def _nms_output(self, output: list[dict]) -> list[dict]:
        """출력 리스트에서 겹치는 bbox 중복 제거.
        같은 영역에 Phase 2 (is_valid_plate=True)와 Phase 1이 동시에 있으면
        Phase 2만 유지하여 깜빡임 방지."""
        if len(output) <= 1:
            return output
        # Phase 2 (확정) 우선 정렬
        output.sort(key=lambda d: (d.get("is_valid_plate", False), d.get("ocr_confidence", 0)), reverse=True)
        keep: list[dict] = []
        suppressed: set[int] = set()
        for i, det_i in enumerate(output):
            if i in suppressed:
                continue
            keep.append(det_i)
            bbox_i = det_i.get("bbox", [])
            if len(bbox_i) < 4:
                continue
            for j in range(i + 1, len(output)):
                if j in suppressed:
                    continue
                bbox_j = output[j].get("bbox", [])
                if len(bbox_j) < 4:
                    continue
                iou = self.calculate_iou(bbox_i, bbox_j)
                if iou >= _NMS_IOU_THRESHOLD:
                    # 겹치는 bbox → 낮은 우선순위(Phase 1) 억제
                    suppressed.add(j)
        return keep

    def reset(self) -> None:
        """모든 track을 초기화한다 (영상 변경 시)."""
        self.tracks.clear()
        self._next_id = 0


__all__ = ["PlateTracker"]
