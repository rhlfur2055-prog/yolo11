
# YOLO11 통합 모델 로더 (Ultralytics 최신 모델)
# YOLO11 특징: NMS-free 엔드투엔드 / YOLO11 대비 +5% 정확도
import os as _os
_os.environ["FLAGS_use_mkldnn"] = "0"

from config import PathConfig, ThresholdConfig, OCRConfig, DisplayConfig
from plate_recognition_4k import (
    correct_ocr_hangul, correct_hangul_similarity,
    _HANGUL_PLATE_CORRECTION, _find_nearest_valid_hangul,
    _VALID_PLATE_HANGUL_ALL, _correct_single_hangul,
    validate_plate_format, _correct_region, _REGION_SET,
    _DIGIT_CORRECTION,
)

# 분리된 모듈 (refactor)
# 단일 책임 원칙(SRP) 적용 — God class를 도메인별로 분할.
# plate_gui.py는 PlateEnginePro/PlateEngineFast/process_frame_unified만 import하므로
# 아래 재-export는 기존 동작과 호환된다.
from preprocessor import ImagePreprocessor, _deskew_and_otsu  # noqa: F401
from validator import PlateValidator  # noqa: F401
from db import PlateDatabase  # noqa: F401
from engine_config import PlateEngineConfig  # noqa: F401  (re-export for scripts/bench_accuracy.py)
from crnn_verifier import CrnnVerifier  # CRNN 교차검증 책임 모듈 (refactor)

def _load_best_model():
    """우선순위에 따라 가장 좋은 모델 자동 로드"""
    from ultralytics import YOLO
    best = PathConfig.find_best_model()
    print(f"[YOLO11] 모델 로드: {best}")
    return YOLO(best)
# -*- coding: utf-8 -*-
# ============================================
# plate_engine_pro.py
# 상용급 번호판 인식 엔진 (비젼인급 품질)
# ============================================

import os
import re
import time
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque


# 분리된 모듈 (refactor)
# draw_korean_text + 캐시는 ui_text.py로 이관 (SRP).
# 본 파일 내 호출처(L2519 부근)는 아래 re-export 덕분에 그대로 동작한다.
from ui_text import draw_korean_text  # noqa: F401

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# OCR 엔진 임포트
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    easyocr = None
    HAS_EASYOCR = False

try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import fast_alpr  # pip install fast-alpr[onnx-gpu]
    HAS_FAST_ALPR = True
except ImportError:
    fast_alpr = None
    HAS_FAST_ALPR = False


def normalize(text: str) -> str:
    """OCR 혼동 문자 교정 (숫자 자리에서 O→0, I→1 등)"""
    text = text.strip().replace(' ', '').upper()
    table = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '8'}
    result = []
    for i, ch in enumerate(text):
        if i in (2, 3) and '\uAC00' <= ch <= '\uD7A3':
            result.append(ch)
        elif ch in table:
            result.append(table[ch])
        else:
            result.append(ch)
    return ''.join(result)


# ImagePreprocessor → preprocessor.py
# PlateValidator    → validator.py
# PlateDatabase     → db.py
# (refactor: SRP 분리 — 위 "분리된 모듈 (refactor)" import 블록 참조)


class PlateEnginePro:
    """
    상용급 번호판 인식 엔진
    [영상입력] → [YOLO탐지] → [ROI추출] → [10종전처리]
    → [멀티OCR] → [검증/보정] → [DB기록] → [경고알림]
    """

    def __init__(self, config=None):
        self.config = config or PlateEngineConfig()
        self.preprocessor = ImagePreprocessor()
        self.validator = PlateValidator()
        self.db = PlateDatabase()

        model_path = Path(self.config.YOLO_MODEL)
        if not model_path.exists():
            model_path = Path(self.config.YOLO_FALLBACK)
        if not model_path.exists():
            model_path = Path("yolo11n.pt")  # ultralytics 기본
        self.model = YOLO(str(model_path))
        print(f"[엔진] 번호판 YOLO 모델 로드: {model_path}")

        # 2-Stage: 차량 탐지 모델 (yolov8n.pt)
        self.model_vehicle = YOLO('yolov8n.pt')
        print("[엔진] 차량 YOLO 모델 로드: yolov8n.pt")

        # Phase1 Fast loop 전용 번호판 모델 (별도 인스턴스 → 스레드 안전)
        self.model_fast = YOLO(str(model_path))
        print(f"[엔진] Phase1 Fast 번호판 모델 로드: {model_path}")

        self.ocr_engines = {}
        if HAS_PADDLEOCR:
            try:
                self.ocr_engines["paddleocr"] = PaddleOCR(
                    lang="korean",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=True, cpu_threads=4,
                )
            except Exception as e:
                print(f"[엔진] PaddleOCR 초기화 실패: {e}")
        if len(self.ocr_engines) == 0 and HAS_EASYOCR and easyocr is not None:
            try:
                self.ocr_engines["easyocr"] = easyocr.Reader(["ko", "en"], gpu=True)
                print("[엔진] PaddleOCR 미사용 → EasyOCR 폴백 사용")
            except Exception as e:
                print(f"[엔진] EasyOCR 폴백 실패: {e}")
        print(f"[엔진] OCR 엔진: {list(self.ocr_engines.keys())}")

        # CRNN 한글 검증 모델 로드 (CrnnVerifier 책임 클래스)
        # 기존 self._crnn_* 속성은 thin proxy 로 유지 — 외부 호출처 호환.
        _crnn_path = Path(__file__).parent / "plate_ocr_crnn.pth"
        self.crnn_verifier = CrnnVerifier(_crnn_path)
        self._crnn_model = self.crnn_verifier.model
        self._crnn_idx2char = self.crnn_verifier.idx2char
        self._crnn_vocab = self.crnn_verifier.vocab

        self.recent_plates = defaultdict(lambda: {"count": 0, "last_seen": 0, "consecutive": 0})
        self.DUPLICATE_THRESHOLD = 3.0
        # ★ 전역 숫자부 기반 번호판 히스토리 (크로스-트랙 안정화용)
        # {digits: {text: (count, last_frame)}} — 프레임 기반 TTL 관리
        self._global_plate_history = {}  # {digit_pattern: {full_text: (vote_count, last_frame_no)}}
        self._gph_ttl_frames = 45        # 45프레임(≈1.5초@30fps) 미갱신 → 투표 만료 (90→45: 잔상 단축)
        self._gph_max_digits = 50        # 최대 50개 숫자 패턴 보관
        self._gph_cleanup_interval = 30  # 30프레임마다 정리 실행
        self._gph_last_cleanup = 0       # 마지막 정리 시점
        # 속도 최적화: 프레임 스킵 캐시
        self._frame_skip_interval = 3   # N프레임마다 1번 YOLO 실행 (5→3: 컬러 번호판 연속 감지 개선)
        self._frame_counter = 0
        self._cached_results = None     # 이전 프레임 결과 재사용 (None=첫 프레임은 반드시 처리)
        # OCR 스킵 최적화 (FPS 개선)
        # 트랙별: {track_key: {"text": str, "conf": float, "same_count": int,
        #          "frame_since_ocr": int, "last_area": float, "bbox": list}}
        self._ocr_track_cache = {}
        # 연속 N프레임 감지 시 표시 (이미지 슬라이드 영상은 PLATE_CONSECUTIVE_FRAMES=1 로 설정)
        _env = os.environ.get("PLATE_CONSECUTIVE_FRAMES")
        default_consecutive = int(_env) if (_env and _env.isdigit()) else getattr(
            self.config, "CONSECUTIVE_FRAMES_REQUIRED", 1
        )
        self.consecutive_required = default_consecutive
        # 멀티프레임: 최근 5프레임 크롭 저장 (번호판 너비 < 80px 시 사용)
        self._multiframe_buffer = deque(maxlen=PlateEngineConfig.MULTIFRAME_SIZE)
        # 리테스트/벤치마크용 통계
        self.stats = {
            "frames_processed": 0,
            "plates_shown": 0,
            "filtered_by_length": 0,
            "filtered_by_pattern": 0,
            "filtered_by_confidence": 0,
            "confidences": [],
            "multiframe_used": 0,
            "singleframe_used": 0,
        }

    def reset_state(self):
        """내부 캐시/상태 초기화 (테스트용 — 이미지 간 오염 방지)"""
        self.recent_plates = defaultdict(lambda: {"count": 0, "last_seen": 0, "consecutive": 0})
        self._frame_counter = 0
        self._cached_results = None
        self._ocr_track_cache = {}
        self._multiframe_buffer.clear()
        self._global_plate_history = {}
        self._gph_last_cleanup = 0

    def _cleanup_global_plate_history(self):
        """★ 전역 히스토리 TTL 정리 — 오래된 투표 제거 + maxlen 방어.
        30프레임마다 실행, 90프레임 미갱신 엔트리 삭제."""
        if self._frame_counter - self._gph_last_cleanup < self._gph_cleanup_interval:
            return
        self._gph_last_cleanup = self._frame_counter

        # 1단계: TTL 만료 투표 제거
        expired_digits = []
        for digits, variants in self._global_plate_history.items():
            expired_texts = [
                t for t, (count, last_frame) in variants.items()
                if self._frame_counter - last_frame > self._gph_ttl_frames
            ]
            for t in expired_texts:
                del variants[t]
            if not variants:
                expired_digits.append(digits)
        for d in expired_digits:
            del self._global_plate_history[d]

        # 2단계: maxlen 초과 시 가장 오래된 패턴 제거
        if len(self._global_plate_history) > self._gph_max_digits:
            by_recency = sorted(
                self._global_plate_history.keys(),
                key=lambda d: max(
                    (lf for _, lf in self._global_plate_history[d].values()),
                    default=0
                )
            )
            while len(self._global_plate_history) > self._gph_max_digits:
                del self._global_plate_history[by_recency.pop(0)]

    def _composite_multiframe(self, crops):
        """5프레임 크롭을 하나로 합성 (median → 노이즈 감소)."""
        if not crops:
            return None
        target_h, target_w = crops[0].shape[:2]
        resized = [cv2.resize(c, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for c in crops]
        stack = np.stack(resized, axis=0)
        return np.median(stack, axis=0).astype(np.uint8)

    def _make_track_key(self, bbox):
        """bbox 중심점 + 크기 기반 트랙 키 생성.
        30px 양자화 + bbox 높이 그룹으로 같은 위치의 다른 크기 차량 분리."""
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        bh = bbox[3] - bbox[1]
        # 30px 양자화 (50→30: 충돌 확률 감소, 같은 차량 추적 유지)
        qx = cx // 30
        qy = cy // 30
        # bbox 높이를 20px 단위로 양자화 → 같은 위치라도 크기 다른 번호판 분리
        qh = bh // 20
        return (qx, qy, qh)

    def _bbox_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _should_skip_ocr(self, track_key, bbox):
        """OCR 스킵 여부 판단.
        - 새 트랙 → False (무조건 OCR)
        - 3프레임마다 강제 OCR (5→3: 차량 교체 대응 속도 향상)
        - bbox 중심 이동 > 15px → False (같은 키라도 위치 변동 시 즉시 OCR)
        - bbox 면적 20% 변화 → False (즉시 OCR)
        - 같은 결과 5프레임 연속 → True (스킵)
        """
        if track_key not in self._ocr_track_cache:
            return False  # 새 트랙

        cache = self._ocr_track_cache[track_key]

        # 3프레임마다 강제 OCR 재실행 (5→3: Ghost 잔상 지속 시간 단축)
        if cache["frame_since_ocr"] >= 3:
            return False

        # ★ bbox 중심점 이동 거리 체크 — 같은 track_key라도 위치 변동 감지
        last_bbox = cache.get("bbox")
        if last_bbox:
            last_cx = (last_bbox[0] + last_bbox[2]) / 2
            last_cy = (last_bbox[1] + last_bbox[3]) / 2
            cur_cx = (bbox[0] + bbox[2]) / 2
            cur_cy = (bbox[1] + bbox[3]) / 2
            displacement = ((cur_cx - last_cx)**2 + (cur_cy - last_cy)**2) ** 0.5
            if displacement > 15:  # 15px 이상 이동 → 다른 차량 가능성
                return False

        # bbox 면적 20% 변화 시 즉시 OCR
        cur_area = self._bbox_area(bbox)
        last_area = cache.get("last_area", 0)
        if last_area > 0:
            area_change = abs(cur_area - last_area) / last_area
            if area_change >= 0.2:
                return False

        # 같은 결과 5프레임 연속이면 스킵
        if cache["same_count"] >= 5:
            return True

        return False

    @staticmethod
    def _extract_hangul_positions(text):
        """텍스트에서 한글 문자와 위치(숫자 기준 상대 위치) 추출.
        예: '70버6393' → digits='706393', kr_map={2: '버'}
            '경기76바7789' → digits='767789', kr_map={-2: '경', -1: '기', 2: '바'}
        위치 키: 뒤 4자리 숫자 기준 한글의 상대 위치."""
        if not text:
            return {}
        # 숫자만 추출하여 뒤 4자리 위치 기준점 설정
        digits_pos = [(i, c) for i, c in enumerate(text) if c.isdigit()]
        kr_map = {}
        for i, c in enumerate(text):
            if '\uac00' <= c <= '\ud7a3':
                # 한글 위치를 원본 인덱스로 저장
                kr_map[i] = c
        return kr_map

    def _update_ocr_cache(self, track_key, bbox, text, conf, did_ocr):
        """OCR 캐시 업데이트 + 한글 투표 카운터 누적 + 전체 텍스트 다수결 투표"""
        cur_area = self._bbox_area(bbox)
        if track_key not in self._ocr_track_cache:
            self._ocr_track_cache[track_key] = {
                "text": text, "conf": conf, "same_count": 1,
                "frame_since_ocr": 0, "last_area": cur_area, "bbox": bbox,
                "best_kr_text": "", "best_kr_conf": 0.0,
                "kr_votes": {},  # {position: {char: count}}
                "vote_count": 0,
                "frames_absent": 0,  # ★ 고스트 방지: 미감지 프레임 카운터
                "text_votes": {},    # ★ 전체 텍스트 다수결: {text: count}
                "text_confs": {},    # ★ 텍스트별 conf 합산: {text: [conf1, ...]}
            }
            if text and re.search(r'[가-힣]', text):
                self._ocr_track_cache[track_key]["best_kr_text"] = text
                self._ocr_track_cache[track_key]["best_kr_conf"] = conf
                # 첫 투표 등록
                for pos, ch in self._extract_hangul_positions(text).items():
                    self._ocr_track_cache[track_key]["kr_votes"][pos] = {ch: 1}
                self._ocr_track_cache[track_key]["vote_count"] = 1
            # 전체 텍스트 투표 등록
            if text:
                self._ocr_track_cache[track_key]["text_votes"][text] = 1
                self._ocr_track_cache[track_key]["text_confs"][text] = [conf]
            return

        cache = self._ocr_track_cache[track_key]
        if did_ocr:
            if text == cache["text"]:
                cache["same_count"] += 1
            else:
                cache["same_count"] = 1
            cache["text"] = text
            cache["conf"] = conf
            cache["frame_since_ocr"] = 0
            if text and re.search(r'[가-힣]', text):
                if conf >= cache.get("best_kr_conf", 0):
                    cache["best_kr_text"] = text
                    cache["best_kr_conf"] = conf
                # 한글 투표 누적
                for pos, ch in self._extract_hangul_positions(text).items():
                    if pos not in cache["kr_votes"]:
                        cache["kr_votes"][pos] = {}
                    cache["kr_votes"][pos][ch] = cache["kr_votes"][pos].get(ch, 0) + 1
                cache["vote_count"] = cache.get("vote_count", 0) + 1
            # ★ 전체 텍스트 다수결 누적
            if text:
                if "text_votes" not in cache:
                    cache["text_votes"] = {}
                    cache["text_confs"] = {}
                cache["text_votes"][text] = cache["text_votes"].get(text, 0) + 1
                if text not in cache.get("text_confs", {}):
                    cache["text_confs"][text] = []
                cache["text_confs"][text].append(conf)
                # ★ 전역 히스토리 즉시 누적 (프레임 번호 기반 TTL 관리)
                _digits = re.sub(r'[^0-9]', '', text)
                if len(_digits) >= 4:
                    if _digits not in self._global_plate_history:
                        self._global_plate_history[_digits] = {}
                    _prev = self._global_plate_history[_digits].get(text, (0, 0))
                    self._global_plate_history[_digits][text] = (
                        _prev[0] + 1, self._frame_counter  # (누적 투표수, 마지막 감지 프레임)
                    )
        else:
            cache["frame_since_ocr"] += 1
        cache["last_area"] = cur_area
        cache["bbox"] = bbox

    def _recover_hangul_from_cache(self, track_key, text, conf):
        """한글 캐시 복원 + 멀티프레임 한글 투표 적용.
        1) 한글 없는 결과 → 캐시의 한글 포함 결과로 복원
        2) 한글 있는 결과 → 5프레임 이상 누적 시 다수결로 한글 교체"""
        cache = self._ocr_track_cache.get(track_key)
        if not cache:
            return text, conf

        has_kr = bool(text and re.search(r'[가-힣]', text))

        # 한글 없음 → 캐시 복원 (기존 로직)
        if not has_kr:
            kr_text = cache.get("best_kr_text", "")
            if kr_text:
                # 복원된 텍스트에도 투표 적용
                text, conf = kr_text, cache.get("best_kr_conf", conf)
                has_kr = True
            else:
                return text, conf

        # 한글 투표 적용: 5프레임 이상 누적 시 다수결로 한글 교체
        vote_count = cache.get("vote_count", 0)
        kr_votes = cache.get("kr_votes", {})
        if vote_count < 5 or not kr_votes:
            return text, conf

        # 현재 텍스트의 한글 위치와 투표 결과 매칭
        result = list(text)
        changed = False
        for pos, ch in self._extract_hangul_positions(text).items():
            if pos in kr_votes:
                votes = kr_votes[pos]
                winner = max(votes, key=votes.get)
                if winner != ch and votes[winner] >= 3:
                    result[pos] = winner
                    changed = True

        if changed:
            return "".join(result), conf
        return text, conf

    def _stabilize_track_text(self, track_key, text, conf):
        """★ 트래킹 기반 번호판 안정화: 전역 히스토리 다수결.
        조건: 승자가 최소 3회 이상 + 현재 텍스트보다 2회 이상 많을 때만 교체.
        조기 교체 방지 — 충분한 데이터 누적 후에만 다수결 적용."""
        if not text:
            return text, conf

        digits = re.sub(r'[^0-9]', '', text)
        if len(digits) < 4:
            return text, conf

        # ★ 전역 히스토리 참조 (최소 3표 + 2표차 + TTL 유효한 투표만)
        history = self._global_plate_history.get(digits, {})
        if history and len(history) >= 2:
            # TTL 유효한 투표만 필터링
            alive = {t: cnt for t, (cnt, lf) in history.items()
                     if self._frame_counter - lf <= self._gph_ttl_frames}
            if len(alive) >= 2:
                winner = max(alive, key=alive.get)
                winner_count = alive[winner]
                current_count = alive.get(text, 0)
                if (winner != text
                        and winner_count >= 3
                        and winner_count >= current_count + 2):
                    print(f"[STABILIZE] 전역 다수결: {text}({current_count}회) → "
                          f"{winner}({winner_count}회)", flush=True)
                    return winner, conf

        return text, conf

    def _merge_partial_plates(self, text, conf):
        """★ OCR 부분 결과 병합: 같은 뒤4자리를 가진 서로 다른 부분 인식 결과를 합쳐서 전체 복원.
        예: "85아6374"(100% x6) + "51우6374"(100% x99) → "8519우6374"
        게임/시뮬레이션 번호판에서 OCR이 앞자리를 분할 인식하는 경우 대응."""
        if not text or len(text) < 5:
            return text, conf

        # 현재 텍스트에서 뒤4자리 숫자 추출
        suffix_m = re.search(r'(\d{4})$', text)
        if not suffix_m:
            return text, conf
        suffix4 = suffix_m.group(1)

        # 현재 텍스트에서 한글과 앞자리 분리
        cur_m = re.match(r'^(\d{1,4})([가-힣])(\d{4})$', text)
        if not cur_m:
            return text, conf

        cur_prefix = cur_m.group(1)  # 예: "51" 또는 "85"
        cur_hangul = cur_m.group(2)  # 예: "우" 또는 "아"
        cur_suffix = cur_m.group(3)  # 예: "6374"

        # recent_plates에서 같은 뒤4자리를 가진 다른 결과 찾기
        _now = time.time()
        candidates = []
        for plate_key, plate_data in self.recent_plates.items():
            if plate_key == text:
                continue
            if (_now - plate_data.get("last_seen", 0)) > 10.0:
                continue  # 10초 이상 오래된 결과 무시
            other_m = re.match(r'^(\d{1,4})([가-힣])(\d{4})$', plate_key)
            if not other_m:
                continue
            if other_m.group(3) != cur_suffix:
                continue  # 뒤4자리 다르면 스킵

            other_prefix = other_m.group(1)
            other_hangul = other_m.group(2)
            other_count = plate_data.get("count", 0)
            candidates.append((other_prefix, other_hangul, other_count, plate_key))

        if not candidates:
            return text, conf

        # ★ 병합 시도: 서로 다른 앞자리 숫자를 합치면 3~4자리가 되는지 확인
        for other_prefix, other_hangul, other_count, other_plate in candidates:
            # 앞자리가 서로 다르고, 합쳤을 때 3~4자리가 되는 경우
            if cur_prefix == other_prefix:
                continue

            # 합산 시도: 긴쪽 앞자리를 기준으로 검증
            # 예: "85" + "51" → "8519" 아니면 "5185"
            merged1 = cur_prefix + other_prefix  # "8551" 또는 "5185"
            merged2 = other_prefix + cur_prefix  # "5185" 또는 "8551"

            for merged_prefix in [merged1, merged2]:
                if len(merged_prefix) < 3 or len(merged_prefix) > 4:
                    continue

                # 더 자주 인식된 쪽의 한글 사용
                cur_count = self.recent_plates.get(text, {}).get("count", 0)
                if other_count > cur_count:
                    merged_hangul = other_hangul
                else:
                    merged_hangul = cur_hangul

                merged_plate = f"{merged_prefix}{merged_hangul}{cur_suffix}"

                # 패턴 유효성 검증
                if re.match(r'^\d{3,4}[가-힣]\d{4}$', merged_plate):
                    print(f"[MERGE-PARTIAL] '{text}' + '{other_plate}' → "
                          f"'{merged_plate}' (prefix={merged_prefix}, hangul={merged_hangul})",
                          flush=True)
                    return merged_plate, max(conf, 0.85)

        return text, conf

    def _get_cached_ocr(self, track_key):
        """캐시된 OCR 결과 반환"""
        cache = self._ocr_track_cache.get(track_key)
        if cache:
            return cache["text"], cache["conf"]
        return "", 0.0

    def _ocr_plate_roi(self, roi, use_multiframe=False):
        """번호판 ROI에서 OCR 수행 → (best_text, best_conf) 반환"""
        roi_h, roi_w = roi.shape[:2]
        roi_for_ocr = roi

        if use_multiframe and roi_w < PlateEngineConfig.MULTIFRAME_PLATE_WIDTH_THRESHOLD:
            self._multiframe_buffer.append(roi.copy())
            if len(self._multiframe_buffer) >= PlateEngineConfig.MULTIFRAME_SIZE:
                crops = list(self._multiframe_buffer)
                roi_for_ocr = self._composite_multiframe(crops)
                self._multiframe_buffer.clear()
                self.stats["multiframe_used"] = self.stats.get("multiframe_used", 0) + 1
            else:
                return "", 0.0  # 버퍼 채울 때까지 스킵
        else:
            if use_multiframe:
                self.stats["singleframe_used"] = self.stats.get("singleframe_used", 0) + 1
            target_w = 300  # PaddleOCR 입력 크기
            if roi_w < target_w:
                scale = target_w / roi_w
            else:
                scale = 1.0
            if scale > 1.0:
                roi_for_ocr = cv2.resize(
                    roi, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_LANCZOS4 if scale > 3.0 else cv2.INTER_CUBIC
                )
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                roi_for_ocr = cv2.filter2D(roi_for_ocr, -1, kernel)
            else:
                roi_for_ocr = roi

        from collections import Counter
        all_candidates = []

        # ★ ROI 색상 감지 (2줄 감지보다 선행): 컬러판은 2줄 threshold 완화
        _is_color_plate = False
        self._last_color_plate = False
        try:
            _hsv_roi = cv2.cvtColor(roi_for_ocr, cv2.COLOR_BGR2HSV)
            _h, _s, _v = cv2.split(_hsv_roi)
            _yellow_mask = cv2.inRange(_hsv_roi, np.array([15, 40, 100]), np.array([35, 255, 255]))
            _yellow_ratio = np.count_nonzero(_yellow_mask) / max(_yellow_mask.size, 1)
            _green_mask = cv2.inRange(_hsv_roi, np.array([35, 40, 40]), np.array([90, 255, 255]))
            _green_ratio = np.count_nonzero(_green_mask) / max(_green_mask.size, 1)
            if _yellow_ratio > 0.15 or _green_ratio > 0.15:
                _is_color_plate = True
                self._last_color_plate = True
                _color_type = "노란" if _yellow_ratio > _green_ratio else "초록"
                print(f"[COLOR-DETECT] {_color_type} 번호판 감지 (yellow={_yellow_ratio:.0%}, green={_green_ratio:.0%}) → 컬러 전처리 강제", flush=True)
        except Exception:
            pass

        # 구형 두 줄 번호판 감지
        # ★ 컬러판(노란/초록)은 대부분 2줄 → threshold 완화 (0.45→0.30)
        extra_crops = []
        roi_nosharp = None
        # ★ 컬러판 2줄 threshold 대폭 완화: 원거리 녹색판(36다7117, 02누2754)도 2줄 분할
        _2line_ratio = 0.20 if _is_color_plate else 0.45
        if roi_h > roi_w * _2line_ratio:
            top_crop = roi_for_ocr[:int(roi_for_ocr.shape[0] * 0.5), :]
            bot_crop = roi_for_ocr[int(roi_for_ocr.shape[0] * 0.4):, :]
            extra_crops = [("top", top_crop), ("bot", bot_crop)]
            if roi_w < 150:
                _nosharp_scale = 5 if roi_w < 80 else 3
                roi_nosharp = cv2.resize(roi, None, fx=_nosharp_scale, fy=_nosharp_scale, interpolation=cv2.INTER_CUBIC)

        # ★ 2줄 번호판 선행 OCR: 전처리 루프보다 먼저 분할 OCR 실행 (속도 우선)
        # 경기76바7789, 36다7117 등 2줄 번호판은 상+하 분할이 가장 효과적
        _2line_early_found = False
        if extra_crops:
            _ec_top_texts, _ec_bot_texts = [], []
            _ec_top_confs, _ec_bot_confs = [], []
            for crop_name, crop_img in extra_crops:
                # ★ 원거리 top crop 강화: 전처리 버전 추가 (green_plate, clahe)
                _crop_variants = [crop_img]
                if crop_name == "top" and _is_color_plate:
                    for _prep_name in ['clahe', 'green_plate', 'sharpen']:
                        _prep_func = getattr(self.preprocessor, _prep_name, None)
                        if _prep_func:
                            try:
                                _crop_variants.append(_prep_func(crop_img.copy()))
                            except Exception:
                                pass
                for _variant in _crop_variants:
                    for eng_name, eng in self.ocr_engines.items():
                        t, c = self._run_ocr(eng_name, eng, _variant)
                        if t and c > 0.2:
                            cleaned_t = self.validator.clean_ocr_text(t)
                            if crop_name == "top":
                                _ec_top_texts.append(cleaned_t); _ec_top_confs.append(c)
                            else:
                                _ec_bot_texts.append(cleaned_t); _ec_bot_confs.append(c)
            for tt in (_ec_top_texts or [""]):
                for bt in (_ec_bot_texts or [""]):
                    combined = (tt + bt).strip()
                    norm = self.validator._normalize_for_validation(combined)
                    if self.validator.is_valid_length(norm):
                        is_v, final = self.validator.validate(norm)
                        if is_v:
                            avg_c = float(np.mean((_ec_top_confs or [0.3]) + (_ec_bot_confs or [0.3])))
                            weight = 6 if re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', final) else 2
                            for _ in range(weight):
                                all_candidates.append((final, avg_c))
                            _2line_early_found = True
                            print(f"[2LINE-EARLY] 선행 분할 OCR 성공: {final} (conf={avg_c:.2f})", flush=True)

        early_exit = False  # original 고신뢰 시 나머지 전처리 스킵
        _base_methods = {"original", "clahe", "sharpen"}  # 기본 전처리 (흰 번호판용)
        # ★ 컬러 번호판이면 컬러 전처리도 기본에 포함
        _color_methods = {"invert_color", "green_plate", "yellow_plate", "color_plate_clahe",
                          "backlight_adaptive", "brightness_normalize"}
        _skip_color = False  # 기본 3종에서 충분한 후보 → 컬러 전처리 불필요
        _ocr_start_time = time.time()  # ★ 시간 기반 강제 종료용
        # ★ 2줄 선행 OCR 성공 시 타임아웃 단축 (이미 결과 있으므로)
        if _2line_early_found:
            _ocr_time_limit = 0.3  # 빠른 보완용
        else:
            _ocr_time_limit = 0.5  # ★ 컬러/일반 동일 500ms (속도 우선)
        _method_count = 0  # ★ 처리한 전처리 수
        for method in self.config.PREPROCESS_METHODS:
            # ★ 시간 기반 강제 종료
            if time.time() - _ocr_start_time > _ocr_time_limit and _method_count >= 2:
                print(f"[OCR-TIMEOUT] {_method_count}종 처리 후 {(time.time()-_ocr_start_time)*1000:.0f}ms → 시간 초과 종료", flush=True)
                break
            # 기본 전처리에서 충분한 후보가 나오면 컬러 전처리 스킵
            # ★ 컬러판도 고신뢰(conf>=0.85) 후보 있으면 조기 종료 허용
            if early_exit and method not in _base_methods:
                if not _is_color_plate:
                    break
                # 컬러판: 고신뢰 후보가 있으면 추가 전처리 스킵
                _best_conf = max((c for _, c in all_candidates), default=0)
                if _best_conf >= 0.85 or len(all_candidates) >= 3:
                    print(f"[COLOR-EARLY-EXIT] 컬러판 고신뢰 조기종료 (best_conf={_best_conf:.2f}, candidates={len(all_candidates)})", flush=True)
                    break
            # ★ 컬러 번호판: 컬러 전처리 절대 스킵 안 함
            if _is_color_plate and method in _color_methods:
                pass  # 강제 실행
            elif not _is_color_plate:
                # 기본 3종 완료 후 후보가 1개 이상이면 컬러 전처리 건너뛰기
                if method not in _base_methods and not _skip_color:
                    if len(all_candidates) >= 1:
                        _skip_color = True
                if _skip_color and method not in _base_methods:
                    continue
            _method_count += 1
            try:
                if method == "original":
                    processed = roi_for_ocr.copy()
                else:
                    proc_func = getattr(self.preprocessor, method, None)
                    if proc_func is None:
                        continue
                    processed = proc_func(roi_for_ocr.copy())

                for engine_name, engine in self.ocr_engines.items():
                    text, ocr_conf = self._run_ocr(engine_name, engine, processed)
                    print(f"[OCR-RAW] text={text!r} conf={ocr_conf:.2f}", flush=True)
                    if not text or ocr_conf < 0.10:
                        continue
                    cleaned = self.validator.clean_ocr_text(text)
                    if not self.validator.is_valid_length(cleaned):
                        # ★ 숫자 4자리만 읽힌 경우: CRNN으로 한글+앞번호 복원 시도
                        # "7117" → CRNN "36다7117" → 복원 가능
                        if (re.fullmatch(r'\d{4}', cleaned)
                                and ocr_conf >= 0.50):
                            _digit_recovered = False
                            # 방법 1: CRNN 직접 복원
                            if hasattr(self, '_crnn_model') and self._crnn_model is not None:
                                _crnn_full = self._crnn_read_plate(roi)
                                if (_crnn_full
                                        and _crnn_full.endswith(cleaned)
                                        and len(_crnn_full) > len(cleaned)
                                        and self.validator.is_valid_length(_crnn_full)):
                                    is_valid_c, final_c = self.validator.validate(_crnn_full)
                                    if is_valid_c:
                                        print(f"[DIGIT-CRNN] 숫자만 → CRNN 복원: {cleaned} → {final_c}", flush=True)
                                        all_candidates.append((final_c, ocr_conf * 0.85))
                                        _digit_recovered = True
                                elif (_crnn_full
                                        and cleaned in _crnn_full
                                        and re.search(r'[가-힣]', _crnn_full)):
                                    # ★ CRNN이 한글 포함 + 숫자가 부분 일치 → 복원 시도
                                    # "36다71170" contains "7117" + 한글 "다" → 앞부분 추출
                                    _crnn_clean = self.validator.clean_ocr_text(_crnn_full)
                                    if self.validator.is_valid_length(_crnn_clean):
                                        is_valid_c, final_c = self.validator.validate(_crnn_clean)
                                        if is_valid_c:
                                            print(f"[DIGIT-CRNN-CONTAINS] CRNN 부분매칭 복원: {cleaned} → {final_c}", flush=True)
                                            all_candidates.append((final_c, ocr_conf * 0.80))
                                            _digit_recovered = True
                                if not _digit_recovered and _crnn_full:
                                    print(f"[DIGIT-CRNN-FAIL] CRNN={_crnn_full!r} ← 숫자={cleaned} (불일치)", flush=True)
                            # ★ 방법 1-1: 4자리 + CRNN 실패 → 즉석 상단 크롭 OCR로 지역명 복원
                            # (원거리에서 컬러 감지 안 될 수 있으므로 항상 시도)
                            if not _digit_recovered and roi_for_ocr is not None:
                                _top_h = int(roi_for_ocr.shape[0] * 0.50)
                                if _top_h > 10:
                                    _top_crop = roi_for_ocr[:_top_h, :]
                                    # 상단 크롭에 전처리 적용
                                    _top_variants = [_top_crop]
                                    for _pn in ['clahe', 'sharpen', 'green_plate']:
                                        _pf = getattr(self.preprocessor, _pn, None)
                                        if _pf:
                                            try: _top_variants.append(_pf(_top_crop.copy()))
                                            except: pass
                                    _top_found_texts = []
                                    for _tv in _top_variants:
                                        for _en, _eg in self.ocr_engines.items():
                                            _tt, _tc = self._run_ocr(_en, _eg, _tv)
                                            if _tt and _tc > 0.20:
                                                _top_clean = re.sub(r'[^가-힣0-9a-zA-Z]', '', _tt)
                                                if _top_clean:
                                                    _top_found_texts.append(_top_clean)
                                                    _combined = _top_clean + cleaned
                                                    _norm = self.validator._normalize_for_validation(_combined)
                                                    if self.validator.is_valid_length(_norm):
                                                        _is_v, _final = self.validator.validate(_norm)
                                                        if _is_v:
                                                            print(f"[DIGIT-TOP-CROP] 상단크롭 복원: {_top_clean}+{cleaned} → {_final}", flush=True)
                                                            all_candidates.append((_final, ocr_conf * 0.80))
                                                            _digit_recovered = True
                                                            break
                                        if _digit_recovered:
                                            break
                                    if not _digit_recovered and _top_found_texts:
                                        print(f"[DIGIT-TOP-CROP-FAIL] 상단크롭 읽음: {_top_found_texts} + {cleaned} → 유효 번호판 없음", flush=True)
                            # ★ 방법 1-2: 2줄 상단 텍스트 + 하단 숫자 결합
                            # 2LINE-EARLY에서 top 텍스트가 있으면 bottom 숫자와 결합 시도
                            if not _digit_recovered and extra_crops and _ec_top_texts:
                                for _top_t in _ec_top_texts:
                                    _combined = _top_t + cleaned
                                    _norm = self.validator._normalize_for_validation(_combined)
                                    if self.validator.is_valid_length(_norm):
                                        _is_v, _final = self.validator.validate(_norm)
                                        if _is_v:
                                            print(f"[DIGIT-2LINE-MERGE] 상단+하단 결합: {_top_t}+{cleaned} → {_final}", flush=True)
                                            all_candidates.append((_final, ocr_conf * 0.80))
                                            _digit_recovered = True
                                            break
                            # 방법 2: 같은 트랙 캐시에서 복원 (4자리도 허용 - 같은 차량)
                            # ★ 같은 bbox 위치(같은 차량)의 이전 인식 결과로 복원
                            if not _digit_recovered:
                                for _tk, _tc in self._ocr_track_cache.items():
                                    _ct = _tc.get("text", "")
                                    if (_ct and _ct.endswith(cleaned)
                                            and len(_ct) > len(cleaned)
                                            and re.search(r'[가-힣]', _ct)):
                                        print(f"[DIGIT-TRACK] 같은 트랙 캐시 복원: {cleaned} → {_ct}", flush=True)
                                        all_candidates.append((_ct, ocr_conf * 0.80))
                                        _digit_recovered = True
                                        break
                            # 방법 3: 글로벌 히스토리 복원 (5자리 이상만 - 교차 오염 방지)
                            if not _digit_recovered and len(cleaned) >= 5:
                                for _known in list(self.recent_plates.keys()):
                                    if _known.endswith(cleaned) and len(_known) > len(cleaned):
                                        print(f"[DIGIT-HIST] 숫자만 → 히스토리 복원: {cleaned} → {_known}", flush=True)
                                        all_candidates.append((_known, ocr_conf * 0.75))
                                        _digit_recovered = True
                                        break
                            if not _digit_recovered and len(cleaned) >= 5:
                                for _tk, _tc in self._ocr_track_cache.items():
                                    _ct = _tc.get("text", "")
                                    if _ct.endswith(cleaned) and len(_ct) > len(cleaned):
                                        print(f"[DIGIT-CACHE] 숫자만 → 캐시 복원: {cleaned} → {_ct}", flush=True)
                                        all_candidates.append((_ct, ocr_conf * 0.75))
                                        _digit_recovered = True
                                        break
                            if _digit_recovered:
                                continue
                        print(f"[OCR-FILTERED-LEN] cleaned={cleaned!r}", flush=True)
                        self.stats["filtered_by_length"] += 1
                        continue
                    print(f"[OCR-DBG] raw={text} cleaned={cleaned} conf={ocr_conf:.2f}", flush=True)
                    is_valid, final_text = self.validator.validate(cleaned)
                    if not is_valid:
                        self.stats["filtered_by_pattern"] += 1
                        continue
                    all_candidates.append((final_text, ocr_conf))
                    # ★ 영상 모드 고속화: 유효 번호판 + conf ≥ 0.6 → 즉시 스킵
                    # (original뿐 아니라 모든 전처리에서 고신뢰 시 종료)
                    if ocr_conf >= 0.6:
                        early_exit = True
                    # ★ 후보 3개 이상 모이면 추가 전처리 불필요
                    if len(all_candidates) >= 3:
                        early_exit = True
            except Exception:
                continue

        if extra_crops:
            top_texts, bot_texts = [], []
            top_confs, bot_confs = [], []
            for crop_name, crop_img in extra_crops:
                for eng_name, eng in self.ocr_engines.items():
                    t, c = self._run_ocr(eng_name, eng, crop_img)
                    if t and c > 0.2:
                        cleaned_t = self.validator.clean_ocr_text(t)
                        if crop_name == "top":
                            top_texts.append(cleaned_t); top_confs.append(c)
                        else:
                            bot_texts.append(cleaned_t); bot_confs.append(c)
            for tt in (top_texts or [""]):
                for bt in (bot_texts or [""]):
                    combined = (tt + bt).strip()
                    norm = self.validator._normalize_for_validation(combined)
                    if self.validator.is_valid_length(norm):
                        is_v, final = self.validator.validate(norm)
                        if is_v:
                            avg_c = float(np.mean((top_confs or [0.3]) + (bot_confs or [0.3])))
                            weight = 6 if re.match(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$', final) else 2
                            for _ in range(weight):
                                all_candidates.append((final, avg_c))

        # 2줄 번호판 보완: 샤프닝 없는 업스케일로 engine.predict 시도
        if roi_nosharp is not None and not all_candidates:
            for eng_name, eng in self.ocr_engines.items():
                try:
                    for res in eng.predict(roi_nosharp):
                        texts = res.get('rec_texts', [])
                        scores = res.get('rec_scores', [])
                        if texts:
                            text = "".join(texts)
                            conf = sum(scores) / len(scores) if scores else 0.0
                            cleaned = self.validator.clean_ocr_text(text)
                            if self.validator.is_valid_length(cleaned):
                                is_v, final = self.validator.validate(cleaned)
                                if is_v:
                                    all_candidates.append((final, conf))
                except Exception:
                    pass

        # ★ 컬러판 강제 2줄 시도: all_candidates 없고 컬러판이면 ROI를 강제 2줄 분할
        # 원거리 녹색판(36다7117, 02누2754)에서 2LINE 미발동 시 마지막 시도
        if not all_candidates and _is_color_plate and not extra_crops:
            _force_top = roi_for_ocr[:int(roi_for_ocr.shape[0] * 0.55), :]
            _force_bot = roi_for_ocr[int(roi_for_ocr.shape[0] * 0.40):, :]
            _force_top_texts, _force_bot_texts = [], []
            _force_top_confs, _force_bot_confs = [], []
            for _fn, _fc in [("top", _force_top), ("bot", _force_bot)]:
                # 전처리 버전도 포함
                _fvariants = [_fc]
                if _fn == "top":
                    for _pn in ['clahe', 'green_plate']:
                        _pf = getattr(self.preprocessor, _pn, None)
                        if _pf:
                            try:
                                _fvariants.append(_pf(_fc.copy()))
                            except Exception:
                                pass
                for _fv in _fvariants:
                    for eng_name, eng in self.ocr_engines.items():
                        t, c = self._run_ocr(eng_name, eng, _fv)
                        if t and c > 0.2:
                            ct = self.validator.clean_ocr_text(t)
                            if _fn == "top":
                                _force_top_texts.append(ct); _force_top_confs.append(c)
                            else:
                                _force_bot_texts.append(ct); _force_bot_confs.append(c)
            # 상단+하단 조합 시도
            for _ft in (_force_top_texts or [""]):
                for _fb in (_force_bot_texts or [""]):
                    _comb = (_ft + _fb).strip()
                    _norm = self.validator._normalize_for_validation(_comb)
                    if self.validator.is_valid_length(_norm):
                        _iv, _ifinal = self.validator.validate(_norm)
                        if _iv:
                            _ac = float(np.mean((_force_top_confs or [0.3]) + (_force_bot_confs or [0.3])))
                            all_candidates.append((_ifinal, _ac))
                            print(f"[FORCE-2LINE] 강제 분할 성공: {_ifinal} (conf={_ac:.2f})", flush=True)

        best_text = ""
        best_conf = 0.0
        # ★ 3DIGIT-MAP 등에서 사용할 원시 숫자 패턴 저장
        self._last_raw_digits = []
        for t, c in all_candidates:
            _d = re.sub(r'[^0-9]', '', t)
            if len(_d) >= 5:
                self._last_raw_digits.append(_d)
        if all_candidates:
            # ★ conf 가중합 투표: 고신뢰 결과에 높은 가중치 (한글 혼동 방지)
            # 소↔조, 버↔조, 무↔오↔보 등 유효 한글 간 혼동 시 conf가 높은 쪽이 승리
            weighted_scores = {}
            for t, c in all_candidates:
                weighted_scores[t] = weighted_scores.get(t, 0.0) + max(c, 0.1)
            best_text = max(weighted_scores, key=weighted_scores.get)
            confs = [c for t, c in all_candidates if t == best_text]
            best_conf = sum(confs) / len(confs)

            # ★ 노란판 신형패턴 보정: 노란판은 영업용/구형이므로 신형 결과 의심
            # 노란판에서 \d{2}[가-힣]\d{4} (신형)이 나오면, 모든 후보를 분석하여
            # 영업용 번호판 복원 시도 (OCR이 지역명을 숫자로 오인식하는 경우)
            # ★ 2LINE-EARLY 결과는 스킵: 분할 OCR의 상단 행 오독 → 잘못된 지역 매핑 방지
            # (CRNN-2LINE이 caller에서 더 정확하게 복원함)
            if (getattr(self, '_last_color_plate', False)
                    and re.fullmatch(r'\d{2}[가-힣]\d{4}', best_text)
                    and not _2line_early_found):
                from collections import Counter
                # ★ 핵심: 각 후보 텍스트에서 한글 앞 2자리 = mid digits 직접 추출
                # "86오8118" → 한글 "오" 앞 2자리 "86" = mid
                # "18바1818" → 한글 "바" 앞 2자리 "18" = mid
                _mid_cnt = Counter()
                _suffix_cnt = Counter()
                _hangul_cnt = Counter()
                for t, c in all_candidates:
                    _m = re.match(r'^(\d+)([가-힣])(\d{4})$', t)
                    if _m:
                        _digits_before = _m.group(1)
                        _h = _m.group(2)
                        _sfx = _m.group(3)
                        _suffix_cnt[_sfx] += c
                        _hangul_cnt[_h] += c
                        if len(_digits_before) >= 4:
                            # ★ 4자리 이상: region+mid 모두 캡처 → mid 신뢰도 높음 (2배 가중)
                            _mid_cnt[_digits_before[-2:]] += c * 2.0
                        elif len(_digits_before) >= 2:
                            # 2자리: region/mid 구분 불가 → 기본 가중치
                            _mid_cnt[_digits_before[-2:]] += c
                if _mid_cnt and _suffix_cnt:
                    _top_mid = _mid_cnt.most_common(1)[0][0]
                    _top_suffix = _suffix_cnt.most_common(1)[0][0]
                    _top_hangul = _hangul_cnt.most_common(1)[0][0] if _hangul_cnt else ''
                    # mid digits → 지역명 매핑 시도
                    # 노란판에서 mid=86 → 실제 충남86자8118이므로 region 추정 필요
                    _y_region_map = {
                        '13': '충남', '15': '충남', '16': '충남', '56': '충남',
                        '53': '충남', '96': '충남', '86': '충남', '58': '충남', '98': '충남',
                        '10': '충북', '50': '충북',
                        '14': '전남', '34': '전남', '11': '전북', '31': '전북',
                        '21': '경기', '12': '경북', '22': '경남',
                        '17': '인천', '37': '인천',
                    }
                    # ★ 지역명 추정: 후보에서 한글 앞 2자리 이전 숫자가 있으면 그것이 region 코드
                    _region_code_cnt = Counter()
                    for t, c in all_candidates:
                        _m = re.match(r'^(\d+)([가-힣])(\d{4})$', t)
                        if _m and len(_m.group(1)) >= 4:
                            # 4자리 이상 → 앞 2자리=region, 뒤 2자리=mid
                            _rc = _m.group(1)[:2]
                            if _rc in _y_region_map:
                                _region_code_cnt[_rc] += c
                    if _region_code_cnt:
                        _region_code = _region_code_cnt.most_common(1)[0][0]
                        _y_region = _y_region_map[_region_code]
                    elif _top_mid in _y_region_map:
                        # 폴백: mid 자체가 region map에 있으면 사용
                        _y_region = _y_region_map[_top_mid]
                    else:
                        _y_region = ''
                    if _y_region and _top_hangul:
                        _yellow_restored = _y_region + _top_mid + _top_hangul + _top_suffix
                        _yv, _yf = self.validator.validate(_yellow_restored)
                        if _yv:
                            print(f"[YELLOW-FIX] 노란판 영업용 복원: {best_text} → {_yf} "
                                  f"(region={_y_region}, mid={_top_mid}, "
                                  f"hangul={_top_hangul}, suffix={_top_suffix})", flush=True)
                            best_text = _yf
                            confs = [c for t, c in all_candidates]
                            best_conf = sum(confs) / len(confs)

            # ★ 노란판 한글 교정: OCR이 자주 혼동하는 한글 쌍 교정
            # 아(ㅇ+ㅏ) ↔ 자(ㅈ+ㅏ): 저해상도에서 ㅈ 상단 획이 ㅇ으로 보임
            if getattr(self, '_last_color_plate', False):
                _yellow_hangul_fix = {'아': '자', '오': '자', '이': '자'}
                _yh_done = False
                # 1) 상용 패턴: "충남86아6118" → "충남86자6118"
                _m_yh2 = re.match(r'^([가-힣]{2,3}\d{2})([가-힣])(\d{4})$', best_text)
                if _m_yh2 and _m_yh2.group(2) in _yellow_hangul_fix:
                    _new_h = _yellow_hangul_fix[_m_yh2.group(2)]
                    _yh_fixed = _m_yh2.group(1) + _new_h + _m_yh2.group(3)
                    _yhv, _yhf = self.validator.validate(_yh_fixed)
                    if _yhv:
                        print(f"[YELLOW-HANGUL] 상용 한글 교정: {best_text} → {_yhf} "
                              f"({_m_yh2.group(2)}→{_new_h})", flush=True)
                        best_text = _yhf
                        _yh_done = True
                # 2) 신형 패턴: "86아6118" → "86자6118"
                if not _yh_done:
                    _m_yh3 = re.match(r'^(\d{2})([가-힣])(\d{4})$', best_text)
                    if _m_yh3 and _m_yh3.group(2) in _yellow_hangul_fix:
                        _new_h = _yellow_hangul_fix[_m_yh3.group(2)]
                        _yh_fixed = _m_yh3.group(1) + _new_h + _m_yh3.group(3)
                        _yhv, _yhf = self.validator.validate(_yh_fixed)
                        if _yhv:
                            print(f"[YELLOW-HANGUL] 신형 한글 교정: {best_text} → {_yhf} "
                                  f"({_m_yh3.group(2)}→{_new_h})", flush=True)
                            best_text = _yhf

            # ★ 8자리 오류 교정: OCR이 잘못된 문자를 추가하는 경우
            # "전41나3234" (한글+2자리+한글+4자리=8자) → "14나3234" (7자) 시도
            if re.fullmatch(r'[가-힣]\d{2}[가-힣]\d{4}', best_text):
                _trimmed = best_text[1:]  # 앞 한글 제거
                _tv, _tf = self.validator.validate(_trimmed)
                if _tv:
                    print(f"[8CHAR-FIX] 앞 한글 제거: {best_text} → {_tf}", flush=True)
                    best_text = _tf
            # ★ 3~4자리+한글+4자리 (8~9자)는 유효한 번호판이므로 절대 삭제하지 않음
            # "851우6374", "8519우6374" 등 게임/시뮬레이션 번호판 보호
            # 기존 "022누2754"→"02누2754" 변환은 폐기 (3자리 앞번호도 유효함)

        return best_text, best_conf

    @staticmethod
    def _extract_last4(plate_text):
        """번호판 텍스트에서 뒤 4자리 숫자 추출 (중복 제거용)"""
        digits = re.findall(r'\d', plate_text)
        return ''.join(digits[-4:]) if len(digits) >= 4 else ''

    # ★ 영상 모드에서 유효한 완전한 번호판 형식만 허용 (부분 인식 제거)
    # config.py KR_PATTERNS 14개와 완전 동기화 + 영업용 노란판 추가
    _STRICT_PLATE_PATTERNS = [
        re.compile(r'^[가-힣]{2}[0-9]{2}[가-힣][0-9]{4}$'),       # 구형: 서울12가3456
        re.compile(r'^[0-9]{2,4}[가-힣][0-9]{4}$'),                # 신형: 12가3456, 123가4567, 8519우6374
        re.compile(r'^[가-힣]{2,3}[0-9]{2}[가-힣][0-9]{4}$'),      # 구형지역: 경기12가3456
        re.compile(r'^[가-힣]{2}[0-9]{2}[바사아자배비하][0-9]{4}$'),# 영업/버스: 서울12바3456
        re.compile(r'^[가-힣]{2,3}[0-9]{4}[가-힣]{1}$'),           # 영업용 변형: 서울1234가
        re.compile(r'^외교[0-9]{3}-?[0-9]{3}$'),                    # 외교: 외교123-456
        re.compile(r'^[가-힣]{2}[0-9]{3}[가-힣]$'),                # 이륜차: 서울123가
        re.compile(r'^[가-힣]{2}[0-9]{1,2}[가-힣]{1,2}[0-9]{4}$'),# 혼합형: 서울1가나3456
        re.compile(r'^전기[0-9]{4}$'),                              # 전기차 구형: 전기1234
        re.compile(r'^[가-힣]{2}전기[0-9]{4}$'),                    # 지역+전기차: 서울전기1234
        re.compile(r'^[0-9]{2}[가-힣][0-9]{4}$'),                  # 신형 전기차: 12가3456
        re.compile(r'^[가-힣][0-9]{2}[가-힣][0-9]{4}$'),           # 영업용 1줄: 충86다6118
        # ★ 삭제: [가-힣]\d{4} (바6286) — 5자리 부분 인식이 확정 결과로 표시되는 원인
        # 2줄판 하단은 2LINE 복원 내부에서만 사용, 독립 결과로 출력 금지
        # ★ 삭제: [가-힣]{2}\d{4} (이나8060) — 부분 인식 오표시 방지
        re.compile(r'^[0-9]{3,4}[가-힣][0-9]{4}$'),                 # 영업용 노란판 + 4자리 앞번호: 586다6118, 8519우6374
    ]

    @classmethod
    def _is_strict_valid_plate(cls, text):
        """★ 엄격한 번호판 형식 검증: 완전한 형식만 허용.
        config.py KR_PATTERNS 14개 + 영업용 노란판(586다6118) 포함.
        바6282(부분인식), 250보5351(비표준) 등 제거."""
        if not text:
            return False
        # 최소 5자 이상 (전기1234=6자, 바6286=5자 등 특수 번호판 허용)
        if len(text) < 5:
            return False
        # 한글이 최소 1자 포함
        if not re.search(r'[가-힣]', text):
            return False
        # 엄격 패턴 매칭
        for pat in cls._STRICT_PLATE_PATTERNS:
            if pat.match(text):
                return True
        return False

    def _deduplicate_results(self, results):
        """같은 프레임 내 뒤 4자리 숫자 동일 번호판 → conf 높은 것만 유지"""
        # ★ conf 필터 (0.40으로 완화 — 오탐 방지 + 원거리 결과 보존 균형)
        results = [r for r in results if r.get("confidence", 0) >= 0.40]
        # ★ 글로벌 2LINE 복원: 부분 인식을 recent_plates에서 전체 형식으로 복원
        # 형식 필터 전에 실행해야 부분 인식이 제거되지 않음
        # 대상: "바7789"(한글+4숫자), "70바9203"(숫자+한글+4숫자, 지역명 누락)
        for i, r in enumerate(results):
            plate = r.get("plate", "")
            # 부분 인식 패턴 확장: 한글+4숫자 또는 숫자2~4+한글+4숫자 (지역명 없는 것)
            # ★ 강화: XX[가-힣]XXXX (7자리)가 recent_plates에서 Y+XX[가-힣]XXXX (8자리)와 매칭 시 복원
            _is_partial = (re.match(r'^[가-힣]\d{4}$', plate)
                          or (re.match(r'^\d{2,4}[가-힣]\d{4}$', plate)
                              and not re.match(r'^[가-힣]', plate))
                          or (re.match(r'^\d{2}[가-힣]\d{4}$', plate)
                              and any(kp.endswith(plate) and len(kp) > len(plate)
                                      for kp in self.recent_plates)))
            if _is_partial:
                # recent_plates에서 이 부분 인식으로 끝나는 더 긴 전체 형식 찾기
                _restored = False
                for known_plate in self.recent_plates:
                    if (known_plate.endswith(plate) and len(known_plate) > len(plate)
                            and self._is_strict_valid_plate(known_plate)):
                        print(f"[2LINE-GLOBAL] 부분 인식 복원: {plate} → {known_plate}", flush=True)
                        results[i] = dict(r)
                        results[i]["plate"] = known_plate
                        _restored = True
                        break
                if not _restored:
                    # _ocr_track_cache에서도 검색
                    for _tk, _tc in self._ocr_track_cache.items():
                        _cached_text = _tc.get("text", "")
                        if (_cached_text.endswith(plate) and len(_cached_text) > len(plate)
                                and self._is_strict_valid_plate(_cached_text)):
                            print(f"[2LINE-GLOBAL] 캐시 기반 복원: {plate} → {_cached_text}", flush=True)
                            results[i] = dict(r)
                            results[i]["plate"] = _cached_text
                            _restored = True
                            break
                # ★ 뒤4자리 매칭 복원: "세7789" → recent_plates에서 "7789"로 끝나는 전체 번호판 찾기
                if not _restored and re.match(r'^[가-힣]\d{4}$', plate):
                    _p_last4 = re.search(r'\d{4}$', plate).group()
                    for known_plate in self.recent_plates:
                        if (known_plate.endswith(_p_last4) and len(known_plate) > len(plate)
                                and self._is_strict_valid_plate(known_plate)):
                            print(f"[2LINE-LAST4] 뒤4자리 매칭 복원: {plate} → {known_plate}", flush=True)
                            results[i] = dict(r)
                            results[i]["plate"] = known_plate
                            _restored = True
                            break
                    if not _restored:
                        for _tk, _tc in self._ocr_track_cache.items():
                            _cached_text = _tc.get("text", "")
                            if (_cached_text.endswith(_p_last4) and len(_cached_text) > len(plate)
                                    and self._is_strict_valid_plate(_cached_text)):
                                print(f"[2LINE-LAST4] 캐시 뒤4자리 복원: {plate} → {_cached_text}", flush=True)
                                results[i] = dict(r)
                                results[i]["plate"] = _cached_text
                                break
        # ★ 엄격한 형식 필터: 부분 인식 / 비표준 형식 제거
        _before = len(results)
        results = [r for r in results if self._is_strict_valid_plate(r.get("plate", ""))]
        if _before > 0 and len(results) < _before:
            print(f"[STRICT-FMT] {_before}개 → {len(results)}개 (형식 불일치 {_before - len(results)}개 제거)",
                  flush=True)
        by_last4 = {}
        for r in results:
            last4 = self._extract_last4(r["plate"])
            if not last4:
                by_last4[id(r)] = r  # 4자리 추출 불가 시 그대로 유지
                continue
            if last4 not in by_last4 or r["confidence"] > by_last4[last4]["confidence"]:
                by_last4[last4] = r
        # ★ 크로스-트랙 다수결: 숫자부 동일한 결과를 전역 히스토리와 대조
        final = []
        for r in by_last4.values():
            plate = r["plate"]
            stabilized = self._cross_track_stabilize(plate, r["confidence"])
            if stabilized != plate:
                print(f"[CROSS-TRACK] {plate} → {stabilized} (전역 다수결)", flush=True)
                r = dict(r)
                r["plate"] = stabilized
            final.append(r)
        return final

    def _cross_track_stabilize(self, text, conf):
        """★ 전역 크로스-트랙 안정화: 전역 히스토리에서 같은 숫자부를 가진
        텍스트 중 최다 득표를 반환 (히스토리 누적은 _update_ocr_cache에서 수행).
        조건: 승자 3회 이상 + 현재보다 2회 이상 많을 때만 교체."""
        digits = re.sub(r'[^0-9]', '', text)
        if len(digits) < 4:
            return text

        history = self._global_plate_history.get(digits, {})
        if not history or len(history) < 2:
            return text

        # TTL 유효 투표만 필터링
        alive = {t: cnt for t, (cnt, lf) in history.items()
                 if self._frame_counter - lf <= self._gph_ttl_frames}
        if len(alive) < 2:
            return text

        winner = max(alive, key=alive.get)
        winner_count = alive[winner]
        current_count = alive.get(text, 0)

        if (winner != text
                and winner_count >= 3
                and winner_count >= current_count + 2):
            return winner
        return text

    def _unify_plate_variants(self, results):
        """★ 최종 변이 통합: 같은 숫자부를 가진 번호판 변이 → 전역 최다 득표로 통일.
        이미 출력된 결과도 소급 교체하여 동일 차량이 여러 번호판으로 나오는 것 방지.
        예: 35오5546 → 35무5546 (전역 히스토리에서 35무5546이 최다)"""
        unified = []
        for r in results:
            plate = r.get("plate", "")
            digits = re.sub(r'[^0-9]', '', plate)
            if len(digits) < 4 or digits not in self._global_plate_history:
                unified.append(r)
                continue
            history = self._global_plate_history[digits]
            alive = {t: cnt for t, (cnt, lf) in history.items()
                     if self._frame_counter - lf <= self._gph_ttl_frames}
            if len(alive) < 2:
                unified.append(r)
                continue
            winner = max(alive, key=alive.get)
            if winner != plate and alive[winner] > alive.get(plate, 0):
                print(f"[UNIFY] {plate} → {winner} "
                      f"(히스토리: {plate}={alive.get(plate,0)}, {winner}={alive[winner]})",
                      flush=True)
                r = dict(r)
                r["plate"] = winner
            unified.append(r)
        return unified

    def detect_only(self, frame) -> list[dict]:
        """번호판 위치 탐지 전용 (model_fast — best.pt 별도 인스턴스).
        Phase2의 self.model과 별개 인스턴스 → 동시 호출 스레드 안전.
        번호판 bbox 반환 → Phase2 결과와 IoU 매칭으로 현재 위치 보정."""
        try:
            results = []
            ch, cw = frame.shape[:2]
            # 별도 인스턴스(model_fast)로 탐지 → Phase2(self.model)와 충돌 없음
            p_res = self.model_fast(frame, conf=self.config.DETECT_CONF,
                                    imgsz=640, verbose=False)
            for pbox in p_res[0].boxes:
                x1, y1, x2, y2 = map(int, pbox.xyxy[0].tolist())
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(cw, x2); y2 = min(ch, y2)
                if x2 - x1 < 20 or y2 - y1 < 10:
                    continue
                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(pbox.conf[0]),
                })
            return results
        except Exception:
            return []

    def process_frame(self, frame, camera_id="CAM01", use_multiframe=False, full_frame=None):
        """
        2-Stage 파이프라인:
          Stage1: frame → YOLO(yolov8n.pt, 차량 탐지) → 차량 크롭
          Stage2: 차량 크롭 → YOLO(best.pt, 번호판 탐지) → 번호판 크롭 → OCR
          폴백: 차량 0대 → 기존 1-Stage (frame → 번호판 직접 탐지)
        """
        self.stats["frames_processed"] += 1

        # 전역 히스토리 정리 (30프레임마다)
        self._cleanup_global_plate_history()
        # 최적화③: 프레임 스킵 (N프레임마다 1번 YOLO, 중간은 캐시 재사용)
        self._frame_counter += 1
        if self._frame_counter % self._frame_skip_interval != 1 and self._cached_results is not None:
            # ★ 스킵 프레임에서도 트랙 노화 처리 — Ghost 방지
            cached_keys = set()
            for r in self._cached_results:
                if "bbox" in r:
                    cached_keys.add(self._make_track_key(r["bbox"]))
            for _tk in list(self._ocr_track_cache.keys()):
                if _tk not in cached_keys:
                    self._ocr_track_cache[_tk]["frames_absent"] = (
                        self._ocr_track_cache[_tk].get("frames_absent", 0) + 1
                    )
                    if self._ocr_track_cache[_tk]["frames_absent"] >= 3:
                        del self._ocr_track_cache[_tk]
            return list(self._cached_results)

        results = []
        # === PRIMARY 생략: 2-Stage 파이프라인만 사용 (YOLO 호출 1회 감소 → FPS 개선) ===

        crop_src = full_frame if full_frame is not None else frame
        ch_full, cw_full = crop_src.shape[:2]
        ch_det, cw_det = frame.shape[:2]
        sx = cw_full / cw_det
        sy = ch_full / ch_det
        frame_area = cw_det * ch_det

        # Stage 1: 차량 탐지 (yolov8n.pt, classes=[2,5,7])
        #   2=car, 5=bus, 7=truck (COCO)
        #   최적화①: imgsz 640→416 (차량은 큰 객체라 충분)
        vehicle_results = self.model_vehicle(frame, conf=0.3, classes=[2, 5, 7],
                                             imgsz=416, verbose=False)
        vehicle_boxes = []
        for det in vehicle_results[0].boxes:
            vx1, vy1, vx2, vy2 = map(int, det.xyxy[0].tolist())
            vconf = float(det.conf[0])
            area = (vx2 - vx1) * (vy2 - vy1)
            vehicle_boxes.append((vx1, vy1, vx2, vy2, vconf, area))

        # 면적 큰 순 정렬 (가까운 차량 우선)
        vehicle_boxes.sort(key=lambda v: v[5], reverse=True)

        # ROI 필터: 차량 bbox 중심점이 ROI 안에 있는 것만 Stage2로 전달
        if ThresholdConfig.ROI_ENABLED and vehicle_boxes:
            roi_x1 = int(ThresholdConfig.ROI_X1 * cw_det / 1920)
            roi_y1 = int(ThresholdConfig.ROI_Y1 * ch_det / 1080)
            roi_x2 = int(ThresholdConfig.ROI_X2 * cw_det / 1920)
            roi_y2 = int(ThresholdConfig.ROI_Y2 * ch_det / 1080)
            before = len(vehicle_boxes)
            vehicle_boxes = [
                v for v in vehicle_boxes
                if roi_x1 <= (v[0] + v[2]) // 2 <= roi_x2
                and roi_y1 <= (v[1] + v[3]) // 2 <= roi_y2
            ]
            if before != len(vehicle_boxes):
                print(f"[ROI] 차량 필터: {before} → {len(vehicle_boxes)} (ROI 밖 {before - len(vehicle_boxes)}대 제외)")

        ratio = vehicle_boxes[0][5] / frame_area if vehicle_boxes else 0.0
        print(f"[2STAGE] vehicles={len(vehicle_boxes)}, frame_area_ratio={ratio:.1%}")

        seen_this_frame = set()
        seen_track_keys = set()  # ★ 고스트 방지: 이번 프레임에 감지된 트랙 키

        # 최적화②: 차량 bbox 면적 ≥ 20% → Stage2 스킵 (가까운 차량 대응)
        if len(vehicle_boxes) >= 1:  # Always use 1-stage direct detection
            # 차량이 프레임 대부분을 차지 → 번호판 직접 탐지 (Stage2 생략)
            vx1, vy1, vx2, vy2, vconf, varea = vehicle_boxes[0]
            vox1, voy1 = int(vx1 * sx), int(vy1 * sy)
            vox2, voy2 = int(vx2 * sx), int(vy2 * sy)
            vox1 = max(0, vox1); voy1 = max(0, voy1)
            vox2 = min(cw_full, vox2); voy2 = min(ch_full, voy2)

            plate_detections = self.model(frame, conf=self.config.DETECT_CONF,
                                          imgsz=640, verbose=False)
            n_detected = len(plate_detections[0].boxes)
            print(f"[PLATE-DBG] detected={n_detected} conf_thresh={self.config.DETECT_CONF}", flush=True)

            # ★ 폴백: best.pt 미탐지 → 각 차량 하단 40% 직접 OCR
            if n_detected == 0:
                for _vx1, _vy1, _vx2, _vy2, _vconf, _varea in vehicle_boxes:
                    _vox1, _voy1 = int(_vx1 * sx), int(_vy1 * sy)
                    _vox2, _voy2 = int(_vx2 * sx), int(_vy2 * sy)
                    _vox1 = max(0, _vox1); _voy1 = max(0, _voy1)
                    _vox2 = min(cw_full, _vox2); _voy2 = min(ch_full, _voy2)
                    _bh = _voy2 - _voy1
                    _bw = _vox2 - _vox1
                    if _bh < 40 or _bw < 50:
                        continue
                    bottom_y = _voy1 + int(_bh * 0.6)
                    bottom_crop = crop_src[bottom_y:_voy2, _vox1:_vox2]
                    if bottom_crop.size == 0:
                        continue
                    fb_text, fb_conf = self._ocr_plate_roi(bottom_crop)
                    if fb_text and fb_conf >= self.config.OCR_CONF:
                        fb_bbox = [_vox1, bottom_y, _vox2, _voy2]
                        print(f"[FALLBACK-OCR] text={fb_text} conf={fb_conf:.2f}", flush=True)
                        seen_this_frame.add(fb_text)
                        plate_info = self.recent_plates[fb_text]
                        plate_info["consecutive"] = plate_info.get("consecutive", 0) + 1
                        plate_info["last_seen"] = time.time()
                        plate_info["count"] += 1
                        if plate_info["consecutive"] >= self.consecutive_required:
                            is_alert, alert_info = (0, None)
                            if plate_info["consecutive"] == self.consecutive_required:
                                try:
                                    is_alert, alert_info = self.db.record_plate(fb_text, fb_conf, camera_id)
                                except Exception:
                                    pass
                            results.append({
                                "plate": fb_text, "confidence": fb_conf,
                                "bbox": fb_bbox, "vehicle_bbox": [_vox1, _voy1, _vox2, _voy2],
                                "is_alert": bool(is_alert), "alert_info": alert_info,
                            })

            for pdet in plate_detections[0].boxes:
                px1, py1, px2, py2 = map(int, pdet.xyxy[0].tolist())
                pconf = float(pdet.conf[0])

                ox1, oy1 = int(px1 * sx), int(py1 * sy)
                ox2, oy2 = int(px2 * sx), int(py2 * sy)

                # ★ 번호판 위치 필터: 차량 bbox 하단 영역에만 허용 (적응형)
                # 앞유리(상단), 그릴(상단 중간), 화물칸(전체) 오탐 방지
                # 원거리(작은 bbox) → 20% 제외, 근거리(큰 bbox) → 35% 제외
                _pcy = (oy1 + oy2) // 2          # 번호판 중심 Y
                _vh = voy2 - voy1                 # 차량 높이
                _pw = ox2 - ox1                   # 번호판 너비
                if _vh > 0:
                    # ★ 적응형 상단 제외: 번호판이 작으면(원거리) 제외 비율 낮춤
                    if _pw < 80:
                        _top_ratio = 0.15  # 원거리: 상단 15% 제외
                    elif _pw < 150:
                        _top_ratio = 0.25  # 중거리: 상단 25% 제외
                    else:
                        _top_ratio = 0.35  # 근거리: 상단 35% 제외
                    _plate_y_min = voy1 + int(_vh * _top_ratio)
                    if _pcy < _plate_y_min:
                        print(f"[POS-FILTER] 번호판 위치 부적합: "
                              f"pcy={_pcy} < zone_y={_plate_y_min} "
                              f"(차량={voy1}~{voy2}, 상단{int(_top_ratio*100)}% 제외, pw={_pw})", flush=True)
                        continue

                # ROI 필터: 번호판 중심이 ROI 밖이면 스킵
                if ThresholdConfig.ROI_ENABLED:
                    pcx = (ox1 + ox2) // 2
                    pcy = (oy1 + oy2) // 2
                    r_x1 = int(ThresholdConfig.ROI_X1 * cw_full / 1920)
                    r_y1 = int(ThresholdConfig.ROI_Y1 * ch_full / 1080)
                    r_x2 = int(ThresholdConfig.ROI_X2 * cw_full / 1920)
                    r_y2 = int(ThresholdConfig.ROI_Y2 * ch_full / 1080)
                    if not (r_x1 <= pcx <= r_x2 and r_y1 <= pcy <= r_y2):
                        continue

                pw = ox2 - ox1
                ph = oy2 - oy1
                margin_x = int(pw * 0.30)
                margin_y = int(ph * 0.20)
                rx1 = max(0, ox1 - margin_x)
                ry1 = max(0, oy1 - margin_y)
                rx2 = min(cw_full, ox2 + margin_x)
                ry2 = min(ch_full, oy2 + margin_y)
                roi = crop_src[ry1:ry2, rx1:rx2]
                if roi.size == 0:
                    continue

                plate_bbox = [ox1, oy1, ox2, oy2]
                track_key = self._make_track_key(plate_bbox)
                seen_track_keys.add(track_key)  # ★ 고스트 방지
                # ★ 캐시 무효화: frames_absent > 0 또는 bbox 중심 급변 시 새 차량으로 판단
                if track_key in self._ocr_track_cache:
                    _cache = self._ocr_track_cache[track_key]
                    _need_reset = False
                    # 조건1: 미감지 이력 있음 → 재출현 = 새 차량
                    if _cache.get("frames_absent", 0) > 0:
                        _need_reset = True
                    # 조건2: bbox 중심 25px 이상 점프 → 같은 키지만 다른 물체
                    _last_bbox = _cache.get("bbox")
                    if _last_bbox and not _need_reset:
                        _lcx = (_last_bbox[0] + _last_bbox[2]) / 2
                        _lcy = (_last_bbox[1] + _last_bbox[3]) / 2
                        _ccx = (plate_bbox[0] + plate_bbox[2]) / 2
                        _ccy = (plate_bbox[1] + plate_bbox[3]) / 2
                        if ((_lcx - _ccx)**2 + (_lcy - _ccy)**2) ** 0.5 > 25:
                            _need_reset = True
                    if _need_reset:
                        print(f"[GHOST-RESET] 캐시 초기화: key={track_key}", flush=True)
                        del self._ocr_track_cache[track_key]
                skip_ocr = self._should_skip_ocr(track_key, plate_bbox)

                if skip_ocr:
                    best_text, best_conf = self._get_cached_ocr(track_key)
                    self._update_ocr_cache(track_key, plate_bbox, best_text, best_conf, did_ocr=False)
                else:
                    best_text, best_conf = self._ocr_plate_roi(roi, use_multiframe)
                    # CRNN 한글 검증 + 2줄 번호판 복원 (1회 호출)
                    if best_text and re.search(r'[가-힣]', best_text):
                        cmx = int(pw * 0.35)
                        cmy_top = int(ph * 0.50)  # 2줄 상단 포함
                        cmy_bot = int(ph * 0.40)
                        crnn_roi = crop_src[
                            max(0, oy1 - cmy_top):min(ch_full, oy2 + cmy_bot),
                            max(0, ox1 - cmx):min(cw_full, ox2 + cmx)
                        ]
                        _2line_restored = False  # ★ 2LINE 복원 플래그 초기화
                        if crnn_roi.size > 0:
                            crnn_text, crnn_conf = self._crnn_read_plate(crnn_roi, return_confidence=True)
                            # 1) 한글 검증 (CRNN 신뢰도 전달로 과적합 방어)
                            _before_crnn = best_text
                            best_text = self._verify_korean_with_crnn(best_text, crnn_roi, crnn_text, crnn_conf)
                            # ★ CRNN-PREFIX 교정 시에도 2LINE 복원 보너스 적용
                            if (_before_crnn != best_text
                                    and re.match(r'^\d{2,4}[가-힣]\d{4}$', best_text)
                                    and re.match(r'^\d{2,4}[가-힣]\d{4}$', _before_crnn)):
                                _b_prefix = re.match(r'^(\d{2,4})', _before_crnn).group(1)
                                _a_prefix = re.match(r'^(\d{2,4})', best_text).group(1)
                                if _b_prefix != _a_prefix:
                                    _2line_restored = True  # prefix 교정도 확인된 결과로 보너스
                            # ★ _verify가 구형/영업용 2줄 결과를 직접 반환한 경우 복원 플래그 설정
                            if (_before_crnn != best_text
                                    and not re.match(r'^\d{2,4}[가-힣]\d{4}$', best_text)
                                    and (re.fullmatch(r'[가-힣]{2,3}\d{2}[가-힣]\d{4}', best_text)
                                         or re.fullmatch(r'[가-힣]\d{2}[가-힣]\d{4}', best_text))):
                                _2line_restored = True
                                print(f"[CRNN-2LINE-FLAG] _verify 구형 복원 → _2line_restored=True", flush=True)
                            # 2) 2줄 번호판 지역명 복원
                            _prev_best = best_text  # ★ 복원 실패 시 롤백용 저장
                            # ★ 후미 매칭: PaddleOCR과 CRNN의 한글+4자리가 일치하면 복원
                            _paddle_suffix_m = re.search(r'[가-힣]\d{4}$', best_text)
                            _crnn_suffix_m = re.search(r'[가-힣]\d{4}$', crnn_text or "")
                            _suffix_match = (_paddle_suffix_m and _crnn_suffix_m
                                             and _paddle_suffix_m.group() == _crnn_suffix_m.group())
                            _full_substr = (crnn_text and best_text
                                            and (crnn_text.endswith(best_text) or best_text in crnn_text))
                            # ★ 영업용 body 매칭: "586다6118" vs "충86다6118" (같은 길이)
                            _comm_body_match = False
                            _comm_p_m = re.match(r'^\d(\d{2}[가-힣]\d{4})$', best_text)
                            _comm_c_m = re.match(r'^[가-힣](\d{2}[가-힣]\d{4})$', crnn_text or "")
                            if _comm_p_m and _comm_c_m:
                                _comm_body_match = (_comm_p_m.group(1) == _comm_c_m.group(1))
                            if (crnn_text
                                    and re.match(r'^\d{0,3}[가-힣]\d{4}$', best_text)
                                    and not re.match(r'^\d{2,4}[가-힣]\d{4}$', best_text)  # ★ 신형 완전 번호판 제외 (02누2754 등)
                                    and (len(crnn_text) > len(best_text)
                                         or _comm_body_match)  # ★ 영업용: 길이 같아도 body 일치 시 통과
                                    and (_full_substr or _suffix_match or _comm_body_match)
                                    and (re.fullmatch(r'[가-힣]{2,3}\d{2}[가-힣]\d{4}', crnn_text)
                                         or re.fullmatch(r'[가-힣]\d{2}[가-힣]\d{4}', crnn_text))):  # ★ 영업용 추가
                                print(f"[2LINE] CRNN 지역명 복원: {best_text} → {crnn_text} "
                                      f"(suffix={'일치' if _suffix_match else 'body' if _comm_body_match else 'substr'})",
                                      flush=True)
                                best_text = crnn_text
                                _2line_restored = True  # ★ 2LINE 복원 성공 플래그
                                # ★ 최종 안전 검증: 복원 결과 패턴 재확인
                                if not (re.fullmatch(r'[가-힣]{2,3}\d{2}[가-힣]\d{4}', best_text)
                                        or re.fullmatch(r'[가-힣]\d{2}[가-힣]\d{4}', best_text)):
                                    print(f"[2LINE] 최종검증 실패, 복원 취소: '{best_text}'", flush=True)
                                    best_text = _prev_best
                                    _2line_restored = False
                            # 3) 지역명 CRNN 교정: OCR 지역명이 틀렸을 경우 CRNN 결과로 보정
                            # 예: 서울76바7789 → 경기76바7789 (CRNN이 다른 지역명을 읽은 경우)
                            if (crnn_text
                                    and re.fullmatch(r'[가-힣]{2,3}\d{2}[가-힣]\d{4}', best_text)
                                    and re.fullmatch(r'[가-힣]{2,3}\d{2}[가-힣]\d{4}', crnn_text)):
                                _b_suffix = re.sub(r'^[가-힣]{2,3}', '', best_text)
                                _c_suffix = re.sub(r'^[가-힣]{2,3}', '', crnn_text)
                                _c_region_m = re.match(r'^[가-힣]{2,3}', crnn_text)
                                _c_region = _c_region_m.group() if _c_region_m else ""
                                if (_b_suffix == _c_suffix
                                        and _c_region in PlateValidator._REGION_PREFIXES
                                        and crnn_text != best_text):
                                    print(f"[CRNN-REGION] 지역명 교정: {best_text} → {crnn_text}", flush=True)
                                    best_text = crnn_text
                            # 4) 영업용 번호판 교정: "596아6118" → "충86아6118"
                            # PaddleOCR이 첫 한글(충,전,경 등)을 숫자(5,3,1)로 오인식하는 경우
                            # 3자리+한글+4자리 → 1한글+2자리+한글+4자리로 교정
                            _comm_m = re.match(r'^(\d{3})([가-힣])(\d{4})$', best_text)
                            # ★ 가드: COMM-FIX는 CRNN이 영업용 패턴을 확인한 경우에만 허용
                            # (A) CRNN이 구형 지역 패턴 → 스킵 (2LINE에서 처리)
                            # (B) CRNN이 영업용 패턴 → 허용 (CRNN 교차검증 완료)
                            # (C) CRNN 미확인/실패 → 스킵 (추측성 교정 방지)
                            # 예: "170바9203" → CRNN "서울70바9203" → (A) 스킵
                            #     "586다6118" → CRNN "충86다6118" → (B) 허용
                            #     "386오6118" → CRNN "589599" → (C) 스킵
                            _comm_skip = True  # ★ 기본값: 스킵 (CRNN 확인 시에만 허용)
                            if _comm_m:
                                if crnn_text:
                                    if re.fullmatch(r'[가-힣]{2,3}\d{2}[가-힣]\d{4}', crnn_text):
                                        # (A) CRNN이 구형 지역 패턴 → 영업용 아님
                                        print(f"[COMM-SKIP] CRNN 구형 지역 감지 → COMM-FIX 스킵: "
                                              f"crnn={crnn_text}", flush=True)
                                    elif re.fullmatch(r'[가-힣]\d{2}[가-힣]\d{4}', crnn_text):
                                        # (B) CRNN이 영업용 패턴 확인 → 허용
                                        _comm_skip = False
                                    elif re.search(r'[가-힣]', crnn_text):
                                        # (B-2) CRNN에 한글 포함 + 뒤 4자리 일치 → 허용
                                        # 예: "충86아6118" → crnn "충866118" (한글 누락)
                                        _crnn_d = re.sub(r'[^0-9]', '', crnn_text)
                                        _comm_d = _comm_m.group(3)  # 뒤 4자리
                                        if _crnn_d.endswith(_comm_d):
                                            _comm_skip = False
                                            print(f"[COMM-ALLOW] CRNN 부분 매칭 (뒤4={_comm_d}): crnn={crnn_text!r}", flush=True)
                                        else:
                                            print(f"[COMM-SKIP] CRNN 숫자 불일치 → COMM-FIX 스킵: "
                                                  f"crnn={crnn_text!r}", flush=True)
                                    else:
                                        # (C) CRNN에 한글 없음 → 추측성 교정 방지
                                        print(f"[COMM-SKIP] CRNN 한글 없음 → COMM-FIX 스킵: "
                                              f"crnn={crnn_text!r}", flush=True)
                                else:
                                    # ★ CRNN 실패 시에도 컬러 번호판이면 허용 (노란판은 영업용 가능성 높음)
                                    if getattr(self, '_last_color_plate', False):
                                        _comm_skip = False
                                        print(f"[COMM-ALLOW] CRNN 실패 but 컬러판 → COMM-FIX 허용", flush=True)
                            # ★ 이미 유효한 신형 번호판이면 COMM-FIX 불필요 (180라9396 등)
                            if _comm_m and not _2line_restored and not _comm_skip:
                                _already_valid, _ = self.validator.validate(best_text)
                                if _already_valid:
                                    _comm_skip = True
                            if _comm_m and not _2line_restored and not _comm_skip:
                                _first_digit = _comm_m.group(1)[0]  # "5" in "596"
                                _rest_digits = _comm_m.group(1)[1:]  # "96" in "596"
                                _hangul = _comm_m.group(2)
                                _suffix = _comm_m.group(3)
                                # 숫자→한글 오인식 매핑 (OCR에서 자주 혼동)
                                # ★ 형태 유사도 기반: 확신 높은 쌍만 유지
                                # 충: ㅊ 상단획+ㅜ 곡선 → 5,6으로 오인식
                                # 전: ㅈ+ㅓ+ㄴ → 1,3으로 오인식
                                # 경: ㄱ+ㅕ 꺾임 → 2로 오인식
                                _digit_to_hangul = {
                                    '5': '충', '6': '충', '9': '충',
                                    '1': '전', '3': '전',
                                    '2': '경',
                                }
                                if _first_digit in _digit_to_hangul:
                                    _region_char = _digit_to_hangul[_first_digit]
                                    _commercial = _region_char + _rest_digits + _hangul + _suffix
                                    is_valid_comm, final_comm = self.validator.validate(_commercial)
                                    if is_valid_comm:
                                        print(f"[COMM-FIX] 영업용 교정: {best_text} → {final_comm} "
                                              f"(첫 숫자 '{_first_digit}'→'{_region_char}')", flush=True)
                                        best_text = final_comm
                                        _2line_restored = True

                            # 4-2) 3자리+한글+4자리 → 2줄 지역 번호판 복원 시도
                            # "136아6118" → 앞 1~2자리가 지역명 오인식 → "충남86아6118"
                            # 전략: CRNN 결과 우선 → 실패 시 2자리 숫자→지역명 매핑 폴백
                            if (not _2line_restored
                                    and re.match(r'^\d{3}[가-힣]\d{4}$', best_text)):
                                _3d_hangul = re.search(r'[가-힣]', best_text).group()
                                _3d_suffix = re.search(r'\d{4}$', best_text).group()
                                _3d_digits = re.match(r'^(\d{3})', best_text).group(1)
                                # (A) CRNN이 지역명 포함 구형 패턴을 읽었으면 직접 채택
                                if crnn_text:
                                    _cr_region_m = re.fullmatch(r'([가-힣]{2,3})(\d{2})([가-힣])(\d{4})', crnn_text)
                                    if (_cr_region_m
                                            and _cr_region_m.group(4) == _3d_suffix
                                            and _cr_region_m.group(1) in PlateValidator._REGION_PREFIXES):
                                        is_v, final_v = self.validator.validate(crnn_text)
                                        if is_v:
                                            print(f"[3DIGIT-CRNN] 3자리→지역명 복원: {best_text} → {final_v} (crnn)", flush=True)
                                            best_text = final_v
                                            _2line_restored = True
                                # (B) CRNN이 영업용 1줄(충86아6118) 읽었으면 채택
                                if not _2line_restored and crnn_text:
                                    _cr_comm_m = re.fullmatch(r'([가-힣])(\d{2})([가-힣])(\d{4})', crnn_text)
                                    if (_cr_comm_m
                                            and _cr_comm_m.group(4) == _3d_suffix):
                                        is_v, final_v = self.validator.validate(crnn_text)
                                        if is_v:
                                            print(f"[3DIGIT-CRNN] 3자리→영업용 복원: {best_text} → {final_v} (crnn)", flush=True)
                                            best_text = final_v
                                            _2line_restored = True
                                # (C) CRNN 실패 시 2자리 숫자→지역명 매핑 폴백
                                # "136" → "13"="충남" + "6" → "충남6" → 재조합 "충남86아6118"
                                # PaddleOCR이 지역명(충남,전남,경기 등)을 숫자로 오인식하는 패턴
                                if not _2line_restored and getattr(self, '_last_color_plate', False):
                                    _2digit_to_region = {
                                        '13': '충남', '15': '충남', '16': '충남',
                                        '56': '충남', '53': '충남', '96': '충남',
                                        '86': '충남', '58': '충남', '98': '충남',
                                        '10': '충북', '50': '충북',
                                        '14': '전남', '34': '전남',
                                        '11': '전북', '31': '전북',
                                        '21': '경기', '12': '경북', '22': '경남',
                                        '17': '인천', '37': '인천',
                                    }
                                    _d2 = _3d_digits[:2]  # 앞 2자리
                                    if _d2 in _2digit_to_region:
                                        _region = _2digit_to_region[_d2]
                                        # ★ 개선: 모든 원시 OCR 숫자에서 mid 2자리 빈도 분석
                                        # suffix 제거 후, 남은 숫자에서 2자리 쌍 빈도 카운트
                                        # suffix 내부 2자리("61","11","18" 등)는 제외
                                        _raw_digits_list = getattr(self, '_last_raw_digits', [])
                                        _suffix_pairs = set()
                                        for _si in range(len(_3d_suffix) - 1):
                                            _suffix_pairs.add(_3d_suffix[_si:_si+2])
                                        _pair_cnt = Counter()
                                        for _rd in _raw_digits_list:
                                            if _rd.endswith(_3d_suffix):
                                                _prefix_part = _rd[:-4]  # suffix 제거
                                            else:
                                                _prefix_part = _rd[:-4] if len(_rd) > 4 else _rd
                                            for _pi in range(len(_prefix_part) - 1):
                                                _pair = _prefix_part[_pi:_pi+2]
                                                if _pair not in _suffix_pairs:
                                                    _pair_cnt[_pair] += 1
                                        # CRNN 힌트도 추가
                                        _crnn_digits = re.sub(r'[^0-9]', '', crnn_text or '')
                                        if len(_crnn_digits) >= 6 and _crnn_digits[-4:] == _3d_suffix:
                                            _crnn_mid = _crnn_digits[:-4][-2:]
                                            if len(_crnn_mid) == 2:
                                                _pair_cnt[_crnn_mid] += 3  # CRNN 가중치
                                        # 가장 빈도 높은 2자리 쌍 = mid digits
                                        _mid_digits = ''
                                        if _pair_cnt:
                                            _mid_digits = _pair_cnt.most_common(1)[0][0]
                                            print(f"[3DIGIT-MAP] mid 빈도분석: {dict(_pair_cnt.most_common(5))}", flush=True)
                                        if not _mid_digits:
                                            _mid_digits = _3d_digits[2] + '6'  # 최후 폴백
                                        if len(_mid_digits) == 2:
                                            _restored = _region + _mid_digits + _3d_hangul + _3d_suffix
                                            is_v, final_v = self.validator.validate(_restored)
                                            if is_v:
                                                print(f"[3DIGIT-MAP] 숫자→지역명 매핑: {best_text} → {final_v} "
                                                      f"('{_d2}'→'{_region}', mid='{_mid_digits}')", flush=True)
                                                best_text = final_v
                                                _2line_restored = True

                            # 5) 2줄 부분 인식 복원: "86아6118" → "충남86아6118"
                            # 신형 패턴이지만 상용 한글 포함 → 지역명 누락된 2줄 가능성
                            if (not _2line_restored
                                    and re.match(r'^\d{2}[가-힣]\d{4}$', best_text)):
                                _partial_hangul = re.search(r'[가-힣]', best_text).group()
                                _is_commercial = _partial_hangul in PlateValidator._COMMERCIAL_CHARS
                                if _is_commercial:
                                    _partial_restored = False
                                    # (A) CRNN에서 지역명 추출
                                    if crnn_text:
                                        _cr_full_m = re.match(r'^([가-힣]{2,3})(\d{2}[가-힣]\d{4})$', crnn_text)
                                        if (_cr_full_m
                                                and _cr_full_m.group(2) == best_text
                                                and _cr_full_m.group(1) in PlateValidator._REGION_PREFIXES):
                                            _restored = crnn_text
                                            is_v, final_v = self.validator.validate(_restored)
                                            if is_v:
                                                print(f"[2LINE-PARTIAL] CRNN 지역명 복원: {best_text} → {final_v} "
                                                      f"(conf={crnn_conf:.2f})", flush=True)
                                                best_text = final_v
                                                _2line_restored = True
                                                _partial_restored = True
                                    # (B) 히스토리/캐시 폴백
                                    if not _partial_restored:
                                        for _known in list(self.recent_plates.keys()):
                                            if (_known.endswith(best_text)
                                                    and len(_known) > len(best_text)
                                                    and self._is_strict_valid_plate(_known)):
                                                print(f"[2LINE-PARTIAL] 히스토리 복원: {best_text} → {_known}", flush=True)
                                                best_text = _known
                                                _2line_restored = True
                                                _partial_restored = True
                                                break
                                    if not _partial_restored:
                                        for _tk, _tc in self._ocr_track_cache.items():
                                            _ct = _tc.get("text", "")
                                            if (_ct.endswith(best_text)
                                                    and len(_ct) > len(best_text)
                                                    and self._is_strict_valid_plate(_ct)):
                                                print(f"[2LINE-PARTIAL] 캐시 복원: {best_text} → {_ct}", flush=True)
                                                best_text = _ct
                                                _2line_restored = True
                                                break

                    # ★ 히스토리 기반 2LINE 복원: CRNN 실패 시 이전 결과에서 지역명 복원
                    if (re.match(r'^[가-힣]\d{4}$', best_text)
                            and track_key in self._ocr_track_cache):
                        _cached = self._ocr_track_cache[track_key]
                        _cached_text = _cached.get("text", "")
                        if (re.fullmatch(r'[가-힣]{2,3}\d{2}[가-힣]\d{4}', _cached_text)
                                and _cached_text.endswith(best_text)):
                            print(f"[2LINE-HIST] 히스토리 기반 복원: {best_text} → {_cached_text}", flush=True)
                            best_text = _cached_text
                    best_text, best_conf = self._recover_hangul_from_cache(track_key, best_text, best_conf)
                    self._update_ocr_cache(track_key, plate_bbox, best_text, best_conf, did_ocr=True)
                    # ★ 트래킹 다수결 안정화
                    best_text, best_conf = self._stabilize_track_text(track_key, best_text, best_conf)
                    # ★ 부분 인식 병합 (게임/시뮬레이션 번호판 대응)
                    best_text, best_conf = self._merge_partial_plates(best_text, best_conf)

                if best_text and best_conf >= self.config.OCR_CONF:
                    seen_this_frame.add(best_text)
                    plate_info = self.recent_plates[best_text]
                    plate_info["consecutive"] = plate_info.get("consecutive", 0) + 1
                    plate_info["last_seen"] = time.time()
                    plate_info["count"] += 1
                    # ★ 2LINE CRNN 복원 보너스: 복원 성공 시 consecutive +1 (2중 검증 효과)
                    if locals().get("_2line_restored", False) and plate_info["consecutive"] == 1:
                        plate_info["consecutive"] = 2
                        print(f"[2LINE-BONUS] CRNN 2LINE 복원 → consecutive=2 (자동 확인)", flush=True)
                    # ★ 후미 기반 consecutive 공유: 같은 한글+4자리를 가진 다른 변형에서 연속 카운트 이어받기
                    # 예: "20나8060"(1/2) → "01나8060"(1/2가 아니라 2/2로 이어받음)
                    # ★ fragment 방어: 새 번호판이 기존보다 짧으면 상속 차단 (86아6118 ← 전86아6118 방지)
                    _suffix_m = re.search(r'[가-힣]\d{4}$', best_text)
                    if _suffix_m and plate_info["consecutive"] == 1:
                        _suffix = _suffix_m.group()
                        _now = time.time()
                        for _other_plate, _other_info in self.recent_plates.items():
                            if (_other_plate != best_text
                                    and _other_plate.endswith(_suffix)
                                    and len(best_text) >= len(_other_plate)  # ★ fragment 차단: 짧은 쪽은 상속 불가
                                    # ★ count 기반: 최근 5초 이내 1회 이상 감지된 적 있으면 공유
                                    and _other_info.get("count", 0) >= 1
                                    and (_now - _other_info.get("last_seen", 0)) < 5.0):
                                _inherited = max(_other_info.get("consecutive", 0), 1) + 1
                                plate_info["consecutive"] = _inherited
                                print(f"[CONSEC-SHARE] '{best_text}' ← '{_other_plate}' "
                                      f"후미 '{_suffix}' count={_other_info['count']} → consecutive={_inherited}",
                                      flush=True)
                                break
                    # 디버그: consecutive 상태 추적
                    print(f"[CONSEC] '{best_text}' conf={best_conf:.2f} "
                          f"consecutive={plate_info['consecutive']}/{self.consecutive_required} "
                          f"→ {'PASS' if plate_info['consecutive'] >= self.consecutive_required else 'WAIT'}",
                          flush=True)
                    if plate_info["consecutive"] >= self.consecutive_required:
                        is_alert, alert_info = (0, None)
                        if plate_info["consecutive"] == self.consecutive_required:
                            try:
                                is_alert, alert_info = self.db.record_plate(
                                    best_text, best_conf, camera_id)
                                if is_alert and alert_info:
                                    self._trigger_alert(best_text, alert_info)
                            except Exception:
                                pass
                        self.stats["plates_shown"] += 1
                        self.stats["confidences"].append(best_conf)
                        results.append({
                            "plate": best_text,
                            "confidence": best_conf,
                            "bbox": plate_bbox,
                            "vehicle_bbox": [vox1, voy1, vox2, voy2],
                            "is_alert": bool(is_alert),
                            "alert_info": alert_info,
                        })

        elif vehicle_boxes:
            # Stage 2: 각 차량 크롭 → 번호판 탐지 → OCR
            for vx1, vy1, vx2, vy2, vconf, varea in vehicle_boxes:
                # 차량 bbox를 원본 프레임 좌표로 변환
                vox1, voy1 = int(vx1 * sx), int(vy1 * sy)
                vox2, voy2 = int(vx2 * sx), int(vy2 * sy)
                vox1 = max(0, vox1); voy1 = max(0, voy1)
                vox2 = min(cw_full, vox2); voy2 = min(ch_full, voy2)

                vehicle_crop = crop_src[voy1:voy2, vox1:vox2]
                bw = vox2 - vox1; bh = voy2 - voy1
                if bw < 50 or bh < 20:
                    continue  # Skip tiny plates
                if vehicle_crop.size == 0:
                    continue

                # Stage2: 번호판 탐지 (best.pt)
                # 차량이 프레임 우측 절반에 있으면 좌측 부분만 크롭 (번호판은 차량 앞쪽)
                if vx1 > cw_det * 0.4:
                    crop_w = vox2 - vox1
                    vehicle_crop = vehicle_crop[:, :int(crop_w * 0.6)]
                    if vehicle_crop.size == 0:
                        vehicle_crop = crop_src[voy1:voy2, vox1:vox2]
                plate_detections = self.model(vehicle_crop, conf=0.05, imgsz=640, verbose=False)
                n_plates = len(plate_detections[0].boxes)
                print(f"[PLATE-DBG2] stage2 plates={n_plates}", flush=True)

                # ★ 폴백: best.pt 미탐지 → 차량 하단 40% 직접 OCR
                if n_plates == 0 and bh > 40:
                    bottom_crop = crop_src[voy1 + int(bh * 0.6):voy2, vox1:vox2]
                    if bottom_crop.size > 0:
                        fb_text, fb_conf = self._ocr_plate_roi(bottom_crop)
                        if fb_text and fb_conf >= self.config.OCR_CONF:
                            fb_bbox = [vox1, voy1 + int(bh * 0.6), vox2, voy2]
                            print(f"[FALLBACK-OCR] text={fb_text} conf={fb_conf:.2f}", flush=True)
                            seen_this_frame.add(fb_text)
                            plate_info = self.recent_plates[fb_text]
                            plate_info["consecutive"] = plate_info.get("consecutive", 0) + 1
                            plate_info["last_seen"] = time.time()
                            plate_info["count"] += 1
                            if plate_info["consecutive"] >= self.consecutive_required:
                                is_alert, alert_info = (0, None)
                                if plate_info["consecutive"] == self.consecutive_required:
                                    try:
                                        is_alert, alert_info = self.db.record_plate(fb_text, fb_conf, camera_id)
                                    except Exception:
                                        pass
                                results.append({
                                    "plate": fb_text, "confidence": fb_conf,
                                    "bbox": fb_bbox, "vehicle_bbox": [vox1, voy1, vox2, voy2],
                                    "is_alert": bool(is_alert), "alert_info": alert_info,
                                })

                for pdet in plate_detections[0].boxes:
                    px1, py1, px2, py2 = map(int, pdet.xyxy[0].tolist())
                    pconf = float(pdet.conf[0])

                    # 번호판 bbox → 원본 프레임 전역 좌표 변환
                    plate_x1_global = vox1 + px1
                    plate_y1_global = voy1 + py1
                    plate_x2_global = vox1 + px2
                    plate_y2_global = voy1 + py2

                    # 마진 확장
                    pw = plate_x2_global - plate_x1_global
                    ph = plate_y2_global - plate_y1_global
                    margin_x = int(pw * 0.2)
                    margin_y = int(ph * 0.25)
                    rx1 = max(0, plate_x1_global - margin_x)
                    ry1 = max(0, plate_y1_global - margin_y)
                    rx2 = min(cw_full, plate_x2_global + margin_x)
                    ry2 = min(ch_full, plate_y2_global + margin_y)
                    roi = crop_src[ry1:ry2, rx1:rx2]

                    if roi.size == 0:
                        continue

                    plate_bbox = [plate_x1_global, plate_y1_global,
                                  plate_x2_global, plate_y2_global]
                    track_key = self._make_track_key(plate_bbox)
                    seen_track_keys.add(track_key)  # ★ 고스트 방지
                    # ★ 캐시 무효화: frames_absent > 0 또는 bbox 중심 급변 시 새 차량으로 판단
                    if track_key in self._ocr_track_cache:
                        _cache = self._ocr_track_cache[track_key]
                        _need_reset = False
                        if _cache.get("frames_absent", 0) > 0:
                            _need_reset = True
                        _last_bbox = _cache.get("bbox")
                        if _last_bbox and not _need_reset:
                            _lcx = (_last_bbox[0] + _last_bbox[2]) / 2
                            _lcy = (_last_bbox[1] + _last_bbox[3]) / 2
                            _ccx = (plate_bbox[0] + plate_bbox[2]) / 2
                            _ccy = (plate_bbox[1] + plate_bbox[3]) / 2
                            if ((_lcx - _ccx)**2 + (_lcy - _ccy)**2) ** 0.5 > 25:
                                _need_reset = True
                        if _need_reset:
                            print(f"[GHOST-RESET] 캐시 초기화: key={track_key}", flush=True)
                            del self._ocr_track_cache[track_key]
                    skip_ocr = self._should_skip_ocr(track_key, plate_bbox)

                    if skip_ocr:
                        best_text, best_conf = self._get_cached_ocr(track_key)
                        self._update_ocr_cache(track_key, plate_bbox, best_text, best_conf, did_ocr=False)
                    else:
                        best_text, best_conf = self._ocr_plate_roi(roi, use_multiframe)
                        # ★ 히스토리 기반 2LINE 복원
                        if (re.match(r'^[가-힣]\d{4}$', best_text)
                                and track_key in self._ocr_track_cache):
                            _cached = self._ocr_track_cache[track_key]
                            _cached_text = _cached.get("text", "")
                            if (re.fullmatch(r'[가-힣]{2,3}\d{2}[가-힣]\d{4}', _cached_text)
                                    and _cached_text.endswith(best_text)):
                                print(f"[2LINE-HIST] 히스토리 기반 복원: {best_text} → {_cached_text}", flush=True)
                                best_text = _cached_text
                        best_text, best_conf = self._recover_hangul_from_cache(track_key, best_text, best_conf)
                        self._update_ocr_cache(track_key, plate_bbox, best_text, best_conf, did_ocr=True)
                    # ★ 트래킹 다수결 안정화
                    best_text, best_conf = self._stabilize_track_text(track_key, best_text, best_conf)
                    # ★ 부분 인식 병합 (게임/시뮬레이션 번호판 대응)
                    best_text, best_conf = self._merge_partial_plates(best_text, best_conf)

                    if best_text and best_conf >= self.config.OCR_CONF:
                        seen_this_frame.add(best_text)
                        plate_info = self.recent_plates[best_text]
                        plate_info["consecutive"] = plate_info.get("consecutive", 0) + 1
                        plate_info["last_seen"] = time.time()
                        plate_info["count"] += 1

                        if plate_info["consecutive"] >= self.consecutive_required:
                            is_alert, alert_info = (0, None)
                            if plate_info["consecutive"] == self.consecutive_required:
                                try:
                                    is_alert, alert_info = self.db.record_plate(
                                        best_text, best_conf, camera_id
                                    )
                                    if is_alert and alert_info:
                                        self._trigger_alert(best_text, alert_info)
                                except Exception:
                                    pass
                            self.stats["plates_shown"] += 1
                            self.stats["confidences"].append(best_conf)
                            results.append({
                                "plate": best_text,
                                "confidence": best_conf,
                                "bbox": plate_bbox,
                                "vehicle_bbox": [vox1, voy1, vox2, voy2],
                                "is_alert": bool(is_alert),
                                "alert_info": alert_info,
                            })
        else:
            # 폴백: 차량 0대 → 기존 1-Stage (번호판 직접 탐지)
            detections = self.model(frame, conf=self.config.DETECT_CONF, verbose=False)

            for det in detections[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                conf = float(det.conf[0])

                ox1, oy1 = int(x1 * sx), int(y1 * sy)
                ox2, oy2 = int(x2 * sx), int(y2 * sy)

                # ROI 필터: 번호판 중심이 ROI 밖이면 스킵
                if ThresholdConfig.ROI_ENABLED:
                    pcx = (ox1 + ox2) // 2
                    pcy = (oy1 + oy2) // 2
                    r_x1 = int(ThresholdConfig.ROI_X1 * cw_full / 1920)
                    r_y1 = int(ThresholdConfig.ROI_Y1 * ch_full / 1080)
                    r_x2 = int(ThresholdConfig.ROI_X2 * cw_full / 1920)
                    r_y2 = int(ThresholdConfig.ROI_Y2 * ch_full / 1080)
                    if not (r_x1 <= pcx <= r_x2 and r_y1 <= pcy <= r_y2):
                        continue

                margin_x = int((ox2 - ox1) * 0.1)
                margin_y = int((oy2 - oy1) * 0.15)
                rx1 = max(0, ox1 - margin_x)
                ry1 = max(0, oy1 - margin_y)
                rx2 = min(cw_full, ox2 + margin_x)
                ry2 = min(ch_full, oy2 + margin_y)
                roi = crop_src[ry1:ry2, rx1:rx2]

                if roi.size == 0:
                    continue

                plate_bbox = [ox1, oy1, ox2, oy2]
                track_key = self._make_track_key(plate_bbox)
                seen_track_keys.add(track_key)  # ★ 고스트 방지
                # ★ 캐시 무효화: frames_absent > 0 또는 bbox 중심 급변 시 새 차량으로 판단
                if track_key in self._ocr_track_cache:
                    _cache = self._ocr_track_cache[track_key]
                    _need_reset = False
                    # 조건1: 미감지 이력 있음 → 재출현 = 새 차량
                    if _cache.get("frames_absent", 0) > 0:
                        _need_reset = True
                    # 조건2: bbox 중심 25px 이상 점프 → 같은 키지만 다른 물체
                    _last_bbox = _cache.get("bbox")
                    if _last_bbox and not _need_reset:
                        _lcx = (_last_bbox[0] + _last_bbox[2]) / 2
                        _lcy = (_last_bbox[1] + _last_bbox[3]) / 2
                        _ccx = (plate_bbox[0] + plate_bbox[2]) / 2
                        _ccy = (plate_bbox[1] + plate_bbox[3]) / 2
                        if ((_lcx - _ccx)**2 + (_lcy - _ccy)**2) ** 0.5 > 25:
                            _need_reset = True
                    if _need_reset:
                        print(f"[GHOST-RESET] 캐시 초기화: key={track_key}", flush=True)
                        del self._ocr_track_cache[track_key]
                skip_ocr = self._should_skip_ocr(track_key, plate_bbox)

                if skip_ocr:
                    best_text, best_conf = self._get_cached_ocr(track_key)
                    self._update_ocr_cache(track_key, plate_bbox, best_text, best_conf, did_ocr=False)
                else:
                    best_text, best_conf = self._ocr_plate_roi(roi, use_multiframe)
                    # ★ 히스토리 기반 2LINE 복원
                    if (re.match(r'^[가-힣]\d{4}$', best_text)
                            and track_key in self._ocr_track_cache):
                        _cached = self._ocr_track_cache[track_key]
                        _cached_text = _cached.get("text", "")
                        if (re.fullmatch(r'[가-힣]{2,3}\d{2}[가-힣]\d{4}', _cached_text)
                                and _cached_text.endswith(best_text)):
                            print(f"[2LINE-HIST] 히스토리 기반 복원: {best_text} → {_cached_text}", flush=True)
                            best_text = _cached_text
                    best_text, best_conf = self._recover_hangul_from_cache(track_key, best_text, best_conf)
                    self._update_ocr_cache(track_key, plate_bbox, best_text, best_conf, did_ocr=True)
                # ★ 트래킹 다수결 안정화
                best_text, best_conf = self._stabilize_track_text(track_key, best_text, best_conf)
                # ★ 부분 인식 병합 (게임/시뮬레이션 번호판 대응)
                best_text, best_conf = self._merge_partial_plates(best_text, best_conf)

                if best_text and best_conf >= self.config.OCR_CONF:
                    seen_this_frame.add(best_text)
                    plate_info = self.recent_plates[best_text]
                    plate_info["consecutive"] = plate_info.get("consecutive", 0) + 1
                    plate_info["last_seen"] = time.time()
                    plate_info["count"] += 1

                    if plate_info["consecutive"] >= self.consecutive_required:
                        is_alert, alert_info = (0, None)
                        if plate_info["consecutive"] == self.consecutive_required:
                            try:
                                is_alert, alert_info = self.db.record_plate(
                                    best_text, best_conf, camera_id
                                )
                                if is_alert and alert_info:
                                    self._trigger_alert(best_text, alert_info)
                            except Exception:
                                pass
                        self.stats["plates_shown"] += 1
                        self.stats["confidences"].append(best_conf)
                        results.append({
                            "plate": best_text,
                            "confidence": best_conf,
                            "bbox": plate_bbox,
                            "vehicle_bbox": None,
                            "is_alert": bool(is_alert),
                            "alert_info": alert_info,
                        })

        # OCR conf 필터 + 중복 제거 (conf ≥ 0.30으로 완화 — 저해상도/2LINE 결과 보호)
        _before_filter = len(results)
        results = [r for r in results if r.get("confidence", 0) >= 0.30]
        results = self._deduplicate_results(results)
        # ★ 최종 변이 통합: 전역 히스토리에서 같은 숫자부의 최다 득표 텍스트로 강제 교체
        results = self._unify_plate_variants(results)
        if _before_filter > 0 and len(results) == 0:
            print(f"[FILTER] {_before_filter}개 결과 → conf<0.30 필터로 전부 제거됨!", flush=True)

        # ★ 이번 프레임에 없던 번호판은 연속 카운트 즉시 리셋 (잔상 방지)
        # grace period 제거: 미감지 즉시 consecutive=0 → 다음 차량에 구 번호판 표시 차단
        for key in list(self.recent_plates.keys()):
            if key not in seen_this_frame:
                self.recent_plates[key]["consecutive"] = 0

        # ★ 고스트 방지: 이번 프레임에 미감지된 트랙 → frames_absent 증가
        # 3프레임 연속 미감지 시 캐시 삭제 → 새 차량이 같은 위치 진입해도 구 트랙 재사용 안 함
        for _tk in list(self._ocr_track_cache.keys()):
            if _tk not in seen_track_keys:
                self._ocr_track_cache[_tk]["frames_absent"] = (
                    self._ocr_track_cache[_tk].get("frames_absent", 0) + 1
                )
                if self._ocr_track_cache[_tk]["frames_absent"] >= 3:
                    print(f"[GHOST] 트랙 만료: key={_tk} (3프레임 미감지)", flush=True)
                    del self._ocr_track_cache[_tk]
            else:
                self._ocr_track_cache[_tk]["frames_absent"] = 0

        # OCR 트랙 캐시: 오래된 엔트리 정리 (15개 초과 시 가장 오래된 것 제거)
        if len(self._ocr_track_cache) > 15:
            sorted_keys = sorted(self._ocr_track_cache.keys(),
                                 key=lambda k: self._ocr_track_cache[k].get("frame_since_ocr", 0),
                                 reverse=True)
            for k in sorted_keys[10:]:
                del self._ocr_track_cache[k]

        # 캐시 갱신 (프레임 스킵 시 재사용)
        self._cached_results = list(results)

        # 디버그: 최종 리턴 결과 추적
        if results:
            print(f"[ENGINE-RETURN] {len(results)}개 결과 → "
                  + ", ".join(f"{r['plate']}({r['confidence']:.2f})" for r in results),
                  flush=True)
        else:
            # 캐시 히트인 경우만 표시 안함 (매 프레임 출력 방지)
            if self._frame_counter % self._frame_skip_interval == 1:
                print(f"[ENGINE-RETURN] YOLO 실행했지만 결과 0개 (consecutive_req={self.consecutive_required})",
                      flush=True)

        return results

    # CRNN 한글 검증 (thin wrappers — 실 구현은 crnn_verifier.CrnnVerifier)
    def _load_crnn(self):
        """[deprecated] CrnnVerifier 가 __init__ 에서 자동 로드.

        과거 시그니처 보존용 호환 메서드 — 외부 호출처가 없으면 제거 가능.
        """
        return

    def _crnn_read_plate(self, roi, return_confidence=False):
        """CRNN 으로 번호판 ROI 읽기 (CrnnVerifier 위임)."""
        return self.crnn_verifier.read_plate(roi, return_confidence=return_confidence)

    def _verify_korean_with_crnn(self, paddle_text, roi, crnn_text=None, crnn_conf=None):
        """PaddleOCR 한글을 CRNN 결과로 교차검증 (CrnnVerifier 위임)."""
        return self.crnn_verifier.verify(paddle_text, roi, crnn_text, crnn_conf)

    # CTC 한글 필터: post_op monkey-patch
    _ctc_patched = False

    @classmethod
    def _patch_ctc_postop(cls, post_op):
        """CTCLabelDecode.__call__에 마스크를 주입 (monkey-patch).

        원본 파이프라인 그대로 유지, argmax 직전에만 마스크 적용.
        """
        if cls._ctc_patched:
            return

        # ★ 번호판 용도 한글: 가~후 전체 (초·추·코·쿠·토·투·포·푸·후·부 누락 수정)
        plate_hangul = set(
            '가거고구나너노누다더도두라러로루마머모무'
            '바배버보부사서소수아어오우자저조주'
            '차처초카커코타터토파퍼포하허호'
            '추쿠투푸후'
        )
        # ★ 지역명 한글: 구형 번호판 "서울12가3456" 등 지역 접두사 인식용
        region_hangul = set(
            '서울부산대구인천광주대전울산세종'
            '경기강원충북충남전북전남경북경남제주'
        )
        digits = set('0123456789')
        valid = plate_hangul | region_hangul | digits
        n = len(post_op.character)
        mask = np.full(n, -1e9, dtype=np.float32)
        mask[0] = 0.0  # blank
        for i, ch in enumerate(post_op.character):
            if ch in valid:
                mask[i] = 0.0
        cnt = int((mask == 0.0).sum())
        print(f"[CTC] 마스크 패치: {cnt}/{n} chars allowed", flush=True)

        _orig_call = post_op.__class__.__call__

        def _patched_call(self_post, pred, return_word_box=False, **kwargs):
            preds = np.array(pred[0])
            # ★ 마스크 적용: 유효 문자 외 -inf
            preds = preds + mask[np.newaxis, :]
            pred_patched = [preds]
            return _orig_call(self_post, pred_patched,
                              return_word_box=return_word_box, **kwargs)

        post_op.__class__.__call__ = _patched_call
        cls._ctc_patched = True

    def _run_ocr(self, engine_name, engine, image):
        """OCR 실행 (CTC 필터 적용 — monkey-patch 방식).
        기존 rec_model.predict() 파이프라인 그대로 사용,
        post_op 단계에서만 logits 마스킹.
        """
        import re as _re
        try:
            if engine_name == "paddleocr":
                # CTC 마스크 패치 (최초 1회)
                rec_model = engine.paddlex_pipeline.text_rec_model
                self._patch_ctc_postop(rec_model.post_op)

                # rec-only (기존과 동일, 내부에서 마스킹됨)
                rec_text, rec_score = "", 0.0
                try:
                    rec_results = list(rec_model.predict([image]))
                    if rec_results:
                        res = rec_results[0]
                        rec_text = res.get('rec_text', '')
                        rec_score = float(res.get('rec_score', 0))
                        if rec_text and rec_score >= 0.7 and _re.search(r'[가-힣]', rec_text):
                            return rec_text, rec_score
                except Exception:
                    pass

                if not rec_text or rec_score < 0.3 or len(rec_text.strip()) < 3:
                    if rec_text and rec_score > 0:
                        return rec_text, rec_score
                    return "", 0.0

                # Fallback: full det+rec
                try:
                    for res in engine.predict(image):
                        texts = res.get('rec_texts', [])
                        scores = res.get('rec_scores', [])
                        if texts:
                            text = "".join(texts)
                            conf = sum(scores) / len(scores) if scores else 0.0
                            if text:
                                return text, conf
                except Exception:
                    pass

                if rec_text:
                    return rec_text, rec_score
        except Exception as _e:
            print(f"[OCR-ERROR] {type(_e).__name__}: {_e}", flush=True)
        return "", 0.0

    def _trigger_alert(self, plate_number, alert_info):
        print("\n" + "=" * 50)
        print("🚨 [경고] 수배 차량 감지!")
        print(f"   번호판: {plate_number}")
        print(f"   유형: {alert_info[2] if alert_info else '미상'}")
        print(f"   시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50 + "\n")

    def process_video(self, source, camera_id="CAM01", show=True, save=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[에러] 영상 열기 실패: {source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save:
            out_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_count = 0
        total_plates = 0
        start_time = time.time()
        print(f"[시작] 영상 처리: {source} ({w}x{h} @ {fps}fps)")

        paused = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if not paused:
                frame_count += 1

                # process_frame 내부의 _frame_skip_interval이 YOLO 스킵 처리
                results = self.process_frame(frame, camera_id)
                total_plates += len(results)

                # 캐시된 결과로 OSD 그리기 (매 프레임)
                for r in results:
                    x1, y1, x2, y2 = r["bbox"]
                    color = (0, 0, 255) if r["is_alert"] else (0, 255, 0)
                    thickness = 3 if r["is_alert"] else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"{r['plate']} ({r['confidence']:.0%})"
                    frame = draw_korean_text(frame, label, (x1, y1 - 30), color, 24)
                    if r["is_alert"]:
                        cv2.putText(frame, "!! ALERT !!", (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                info = f"FPS: {current_fps:.1f} | Plates: {total_plates}"
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if writer:
                    writer.write(frame)
            if show:
                disp = cv2.resize(frame, (960, 540)) if frame.shape[1] > 960 else frame
                cv2.imshow("ANPR Pro", disp)
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):  # spacebar toggle pause
                    paused = not paused

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"\n[완료] {frame_count}프레임, {total_plates}대 인식, 평균 {frame_count/elapsed:.1f} FPS")


# ============================================
# [통합1] FastALPR → ONNX 고속 엔진 (출처: github.com/ankandrew/fast-alpr)
# ============================================
class PlateEngineFast:
    """FastALPR ONNX 고속 엔진. pip install fast-alpr[onnx-gpu] 필요."""

    def __init__(self):
        self._engine = None
        self._validator = PlateValidator()
        if HAS_FAST_ALPR and fast_alpr is not None:
            try:
                if hasattr(fast_alpr, "ALPR"):
                    self._engine = fast_alpr.ALPR()
                elif hasattr(fast_alpr, "Pipeline"):
                    self._engine = fast_alpr.Pipeline()
                elif callable(getattr(fast_alpr, "run", None)):
                    self._engine = fast_alpr
                else:
                    self._engine = None
            except Exception as e:
                print(f"[FastALPR] 초기화 실패: {e}")
                self._engine = None
        if self._engine is None and HAS_FAST_ALPR:
            print("[FastALPR] API 불일치. pip install fast-alpr[onnx-gpu] 후 문서 참조")
        elif not HAS_FAST_ALPR:
            print("[FastALPR] 미설치. pip install fast-alpr[onnx-gpu] 권장")

    @property
    def available(self):
        return self._engine is not None

    def process_frame(self, frame, camera_id="CAM01"):
        """프레임 처리 → [{plate, confidence, bbox}, ...] (Pro와 동일 형식)."""
        results = []
        if not self._engine:
            return results
        try:
            t0 = time.time()
            # fast_alpr 일반적 사용: run(frame) 또는 detect(frame)
            if hasattr(self._engine, "run"):
                raw = self._engine.run(frame)
            elif hasattr(self._engine, "detect"):
                raw = self._engine.detect(frame)
            else:
                raw = []
            elapsed_ms = (time.time() - t0) * 1000
            if not raw:
                return results
            # raw가 리스트 of (text, conf, box) 또는 dict 리스트 등으로 올 수 있음
            for item in (raw if isinstance(raw, list) else [raw]):
                if isinstance(item, dict):
                    text = item.get("plate", item.get("text", ""))
                    conf = float(item.get("confidence", item.get("conf", 0)))
                    bbox = item.get("bbox", item.get("box", [0, 0, 0, 0]))
                else:
                    text = str(item[0]) if len(item) > 0 else ""
                    conf = float(item[1]) if len(item) > 1 else 0
                    bbox = list(item[2]) if len(item) > 2 else [0, 0, 0, 0]
                clean = self._validator.clean_ocr_text(text)
                if not self._validator.is_valid_length(clean):
                    continue
                valid, final = self._validator.validate(clean)
                if valid and conf >= PlateEngineConfig.OCR_CONF:
                    results.append({
                        "plate": final,
                        "confidence": conf,
                        "bbox": bbox,
                        "is_alert": False,
                        "alert_info": None,
                        "engine": "Fast",
                    })
        except Exception as e:
            pass
        return results


def process_frame_unified(
    frame,
    camera_id="CAM01",
    engine_pro=None,
    engine_fast=None,
    engine_mode="pro",
    use_multiframe=False,
):
    """
    Pro / Fast 병렬 실행 후 engine_mode에 따라 결과 반환.
    engine_mode: "pro" | "fast" | "auto"(높은 confidence 채택)
    반환: (results, process_ms_pro, process_ms_fast)
    """
    results = []
    ms_pro, ms_fast = 0.0, 0.0
    pro_results, fast_results = [], []

    if engine_mode in ("pro", "auto") and engine_pro is not None:
        t0 = time.time()
        pro_results = engine_pro.process_frame(frame, camera_id, use_multiframe=use_multiframe)
        ms_pro = (time.time() - t0) * 1000

    if engine_mode in ("fast", "auto") and engine_fast is not None and getattr(engine_fast, "available", True):
        t0 = time.time()
        fast_results = engine_fast.process_frame(frame, camera_id)
        ms_fast = (time.time() - t0) * 1000

    if engine_mode == "pro":
        results = pro_results
    elif engine_mode == "fast":
        results = fast_results
    else:
        # auto: 동일 번호판이면 confidence 높은 것 채택
        by_plate = {}
        for r in pro_results:
            by_plate[r["plate"]] = {**r, "engine": "Pro", "ms": ms_pro}
        for r in fast_results:
            p = r["plate"]
            if p not in by_plate or r["confidence"] > by_plate[p]["confidence"]:
                by_plate[p] = {**r, "engine": "Fast", "ms": ms_fast}
        results = [by_plate[p] for p in by_plate]

    return results, ms_pro, ms_fast


# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ANPR Pro Engine")
    parser.add_argument("--input", default="0", help="영상 소스 (0=웹캠, 파일경로, rtsp)")
    parser.add_argument("--camera", default="CAM01", help="카메라 ID")
    parser.add_argument("--no-show", action="store_true", help="화면 표시 안 함")
    parser.add_argument("--no-save", action="store_true", help="결과 영상 저장 안 함")
    parser.add_argument("--alert-add", help="경고 목록에 번호판 추가")
    args = parser.parse_args()

    engine = PlateEnginePro()

    if args.alert_add:
        engine.db.add_alert(args.alert_add)
        print(f"[경고등록] {args.alert_add}")
    else:
        source = int(args.input) if args.input.isdigit() else args.input
        engine.process_video(source, args.camera, show=not args.no_show, save=not args.no_save)
