# -*- coding: utf-8 -*-
"""CRNN 교차검증 책임 모듈 (Single Responsibility).

PlateEnginePro 의 한글 교차검증 로직을 캡슐화한 책임 클래스.

설계 의도
---------
- PaddleOCR: 숫자 정확, 한글 부정확
- CRNN(전용 79자 사전): 한글 정확, 과적합 위험
- 두 결과를 신뢰도 기반으로 합성 → 최적 한국 번호판 텍스트

원래 ``PlateEnginePro._verify_korean_with_crnn`` 안에 약 120줄로 존재하던
교차검증 로직을 ``CrnnVerifier.verify`` 로 추출하여 테스트 용이성과 응집도를
개선한다 (refactor: SRP). PaddleOCR / PlateValidator 같은 외부 모듈은
순수 함수 경로로만 import 하여 순환 의존을 피한다.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 외부 도메인 상수 (read-only 사용)
from plate_recognition_4k import _VALID_PLATE_HANGUL_ALL
from validator import PlateValidator

# 매직 넘버 상수
CRNN_INPUT_HEIGHT: int = 64          # CRNN 입력 표준 높이
CRNN_INPUT_MAX_WIDTH: int = 256      # CRNN 입력 최대 너비 (패딩 기준)
CRNN_CONF_MIN_TRUST: float = 0.70    # CRNN 결과 채택 최소 신뢰도
CRNN_CONF_KR_REPLACE: float = 0.90   # 한글 교체 임계값
CRNN_CONF_PREFIX_FIX: float = 0.88   # 앞자리 교정 임계값
CRNN_CONF_COMM_BODY: float = 0.75    # 영업용 body 매칭 임계값
SUFFIX_SEARCH_PAD: int = 5           # 뒤 4자리 검색 시 여유 자릿수


class CrnnVerifier:
    """CRNN 한글 교차검증 책임 클래스.

    PaddleOCR 결과의 한글 위치를 CRNN 결과와 비교하여 한국 번호판
    텍스트를 보정한다. 모델 로드 실패(파일 부재) 시에도 ``verify``
    는 PaddleOCR 결과를 그대로 반환한다 — 호출측 분기 불필요.

    Attributes
    ----------
    model : torch.nn.Module | None
        로드된 CRNN 추론 모델. 파일 없거나 로드 실패 시 ``None``.
    idx2char : dict[int, str]
        CTC index → 문자 매핑.
    vocab : set[str]
        모델 사전(디버깅용).
    """

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        """CRNN 체크포인트를 로드한다.

        Parameters
        ----------
        model_path:
            ``plate_ocr_crnn.pth`` 경로. 없으면 한글 검증 비활성화.
        device:
            ``torch.load`` map_location. 기본 ``"cpu"``.
        """
        self.model_path: Path = Path(model_path)
        self.device: str = device
        self.model = None  # torch.nn.Module | None
        self.idx2char: dict[int, str] = {}
        self.vocab: set[str] = set()
        self._load()

    # ─
    # public API
    # ─
    def read_plate(
        self,
        roi: np.ndarray,
        return_confidence: bool = False,
    ) -> str | tuple[str, float]:
        """CRNN 으로 번호판 ROI 를 읽는다.

        Parameters
        ----------
        roi:
            번호판 ROI(BGR 또는 GRAY).
        return_confidence:
            True 면 ``(text, avg_conf)`` 튜플 반환.
        """
        if self.model is None:
            return ("", 0.0) if return_confidence else ""
        return self._infer(roi, return_confidence)

    def verify(
        self,
        paddle_text: str,
        roi: np.ndarray,
        crnn_text: Optional[str] = None,
        crnn_conf: Optional[float] = None,
    ) -> str:
        """PaddleOCR 결과를 CRNN 결과로 교차검증한다.

        과적합 방어 2단(신뢰도 + 숫자 일치) 후 한글/앞자리 교정.

        Parameters
        ----------
        paddle_text:
            PaddleOCR 1차 결과 (예: ``"86다6118"``).
        roi:
            ``crnn_text`` 미제공 시 재추론에 사용할 ROI.
        crnn_text, crnn_conf:
            외부에서 미리 계산된 CRNN 결과(중복 호출 방지).
        """
        if self.model is None:
            return paddle_text

        m = re.match(r'^(\d{2,3})([가-힣])(\d{4})$', paddle_text)
        if not m:
            return paddle_text

        if crnn_conf is None:
            crnn_conf = 1.0
        if crnn_text is None:
            crnn_text, crnn_conf = self._infer(roi, return_confidence=True)
        if not crnn_text:
            return paddle_text

        # 과적합 방어 1: CRNN 신뢰도
        if crnn_conf < CRNN_CONF_MIN_TRUST:
            print(
                f"[CRNN-SKIP] 신뢰도 부족 {crnn_conf:.2f}<{CRNN_CONF_MIN_TRUST:.2f}, "
                f"PaddleOCR 유지: {paddle_text}",
                flush=True,
            )
            return paddle_text

        # 과적합 방어 2: 숫자 교차검증
        paddle_digits = m.group(1) + m.group(3)
        crnn_digits = re.sub(r'[^0-9]', '', crnn_text)
        if crnn_digits and paddle_digits:
            paddle_prefix = m.group(1)
            paddle_suffix = m.group(3)
            crnn_has_prefix = paddle_prefix in crnn_digits[: len(paddle_prefix) + 1]
            crnn_has_suffix = paddle_suffix in crnn_digits[-SUFFIX_SEARCH_PAD:]
            if not crnn_has_prefix and not crnn_has_suffix:
                print(
                    f"[CRNN-SKIP] 숫자 불일치 paddle={paddle_digits} "
                    f"crnn={crnn_digits}, PaddleOCR 유지: {paddle_text}",
                    flush=True,
                )
                return paddle_text

        # 한글 추출
        mc = re.match(r'^\d{2,3}([가-힣])\d{4}$', crnn_text)
        if not mc:
            crnn_hangul = [ch for ch in crnn_text if '가' <= ch <= '힣']

            if len(crnn_hangul) >= 2:
                # 2줄 번호판 복원 분기
                restored = self._try_two_line_restore(
                    paddle_text, crnn_text, crnn_conf, m
                )
                if restored is not None:
                    return restored
                return paddle_text

            if len(crnn_hangul) != 1:
                return paddle_text
            crnn_kr = crnn_hangul[0]
        else:
            crnn_kr = mc.group(1)

        paddle_kr = m.group(2)
        # 한글 교정 (conf ≥ 0.90)
        kr_conf_ok = crnn_conf and crnn_conf >= CRNN_CONF_KR_REPLACE
        corrected_kr = (
            crnn_kr
            if (kr_conf_ok and crnn_kr != paddle_kr and crnn_kr in _VALID_PLATE_HANGUL_ALL)
            else paddle_kr
        )
        if corrected_kr != paddle_kr:
            print(
                f"[CRNN-VERIFY] {paddle_kr}→{corrected_kr} "
                f"(crnn={crnn_text}, conf={crnn_conf:.2f})",
                flush=True,
            )

        # 앞자리 prefix 교정 (conf ≥ 0.88)
        corrected_prefix = m.group(1)
        if mc and crnn_conf and crnn_conf >= CRNN_CONF_PREFIX_FIX:
            crnn_prefix_m = re.match(r'^(\d{2,3})[가-힣]\d{4}$', crnn_text)
            if crnn_prefix_m:
                crnn_prefix = crnn_prefix_m.group(1)
                crnn_suffix_m = re.search(r'(\d{4})$', crnn_text)
                paddle_suffix = m.group(3)
                if (
                    crnn_suffix_m
                    and crnn_suffix_m.group(1) == paddle_suffix
                    and crnn_prefix != corrected_prefix
                    and len(crnn_prefix) == len(corrected_prefix)
                ):
                    print(
                        f"[CRNN-PREFIX] 앞자리 교정: {corrected_prefix}→{crnn_prefix} "
                        f"(뒤4자리 '{paddle_suffix}' 일치, conf={crnn_conf:.2f})",
                        flush=True,
                    )
                    corrected_prefix = crnn_prefix

        result = corrected_prefix + corrected_kr + m.group(3)
        return result if result != paddle_text else paddle_text

    # ─
    # private helpers
    # ─
    def _load(self) -> None:
        """체크포인트 로드 (실패 시 self.model=None 유지)."""
        import torch
        import torch.nn as nn

        if not self.model_path.exists():
            print("[CRNN] plate_ocr_crnn.pth 없음 — 한글 검증 비활성화")
            return
        try:
            ckpt = torch.load(
                str(self.model_path), map_location=self.device, weights_only=False
            )
            self.idx2char = {int(k): v for k, v in ckpt["idx2char"].items()}
            self.vocab = set(ckpt["vocab"])
            nc = ckpt["num_classes"]
            hid = ckpt["hidden_size"]
            nl = ckpt["num_layers"]

            class _CRNN(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.cnn = nn.Sequential(
                        nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
                        nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
                        nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                        nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                        nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                        nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                        nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                        nn.Conv2d(512, 512, (2, 1), 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
                    )
                    s.rnn = nn.LSTM(
                        512, hid, nl, bidirectional=True, batch_first=True, dropout=0.2
                    )
                    s.fc = nn.Linear(hid * 2, nc)

                def forward(s, x):
                    c = s.cnn(x).squeeze(2).permute(0, 2, 1)
                    r, _ = s.rnn(c)
                    return s.fc(r).permute(1, 0, 2)

            m = _CRNN()
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            self.model = m
            print(f"[CRNN] 로드 완료: {nc} classes, acc={ckpt.get('accuracy', '?')}")
        except Exception as e:  # pragma: no cover — 환경 의존
            print(f"[CRNN] 로드 실패: {e}")

    def _infer(
        self, roi: np.ndarray, return_confidence: bool
    ) -> str | tuple[str, float]:
        """CRNN 순전파 + CTC 디코딩."""
        import torch

        h, w = roi.shape[:2]
        ratio = CRNN_INPUT_HEIGHT / h
        nw = min(int(w * ratio), CRNN_INPUT_MAX_WIDTH)
        resized = cv2.resize(roi, (nw, CRNN_INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
        gray = (
            cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if len(resized.shape) == 3
            else resized
        )
        if nw < CRNN_INPUT_MAX_WIDTH:
            pad = np.ones((CRNN_INPUT_HEIGHT, CRNN_INPUT_MAX_WIDTH - nw), dtype=np.uint8) * 255
            gray = np.concatenate([gray, pad], axis=1)
        t = torch.FloatTensor(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self.model(t)
        probs = torch.nn.functional.softmax(out, dim=2)
        max_probs, preds = probs.max(2)
        preds = preds.transpose(0, 1)
        max_probs = max_probs.transpose(0, 1)

        chars: list[str] = []
        char_confs: list[float] = []
        prev = -1
        for ti in range(preds.size(1)):
            p = preds[0, ti].item()
            if p != 0 and p != prev and p in self.idx2char:
                chars.append(self.idx2char[p])
                char_confs.append(max_probs[0, ti].item())
            prev = p
        text = "".join(chars)
        if return_confidence:
            avg_conf = sum(char_confs) / len(char_confs) if char_confs else 0.0
            return text, avg_conf
        return text

    @staticmethod
    def _try_two_line_restore(
        paddle_text: str,
        crnn_text: str,
        crnn_conf: float,
        paddle_match: re.Match,
    ) -> Optional[str]:
        """2줄 번호판(구형/영업용) 복원 시도.

        Returns
        -------
        restored or None — 복원 성공 시 새 텍스트, 실패 시 None.
        """
        p_tail = paddle_match.group(2) + paddle_match.group(3)
        c_tail_m = re.search(r'[가-힣]\d{4}$', crnn_text)
        if not (c_tail_m and c_tail_m.group() == p_tail):
            return None

        # (A) 구형 지역: "경기76바7789", "충남86아6118"
        cr_region_m = re.match(r'^([가-힣]{2,3})\d{2}[가-힣]\d{4}$', crnn_text)
        if cr_region_m and cr_region_m.group(1) in PlateValidator._REGION_PREFIXES:
            print(
                f"[CRNN-2LINE] 구형 복원: {paddle_text} → {crnn_text} "
                f"(conf={crnn_conf:.2f})",
                flush=True,
            )
            return crnn_text

        # (B) 영업용 1글자 지역: "충86다6118"
        if re.fullmatch(r'[가-힣]\d{2}[가-힣]\d{4}', crnn_text):
            print(
                f"[CRNN-COMM] 영업용 복원: {paddle_text} → {crnn_text} "
                f"(conf={crnn_conf:.2f})",
                flush=True,
            )
            return crnn_text

        # (C) 영업용 body 매칭: PaddleOCR "586다6118" → CRNN "충86다6118"
        if len(paddle_match.group(1)) == 3:
            crnn_comm_m = re.match(r'^([가-힣])(\d{2}[가-힣]\d{4})$', crnn_text)
            if crnn_comm_m:
                paddle_body = (
                    paddle_match.group(1)[1:]
                    + paddle_match.group(2)
                    + paddle_match.group(3)
                )
                if (
                    crnn_comm_m.group(2) == paddle_body
                    and crnn_conf >= CRNN_CONF_COMM_BODY
                ):
                    print(
                        f"[CRNN-COMM] 영업용 body: {paddle_text} → {crnn_text} "
                        f"(conf={crnn_conf:.2f})",
                        flush=True,
                    )
                    return crnn_text
        return None
