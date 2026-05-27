"""ui_text.py — 캐시 기반 한글 텍스트 오버레이 도메인 모듈.

plate_engine_pro.py에서 분리한 `draw_korean_text` + `_kr_text_cache`.
PIL 텍스트 렌더링 결과를 (RGBA→BGR + alpha 채널) 형태로 캐싱하여
프레임마다 폰트를 새로 그리는 오버헤드를 제거한다.

시그니처는 plate_engine_pro.py 호출부와 호환:
    draw_korean_text(frame, text, pos, color=(0,255,0), size=24)

plate_gui.py의 `draw_korean_text`는 외곽선/배경 옵션을 가진 별도 구현이므로
이 모듈로 통합하지 않는다 (시그니처 불일치).
"""

from __future__ import annotations

from typing import Final

import cv2  # noqa: F401  (이 모듈은 cv2 직접 호출은 없지만 다른 호출자가 import 순서를 가정)
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 매직 넘버 Final 상수

_DEFAULT_FONT_FACE: Final[str] = "malgun.ttf"
_TEXT_BOX_VERTICAL_PAD: Final[int] = 10  # 텍스트 박스 세로 여백(px)


# 캐시 (프로세스 전역 — 폰트 로드/텍스트 렌더링 결과 공유)

_kr_font_cache: dict = {}
_kr_text_cache: dict = {}


def draw_korean_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 0),
    size: int = 24,
) -> np.ndarray:
    """BGR 프레임 위에 한글 텍스트를 캐시 기반으로 렌더링한다.

    PIL로 한 번 그린 텍스트 RGBA 결과를 alpha 채널과 함께 캐싱하여
    이후 동일 (text, color, size) 호출 시 numpy 알파 블렌딩만 수행한다.
    """
    cache_key = (text, color, size)
    if cache_key in _kr_text_cache:
        tmp_np, alpha = _kr_text_cache[cache_key]
    else:
        if size not in _kr_font_cache:
            try:
                _kr_font_cache[size] = ImageFont.truetype(_DEFAULT_FONT_FACE, size)
            except Exception:
                _kr_font_cache[size] = ImageFont.load_default()
        font = _kr_font_cache[size]
        b, g, r = color
        tmp = Image.new("RGBA", (len(text) * size, size + _TEXT_BOX_VERTICAL_PAD), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tmp)
        draw.text((0, 0), text, font=font, fill=(r, g, b, 255))
        tmp_np = np.array(tmp)
        alpha = tmp_np[:, :, 3:4].astype(np.float32) / 255.0
        # BGR 채널 순서 변환 (PIL RGB → OpenCV BGR)
        tmp_np = tmp_np[:, :, :3][:, :, ::-1].astype(np.float32)
        _kr_text_cache[cache_key] = (tmp_np, alpha)
    x, y = int(pos[0]), int(pos[1])
    h, w = tmp_np.shape[:2]
    fh, fw = frame.shape[:2]
    y = max(0, min(y, fh - h))
    x = max(0, min(x, fw - w))
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (alpha * tmp_np + (1 - alpha) * roi).astype(np.uint8)
    return frame


__all__ = ["draw_korean_text"]
