#!/usr/bin/env python3
"""
YOLO26 plate_gui.py 패치 스크립트
- MareArts ANPR 스타일 오버레이
- 탐지 결과 2초 유지 (잔상 방지)
- 좌상단 FPS/해상도 표시

사용법: cd /c/tool/yolo26-main && py patch_marearts_style.py
"""
import re
import shutil
import os

GUI_PATH = "plate_gui.py"

if not os.path.exists(GUI_PATH):
    print(f"[ERROR] {GUI_PATH} 파일을 찾을 수 없습니다.")
    exit(1)

# 백업
shutil.copy2(GUI_PATH, GUI_PATH + ".bak_marearts")
print(f"[OK] 백업 생성: {GUI_PATH}.bak_marearts")

with open(GUI_PATH, "r", encoding="utf-8") as f:
    code = f.read()

# ═══════════════════════════════════════════════
# 패치 1: _refresh_display_inner에서 탐지 결과 2초 유지
# ═══════════════════════════════════════════════
# "self._latest_detections = tracked" 를 시간 기반 유지로 교체

old_det_update = "self._latest_detections = tracked"

new_det_update = """# ★ MareArts 스타일: 탐지 결과 2초 유지 (빈 결과로 바로 안 사라짐)
                    import time as _t
                    if tracked:
                        self._latest_detections = tracked
                        self._last_det_time = _t.time()
                    else:
                        # 마지막 탐지 후 2초 지나면 지우기
                        if hasattr(self, '_last_det_time') and _t.time() - self._last_det_time > 2.0:
                            self._latest_detections = []"""

if old_det_update in code:
    # "if tracked: self._latest_detections = tracked" 형태도 체크
    code = code.replace(old_det_update, new_det_update)
    print("[OK] 패치 1: 탐지 결과 2초 유지 적용")
elif "if tracked: self._latest_detections = tracked" in code:
    code = code.replace("if tracked: self._latest_detections = tracked", new_det_update)
    print("[OK] 패치 1: 탐지 결과 2초 유지 적용 (if tracked 버전)")
else:
    print("[SKIP] 패치 1: 이미 적용되었거나 코드 구조 다름")


# ═══════════════════════════════════════════════
# 패치 2: _draw_overlay를 MareArts ANPR 스타일로 완전 교체
# ═══════════════════════════════════════════════

# 기존 _draw_overlay 함수를 찾아서 교체
# "def _draw_overlay" 부터 다음 "def " 또는 "# ─── Detection Log" 전까지
pattern = r'(    def _draw_overlay\(self.*?\n)(.*?)(\n    # ─── Detection Log)'
match = re.search(pattern, code, re.DOTALL)

if match:
    new_draw_overlay = '''    def _draw_overlay(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """MareArts ANPR 스타일 오버레이
        - 좌상단: YOLO26 ANPR + FPS + 해상도
        - 초록 bbox + 번호판 크롭 이미지
        - 라벨: 48보7639 D89/R99 (Detection/Recognition conf)
        - 최대 5대 동시 표시
        """
        result = frame.copy()
        h, w = result.shape[:2]

        # ── 좌상단 정보 표시 ──
        fps = 0
        if hasattr(self, '_det_fps_samples') and self._det_fps_samples:
            fps = sum(self._det_fps_samples) / len(self._det_fps_samples)
        result = draw_korean_text(result, "YOLO26 ANPR", (10, 10),
                                  font_size=22, color=(255, 255, 255))
        result = draw_korean_text(result, f"FPS: {fps:.1f} {w}x{h}", (10, 38),
                                  font_size=18, color=(0, 255, 255))

        # ── OCR 확정 필터 ──
        phase2_dets = []
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            text = det.get("text") or det.get("plate", "")
            is_valid = det.get("is_valid_plate", False)
            if is_valid and text:
                phase2_dets.append(det)

        if not phase2_dets:
            return result

        # 신뢰도 내림차순 정렬 후 상위 5개
        phase2_dets.sort(key=lambda d: d.get("ocr_confidence", d.get("confidence", 0)), reverse=True)
        phase2_dets = phase2_dets[:5]

        for det in phase2_dets:
            bbox = det.get("bbox", [])
            x1, y1, x2, y2 = [int(v) for v in bbox]
            text = det.get("text") or det.get("plate", "")
            det_conf = det.get("pattern_score", det.get("ocr_confidence", 0))
            ocr_conf = det.get("ocr_confidence", det.get("confidence", 0))

            # D/R 값 (0~99 스케일)
            d_val = min(99, int(det_conf * 100))
            r_val = min(99, int(ocr_conf * 100))

            # ── 초록 bbox (MareArts 스타일: 두꺼운 선) ──
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ── 번호판 크롭 이미지를 bbox 아래에 표시 ──
            plate_img = det.get("plate_image", None)
            crop_bottom = y2 + 4
            if plate_img is not None and plate_img.size > 0:
                crop_h = max(35, min(50, y2 - y1 + 10))
                aspect = plate_img.shape[1] / max(plate_img.shape[0], 1)
                crop_w = int(crop_h * aspect)
                crop_w = max(crop_w, 80)  # 최소 너비
                try:
                    crop_resized = cv2.resize(plate_img, (crop_w, crop_h))
                    # 흰색 테두리
                    cv2.rectangle(result, (x1-1, crop_bottom-1),
                                  (x1+crop_w+1, crop_bottom+crop_h+1), (255,255,255), 1)
                    # 화면 범위 체크
                    if crop_bottom + crop_h < h and x1 + crop_w < w and x1 >= 0:
                        result[crop_bottom:crop_bottom+crop_h, x1:x1+crop_w] = crop_resized
                        crop_bottom += crop_h + 2
                except Exception:
                    pass

            # ── MareArts 스타일 라벨: "48보7639 D89/R99" ──
            label = f"{text} D{d_val}/R{r_val}"
            font_sz = max(20, min(28, (x2 - x1) // 3))
            label_y = max(0, y1 - font_sz - 8)
            result = draw_korean_text(result, label, (x1, label_y),
                                      font_size=font_sz, color=(0, 255, 0))

        return result
'''
    code = code[:match.start()] + new_draw_overlay + "\n    # ─── Detection Log" + code[match.end():]
    print("[OK] 패치 2: MareArts ANPR 스타일 _draw_overlay 적용")
else:
    print("[SKIP] 패치 2: _draw_overlay 패턴 매치 실패 — 수동 확인 필요")


# ═══════════════════════════════════════════════
# 패치 3: MAX_PHASE2_DISPLAY가 없으면 추가
# ═══════════════════════════════════════════════
if "MAX_PHASE2_DISPLAY" not in code:
    # __init__에서 추가
    init_pattern = "self._latest_detections = []"
    if init_pattern in code:
        code = code.replace(
            init_pattern,
            init_pattern + "\n        self.MAX_PHASE2_DISPLAY = 5\n        self._last_det_time = 0"
        )
        print("[OK] 패치 3: MAX_PHASE2_DISPLAY=5 추가")
else:
    print("[SKIP] 패치 3: MAX_PHASE2_DISPLAY 이미 존재")


# 저장
with open(GUI_PATH, "w", encoding="utf-8") as f:
    f.write(code)

print(f"\n[완료] {GUI_PATH} 패치 적용됨")
print("실행: py plate_gui.py \"C:/tool/yolo26-main/movie/hiway.mp4\"")
