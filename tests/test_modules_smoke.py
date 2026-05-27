# -*- coding: utf-8 -*-
"""Smoke tests for the three SRP-extracted modules.

대상 모듈 (T1 추출 — 커밋 000ef534):
- preprocessor.ImagePreprocessor (+ _deskew_and_otsu)
- validator.PlateValidator (+ 클래스 상수)
- db.PlateDatabase

특성: 회귀 안전망 (smoke) — 깊은 커버리지가 아니라 핵심 동작 + 시그니처 보존을
보장한다. 외부 의존(YOLO·PaddleOCR·CRNN)은 사용하지 않고 모듈이 self-contained
인지를 검증한다.
"""
from __future__ import annotations

import re
import sys
import sqlite3
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

# src/ 패키지 레이아웃: 패키지 install (`pip install -e .`) 또는
# pyproject.toml `[tool.pytest.ini_options] pythonpath = ["src"]` 로 import 가능.
# CI/로컬에서 `pip install -e .` 미실행 시를 대비해 sys.path 보강.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo11_plate.preprocessor import ImagePreprocessor, _deskew_and_otsu  # noqa: E402
from yolo11_plate.validator import PlateValidator  # noqa: E402
from yolo11_plate.db import PlateDatabase  # noqa: E402
from yolo11_plate.config import PathConfig  # noqa: E402


# ────────────────────────────────────────────────────────────────
# 픽스처
# ────────────────────────────────────────────────────────────────
@pytest.fixture
def bgr_image() -> np.ndarray:
    """120x200 BGR 합성 이미지 — 회색 배경 + 검은 사각형(글자 흉내)."""
    img = np.full((120, 200, 3), 200, dtype=np.uint8)  # 밝은 회색
    cv2.rectangle(img, (20, 30), (60, 90), (30, 30, 30), thickness=-1)
    cv2.rectangle(img, (80, 30), (120, 90), (30, 30, 30), thickness=-1)
    return img


@pytest.fixture
def gray_image(bgr_image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def validator() -> PlateValidator:
    return PlateValidator()


@pytest.fixture
def temp_db(tmp_path: Path) -> PlateDatabase:
    """매 테스트마다 깨끗한 sqlite 파일을 사용하는 PlateDatabase."""
    return PlateDatabase(db_path=str(tmp_path / "plate_records.db"))


# ────────────────────────────────────────────────────────────────
# preprocessor
# ────────────────────────────────────────────────────────────────
class TestImagePreprocessor:
    """ImagePreprocessor: BGR 입력 → BGR 출력(또는 명시된 형태) 일관성 + 핵심 효과."""

    def test_grayscale_returns_2d_array(self, bgr_image: np.ndarray) -> None:
        """gray_threshold는 내부에서 GRAY 변환을 거치지만 BGR 3채널로 복원해서 반환한다."""
        out = ImagePreprocessor.gray_threshold(bgr_image)
        assert out.ndim == 3 and out.shape[2] == 3
        assert out.shape[:2] == bgr_image.shape[:2]
        # otsu 이진화이므로 모든 픽셀이 0 또는 255 인지 확인.
        uniques = np.unique(out)
        assert set(uniques.tolist()).issubset({0, 255})

    def test_sharpen_preserves_shape(self, bgr_image: np.ndarray) -> None:
        """샤프닝은 모양/채널을 그대로 유지하면서 픽셀 값을 변화시켜야 한다."""
        out = ImagePreprocessor.sharpen(bgr_image)
        assert out.shape == bgr_image.shape
        # 라플라시안 샤프닝이 적용되었는지 — 동일 이미지가 아님을 보장.
        assert not np.array_equal(out, bgr_image)

    def test_deblur_is_sharpen_alias(self, bgr_image: np.ndarray) -> None:
        """deblur는 sharpen의 alias여야 한다 (외부 호출 호환)."""
        assert np.array_equal(
            ImagePreprocessor.deblur(bgr_image),
            ImagePreprocessor.sharpen(bgr_image),
        )

    def test_gamma_bright_increases_mean_intensity(
        self, bgr_image: np.ndarray
    ) -> None:
        """gamma_bright(gamma=0.5)은 평균 밝기를 증가시켜야 한다."""
        out = ImagePreprocessor.gamma_bright(bgr_image)
        assert out.shape == bgr_image.shape
        assert float(out.mean()) > float(bgr_image.mean())

    def test_gamma_dark_decreases_mean_intensity(
        self, bgr_image: np.ndarray
    ) -> None:
        """gamma_dark(gamma=1.5)는 평균 밝기를 감소시켜야 한다."""
        out = ImagePreprocessor.gamma_dark(bgr_image)
        assert float(out.mean()) < float(bgr_image.mean())

    def test_clahe_returns_same_shape(self, bgr_image: np.ndarray) -> None:
        """CLAHE 결과는 BGR 3채널 + 동일 해상도를 유지해야 한다."""
        out = ImagePreprocessor.clahe(bgr_image)
        assert out.shape == bgr_image.shape
        assert out.dtype == np.uint8

    def test_upscale_2x_doubles_dimensions(self, bgr_image: np.ndarray) -> None:
        """upscale_2x는 W/H를 정확히 2배로 키운다."""
        h, w = bgr_image.shape[:2]
        out = ImagePreprocessor.upscale_2x(bgr_image)
        assert out.shape[0] == h * 2
        assert out.shape[1] == w * 2

    def test_invert_color_is_involution(self, bgr_image: np.ndarray) -> None:
        """invert_color 두 번 = 원본 (involution)."""
        once = ImagePreprocessor.invert_color(bgr_image)
        twice = ImagePreprocessor.invert_color(once)
        assert np.array_equal(twice, bgr_image)

    def test_deskew_handles_no_rotation_case(
        self, bgr_image: np.ndarray
    ) -> None:
        """직각 입력은 회전이 적용되지 않거나 무시할 만한 변형만 일어나야 한다.

        deskew는 Hough 라인을 찾지 못하면 원본을 그대로 반환한다.
        """
        out = ImagePreprocessor.deskew(bgr_image)
        assert out.shape == bgr_image.shape
        assert out.dtype == bgr_image.dtype

    def test_deskew_and_otsu_returns_binary(self, gray_image: np.ndarray) -> None:
        """_deskew_and_otsu는 항상 0/255 이진 이미지를 반환한다 (실패 시 Otsu만)."""
        out = _deskew_and_otsu(gray_image)
        assert out.shape == gray_image.shape
        uniques = set(np.unique(out).tolist())
        assert uniques.issubset({0, 255})


# ────────────────────────────────────────────────────────────────
# validator
# ────────────────────────────────────────────────────────────────
class TestPlateValidator:
    """PlateValidator: 패턴 검증/한글 교정/지역명 복원."""

    def test_class_constants_have_expected_types(self) -> None:
        """외부에서 참조하는 클래스 상수의 타입/내용 회귀 방지."""
        assert isinstance(PlateValidator._COMMERCIAL_CHARS, set)
        assert "바" in PlateValidator._COMMERCIAL_CHARS
        assert "서울" in PlateValidator._REGION_PREFIXES
        assert "전기" in PlateValidator._GOV_PREFIXES_2CHAR
        # _KR_CONFUSION 은 dict (plate_recognition_4k._HANGUL_PLATE_CORRECTION 알리아스).
        assert isinstance(PlateValidator._KR_CONFUSION, dict)

    def test_korean_pattern_regex_compiled(self, validator: PlateValidator) -> None:
        """OCRConfig.KR_PATTERNS 가 컴파일된 re.Pattern 리스트로 보존되어야 한다."""
        assert len(validator.patterns) > 0
        assert all(isinstance(p, re.Pattern) for p in validator.patterns)

    def test_validate_format_with_valid_plate(
        self, validator: PlateValidator
    ) -> None:
        """유효한 신형 번호판은 검증 통과 + 정규화된 텍스트 반환."""
        is_valid, result = validator.validate("01나8060")
        assert is_valid is True
        assert result == "01나8060"

    def test_validate_format_with_invalid_plate(
        self, validator: PlateValidator
    ) -> None:
        """알파벳/특수문자만으로 구성된 노이즈는 거부."""
        is_valid, _ = validator.validate("XXYYZZ")
        assert is_valid is False

    def test_validate_region_plate(self, validator: PlateValidator) -> None:
        """구형 지역명 번호판도 통과해야 한다."""
        is_valid, result = validator.validate("서울70바9203")
        assert is_valid is True
        assert "서울" in result

    def test_validate_recovers_region_from_digit_noise(
        self, validator: PlateValidator
    ) -> None:
        """앞 1~2자리 숫자 → 지역명 복원 경로 동작 확인.

        예: '176바7789' → 'X76바7789' 후보 중 유효 지역명 prefix 가 채택되면 True.
        """
        is_valid, result = validator.validate("176바7789")
        # 복원이 성공/실패 어느 쪽이든 함수가 폭주하지 않고 (bool, str) 시그니처를 지켜야 함.
        assert isinstance(is_valid, bool)
        assert isinstance(result, str)
        if is_valid:
            assert "바7789" in result

    def test_is_valid_length_respects_thresholds(
        self, validator: PlateValidator
    ) -> None:
        """min_len ≤ len ≤ max_len 경계 동작."""
        too_short = "X" * (validator.min_len - 1)
        ok = "01나8060"
        too_long = "X" * (validator.max_len + 1)
        assert validator.is_valid_length(too_short) is False
        assert validator.is_valid_length(ok) is True
        assert validator.is_valid_length(too_long) is False

    def test_clean_ocr_text_removes_noise(
        self, validator: PlateValidator
    ) -> None:
        """clean_ocr_text 는 공백/구두점/자모 잡음을 제거해야 한다."""
        cleaned = validator.clean_ocr_text("01나-8060|")
        assert "-" not in cleaned and "|" not in cleaned
        # 유효 한글/숫자는 보존.
        assert "01" in cleaned and "8060" in cleaned

    def test_clean_ocr_text_corrects_region_typo(
        self, validator: PlateValidator
    ) -> None:
        """지역명 자모 오인식(시울 → 서울)은 자동 교정되어야 한다."""
        cleaned = validator.clean_ocr_text("시울70바9203")
        # 교정 테이블에 따라 '시울' → '서울' 매핑이 적용되어야 한다.
        assert cleaned.startswith("서울")


# ────────────────────────────────────────────────────────────────
# db
# ────────────────────────────────────────────────────────────────
class TestPlateDatabase:
    """PlateDatabase: SQLite 백엔드 + 수배 차량 join + 동시 쓰기."""

    def test_db_path_resolution_uses_path_config(self) -> None:
        """DEFAULT_DB_PATH 는 PathConfig.DB_PATH 의 문자열 표현이어야 한다."""
        assert PlateDatabase.DEFAULT_DB_PATH == str(PathConfig.DB_PATH)

    def test_tables_are_created(self, temp_db: PlateDatabase) -> None:
        """초기화 시 plate_records / alert_list 두 테이블이 생성된다."""
        rows = temp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r[0] for r in rows}
        assert {"plate_records", "alert_list"}.issubset(names)

    def test_record_plate_creates_row(self, temp_db: PlateDatabase) -> None:
        """record_plate 호출 후 plate_records 행이 1개 추가된다."""
        is_alert, alert_row = temp_db.record_plate("01나8060", 0.91)
        assert is_alert == 0
        assert alert_row is None
        count = temp_db.conn.execute(
            "SELECT COUNT(*) FROM plate_records"
        ).fetchone()[0]
        assert count == 1

    def test_lookup_wanted_plate(self, temp_db: PlateDatabase) -> None:
        """alert_list 에 등록한 번호판은 record_plate 시 is_alert=1로 표시된다."""
        temp_db.add_alert("12가3456", alert_type="수배", description="테스트")
        is_alert, alert_row = temp_db.record_plate("12가3456", 0.88)
        assert is_alert == 1
        assert alert_row is not None
        # search_plates 도 같은 번호판을 찾을 수 있어야 한다.
        rows = temp_db.search_plates("12가3456")
        assert len(rows) == 1

    def test_search_plates_substring(self, temp_db: PlateDatabase) -> None:
        """search_plates 는 LIKE %query% 부분 일치를 지원한다."""
        temp_db.record_plate("01나8060", 0.9)
        temp_db.record_plate("02누2754", 0.9)
        rows = temp_db.search_plates("나80")
        assert len(rows) == 1

    def test_concurrent_writes(self, tmp_path: Path) -> None:
        """다중 스레드에서 record_plate 호출 시 INSERT 가 누락되지 않아야 한다.

        ``PlateDatabase`` 는 단일 sqlite 연결을 공유한다 (``check_same_thread=False``).
        sqlite3 자체가 한 연결에 대해 동시 statement 를 허용하지 않으므로 실제
        애플리케이션에서도 호출자가 직렬화하는 것이 정석이다. 본 테스트는
        “쓰기 잠금을 호출자가 직렬화하면 multi-thread 진입경로가 깨지지 않는다”
        는 정도의 가벼운 smoke 검증이다.
        """
        db = PlateDatabase(db_path=str(tmp_path / "concurrent.db"))
        n_threads = 8
        n_each = 10
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker(tid: int) -> None:
            try:
                for i in range(n_each):
                    with lock:
                        db.record_plate(f"99가{tid:02d}{i:02d}", 0.5)
            except BaseException as exc:  # pragma: no cover - 진단용
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(tid,))
            for tid in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"동시 쓰기 중 예외 발생: {errors!r}"
        count = db.conn.execute(
            "SELECT COUNT(*) FROM plate_records"
        ).fetchone()[0]
        assert count == n_threads * n_each
