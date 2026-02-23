# -*- coding: utf-8 -*-
# ============================================
# plate_engine_pro.py
# 상용급 번호판 인식 엔진 (비젼인급 품질)
# ============================================

import re
import time
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# ── OCR 엔진 임포트 ──
try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


class PlateEngineConfig:
    """엔진 설정"""
    # ── 모델 경로 ──
    YOLO_MODEL = "runs/plate_detect/yolo26_plate_v1/weights/best.pt"
    YOLO_FALLBACK = "yolo26.pt"

    # ── 인식 임계값 ──
    DETECT_CONF = 0.45
    OCR_CONF = 0.60

    # ── 한국 번호판 정규식 패턴 ──
    KR_PATTERNS = [
        r'^\d{2,3}[가-힣]\d{4}$',
        r'^[가-힣]{2}\d{2}[가-힣]\d{4}$',
        r'^[가-힣]{2}\d{2}[바사아자]\d{4}$',
        r'^외교\d{3}-?\d{3}$',
        r'^[가-힣]{2}\d{3}[가-힣]$',
    ]

    DB_PATH = "plate_records.db"

    PREPROCESS_METHODS = [
        "original",
        "gray_threshold",
        "adaptive_threshold",
        "clahe",
        "deblur",
        "gamma_bright",
        "gamma_dark",
        "bilateral",
        "morphology",
        "deskew",
    ]


class ImagePreprocessor:
    """10종 이미지 전처리 파이프라인"""

    @staticmethod
    def gray_threshold(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adaptive_threshold(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def clahe(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def deblur(img):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def gamma_bright(img, gamma=0.5):
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    @staticmethod
    def gamma_dark(img, gamma=1.5):
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    @staticmethod
    def bilateral(img):
        return cv2.bilateralFilter(img, 11, 75, 75)

    @staticmethod
    def morphology(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def deskew(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 30:
                    angles.append(angle)
            if angles:
                median_angle = np.median(angles)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
                return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return img


class PlateValidator:
    """번호판 유효성 검증기"""

    def __init__(self):
        self.patterns = [re.compile(p) for p in PlateEngineConfig.KR_PATTERNS]

    def validate(self, text):
        clean = re.sub(r"[\s\-\.]", "", text)
        for pattern in self.patterns:
            if pattern.match(clean):
                return True, clean
        return False, clean

    def clean_ocr_text(self, text):
        replacements = {
            "O": "0", "I": "1", "Z": "2", "S": "5",
            "B": "8", "D": "0", "Q": "0", "G": "6",
            "ㅇ": "0", "ㅣ": "1",
        }
        clean = text.strip()
        result = []
        for i, ch in enumerate(clean):
            if ch in replacements and self._should_be_digit(clean, i):
                result.append(replacements[ch])
            else:
                result.append(ch)
        return "".join(result)

    def _should_be_digit(self, text, pos):
        if pos > 0 and text[pos - 1].isdigit():
            return True
        if pos < len(text) - 1 and text[pos + 1].isdigit():
            return True
        return False


class PlateDatabase:
    """번호판 기록 데이터베이스"""

    def __init__(self, db_path=None):
        db_path = db_path or PlateEngineConfig.DB_PATH
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS plate_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                camera_id TEXT,
                image_path TEXT,
                vehicle_type TEXT,
                vehicle_color TEXT,
                speed_estimate REAL,
                direction TEXT,
                is_alert INTEGER DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_list (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT UNIQUE NOT NULL,
                alert_type TEXT,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_number ON plate_records(plate_number)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON plate_records(timestamp)")
        self.conn.commit()

    def record_plate(self, plate_number, confidence, camera_id="CAM01",
                     image_path=None, vehicle_type=None, vehicle_color=None):
        alert = self.conn.execute(
            "SELECT * FROM alert_list WHERE plate_number=?",
            (plate_number,)
        ).fetchone()
        is_alert = 1 if alert else 0
        self.conn.execute("""
            INSERT INTO plate_records
            (plate_number, confidence, camera_id, image_path,
             vehicle_type, vehicle_color, is_alert)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (plate_number, confidence, camera_id, image_path,
              vehicle_type, vehicle_color, is_alert))
        self.conn.commit()
        return is_alert, alert

    def add_alert(self, plate_number, alert_type="수배", description=""):
        self.conn.execute("""
            INSERT OR REPLACE INTO alert_list
            (plate_number, alert_type, description)
            VALUES (?, ?, ?)
        """, (plate_number, alert_type, description))
        self.conn.commit()

    def search_plates(self, query, limit=100):
        return self.conn.execute("""
            SELECT * FROM plate_records
            WHERE plate_number LIKE ?
            ORDER BY timestamp DESC LIMIT ?
        """, (f"%{query}%", limit)).fetchall()


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
        print(f"[엔진] YOLO 모델 로드: {model_path}")

        self.ocr_engines = {}
        if HAS_PADDLE:
            self.ocr_engines["paddle"] = PaddleOCR(
                use_angle_cls=True, lang="korean",
                show_log=False, use_gpu=True
            )
        if HAS_EASYOCR:
            self.ocr_engines["easyocr"] = easyocr.Reader(["ko", "en"], gpu=True)
        print(f"[엔진] OCR 엔진: {list(self.ocr_engines.keys())}")

        self.recent_plates = defaultdict(lambda: {"count": 0, "last_seen": 0})
        self.DUPLICATE_THRESHOLD = 3.0

    def process_frame(self, frame, camera_id="CAM01"):
        results = []
        detections = self.model(frame, conf=self.config.DETECT_CONF, verbose=False)

        for det in detections[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0])

            h, w = frame.shape[:2]
            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.15)
            rx1 = max(0, x1 - margin_x)
            ry1 = max(0, y1 - margin_y)
            rx2 = min(w, x2 + margin_x)
            ry2 = min(h, y2 + margin_y)
            roi = frame[ry1:ry2, rx1:rx2]

            if roi.size == 0:
                continue

            roi_h, roi_w = roi.shape[:2]
            if roi_w < 200:
                scale = 200 / roi_w
                roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            best_text = ""
            best_conf = 0.0

            for method in self.config.PREPROCESS_METHODS:
                try:
                    if method == "original":
                        processed = roi.copy()
                    else:
                        proc_func = getattr(self.preprocessor, method)
                        processed = proc_func(roi.copy())

                    for engine_name, engine in self.ocr_engines.items():
                        text, ocr_conf = self._run_ocr(engine_name, engine, processed)
                        if text and ocr_conf > best_conf:
                            cleaned = self.validator.clean_ocr_text(text)
                            is_valid, final_text = self.validator.validate(cleaned)
                            if is_valid and ocr_conf > best_conf:
                                best_text = final_text
                                best_conf = ocr_conf
                except Exception:
                    continue

            if best_text and best_conf >= self.config.OCR_CONF:
                now = time.time()
                plate_info = self.recent_plates[best_text]
                if now - plate_info["last_seen"] > self.DUPLICATE_THRESHOLD:
                    plate_info["count"] += 1
                    plate_info["last_seen"] = now
                    is_alert, alert_info = self.db.record_plate(
                        best_text, best_conf, camera_id
                    )
                    results.append({
                        "plate": best_text,
                        "confidence": best_conf,
                        "bbox": [x1, y1, x2, y2],
                        "is_alert": bool(is_alert),
                        "alert_info": alert_info,
                    })
                    if is_alert:
                        self._trigger_alert(best_text, alert_info)

        return results

    def _run_ocr(self, engine_name, engine, image):
        try:
            if engine_name == "paddle":
                result = engine.ocr(image, cls=True)
                if result and result[0]:
                    texts = [line[1][0] for line in result[0]]
                    confs = [line[1][1] for line in result[0]]
                    return "".join(texts), float(np.mean(confs))
            elif engine_name == "easyocr":
                result = engine.readtext(image)
                if result:
                    texts = [r[1] for r in result]
                    confs = [r[2] for r in result]
                    return "".join(texts), float(np.mean(confs))
        except Exception:
            pass
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            results = self.process_frame(frame, camera_id)
            total_plates += len(results)

            for r in results:
                x1, y1, x2, y2 = r["bbox"]
                color = (0, 0, 255) if r["is_alert"] else (0, 255, 0)
                thickness = 3 if r["is_alert"] else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f"{r['plate']} ({r['confidence']:.0%})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
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
                cv2.imshow("ANPR Pro", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"\n[완료] {frame_count}프레임, {total_plates}대 인식, 평균 {frame_count/elapsed:.1f} FPS")


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
