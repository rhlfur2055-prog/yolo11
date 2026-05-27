# -*- coding: utf-8 -*-
"""SQLite-backed plate record / alert database.

Extracted from plate_engine_pro.py (refactor — SRP 분리).
공개 API: PlateDatabase 클래스. 외부에서 ``from db import PlateDatabase``로 사용.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional

from .config import PathConfig


class PlateDatabase:
    """번호판 기록 + 수배차량(alert) 테이블 관리.

    한 SQLite 파일에 두 테이블(``plate_records``, ``alert_list``)을 두고,
    record_plate 시 자동으로 수배 차량 여부를 join 검사한다.
    """

    DEFAULT_DB_PATH: str = str(PathConfig.DB_PATH)

    def __init__(self, db_path: Optional[str] = None) -> None:
        path = db_path or self.DEFAULT_DB_PATH
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
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
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_plate_number ON plate_records(plate_number)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON plate_records(timestamp)"
        )
        self.conn.commit()

    def record_plate(
        self,
        plate_number: str,
        confidence: float,
        camera_id: str = "CAM01",
        image_path: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        vehicle_color: Optional[str] = None,
    ) -> tuple[int, Optional[Any]]:
        alert = self.conn.execute(
            "SELECT * FROM alert_list WHERE plate_number=?",
            (plate_number,),
        ).fetchone()
        is_alert = 1 if alert else 0
        self.conn.execute(
            """
            INSERT INTO plate_records
            (plate_number, confidence, camera_id, image_path,
             vehicle_type, vehicle_color, is_alert)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plate_number, confidence, camera_id, image_path,
                vehicle_type, vehicle_color, is_alert,
            ),
        )
        self.conn.commit()
        return is_alert, alert

    def add_alert(
        self,
        plate_number: str,
        alert_type: str = "수배",
        description: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO alert_list
            (plate_number, alert_type, description)
            VALUES (?, ?, ?)
            """,
            (plate_number, alert_type, description),
        )
        self.conn.commit()

    def search_plates(self, query: str, limit: int = 100) -> list[Any]:
        return self.conn.execute(
            """
            SELECT * FROM plate_records
            WHERE plate_number LIKE ?
            ORDER BY timestamp DESC LIMIT ?
            """,
            (f"%{query}%", limit),
        ).fetchall()
