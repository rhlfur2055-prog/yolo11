"""build_dashboard.py — YOLO26 증거 대시보드 HTML 생성기.

evidence_output/ 안의 모든 증거 패키지를 스캔하여
브라우저에서 바로 열 수 있는 단일 HTML 파일을 생성한다.

지원 시나리오:
    - GoldenTime 2.0: 긴급차량 통행방해 채증 (evidence_YYYYMMDD_...)
    - SafePlate 4K:   물피도주 의심 차량 채증 (safeplate_YYYYMMDD_...)

리팩터링 메모(2026-05):
    - PEP 8 + 타입 힌트 + Guard Clause
    - 매직 넘버/문자열 상수화, CSS/JS 모듈 상수로 분리
    - dataclass(DashboardStats / DashboardTheme) 도입
    - 카드 공통부(meta row, 스크린샷, 제출 대상, 원본 데이터 탭) 헬퍼 추출
    - argparse CLI 도입 (기본 경로는 기존과 동일)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final


# ─
# 모듈 상수
# ─
DEFAULT_BASE_DIR: Final[Path] = Path("C:/tool/yolo26-main/evidence_output")
DEFAULT_OUTPUT_FILE: Final[Path] = Path(
    "C:/tool/yolo26-main/evidence_dashboard.html"
)

MAX_PACKAGES_DEFAULT: Final[int] = 20
IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png")
BYTES_PER_KB: Final[int] = 1024
BYTES_PER_MB: Final[int] = 1024 * 1024

FOREIGN_PLATE_TAG: Final[str] = "FOREIGN_PLATE"
SAFEPLATE_PREFIX: Final[str] = "safeplate_"
DAMAGE_HIT_RUN_SCENARIO: Final[str] = "물피도주"

DEPARTURE_DIRECTION_LABELS: Final[dict[str, str]] = {
    "left": "좌측 이탈",
    "right": "우측 이탈",
    "top": "상방 이탈",
    "bottom": "하방 이탈",
    "vanished": "화면 소멸",
}

THEME_TITLE_BOTH: Final[str] = "YOLO26 통합 증거 대시보드"
THEME_SUBTITLE_BOTH: Final[str] = "긴급차량 통행방해 + 물피도주 의심 차량 채증"
THEME_TITLE_SAFEPLATE: Final[str] = "SafePlate 4K"
THEME_SUBTITLE_SAFEPLATE: Final[str] = "물피도주 의심 차량 자동 채증 대시보드"
THEME_TITLE_GOLDENTIME: Final[str] = "GoldenTime 2.0"
THEME_SUBTITLE_GOLDENTIME: Final[str] = "긴급차량 통행방해 증거 대시보드"


# ─
# 데이터 구조
# ─
@dataclass(frozen=True)
class DashboardStats:
    """대시보드 상단 통계 바에 표시되는 집계값."""

    total_pkgs: int
    unique_plates: int
    total_detections: int
    total_violations: int
    total_departures: int


@dataclass(frozen=True)
class DashboardTheme:
    """헤더 타이틀/서브타이틀."""

    title: str
    subtitle: str


# ─
# 스캐닝 — 디스크 → 패키지 dict 리스트
# ─
def _encode_image_to_data_uri(img_path: Path) -> tuple[str, float]:
    """단일 이미지를 base64 data-URI로 인코딩 → (data_uri, size_kb)."""
    ext = img_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    with img_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    size_kb = round(img_path.stat().st_size / BYTES_PER_KB, 1)
    return f"data:{mime};base64,{b64}", size_kb


def _load_screenshots(ss_dir: Path) -> list[dict]:
    """스크린샷 디렉토리를 스캔해 base64 인코딩된 이미지 메타 리스트 반환."""
    if not ss_dir.is_dir():
        return []

    shots: list[dict] = []
    for name in sorted(os.listdir(ss_dir)):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        img_path = ss_dir / name
        data_uri, size_kb = _encode_image_to_data_uri(img_path)
        shots.append({"name": name, "data_uri": data_uri, "size_kb": size_kb})
    return shots


def _load_json(pkg_dir: Path) -> dict:
    pjson = pkg_dir / "plates.json"
    if not pjson.exists():
        return {}
    with pjson.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_report(pkg_dir: Path) -> str:
    rpt = pkg_dir / "report.txt"
    if not rpt.exists():
        return "(보고서 없음)"
    with rpt.open("r", encoding="utf-8") as f:
        return f.read()


def _video_size_mb(pkg_dir: Path) -> float:
    vid = pkg_dir / "video.mp4"
    if not vid.exists():
        return 0.0
    return round(vid.stat().st_size / BYTES_PER_MB, 1)


def scan_packages(
    base_dir: str | Path,
    max_packages: int = MAX_PACKAGES_DEFAULT,
    skip_foreign: bool = True,
) -> list[dict]:
    """증거 패키지 디렉토리 스캔.

    Args:
        base_dir: 증거 패키지 루트 디렉토리
        max_packages: 최대 스캔 수 (기본 20, 대시보드 크기 제한)
        skip_foreign: FOREIGN_PLATE 패키지 스킵 (기본 True)

    Returns:
        패키지 dict 리스트 — 키: name, path, json_data, report, screenshots, video_mb
    """
    base = Path(base_dir)
    if not base.is_dir():
        return []

    packages: list[dict] = []
    # 최신 우선 (디렉토리명 역정렬)
    for name in sorted(os.listdir(base), reverse=True):
        if len(packages) >= max_packages:
            break

        pkg_dir = base / name
        if not pkg_dir.is_dir():
            continue
        if skip_foreign and FOREIGN_PLATE_TAG in name:
            continue

        packages.append({
            "name": name,
            "path": str(pkg_dir),
            "json_data": _load_json(pkg_dir),
            "report": _load_report(pkg_dir),
            "screenshots": _load_screenshots(pkg_dir / "screenshots"),
            "video_mb": _video_size_mb(pkg_dir),
        })

    return packages


# ─
# 분류 / 통계 / 테마
# ─
def _detect_package_type(pkg: dict) -> str:
    """패키지 유형 판별 → 'safeplate' 또는 'goldentime'."""
    name = pkg.get("name", "")
    jd = pkg.get("json_data", {})

    if name.startswith(SAFEPLATE_PREFIX):
        return "safeplate"
    if jd.get("scenario", "").startswith(DAMAGE_HIT_RUN_SCENARIO):
        return "safeplate"
    if "departure_info" in jd:
        return "safeplate"
    return "goldentime"


def _compute_stats(packages: list[dict]) -> DashboardStats:
    """패키지 리스트 → 헤더 통계 집계."""
    total_violations = 0
    total_detections = 0
    total_departures = 0

    for pkg in packages:
        jd = pkg.get("json_data", {})
        total_violations += jd.get("violation_summary", {}).get("violation_count", 0)
        total_detections += jd.get("evidence", {}).get("detection_count", 0)

        if _detect_package_type(pkg) == "safeplate":
            total_departures += 1
            # SafePlate 패키지는 departure_info에 detection_count가 있음
            total_detections += jd.get("departure_info", {}).get("detection_count", 0)

    unique_plates = {
        pkg.get("json_data", {}).get("plate", "") for pkg in packages
    }

    return DashboardStats(
        total_pkgs=len(packages),
        unique_plates=len(unique_plates),
        total_detections=total_detections,
        total_violations=total_violations,
        total_departures=total_departures,
    )


def _resolve_theme(packages: list[dict]) -> DashboardTheme:
    """패키지 구성에 따라 헤더 타이틀/서브타이틀 결정."""
    types = [_detect_package_type(p) for p in packages]
    has_safe = "safeplate" in types
    has_gold = "goldentime" in types

    if has_safe and has_gold:
        return DashboardTheme(THEME_TITLE_BOTH, THEME_SUBTITLE_BOTH)
    if has_safe:
        return DashboardTheme(THEME_TITLE_SAFEPLATE, THEME_SUBTITLE_SAFEPLATE)
    return DashboardTheme(THEME_TITLE_GOLDENTIME, THEME_SUBTITLE_GOLDENTIME)


def _parse_time(gen_at: str) -> str:
    """ISO 시간 문자열을 표시용 문자열로 파싱."""
    try:
        dt = datetime.fromisoformat(gen_at)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return gen_at


# ─
# 공용 HTML 헬퍼
# ─
def _build_gallery(pkg: dict) -> str:
    """스크린샷 갤러리 HTML 생성."""
    parts: list[str] = []
    for ss in pkg["screenshots"]:
        label = ss["name"].replace(".jpg", "").replace("_", " ").title()
        parts.append(f'''
            <div class="thumb-card">
                <img src="{ss['data_uri']}" alt="{ss['name']}"
                     onclick="openModal(this.src, '{ss['name']}')" />
                <div class="thumb-label">{label}</div>
                <div class="thumb-size">{ss['size_kb']}KB</div>
            </div>''')
    return "".join(parts)


def _build_data_table_rows(rows: list[tuple[str, str]]) -> str:
    """라벨/값 쌍 리스트를 <tbody> 행 HTML로 직렬화."""
    return "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows
    )


def _build_screenshot_section(pkg: dict) -> str:
    return f'''
        <div class="section">
            <h3>&#128248; 채증 스크린샷</h3>
            <div class="gallery">{_build_gallery(pkg)}
            </div>
        </div>'''


def _build_target_section(targets: list[str]) -> str:
    items_html = "".join(f"<li>{t}</li>" for t in targets)
    return f'''
        <div class="section">
            <h3>&#128230; 제출 대상</h3>
            <ul class="target-list">{items_html}</ul>
        </div>'''


def _build_source_data_section(idx: int, json_data: dict, report: str) -> str:
    """plates.json + report.txt 탭 섹션 — goldentime/safeplate 공용."""
    json_pretty = json.dumps(json_data, indent=2, ensure_ascii=False)
    return f'''
        <div class="section">
            <h3>&#128196; 원본 데이터</h3>
            <div class="tab-container">
                <button class="tab-btn active" onclick="switchTab(this, 'json-{idx}')">plates.json</button>
                <button class="tab-btn" onclick="switchTab(this, 'report-{idx}')">report.txt</button>
            </div>
            <div class="tab-content" id="json-{idx}">
                <pre class="code-block">{json_pretty}</pre>
            </div>
            <div class="tab-content" id="report-{idx}" style="display:none;">
                <pre class="code-block report-block">{report}</pre>
            </div>
        </div>'''


# ─
# 카드 빌더 — GoldenTime / SafePlate
# ─
def _build_violation_rows(dv_list: list[dict]) -> str:
    """GoldenTime 거리 위반 행."""
    if not dv_list:
        return '<tr><td colspan="5" class="no-data">거리 미확보 위반 없음</td></tr>'

    parts: list[str] = []
    for i, dv in enumerate(dv_list, 1):
        parts.append(f'''
                <tr>
                    <td>{i}</td>
                    <td>F{dv.get('frame_idx', '?')}</td>
                    <td><strong>{dv.get('distance_label', '?')}</strong></td>
                    <td>{dv.get('close_duration_sec', 0):.1f}초</td>
                    <td>{dv.get('bbox_ratio', 0) * 100:.3f}%</td>
                </tr>''')
    return "".join(parts)


def _build_card_goldentime(idx: int, pkg: dict) -> str:
    """GoldenTime 증거 카드 HTML 생성."""
    jd = pkg["json_data"]
    plate = jd.get("plate", "알 수 없음")
    time_str = _parse_time(jd.get("generated_at", ""))
    ev = jd.get("evidence", {})
    vs = jd.get("violation_summary", {})
    dv_list = jd.get("distance_violations", [])
    targets = jd.get("submission_targets", [])

    has_violation = vs.get("has_distance_violation", False)
    v_count = vs.get("violation_count", 0)
    badge_class = "badge-danger" if has_violation else "badge-safe"
    badge_text = f"위반 {v_count}건" if has_violation else "위반 없음"

    detail_rows = _build_data_table_rows([
        ("번호판", f"<strong>{plate}</strong>"),
        ("생성 시각", time_str),
        ("시작 프레임", f"F{ev.get('start_frame', '?')}"),
        ("종료 프레임", f"F{ev.get('end_frame', '?')}"),
        ("총 프레임", str(ev.get("total_frames", 0))),
        ("영상 길이", f"{ev.get('total_duration_sec', 0):.1f}초"),
        ("최초 감지 시각", f"{ev.get('first_seen_time_sec', 0):.2f}초"),
        ("연속 감지 시간", f"{ev.get('continuous_duration_sec', 0):.2f}초"),
        ("감지 횟수", f"{ev.get('detection_count', 0)}회"),
        ("최종 신뢰도", f"{ev.get('last_confidence', 0) * 100:.0f}%"),
        ("최종 bbox", str(ev.get("last_bbox", []))),
    ])

    violation_rows = _build_violation_rows(dv_list)

    return f'''
    <div class="evidence-card" id="pkg-{idx}">
        <div class="card-header">
            <div class="card-title-row">
                <h2><span class="plate-num">{plate}</span></h2>
                <span class="badge badge-goldentime">GoldenTime</span>
                <span class="badge {badge_class}">{badge_text}</span>
            </div>
            <div class="card-meta">
                <span class="meta-item">
                    <span class="meta-icon">&#128197;</span> {time_str}
                </span>
                <span class="meta-item">
                    <span class="meta-icon">&#127909;</span> {ev.get('total_duration_sec', 0):.0f}초 ({ev.get('total_frames', 0)} 프레임)
                </span>
                <span class="meta-item">
                    <span class="meta-icon">&#128269;</span> {ev.get('detection_count', 0)}회 감지, 신뢰도 {ev.get('last_confidence', 0) * 100:.0f}%
                </span>
                <span class="meta-item">
                    <span class="meta-icon">&#9201;</span> 연속 {ev.get('continuous_duration_sec', 0):.1f}초
                </span>
                <span class="meta-item">
                    <span class="meta-icon">&#128190;</span> video.mp4 ({pkg['video_mb']}MB)
                </span>
            </div>
            <div class="card-folder">{pkg['name']}</div>
        </div>
        {_build_screenshot_section(pkg)}
        <div class="section">
            <h3>&#128202; 증거 데이터</h3>
            <table class="data-table">
                <thead><tr><th>항목</th><th>값</th></tr></thead>
                <tbody>{detail_rows}</tbody>
            </table>
        </div>
        <div class="section">
            <h3>&#9888;&#65039; 거리 미확보 위반</h3>
            <table class="data-table violation-table">
                <thead><tr><th>#</th><th>프레임</th><th>거리</th><th>지속시간</th><th>bbox 비율</th></tr></thead>
                <tbody>{violation_rows}</tbody>
            </table>
        </div>
        {_build_target_section(targets)}
        {_build_source_data_section(idx, jd, pkg["report"])}
    </div>
'''


def _build_card_safeplate(idx: int, pkg: dict) -> str:
    """SafePlate 4K 증거 카드 HTML 생성."""
    jd = pkg["json_data"]
    plate = jd.get("plate", "알 수 없음")
    time_str = _parse_time(jd.get("generated_at", ""))
    ev = jd.get("evidence", {})
    dep = jd.get("departure_info", {})
    shock = jd.get("shock_event", {})
    targets = jd.get("submission_targets", [])

    direction = dep.get("departure_direction", "unknown")
    dir_label = DEPARTURE_DIRECTION_LABELS.get(direction, direction)

    detail_rows = _build_data_table_rows([
        ("번호판", f"<strong>{plate}</strong>"),
        ("생성 시각", time_str),
        ("이탈 방향", f'<strong style="color:#ff6b6b">{dir_label}</strong>'),
        ("이탈 시각", f"영상 {dep.get('departure_time_sec', 0):.1f}초"),
        (
            "충격 시각",
            f"영상 {shock.get('shock_timestamp_sec', 0):.1f}초 "
            f"(프레임 #{shock.get('shock_frame_idx', 0)})",
        ),
        ("최초 감지", f"{dep.get('first_seen_time_sec', 0):.2f}초"),
        ("마지막 감지", f"{dep.get('last_seen_time_sec', 0):.2f}초"),
        ("감지 횟수", f"{dep.get('detection_count', 0)}회"),
        ("신뢰도", f"{dep.get('confidence', 0) * 100:.0f}%"),
        ("이동 벡터", str(dep.get("movement_vector", [0, 0]))),
        ("최초 bbox", str(dep.get("first_bbox", []))),
        ("마지막 bbox", str(dep.get("last_bbox", []))),
        ("총 프레임", str(ev.get("total_frames", 0))),
        ("영상 길이", f"{ev.get('total_duration_sec', 0):.1f}초"),
        ("전 버퍼", f"{ev.get('pre_buffer_frames', 0)} 프레임"),
        ("후 버퍼", f"{ev.get('post_buffer_frames', 0)} 프레임"),
    ])

    return f'''
    <div class="evidence-card safeplate-card" id="pkg-{idx}">
        <div class="card-header card-header-safeplate">
            <div class="card-title-row">
                <h2><span class="plate-num plate-num-safeplate">{plate}</span></h2>
                <span class="badge badge-safeplate">SafePlate 4K</span>
                <span class="badge badge-danger">{dir_label}</span>
            </div>
            <div class="card-meta">
                <span class="meta-item">
                    <span class="meta-icon">&#128197;</span> {time_str}
                </span>
                <span class="meta-item">
                    <span class="meta-icon">&#127909;</span> {ev.get('total_duration_sec', 0):.0f}초 ({ev.get('total_frames', 0)} 프레임)
                </span>
                <span class="meta-item">
                    <span class="meta-icon">&#128663;</span> 감지 {dep.get('detection_count', 0)}회, 신뢰도 {dep.get('confidence', 0) * 100:.0f}%
                </span>
                <span class="meta-item">
                    <span class="meta-icon">&#128190;</span> video.mp4 ({pkg['video_mb']}MB)
                </span>
            </div>
            <div class="card-folder">{pkg['name']}</div>
        </div>
        {_build_screenshot_section(pkg)}
        <div class="section">
            <h3>&#128663; 이탈 차량 정보</h3>
            <table class="data-table">
                <thead><tr><th>항목</th><th>값</th></tr></thead>
                <tbody>{detail_rows}</tbody>
            </table>
        </div>
        {_build_target_section(targets)}
        {_build_source_data_section(idx, jd, pkg["report"])}
    </div>
'''


# ─
# CSS / JS — 모듈 상수
# ─
DASHBOARD_CSS: Final[str] = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0a0e1a;
    color: #e0e0e0;
    line-height: 1.6;
}
.dashboard { max-width: 1200px; margin: 0 auto; padding: 20px; }

/* 헤더 */
.header {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    margin-bottom: 30px;
    border: 1px solid #1e3a5f;
}
.header h1 {
    font-size: 2.2em;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}
.header .subtitle { color: #8899aa; font-size: 1.1em; }

/* 통계 바 */
.stats-bar { display: flex; gap: 16px; margin-bottom: 30px; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 140px;
    background: #141b2d; border: 1px solid #1e3050;
    border-radius: 12px; padding: 20px; text-align: center;
}
.stat-card .stat-num { font-size: 2em; font-weight: 700; color: #4d96ff; }
.stat-card .stat-label { color: #8899aa; font-size: 0.85em; margin-top: 4px; }
.stat-card.danger .stat-num { color: #ff6b6b; }
.stat-card.success .stat-num { color: #6bcb77; }
.stat-card.warning .stat-num { color: #ff9f43; }

/* SafePlate 테마 */
.badge-safeplate {
    background: rgba(255,159,67,0.15); color: #ff9f43;
    border: 1px solid rgba(255,159,67,0.3);
}
.badge-goldentime {
    background: rgba(77,150,255,0.15); color: #4d96ff;
    border: 1px solid rgba(77,150,255,0.3);
}
.card-header-safeplate {
    background: linear-gradient(135deg, #2a1a0e 0%, #3a2010 50%, #2e1a08 100%) !important;
    border-bottom: 1px solid #4a2a10 !important;
}
.plate-num-safeplate {
    color: #ff9f43 !important;
    background: rgba(255,159,67,0.1) !important;
    border-color: rgba(255,159,67,0.3) !important;
}
.safeplate-card { border-color: #3a2010 !important; }

/* 증거 카드 */
.evidence-card {
    background: #141b2d; border: 1px solid #1e3050;
    border-radius: 16px; margin-bottom: 30px; overflow: hidden;
}
.card-header {
    padding: 24px;
    background: linear-gradient(135deg, #1a2332 0%, #1e2d42 100%);
    border-bottom: 1px solid #1e3050;
}
.card-title-row { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }
.plate-num {
    font-size: 1.6em; font-weight: 800; color: #ffd93d;
    letter-spacing: 2px; background: rgba(255,217,61,0.1);
    padding: 4px 16px; border-radius: 8px;
    border: 2px solid rgba(255,217,61,0.3);
}
.badge { padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; }
.badge-danger {
    background: rgba(255,107,107,0.15); color: #ff6b6b;
    border: 1px solid rgba(255,107,107,0.3);
}
.badge-safe {
    background: rgba(107,203,119,0.15); color: #6bcb77;
    border: 1px solid rgba(107,203,119,0.3);
}
.card-meta { display: flex; gap: 20px; flex-wrap: wrap; }
.meta-item { font-size: 0.88em; color: #8899aa; }
.meta-icon { margin-right: 4px; }
.card-folder { margin-top: 8px; font-family: monospace; font-size: 0.78em; color: #556677; }

/* 섹션 */
.section { padding: 24px; border-bottom: 1px solid #1a2538; }
.section:last-child { border-bottom: none; }
.section h3 { color: #c0d0e0; margin-bottom: 16px; font-size: 1.1em; }

/* 갤러리 */
.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
}
.thumb-card {
    background: #0d1420; border-radius: 8px; overflow: hidden;
    border: 1px solid #1e2d42;
    transition: transform 0.2s, border-color 0.2s; cursor: pointer;
}
.thumb-card:hover { transform: translateY(-3px); border-color: #4d96ff; }
.thumb-card img { width: 100%; height: 140px; object-fit: cover; display: block; }
.thumb-label { padding: 6px 10px; font-size: 0.82em; color: #b0c0d0; font-weight: 600; }
.thumb-size { padding: 0 10px 6px; font-size: 0.72em; color: #556677; }

/* 테이블 */
.data-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
.data-table th {
    background: #0d1420; padding: 10px 14px; text-align: left;
    color: #8899bb; font-weight: 600; border-bottom: 2px solid #1e3050;
}
.data-table td { padding: 8px 14px; border-bottom: 1px solid #1a2538; }
.data-table tr:hover td { background: rgba(77,150,255,0.05); }
.violation-table td:nth-child(3) { color: #ff6b6b; font-weight: 700; }
.no-data {
    text-align: center; color: #556677; font-style: italic;
    padding: 20px !important;
}

/* 제출 대상 */
.target-list { list-style: none; padding: 0; }
.target-list li {
    padding: 10px 16px; margin-bottom: 8px;
    background: rgba(77,150,255,0.08); border-left: 3px solid #4d96ff;
    border-radius: 0 8px 8px 0; font-size: 0.9em;
}

/* 탭 */
.tab-container { display: flex; gap: 4px; margin-bottom: 12px; }
.tab-btn {
    padding: 8px 20px; background: #0d1420; color: #8899aa;
    border: 1px solid #1e2d42; border-radius: 8px 8px 0 0;
    cursor: pointer; font-size: 0.85em;
    font-family: inherit; transition: all 0.2s;
}
.tab-btn.active {
    background: #1a2a40; color: #4d96ff;
    border-color: #4d96ff; border-bottom-color: #1a2a40;
}
.tab-btn:hover { color: #fff; }
.code-block {
    background: #0a0e18; padding: 16px;
    border-radius: 0 8px 8px 8px; border: 1px solid #1e2d42;
    overflow-x: auto;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.82em; line-height: 1.5; color: #a0b8d0;
    max-height: 400px; overflow-y: auto; white-space: pre;
}

/* 모달 */
.modal-overlay {
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.85); z-index: 1000;
    justify-content: center; align-items: center; cursor: pointer;
}
.modal-overlay.active { display: flex; }
.modal-img {
    max-width: 90vw; max-height: 90vh;
    border-radius: 8px; box-shadow: 0 0 40px rgba(0,0,0,0.5);
}
.modal-label {
    position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%);
    color: #fff; background: rgba(0,0,0,0.7);
    padding: 8px 24px; border-radius: 20px; font-size: 0.9em;
}

/* 푸터 */
.footer { text-align: center; padding: 30px; color: #445566; font-size: 0.82em; }

/* 반응형 */
@media (max-width: 768px) {
    .stats-bar { flex-direction: column; }
    .gallery { grid-template-columns: repeat(2, 1fr); }
    .card-meta { flex-direction: column; gap: 6px; }
    .header h1 { font-size: 1.6em; }
}
"""

DASHBOARD_JS: Final[str] = """
function openModal(src, label) {
    document.getElementById('modalImg').src = src;
    document.getElementById('modalLabel').textContent = label;
    document.getElementById('imgModal').classList.add('active');
}
function closeModal() {
    document.getElementById('imgModal').classList.remove('active');
}
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeModal();
});
function switchTab(btn, contentId) {
    var parent = btn.parentElement.parentElement;
    parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    parent.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
    btn.classList.add('active');
    document.getElementById(contentId).style.display = 'block';
}
"""


# ─
# HTML 셸 빌더
# ─
def _build_stats_bar(stats: DashboardStats) -> str:
    return f'''
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-num">{stats.total_pkgs}</div>
            <div class="stat-label">증거 패키지</div>
        </div>
        <div class="stat-card success">
            <div class="stat-num">{stats.unique_plates}</div>
            <div class="stat-label">고유 번호판</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{stats.total_detections}</div>
            <div class="stat-label">총 감지 횟수</div>
        </div>
        <div class="stat-card danger">
            <div class="stat-num">{stats.total_violations}</div>
            <div class="stat-label">거리 위반</div>
        </div>
        <div class="stat-card warning">
            <div class="stat-num">{stats.total_departures}</div>
            <div class="stat-label">이탈 차량</div>
        </div>
    </div>'''


def _build_cards(packages: list[dict]) -> str:
    parts: list[str] = []
    for idx, pkg in enumerate(packages):
        if _detect_package_type(pkg) == "safeplate":
            parts.append(_build_card_safeplate(idx, pkg))
        else:
            parts.append(_build_card_goldentime(idx, pkg))
    return "".join(parts)


def build_html(packages: list[dict]) -> str:
    """HTML 대시보드 문자열 생성 — GoldenTime + SafePlate 통합."""
    theme = _resolve_theme(packages)
    stats = _compute_stats(packages)
    cards_html = _build_cards(packages)
    stats_html = _build_stats_bar(stats)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{theme.title} - 증거 대시보드</title>
<style>{DASHBOARD_CSS}</style>
</head>
<body>

<div class="dashboard">
    <div class="header">
        <h1>{theme.title}</h1>
        <div class="subtitle">{theme.subtitle}</div>
    </div>
    {stats_html}
    {cards_html}
    <div class="footer">
        {theme.title} Evidence Dashboard &mdash;
        생성: {now_str} &mdash;
        YOLO26 Simulation System
    </div>
</div>

<div class="modal-overlay" id="imgModal" onclick="closeModal()">
    <img class="modal-img" id="modalImg" src="" alt="" />
    <div class="modal-label" id="modalLabel"></div>
</div>

<script>{DASHBOARD_JS}</script>

</body>
</html>'''


# ─
# CLI
# ─
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26 증거 대시보드(HTML) 생성기 — "
                    "GoldenTime + SafePlate 통합",
    )
    parser.add_argument(
        "--base", type=Path, default=DEFAULT_BASE_DIR,
        help=f"증거 패키지 루트 디렉토리 (기본: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_FILE,
        help=f"출력 HTML 파일 경로 (기본: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--max", type=int, default=MAX_PACKAGES_DEFAULT, dest="max_packages",
        help=f"최대 스캔 패키지 수 (기본: {MAX_PACKAGES_DEFAULT})",
    )
    parser.add_argument(
        "--include-foreign", action="store_true",
        help="FOREIGN_PLATE 패키지 포함 (기본: 제외)",
    )
    return parser.parse_args(argv)


def _force_utf8_stdout() -> None:
    """Windows 콘솔에서 한글 깨짐 방지. (side effect는 main()에서 1회만)"""
    if isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            return
        except (AttributeError, OSError):
            pass
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _print_summary(packages: list[dict]) -> None:
    print(f"[OK] {len(packages)}개 증거 패키지 스캔 완료")
    for pkg in packages:
        plate = pkg.get("json_data", {}).get("plate", "?")
        ss = len(pkg["screenshots"])
        viol = pkg.get("json_data", {}).get(
            "violation_summary", {}
        ).get("violation_count", 0)
        print(f"  - {pkg['name']}")
        print(f"    번호판: {plate}, 스크린샷: {ss}장, 위반: {viol}건")
    print()


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = _parse_args(argv)

    print("=" * 60)
    print("  YOLO26 증거 대시보드 생성기 (GoldenTime + SafePlate)")
    print("=" * 60)
    print()

    packages = scan_packages(
        args.base,
        max_packages=args.max_packages,
        skip_foreign=not args.include_foreign,
    )
    _print_summary(packages)

    html = build_html(packages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    size_kb = args.output.stat().st_size / BYTES_PER_KB
    print("[OK] 대시보드 생성 완료!")
    print(f"  파일: {args.output}")
    print(f"  크기: {size_kb:.0f}KB")
    print("  이미지: base64 임베딩 (외부 의존성 없음)")
    print()
    print("  브라우저에서 열기:")
    print(f"  start {args.output}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
