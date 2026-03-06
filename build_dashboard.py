"""
build_dashboard.py - YOLO26 증거 대시보드 HTML 생성기

evidence_output/ 안의 모든 증거 패키지를 스캔하여
브라우저에서 바로 열 수 있는 단일 HTML 파일을 생성한다.

지원 시나리오:
    - GoldenTime 2.0: 긴급차량 통행방해 채증 (evidence_YYYYMMDD_...)
    - SafePlate 4K:   물피도주 의심 차량 채증 (safeplate_YYYYMMDD_...)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import base64
from datetime import datetime


def scan_packages(base_dir):
    """증거 패키지 디렉토리 스캔"""
    packages = []
    for d in sorted(os.listdir(base_dir)):
        pkg_dir = os.path.join(base_dir, d)
        if not os.path.isdir(pkg_dir):
            continue

        pkg = {"name": d, "path": pkg_dir}

        # plates.json 로드
        pjson = os.path.join(pkg_dir, "plates.json")
        if os.path.exists(pjson):
            with open(pjson, "r", encoding="utf-8") as f:
                pkg["json_data"] = json.load(f)
        else:
            pkg["json_data"] = {}

        # report.txt 로드
        rpt = os.path.join(pkg_dir, "report.txt")
        if os.path.exists(rpt):
            with open(rpt, "r", encoding="utf-8") as f:
                pkg["report"] = f.read()
        else:
            pkg["report"] = "(보고서 없음)"

        # 스크린샷 base64 인코딩
        ss_dir = os.path.join(pkg_dir, "screenshots")
        pkg["screenshots"] = []
        if os.path.isdir(ss_dir):
            for img in sorted(os.listdir(ss_dir)):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(ss_dir, img)
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    ext = img.rsplit(".", 1)[-1].lower()
                    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
                    pkg["screenshots"].append({
                        "name": img,
                        "data_uri": f"data:{mime};base64,{b64}",
                        "size_kb": round(os.path.getsize(img_path) / 1024, 1),
                    })

        # video.mp4 정보
        vid = os.path.join(pkg_dir, "video.mp4")
        if os.path.exists(vid):
            pkg["video_mb"] = round(os.path.getsize(vid) / (1024 * 1024), 1)
        else:
            pkg["video_mb"] = 0

        packages.append(pkg)

    return packages


def _detect_package_type(pkg: dict) -> str:
    """패키지 유형 판별: 'safeplate' 또는 'goldentime'"""
    name = pkg.get("name", "")
    jd = pkg.get("json_data", {})

    if name.startswith("safeplate_"):
        return "safeplate"
    if jd.get("scenario", "").startswith("물피도주"):
        return "safeplate"
    if "departure_info" in jd:
        return "safeplate"
    return "goldentime"


def _build_card_goldentime(idx: int, pkg: dict) -> str:
    """GoldenTime 증거 카드 HTML 생성"""
    jd = pkg["json_data"]
    plate = jd.get("plate", "알 수 없음")
    gen_at = jd.get("generated_at", "")
    ev = jd.get("evidence", {})
    vs = jd.get("violation_summary", {})
    dv_list = jd.get("distance_violations", [])
    targets = jd.get("submission_targets", [])

    time_str = _parse_time(gen_at)

    has_violation = vs.get("has_distance_violation", False)
    v_count = vs.get("violation_count", 0)
    badge_class = "badge-danger" if has_violation else "badge-safe"
    badge_text = f"위반 {v_count}건" if has_violation else "위반 없음"

    gallery_html = _build_gallery(pkg)

    # 거리 위반 테이블
    violation_rows = ""
    if dv_list:
        for i, dv in enumerate(dv_list, 1):
            violation_rows += f'''
                <tr>
                    <td>{i}</td>
                    <td>F{dv.get('frame_idx', '?')}</td>
                    <td><strong>{dv.get('distance_label', '?')}</strong></td>
                    <td>{dv.get('close_duration_sec', 0):.1f}초</td>
                    <td>{dv.get('bbox_ratio', 0)*100:.3f}%</td>
                </tr>'''
    else:
        violation_rows = '<tr><td colspan="5" class="no-data">거리 미확보 위반 없음</td></tr>'

    targets_html = "".join(f'<li>{t}</li>' for t in targets)
    json_pretty = json.dumps(jd, indent=2, ensure_ascii=False)
    report_text = pkg["report"]

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
                    <span class="meta-icon">&#128269;</span> {ev.get('detection_count', 0)}회 감지, 신뢰도 {ev.get('last_confidence', 0)*100:.0f}%
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

        <div class="section">
            <h3>&#128248; 채증 스크린샷</h3>
            <div class="gallery">{gallery_html}
            </div>
        </div>

        <div class="section">
            <h3>&#128202; 증거 데이터</h3>
            <table class="data-table">
                <thead><tr><th>항목</th><th>값</th></tr></thead>
                <tbody>
                    <tr><td>번호판</td><td><strong>{plate}</strong></td></tr>
                    <tr><td>생성 시각</td><td>{time_str}</td></tr>
                    <tr><td>시작 프레임</td><td>F{ev.get('start_frame', '?')}</td></tr>
                    <tr><td>종료 프레임</td><td>F{ev.get('end_frame', '?')}</td></tr>
                    <tr><td>총 프레임</td><td>{ev.get('total_frames', 0)}</td></tr>
                    <tr><td>영상 길이</td><td>{ev.get('total_duration_sec', 0):.1f}초</td></tr>
                    <tr><td>최초 감지 시각</td><td>{ev.get('first_seen_time_sec', 0):.2f}초</td></tr>
                    <tr><td>연속 감지 시간</td><td>{ev.get('continuous_duration_sec', 0):.2f}초</td></tr>
                    <tr><td>감지 횟수</td><td>{ev.get('detection_count', 0)}회</td></tr>
                    <tr><td>최종 신뢰도</td><td>{ev.get('last_confidence', 0)*100:.0f}%</td></tr>
                    <tr><td>최종 bbox</td><td>{ev.get('last_bbox', [])}</td></tr>
                </tbody>
            </table>
        </div>

        <div class="section">
            <h3>&#9888;&#65039; 거리 미확보 위반</h3>
            <table class="data-table violation-table">
                <thead><tr><th>#</th><th>프레임</th><th>거리</th><th>지속시간</th><th>bbox 비율</th></tr></thead>
                <tbody>{violation_rows}</tbody>
            </table>
        </div>

        <div class="section">
            <h3>&#128230; 제출 대상</h3>
            <ul class="target-list">{targets_html}</ul>
        </div>

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
                <pre class="code-block report-block">{report_text}</pre>
            </div>
        </div>
    </div>
'''


def _build_card_safeplate(idx: int, pkg: dict) -> str:
    """SafePlate 4K 증거 카드 HTML 생성"""
    jd = pkg["json_data"]
    plate = jd.get("plate", "알 수 없음")
    gen_at = jd.get("generated_at", "")
    ev = jd.get("evidence", {})
    dep = jd.get("departure_info", {})
    shock = jd.get("shock_event", {})
    targets = jd.get("submission_targets", [])

    time_str = _parse_time(gen_at)

    # 이탈 방향 한글화
    dir_map = {
        "left": "좌측 이탈", "right": "우측 이탈",
        "top": "상방 이탈", "bottom": "하방 이탈",
        "vanished": "화면 소멸",
    }
    direction = dep.get("departure_direction", "unknown")
    dir_label = dir_map.get(direction, direction)

    gallery_html = _build_gallery(pkg)
    targets_html = "".join(f'<li>{t}</li>' for t in targets)
    json_pretty = json.dumps(jd, indent=2, ensure_ascii=False)
    report_text = pkg["report"]

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
                    <span class="meta-icon">&#128663;</span> 감지 {dep.get('detection_count', 0)}회, 신뢰도 {dep.get('confidence', 0)*100:.0f}%
                </span>
                <span class="meta-item">
                    <span class="meta-icon">&#128190;</span> video.mp4 ({pkg['video_mb']}MB)
                </span>
            </div>
            <div class="card-folder">{pkg['name']}</div>
        </div>

        <div class="section">
            <h3>&#128248; 채증 스크린샷</h3>
            <div class="gallery">{gallery_html}
            </div>
        </div>

        <div class="section">
            <h3>&#128663; 이탈 차량 정보</h3>
            <table class="data-table">
                <thead><tr><th>항목</th><th>값</th></tr></thead>
                <tbody>
                    <tr><td>번호판</td><td><strong>{plate}</strong></td></tr>
                    <tr><td>생성 시각</td><td>{time_str}</td></tr>
                    <tr><td>이탈 방향</td><td><strong style="color:#ff6b6b">{dir_label}</strong></td></tr>
                    <tr><td>이탈 시각</td><td>영상 {dep.get('departure_time_sec', 0):.1f}초</td></tr>
                    <tr><td>충격 시각</td><td>영상 {shock.get('shock_timestamp_sec', 0):.1f}초 (프레임 #{shock.get('shock_frame_idx', 0)})</td></tr>
                    <tr><td>최초 감지</td><td>{dep.get('first_seen_time_sec', 0):.2f}초</td></tr>
                    <tr><td>마지막 감지</td><td>{dep.get('last_seen_time_sec', 0):.2f}초</td></tr>
                    <tr><td>감지 횟수</td><td>{dep.get('detection_count', 0)}회</td></tr>
                    <tr><td>신뢰도</td><td>{dep.get('confidence', 0)*100:.0f}%</td></tr>
                    <tr><td>이동 벡터</td><td>{dep.get('movement_vector', [0, 0])}</td></tr>
                    <tr><td>최초 bbox</td><td>{dep.get('first_bbox', [])}</td></tr>
                    <tr><td>마지막 bbox</td><td>{dep.get('last_bbox', [])}</td></tr>
                    <tr><td>총 프레임</td><td>{ev.get('total_frames', 0)}</td></tr>
                    <tr><td>영상 길이</td><td>{ev.get('total_duration_sec', 0):.1f}초</td></tr>
                    <tr><td>전 버퍼</td><td>{ev.get('pre_buffer_frames', 0)} 프레임</td></tr>
                    <tr><td>후 버퍼</td><td>{ev.get('post_buffer_frames', 0)} 프레임</td></tr>
                </tbody>
            </table>
        </div>

        <div class="section">
            <h3>&#128230; 제출 대상</h3>
            <ul class="target-list">{targets_html}</ul>
        </div>

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
                <pre class="code-block report-block">{report_text}</pre>
            </div>
        </div>
    </div>
'''


def _parse_time(gen_at: str) -> str:
    """ISO 시간 문자열을 표시용 문자열로 파싱"""
    try:
        dt = datetime.fromisoformat(gen_at)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return gen_at


def _build_gallery(pkg: dict) -> str:
    """스크린샷 갤러리 HTML 생성"""
    gallery_html = ""
    for ss in pkg["screenshots"]:
        label = ss["name"].replace(".jpg", "").replace("_", " ").title()
        gallery_html += f'''
            <div class="thumb-card">
                <img src="{ss['data_uri']}" alt="{ss['name']}"
                     onclick="openModal(this.src, '{ss['name']}')" />
                <div class="thumb-label">{label}</div>
                <div class="thumb-size">{ss['size_kb']}KB</div>
            </div>'''
    return gallery_html


def build_html(packages):
    """HTML 대시보드 문자열 생성 — GoldenTime + SafePlate 통합"""

    # 각 패키지별 카드 HTML
    cards_html = ""
    safeplate_count = 0
    goldentime_count = 0

    for idx, pkg in enumerate(packages):
        pkg_type = _detect_package_type(pkg)
        if pkg_type == "safeplate":
            cards_html += _build_card_safeplate(idx, pkg)
            safeplate_count += 1
        else:
            cards_html += _build_card_goldentime(idx, pkg)
            goldentime_count += 1

    # 통계 요약
    total_pkgs = len(packages)
    total_violations = sum(
        p.get("json_data", {}).get("violation_summary", {}).get("violation_count", 0)
        for p in packages
    )
    total_departures = sum(
        1 for p in packages if _detect_package_type(p) == "safeplate"
    )
    total_detections = sum(
        p.get("json_data", {}).get("evidence", {}).get("detection_count", 0)
        for p in packages
    )
    # SafePlate 패키지는 departure_info에 detection_count가 있음
    for p in packages:
        if _detect_package_type(p) == "safeplate":
            dep = p.get("json_data", {}).get("departure_info", {})
            total_detections += dep.get("detection_count", 0)
    unique_plates = set(
        p.get("json_data", {}).get("plate", "") for p in packages
    )

    # 타이틀 결정
    if safeplate_count > 0 and goldentime_count > 0:
        title = "YOLO26 통합 증거 대시보드"
        subtitle = "긴급차량 통행방해 + 물피도주 의심 차량 채증"
    elif safeplate_count > 0:
        title = "SafePlate 4K"
        subtitle = "물피도주 의심 차량 자동 채증 대시보드"
    else:
        title = "GoldenTime 2.0"
        subtitle = "긴급차량 통행방해 증거 대시보드"

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - 증거 대시보드</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0a0e1a;
    color: #e0e0e0;
    line-height: 1.6;
}}
.dashboard {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}}

/* 헤더 */
.header {{
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    margin-bottom: 30px;
    border: 1px solid #1e3a5f;
}}
.header h1 {{
    font-size: 2.2em;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}}
.header .subtitle {{
    color: #8899aa;
    font-size: 1.1em;
}}

/* 통계 바 */
.stats-bar {{
    display: flex;
    gap: 16px;
    margin-bottom: 30px;
    flex-wrap: wrap;
}}
.stat-card {{
    flex: 1;
    min-width: 140px;
    background: #141b2d;
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}}
.stat-card .stat-num {{
    font-size: 2em;
    font-weight: 700;
    color: #4d96ff;
}}
.stat-card .stat-label {{
    color: #8899aa;
    font-size: 0.85em;
    margin-top: 4px;
}}
.stat-card.danger .stat-num {{ color: #ff6b6b; }}
.stat-card.success .stat-num {{ color: #6bcb77; }}
.stat-card.warning .stat-num {{ color: #ff9f43; }}

/* SafePlate 테마 */
.badge-safeplate {{
    background: rgba(255,159,67,0.15);
    color: #ff9f43;
    border: 1px solid rgba(255,159,67,0.3);
}}
.badge-goldentime {{
    background: rgba(77,150,255,0.15);
    color: #4d96ff;
    border: 1px solid rgba(77,150,255,0.3);
}}
.card-header-safeplate {{
    background: linear-gradient(135deg, #2a1a0e 0%, #3a2010 50%, #2e1a08 100%) !important;
    border-bottom: 1px solid #4a2a10 !important;
}}
.plate-num-safeplate {{
    color: #ff9f43 !important;
    background: rgba(255,159,67,0.1) !important;
    border-color: rgba(255,159,67,0.3) !important;
}}
.safeplate-card {{
    border-color: #3a2010 !important;
}}

/* 증거 카드 */
.evidence-card {{
    background: #141b2d;
    border: 1px solid #1e3050;
    border-radius: 16px;
    margin-bottom: 30px;
    overflow: hidden;
}}
.card-header {{
    padding: 24px;
    background: linear-gradient(135deg, #1a2332 0%, #1e2d42 100%);
    border-bottom: 1px solid #1e3050;
}}
.card-title-row {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
}}
.plate-num {{
    font-size: 1.6em;
    font-weight: 800;
    color: #ffd93d;
    letter-spacing: 2px;
    background: rgba(255,217,61,0.1);
    padding: 4px 16px;
    border-radius: 8px;
    border: 2px solid rgba(255,217,61,0.3);
}}
.badge {{
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
}}
.badge-danger {{
    background: rgba(255,107,107,0.15);
    color: #ff6b6b;
    border: 1px solid rgba(255,107,107,0.3);
}}
.badge-safe {{
    background: rgba(107,203,119,0.15);
    color: #6bcb77;
    border: 1px solid rgba(107,203,119,0.3);
}}
.card-meta {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}}
.meta-item {{
    font-size: 0.88em;
    color: #8899aa;
}}
.meta-icon {{
    margin-right: 4px;
}}
.card-folder {{
    margin-top: 8px;
    font-family: monospace;
    font-size: 0.78em;
    color: #556677;
}}

/* 섹션 */
.section {{
    padding: 24px;
    border-bottom: 1px solid #1a2538;
}}
.section:last-child {{
    border-bottom: none;
}}
.section h3 {{
    color: #c0d0e0;
    margin-bottom: 16px;
    font-size: 1.1em;
}}

/* 갤러리 */
.gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
}}
.thumb-card {{
    background: #0d1420;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #1e2d42;
    transition: transform 0.2s, border-color 0.2s;
    cursor: pointer;
}}
.thumb-card:hover {{
    transform: translateY(-3px);
    border-color: #4d96ff;
}}
.thumb-card img {{
    width: 100%;
    height: 140px;
    object-fit: cover;
    display: block;
}}
.thumb-label {{
    padding: 6px 10px;
    font-size: 0.82em;
    color: #b0c0d0;
    font-weight: 600;
}}
.thumb-size {{
    padding: 0 10px 6px;
    font-size: 0.72em;
    color: #556677;
}}

/* 테이블 */
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
}}
.data-table th {{
    background: #0d1420;
    padding: 10px 14px;
    text-align: left;
    color: #8899bb;
    font-weight: 600;
    border-bottom: 2px solid #1e3050;
}}
.data-table td {{
    padding: 8px 14px;
    border-bottom: 1px solid #1a2538;
}}
.data-table tr:hover td {{
    background: rgba(77,150,255,0.05);
}}
.violation-table td:nth-child(3) {{
    color: #ff6b6b;
    font-weight: 700;
}}
.no-data {{
    text-align: center;
    color: #556677;
    font-style: italic;
    padding: 20px !important;
}}

/* 제출 대상 */
.target-list {{
    list-style: none;
    padding: 0;
}}
.target-list li {{
    padding: 10px 16px;
    margin-bottom: 8px;
    background: rgba(77,150,255,0.08);
    border-left: 3px solid #4d96ff;
    border-radius: 0 8px 8px 0;
    font-size: 0.9em;
}}

/* 탭 */
.tab-container {{
    display: flex;
    gap: 4px;
    margin-bottom: 12px;
}}
.tab-btn {{
    padding: 8px 20px;
    background: #0d1420;
    color: #8899aa;
    border: 1px solid #1e2d42;
    border-radius: 8px 8px 0 0;
    cursor: pointer;
    font-size: 0.85em;
    font-family: inherit;
    transition: all 0.2s;
}}
.tab-btn.active {{
    background: #1a2a40;
    color: #4d96ff;
    border-color: #4d96ff;
    border-bottom-color: #1a2a40;
}}
.tab-btn:hover {{
    color: #fff;
}}
.code-block {{
    background: #0a0e18;
    padding: 16px;
    border-radius: 0 8px 8px 8px;
    border: 1px solid #1e2d42;
    overflow-x: auto;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.82em;
    line-height: 1.5;
    color: #a0b8d0;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre;
}}

/* 모달 */
.modal-overlay {{
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.85);
    z-index: 1000;
    justify-content: center;
    align-items: center;
    cursor: pointer;
}}
.modal-overlay.active {{
    display: flex;
}}
.modal-img {{
    max-width: 90vw;
    max-height: 90vh;
    border-radius: 8px;
    box-shadow: 0 0 40px rgba(0,0,0,0.5);
}}
.modal-label {{
    position: fixed;
    bottom: 30px;
    left: 50%;
    transform: translateX(-50%);
    color: #fff;
    background: rgba(0,0,0,0.7);
    padding: 8px 24px;
    border-radius: 20px;
    font-size: 0.9em;
}}

/* 푸터 */
.footer {{
    text-align: center;
    padding: 30px;
    color: #445566;
    font-size: 0.82em;
}}

/* 반응형 */
@media (max-width: 768px) {{
    .stats-bar {{ flex-direction: column; }}
    .gallery {{ grid-template-columns: repeat(2, 1fr); }}
    .card-meta {{ flex-direction: column; gap: 6px; }}
    .header h1 {{ font-size: 1.6em; }}
}}
</style>
</head>
<body>

<div class="dashboard">
    <!-- 헤더 -->
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>

    <!-- 통계 바 -->
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-num">{total_pkgs}</div>
            <div class="stat-label">증거 패키지</div>
        </div>
        <div class="stat-card success">
            <div class="stat-num">{len(unique_plates)}</div>
            <div class="stat-label">고유 번호판</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{total_detections}</div>
            <div class="stat-label">총 감지 횟수</div>
        </div>
        <div class="stat-card danger">
            <div class="stat-num">{total_violations}</div>
            <div class="stat-label">거리 위반</div>
        </div>
        <div class="stat-card warning">
            <div class="stat-num">{total_departures}</div>
            <div class="stat-label">이탈 차량</div>
        </div>
    </div>

    <!-- 증거 카드 -->
    {cards_html}

    <!-- 푸터 -->
    <div class="footer">
        {title} Evidence Dashboard &mdash;
        생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &mdash;
        YOLO26 Simulation System
    </div>
</div>

<!-- 이미지 모달 -->
<div class="modal-overlay" id="imgModal" onclick="closeModal()">
    <img class="modal-img" id="modalImg" src="" alt="" />
    <div class="modal-label" id="modalLabel"></div>
</div>

<script>
function openModal(src, label) {{
    document.getElementById('modalImg').src = src;
    document.getElementById('modalLabel').textContent = label;
    document.getElementById('imgModal').classList.add('active');
}}
function closeModal() {{
    document.getElementById('imgModal').classList.remove('active');
}}
document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') closeModal();
}});
function switchTab(btn, contentId) {{
    var parent = btn.parentElement.parentElement;
    parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    parent.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
    btn.classList.add('active');
    document.getElementById(contentId).style.display = 'block';
}}
</script>

</body>
</html>'''
    return html


def main():
    base_dir = "C:/tool/yolo26-main/evidence_output"
    output_file = "C:/tool/yolo26-main/evidence_dashboard.html"

    print("=" * 60)
    print("  YOLO26 증거 대시보드 생성기 (GoldenTime + SafePlate)")
    print("=" * 60)
    print()

    # 패키지 스캔
    packages = scan_packages(base_dir)
    print(f"[OK] {len(packages)}개 증거 패키지 스캔 완료")
    for p in packages:
        plate = p.get("json_data", {}).get("plate", "?")
        ss = len(p["screenshots"])
        viol = p.get("json_data", {}).get("violation_summary", {}).get("violation_count", 0)
        print(f"  - {p['name']}")
        print(f"    번호판: {plate}, 스크린샷: {ss}장, 위반: {viol}건")
    print()

    # HTML 생성
    html = build_html(packages)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(output_file) / 1024
    print(f"[OK] 대시보드 생성 완료!")
    print(f"  파일: {output_file}")
    print(f"  크기: {size_kb:.0f}KB")
    print(f"  이미지: base64 임베딩 (외부 의존성 없음)")
    print()
    print("  브라우저에서 열기:")
    print(f"  start {output_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
