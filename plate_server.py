"""
plate_server.py - 번호판 인식 결과 로컬 뷰어
────────────────────────────────────────────
사용법:
    python plate_server.py
    python plate_server.py --port 8080
    python plate_server.py --results plate_results_v2
"""

import argparse
import json
import os
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

RESULTS_DIR = "plate_results_v2"
DEFAULT_PORT = 8000

# 자동 스캔 대상 디렉토리 패턴
RESULT_DIR_PREFIX = "plate_results"


def find_all_result_dirs(base_dir: str = ".") -> list[str]:
    """base_dir 아래에서 plate_results* 디렉토리를 모두 찾아 반환."""
    dirs = []
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith(RESULT_DIR_PREFIX):
            json_path = os.path.join(full, "results.json")
            if os.path.isfile(json_path):
                dirs.append(name)
    return dirs


def load_results(results_dir: str) -> list[dict]:
    json_path = os.path.join(results_dir, "results.json")
    if not os.path.isfile(json_path):
        return []
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def load_all_results(result_dirs: list[str]) -> list[dict]:
    """여러 디렉토리의 results.json을 합쳐서 반환. source 필드 추가."""
    all_results = []
    for d in result_dirs:
        results = load_results(d)
        for r in results:
            r["source_dir"] = d
        all_results.extend(results)
    return all_results


def render_html(results: list[dict], results_dir: str) -> str:
    total = len(results)
    valid = sum(1 for r in results if r.get("is_valid_plate"))
    ocr_detected = sum(1 for r in results if r.get("text"))

    # TOP 10 by OCR confidence
    top10 = sorted(results, key=lambda r: r.get("ocr_confidence", 0), reverse=True)[:10]

    rows = ""
    for i, r in enumerate(results):
        idx = r.get("index", i)
        text = r.get("text", "") or "—"
        ocr_conf = r.get("ocr_confidence", 0)
        det_conf = r.get("detection_confidence", 0)
        sharpness = r.get("sharpness", 0)
        pattern = r.get("pattern_score", 0)
        method = r.get("detection_method", "—")
        valid_mark = "✅" if r.get("is_valid_plate") else "❌"
        frame = r.get("frame_idx", "—")

        img_url = f"/image/{results_dir}/plate_{idx:04d}.png"
        pre_url = f"/image/{results_dir}/plate_{idx:04d}_preprocessed.png"

        conf_class = "high" if ocr_conf >= 0.7 else ("mid" if ocr_conf >= 0.3 else "low")

        rows += f"""
        <tr class="{'valid-row' if r.get('is_valid_plate') else ''}">
          <td class="num">#{idx}</td>
          <td class="plate-text">{text}</td>
          <td class="conf {conf_class}">{ocr_conf:.3f}</td>
          <td>{det_conf:.3f}</td>
          <td>{pattern:.3f}</td>
          <td>{sharpness:.1f}</td>
          <td class="frame">{frame}</td>
          <td><span class="method {method}">{method}</span></td>
          <td class="valid-cell">{valid_mark}</td>
          <td class="img-cell">
            <img src="{img_url}" alt="plate_{idx}" loading="lazy"
                 onclick="openModal('{img_url}', '{pre_url}', '{text}', {ocr_conf:.3f})">
          </td>
        </tr>
        """

    top10_cards = ""
    for rank, r in enumerate(top10, 1):
        idx = r.get("index", 0)
        text = r.get("text", "") or "—"
        ocr_conf = r.get("ocr_confidence", 0)
        img_url = f"/image/{results_dir}/plate_{idx:04d}.png"
        valid_cls = "card-valid" if r.get("is_valid_plate") else "card-invalid"
        top10_cards += f"""
        <div class="top-card {valid_cls}">
          <div class="top-rank">#{rank}</div>
          <img src="{img_url}" alt="{text}">
          <div class="top-plate">{text}</div>
          <div class="top-conf">{ocr_conf:.3f}</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>번호판 인식 결과 뷰어</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', -apple-system, sans-serif; background: #0f1117; color: #e0e0e0; }}

  header {{
    background: linear-gradient(135deg, #1a1d27 0%, #252836 100%);
    padding: 24px 32px;
    border-bottom: 1px solid #2d3148;
    display: flex; align-items: center; gap: 16px;
  }}
  header h1 {{ font-size: 1.6rem; font-weight: 700; color: #fff; }}
  header .badge {{ background: #3d5afe; color: #fff; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; }}

  .stats {{
    display: flex; gap: 16px; padding: 24px 32px; flex-wrap: wrap;
  }}
  .stat-card {{
    background: #1a1d27; border: 1px solid #2d3148; border-radius: 12px;
    padding: 16px 24px; min-width: 140px; text-align: center;
  }}
  .stat-card .value {{ font-size: 2rem; font-weight: 700; color: #3d8afe; }}
  .stat-card .label {{ font-size: 0.8rem; color: #888; margin-top: 4px; }}

  .section {{ padding: 0 32px 32px; }}
  .section h2 {{ font-size: 1.1rem; color: #aaa; margin-bottom: 16px; letter-spacing: 0.05em; }}

  /* TOP 10 */
  .top10 {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .top-card {{
    background: #1a1d27; border: 1px solid #2d3148; border-radius: 10px;
    padding: 12px; width: 120px; text-align: center; transition: transform 0.15s;
    cursor: pointer;
  }}
  .top-card:hover {{ transform: translateY(-2px); border-color: #3d5afe; }}
  .top-card img {{ width: 100%; height: 50px; object-fit: contain; border-radius: 4px; background: #0f1117; }}
  .top-rank {{ font-size: 0.7rem; color: #888; margin-bottom: 4px; }}
  .top-plate {{ font-size: 0.85rem; font-weight: 700; margin-top: 6px; color: #fff; word-break: break-all; }}
  .top-conf {{ font-size: 0.75rem; color: #4caf50; margin-top: 2px; }}
  .card-valid {{ border-color: #2e7d32; }}
  .card-invalid {{ border-color: #424242; }}

  /* 필터 */
  .filter-bar {{
    display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap;
  }}
  .filter-bar input, .filter-bar select {{
    background: #1a1d27; border: 1px solid #2d3148; color: #e0e0e0;
    padding: 8px 14px; border-radius: 8px; font-size: 0.9rem; outline: none;
  }}
  .filter-bar input:focus, .filter-bar select:focus {{ border-color: #3d5afe; }}
  .filter-bar label {{ color: #888; font-size: 0.85rem; }}

  /* 테이블 */
  .table-wrap {{ overflow-x: auto; border-radius: 12px; border: 1px solid #2d3148; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  thead th {{
    background: #1e2133; padding: 12px 14px; text-align: left;
    color: #888; font-weight: 600; letter-spacing: 0.03em;
    position: sticky; top: 0; cursor: pointer; user-select: none;
    white-space: nowrap;
  }}
  thead th:hover {{ color: #fff; }}
  thead th.sorted-asc::after {{ content: ' ▲'; }}
  thead th.sorted-desc::after {{ content: ' ▼'; }}
  tbody tr {{ border-top: 1px solid #1e2133; transition: background 0.1s; }}
  tbody tr:hover {{ background: #1a1d27; }}
  tbody tr.valid-row {{ background: rgba(46, 125, 50, 0.05); }}
  td {{ padding: 10px 14px; vertical-align: middle; }}
  td.num {{ color: #666; font-size: 0.75rem; }}
  td.plate-text {{ font-weight: 700; font-size: 1rem; color: #fff; letter-spacing: 0.05em; min-width: 100px; }}
  td.conf {{ font-weight: 600; }}
  td.conf.high {{ color: #4caf50; }}
  td.conf.mid {{ color: #ff9800; }}
  td.conf.low {{ color: #f44336; }}
  td.frame {{ color: #666; font-size: 0.75rem; }}
  td.valid-cell {{ text-align: center; font-size: 1rem; }}
  .method {{ padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
  .method.plate_model {{ background: #1a3a6b; color: #7eb8f7; }}
  .method.coco_fallback {{ background: #3a1a1a; color: #f77; }}
  td.img-cell img {{
    height: 36px; max-width: 120px; object-fit: contain;
    cursor: pointer; border-radius: 4px; background: #0f1117;
    transition: transform 0.15s; border: 1px solid #2d3148;
  }}
  td.img-cell img:hover {{ transform: scale(1.5); z-index: 10; position: relative; }}

  /* 모달 */
  .modal {{
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85);
    z-index: 1000; align-items: center; justify-content: center;
  }}
  .modal.open {{ display: flex; }}
  .modal-box {{
    background: #1a1d27; border: 1px solid #2d3148; border-radius: 16px;
    padding: 24px; max-width: 680px; width: 90%; text-align: center;
  }}
  .modal-box h3 {{ font-size: 1.4rem; color: #fff; margin-bottom: 16px; }}
  .modal-imgs {{ display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }}
  .modal-imgs div {{ text-align: center; }}
  .modal-imgs span {{ display: block; font-size: 0.75rem; color: #888; margin-bottom: 4px; }}
  .modal-imgs img {{
    max-width: 280px; max-height: 200px; object-fit: contain;
    border-radius: 8px; background: #0f1117; border: 1px solid #2d3148;
  }}
  .modal-close {{
    margin-top: 16px; background: #2d3148; color: #e0e0e0;
    border: none; padding: 8px 24px; border-radius: 8px; cursor: pointer;
    font-size: 0.9rem;
  }}
  .modal-close:hover {{ background: #3d4468; }}

  .footer {{ text-align: center; padding: 32px; color: #444; font-size: 0.8rem; }}
</style>
</head>
<body>

<header>
  <h1>🚗 번호판 인식 결과 뷰어</h1>
  <span class="badge">{results_dir}</span>
</header>

<div class="stats">
  <div class="stat-card"><div class="value">{total}</div><div class="label">전체 탐지</div></div>
  <div class="stat-card"><div class="value" style="color:#4caf50">{valid}</div><div class="label">유효 번호판</div></div>
  <div class="stat-card"><div class="value" style="color:#ff9800">{ocr_detected}</div><div class="label">OCR 인식</div></div>
  <div class="stat-card"><div class="value" style="color:#9c27b0">{total - valid}</div><div class="label">미확인</div></div>
</div>

<div class="section">
  <h2>TOP 10 (OCR 신뢰도 순)</h2>
  <div class="top10">{top10_cards}</div>
</div>

<div class="section">
  <h2>전체 결과</h2>
  <div class="filter-bar">
    <input type="text" id="search" placeholder="번호판 검색..." oninput="filterTable()">
    <select id="filterValid" onchange="filterTable()">
      <option value="">전체</option>
      <option value="valid">유효만</option>
      <option value="invalid">미확인만</option>
    </select>
    <select id="filterMethod" onchange="filterTable()">
      <option value="">모든 방식</option>
      <option value="plate_model">plate_model</option>
      <option value="coco_fallback">coco_fallback</option>
    </select>
    <label><input type="checkbox" id="hideEmpty" onchange="filterTable()"> 텍스트 없는 항목 숨기기</label>
  </div>
  <div class="table-wrap">
    <table id="mainTable">
      <thead>
        <tr>
          <th onclick="sortTable(0)" data-col="0">#</th>
          <th onclick="sortTable(1)" data-col="1">번호판</th>
          <th onclick="sortTable(2)" data-col="2">OCR신뢰도</th>
          <th onclick="sortTable(3)" data-col="3">탐지신뢰도</th>
          <th onclick="sortTable(4)" data-col="4">패턴점수</th>
          <th onclick="sortTable(5)" data-col="5">선명도</th>
          <th onclick="sortTable(6)" data-col="6">프레임</th>
          <th>방식</th>
          <th>유효</th>
          <th>이미지</th>
        </tr>
      </thead>
      <tbody id="tableBody">
        {rows}
      </tbody>
    </table>
  </div>
</div>

<div class="modal" id="modal" onclick="closeModal(event)">
  <div class="modal-box">
    <h3 id="modalTitle"></h3>
    <div class="modal-imgs">
      <div><span>원본</span><img id="modalImg" src="" alt=""></div>
      <div><span>전처리</span><img id="modalPre" src="" alt=""></div>
    </div>
    <button class="modal-close" onclick="document.getElementById('modal').classList.remove('open')">닫기</button>
  </div>
</div>

<div class="footer">plate_server.py — 로컬 전용 뷰어</div>

<script>
let sortCol = -1, sortAsc = true;

function filterTable() {{
  const search = document.getElementById('search').value.toLowerCase();
  const filterValid = document.getElementById('filterValid').value;
  const filterMethod = document.getElementById('filterMethod').value;
  const hideEmpty = document.getElementById('hideEmpty').checked;
  const rows = document.querySelectorAll('#tableBody tr');
  rows.forEach(row => {{
    const cells = row.querySelectorAll('td');
    const text = cells[1].textContent.trim().toLowerCase();
    const isValid = row.classList.contains('valid-row');
    const method = cells[7].textContent.trim();
    const isEmpty = text === '—' || text === '';
    let show = true;
    if (search && !text.includes(search)) show = false;
    if (filterValid === 'valid' && !isValid) show = false;
    if (filterValid === 'invalid' && isValid) show = false;
    if (filterMethod && !method.includes(filterMethod)) show = false;
    if (hideEmpty && isEmpty) show = false;
    row.style.display = show ? '' : 'none';
  }});
}}

function sortTable(col) {{
  const headers = document.querySelectorAll('thead th');
  headers.forEach((h, i) => {{
    h.classList.remove('sorted-asc', 'sorted-desc');
  }});
  if (sortCol === col) sortAsc = !sortAsc;
  else {{ sortCol = col; sortAsc = false; }}
  headers[col].classList.add(sortAsc ? 'sorted-asc' : 'sorted-desc');

  const tbody = document.getElementById('tableBody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {{
    const aVal = a.querySelectorAll('td')[col].textContent.trim();
    const bVal = b.querySelectorAll('td')[col].textContent.trim();
    const aNum = parseFloat(aVal);
    const bNum = parseFloat(bVal);
    if (!isNaN(aNum) && !isNaN(bNum)) return sortAsc ? aNum - bNum : bNum - aNum;
    return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

function openModal(imgUrl, preUrl, text, conf) {{
  document.getElementById('modalTitle').textContent = text + ' (OCR: ' + conf + ')';
  document.getElementById('modalImg').src = imgUrl;
  document.getElementById('modalPre').src = preUrl;
  document.getElementById('modal').classList.add('open');
}}

function closeModal(e) {{
  if (e.target === document.getElementById('modal'))
    document.getElementById('modal').classList.remove('open');
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') document.getElementById('modal').classList.remove('open');
}});

// 기본 정렬: OCR 신뢰도 내림차순
window.addEventListener('load', () => sortTable(2));
</script>
</body>
</html>
"""


def render_multi_html(all_results: list[dict], dirs: list[str]) -> str:
    """멀티 디렉토리 통합 뷰 HTML."""
    total = len(all_results)
    valid = sum(1 for r in all_results if r.get("is_valid_plate"))

    dir_links = ""
    for d in dirs:
        count = sum(1 for r in all_results if r.get("source_dir") == d)
        dir_links += f'<a href="/view/{d}" style="display:inline-block;background:#1a1d27;border:1px solid #2d3148;border-radius:8px;padding:10px 20px;margin:6px;color:#e0e0e0;text-decoration:none;font-size:0.9rem;">{d} <span style="color:#3d5afe;">({count})</span></a>\n'

    # TOP 10 across all dirs
    top10 = sorted(all_results, key=lambda r: r.get("ocr_confidence", 0), reverse=True)[:10]
    top10_cards = ""
    for rank, r in enumerate(top10, 1):
        idx = r.get("index", 0)
        text = r.get("text", "") or "—"
        ocr_conf = r.get("ocr_confidence", 0)
        src = r.get("source_dir", "")
        img_url = f"/image/{src}/plate_{idx:04d}.png"
        valid_cls = "card-valid" if r.get("is_valid_plate") else "card-invalid"
        top10_cards += f"""
        <div class="top-card {valid_cls}" style="background:#1a1d27;border:1px solid #2d3148;border-radius:10px;padding:12px;width:120px;text-align:center;">
          <div style="font-size:0.7rem;color:#888;">#{rank}</div>
          <img src="{img_url}" alt="{text}" style="width:100%;height:50px;object-fit:contain;border-radius:4px;background:#0f1117;">
          <div style="font-size:0.85rem;font-weight:700;margin-top:6px;color:#fff;word-break:break-all;">{text}</div>
          <div style="font-size:0.75rem;color:#4caf50;">{ocr_conf:.3f}</div>
          <div style="font-size:0.65rem;color:#666;">{src}</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>번호판 인식 - 통합 뷰어</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; }}
  header {{ background: linear-gradient(135deg, #1a1d27, #252836); padding: 24px 32px; border-bottom: 1px solid #2d3148; }}
  header h1 {{ font-size: 1.6rem; color: #fff; }}
  .section {{ padding: 24px 32px; }}
  .section h2 {{ font-size: 1.1rem; color: #aaa; margin-bottom: 12px; }}
  .stats {{ display: flex; gap: 16px; padding: 24px 32px; }}
  .stat-card {{ background: #1a1d27; border: 1px solid #2d3148; border-radius: 12px; padding: 16px 24px; min-width: 140px; text-align: center; }}
  .stat-card .value {{ font-size: 2rem; font-weight: 700; color: #3d8afe; }}
  .stat-card .label {{ font-size: 0.8rem; color: #888; margin-top: 4px; }}
  .top10 {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .footer {{ text-align: center; padding: 32px; color: #444; font-size: 0.8rem; }}
</style>
</head>
<body>
<header><h1>번호판 인식 통합 뷰어</h1></header>
<div class="stats">
  <div class="stat-card"><div class="value">{total}</div><div class="label">전체 탐지</div></div>
  <div class="stat-card"><div class="value" style="color:#4caf50">{valid}</div><div class="label">유효 번호판</div></div>
  <div class="stat-card"><div class="value" style="color:#9c27b0">{len(dirs)}</div><div class="label">결과 폴더</div></div>
</div>
<div class="section">
  <h2>결과 디렉토리 선택</h2>
  <div>{dir_links}</div>
</div>
<div class="section">
  <h2>TOP 10 (전체 통합, OCR 신뢰도 순)</h2>
  <div class="top10">{top10_cards}</div>
</div>
<div class="footer">plate_server.py — 멀티 디렉토리 통합 뷰어</div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    results_dir = RESULTS_DIR

    def log_message(self, fmt, *args):
        print(f"  [{self.address_string()}] {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # 이미지 서빙: /image/<dir>/<filename>
        if path.startswith("/image/"):
            file_rel = unquote(path[len("/image/"):])
            file_path = Path(file_rel)
            if file_path.exists() and file_path.is_file():
                mime, _ = mimetypes.guess_type(str(file_path))
                mime = mime or "application/octet-stream"
                data = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._404()
            return

        # JSON API: /api/results
        if path == "/api/results":
            results = load_results(self.results_dir)
            body = json.dumps(results, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 특정 디렉토리 뷰: /view/<dir_name>
        if path.startswith("/view/"):
            dir_name = unquote(path[len("/view/"):]).strip("/")
            if dir_name and os.path.isdir(dir_name):
                results = load_results(dir_name)
                html = render_html(results, dir_name)
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._404()
            return

        # 메인 페이지: 자동 스캔한 디렉토리 목록 또는 단일 뷰
        if path in ("/", "/index.html"):
            all_dirs = find_all_result_dirs(".")
            if len(all_dirs) <= 1:
                # 단일 디렉토리 → 기존 동작
                use_dir = all_dirs[0] if all_dirs else self.results_dir
                results = load_results(use_dir)
                html = render_html(results, use_dir)
            else:
                # 멀티 디렉토리 → 통합 뷰 + 디렉토리 선택 링크
                all_results = load_all_results(all_dirs)
                html = render_multi_html(all_results, all_dirs)
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self._404()

    def _404(self):
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Not Found")


def main():
    parser = argparse.ArgumentParser(description="번호판 인식 결과 로컬 뷰어")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"포트 번호 (기본: {DEFAULT_PORT})")
    parser.add_argument("--results", default=None, help="결과 디렉토리 (미지정 시 자동 스캔)")
    args = parser.parse_args()

    # 자동 스캔 또는 지정 디렉토리
    if args.results:
        if not os.path.isdir(args.results):
            print(f"[오류] 결과 디렉토리를 찾을 수 없습니다: {args.results}")
            return
        Handler.results_dir = args.results
        found_dirs = [args.results]
    else:
        found_dirs = find_all_result_dirs(".")
        if not found_dirs:
            print("[오류] plate_results* 디렉토리를 찾을 수 없습니다.")
            print("  python plate_server.py --results <디렉토리> 로 지정하세요.")
            return
        Handler.results_dir = found_dirs[0]

    server = HTTPServer(("localhost", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print()
    print("=" * 50)
    print("  번호판 인식 결과 뷰어")
    print("=" * 50)
    print(f"  발견된 결과 디렉토리:")
    for d in found_dirs:
        count = len(load_results(d))
        print(f"    - {d} ({count}건)")
    print(f"  서버 주소:     {url}")
    print(f"  JSON API:      {url}/api/results")
    print()
    print("  브라우저에서 위 주소를 열어주세요.")
    print("  종료: Ctrl+C")
    print("=" * 50)
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  서버 종료.")


if __name__ == "__main__":
    main()
