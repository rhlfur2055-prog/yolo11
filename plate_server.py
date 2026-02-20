"""
plate_server.py - 번호판 인식 웹 서버 (Flask)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용법:
    python plate_server.py
    python plate_server.py --port 5000

기능:
    - http://localhost:5000 에서 영상 업로드
    - 번호판 인식 엔진(plate_recognition_4k)으로 처리
    - 실시간 결과 JSON + 웹 UI 표시
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify, send_from_directory, Response
from plate_recognition_4k import PlateRecognizer, NumpyEncoder

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_BASE = os.path.join(os.path.dirname(__file__), "server_results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_BASE, exist_ok=True)

# 진행중인 작업 상태
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# 기존 결과 디렉토리 자동 스캔
RESULT_DIR_PREFIX = "plate_results"


def find_all_result_dirs(base_dir: str = ".") -> list[str]:
    dirs = []
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith(RESULT_DIR_PREFIX):
            json_path = os.path.join(full, "results.json")
            if os.path.isfile(json_path):
                dirs.append(name)
    # server_results 하위도 탐색
    if os.path.isdir(RESULTS_BASE):
        for name in sorted(os.listdir(RESULTS_BASE)):
            full = os.path.join(RESULTS_BASE, name)
            rj = os.path.join(full, "results.json")
            if os.path.isdir(full) and os.path.isfile(rj):
                dirs.append(os.path.join("server_results", name))
    return dirs


def process_video_job(job_id: str, video_path: str):
    """백그라운드 영상 처리 스레드."""
    result_dir = os.path.join(RESULTS_BASE, job_id)
    os.makedirs(result_dir, exist_ok=True)

    with _jobs_lock:
        _jobs[job_id]["status"] = "loading"
        _jobs[job_id]["message"] = "모델 로딩중..."

    try:
        recognizer = PlateRecognizer()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        with _jobs_lock:
            _jobs[job_id].update({
                "status": "processing",
                "message": "영상 처리중...",
                "total_frames": total_frames,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "processed_frames": 0,
                "plates_found": 0,
                "plates": [],
            })

        all_results = recognizer.process_video(
            video_path,
            output_dir=result_dir,
            progress_callback=lambda frame_idx, det_count: _update_progress(
                job_id, frame_idx, total_frames, det_count
            ),
        )

        confirmed = recognizer.get_confirmed_plates()

        # 결과 저장
        _EXCLUDE_KEYS = {"plate_image", "preprocessed_image", "plate_img", "preprocessed"}
        clean_results = []
        for r in all_results:
            clean = {k: v for k, v in r.items()
                     if k not in _EXCLUDE_KEYS and not isinstance(v, np.ndarray)}
            clean_results.append(clean)

        with open(os.path.join(result_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        clean_confirmed = []
        for c in confirmed:
            cc = {k: v for k, v in c.items()
                  if k not in _EXCLUDE_KEYS and not isinstance(v, np.ndarray)}
            clean_confirmed.append(cc)

        with _jobs_lock:
            _jobs[job_id].update({
                "status": "completed",
                "message": "완료",
                "plates": clean_results,
                "confirmed": clean_confirmed,
                "total_plates": len(clean_results),
                "total_confirmed": len(clean_confirmed),
            })

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "error",
                "message": str(e),
            })


def _update_progress(job_id: str, frame_idx: int, total: int, det_count: int):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["processed_frames"] = frame_idx
            _jobs[job_id]["plates_found"] = det_count
            pct = frame_idx / total * 100 if total > 0 else 0
            _jobs[job_id]["progress_pct"] = round(pct, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML 페이지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INDEX_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>번호판 인식 시스템</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', -apple-system, sans-serif; background: #0f1117; color: #e0e0e0; min-height: 100vh; }
  header { background: linear-gradient(135deg, #1a1d27, #252836); padding: 20px 32px; border-bottom: 1px solid #2d3148; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.5rem; font-weight: 700; color: #fff; }
  header .badge { background: #3d5afe; color: #fff; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; }
  .container { max-width: 1000px; margin: 0 auto; padding: 32px; }

  /* 업로드 영역 */
  .upload-box {
    border: 2px dashed #3d5afe; border-radius: 16px; padding: 48px;
    text-align: center; cursor: pointer; transition: all 0.2s;
    background: rgba(61,90,254,0.05); margin-bottom: 32px;
  }
  .upload-box:hover { background: rgba(61,90,254,0.12); border-color: #536dfe; }
  .upload-box.dragover { background: rgba(61,90,254,0.2); border-color: #7c8eff; }
  .upload-box h2 { font-size: 1.3rem; color: #7c8eff; margin-bottom: 8px; }
  .upload-box p { color: #888; font-size: 0.9rem; }
  .upload-box input { display: none; }

  /* 진행률 */
  .progress-section { display: none; margin-bottom: 32px; }
  .progress-section.active { display: block; }
  .progress-bar-wrap { background: #1a1d27; border-radius: 8px; height: 24px; overflow: hidden; border: 1px solid #2d3148; }
  .progress-bar { height: 100%; background: linear-gradient(90deg, #3d5afe, #536dfe); border-radius: 8px; transition: width 0.3s; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600; min-width: 40px; }
  .progress-info { display: flex; gap: 24px; margin-top: 12px; flex-wrap: wrap; }
  .progress-info .item { background: #1a1d27; border: 1px solid #2d3148; border-radius: 8px; padding: 12px 20px; }
  .progress-info .item .val { font-size: 1.4rem; font-weight: 700; color: #3d8afe; }
  .progress-info .item .lbl { font-size: 0.75rem; color: #888; margin-top: 2px; }

  /* 결과 */
  .results-section { display: none; }
  .results-section.active { display: block; }
  .results-section h2 { font-size: 1.2rem; color: #aaa; margin-bottom: 16px; }
  .confirmed-list { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }
  .plate-card { background: #1a1d27; border: 1px solid #2d3148; border-radius: 12px; padding: 16px; min-width: 160px; text-align: center; }
  .plate-card .plate-text { font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 4px; letter-spacing: 0.08em; }
  .plate-card .plate-conf { font-size: 0.8rem; color: #4caf50; }
  .plate-card .plate-frames { font-size: 0.7rem; color: #666; margin-top: 4px; }

  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 16px; }
  thead th { background: #1e2133; padding: 10px 12px; text-align: left; color: #888; font-weight: 600; position: sticky; top: 0; }
  tbody tr { border-top: 1px solid #1e2133; }
  tbody tr:hover { background: #1a1d27; }
  td { padding: 8px 12px; }
  .text-col { font-weight: 700; color: #fff; letter-spacing: 0.05em; }
  .conf-high { color: #4caf50; font-weight: 600; }
  .conf-mid { color: #ff9800; }
  .conf-low { color: #f44336; }
  .method-tag { padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
  .method-tag.plate_model { background: #1a3a6b; color: #7eb8f7; }
  .method-tag.log_ocr { background: #1a4a2b; color: #7ef7a8; }

  /* 기존 결과 */
  .history-section { margin-top: 32px; padding-top: 24px; border-top: 1px solid #2d3148; }
  .history-section h2 { font-size: 1.1rem; color: #aaa; margin-bottom: 12px; }
  .dir-links { display: flex; flex-wrap: wrap; gap: 8px; }
  .dir-link { background: #1a1d27; border: 1px solid #2d3148; border-radius: 8px; padding: 10px 20px; color: #e0e0e0; text-decoration: none; font-size: 0.85rem; transition: border-color 0.2s; }
  .dir-link:hover { border-color: #3d5afe; }
  .dir-link span { color: #3d5afe; }

  .footer { text-align: center; padding: 32px; color: #444; font-size: 0.8rem; }
</style>
</head>
<body>

<header>
  <h1>번호판 인식 시스템</h1>
  <span class="badge">v6 AI Hub</span>
</header>

<div class="container">
  <!-- 업로드 -->
  <div class="upload-box" id="uploadBox" onclick="document.getElementById('fileInput').click()">
    <h2>영상 파일 업로드</h2>
    <p>클릭하거나 파일을 드래그하세요 (.mp4, .avi, .mkv)</p>
    <input type="file" id="fileInput" accept="video/*">
  </div>

  <!-- 진행률 -->
  <div class="progress-section" id="progressSection">
    <div class="progress-bar-wrap">
      <div class="progress-bar" id="progressBar" style="width: 0%">0%</div>
    </div>
    <div class="progress-info">
      <div class="item"><div class="val" id="statStatus">대기</div><div class="lbl">상태</div></div>
      <div class="item"><div class="val" id="statFrames">0</div><div class="lbl">처리 프레임</div></div>
      <div class="item"><div class="val" id="statPlates">0</div><div class="lbl">인식 번호판</div></div>
      <div class="item"><div class="val" id="statRes">-</div><div class="lbl">해상도</div></div>
    </div>
  </div>

  <!-- 결과 -->
  <div class="results-section" id="resultsSection">
    <h2>확정 번호판</h2>
    <div class="confirmed-list" id="confirmedList"></div>

    <h2>전체 인식 결과 (<span id="totalCount">0</span>건)</h2>
    <div style="overflow-x:auto; border-radius:12px; border:1px solid #2d3148;">
    <table>
      <thead>
        <tr><th>#</th><th>번호판</th><th>OCR</th><th>탐지</th><th>패턴</th><th>방식</th></tr>
      </thead>
      <tbody id="resultsBody"></tbody>
    </table>
    </div>

    <div style="margin-top:16px;">
      <a id="jsonLink" href="#" target="_blank" style="color:#3d5afe; font-size:0.85rem;">결과 JSON 다운로드</a>
    </div>
  </div>

  <!-- 기존 결과 -->
  <div class="history-section" id="historySection"></div>
</div>

<div class="footer">plate_server.py Flask v6 | plate_recognition_4k engine</div>

<script>
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');

// 드래그 앤 드롭
uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', e => {
  e.preventDefault(); uploadBox.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length > 0) uploadFile(fileInput.files[0]); });

function uploadFile(file) {
  const form = new FormData();
  form.append('video', file);
  uploadBox.innerHTML = '<h2>업로드중: ' + file.name + '</h2><p>' + (file.size / 1024 / 1024).toFixed(1) + ' MB</p>';
  progressSection.classList.add('active');
  resultsSection.classList.remove('active');
  document.getElementById('statStatus').textContent = '업로드중...';

  fetch('/api/upload', { method: 'POST', body: form })
    .then(r => r.json())
    .then(data => {
      if (data.job_id) pollProgress(data.job_id);
      else alert('업로드 실패: ' + (data.error || '알 수 없는 오류'));
    })
    .catch(err => alert('업로드 오류: ' + err));
}

function pollProgress(jobId) {
  const interval = setInterval(() => {
    fetch('/api/status/' + jobId)
      .then(r => r.json())
      .then(data => {
        const pct = data.progress_pct || 0;
        document.getElementById('progressBar').style.width = pct + '%';
        document.getElementById('progressBar').textContent = pct + '%';
        document.getElementById('statStatus').textContent = data.message || data.status;
        document.getElementById('statFrames').textContent = (data.processed_frames || 0) + '/' + (data.total_frames || '?');
        document.getElementById('statPlates').textContent = data.plates_found || 0;
        document.getElementById('statRes').textContent = data.resolution || '-';

        if (data.status === 'completed') {
          clearInterval(interval);
          document.getElementById('progressBar').style.width = '100%';
          document.getElementById('progressBar').textContent = '100%';
          showResults(data, jobId);
        } else if (data.status === 'error') {
          clearInterval(interval);
          document.getElementById('statStatus').textContent = 'ERROR: ' + data.message;
        }
      });
  }, 1000);
}

function showResults(data, jobId) {
  resultsSection.classList.add('active');
  document.getElementById('jsonLink').href = '/api/results/' + jobId;

  // 확정 번호판
  const confirmed = data.confirmed || [];
  const clist = document.getElementById('confirmedList');
  clist.innerHTML = '';
  confirmed.forEach(c => {
    const card = document.createElement('div');
    card.className = 'plate-card';
    card.innerHTML = '<div class="plate-text">' + (c.text || '-') + '</div>'
      + '<div class="plate-conf">score: ' + (c.pattern_score || 0).toFixed(2) + '</div>'
      + '<div class="plate-frames">' + (c.detection_count || 0) + ' frames</div>';
    clist.appendChild(card);
  });

  // 전체 결과
  const plates = data.plates || [];
  document.getElementById('totalCount').textContent = plates.length;
  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = '';
  plates.forEach((p, i) => {
    const ocr = (p.ocr_confidence || 0);
    const confClass = ocr >= 0.7 ? 'conf-high' : (ocr >= 0.3 ? 'conf-mid' : 'conf-low');
    const method = p.detection_method || '-';
    const methodClass = method === 'plate_model' ? 'plate_model' : (method === 'log_ocr' ? 'log_ocr' : '');
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + i + '</td>'
      + '<td class="text-col">' + (p.text || '-') + '</td>'
      + '<td class="' + confClass + '">' + ocr.toFixed(3) + '</td>'
      + '<td>' + (p.det_confidence || 0).toFixed(3) + '</td>'
      + '<td>' + (p.pattern_score || 0).toFixed(2) + '</td>'
      + '<td><span class="method-tag ' + methodClass + '">' + method + '</span></td>';
    tbody.appendChild(tr);
  });
}

// 기존 결과 디렉토리 로드
fetch('/api/history')
  .then(r => r.json())
  .then(dirs => {
    if (dirs.length === 0) return;
    const sec = document.getElementById('historySection');
    let html = '<h2>기존 인식 결과</h2><div class="dir-links">';
    dirs.forEach(d => {
      html += '<a class="dir-link" href="/view/' + d.name + '">' + d.name + ' <span>(' + d.count + '건)</span></a>';
    });
    html += '</div>';
    sec.innerHTML = html;
  });
</script>
</body>
</html>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Flask 라우트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "video 파일이 없습니다"}), 400

    video = request.files["video"]
    if not video.filename:
        return jsonify({"error": "파일명이 없습니다"}), 400

    job_id = uuid.uuid4().hex[:8]
    ext = Path(video.filename).suffix or ".mp4"
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    video.save(save_path)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "message": "대기중...",
            "filename": video.filename,
            "video_path": save_path,
            "progress_pct": 0,
        }

    thread = threading.Thread(target=process_video_job, args=(job_id, save_path), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다"}), 404
    return Response(
        json.dumps(job, ensure_ascii=False, cls=NumpyEncoder),
        content_type="application/json; charset=utf-8",
    )


@app.route("/api/results/<job_id>")
def api_results(job_id):
    result_path = os.path.join(RESULTS_BASE, job_id, "results.json")
    if not os.path.isfile(result_path):
        return jsonify({"error": "결과 없음"}), 404
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    return Response(
        json.dumps(data, ensure_ascii=False, cls=NumpyEncoder),
        content_type="application/json; charset=utf-8",
    )


@app.route("/api/history")
def api_history():
    dirs = find_all_result_dirs(os.path.dirname(__file__) or ".")
    result = []
    for d in dirs:
        rj = os.path.join(d, "results.json")
        count = 0
        if os.path.isfile(rj):
            try:
                with open(rj, encoding="utf-8") as f:
                    count = len(json.load(f))
            except Exception:
                pass
        result.append({"name": d, "count": count})
    return jsonify(result)


@app.route("/view/<path:dir_name>")
def view_results(dir_name):
    rj = os.path.join(dir_name, "results.json")
    if not os.path.isfile(rj):
        return "결과 없음", 404
    with open(rj, encoding="utf-8") as f:
        data = json.load(f)
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2, cls=NumpyEncoder),
        content_type="application/json; charset=utf-8",
    )


@app.route("/image/<path:filepath>")
def serve_image(filepath):
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    return send_from_directory(directory, filename)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="번호판 인식 웹 서버")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print()
    print("=" * 50)
    print("  번호판 인식 웹 서버")
    print("=" * 50)
    print(f"  주소: http://localhost:{args.port}")
    print(f"  엔진: plate_recognition_4k v6 (AI Hub)")
    print()
    print("  기능:")
    print("    - 영상 업로드 + 실시간 번호판 인식")
    print("    - 인식 결과 JSON API")
    print("    - 기존 결과 뷰어")
    print()
    print("  종료: Ctrl+C")
    print("=" * 50)
    print()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
