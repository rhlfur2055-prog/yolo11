#!/usr/bin/env python3
"""
yolo26 프로젝트 완전 분석 스크립트
- 모든 .py 파일의 역할 파악
- 중복 함수/클래스 탐지
- import 의존성 맵
- 전처리/후처리 파이프라인 추적
- 실시간 처리 병목 후보 탐지
- 미사용 파일 탐지

사용법: python analyze_project.py > analysis_report.txt
"""

import os
import re
import ast
import sys
from collections import defaultdict, Counter
from pathlib import Path

# ============================================================
# 설정
# ============================================================
PROJECT_ROOT = r"C:\tool\yolo26-main"
SKIP_DIRS = {'.git', '__pycache__', '.claude', 'node_modules', '.venv', 'venv', 'runs'}
REPORT_FILE = os.path.join(PROJECT_ROOT, "analysis_report.txt")

# ============================================================
# 1. 파일 수집
# ============================================================
def collect_py_files(root):
    """모든 .py 파일 수집 (SKIP_DIRS 제외)"""
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith('.py'):
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root)
                py_files.append((rel, full))
    return sorted(py_files)

# ============================================================
# 2. 파일별 AST 분석
# ============================================================
def analyze_file(filepath):
    """단일 .py 파일을 AST로 파싱하여 정보 추출"""
    info = {
        'imports': [],           # import 목록
        'from_imports': [],      # from X import Y 목록
        'functions': [],         # def 함수명 (top-level)
        'classes': [],           # class 명
        'class_methods': defaultdict(list),  # 클래스별 메서드
        'global_vars': [],       # 전역 변수 할당
        'has_main': False,       # if __name__ == '__main__' 존재?
        'has_argparse': False,   # argparse 사용?
        'has_flask': False,      # Flask 사용?
        'has_gradio': False,     # Gradio 사용?
        'has_streamlit': False,  # Streamlit 사용?
        'has_cv2_videocapture': False,  # cv2.VideoCapture 사용?
        'has_yolo': False,       # YOLO 모델 로드?
        'has_ocr': [],           # OCR 엔진 사용 목록
        'line_count': 0,
        'docstring': '',
        'error': None,
        'raw_imports_str': [],   # 문자열 기반 import 추출 (AST 실패 시 fallback)
        'key_patterns': [],      # 주요 패턴 탐지
    }

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
    except Exception as e:
        info['error'] = f"읽기 실패: {e}"
        return info

    lines = source.split('\n')
    info['line_count'] = len(lines)

    # --- 문자열 기반 패턴 탐지 (AST 파싱 실패해도 작동) ---
    source_lower = source.lower()

    # OCR 엔진 탐지
    if 'paddleocr' in source_lower or 'from paddleocr' in source_lower:
        info['has_ocr'].append('PaddleOCR')
    if 'easyocr' in source_lower:
        info['has_ocr'].append('EasyOCR')
    if 'pytesseract' in source_lower or 'tesseract' in source_lower:
        info['has_ocr'].append('Tesseract')

    # YOLO 탐지
    if 'from ultralytics' in source or 'YOLO(' in source or 'yolo' in source_lower:
        info['has_yolo'] = True

    # 웹 프레임워크 탐지
    if 'flask' in source_lower and ('Flask(' in source or 'from flask' in source_lower):
        info['has_flask'] = True
    if 'gradio' in source_lower:
        info['has_gradio'] = True
    if 'streamlit' in source_lower or 'import st' in source:
        info['has_streamlit'] = True

    # VideoCapture 탐지
    if 'videocapture' in source_lower or 'cap.read' in source_lower:
        info['has_cv2_videocapture'] = True

    # argparse 탐지
    if 'argparse' in source or 'ArgumentParser' in source:
        info['has_argparse'] = True

    # 주요 키워드 패턴 탐지
    pattern_keywords = {
        '전처리/preprocess': r'(?i)(preprocess|전처리|pre_process|enhance|sharpen|denoise|resize|gray|threshold|blur|clahe|adaptive)',
        '후처리/postprocess': r'(?i)(postprocess|후처리|post_process|nms|filter_result|merge_result|ensemble|voting)',
        'ROI/크롭': r'(?i)(roi|crop|region.of.interest|bbox.*crop|plate.*crop)',
        'DB/저장': r'(?i)(sqlite|mysql|postgres|database|db_|\.db|insert.*into|create.*table)',
        '웹소켓/실시간통신': r'(?i)(websocket|socketio|socket\.io|emit\(|sse|server.sent)',
        'GUI': r'(?i)(tkinter|pyqt|pyside|ttk\.|tk\.|canvas|mainloop|root\.)',
        '멀티스레드/프로세스': r'(?i)(threading|multiprocessing|thread|queue|concurrent|pool|async|await)',
        '영상스트리밍': r'(?i)(rtsp|rtmp|hls|m3u8|stream|mjpeg|youtube.*url|yt_dlp|pafy)',
        '번호판정규식': r'(?i)(plate.*pattern|regex.*plate|번호판.*패턴|[0-9]+[가-힣]|re\.match|re\.compile)',
        'GPU/CUDA': r'(?i)(cuda|gpu|device.*0|\.to\(|half\(\)|fp16)',
        '로깅': r'(?i)(logging\.|logger|log_file|print.*result|save.*log)',
        '설정/Config': r'(?i)(config|\.yaml|\.yml|\.json|\.ini|load.*config|parse.*config)',
    }

    for label, pattern in pattern_keywords.items():
        if re.search(pattern, source):
            info['key_patterns'].append(label)

    # --- AST 파싱 ---
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        info['error'] = f"SyntaxError: {e.lineno}:{e.offset} - {e.msg}"
        # AST 실패해도 문자열 기반 import 추출
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                info['raw_imports_str'].append(stripped)
        return info

    # 모듈 docstring
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, (ast.Str, ast.Constant))):
        val = tree.body[0].value
        info['docstring'] = val.s if isinstance(val, ast.Str) else str(getattr(val, 'value', ''))
        info['docstring'] = info['docstring'][:200]  # 200자 제한

    for node in ast.iter_child_nodes(tree):
        # import X
        if isinstance(node, ast.Import):
            for alias in node.names:
                info['imports'].append(alias.name)

        # from X import Y
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            for alias in node.names:
                info['from_imports'].append(f"{mod}.{alias.name}")

        # def func():
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            info['functions'].append(node.name)

        # class Foo:
        elif isinstance(node, ast.ClassDef):
            info['classes'].append(node.name)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info['class_methods'][node.name].append(item.name)

        # 전역 변수 (X = ...)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    info['global_vars'].append(target.id)

    # if __name__ == '__main__' 탐지
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            try:
                test = ast.dump(node.test)
                if '__name__' in test and '__main__' in test:
                    info['has_main'] = True
            except:
                pass

    return info

# ============================================================
# 3. 역할 자동 분류
# ============================================================
def classify_role(relpath, info):
    """파일의 역할을 자동 분류"""
    roles = []
    name = os.path.basename(relpath).lower()

    # 진입점
    if info['has_main']:
        roles.append('🚀 실행가능(진입점)')
    if info['has_flask']:
        roles.append('🌐 Flask서버')
    if info['has_gradio']:
        roles.append('🌐 Gradio서버')
    if info['has_streamlit']:
        roles.append('🌐 Streamlit앱')

    # 핵심 기능
    if info['has_yolo']:
        roles.append('🔍 YOLO모델사용')
    if info['has_ocr']:
        roles.append(f"📝 OCR({','.join(info['has_ocr'])})")
    if info['has_cv2_videocapture']:
        roles.append('📹 영상캡처')

    # 패턴 기반
    for p in info['key_patterns']:
        if '전처리' in p:
            roles.append('⚙️ 전처리')
        elif '후처리' in p:
            roles.append('⚙️ 후처리')
        elif 'ROI' in p:
            roles.append('✂️ ROI/크롭')
        elif 'DB' in p:
            roles.append('💾 DB/저장')
        elif 'GUI' in p:
            roles.append('🖥️ GUI')
        elif '멀티스레드' in p:
            roles.append('⚡ 멀티스레드')
        elif '영상스트리밍' in p:
            roles.append('📡 스트리밍')
        elif '번호판정규식' in p:
            roles.append('🔤 번호판정규식')
        elif 'GPU' in p:
            roles.append('🎮 GPU사용')

    # 파일명 기반 추가 힌트
    if 'test' in name or 'debug' in name:
        roles.append('🧪 테스트/디버그')
    if 'config' in name or 'setting' in name:
        roles.append('⚙️ 설정파일')
    if 'util' in name or 'helper' in name or 'tool' in name:
        roles.append('🔧 유틸리티')
    if 'train' in name:
        roles.append('🏋️ 학습')
    if 'server' in name or 'api' in name or 'app' in name:
        roles.append('🌐 서버/API')

    if not roles:
        roles.append('❓ 역할불명')

    return list(dict.fromkeys(roles))  # 중복 제거, 순서 유지

# ============================================================
# 4. 중복 탐지
# ============================================================
def find_duplicates(all_info):
    """함수명/클래스명 중복 탐지"""
    func_map = defaultdict(list)  # 함수명 → [파일들]
    class_map = defaultdict(list)  # 클래스명 → [파일들]

    for relpath, info in all_info.items():
        for fn in info['functions']:
            if fn.startswith('_'):
                continue  # private 제외
            func_map[fn].append(relpath)
        for cls in info['classes']:
            class_map[cls].append(relpath)

    dup_funcs = {k: v for k, v in func_map.items() if len(v) > 1}
    dup_classes = {k: v for k, v in class_map.items() if len(v) > 1}
    return dup_funcs, dup_classes

# ============================================================
# 5. 파이프라인 추적
# ============================================================
def trace_pipeline(all_info):
    """전처리 → YOLO → OCR → 후처리 파이프라인 파일 매핑"""
    stages = {
        '1_입력(영상/이미지)': [],
        '2_전처리': [],
        '3_YOLO탐지': [],
        '4_ROI크롭': [],
        '5_OCR인식': [],
        '6_후처리(정규식/앙상블)': [],
        '7_출력(GUI/서버/저장)': [],
    }

    for relpath, info in all_info.items():
        if info['has_cv2_videocapture'] or '영상스트리밍' in str(info['key_patterns']):
            stages['1_입력(영상/이미지)'].append(relpath)
        if '전처리/preprocess' in str(info['key_patterns']):
            stages['2_전처리'].append(relpath)
        if info['has_yolo']:
            stages['3_YOLO탐지'].append(relpath)
        if 'ROI/크롭' in str(info['key_patterns']):
            stages['4_ROI크롭'].append(relpath)
        if info['has_ocr']:
            stages['5_OCR인식'].append(relpath)
        if '후처리/postprocess' in str(info['key_patterns']) or '번호판정규식' in str(info['key_patterns']):
            stages['6_후처리(정규식/앙상블)'].append(relpath)
        if info['has_flask'] or info['has_gradio'] or info['has_streamlit'] or 'GUI' in str(info['key_patterns']) or 'DB/저장' in str(info['key_patterns']):
            stages['7_출력(GUI/서버/저장)'].append(relpath)

    return stages

# ============================================================
# 6. 실시간 병목 후보 탐지
# ============================================================
def detect_bottlenecks(all_info):
    """실시간 처리 병목이 될 수 있는 패턴 탐지"""
    issues = []

    for relpath, info in all_info.items():
        # OCR 다중 호출 (느림의 주범)
        if len(info['has_ocr']) >= 2:
            issues.append(f"⚠️  [{relpath}] OCR 엔진 {len(info['has_ocr'])}개 동시 사용 ({', '.join(info['has_ocr'])}) → 프레임당 OCR 3번 = 병목 확정")

        # VideoCapture + OCR 같은 파일 (동기 처리 의심)
        if info['has_cv2_videocapture'] and info['has_ocr']:
            if '멀티스레드/프로세스' not in str(info['key_patterns']):
                issues.append(f"⚠️  [{relpath}] 영상캡처 + OCR이 같은 파일인데 멀티스레드 없음 → 동기 처리로 프레임 드롭 가능")

        # YOLO + OCR 같은 파일 (파이프라인이 한 파일에 몰림)
        if info['has_yolo'] and info['has_ocr']:
            issues.append(f"🔎 [{relpath}] YOLO+OCR 통합파일 → 분리 여부 확인 필요")

        # 파일이 너무 큼 (god object)
        if info['line_count'] > 500:
            issues.append(f"📏 [{relpath}] {info['line_count']}줄 → 파일이 너무 커서 유지보수 어려움")

    return issues

# ============================================================
# 7. 비실행 파일 탐지 (import 안 되는 고아 파일)
# ============================================================
def find_orphan_files(all_info):
    """다른 파일에서 import 되지 않고, 진입점도 아닌 파일"""
    # 모든 import 대상 모듈 수집
    imported_modules = set()
    for relpath, info in all_info.items():
        for imp in info['imports']:
            imported_modules.add(imp.split('.')[0])
        for imp in info['from_imports']:
            imported_modules.add(imp.split('.')[0])

    orphans = []
    for relpath, info in all_info.items():
        module_name = os.path.splitext(os.path.basename(relpath))[0]
        # 진입점이면 OK
        if info['has_main'] or info['has_flask'] or info['has_gradio']:
            continue
        # 다른 파일에서 import 되면 OK
        if module_name in imported_modules:
            continue
        # scripts/ 폴더는 독립 실행일 수 있으므로 제외
        if 'scripts' in relpath.lower():
            continue
        orphans.append(relpath)

    return orphans

# ============================================================
# 8. 비-Python 주요 파일 수집
# ============================================================
def collect_other_files(root):
    """주요 비-Python 파일 수집 (.bat, .yaml, .json, .sh, .cfg, .pt 등)"""
    important_exts = {'.bat', '.sh', '.yaml', '.yml', '.json', '.cfg', '.ini', '.pt', '.onnx', '.pth', '.pdf', '.md', '.txt'}
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        # dataset 폴더 내부는 건너뛰기 (이미지/라벨 수천 개)
        if 'dataset' in os.path.basename(dirpath).lower():
            dirnames.clear()
            files.append(os.path.relpath(dirpath, root) + "/ (데이터셋 폴더 - 내부 생략)")
            continue
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in important_exts:
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root)
                size = os.path.getsize(full)
                files.append(f"{rel} ({size:,} bytes)")
    return sorted(files)

# ============================================================
# 메인 리포트 생성
# ============================================================
def main():
    output_lines = []
    def out(text=""):
        output_lines.append(text)

    out("=" * 80)
    out("  YOLO26 프로젝트 완전 분석 리포트")
    out(f"  분석 대상: {PROJECT_ROOT}")
    out("=" * 80)

    # 파일 수집
    py_files = collect_py_files(PROJECT_ROOT)
    out(f"\n📂 Python 파일 총 {len(py_files)}개 발견\n")

    # 전체 분석
    all_info = {}
    for relpath, fullpath in py_files:
        info = analyze_file(fullpath)
        all_info[relpath] = info

    # ─── 섹션 1: 파일별 상세 분석 ───
    out("=" * 80)
    out("  [1] 파일별 상세 분석")
    out("=" * 80)

    for relpath, info in all_info.items():
        roles = classify_role(relpath, info)
        out(f"\n{'─' * 70}")
        out(f"📄 {relpath}  ({info['line_count']}줄)")
        out(f"   역할: {' | '.join(roles)}")

        if info['docstring']:
            out(f"   설명: {info['docstring']}")
        if info['error']:
            out(f"   ❌ 파싱에러: {info['error']}")

        if info['classes']:
            out(f"   클래스: {', '.join(info['classes'])}")
            for cls, methods in info['class_methods'].items():
                out(f"      {cls} 메서드: {', '.join(methods)}")

        if info['functions']:
            out(f"   함수: {', '.join(info['functions'])}")

        if info['has_ocr']:
            out(f"   OCR엔진: {', '.join(info['has_ocr'])}")

        if info['key_patterns']:
            out(f"   탐지패턴: {', '.join(info['key_patterns'])}")

        # 핵심 import만 표시
        key_imports = [i for i in info['imports'] + [x.split('.')[0] for x in info['from_imports']]
                       if i in ('cv2', 'numpy', 'torch', 'ultralytics', 'flask', 'gradio',
                                'paddleocr', 'easyocr', 'pytesseract', 'threading', 'multiprocessing',
                                'asyncio', 'sqlite3', 'requests', 'fastapi', 'uvicorn', 'tkinter')]
        if key_imports:
            out(f"   핵심import: {', '.join(sorted(set(key_imports)))}")

    # ─── 섹션 2: 파이프라인 매핑 ───
    out(f"\n\n{'=' * 80}")
    out("  [2] 번호판 인식 파이프라인 매핑")
    out("=" * 80)
    out("  영상입력 → 전처리 → YOLO탐지 → ROI크롭 → OCR인식 → 후처리 → 출력\n")

    stages = trace_pipeline(all_info)
    for stage, files in stages.items():
        out(f"  {stage}:")
        if files:
            for f in files:
                out(f"    → {f}")
        else:
            out(f"    (해당 파일 없음)")
        out()

    # ─── 섹션 3: 중복 탐지 ───
    out(f"\n{'=' * 80}")
    out("  [3] 중복 함수/클래스 탐지")
    out("=" * 80)

    dup_funcs, dup_classes = find_duplicates(all_info)
    if dup_funcs:
        out("\n  🔴 중복 함수명:")
        for fn, files in sorted(dup_funcs.items()):
            out(f"    {fn}() → {', '.join(files)}")
    else:
        out("\n  ✅ 중복 함수 없음")

    if dup_classes:
        out("\n  🔴 중복 클래스명:")
        for cls, files in sorted(dup_classes.items()):
            out(f"    {cls} → {', '.join(files)}")
    else:
        out("\n  ✅ 중복 클래스 없음")

    # ─── 섹션 4: 실시간 병목 분석 ───
    out(f"\n\n{'=' * 80}")
    out("  [4] 실시간 처리 병목 후보")
    out("=" * 80)

    issues = detect_bottlenecks(all_info)
    if issues:
        for issue in issues:
            out(f"  {issue}")
    else:
        out("  ✅ 명확한 병목 패턴 미탐지")

    # ─── 섹션 5: 고아 파일 ───
    out(f"\n\n{'=' * 80}")
    out("  [5] 미사용 의심 파일 (import 안 됨 + 진입점 아님)")
    out("=" * 80)

    orphans = find_orphan_files(all_info)
    if orphans:
        for o in orphans:
            out(f"  🗑️  {o}")
    else:
        out("  ✅ 모든 파일이 사용됨")

    # ─── 섹션 6: 진입점 정리 ───
    out(f"\n\n{'=' * 80}")
    out("  [6] 실행 가능한 진입점 (if __name__ == '__main__')")
    out("=" * 80)

    for relpath, info in all_info.items():
        if info['has_main']:
            roles = classify_role(relpath, info)
            out(f"  ▶ {relpath} ({info['line_count']}줄) - {' | '.join(roles)}")

    # ─── 섹션 7: 비-Python 주요 파일 ───
    out(f"\n\n{'=' * 80}")
    out("  [7] 주요 비-Python 파일 (.bat, .yaml, .json, .pt 등)")
    out("=" * 80)

    other_files = collect_other_files(PROJECT_ROOT)
    for f in other_files:
        out(f"  {f}")

    # ─── 섹션 8: 통계 요약 ───
    out(f"\n\n{'=' * 80}")
    out("  [8] 통계 요약")
    out("=" * 80)

    total_lines = sum(info['line_count'] for info in all_info.values())
    ocr_files = [r for r, i in all_info.items() if i['has_ocr']]
    yolo_files = [r for r, i in all_info.items() if i['has_yolo']]
    server_files = [r for r, i in all_info.items() if i['has_flask'] or i['has_gradio'] or i['has_streamlit']]
    video_files = [r for r, i in all_info.items() if i['has_cv2_videocapture']]
    threaded_files = [r for r, i in all_info.items() if '멀티스레드/프로세스' in str(i['key_patterns'])]

    out(f"  총 Python 파일: {len(py_files)}개")
    out(f"  총 코드 라인: {total_lines:,}줄")
    out(f"  YOLO 사용 파일: {len(yolo_files)}개 → {', '.join(yolo_files) if yolo_files else 'N/A'}")
    out(f"  OCR 사용 파일: {len(ocr_files)}개 → {', '.join(ocr_files) if ocr_files else 'N/A'}")
    out(f"  서버/GUI 파일: {len(server_files)}개 → {', '.join(server_files) if server_files else 'N/A'}")
    out(f"  영상캡처 파일: {len(video_files)}개 → {', '.join(video_files) if video_files else 'N/A'}")
    out(f"  멀티스레드 파일: {len(threaded_files)}개 → {', '.join(threaded_files) if threaded_files else 'N/A'}")
    out(f"  중복 함수: {len(dup_funcs)}개")
    out(f"  중복 클래스: {len(dup_classes)}개")
    out(f"  고아 파일: {len(orphans)}개")
    out(f"  병목 이슈: {len(issues)}개")

    # ─── 섹션 9: CMD 12분할 운영 가이드 (예비) ───
    out(f"\n\n{'=' * 80}")
    out("  [9] CMD 12분할 실시간 파이프라인 구조 제안")
    out("=" * 80)
    out("""
  현재 구조에서 CMD 12개로 분리 운영하려면:

  [프론트엔드 그룹]
    CMD 1: Flask/Gradio 웹서버 (GUI 담당)
    CMD 2: 웹소켓 or SSE 실시간 스트림 전송

  [영상 입력 그룹]
    CMD 3: 카메라/RTSP 입력 + 프레임 큐 적재
    CMD 4: YouTube/파일 입력 + 프레임 큐 적재

  [AI 처리 그룹 - 핵심]
    CMD 5: YOLO 탐지 워커 #1 (GPU)
    CMD 6: YOLO 탐지 워커 #2 (GPU, 부하 분산)
    CMD 7: 전처리 (리사이즈/그레이/CLAHE)
    CMD 8: OCR 워커 #1 (PaddleOCR)
    CMD 9: OCR 워커 #2 (EasyOCR)
    CMD 10: OCR 워커 #3 (Tesseract)

  [후처리 그룹]
    CMD 11: 앙상블 투표 + 정규식 검증 + DB 저장
    CMD 12: 로깅/모니터링/성능 대시보드

  ⚠️ 핵심: 각 CMD 간 통신은 Redis Queue 또는 multiprocessing.Queue 사용
  ⚠️ 현재 코드가 이 구조인지, 아니면 단일 파일에 다 몰려있는지
     → 위 [2]번 파이프라인 매핑 결과로 판단 가능
""")

    # 파일 저장
    report_text = '\n'.join(output_lines)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Windows cp949 인코딩 이모지 출력 오류 방지
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print(report_text)
    print(f"\n\n💾 리포트 저장 완료: {REPORT_FILE}")

if __name__ == '__main__':
    main()
