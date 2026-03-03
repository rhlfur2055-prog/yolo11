#!/usr/bin/env python3
"""
plate_engine_pro.py + 관련 파일에서
전처리/후처리 파이프라인 코드를 정확히 추출

실행: python extract_pipeline.py > pipeline_map.txt
"""

import os
import re
import ast
import sys
import io

# Windows cp949 이모지 출력 오류 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = r"C:\tool\yolo26-main"
REPORT_FILE = os.path.join(PROJECT_ROOT, "pipeline_map.txt")

# ============================================================
# 핵심 파일 목록 (analysis_report 기반)
# ============================================================
KEY_FILES = [
    "plate_engine_pro.py",
    "plate_recognition_4k.py",
    "plate_gui.py",
    "plate_ocr_postfilter.py",
    "server.py",
]

# + scripts/ 폴더 내 .py 전부
# + 루트의 나머지 .py 전부

def collect_all_py(root):
    """루트 + scripts/ 의 모든 .py"""
    files = []
    for f in os.listdir(root):
        if f.endswith('.py'):
            files.append(f)
    scripts_dir = os.path.join(root, 'scripts')
    if os.path.isdir(scripts_dir):
        for f in os.listdir(scripts_dir):
            if f.endswith('.py'):
                files.append(os.path.join('scripts', f))
    return files

# ============================================================
# 함수 단위 추출 (AST 기반)
# ============================================================
def extract_functions_and_classes(filepath):
    """파일에서 모든 함수/클래스를 줄번호와 함께 추출"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
            lines = source.split('\n')
    except:
        return [], [], len([])

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], [], len(lines)

    functions = []
    classes = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # 함수의 끝 줄 찾기
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 10
            # docstring 추출
            docstring = ''
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                val = node.body[0].value
                docstring = val.s if isinstance(val, ast.Str) else str(getattr(val, 'value', ''))
                docstring = docstring.strip()[:150]

            # 인자 추출
            args = []
            for arg in node.args.args:
                args.append(arg.arg)

            # 호출하는 다른 함수 추출
            calls = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.add(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.add(child.func.attr)

            functions.append({
                'name': node.name,
                'line_start': node.lineno,
                'line_end': end_line,
                'line_count': end_line - node.lineno + 1,
                'args': args,
                'docstring': docstring,
                'calls': calls,
                'decorators': [ast.dump(d) for d in node.decorator_list] if node.decorator_list else [],
            })

        elif isinstance(node, ast.ClassDef):
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 50
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_end = item.end_lineno if hasattr(item, 'end_lineno') else item.lineno + 10

                    # 메서드가 호출하는 함수들
                    m_calls = set()
                    for child in ast.walk(item):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                m_calls.add(child.func.id)
                            elif isinstance(child.func, ast.Attribute):
                                m_calls.add(child.func.attr)

                    methods.append({
                        'name': item.name,
                        'line_start': item.lineno,
                        'line_end': m_end,
                        'line_count': m_end - item.lineno + 1,
                        'args': [a.arg for a in item.args.args],
                        'calls': m_calls,
                    })

            classes.append({
                'name': node.name,
                'line_start': node.lineno,
                'line_end': end_line,
                'line_count': end_line - node.lineno + 1,
                'methods': methods,
            })

    return functions, classes, len(lines)

# ============================================================
# 파이프라인 스테이지 분류
# ============================================================
STAGE_KEYWORDS = {
    'S1_입력': ['videocapture', 'cap.read', 'imread', 'youtube', 'rtsp', 'upload', 'open.*video', 'load.*image', 'frame_queue', 'yt_dlp', 'pafy'],
    'S2_전처리': ['preprocess', 'pre_process', 'enhance', 'sharpen', 'denoise', 'resize', 'gray', 'cvtcolor', 'threshold', 'blur', 'clahe', 'adaptive', 'equalize', 'morpholog', 'dilate', 'erode', 'bilateral', 'gaussian', 'contrast', 'brightness', 'normalize_image', 'pad', 'letterbox'],
    'S3_YOLO탐지': ['yolo', 'ultralytics', 'model.*predict', 'model.*detect', 'results.*boxes', 'conf.*threshold', 'detect_plate', 'detect_license'],
    'S4_ROI크롭': ['crop', 'roi', 'extract.*plate', 'bbox', 'x1.*y1.*x2.*y2', 'cut.*region', 'slice.*image'],
    'S5_OCR인식': ['paddleocr', 'easyocr', 'pytesseract', 'tesseract', 'ocr.*read', 'ocr.*recognize', 'text.*recognize', 'read.*text'],
    'S6_후처리': ['postprocess', 'post_process', 'ensemble', 'voting', 'vote', 'normalize_plate', 'validate_plate', 'plate_pattern', 'regex.*plate', 'filter_result', 'merge_result', 'confidence.*filter', 'nms', 'suppress'],
    'S7_추적': ['tracker', 'tracking', 'track_id', 'plate_history', 'deque', 'seen_plate', 'plate_buffer', 'smooth', 'stabilize'],
    'S8_DB저장': ['sqlite', 'database', 'db_', 'insert.*into', 'create.*table', 'save.*result', 'write.*csv', 'json.*dump', 'log.*result'],
    'S9_출력GUI': ['flask', 'gradio', 'streamlit', 'tkinter', 'pyqt', 'imshow', 'puttext', 'rectangle', 'draw', 'render', 'display', 'canvas', 'mainloop', 'emit', 'websocket', 'sse', 'response', 'template'],
}

def classify_function_stage(func_name, source_snippet=""):
    """함수가 어느 파이프라인 스테이지에 속하는지 분류"""
    combined = (func_name + ' ' + source_snippet).lower()
    stages = []
    for stage, keywords in STAGE_KEYWORDS.items():
        for kw in keywords:
            if re.search(kw, combined):
                stages.append(stage)
                break
    return list(dict.fromkeys(stages)) if stages else ['S?_미분류']

# ============================================================
# 메인 분석
# ============================================================
def main():
    out = []
    def p(text=""):
        out.append(text)

    p("=" * 80)
    p("  YOLO26 파이프라인 함수 단위 매핑")
    p("  목적: CMD 12분할을 위한 함수별 스테이지 분류")
    p("=" * 80)

    all_py = collect_all_py(PROJECT_ROOT)
    p(f"\n총 {len(all_py)}개 Python 파일 분석\n")

    # 파일별 분석 결과 저장
    file_results = {}

    for rel in sorted(all_py):
        fullpath = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isfile(fullpath):
            continue

        # 소스 읽기
        try:
            with open(fullpath, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
        except:
            continue

        functions, classes, total_lines = extract_functions_and_classes(fullpath)

        p(f"\n{'=' * 70}")
        p(f"  {rel}  ({total_lines}줄)")
        p(f"{'=' * 70}")

        # 클래스 분석
        for cls in classes:
            p(f"\n  [CLASS] {cls['name']}  (L{cls['line_start']}-{cls['line_end']}, {cls['line_count']}줄)")

            for method in cls['methods']:
                # 해당 메서드의 소스 코드 범위 추출
                src_lines = source.split('\n')[method['line_start']-1:method['line_end']]
                src_snippet = ' '.join(src_lines)[:500]

                stages = classify_function_stage(method['name'], src_snippet)
                stage_str = ' + '.join(stages)

                # 핵심 호출 (self.xxx 제외, 외부 함수만)
                key_calls = [c for c in method['calls']
                             if not c.startswith('_') and c not in ('self', 'super', 'print', 'len', 'range', 'int', 'str', 'float', 'list', 'dict', 'set', 'tuple', 'isinstance', 'getattr', 'setattr', 'hasattr', 'format', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'min', 'max', 'abs', 'round', 'type', 'open', 'append', 'extend', 'update', 'get', 'join', 'split', 'strip', 'replace', 'lower', 'upper', 'format', 'items', 'keys', 'values')]

                p(f"    def {method['name']}({', '.join(method['args'][:5])})  [{stage_str}]  L{method['line_start']}-{method['line_end']} ({method['line_count']}줄)")
                if key_calls:
                    p(f"        호출: {', '.join(sorted(key_calls)[:15])}")

        # 독립 함수 분석
        for func in functions:
            src_lines = source.split('\n')[func['line_start']-1:func['line_end']]
            src_snippet = ' '.join(src_lines)[:500]

            stages = classify_function_stage(func['name'], src_snippet)
            stage_str = ' + '.join(stages)

            key_calls = [c for c in func['calls']
                         if not c.startswith('_') and c not in ('self', 'super', 'print', 'len', 'range', 'int', 'str', 'float', 'list', 'dict', 'set', 'tuple', 'isinstance', 'getattr', 'setattr', 'hasattr', 'format', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'min', 'max', 'abs', 'round', 'type', 'open', 'append', 'extend', 'update', 'get', 'join', 'split', 'strip', 'replace', 'lower', 'upper', 'format', 'items', 'keys', 'values')]

            p(f"\n  [FUNC] def {func['name']}({', '.join(func['args'][:5])})  [{stage_str}]  L{func['line_start']}-{func['line_end']} ({func['line_count']}줄)")
            if func['docstring']:
                p(f"    설명: {func['docstring']}")
            if key_calls:
                p(f"    호출: {', '.join(sorted(key_calls)[:15])}")

        file_results[rel] = {'functions': functions, 'classes': classes, 'total_lines': total_lines}

    # ============================================================
    # 스테이지별 함수 집계
    # ============================================================
    p(f"\n\n{'=' * 80}")
    p("  스테이지별 함수 집계 (CMD 분할 기준)")
    p("=" * 80)

    stage_funcs = {}  # stage -> [(file, func_name, line_count)]
    for stage_key in STAGE_KEYWORDS:
        stage_funcs[stage_key] = []

    stage_funcs['S?_미분류'] = []

    for rel, data in file_results.items():
        try:
            with open(os.path.join(PROJECT_ROOT, rel), 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
        except:
            continue

        # 클래스 메서드
        for cls in data['classes']:
            for method in cls['methods']:
                src_lines = source.split('\n')[method['line_start']-1:method['line_end']]
                src_snippet = ' '.join(src_lines)[:500]
                stages = classify_function_stage(method['name'], src_snippet)
                for s in stages:
                    if s in stage_funcs:
                        stage_funcs[s].append((rel, f"{cls['name']}.{method['name']}", method['line_count']))

        # 독립 함수
        for func in data['functions']:
            src_lines = source.split('\n')[func['line_start']-1:func['line_end']]
            src_snippet = ' '.join(src_lines)[:500]
            stages = classify_function_stage(func['name'], src_snippet)
            for s in stages:
                if s in stage_funcs:
                    stage_funcs[s].append((rel, func['name'], func['line_count']))

    for stage, funcs in sorted(stage_funcs.items()):
        if not funcs:
            continue
        total = sum(f[2] for f in funcs)
        p(f"\n  {stage} ({len(funcs)}개 함수, 총 {total}줄)")
        p(f"  {'─' * 60}")
        for file, name, lc in sorted(funcs, key=lambda x: -x[2]):
            p(f"    {name:45s} {lc:4d}줄  ({file})")

    # ============================================================
    # CMD 1 & 2 전처리 분할 제안
    # ============================================================
    p(f"\n\n{'=' * 80}")
    p("  CMD 1 + CMD 2: 전처리 분할 제안")
    p("=" * 80)
    p("""
  위 S2_전처리 함수 목록을 기반으로:

  [CMD 1: 이미지 기본 전처리]
  - 리사이즈/패딩 (letterbox, resize)
  - 색공간 변환 (grayscale, BGR→RGB)
  - 기본 필터링 (blur, denoise)

  [CMD 2: 번호판 특화 전처리]
  - CLAHE (대비 향상)
  - 이진화 (threshold, adaptive)
  - 모폴로지 연산 (dilate, erode)
  - 번호판 영역 보정 (기울기 보정, 원근 변환)

  ※ 위는 일반적 분할안이며, 실제 코드의 S2 함수 목록을 보고
     구체적으로 어떤 함수를 어느 CMD에 배치할지 결정해야 합니다.
""")

    # 저장
    report_text = '\n'.join(out)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n\n  리포트 저장: {REPORT_FILE}")


if __name__ == '__main__':
    main()
