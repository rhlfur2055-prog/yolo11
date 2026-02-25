"""portfolio.html 생성 스크립트"""
import json, os

with open('_img_data.json', 'r') as f:
    img = json.load(f)

html = '<!DOCTYPE html>\n<html lang="ko">\n<head>\n<meta charset="UTF-8">\n<title>YOLO26 번호판 인식 시스템 포트폴리오</title>\n'
html += '''<style>
@page{size:A4;margin:12mm 15mm}*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;background:#fff;color:#222;line-height:1.65;padding:30px 50px;max-width:960px;margin:0 auto}
.slide{page-break-after:always;min-height:90vh;padding-bottom:20px}.slide:last-child{page-break-after:avoid}
.cover{text-align:center;padding:80px 0 40px;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:90vh}
.cover h1{font-size:2.5em;color:#1a237e;margin-bottom:8px}.cover .sub{font-size:1.3em;color:#546e7a;margin-bottom:35px}
.cover .mimg{width:85%;border-radius:12px;box-shadow:0 6px 25px rgba(0,0,0,.18);margin-bottom:30px}
.cover .info{font-size:.95em;color:#555;line-height:2.2}.cover .info strong{color:#1a237e}
.bg{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:20px}
.bg span{background:#e8eaf6;color:#283593;padding:5px 15px;border-radius:20px;font-size:.85em;font-weight:600}
h2{font-size:1.55em;color:#1a237e;border-bottom:3px solid #3949ab;padding-bottom:6px;margin:30px 0 16px}
h3{font-size:1.1em;color:#283593;margin:18px 0 10px}
.code{background:#1e1e2e;color:#cdd6f4;border-radius:8px;padding:14px 18px;margin:10px 0 16px;overflow-x:auto;font-family:Consolas,'D2Coding',monospace;font-size:.78em;line-height:1.5;white-space:pre;position:relative}
.code .fn{position:absolute;top:0;right:0;background:#45475a;color:#a6adc8;padding:2px 10px;border-radius:0 8px 0 6px;font-size:.85em}
.cm{color:#6c7086}.kw{color:#cba6f7}.fc{color:#89b4fa}.st{color:#a6e3a1}.nm{color:#fab387}.dc{color:#f9e2af}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}
.card{border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;background:#fafafa}
.card img{width:100%;height:auto;display:block}
.cap{padding:8px 12px;font-size:.85em;text-align:center}
.cap .pl{font-weight:700;color:#1a237e;font-size:1.1em}.cap .cf{color:#e65100;font-weight:600}
.ty{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.78em;font-weight:600}
.ty-y{background:#fff9c4;color:#f57f17;border:1px solid #fdd835}
.ty-w{background:#e3f2fd;color:#1565c0;border:1px solid #64b5f6}
.ty-g{background:#e8f5e9;color:#2e7d32;border:1px solid #66bb6a}
.feat{background:linear-gradient(135deg,#e8eaf6,#f3e5f5);border-radius:8px;padding:14px 18px;margin:10px 0;border-left:4px solid #3949ab}
.feat h4{color:#283593;margin-bottom:4px;font-size:1em}.feat p{font-size:.88em;color:#444}
.arch{background:#f5f5f5;border:1px solid #ddd;border-radius:8px;padding:16px;margin:14px 0}
.arch pre{font-family:Consolas,'D2Coding',monospace;font-size:.8em;line-height:1.45;color:#333}
.pipe{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px dashed #e0e0e0}
.pipe .num{width:30px;height:30px;background:#3949ab;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.85em;flex-shrink:0}
.pipe .txt{font-size:.88em}.pipe .txt strong{color:#1a237e}
.compare{display:flex;gap:12px;align-items:center;justify-content:center;margin:14px 0}
.compare img{max-height:75px;border-radius:4px;border:2px solid #ddd}
.arrow{font-size:1.5em;color:#3949ab}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:.9em}
th{background:#e8eaf6;padding:9px;text-align:left;border:1px solid #ccc}
td{padding:8px;border:1px solid #ddd}
.future{background:#e3f2fd;border:2px solid #42a5f5;border-radius:10px;padding:20px;margin:16px 0}
.future h3{color:#1565c0;margin:0 0 10px}.future ul{padding-left:20px}
.future li{margin:6px 0;font-size:.92em}
.future .hl{font-size:1.3em;color:#1565c0;font-weight:700;text-align:center;margin:15px 0}
@media print{body{padding:0}.slide{page-break-after:always}.code{break-inside:avoid}.grid{break-inside:avoid}}
</style>\n</head>\n<body>\n'''

# Slide 1: Cover
html += f'''<div class="slide cover">
<h1>교통안전을 위한 AI 솔루션</h1>
<div class="sub">YOLO26 기반 실시간 차량 번호판 인식 시스템</div>
<img class="mimg" src="{img['det_frame1']}">
<div class="info"><strong>능력단위:</strong> 교통안전을 위한 AI 솔루션<br><strong>평가방법:</strong> 포트폴리오<br><strong>평가일시:</strong> 2026-02-25<br><strong>훈련기간:</strong> 2025-10-31 ~ 2026-05-04</div>
<div class="bg"><span>YOLO26</span><span>OpenCV</span><span>PaddleOCR</span><span>EasyOCR</span><span>Python</span><span>Flask</span><span>Roboflow</span></div>
</div>\n'''

# Slide 2: Overview + Architecture
html += '''<div class="slide">
<h2>1. 프로젝트 개요</h2>
<p>고속도로 톨게이트 CCTV 영상에서 <strong>실시간으로 차량 번호판을 검출·인식</strong>하는 AI 시스템입니다.<br>구형 노란 번호판(버스/트럭), 신형 흰색 번호판, 녹색 번호판을 모두 인식할 수 있습니다.</p>
<h2>2. 시스템 아키텍처</h2>
<div class="arch"><pre>
┌─────────────────────────────────────────────────┐
│               사용자 인터페이스                      │
│  plate_gui.py (GUI)  │  plate_server.py (웹서버)   │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────┐
│            YOLO26 Detection Engine               │
│  번호판 전용 모델 (yolo11x_plate.pt, mAP@50=98.4%) │
│  상태머신: SCANNING → TRACKING → CAPTURING        │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────┐
│           이미지 전처리 파이프라인                    │
│  ROI → CROP → CLAHE → Homography → 18종 전처리    │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────┐
│           앙상블 OCR + 후처리                      │
│  PaddleOCR(한글) + EasyOCR(영문) → 투표 → 검증     │
└─────────────────────────────────────────────────┘
</pre></div>
</div>\n'''

# Slide 3: Yellow plates
html += f'''<div class="slide">
<h2>3. 실제 번호판 인식 결과</h2>
<h3>3-1. 구형 노란 번호판 (버스/트럭) 인식</h3>
<p>영업용 차량의 구형 노란색 번호판을 정확하게 인식합니다.</p>
<div class="grid">
<div class="card"><img src="{img['truck_6286']}"><div class="cap"><span class="pl">경기91바6286</span> <span class="cf">79%</span> <span class="ty ty-y">영업용</span><br>현대 트럭 · 구형 노란판</div></div>
<div class="card"><img src="{img['bus_9203']}"><div class="cap"><span class="pl">서울바9203</span> <span class="cf">72%</span> <span class="ty ty-y">영업용</span><br>마이버스 · 구형 노란판</div></div>
<div class="card"><img src="{img['bus_7789']}"><div class="cap"><span class="pl">경기76바7789</span> <span class="cf">85%</span> <span class="ty ty-y">영업용</span><br>한양대학교 버스 · 구형 노란판</div></div>
<div class="card"><img src="{img['kia_8060']}"><div class="cap"><span class="pl">01나8060</span> <span class="cf">71%</span> <span class="ty ty-g">녹색판</span><br>기아 스포티지 · 구형 녹색판</div></div>
</div>
</div>\n'''

# Slide 4: White plates
html += f'''<div class="slide">
<h3>3-2. 신형 흰색 번호판 인식</h3>
<p>일반 자가용의 신형 흰색 번호판을 다양한 차종에서 인식합니다.</p>
<div class="grid">
<div class="card"><img src="{img['kia_9392']}"><div class="cap"><span class="pl">55저9392</span> <span class="cf">88%</span> <span class="ty ty-w">자가용</span><br>기아 쏘렌토</div></div>
<div class="card"><img src="{img['kia_9599']}"><div class="cap"><span class="pl">58두9599</span> <span class="cf">82%</span> <span class="ty ty-w">자가용</span><br>기아 K5</div></div>
<div class="card"><img src="{img['hyundai_6393']}"><div class="cap"><span class="pl">70버6393</span> <span class="cf">85%</span> <span class="ty ty-w">자가용</span><br>현대 스타렉스</div></div>
<div class="card"><img src="{img['kia_5915']}"><div class="cap"><span class="pl">80부5915</span> <span class="cf">83%</span> <span class="ty ty-w">자가용</span><br>기아 봉고</div></div>
<div class="card"><img src="{img['sonata_7062']}"><div class="cap"><span class="pl">48보7062</span> <span class="cf">78%</span> <span class="ty ty-w">자가용</span><br>현대 쏘나타</div></div>
<div class="card"><img src="{img['kia_3234']}"><div class="cap"><span class="pl">14니3234</span> <span class="cf">86%</span> <span class="ty ty-w">자가용</span><br>기아 옵티마</div></div>
<div class="card"><img src="{img['suv_7117']}"><div class="cap"><span class="pl">36다7117</span> <span class="cf">75%</span> <span class="ty ty-g">녹색판</span><br>쌍용 렉스턴</div></div>
<div class="card"><img src="{img['van_2754']}"><div class="cap"><span class="pl">02누2754</span> <span class="cf">77%</span> <span class="ty ty-w">자가용</span><br>기아 카니발</div></div>
</div>
</div>\n'''

# Slide 5: Realtime detection screens + crop compare
html += f'''<div class="slide">
<h2>4. 실시간 인식 화면</h2>
<p>YOLO26이 번호판을 검출하고 OCR로 텍스트를 인식한 후 화면에 오버레이합니다.</p>
<div class="grid">
<div class="card"><img src="{img['det_frame1']}"><div class="cap">톨게이트 실시간 검출 #1 - 바운딩 박스 + OCR</div></div>
<div class="card"><img src="{img['det_frame3']}"><div class="cap">톨게이트 실시간 검출 #2 - 진입 차량 인식</div></div>
</div>
<h3>4-1. 번호판 크롭 &rarr; 전처리 &rarr; OCR 과정</h3>
<div class="compare">
<div style="text-align:center"><img src="{img['plate_crop']}" style="max-height:65px"><br><small>원본 크롭</small></div>
<div class="arrow">&rarr;</div>
<div style="text-align:center"><img src="{img['plate_preproc']}" style="max-height:65px"><br><small>CLAHE 전처리</small></div>
<div class="arrow">&rarr;</div>
<div style="text-align:center;font-weight:700;color:#1a237e;font-size:1.3em">35주5546</div>
</div>
<div class="compare">
<div style="text-align:center"><img src="{img['crop_9203']}" style="max-height:65px"><br><small>노란판 크롭</small></div>
<div class="arrow">&rarr;</div>
<div style="text-align:center;font-weight:700;color:#1a237e;font-size:1.3em">서울바9203</div>
</div>
</div>\n'''

# Slide 6: Code review - ROI + CROP
html += '''<div class="slide">
<h2>5. 핵심 코드 리뷰</h2>
<h3>5-1. Roboflow &rarr; YOLO 학습 파이프라인</h3>
<div class="feat"><h4>Object Detection 모델 학습</h4><p>Roboflow에서 번호판 이미지를 라벨링하여 데이터셋을 생성하고, YOLO 모델을 학습시켜 best.pt를 생성했습니다.</p></div>
<div class="code"><span class="fn">학습 파이프라인</span>
<span class="cm"># 1. Roboflow 데이터셋 다운로드</span>
<span class="kw">from</span> roboflow <span class="kw">import</span> Roboflow
rf = Roboflow(api_key=<span class="st">"..."</span>)
project = rf.workspace().project(<span class="st">"license-plate-detection"</span>)
dataset = project.version(<span class="nm">1</span>).download(<span class="st">"yolov11"</span>)

<span class="cm"># 2. YOLO 모델 학습 → best.pt 생성</span>
<span class="kw">from</span> ultralytics <span class="kw">import</span> YOLO
model = YOLO(<span class="st">"yolo11x.pt"</span>)
model.train(data=<span class="st">"dataset/data.yaml"</span>, epochs=<span class="nm">100</span>, imgsz=<span class="nm">640</span>)
<span class="cm"># → best.pt (mAP@50 = 98.4%)</span></div>

<h3>5-2. ROI (Region of Interest) - 관심 영역 설정</h3>
<div class="feat"><h4>불필요한 영역 제거로 처리 속도 향상</h4><p>번호판이 나타날 수 있는 영역만 추출합니다.</p></div>
<div class="code"><span class="fn">exam_realtime_plate.py</span>
<span class="kw">def</span> <span class="fc">apply_roi</span>(frame, bbox, margin_x_ratio=<span class="nm">0.25</span>, margin_y_ratio=<span class="nm">0.30</span>):
    <span class="st">"""ROI(Region of Interest) 추출"""</span>
    h, w = frame.shape[:<span class="nm">2</span>]
    x1, y1, x2, y2 = bbox
    margin_x = <span class="fc">int</span>((x2 - x1) * margin_x_ratio)
    margin_y = <span class="fc">int</span>((y2 - y1) * margin_y_ratio)
    roi_x1 = <span class="fc">max</span>(<span class="nm">0</span>, x1 - margin_x)
    roi_y1 = <span class="fc">max</span>(<span class="nm">0</span>, y1 - margin_y)
    roi_x2 = <span class="fc">min</span>(w, x2 + margin_x)
    roi_y2 = <span class="fc">min</span>(h, y2 + margin_y)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    <span class="kw">return</span> roi, (roi_x1, roi_y1, roi_x2, roi_y2)</div>

<h3>5-3. CROP - 번호판 크롭 + 자동 업스케일</h3>
<div class="feat"><h4>작은 번호판 자동 확대</h4><p>너비 300px 미만이면 자동 업스케일 + 샤프닝으로 OCR 정확도를 높입니다.</p></div>
<div class="code"><span class="fn">exam_realtime_plate.py</span>
<span class="kw">def</span> <span class="fc">crop_plate</span>(frame, bbox):
    <span class="st">"""번호판 영역 크롭 + 업스케일"""</span>
    x1, y1, x2, y2 = bbox
    cropped = frame[y1:y2, x1:x2]
    crop_w = cropped.shape[<span class="nm">1</span>]
    <span class="kw">if</span> crop_w < <span class="nm">300</span>:
        scale = <span class="nm">300.0</span> / crop_w
        cropped = cv2.<span class="fc">resize</span>(cropped, <span class="kw">None</span>, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[<span class="nm">0</span>,<span class="nm">-1</span>,<span class="nm">0</span>],[<span class="nm">-1</span>,<span class="nm">5</span>,<span class="nm">-1</span>],[<span class="nm">0</span>,<span class="nm">-1</span>,<span class="nm">0</span>]])
        cropped = cv2.<span class="fc">filter2D</span>(cropped, <span class="nm">-1</span>, kernel)
    <span class="kw">return</span> cropped</div>
</div>\n'''

# Slide 7: CLAHE + Homography
html += '''<div class="slide">
<h3>5-4. CLAHE - 적응형 대비 향상</h3>
<div class="feat"><h4>Contrast Limited Adaptive Histogram Equalization</h4><p>LAB 색공간의 L 채널에 CLAHE를 적용하여 어두운/밝은 환경에서도 번호판 가독성을 개선합니다.</p></div>
<div class="code"><span class="fn">exam_realtime_plate.py</span>
<span class="kw">def</span> <span class="fc">apply_clahe</span>(image, clip_limit=<span class="nm">4.0</span>, tile_size=(<span class="nm">8</span>, <span class="nm">8</span>)):
    <span class="st">"""CLAHE - 적응형 히스토그램 평활화로 대비 향상"""</span>
    lab = cv2.<span class="fc">cvtColor</span>(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.<span class="fc">split</span>(lab)
    clahe = cv2.<span class="fc">createCLAHE</span>(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.<span class="fc">apply</span>(l_channel)
    enhanced_lab = cv2.<span class="fc">merge</span>([l_enhanced, a_channel, b_channel])
    <span class="kw">return</span> cv2.<span class="fc">cvtColor</span>(enhanced_lab, cv2.COLOR_LAB2BGR)</div>

<h3>5-5. Homography - 기울어진 번호판 원근 보정</h3>
<div class="feat"><h4>에지 검출 &rarr; 윤곽선 &rarr; 4점 원근 변환</h4><p>Canny 에지 검출로 번호판 윤곽을 찾고, 4개 꼭짓점으로 원근 변환하여 정면 보정합니다.</p></div>
<div class="code"><span class="fn">exam_realtime_plate.py</span>
<span class="kw">def</span> <span class="fc">apply_homography</span>(image):
    <span class="st">"""Homography 변환 (원근 보정)"""</span>
    gray = cv2.<span class="fc">cvtColor</span>(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.<span class="fc">Canny</span>(cv2.<span class="fc">GaussianBlur</span>(gray,(<span class="nm">5</span>,<span class="nm">5</span>),<span class="nm">0</span>), <span class="nm">50</span>, <span class="nm">150</span>)
    contours, _ = cv2.<span class="fc">findContours</span>(edges, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    largest = <span class="fc">max</span>(contours, key=cv2.contourArea)
    approx = cv2.<span class="fc">approxPolyDP</span>(largest,
                  <span class="nm">0.02</span> * cv2.arcLength(largest, <span class="kw">True</span>), <span class="kw">True</span>)
    <span class="kw">if</span> <span class="fc">len</span>(approx) == <span class="nm">4</span>:
        pts = approx.reshape(<span class="nm">4</span>, <span class="nm">2</span>).astype(np.float32)
        h, w = image.shape[:<span class="nm">2</span>]
        dst = np.array([[<span class="nm">0</span>,<span class="nm">0</span>],[w,<span class="nm">0</span>],[w,h],[<span class="nm">0</span>,h]], dtype=np.float32)
        M = cv2.<span class="fc">getPerspectiveTransform</span>(pts, dst)
        <span class="kw">return</span> cv2.<span class="fc">warpPerspective</span>(image, M, (w, h))</div>
</div>\n'''

# Slide 8: 18 preprocessors + OCR ensemble
html += '''<div class="slide">
<h3>5-6. 18종 이미지 전처리 파이프라인</h3>
<div class="feat"><h4>다양한 환경에 대응하는 앙상블 전처리</h4><p>18가지 서로 다른 전처리를 적용하고 최적의 OCR 결과를 선택합니다.</p></div>
<div class="code"><span class="fn">plate_engine_pro.py - ImagePreprocessor</span>
<span class="kw">class</span> <span class="fc">ImagePreprocessor</span>:
    <span class="st">"""18종 이미지 전처리 파이프라인"""</span>
    <span class="kw">def</span> <span class="fc">clahe</span>(img):          <span class="cm"># CLAHE 대비 향상</span>
    <span class="kw">def</span> <span class="fc">deskew</span>(img):         <span class="cm"># Hough 기울기 보정</span>
    <span class="kw">def</span> <span class="fc">sharpen</span>(img):        <span class="cm"># 샤프닝</span>
    <span class="kw">def</span> <span class="fc">median_blur</span>(img):    <span class="cm"># 점잡음 제거</span>
    <span class="kw">def</span> <span class="fc">otsu_inv</span>(img):       <span class="cm"># Otsu 반전 이진화</span>
    <span class="kw">def</span> <span class="fc">upscale_2x</span>(img):     <span class="cm"># 2배 업스케일</span>
    <span class="kw">def</span> <span class="fc">gamma_bright</span>(img):   <span class="cm"># 밝기 감마 보정</span>
    <span class="kw">def</span> <span class="fc">gamma_dark</span>(img):     <span class="cm"># 어두운 감마 보정</span>
    <span class="kw">def</span> <span class="fc">bilateral</span>(img):      <span class="cm"># 양방향 필터</span>
    <span class="kw">def</span> <span class="fc">morphology</span>(img):     <span class="cm"># 모폴로지 연산</span>
    <span class="kw">def</span> <span class="fc">deskew_otsu</span>(img):    <span class="cm"># 기울기 보정 + Otsu</span>
    <span class="kw">def</span> <span class="fc">denoise</span>(img):        <span class="cm"># 노이즈 제거</span>
    <span class="kw">def</span> <span class="fc">deblur</span>(img):         <span class="cm"># 블러 제거</span>
    <span class="kw">def</span> <span class="fc">hist_equalize</span>(img):  <span class="cm"># 히스토그램 평활화</span>
    <span class="kw">def</span> <span class="fc">adaptive_mean</span>(img):  <span class="cm"># 적응 평균 이진화</span>
    <span class="kw">def</span> <span class="fc">adaptive_threshold</span>: <span class="cm"># 적응 가우시안 이진화</span>
    <span class="kw">def</span> <span class="fc">gray_threshold</span>:     <span class="cm"># Otsu 이진화</span>
    <span class="kw">def</span> <span class="fc">brightness_boost</span>:   <span class="cm"># 밝기 증가</span></div>

<h3>5-7. 앙상블 OCR (PaddleOCR + EasyOCR)</h3>
<div class="feat"><h4>이중 OCR 엔진 투표 방식</h4><p>PaddleOCR(한글)과 EasyOCR(영문) 결과를 투표하여 최종 텍스트를 결정합니다.</p></div>
<div class="code"><span class="fn">hiway_exam_final.py - PlateOCR</span>
<span class="kw">class</span> <span class="fc">PlateOCR</span>:
    <span class="st">"""PaddleOCR + EasyOCR 앙상블
    3가지 전처리 x 2개 엔진 = 최대 6회 시도"""</span>
    <span class="kw">def</span> <span class="fc">recognize</span>(crop):
        versions, is_yellow = PlatePreprocessor.<span class="fc">get_versions</span>(crop)
        candidates = []
        <span class="kw">for</span> v <span class="kw">in</span> versions:
            candidates.append(paddle_ocr.<span class="fc">ocr</span>(v))
            candidates.append(easy_reader.<span class="fc">readtext</span>(v))
        <span class="kw">return</span> Counter(candidates).<span class="fc">most_common</span>(<span class="nm">1</span>)[<span class="nm">0</span>]</div>
</div>\n'''

# Slide 9: Yellow/white detection + web server
html += '''<div class="slide">
<h3>5-8. 노란/흰색 번호판 자동 구분</h3>
<div class="feat"><h4>HSV 색공간 분석으로 자동 판별</h4><p>노란색 비율 20% 이상이면 구형 노란 번호판으로 판단, 맞춤 전처리를 적용합니다.</p></div>
<div class="code"><span class="fn">hiway_exam_final.py</span>
<span class="kw">def</span> <span class="fc">is_yellow_plate</span>(crop):
    <span class="st">"""노란 번호판 판단 (HSV 노란색 비율)"""</span>
    hsv = cv2.<span class="fc">cvtColor</span>(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.<span class="fc">inRange</span>(hsv, (<span class="nm">15</span>,<span class="nm">80</span>,<span class="nm">80</span>), (<span class="nm">40</span>,<span class="nm">255</span>,<span class="nm">255</span>))
    ratio = np.<span class="fc">sum</span>(mask > <span class="nm">0</span>) / (crop.shape[<span class="nm">0</span>] * crop.shape[<span class="nm">1</span>])
    <span class="kw">return</span> ratio > <span class="nm">0.20</span>

<span class="kw">def</span> <span class="fc">preprocess_yellow</span>(crop):
    <span class="st">"""노란판: 반전 + 채도 제거 + CLAHE"""</span>
    v1 = cv2.<span class="fc">bitwise_not</span>(gray)
    v2 = hsv[:, :, <span class="nm">1</span>]
    v3 = clahe.<span class="fc">apply</span>(cv2.bitwise_not(gray))</div>

<h3>5-9. Flask 실시간 웹 서버</h3>
<div class="feat"><h4>원격 모니터링 웹 인터페이스</h4><p>비디오 업로드, MJPEG 실시간 스트리밍, REST API를 제공합니다.</p></div>
<div class="code"><span class="fn">plate_server.py</span>
<span class="dc">@app.route</span>(<span class="st">"/"</span>)           <span class="cm"># 메인 업로드 UI</span>
<span class="dc">@app.route</span>(<span class="st">"/realtime"</span>)   <span class="cm"># 실시간 스트리밍</span>
<span class="dc">@app.route</span>(<span class="st">"/api/upload"</span>)  <span class="cm"># 비디오 업로드 (POST)</span>
<span class="dc">@app.route</span>(<span class="st">"/stream"</span>)      <span class="cm"># MJPEG 피드</span>

<span class="kw">def</span> <span class="fc">_mjpeg_stream</span>(source, roi):
    recognizer = PlateRecognizer()
    cap = cv2.<span class="fc">VideoCapture</span>(source)
    <span class="kw">while</span> cap.isOpened():
        ret, frame = cap.read()
        results = recognizer.<span class="fc">process_frame</span>(frame)
        <span class="kw">for</span> r <span class="kw">in</span> results:
            cv2.<span class="fc">rectangle</span>(frame, ...)
            cv2.<span class="fc">putText</span>(frame, r[<span class="st">'plate'</span>])
        <span class="kw">yield</span> frame</div>
</div>\n'''

# Slide 10: Pipeline + file structure
html += '''<div class="slide">
<h2>6. 전체 처리 파이프라인</h2>
<div class="pipe"><div class="num">1</div><div class="txt"><strong>영상 입력</strong> &mdash; MP4 / RTSP / 웹캠 / YouTube URL</div></div>
<div class="pipe"><div class="num">2</div><div class="txt"><strong>ROI 필터링</strong> &mdash; 관심 영역만 추출, 불필요 영역 제거</div></div>
<div class="pipe"><div class="num">3</div><div class="txt"><strong>YOLO26 Detection</strong> &mdash; 번호판 바운딩 박스 검출 (NMS-free)</div></div>
<div class="pipe"><div class="num">4</div><div class="txt"><strong>CROP + 업스케일</strong> &mdash; 번호판 크롭, 소형 번호판 6배 확대</div></div>
<div class="pipe"><div class="num">5</div><div class="txt"><strong>CLAHE 대비 향상</strong> &mdash; LAB 색공간 적응형 히스토그램 평활화</div></div>
<div class="pipe"><div class="num">6</div><div class="txt"><strong>Homography 보정</strong> &mdash; 기울어진 번호판 원근 변환 보정</div></div>
<div class="pipe"><div class="num">7</div><div class="txt"><strong>18종 전처리</strong> &mdash; 이진화, 샤프닝, 감마 보정, 노이즈 제거 등</div></div>
<div class="pipe"><div class="num">8</div><div class="txt"><strong>앙상블 OCR</strong> &mdash; PaddleOCR + EasyOCR 투표 방식</div></div>
<div class="pipe"><div class="num">9</div><div class="txt"><strong>후처리 검증</strong> &mdash; 한글 혼동 교정 (O&rarr;0, L&rarr;나), 패턴 검증</div></div>
<div class="pipe"><div class="num">10</div><div class="txt"><strong>결과 출력</strong> &mdash; 화면 오버레이 + JSON + DB 캐싱</div></div>

<h2>7. 프로젝트 파일 구조</h2>
<div class="arch"><pre>yolo26-main/
├── <strong>plate_recognition_4k.py</strong>  (2,925줄) ─ 핵심 엔진
├── <strong>plate_engine_pro.py</strong>      (1,161줄) ─ Pro 엔진
├── <strong>plate_server.py</strong>          (1,644줄) ─ Flask 웹서버
├── <strong>plate_gui.py</strong>               (926줄) ─ GUI 뷰어
├── <strong>exam_realtime_plate.py</strong>           ─ ROI/CROP/CLAHE/Homography
├── <strong>hiway_exam_final.py</strong>              ─ 역할별 클래스 아키텍처
├── models/
│   ├── yolo11x_plate.pt (110MB) ─ 번호판 전용 (mAP@50=98.4%)
│   └── yolo26n.pt (5MB)         ─ YOLO26 NMS-free
└── dataset/                      ─ Roboflow 라벨링 데이터셋</pre></div>
</div>\n'''

# Slide 11: Tech stack + Future plan
html += '''<div class="slide">
<h2>8. 사용 기술 스택</h2>
<table>
<tr><th>분류</th><th>기술</th><th>용도</th></tr>
<tr><td><strong>Detection</strong></td><td>YOLO26, YOLOv11x</td><td>번호판 검출 (NMS-free)</td></tr>
<tr><td><strong>OCR</strong></td><td>PaddleOCR, EasyOCR</td><td>한글/영문 텍스트 인식 앙상블</td></tr>
<tr><td><strong>영상처리</strong></td><td>OpenCV</td><td>ROI, CROP, CLAHE, Homography</td></tr>
<tr><td><strong>딥러닝</strong></td><td>PyTorch, Ultralytics</td><td>모델 추론 프레임워크</td></tr>
<tr><td><strong>데이터셋</strong></td><td>Roboflow</td><td>라벨링 + 데이터셋 생성 + 증강</td></tr>
<tr><td><strong>웹서버</strong></td><td>Flask, FastAPI</td><td>실시간 스트리밍, REST API</td></tr>
<tr><td><strong>GUI</strong></td><td>Tkinter</td><td>데스크탑 실시간 뷰어</td></tr>
<tr><td><strong>DB</strong></td><td>SQLite</td><td>중복 인식 방지 캐시</td></tr>
</table>

<h2>9. 향후 개선 계획</h2>
<div class="future">
<div class="hl">목표: 인식 정확도 99% 달성</div>
<h3>데이터 확충 계획</h3>
<ul>
<li><strong>1,000장 이상의 번호판 이미지를 추가 수집·라벨링</strong>하여 학습 데이터를 대폭 확충</li>
<li>야간, 역광, 우천, 저해상도 등 <strong>다양한 악조건 데이터</strong>를 집중 수집</li>
<li>구형 노란 번호판, 전기차 번호판 등 <strong>특수 번호판 비율을 높여</strong> 학습</li>
</ul>
<h3>모델 개선 계획</h3>
<ul>
<li>데이터 증강(Augmentation) 강화: 회전, 블러, 밝기 변화, 노이즈 추가</li>
<li>YOLO26 최신 아키텍처로 검출 속도 + 정확도 동시 향상</li>
<li>OCR 후처리 고도화: 한글 혼동 매트릭스 확장, 지역명 사전 추가</li>
<li>TensorRT/ONNX 최적화로 실시간 15FPS 이상 달성</li>
</ul>
</div>
<div style="text-align:center;padding:30px 0;color:#999;font-size:.9em">&mdash; End of Portfolio &mdash;</div>
</div>\n'''

html += '</body>\n</html>'

with open('portfolio.html', 'w', encoding='utf-8') as f:
    f.write(html)

size = os.path.getsize('portfolio.html')
print(f"OK: portfolio.html ({size:,} bytes, {size/1024/1024:.1f} MB)")
