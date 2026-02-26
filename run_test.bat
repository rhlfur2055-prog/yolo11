@echo off
chcp 65001 >nul 2>&1
title YOLO26 번호판 인식 - 실행 테스트
color 0A

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   YOLO26 번호판 인식 시스템 - 실행 테스트
echo   경로: c:\tools\yolo26
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

cd /d c:\tools\yolo26

echo [1/6] 파일 존재 확인...
echo ─────────────────────────────
set PASS=0
set FAIL=0

if exist plate_gui.py (echo   ✅ plate_gui.py & set /a PASS+=1) else (echo   ❌ plate_gui.py 없음 & set /a FAIL+=1)
if exist plate_recognition_4k.py (echo   ✅ plate_recognition_4k.py & set /a PASS+=1) else (echo   ❌ plate_recognition_4k.py 없음 & set /a FAIL+=1)
if exist plate_engine_pro.py (echo   ✅ plate_engine_pro.py & set /a PASS+=1) else (echo   ❌ plate_engine_pro.py 없음 & set /a FAIL+=1)
if exist plate_ocr_postfilter.py (echo   ✅ plate_ocr_postfilter.py & set /a PASS+=1) else (echo   ❌ plate_ocr_postfilter.py 없음 & set /a FAIL+=1)
if exist plate_server.py (echo   ✅ plate_server.py & set /a PASS+=1) else (echo   ❌ plate_server.py 없음 & set /a FAIL+=1)
if exist youtube_helper.py (echo   ✅ youtube_helper.py & set /a PASS+=1) else (echo   ❌ youtube_helper.py 없음 & set /a FAIL+=1)
if exist RUN.md (echo   ✅ RUN.md & set /a PASS+=1) else (echo   ❌ RUN.md 없음 & set /a FAIL+=1)
echo.
echo   결과: %PASS%개 확인 / %FAIL%개 누락
echo.

echo [2/6] Python 패키지 확인...
echo ─────────────────────────────
python -c "import ultralytics; print(f'  ✅ ultralytics {ultralytics.__version__}')" 2>nul || echo   ❌ ultralytics 미설치 (pip install ultralytics)
python -c "import easyocr; print('  ✅ easyocr')" 2>nul || echo   ❌ easyocr 미설치 (pip install easyocr)
python -c "import cv2; print(f'  ✅ opencv {cv2.__version__}')" 2>nul || echo   ❌ opencv 미설치 (pip install opencv-python)
python -c "import flask; print(f'  ✅ flask {flask.__version__}')" 2>nul || echo   ❌ flask 미설치 (pip install flask)
python -c "import yt_dlp; print(f'  ✅ yt-dlp {yt_dlp.version.__version__}')" 2>nul || echo   ⚠️ yt-dlp 미설치 (--youtube 사용 시 필요: pip install yt-dlp)
python -c "import PIL; print(f'  ✅ pillow {PIL.__version__}')" 2>nul || echo   ❌ pillow 미설치 (pip install pillow)
python -c "import numpy; print(f'  ✅ numpy {numpy.__version__}')" 2>nul || echo   ❌ numpy 미설치 (pip install numpy)
python -c "import paddleocr; print('  ✅ paddleocr')" 2>nul || echo   ⚠️ paddleocr 미설치 (선택: pip install paddlepaddle paddleocr)
echo.

echo [3/6] YOLO 모델 파일 확인...
echo ─────────────────────────────
if exist yolo26.engine (echo   ✅ yolo26.engine [TensorRT FP16 - 1순위]) else (echo   · yolo26.engine 없음)
if exist yolo26.onnx (echo   ✅ yolo26.onnx [ONNX Runtime - 2순위]) else (echo   · yolo26.onnx 없음)
if exist yolo26.pt (echo   ✅ yolo26.pt [PyTorch - 4순위]) else (echo   · yolo26.pt 없음)
if exist yolo11n.pt (echo   ✅ yolo11n.pt [COCO fallback - 5순위]) else (echo   · yolo11n.pt 없음)
if exist yolo26n.pt (echo   ✅ yolo26n.pt [YOLO26 nano]) else (echo   · yolo26n.pt 없음)
if exist yolo26s.pt (echo   ✅ yolo26s.pt [YOLO26 small]) else (echo   · yolo26s.pt 없음)
python -c "from huggingface_hub import hf_hub_download; print('  ✅ huggingface_hub (3순위 HF 다운로드 가능)')" 2>nul || echo   · huggingface_hub 없음 (자동 다운로드 불가)
echo.

echo [4/6] GPU 확인...
echo ─────────────────────────────
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"없음 (CPU 모드)\"}'); print(f'  PyTorch: {torch.__version__}')" 2>nul || echo   ⚠️ PyTorch 미설치 또는 CUDA 불가
echo.

echo [5/6] 테스트 영상 확인...
echo ─────────────────────────────
set VIDEO_FOUND=0
for %%f in (*.mp4 *.avi *.mov *.mkv) do (
    echo   📹 %%f
    set VIDEO_FOUND=1
)
if exist temp_youtube\*.mp4 (
    echo   📹 temp_youtube\ 폴더:
    for %%f in (temp_youtube\*.mp4) do echo      %%f
    set VIDEO_FOUND=1
)
if %VIDEO_FOUND%==0 (
    echo   ⚠️ mp4/avi/mov 파일 없음
    echo   → 동영상을 c:\tools\yolo26\ 에 넣거나
    echo   → --youtube 옵션으로 다운로드하세요
)
echo.

echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   환경 확인 완료. 실행할 모드를 선택하세요:
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo   ★ 실시간 GUI를 localhost에 띄우려면 [4] 선택
echo.
echo   [4] 실시간 GUI (localhost) — 웹 서버 실행 후 브라우저에 실시간 번호판 인식
echo   [1] plate_gui.py        — GUI 실시간 인식 (파일 선택)
echo   [2] plate_gui.py 영상   — GUI 실시간 인식 (첫 번째 mp4 자동)
echo   [3] --youtube           — YouTube URL 다운로드 후 GUI 재생
echo   [5] plate_engine_pro.py — Pro 엔진 CLI (웹캠)
echo   [6] 배치 처리            — plate_recognition_4k.py (첫 번째 mp4)
echo   [7] 전체 패키지 설치     — pip install 모든 의존성
echo   [8] YouTube 단독 테스트  — youtube_helper.py URL 입력
echo   [0] 종료
echo.

:MENU
set /p CHOICE=  선택 (0-8): 

if "%CHOICE%"=="4" goto RUN_SERVER
if "%CHOICE%"=="1" goto RUN_GUI
if "%CHOICE%"=="2" goto RUN_GUI_AUTO
if "%CHOICE%"=="3" goto RUN_YOUTUBE
if "%CHOICE%"=="5" goto RUN_PRO
if "%CHOICE%"=="6" goto RUN_BATCH
if "%CHOICE%"=="7" goto INSTALL_ALL
if "%CHOICE%"=="8" goto RUN_YT_HELPER
if "%CHOICE%"=="0" goto END

echo   잘못된 선택. 다시 입력하세요.
goto MENU

:RUN_GUI
echo.
echo   ▶ python plate_gui.py
echo   (파일 다이얼로그에서 동영상 선택)
echo.
python plate_gui.py
goto DONE

:RUN_GUI_AUTO
echo.
set FIRST_MP4=
for %%f in (*.mp4) do (
    if not defined FIRST_MP4 set FIRST_MP4=%%f
)
if not defined FIRST_MP4 (
    echo   ❌ mp4 파일이 없습니다. c:\tools\yolo26\ 에 넣어주세요.
    goto DONE
)
echo   ▶ python plate_gui.py "%FIRST_MP4%"
echo.
python plate_gui.py "%FIRST_MP4%"
goto DONE

:RUN_YOUTUBE
echo.
set /p YT_URL=  YouTube URL 입력: 
if "%YT_URL%"=="" (
    echo   URL이 비어있습니다.
    goto DONE
)
echo.
echo   ▶ python plate_gui.py --youtube "%YT_URL%"
echo.
python plate_gui.py --youtube "%YT_URL%"
goto DONE

:RUN_SERVER
echo.
echo   ▶ 실시간 GUI를 localhost에 띄우는 중...
echo   → http://127.0.0.1:5000 (실시간 번호판 인식 화면)
echo   → 동영상 경로/YouTube URL 입력 후 [동영상 번호판 인식 시작]
echo   → 서버가 켜지면 브라우저가 자동으로 열립니다. 종료: Ctrl+C
echo.
python plate_server.py
goto DONE

:RUN_PRO
echo.
echo   ▶ python plate_engine_pro.py --input 0
echo   (웹캠 실시간 - 'q'키로 종료)
echo.
python plate_engine_pro.py --input 0
goto DONE

:RUN_BATCH
echo.
set FIRST_MP4=
for %%f in (*.mp4) do (
    if not defined FIRST_MP4 set FIRST_MP4=%%f
)
if not defined FIRST_MP4 (
    echo   ❌ mp4 파일이 없습니다.
    goto DONE
)
echo   ▶ python plate_recognition_4k.py "%FIRST_MP4%" -o ./plate_results
echo.
python plate_recognition_4k.py "%FIRST_MP4%" -o ./plate_results
goto DONE

:INSTALL_ALL
echo.
echo   ▶ 전체 패키지 설치 중...
echo.
pip install ultralytics easyocr opencv-python flask pillow numpy huggingface_hub yt-dlp
echo.
echo   ⚠️ PaddleOCR (선택):
echo     pip install paddlepaddle paddleocr
echo.
echo   ⚠️ GPU 가속 (선택):
echo     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.
goto DONE

:RUN_YT_HELPER
echo.
set /p YT_URL2=  YouTube URL 입력: 
if "%YT_URL2%"=="" (
    echo   URL이 비어있습니다.
    goto DONE
)
echo.
echo   ▶ python youtube_helper.py "%YT_URL2%"
echo.
python youtube_helper.py "%YT_URL2%"
goto DONE

:DONE
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pause
goto MENU

:END
echo.
echo   종료합니다.
