@echo off
chcp 65001 >nul 2>&1
title 실시간 번호판 인식 (localhost)
cd /d C:\tools\yolo26

echo.
echo   실시간 GUI를 localhost에 띄웁니다.
echo   → http://127.0.0.1:5000
echo   → 브라우저가 자동으로 열립니다.
echo   → 종료하려면 이 창에서 Ctrl+C
echo.
python plate_server.py --port 5000
pause
