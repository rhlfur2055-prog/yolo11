@echo off
chcp 65001 >nul 2>&1
title 번호판 인식 - 실시간 GUI 서버
cd /d C:\tools\yolo26

echo.
echo 기존에 5000 포트에서 서버가 켜져 있으면 먼저 그 창에서 Ctrl+C 로 종료하세요.
echo.
echo 실시간 GUI 주소: http://127.0.0.1:5000/live
echo (또는 http://127.0.0.1:5000 입력 시 자동으로 /live 로 이동)
echo.
python plate_server.py --port 5000
pause
