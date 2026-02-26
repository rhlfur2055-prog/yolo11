@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ========================================
echo   번호판 인식 웹 서버 시작
echo ========================================
echo   접속 주소: http://localhost:5000
echo   브라우저가 자동으로 열립니다.
echo   종료: 이 창에서 Ctrl+C
echo ========================================
echo.
python plate_server.py --port 5000
pause
