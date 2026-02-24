@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ========================================
echo   실시간 번호판 인식 (웹캠)
echo ========================================
echo   종료: 창에서 [q] 키
echo ========================================
echo.
python realtime_plate.py --source 0
pause
