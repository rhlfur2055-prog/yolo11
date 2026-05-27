# PowerShell: 번호판 인식 웹 서버 실행
# 사용법: .\run_plate_server.ps1  (또는 우클릭 → PowerShell에서 실행)
$Host.UI.RawUI.OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "========================================"
Write-Host "  번호판 인식 웹 서버 시작"
Write-Host "========================================"
Write-Host "  접속 주소: http://localhost:5000"
Write-Host "  브라우저가 자동으로 열립니다."
Write-Host "  종료: 이 창에서 Ctrl+C"
Write-Host "========================================"
Write-Host ""

python plate_server.py --port 5000

Read-Host "종료하려면 Enter 키를 누르세요"
