# ============================================
# AI허브 데이터 다운로드 준비 (Windows PowerShell)
# ============================================
$ErrorActionPreference = "Stop"
$Base = "C:\tools\yolo26"
$Dataset = "$Base\dataset"
$AihubRaw = "$Base\dataset\aihub_raw"

New-Item -ItemType Directory -Force -Path $Dataset | Out-Null
New-Item -ItemType Directory -Force -Path $AihubRaw | Out-Null
Write-Host "[OK] 디렉터리 생성: $AihubRaw"

# pip 설치 (선택)
Write-Host "`n[안내] AI허브 CLI 사용 시:"
Write-Host "  pip install aihubshell"
Write-Host "  cd $Base\dataset"
Write-Host "  aihubshell -mode l   # 로그인 후 API KEY 인증"
Write-Host "`n[안내] 수동 다운로드 권장:"
Write-Host "  1. https://aihub.or.kr 로그인"
Write-Host "  2. 마이페이지 → 데이터 신청 내역"
Write-Host "  3. '자동차 차종/연식/번호판 인식용 영상' → 다운로드"
Write-Host "  4. 압축 해제 후 파일을 아래 경로에 저장:"
Write-Host "     $AihubRaw"
Write-Host ""
