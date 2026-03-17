#!/bin/bash
# ==============================================
# YOLO26 PostgreSQL 환경 자동 셋업 & 헬스체크
# ==============================================
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "  YOLO26 PostgreSQL Setup"
echo "========================================"

# ── 1. Docker 실행 확인 ──
echo -e "\n${YELLOW}[1/3]${NC} Docker 상태 확인..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker가 설치되어 있지 않습니다.${NC}"
    echo "  → https://docs.docker.com/get-docker/ 에서 설치하세요."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}✗ Docker 데몬이 실행 중이 아닙니다.${NC}"
    echo "  → Docker Desktop을 실행하세요."
    exit 1
fi

echo -e "${GREEN}✓ Docker 정상 실행 중${NC}"

# ── 2. PostgreSQL 컨테이너 시작 (DB 자동 생성) ──
echo -e "\n${YELLOW}[2/3]${NC} PostgreSQL 컨테이너 시작..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

docker compose up -d

echo "  헬스체크 대기 중..."
RETRIES=0
MAX_RETRIES=15
until docker compose exec -T postgres pg_isready -U mbc -d yolo26 &> /dev/null; do
    RETRIES=$((RETRIES + 1))
    if [ $RETRIES -ge $MAX_RETRIES ]; then
        echo -e "${RED}✗ PostgreSQL 시작 시간 초과${NC}"
        docker compose logs postgres
        exit 1
    fi
    sleep 2
done

echo -e "${GREEN}✓ PostgreSQL 정상 실행 (yolo26 DB 생성 완료)${NC}"

# ── 3. 테이블 자동 생성 확인 ──
echo -e "\n${YELLOW}[3/3]${NC} 테이블 생성 확인..."

TABLE_COUNT=$(docker compose exec -T postgres psql -U mbc -d yolo26 -t -c \
    "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")
TABLE_COUNT=$(echo "$TABLE_COUNT" | tr -d ' ')

if [ "$TABLE_COUNT" -ge 3 ]; then
    echo -e "${GREEN}✓ 테이블 ${TABLE_COUNT}개 확인됨${NC}"
    docker compose exec -T postgres psql -U mbc -d yolo26 -c \
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;"
else
    echo -e "${RED}✗ 테이블이 생성되지 않았습니다 (${TABLE_COUNT}개)${NC}"
    exit 1
fi

echo -e "\n========================================"
echo -e "  ${GREEN}셋업 완료! 엔진을 실행하세요.${NC}"
echo -e "========================================"
echo ""
echo "  접속 정보:"
echo "    Host: localhost"
echo "    Port: 5432"
echo "    DB:   yolo26"
echo "    User: mbc"
echo "    PW:   1234"
echo ""
echo "  명령어:"
echo "    시작: docker compose up -d"
echo "    중지: docker compose down"
echo "    로그: docker compose logs -f postgres"
echo "    접속: docker compose exec postgres psql -U mbc -d yolo26"
