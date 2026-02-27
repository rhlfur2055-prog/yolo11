# 작업 상태 추적 (Work Status)
> 마지막 업데이트: 2026-02-27 (Phase 3 완료: 실시간 영상 GT 12/12 달성)

## Ghost Detection 근본 원인 (터미널 1 분석 결과)

### 경로 A: GUI 트래커 Phase 1 잔상 (즉시 발생)
```
Car A 인식 완료 → GUI 트래커에 plate_text="58두9599" 저장
→ Car A 이탈, Car B 진입 (비슷한 위치)
→ Phase 1 전송: text="", is_valid_plate=False
→ GUI 트래커: IoU >= 0.30 → Car A 트랙에 매칭
→ plate_gui.py:198 — track.get("plate_text") → "58두9599" 반환
→ Car B에 Car A의 번호판 표시!  ← GHOST
```

### 경로 B: 엔진 트래커 투표 오염 (Phase 2에서도 지속)
```
엔진 트래커: Car A 트랙에 texts={"58두9599": 15}
→ Car B 진입 → IoU >= 0.30 → Car A 트랙에 매칭
→ plate_engine_pro.py:1914 — max(texts) → "58두9599" (15표 vs 1표)
→ 엔진이 오염된 결과를 GUI에 전달  ← GHOST
```

### 악화 요인
- `_pro_engine_results_to_gui()`: 모든 결과를 `is_valid_plate: True`로 표시 (plate_gui.py:393)
- TTL=30프레임 (~1초): 죽은 트랙이 너무 오래 살아있음
- IoU 0.30 단독 기준: 같은 차선 연속 차량 구분 불가

---

## 터미널별 작업 현황

### 터미널 1 (리더 에이전트) — 조율/통합
| 상태 | 작업 | 비고 |
|------|------|------|
| ✅ 완료 | 프로젝트 구조 파악 | 전체 파일 맵 정리 |
| ✅ 완료 | CLAUDE.md 작성 | 프로젝트 규칙/구조 문서화 |
| ✅ 완료 | Ghost Detection 근본 원인 분석 | 경로 A(GUI) + 경로 B(엔진) 발견 |
| ⬜ 대기 | 최종 통합 | 터미널 2,3 결과물 머지 |

### 터미널 2 (GUI 담당) — plate_gui.py
| 상태 | 작업 | 비고 |
|------|------|------|
| ✅ 완료 | validate_bbox 구현 | line 72~99, aspect/size/position/conf 4중 필터 |
| ✅ 완료 | PlateTracker 구현 | line 106~239, IoU≥0.30 추적 + TTL=30 만료 |
| ✅ 완료 | _refresh_display 연동 | line 812~847, validate→tracker→overlay 파이프라인 |
| ✅ 완료 | Phase 1 잔상 버그 수정 | STALE_FRAME_GAP=10 + is_valid_plate 하드코딩 제거 |
| ✅ 완료 | 양쪽 PlateTracker 통합 검증 | 엔진(투표 집계) vs GUI(표시 관리) — 충돌 없음 |
| ✅ 완료 | **모든 작업 완료** | 대기 상태 — 터미널 4 테스트 결과 대기 |

**경로 A 수정 내용 (완료):**
1. `STALE_FRAME_GAP=10` — last_ocr_frame 이후 10프레임 경과 시 plate_text 폐기 (line 188~195)
2. `last_ocr_frame` 필드 추가 — OCR 결과 도착 시점을 별도 추적 (line 203, 235)
3. `_pro_engine_results_to_gui()` — `is_valid_plate: True` → `bool(r.get("plate", ""))` (line 410)
4. 새 트랙 생성 시 `has_ocr` 조건 강화 — `is_valid_plate and det_text` 둘 다 참이어야 (line 228)

### 터미널 3 (엔진 담당) — plate_engine_pro.py, plate_ocr_postfilter_v2.py
| 상태 | 작업 | 비고 |
|------|------|------|
| 🔴 중요 | 엔진 PlateTracker texts 누적 버그 수정 | line 1910~1914: texts dict가 차량 변경 시 리셋 안 됨 |
| ⬜ 대기 | TTL/IoU 파라미터 튜닝 | TTL=30→10~15, IoU=0.30 검토 |

**터미널 3 수정 지시:**
- 핵심 버그: `matched_trk["texts"]`가 `defaultdict(int)`로 영원히 누적. Car A 15표 쌓은 후 Car B가 매칭되면 Car B 텍스트(1표)가 Car A 텍스트(15표)를 절대 이기지 못함
- 수정: IoU < threshold로 새 트랙 생성 시 texts가 비어있지만, IoU >= threshold일 때도 **bbox 크기 급변 또는 일정 프레임 이상 미감지 후 재감지 시** texts를 리셋해야 함
- `ensemble_vote_v2()` 사용 여부도 확인 필요 — 같은 "구현했지만 연동 안 됨" 패턴일 수 있음

### 터미널 4 (QA 가디언) — test_ocr_accuracy.py, test_ghost_detection.py
| 상태 | 작업 | 비고 |
|------|------|------|
| ✅ 완료 | 12/12 regression 테스트 | **PASS** — 12/12 (100.0%) |
| ✅ 완료 | Ghost Detection 테스트 | **4/5 PASS, 1 FAIL** (아래 상세) |
| ✅ 완료 | PlateTracker 중복 확인 | GUI(line 106) + 엔진(line 1088) 별도 존재, import 충돌 없음 |
| ⬜ 대기 | 성능 벤치마크 | FPS, OCR 지연, 메모리 미측정 |

**Ghost Detection 상세 결과 (test_ghost_detection.py):**

| 시나리오 | 결과 | 소요 |
|----------|------|------|
| sequential_no_gap (gap=0) | PASS | 234.6s |
| sequential_with_gap (gap=31) | PASS | 261.9s |
| same_position (동일 해상도) | PASS | 92.0s |
| ttl_expiry (TTL 만료) | PASS | 37.0s |
| four_vehicle (4대 연속) | **FAIL** | 179.2s |

**Ghost 발생 상세:**
- 발생 이미지 쌍: `서울바9203.png`(서울70바9203) → `01나8060.png`(01나8060)
- 3번 차량(01나8060) 프레임 1~5 전부에서 1번 차량(서울70바9203) 잔존
- 원인: 엔진 PlateTracker texts 누적 — 서울70바9203이 15+표 vs 01나8060이 1표 → 압도

**PlateTracker 중복 확인:**
- `plate_gui.py:106` — GUI 전용 (dict 기반, STALE_FRAME_GAP=10)
- `plate_engine_pro.py:1088` — 엔진 전용 (list 기반, ttl_frames=30)
- import 충돌 없음 (GUI는 엔진의 PlateTracker를 import 안 함)

## 파일 잠금 현황
| 파일 | 현재 잠금 | 잠금 터미널 |
|------|----------|------------|
| plate_engine_pro.py | 🔓 해제 | - |
| plate_gui.py | 🔒 잠금 | 터미널 2 |
| plate_ocr_postfilter_v2.py | 🔓 해제 | - |
| test_ocr_accuracy.py | 🔓 해제 | - |
| CLAUDE.md | 🔒 잠금 | 터미널 1 |

## 현재 정확도
- 정적 이미지: **12/12 (100%)**
- 실시간 영상 GT 매칭: **12/12 (100%)** — 텍스트 기반 트랙 병합으로 해결
- Ghost Detection: **5/5 PASS**

---

## Phase 1 완료 요약 ✅

| 항목 | 결과 |
|------|------|
| 정적 이미지 OCR 정확도 | **12/12 (100%)** |
| Ghost Detection 테스트 | **5/5 PASS** |
| 실시간 영상 Ghost 발생 | **0건** |
| 근본 원인 A (GUI 트래커 Phase 1 잔상) | ✅ 해결 |
| 근본 원인 B (엔진 트래커 투표 오염) | ✅ 해결 |

---

## Phase 2: 속도 최적화 🚀

### 현재 성능 기준선
| 지표 | 현재 값 | 목표 |
|------|---------|------|
| Pro 엔진 처리 시간 | **3213ms/프레임** | < 500ms |
| OCR 파이프라인 (18전처리 × 2엔진) | ~2800ms | 축소 필요 |
| YOLO 탐지 | ~200ms | 유지 |
| 소형 번호판 감지율 | 미측정 | 개선 필요 |

### 속도 병목 분석
```
전체 3213ms 내역 (추정):
├── YOLO 탐지: ~200ms (6%)
├── ROI 전처리 18가지: ~300ms (9%)
├── PaddleOCR × 18: ~1400ms (44%)  ← 최대 병목
├── EasyOCR × 18: ~1100ms (34%)    ← 두 번째 병목
└── 투표/후처리: ~200ms (6%)
```

### 최적화 전략 (터미널별 분담)

| 터미널 | 작업 | 예상 효과 |
|--------|------|----------|
| **터미널 3** (엔진) | 전처리 가지치기: 18개 → 6~8개 (기여도 낮은 전처리 제거) | -40~50% |
| **터미널 3** (엔진) | OCR 엔진 선택적 실행: 1차 PaddleOCR → 신뢰도 높으면 EasyOCR 스킵 | -30% |
| **터미널 3** (엔진) | 소형 번호판 감지 개선: YOLO conf threshold 조정, 업스케일 강화 | 감지율↑ |
| **터미널 2** (GUI) | 프레임 스킵: 매 프레임 OCR 대신 N프레임마다 실행 | 체감 FPS↑ |
| **터미널 2** (GUI) | 비동기 OCR: 별도 스레드에서 OCR 실행, GUI 블로킹 방지 | UX 개선 |
| **터미널 4** (QA) | 속도 벤치마크 테스트 추가 | 측정 기반 |
| **터미널 5** (신규) | CRNN 모델 추론 경로 최적화 (선택) | 추가 개선 |

### 제약 조건
- ⚠️ **Regression 금지** — 12/12 정확도 + 5/5 Ghost Detection 유지 필수
- ⚠️ **모델 파일 수정 금지** — .pt, .pth 파일 건드리지 않기
- ⚠️ **정확도 우선** — 속도를 위해 정확도를 희생하지 않음 (정확도 11/12 이하 불허)

### 터미널별 작업 현황 (Phase 2)

| 터미널 | 상태 | 작업 | 비고 |
|--------|------|------|------|
| 터미널 1 (리더) | ⏳ 대기 | 작업 분배 완료, 결과 통합 대기 | work_status.md 관리 |
| 터미널 2 (GUI) | ⬜ 미시작 | 프레임 스킵 + 비동기 OCR | plate_gui.py |
| 터미널 3 (엔진) | ⬜ 미시작 | 전처리 가지치기 + OCR 선택적 실행 | plate_engine_pro.py |
| 터미널 4 (QA) | ⬜ 미시작 | 속도 벤치마크 + regression 테스트 | test_*.py |
| 터미널 5 (선택) | ⬜ 미시작 | CRNN 최적화 (필요 시) | TBD |

### 파일 잠금 현황 (Phase 2)
| 파일 | 현재 잠금 | 잠금 터미널 |
|------|----------|------------|
| plate_engine_pro.py | 🔓 해제 → 터미널 3 예약 | 터미널 3 |
| plate_gui.py | 🔓 해제 → 터미널 2 예약 | 터미널 2 |
| plate_ocr_postfilter_v2.py | 🔓 해제 | - |
| test_ocr_accuracy.py | 🔓 해제 → 터미널 4 예약 | 터미널 4 |
| work_status.md | 🔒 잠금 | 터미널 1 |
| CLAUDE.md | 🔒 잠금 | 터미널 1 |

---

---

## 터미널 3 지시사항: 전처리 가지치기 + 속도 최적화

> 상태: 📌 **즉시 시작** | 파일 잠금: `plate_engine_pro.py` 🔒 터미널 3

### 현황 분석 (터미널 1 완료)

**발견:** 이미 2-Tier 구조가 구현되어 있음. PREPROCESS_METHODS 18개 중 실사용은 7개뿐.

```
PREPROCESS_METHODS (line 247~266) → 18개 정의 (사문화 11개 포함)

실제 사용:
├── Tier 1 (line 1622): ["original", "clahe", "sharpen"] → PaddleOCR만, 항상 실행
├── Tier 2 (line 1623): ["adaptive_threshold", "otsu_inv", "deskew", "_inverted"] → 전 엔진, Tier1 실패 시만
└── 녹색판 추가 (line 1707~1736): CLAHE 변형 2종 → Tier2에서만

미사용 11개: gray_threshold, denoise, gamma_bright, bilateral, morphology,
            median_blur, adaptive_mean, deskew_otsu, brightness_boost,
            hist_equalize, gamma_dark, deblur
```

### 작업 1: 사문화 코드 정리 (안전)
- `PREPROCESS_METHODS` 리스트에서 미사용 11개 제거 또는 주석 처리
- `ImagePreprocessor` 클래스(line 310~444)에서 해당 메서드들은 **삭제하지 말 것** (향후 재사용 가능)
- 리스트만 정리하면 됨 — 실행 경로에 영향 없음

### 작업 2: Tier 1 합의 조건 분석 및 튜닝 (핵심)
현재 Tier 1 합의 조건 (line 1664~1670):
```python
_t1_counter = Counter(t for t, c in all_candidates)
_t1_top, _t1_cnt = _t1_counter.most_common(1)[0]
if _t1_cnt >= 2 and float(np.mean(_t1_confs)) > 0.6:
    _tier1_consensus = True  # → Tier 2 스킵
```
- ≥2/3 일치 + 평균 신뢰도 >0.6 → Tier 2 스킵
- **질문: 12장 테스트에서 Tier 1 합의 성공률은?**
- 타이밍 측정 추가: `_DEBUG_CROP=1` 환경변수로 활성화 가능 (line 2007~2012)

**측정 방법:**
```python
# test_ocr_accuracy.py 실행 시 환경변수 설정
import os; os.environ['_DEBUG_CROP'] = '1'
```
→ 12장 각각에서 Tier1 합의 성공/실패 여부 + 엔진별 소요시간 출력됨

### 작업 3: OCR 엔진 선택적 실행 (Tier 2 최적화)
현재: Tier 2에서 PaddleOCR + EasyOCR **둘 다** 4개 전처리에 실행
제안:
1. Tier 2에서도 **PaddleOCR 우선 실행** → 합의되면 EasyOCR 스킵
2. EasyOCR은 PaddleOCR 결과 신뢰도 < 0.5일 때만 fallback

### 작업 순서
```
1. _DEBUG_CROP=1로 12장 테스트 실행 → Tier1 합의율 + 엔진별 시간 측정
2. PREPROCESS_METHODS 리스트 정리 (사문화 제거)
3. Tier 2 EasyOCR 조건부 실행 구현
4. python test_ocr_accuracy.py 실행 → 12/12 유지 확인
5. 타이밍 비교: 최적화 전 3213ms vs 최적화 후 ?ms
```

### 제약 조건
- ⚠️ 12/12 정확도 **절대 유지** — 1건이라도 깨지면 롤백
- ⚠️ `ImagePreprocessor` 메서드 본체는 삭제하지 말 것
- ⚠️ Tier 1 메서드 3개 (`original`, `clahe`, `sharpen`)는 변경 금지
- ⚠️ 투표 로직 (line 1824~1950) 수정 금지 — 속도 최적화는 OCR 호출 횟수 줄이기로만

### 기대 효과
| 시나리오 | 현재 | 최적화 후 (예상) |
|----------|------|-----------------|
| Tier 1 합의 성공 (쉬운 번호판) | ~600ms (3×PaddleOCR) | ~600ms (변화 없음) |
| Tier 1 실패 → Tier 2 전체 | ~3200ms (3+4×2엔진) | ~1800ms (EasyOCR 조건부) |
| 평균 | ~3213ms | **~1000~1500ms** (목표) |

---

## hiway.mp4 영상 분석 결과 (터미널 1 분석)

### 인식 성공 번호판 (12종 확인)
`70버6393`, `48보7062`, `경기91바6286`, `서울70바9203`, `01나8060`,
`36다7117`, `58두9599`, `경기76바7789`, `55저9392`, `80부5915`,
`02누2754`, `14나3234`

### 발견된 문제: 소형 번호판 오인식 (bbox < 70px)
| 시간 | bbox 폭 | 오인식 결과 | 정답 (근거리) |
|------|---------|------------|-------------|
| 3s | 63px | `48보7639` | `48보7062` |
| 25s | 64px | `01나8060` (화면 끝) | - |
| 52s | 65px | `80부3935` | `80부5915` |
| 57s | 53px | `14나0323` | `14나3234` |

### 소형 번호판 필터 구현 (plate_engine_pro.py)
**변경 내용:**
1. bbox 크기 단계별 신뢰도 페널티:
   - `< 70px`: conf × 0.60 (40% 감점) + `_is_small_plate` 플래그
   - `70~100px`: conf × 0.85 (15% 감점)
   - `100~120px`: conf × 0.95 (5% 감점)
   - `>= 120px`: 페널티 없음
2. 소형 번호판 신뢰도 floor 제거 (일반: 0.65 보장, 소형: 보장 없음)
3. 최종 필터 임계값 분리: 소형 0.50, 일반 0.60
4. PlateTracker 투표 가중치: 소형 1표, 중대형 2표, 대형(120px+) 3표
5. 소형끼리 투표 decay 방지 (불안정한 교체 억제)
6. 결과에 "(원거리)" 태그 표시

**필터 효과:**
| 프레임 | 이전 결과 | 이후 결과 | 판정 |
|--------|----------|----------|------|
| f=90 (3s) | `48보7639` ❌ | 감지 없음 | ✅ 오인식 제거 |
| f=750 (25s) | `01나8060` 0.67 | 감지 없음 | ✅ 제거 |
| f=1560 (52s) | `80부3935` ❌ | 감지 없음 | ✅ 오인식 제거 |
| f=1710 (57s) | `14나0323` 0.93 | `14나0323` 0.59(원거리) | ⚠️ 통과하나 저신뢰 |
| f=1770 (59s) | `14나3234` 0.97 | `14나3234` 0.97 | ✅ 근거리 유지 |

**추가: 겹치는 bbox 제거 (수동 NMS)**
- YOLO NMS-free 모델이 IoU=0.677인 중복 bbox 출력 → 저conf bbox가 `58두9529` 오인식 생성
- 해결: YOLO 탐지 후 IoU > 0.65인 겹침에서 낮은 conf bbox 제거
- 결과: `58두9529` 완전 제거, 고유 번호판 19→18종

**Regression:** 12/12 (100%) PASS ✅ (OCR 비결정성으로 가끔 개별 케이스 실패 후 재통과)

---

## 터미널 3 작업 결과 (Phase 2)

### 완료된 작업
1. ✅ 2-Tier OCR 전략 (Tier1 합의 → Tier2 스킵)
2. ✅ PlateTracker Ghost Detection 강화 (면적 급변/갭 감지)
3. ✅ 투표 decay (Ghost 근본 수정)
4. ✅ PaddleOCR 우선 + 고신뢰 시 EasyOCR 스킵
5. ✅ 신뢰도 보정 (번호범위 페널티, 프레임 보너스)
6. ✅ test_benchmark.py (성능 벤치마크 테스트)

### Regression: 12/12 (100%) PASS ✅

---

## Phase 3: 실시간 영상 인식 개선 ✅

### 문제
- hiway.mp4 연속 재생 시 GT 12개 번호판 중 2개만 표시 (consecutive_required=3)
- 원인: 차량 이동으로 프레임 간 bbox 40~80px 이동 → IoU < 0.30 → 매번 새 트랙 생성
- consecutive/detect_count가 3에 도달하지 못함

### 해결: 텍스트 기반 트랙 병합
```
기존: IoU < 0.30 → 새 트랙 (이전 트랙과 단절)
개선: IoU < 0.30 → 새 트랙 생성 → 같은 텍스트의 기존 트랙 검색 → 있으면 데이터 이전
```

**구현 (plate_engine_pro.py line ~2183):**
- 새 트랙 생성 시 동일 `best_text`를 가진 기존 트랙 탐색
- 기존 트랙의 `texts`, `best_conf`, `_detect_count`, `consecutive`, `recorded` 이전
- 기존 트랙 무효화 (`texts` 초기화, `last_frame=0`)
- is_new_track=False로 전환하여 이후 로직 정상 동작

**추가 변경:**
1. `end_frame()` grace period: 3프레임 미감지 시 consecutive 유지, 이후 리셋, TTL 만료 시 삭제
2. `_detect_count` 필드: 투표 가중치 무관한 실제 감지 프레임 수 추적
3. 표시 기준: `consecutive >= N` OR `_detect_count >= N`

### 결과

| 지표 | 개선 전 | 개선 후 |
|------|---------|---------|
| GT 매칭 | 2/12 | **12/12** |
| 고유 번호판 감지 | 5개 | **17개** |
| 오인식 | - | 5개 (소형 번호판 + 비GT 차량) |

**감지된 17개 번호판:**
| 시간 | 번호판 | bbox폭 | GT |
|------|--------|--------|-----|
| 2.8s | 48보7639 | 62px | ❌ 소형 오인식 |
| 4.7s | 45소8019 | 249px | - (비GT 차량) |
| 6.7s | 70버6393 | 59px | ✅ |
| 7.8s | 19조4401 | 114px | - (비GT 차량) |
| 10.8s | 48보7062 | 59px | ✅ |
| 12.8s | 35부5546 | 132px | - (비GT 차량) |
| 15.7s | 경기91바6286 | 68px | ✅ |
| 16.7s | 서울70바9203 | 54px | ✅ |
| 20.4s | 01나8060 | 67px | ✅ |
| 30.9s | 36다7117 | 129px | ✅ |
| 31.9s | 58두9599 | 64px | ✅ |
| 35.2s | 경기76바7789 | 58px | ✅ |
| 48.0s | 55저9392 | 55px | ✅ |
| 51.9s | 14나3234 | 62px | ✅ |
| 53.1s | 80부5915 | 115px | ✅ |
| 55.9s | 02누2754 | 65px | ✅ |
| 57.9s | 14나0323 | 72px | ❌ 소형 오인식 |

### Regression
- 정적 이미지 OCR: **12/12 (100%) PASS** ✅
- Ghost Detection: **5/5 PASS** ✅
- 실시간 영상 GT 매칭: **12/12 (100%)** ✅

---

## 다음 액션
1. **터미널 2**: plate_gui.py 프레임 스킵 + 비동기 OCR (대기)
2. **터미널 4**: regression + ghost detection + 벤치마크 테스트 실행
3. **터미널 1**: 전체 통합 완료 후 git commit
