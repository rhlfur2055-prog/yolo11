import sys
sys.path.insert(0, 'C:/tools/yolo26')
from plate_engine_pro import PlateValidator

v = PlateValidator()

print("=" * 60)
print("1. validate() 패턴 테스트")
print("=" * 60)
tests = [
    ("376비7789",   True,  "지역명 교정 -> 서울/경기76비7789"),
    ("45소8019",    True,  "정상 인식"),
    ("서울70비9203", True,  "구형 지역번호판"),
    ("경기76비7789", True,  "구형 지역번호판"),
    ("전기8060",    True,  "전기차 번호판"),
    ("123가4567",   True,  "신형 번호판"),
    ("vin492031스", False, "쓰레기 → 차단"),
    ("씨70버9203",  False, "쓰레기 → 차단"),
    ("7457v6286",   False, "영문 혼입 → 차단"),
]
passed = 0
for inp, expect_ok, note in tests:
    ok, result = v.validate(inp)
    status = "PASS" if (ok == expect_ok) else "FAIL"
    if status == "PASS":
        passed += 1
    print(f"  {status} | {inp:14} -> {result:14} | {note}")

print(f"\n  결과: {passed}/{len(tests)} 통과\n")

print("=" * 60)
print("2. clean_ocr_text 특수문자 제거 테스트")
print("=" * 60)
garbage = [
    "vin;{492031스",
    "씨:70'버9203",
    "*1?1'16286",
    "5546'55무",
    "45소8019",
]
for t in garbage:
    cleaned = v.clean_ocr_text(t)
    ok, result = v.validate(cleaned)
    verdict = f"OK → {result}" if ok else "FAIL (차단됨)"
    print(f"  입력: {t:20} → 정제: {cleaned:12} → {verdict}")

print()
print("=" * 60)
print("3. 전역 쿨다운 변수 확인")
print("=" * 60)
try:
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("plate_server", "C:/tools/yolo26/plate_server.py")
    mod = importlib.util.module_from_spec(spec)
    # 실제 import 없이 변수 유무만 확인
    with open("C:/tools/yolo26/plate_server.py", encoding="utf-8") as f:
        src = f.read()
    has_global = "_stream_recent_plates: dict[str, float]" in src
    has_cooldown = "_STREAM_PLATE_COOLDOWN = 30.0" in src
    print(f"  전역 _stream_recent_plates : {'있음 ✓' if has_global else '없음 ✗'}")
    print(f"  _STREAM_PLATE_COOLDOWN=30초: {'있음 ✓' if has_cooldown else '없음 ✗'}")
except Exception as e:
    print(f"  확인 실패: {e}")
