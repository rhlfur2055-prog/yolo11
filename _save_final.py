import json, re, os

with open('plate_results_v3/results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Valid plate patterns
valid_patterns = [
    r'^[가-힣]{2,3}\d{2}[가-힣]\d{4}$',   # 서울70바9203
    r'^\d{2,3}[가-힣]\d{4}$',               # 14구3234
]

# Filter valid plates, deduplicate by best confidence
best = {}
for d in data:
    t = d['text']
    c = d.get('ocr_confidence', 0)
    is_valid = any(re.match(p, t) for p in valid_patterns)
    if not is_valid:
        continue
    if t not in best or c > best[t]['ocr_confidence']:
        best[t] = d

result = sorted(best.values(), key=lambda x: -x['ocr_confidence'])

# Write analysis
with open('_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(f'Valid unique plates: {len(result)}\n')
    for r in result:
        f.write(f'  {r["text"]} ({r["ocr_confidence"]:.1%})\n')

# Save to desktop
desktop = os.path.join(os.path.expanduser('~'), 'OneDrive', '바탕 화면')
if not os.path.isdir(desktop):
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')

out_path = os.path.join(desktop, '최종.json')
clean = []
for r2 in result:
    c2 = {k: v for k, v in r2.items() if k not in ('plate_image', 'preprocessed_image', 'index')}
    clean.append(c2)

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(clean, f, ensure_ascii=False, indent=2)

print(f'Saved {len(clean)} plates to {out_path}')
