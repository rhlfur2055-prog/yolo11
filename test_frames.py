"""Quick recognition test on CCTV video frames."""
import cv2
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

from plate_recognition_4k import PlateRecognizer

print('=== 번호판 인식 테스트 ===', flush=True)
rec = PlateRecognizer(model_size='n', confidence_threshold=0.15)

VIDEO = 'temp_youtube/KakaoTalk_20260219_123507625.mp4'
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'FPS: {fps:.1f}, 총 프레임: {total}', flush=True)

found = 0
for frame_idx in [30, 60, 90, 120, 150, 180, 210, 240, 300, 360]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    results = rec.process_frame(frame, frame_idx)
    if results:
        for r in results:
            txt = r.get('text', '')
            if txt:
                print(f'  Frame {frame_idx:4d}: [{txt}] score={r["pattern_score"]:.2f} valid={r["is_valid_plate"]}', flush=True)
                found += 1
    else:
        print(f'  Frame {frame_idx:4d}: 탐지없음', flush=True)

cap.release()
print(f'=== 총 {found}건 인식 ===', flush=True)
