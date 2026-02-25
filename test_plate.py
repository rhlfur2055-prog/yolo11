# -*- coding: utf-8 -*-
import cv2
import zipfile
import re
from pathlib import Path
from ultralytics import YOLO

# 설정
VIDEO_PATH   = r"C:\tool\yolo26-main\movie\hiway.mp4"
ZIP_PATH     = r"C:\Users\jomg2\OneDrive\바탕 화면\22\55저9392.zip"
MODEL_PATHS  = [
    r"C:\tool\yolo26-main\yolo11x_plate.pt",
    r"C:\tool\yolo26-main\best.pt",
]

# 정답 번호판 로드
def load_ref_plates(zip_path):
    plates = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                stem = Path(name).stem
                plate = re.sub(r'[^가-힣0-9]', '', stem)
                if len(plate) >= 4:
                    plates.append(plate)
        print(f"[정답] {len(plates)}개 로드: {plates}")
    except Exception as e:
        print(f"[오류] ZIP 읽기 실패: {e}")
    return plates

# 모델 로드
def load_model():
    for path in MODEL_PATHS:
        try:
            m = YOLO(path)
            print(f"[모델] 로드 성공: {path}")
            return m
        except Exception as e:
            print(f"[모델] 실패 {path}: {e}")
    return None

# 프레임 샘플 테스트 (2초마다 1프레임)
def test_video(model, ref_plates):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[오류] 영상 열기 실패")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[영상] {total}프레임 / {fps:.1f}fps / {total/fps:.0f}초")

    detected = []
    for sec in range(0, int(total/fps), 2):  # 2초마다
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, conf=0.3, verbose=False)[0]
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            h, w = frame.shape[:2]
            pad = 25
            crop = frame[max(0,y1-pad):min(h,y2+pad),
                         max(0,x1-pad):min(w,x2+pad)]
            # crop 저장
            save_path = fr"C:\tool\yolo26-main\crops\crop_{sec:03d}s_{conf:.2f}.jpg"
            import os; os.makedirs(r"C:\tool\yolo26-main\crops", exist_ok=True)
            cv2.imwrite(save_path, crop)
            detected.append({'sec': sec, 'conf': conf, 'crop': save_path})
            print(f"  [{sec:3d}초] 번호판 감지 conf={conf:.2f} → {save_path}")

    cap.release()

    print(f"\n[결과] 총 {len(detected)}개 번호판 감지")
    print(f"[정답] {ref_plates}")
    print(f"[저장] C:\\tool\\yolo26-main\\crops\\ 폴더 확인")
    return detected

if __name__ == "__main__":
    ref   = load_ref_plates(ZIP_PATH)
    model = load_model()
    if model:
        test_video(model, ref)
    else:
        print("[실패] 모델 없음 — .pt 파일 경로 확인 필요")
