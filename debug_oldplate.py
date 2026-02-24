import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(["ko", "en"], gpu=True)

# frame 400, 410, 490, 500 크롭 파일 OCR 테스트
import os
for f in sorted(os.listdir(".")):
    if not f.startswith("test_crop_") or not f.endswith(".jpg"):
        continue
    img = cv2.imread(f)
    if img is None:
        continue
    h, w = img.shape[:2]
    print(f"\n=== {f} ({w}x{h}) ===")

    # 원본 OCR
    res = reader.readtext(img, detail=1, paragraph=False)
    res_sorted = sorted(res, key=lambda r: r[0][0][1])  # y좌표 정렬
    texts = [r[1] for r in res_sorted]
    confs = [r[2] for r in res_sorted]
    print(f"  원본: {texts} | confs={[round(c,2) for c in confs]}")

    # 업스케일 4x + 선명화
    scale = 4.0
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    big = cv2.filter2D(big, -1, kernel)
    res2 = reader.readtext(big, detail=1, paragraph=False)
    res2_sorted = sorted(res2, key=lambda r: r[0][0][1])
    texts2 = [r[1] for r in res2_sorted]
    confs2 = [r[2] for r in res2_sorted]
    print(f"  4x선명: {texts2} | confs={[round(c,2) for c in confs2]}")

    # 하단 60%만 (주 번호 영역)
    bottom = img[int(h*0.4):, :]
    bottom_big = cv2.resize(bottom, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    res3 = reader.readtext(bottom_big, detail=1, paragraph=False)
    texts3 = [r[1] for r in res3]
    print(f"  하단60%: {texts3}")
