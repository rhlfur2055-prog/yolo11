import cv2
from ultralytics import YOLO

# Stage 1: 차량 탐지 (COCO yolov8n)
model_vehicle = YOLO('yolov8n.pt')
# Stage 2: 번호판 탐지 (커스텀 plate_korean_3k_v2)
model_plate = YOLO('runs/detect/plate_korean_3k_v2/weights/best.pt')

cap = cv2.VideoCapture('movie/hiway.mp4')

for f in [400, 500, 600, 900, 1000]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if not ret: continue

    # Stage 1: 차량 찾기
    v_results = model_vehicle(frame, conf=0.3, classes=[2,5,7], imgsz=640, verbose=False)
    vehicles = v_results[0].boxes

    print(f'\nF{f}: 차량 {len(vehicles)}대')

    for v in vehicles:
        vx1,vy1,vx2,vy2 = [int(x) for x in v.xyxy[0].tolist()]
        vehicle_crop = frame[vy1:vy2, vx1:vx2]

        if vehicle_crop.size == 0:
            continue

        # Stage 2: 차량 크롭 내에서 번호판 찾기
        p_results = model_plate(vehicle_crop, conf=0.25, imgsz=640, verbose=False)
        plates = p_results[0].boxes

        cls_name = {2:'car',5:'bus',7:'truck'}.get(int(v.cls[0]),'?')
        print(f'  {cls_name} ({vx2-vx1}x{vy2-vy1}) → 번호판 {len(plates)}개')

        for p in plates:
            px1,py1,px2,py2 = [int(x) for x in p.xyxy[0].tolist()]
            pw = px2-px1
            ph = py2-py1
            print(f'    plate bbox: {pw}x{ph} conf={p.conf[0]:.2f}')

cap.release()
print('\n완료')
