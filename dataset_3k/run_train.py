"""
YOLO 번호판 학습 스크립트
실행: python C:\tool\yolo26-main\dataset_3k\run_train.py
"""
import multiprocessing


def main():
    from ultralytics import YOLO

    # 모델 로딩 (전이학습)
    model = YOLO("yolo11n.pt")

    # 학습 시작!
    results = model.train(
        data=r"C:\tool\yolo26-main\dataset_3k\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="plate_korean_3k",
        patience=20,
        device=0,
        workers=0,             # Windows 멀티프로세싱 문제 방지

        # 한국 번호판 최적화 옵션
        lr0=0.01,          # 초기 학습률
        lrf=0.01,          # 최종 학습률 비율
        warmup_epochs=3,   # 워밍업
        hsv_h=0.015,       # 색상 변환 (번호판은 낮게)
        hsv_s=0.4,         # 채도 변환
        hsv_v=0.4,         # 밝기 변환
        degrees=5.0,       # 회전 (번호판은 작게!)
        translate=0.1,     # 이동
        scale=0.3,         # 스케일
        flipud=0.0,        # 상하반전 OFF (번호판!)
        fliplr=0.0,        # 좌우반전 OFF (번호판!)
        mosaic=1.0,        # 모자이크 증강
        mixup=0.1,         # MixUp 증강

        # 저장
        save=True,
        save_period=10,    # 10에폭마다 체크포인트
        plots=True,        # 학습 그래프 저장
    )

    print()
    print("=" * 50)
    print("  학습 완료!")
    print(f"  best.pt: runs/detect/plate_korean_3k/weights/best.pt")
    print("=" * 50)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
