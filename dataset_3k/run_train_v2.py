"""
YOLO 번호판 추가 학습 스크립트 (Fine-tuning v2)
best.pt 기반 + 낮은 학습률 + 80에폭 추가
실행: python C:\tool\yolo26-main\dataset_3k\run_train_v2.py
"""
import multiprocessing


def main():
    from ultralytics import YOLO

    # 1차 학습 best.pt 기반 Fine-tuning
    model = YOLO(r"C:\tool\yolo26-main\runs\detect\plate_korean_3k3\weights\best.pt")

    # 추가 학습 (Fine-tuning - 낮은 학습률)
    results = model.train(
        data=r"C:\tool\yolo26-main\dataset_3k\data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        name="plate_korean_3k_v2",
        patience=30,            # 더 인내심 있게
        device=0,
        workers=0,

        # Fine-tuning: 낮은 학습률
        lr0=0.002,             # 1차보다 5x 낮은 학습률
        lrf=0.01,              # 최종 학습률 비율
        warmup_epochs=2,       # 짧은 워밍업
        optimizer="AdamW",     # AdamW 최적화

        # 번호판 최적화 증강
        hsv_h=0.01,            # 색상 변환 (더 보수적)
        hsv_s=0.3,             # 채도 변환
        hsv_v=0.3,             # 밝기 변환
        degrees=3.0,           # 회전 (더 작게)
        translate=0.08,        # 이동
        scale=0.2,             # 스케일
        flipud=0.0,            # 상하반전 OFF
        fliplr=0.0,            # 좌우반전 OFF
        mosaic=0.8,            # 모자이크 약간 줄임
        mixup=0.05,            # MixUp 줄임
        close_mosaic=15,       # 마지막 15에폭 모자이크 OFF

        # 저장
        save=True,
        save_period=20,
        plots=True,
    )

    print()
    print("=" * 50)
    print("  추가 학습 완료!")
    print(f"  best.pt: runs/detect/plate_korean_3k_v2/weights/best.pt")
    print("=" * 50)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
