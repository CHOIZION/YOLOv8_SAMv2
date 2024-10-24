from ultralytics import YOLO

if __name__ == "__main__":
    # 모델 로드
    model = YOLO('yolov8x.pt')  # 'yolov8x.pt'는 사전 훈련된 모델입니다.

    # 학습
    results = model.train(
        data='Fan_Test.v1i.yolov8/data.yaml',  # 데이터셋 경로
        epochs=100,                            # 총 100 epochs
        imgsz=640,                             # 이미지 크기 (기본 640)
        batch=16,                              # 배치 크기
        workers=8,                             # 데이터 로딩에 사용할 CPU 스레드 수
        patience=10,                           # early stopping patience
        split=0.75,                            # train/val split 비율 (0.75 대 0.25)
        save=True,                             # 학습 결과 저장
        model='yolov8x.pt',                    # 가중치 파일 사용
        name='custom_yolov8_experiment'        # 결과 저장 폴더 이름
    )

    # 학습 완료 후 최상의 weight 파일(best.pt)을 사용하여 추론
    model = YOLO('runs/train/custom_yolov8_experiment/weights/best.pt')
