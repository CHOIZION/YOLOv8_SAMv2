import cv2
import torch
from ultralytics import YOLO
from ultralytics import SAM

# YOLOv8 모델 로드
yolo_model = YOLO('yolov8n.pt')  # YOLOv8 모델 로드 (v8n, v8s 등)

# SAM 모델 로드
sam_model = SAM("sam2_t.pt")

# GPU 사용 여부를 로그로 출력
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("Using CPU")

# 비디오/카메라 입력 설정 (0이면 기본 카메라)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

with torch.inference_mode():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame. Exiting...")
            break

        # YOLOv8을 사용하여 객체 탐지 수행
        results = yolo_model(frame)

        # 탐지된 객체에 대한 경계 상자 추출
        boxes = []
        for result in results:
            for box in result.boxes.xyxy:
                # 경계 상자를 SAM 모델로 전달할 형식으로 변환
                x1, y1, x2, y2 = map(int, box)
                boxes.append([x1, y1, x2, y2])

        # SAM 모델을 사용하여 세분화 수행
        if boxes:
            sam_results = sam_model(frame, bboxes=boxes)

            # results는 리스트 형태이므로, 각 결과에서 마스크를 추출
            if sam_results and hasattr(sam_results[0], 'masks'):
                masks = sam_results[0].masks  # Masks 객체
                masks_data = masks.data.cpu().numpy()  # Masks 객체에서 데이터를 NumPy 배열로 추출

                output_frame = frame.copy()  # 원본 프레임 복사

                # 여러 객체에 대한 마스크 적용
                for mask in masks_data:
                    mask = (mask > 0).astype('uint8') * 255  # 마스크를 이진 이미지로 변환
                    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  # 마스크를 컬러로 변환
                    output_frame = cv2.addWeighted(output_frame, 1, colored_mask, 0.5, 0)  # 마스크 오버레이

                # 결과 시각화
                cv2.imshow("Segmented Frame", output_frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
