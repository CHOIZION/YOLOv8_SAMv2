import cv2
from ultralytics import SAM
import torch

# SAM 모델 로드
model = SAM("sam2_t.pt")

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

        # 프레임 크기 설정
        height, width = frame.shape[:2]

        # 예시: 특정 박스 좌표로 객체 분할 실행
        bboxes = [100, 100, 200, 200]  # 좌표는 필요에 맞게 수정
        results = model(frame, bboxes=bboxes)

        # results는 리스트 형태이므로, 각 결과에서 마스크를 추출
        if results and hasattr(results[0], 'masks'):
            masks = results[0].masks  # Masks 객체
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
