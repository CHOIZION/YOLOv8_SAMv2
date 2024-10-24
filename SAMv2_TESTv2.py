import torch
import cv2
import time  # 시간을 측정하기 위해 time 모듈 추가
from sam2.build_sam import build_sam2_camera_predictor

# 모델 체크포인트 및 설정 파일 경로
checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

# 카메라 프레딕터 빌드
predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

# 비디오/카메라 입력 설정 (0이면 기본 카메라)
cap = cv2.VideoCapture(0)  # 또는 'video.mp4' 같은 파일 경로

if_init = False
frame_idx = 0  # 프레임 인덱스 초기화

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        start_time = time.time()  # 시작 시간 기록

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame. Exiting...")
            break

        # 프레임 크기 설정
        width, height = frame.shape[:2][::-1]

        # 첫 프레임에 프롬프트 설정하여 객체 분할 수행
        if not if_init:
            predictor.load_first_frame(frame)
            if_init = True

            # 예시: 박스 프롬프트 설정 (해당 위치에 맞는 프롬프트를 지정해야 함)
            bbox_prompt = [100, 100, 200, 200]  # 좌표는 원하는 박스로 설정 가능
            obj_id = 1  # 객체 ID 설정 (분할할 객체를 구분하는 값)
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(bbox=bbox_prompt, obj_id=obj_id, frame_idx=frame_idx)

        else:
            # 나머지 프레임에 대한 객체 추적
            out_obj_ids, out_mask_logits = predictor.track(frame)

        # 분할 결과를 마스크로 변환
        mask = (out_mask_logits > 0).to(torch.uint8) * 255  # 마스크 생성
        mask = mask.squeeze().cpu().numpy()  # 텐서에서 차원을 축소하고 NumPy 배열로 변환

        # 마스크가 2D 배열인지 확인 후 빨간색 마스크로 변환
        if mask.ndim == 2:
            # 빨간색 마스크 생성 (세 번째 채널인 빨간색에 값을 넣고 나머지는 0으로 설정)
            red_mask = cv2.merge([mask, mask * 0, mask * 0])  # R 채널에만 값, G와 B는 0

            output_frame = cv2.addWeighted(frame, 1, red_mask, 0.5, 0)  # 프레임에 마스크 오버레이

            # 결과 시각화
            cv2.imshow("Segmented Frame", output_frame)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # 프레임 인덱스 증가
        frame_idx += 1

        # 레이턴시 계산
        end_time = time.time()  # 종료 시간 기록
        latency = (end_time - start_time) * 1000  # 밀리초(ms) 단위로 레이턴시 계산
        print(f"Latency: {latency:.2f} ms")  # 레이턴시 출력

# 자원 해제
cap.release()
cv2.destroyAllWindows()
