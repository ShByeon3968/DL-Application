import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchinfo

from mmaction.apis import inference_recognizer, init_recognizer

def model_load(config_path:str,checkpoint_path:str):
    model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device can be 'cuda:0'
    return model


if __name__ == '__main__':
    config_path = 'mmaction2/configs/recognition/abuse/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_abuse-rgb.py'
    checkpoint_path = './work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_abuse-rgb/best_acc_top1_epoch_46.pth' # can be a local path
    video_path = './output_video_2.mp4'   # you can specify your own picture path
    model = model_load(config_path,checkpoint_path)
    model.eval()

    # 비디오 로드 및 프레임 추출
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 예제 프레임 선택 (여기서는 첫 번째 프레임 사용)
    frame = frames[0]
    img_tensor = preprocess(frame).unsqueeze(0).to("cuda:0")

    # Grad-CAM 대상 레이어 설정
    target_layer = model.backbone.layer4

    # Grad-CAM 객체 생성
    cam = GradCAM(model,target_layer)

    # 클래스 인덱스 (예: 0 = tench, 1 = goldfish, ...)
    target_category = None  # None이면 모델이 가장 확신하는 클래스를 사용

    # Grad-CAM 히트맵 계산
    grayscale_cam = cam(img_tensor)

    # 히트맵을 원본 프레임에 overlay
    grayscale_cam = grayscale_cam[0, :]
    img_with_cam = show_cam_on_image(frame / 255., grayscale_cam, use_rgb=True)

    # 원본 프레임과 Grad-CAM 히트맵을 함께 표시
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Original Frame')

    plt.subplot(1, 2, 2)
    plt.imshow(img_with_cam)
    plt.title('Grad-CAM')
    plt.show()


