import torch
import torchvision
import torch.nn as nn
from typing import Dict, Iterable, Callable

from Model import Attention

###################### accident box 만들기 #############################
_,alpha = lstm(video)
bbx_indices = alpha[alpha >= 0.6]

import json
from pathlib import Path
import os

def list_files_in_directory(directory):
    # 디렉토리 내의 모든 파일 나열
    p = Path(directory)
    files = [file.name for file in p.iterdir() if file.is_file()]
    
    return files

directory_path = '/home/aikusrv02/ai_hanmuncheol/yolo_features/testing/negative'
files = list_files_in_directory(directory_path)
print(files[0])
file_name = files[0]

file_path = os.path.join(directory_path, file_name)

with open(file_path,'r') as f:
    data = json.load(f)
# print("JSON 데이터:", data)  
accident_box = torch.tensor(data[bbx_indices[1]]['box'][bbx_indices[2]]) 
print(accident_box)


###################### 만든 accident를 image에 시각화 #############################
import torch
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt


def visualization(video, bboxes):
    video_with_boxes = []
    for i in range(video.shape[0]):       
        image = video[i]
        height, width, _ = image.shape

        boxes = []
        # 바운딩 박스 그리기
        for bbox in bboxes:
            x_center, y_center, w, h = bbox
            # YOLO 좌표를 픽셀 좌표로 변환
            x_center = x_center * width
            y_center = y_center * height
            w = w * width
            h = h * height
            
            # 좌상단 좌표와 우하단 좌표 계산
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            # 사각형 그리기
            boxes.append([x1,y1,x2,y2])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        image_with_boxes = draw_bounding_boxes(image, boxes, colors="red", width=2)
        
        video_with_boxes.append(image_with_boxes)
    
    video_with_boxes_tensor = torch.stack(video_with_boxes, dim=0)
    
    video_name = 'accident_video.mp4'
    tensor_to_video(video_with_boxes_tensor, video_name)
    
    print(f"Video saved as {video_name}")
    
    return

###################### accident video 생성 #############################

import cv2

def tensor_to_video(tensor, video_name, frame_rate=30):
    # 텐서 형태 확인
    assert tensor.ndimension() == 4, "Tensor must be 4-dimensional"
    assert tensor.size(1) == 3, "Tensor second dimension must have size 3 (RGB channels)"

    # 텐서에서 프레임 수, 채널, 높이, 너비 가져오기
    frames, channels, height, width = tensor.shape

    # 비디오 작성기 초기화
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for i in range(frames):
        # 텐서에서 한 프레임 가져오기
        frame = tensor[i]
        
        # PyTorch 텐서를 NumPy 배열로 변환
        frame = frame.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        
        # NumPy 배열을 uint8 형식으로 변환 (OpenCV는 이 형식을 사용함)
        frame = (frame * 255).astype('uint8')

        # BGR 형식으로 변환 (OpenCV는 BGR 형식을 사용함)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 비디오에 프레임 추가
        video.write(frame)

    # 비디오 작성기 릴리즈
    video.release()