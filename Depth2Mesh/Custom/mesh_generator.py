import numpy as np
import open3d as o3d
import argparse
import cv2
from utils import *

parser = argparse.ArgumentParser(description='Generate 3D mesh from depth map and RGB image')
parser.add_argument('--depth_map', type=str, required=True, help='Path to depth map file')
parser.add_argument('--rgb_image', type=str, required=True, help='Path to RGB image file')
args = parser.parse_args()

# 1. Depth Map과 RGB 이미지 읽기
depth_map = cv2.imread(args.depth_map,cv2.IMREAD_UNCHANGED)
rgb_image = cv2.imread(args.rgb_image, cv2.IMREAD_COLOR)
scale_factor = np.sqrt(depth_map.shape[0] * depth_map.shape[1]) * 0.005
print(f'scale_factor: {scale_factor}')

# 2. Depth Map을 이용하여 Textured Mesh 생성
make_textured_mesh(args.rgb_image, args.depth_map, scale=scale_factor)
