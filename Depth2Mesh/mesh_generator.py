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
# depth_map = np.load('path_to_depth_map.npy')  # depth map을 numpy 배열로 불러오기
# rgb_image = o3d.io.read_image('path_to_rgb_image.png')  # RGB 이미지를 Open3D 이미지로 불러오기
scale_factor = np.sqrt(depth_map.shape[0] * depth_map.shape[1]) * 0.005
print(f'scale_factor: {scale_factor}')
make_textured_mesh(args.rgb_image, args.depth_map, scale=scale_factor)

# # 2. Point Cloud 생성 (RGB 정보 포함)
# fx = 525.0  # 카메라의 초점 거리 (focal length) 예시
# fy = 525.0  # 카메라의 초점 거리 (focal length) 예시
# cx = width // 2  # 카메라의 중심 좌표 (cx) 예시
# cy = height // 2  # 카메라의 중심 좌표 (cy) 예시

# # Z 축 (depth) 값을 이용하여 3D 좌표로 변환
# points = []
# colors = []
# for v in range(height):
#     for u in range(width):
#         z = depth_map[v, u] / 1000.0  # mm to meters
#         if z == 0:  # depth 값이 0인 경우 skip
#             continue
#         x = (u - cx) * z / fx
#         y = (v - cy) * z / fy
#         points.append([x, y, z])
#         colors.append(rgb_image[v, u, :] / 255.0)  # RGB 값 (0~1 사이로 정규화)

# points = np.array(points)
# colors = np.array(colors)

# # Open3D를 이용하여 포인트 클라우드 생성
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# # 3. Mesh 생성
# pcd.estimate_normals()
# poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]

# # 메쉬를 정리하고 시각화
# bbox = pcd.get_axis_aligned_bounding_box()
# poisson_mesh = poisson_mesh.crop(bbox)

# # 4. 텍스쳐 입히기
# poisson_mesh.vertex_colors = pcd.colors

# # 5. 시각화
# o3d.visualization.draw_geometries([poisson_mesh], window_name="Textured 3D Mesh from Depth Map and RGB Image")
