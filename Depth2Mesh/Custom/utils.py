import numpy as np
from tqdm import tqdm
import cv2
import open3d as o3d

def move_left(mask): 
    return np.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:, 1:]
    
def move_top_right(mask): 
    return np.pad(mask,((0,1),(1,0)),'constant',constant_values=0)[1:, :-1]

def move_top(mask): 
    return np.pad(mask,((0,1),(0,0)),'constant',constant_values=0)[1:, :]

def move_top_left(mask): 
    return np.pad(mask,((0,1),(0,1)),'constant',constant_values=0)[1:, 1:]

def move_right(mask):
    return np.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:, :-1]

def move_bottom_right(mask):
    return np.pad(mask,((1,0),(1,0)),'constant',constant_values=0)[:-1, :-1]

def move_bottom(mask):
    return np.pad(mask,((1,0),(0,0)),'constant',constant_values=0)[:-1, :]

def map_depth_map_to_point_clouds(depth_map, step_size=1):
    H, W = depth_map.shape[:2]
    yy, xx = np.meshgrid(range(W), range(H))

    vertices = np.zeros((H, W, 3))
    vertices[..., 1] = xx * step_size
    vertices[..., 0] = yy * step_size
    vertices[..., 2] = depth_map

    return vertices

def construct_facets_from(mask):
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)
    facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))

    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    return np.stack((4 * np.ones(np.sum(facet_top_left_mask)),
               idx[facet_top_left_mask],
               idx[facet_bottom_left_mask],
               idx[facet_bottom_right_mask],
               idx[facet_top_right_mask]), axis=-1).astype(int)

def get_mesh_from_depth(depth_map, scale=None):
    depth_map = cv2.imread(depth_map, cv2.IMREAD_GRAYSCALE)
    if scale is None:
        scale = np.sqrt(depth_map.shape[0] * depth_map.shape[1])
    vertices = map_depth_map_to_point_clouds((1-depth_map) * scale)
    facets = construct_facets_from(np.ones(depth_map.shape).astype(bool))

    faces = []
    with tqdm(facets) as pbar:
        pbar.set_description(f'[Info] Constructing triangular faces')
        for face in pbar:
            _, v1, v2, v3, v4 = face
            faces.append([3, v1, v2, v3])
            faces.append([3, v1, v3, v4])
    faces = np.array(faces)

    return vertices, faces

def make_textured_mesh(rgb_image, depth_map,scale=None):
    textures = cv2.imread(rgb_image)
    textures = cv2.cvtColor(textures, cv2.COLOR_BGR2RGB)
    textures = textures / 255
    vertices, faces = get_mesh_from_depth(depth_map, scale)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(faces[:, 1:])
    mesh.vertex_colors = o3d.utility.Vector3dVector(textures.reshape(-1, 3))

    o3d.io.write_triangle_mesh(f'./{rgb_image[:-4]}_textured_mesh.ply', mesh)

    return mesh

def merge_pcd(pcd_files: list, output_file: str):
    # 첫 번째 포인트클라우드를 기준으로 사용
    base_pcd = o3d.io.read_point_cloud(pcd_files[0])

    # 나머지 포인트클라우드를 반복적으로 병합
    for pcd_file in pcd_files[1:]:
        current_pcd = o3d.io.read_point_cloud(pcd_file)
        
        # ICP를 사용하여 현재 포인트클라우드를 기준 포인트클라우드에 정렬
        threshold = 0.02  # ICP 알고리즘의 거리 임계값
        trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])  # 초기 변환 행렬
        reg_p2p = o3d.pipelines.registration.registration_icp(
            current_pcd, base_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        # 현재 포인트클라우드에 변환 행렬 적용
        current_pcd.transform(reg_p2p.transformation)
        
        # 병합
        base_pcd += current_pcd

    # 결과를 저장
    o3d.io.write_point_cloud(output_file, base_pcd)
    return base_pcd

def registration_pcd(source_pcd, target_pcd):
    threshold = 0.02  # ICP 알고리즘의 거리 임계값
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p
