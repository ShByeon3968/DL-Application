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

def align_point_clouds(pcd_list):
    """
    여러 포인트 클라우드를 정합하여 하나의 포인트 클라우드로 만듭니다.
    
    Parameters:
    pcd_list (list): 3D vertices numpy 배열들이 담긴 리스트
    
    Returns:
    open3d.geometry.PointCloud: 정합된 포인트 클라우드
    """
    if len(pcd_list) == 0:
        raise ValueError("The input point cloud list is empty.")
    
    # Open3D 포인트 클라우드 객체로 변환
    o3d_pcd_list = [o3d.geometry.PointCloud() for _ in pcd_list]
    for o3d_pcd, vertices in zip(o3d_pcd_list, pcd_list):
        vertices_reshaped = vertices.reshape(-1, 3)
        o3d_pcd.points = o3d.utility.Vector3dVector(vertices_reshaped)
    
    # 초기 포인트 클라우드 설정
    merged_pcd = o3d_pcd_list[0]
    
    # ICP 매개변수 설정
    threshold = 0.02
    trans_init = np.eye(4)
    
    # 나머지 포인트 클라우드 정합 및 병합
    for i in tqdm(range(1, len(o3d_pcd_list)), desc="Merging Point Clouds"):
        source = o3d_pcd_list[i]
        target = merged_pcd
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        source.transform(reg_p2p.transformation)
        merged_pcd += source
    
    return merged_pcd
