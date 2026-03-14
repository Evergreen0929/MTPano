import torch
import torch.nn.functional as F
import torchvision
import math
import os
from PIL import Image
import numpy as np

import torch
import math

NYU_PALETTE_RGB = [
    (0, 0, 0), (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120), 
    (188, 189, 34), (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213), 
    (148, 103, 189), (196, 156, 148), (23, 190, 207), (178, 76, 76), (247, 182, 210), 
    (66, 188, 102), (219, 219, 141), (140, 57, 197), (202, 185, 52), (51, 176, 203), 
    (200, 54, 131), (92, 126, 205), (183, 117, 112), (83, 40, 154), (99, 109, 207), 
    (101, 149, 204), (107, 158, 60), (189, 200, 204), (226, 186, 174), (150, 36, 62), 
    (17, 119, 95), (8, 182, 138), (158, 202, 215), (15, 124, 157), (133, 108, 169), 
    (177, 126, 166), (194, 211, 238), (175, 194, 180), (158, 185, 199), (181, 197, 216)
]
NYUDV2_PALETTE_FLAT = []
for color in NYU_PALETTE_RGB:
    NYUDV2_PALETTE_FLAT.extend(color)
# 填充剩余的 256-40=216 个索引为黑色
NYUDV2_PALETTE_FLAT.extend([0, 0, 0] * (256 - len(NYU_PALETTE_RGB)))

def generate_box_panorama(pano_h, pano_w, side_length=2.0, device='cpu'):
    """
    生成一个代表方盒子的虚拟全景图输入 (RGB, Depth, Normal)。

    此版本最终修正了法线计算和坐标系手性问题，以完全匹配您的要求。
    
    Args:
        pano_h (int): 全景图高度。
        pano_w (int): 全景图宽度。
        side_length (float): 方盒子的边长 (米)。
        device (str): torch设备 ('cpu' 或 'cuda')。
        
    Returns:
        tuple: (pano_rgb, pano_depth, pano_normal)
    """
    # 1. 为每个全景像素计算其在标准Y-Up世界坐标系中的方向向量 (S-Space)
    v, u = torch.meshgrid(
        torch.linspace(-1, 1, pano_h, device=device),
        torch.linspace(-1, 1, pano_w, device=device),
        indexing='ij'
    )
    theta = u * math.pi
    phi = v * math.pi / 2
    
    s_x = torch.cos(phi) * torch.sin(theta)
    s_y = torch.sin(phi)
    s_z = -torch.cos(phi) * torch.cos(theta)
    
    rays_s_space = torch.stack([s_x, s_y, s_z], dim=-1)

    # 2. 计算射线与方盒子的交点和深度
    half_side = side_length / 2.0
    rays_safe = rays_s_space.clone()
    rays_safe[rays_safe.abs() < 1e-8] = 1e-8
    t = half_side / rays_safe.abs()
    t_intersect, face_axis_indices = t.min(dim=-1)
    pano_depth = t_intersect.unsqueeze(0).unsqueeze(0)

    # 3. *** 核心修正 1: 正确计算S-Space法线 ***
    #    法线的方向由射线击中的轴和射线的方向共同决定。
    axis_hot = F.one_hot(face_axis_indices, num_classes=3)
    ray_signs = torch.sign(rays_s_space)
    # 通过点乘得到每个射线击中平面的正确法线方向符号
    correct_signs = (ray_signs * axis_hot).sum(dim=-1, keepdim=True)
    s_space_normals = axis_hot.float() * correct_signs
    
    # 4. *** 核心修正 2: 翻转手性以匹配您的坐标系 ***
    #    将标准坐标系法线 (S-Space) 变换到您要求的法线坐标系 (N-Space)。
    #    通过改变一个符号，我们引入了一个反射变换，修正了手性。
    s_nx, s_ny, s_nz = s_space_normals.unbind(dim=-1)
    n_x = s_nz  # 此处移除了负号，以翻转手性
    n_y = s_ny
    n_z = s_nx
    rotated_normals = torch.stack([n_x, n_y, n_z], dim=-1)
    
    # 5. 创建最终的法线图、RGB图张量
    pano_normal = rotated_normals.permute(2, 0, 1).unsqueeze(0)
    pano_rgb = torch.full((1, 3, pano_h, pano_w), fill_value=0.5, device=device)
    
    print(f"--- Generated dummy box data with CORRECTED normals and handedness (Side Length: {side_length}m) ---")
    return pano_rgb, pano_depth, pano_normal


# --- 辅助函数：加载数据 (不变) ---
def load_rgb_image(path, device='cpu'):
    with Image.open(path).convert('RGB') as img:
        arr = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(arr / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)

def load_semantic_map(path, device='cpu'):
    with Image.open(path).convert('P') as img:
        arr = np.array(img, dtype=np.uint8)

    max_label = arr.max()
    num_classes = max_label + 1
    
    tensor = torch.from_numpy(arr).to(torch.long).to(device).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    
    one_hot = F.one_hot(tensor, num_classes=num_classes).squeeze(0).permute(0, 3, 1, 2).float()
    
    return one_hot, num_classes

def load_depth_map(path, device='cpu'):
    with Image.open(path) as img:
        depth_mm = np.array(img, dtype=np.uint16)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    tensor = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)

def load_normal_map(path, device='cpu'):
    with Image.open(path).convert('RGB') as img:
        arr = np.array(img, dtype=np.float32)
    normals = (arr / 255.0) * 2.0 - 1.0
    tensor = torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)

def save_semantic_map(tensor, path):
    indices = torch.argmax(tensor.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
    
    img = Image.fromarray(indices, 'P')
    # 应用 NYUDv2 调色板
    img.putpalette(NYUDV2_PALETTE_FLAT)
    img.save(path)

def save_depth_map(tensor, path, global_max_val=None):
    tensor_normalized = tensor.clone()
    if global_max_val is None:
        valid_mask = tensor_normalized > 0
        if valid_mask.any():
            global_max_val = torch.max(tensor_normalized[valid_mask])
    
    if global_max_val is not None and global_max_val > 0:
        tensor_normalized /= global_max_val
        
    tensor_normalized = torch.clamp(tensor_normalized, 0, 1)
    torchvision.utils.save_image(tensor_normalized.repeat(1, 3, 1, 1), path)

def save_normal_map(tensor, path):
    torchvision.utils.save_image((tensor * 0.5) + 0.5, path)

# --- 核心变换函数 ---

def get_camera_matrices(yaw, pitch, device='cpu'):
    """
    构建相机到世界的旋转矩阵和世界到相机的旋转矩阵。

    - 世界坐标系 (N-Space): +Y Up, +X Forward, +Z Right
    - 相机坐标系: +Y Up, +X Right, -Z Forward

    Args:
        yaw (float): 偏航角 (度)。围绕世界Y轴旋转。
        pitch (float): 俯仰角 (度)。围绕相机局部X轴旋转。
        device (str): torch设备。

    Returns:
        tuple: (cam_to_world_rot, world_to_cam_rot)
    """
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)

    # 围绕世界Y轴的偏航旋转
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    R_yaw = torch.tensor([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ], device=device, dtype=torch.float32)

    # 围绕相机局部X轴的俯仰旋转
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    R_pitch = torch.tensor([
        [1, 0, 0],
        [0, cos_p, -sin_p],
        [0, sin_p, cos_p]
    ], device=device, dtype=torch.float32)

    # 组合旋转: 首先应用俯仰，然后应用偏航
    cam_to_world_rot = R_yaw @ R_pitch
    world_to_cam_rot = cam_to_world_rot.T

    return cam_to_world_rot, world_to_cam_rot


def pano_to_perspective_correct(pano_rgb, pano_semantic, pano_depth, pano_normal, h_fov, yaw, pitch, persp_h, persp_w):
    """正向变换: Panorama -> Perspective (最终坐标系对齐版)"""
    device = pano_rgb.device
    B, _, PANO_H, PANO_W = pano_rgb.shape

    # 1. 为透视图像的每个像素计算相机空间中的射线
    cam_to_world_rot, world_to_cam_rot = get_camera_matrices(yaw, pitch, device)
    
    h_fov_rad = math.radians(h_fov)
    focal_length = persp_w / (2 * math.tan(h_fov_rad / 2))

    y, x = torch.meshgrid(
        torch.linspace(-persp_h / 2 + 0.5, persp_h / 2 - 0.5, persp_h, device=device),
        torch.linspace(-persp_w / 2 + 0.5, persp_w / 2 - 0.5, persp_w, device=device),
        indexing='ij'
    )

    # 相机坐标系: +X Right, +Y Up, -Z Forward
    dirs_cam = torch.stack([x, -y, -torch.full_like(x, focal_length)], dim=-1)
    dirs_cam_normalized = F.normalize(dirs_cam, p=2, dim=-1)

    # 2. 将相机射线变换到世界空间
    # (H, W, 3) @ (3, 3) -> (H, W, 3)
    rays_world_N_space = dirs_cam_normalized.view(-1, 3) @ cam_to_world_rot.T
    rays_world_N_space = rays_world_N_space.view(persp_h, persp_w, 3)

    # 3. 将世界射线(N-Space)转换为全景图采样坐标(S-Space)
    # N-Space to S-Space: (nx, ny, nz) -> (sz, ny, nx) = (sx, sy, sz)
    rays_S_space = torch.stack([
        rays_world_N_space[..., 2], 
        rays_world_N_space[..., 1],
        rays_world_N_space[..., 0]
    ], dim=-1)

    # 4. 将S-Space射线转换为球面坐标 (theta, phi) 和采样网格 (u, v)
    s_x, s_y, s_z = rays_S_space.unbind(dim=-1)
    theta = torch.atan2(s_x, -s_z)  # [-pi, pi]
    phi = torch.asin(s_y)           # [-pi/2, pi/2]

    # 归一化到 [-1, 1]
    u = theta / math.pi
    v = phi / (math.pi / 2)
    
    grid = torch.stack([u, v], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    grid = grid.expand(B, -1, -1, -1)

    # 5. 使用 grid_sample 从全景图采样
    sampled_rgb = F.grid_sample(pano_rgb, grid, mode='bilinear', padding_mode='border', align_corners=False)
    sampled_radial_depth = F.grid_sample(pano_depth, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    sampled_world_normal = F.grid_sample(pano_normal, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    persp_semantic = F.grid_sample(pano_semantic, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    
    # 6. 修正深度和法线
    # 6.1 重建世界坐标点 P_world (保持批量维度)
    P_world = rays_world_N_space.unsqueeze(0) * sampled_radial_depth.permute(0, 2, 3, 1)
    
    # 6.2 将 P_world 变换到相机空间 P_camera (保持批量维度)
    P_camera = P_world.view(B, -1, 3) @ world_to_cam_rot.T
    P_camera = P_camera.view(B, persp_h, persp_w, 3) # <--- 这是修复您看到的错误的直接原因

    # 6.3 Z-Depth 是 P_camera 的Z分量的相反数 (保持批量维度)
    persp_z_depth = -P_camera[..., 2].unsqueeze(1)

    # 6.4 将世界法线变换到相机法线 (保持批量维度)
    sampled_world_normal_flat = sampled_world_normal.permute(0, 2, 3, 1).reshape(B, -1, 3)
    persp_cam_normal_flat = sampled_world_normal_flat @ world_to_cam_rot.T
    persp_cam_normal = persp_cam_normal_flat.reshape(B, persp_h, persp_w, 3).permute(0, 3, 1, 2)

    # 掩码掉无效深度区域
    valid_mask = sampled_radial_depth > 0
    persp_rgb = sampled_rgb * valid_mask
    persp_cam_normal = persp_cam_normal * valid_mask

    persp_cam_normal = -persp_cam_normal

    return persp_rgb, persp_semantic, persp_z_depth, persp_cam_normal

def pano_to_fisheye_stereographic(pano_rgb, pano_semantic, pano_depth, pano_normal, h_fov, yaw, pitch, persp_h, persp_w):
    """
    鱼眼/小地球变换 (Stereographic Projection)
    """
    device = pano_rgb.device
    B, _, PANO_H, PANO_W = pano_rgb.shape

    # 1. 获取相机矩阵
    cam_to_world_rot, world_to_cam_rot = get_camera_matrices(yaw, pitch, device)
    
    # --- 核心修改：立体投影 (Stereographic) 的焦距计算 ---
    h_fov_rad = math.radians(h_fov)
    # 对于立体投影，边缘点满足 r = 2f * tan(theta/2)
    # 此时 theta 是相对于中心轴的角度，即 h_fov / 2
    focal_length = persp_w / (4 * math.tan(h_fov_rad / 4))

    y, x = torch.meshgrid(
        torch.linspace(-persp_h / 2 + 0.5, persp_h / 2 - 0.5, persp_h, device=device),
        torch.linspace(-persp_w / 2 + 0.5, persp_w / 2 - 0.5, persp_w, device=device),
        indexing='ij'
    )

    # --- 核心修改：鱼眼射线生成 ---
    r = torch.sqrt(x**2 + y**2)
    # 防止除零
    r_safe = torch.clamp(r, min=1e-8)
    
    # 计算射线相对于光轴的角度 theta
    # Stereographic: theta = 2 * atan(r / (2 * f))
    theta_cam = 2 * torch.atan(r / (2 * focal_length))
    
    # 生成相机坐标系下的单位射线方向: +X Right, +Y Up, -Z Forward
    # sin(theta) * (x/r) 即为单位圆在平面上的投影分量
    sin_t = torch.sin(theta_cam)
    dx = (x / r_safe) * sin_t
    dy = (-y / r_safe) * sin_t # 使用 -y 保持 Up 为正
    dz = -torch.cos(theta_cam)
    
    # 处理中心点 (r=0)
    dirs_cam_normalized = torch.stack([dx, dy, dz], dim=-1)
    mask_center = (r < 1e-8)
    dirs_cam_normalized[mask_center] = torch.tensor([0, 0, -1.0], device=device)

    # 2. 将相机射线变换到世界空间 (保持原有逻辑)
    rays_world_N_space = dirs_cam_normalized.view(-1, 3) @ cam_to_world_rot.T
    rays_world_N_space = rays_world_N_space.view(persp_h, persp_w, 3)

    # 3. N-Space to S-Space 采样逻辑 (保持原有逻辑)
    rays_S_space = torch.stack([
        rays_world_N_space[..., 2], 
        rays_world_N_space[..., 1],
        rays_world_N_space[..., 0]
    ], dim=-1)

    s_x, s_y, s_z = rays_S_space.unbind(dim=-1)
    theta_pano = torch.atan2(s_x, -s_z)
    phi_pano = torch.asin(torch.clamp(s_y, -1.0, 1.0))

    u = theta_pano / math.pi
    v = phi_pano / (math.pi / 2)
    
    grid = torch.stack([u, v], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # 4. 采样 (保持原有逻辑)
    sampled_rgb = F.grid_sample(pano_rgb, grid, mode='bilinear', padding_mode='border', align_corners=False)
    sampled_radial_depth = F.grid_sample(pano_depth, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    sampled_world_normal = F.grid_sample(pano_normal, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    persp_semantic = F.grid_sample(pano_semantic, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    
    # 5. 深度和法线修正 (保持原有逻辑)
    P_world = rays_world_N_space.unsqueeze(0) * sampled_radial_depth.permute(0, 2, 3, 1)
    P_camera = P_world.view(B, -1, 3) @ world_to_cam_rot.T
    P_camera = P_camera.view(B, persp_h, persp_w, 3)
    persp_radial_depth = sampled_radial_depth 

    # 法线修正逻辑保持不变
    sampled_world_normal_flat = sampled_world_normal.permute(0, 2, 3, 1).reshape(B, -1, 3)
    persp_cam_normal_flat = sampled_world_normal_flat @ world_to_cam_rot.T
    persp_cam_normal = persp_cam_normal_flat.reshape(B, persp_h, persp_w, 3).permute(0, 3, 1, 2)

    valid_mask = sampled_radial_depth > 0
    persp_rgb = sampled_rgb
    persp_cam_normal = -persp_cam_normal * valid_mask

    # 返回值中的深度现在是径向深度
    return persp_rgb, persp_semantic, persp_radial_depth, persp_cam_normal


def perspective_to_pano_correct(persp_rgb, persp_semantic, persp_z_depth, persp_cam_normal, h_fov, yaw, pitch, pano_h, pano_w):
    """逆向变换: Perspective -> Panorama (最终坐标系对齐版)"""
    device = persp_rgb.device
    B, _, persp_h, persp_w = persp_rgb.shape
    
    cam_to_world_rot, world_to_cam_rot = get_camera_matrices(yaw, pitch, device)

    # 1. 为全景图的每个像素计算世界空间中的射线 (N-Space)
    v, u = torch.meshgrid(
        torch.linspace(-1, 1, pano_h, device=device),
        torch.linspace(-1, 1, pano_w, device=device),
        indexing='ij'
    )
    theta = u * math.pi
    phi = v * math.pi / 2
    
    # S-Space rays
    s_x = torch.cos(phi) * torch.sin(theta)
    s_y = torch.sin(phi)
    s_z = -torch.cos(phi) * torch.cos(theta)
    
    # N-Space rays (World space)
    rays_world_N_space = torch.stack([s_z, s_y, s_x], dim=-1)

    # 2. 将世界射线变换到相机空间
    rays_cam = rays_world_N_space.view(-1, 3) @ world_to_cam_rot.T
    rays_cam = rays_cam.view(pano_h, pano_w, 3)

    # 3. 将相机射线投影到透视图像平面以创建采样网格
    cx, cy, cz = rays_cam.unbind(dim=-1)
    
    # 创建一个掩码，只处理指向相机前方的射线
    mask = (cz < -1e-8)
    
    h_fov_rad = math.radians(h_fov)
    focal_length = persp_w / (2 * math.tan(h_fov_rad / 2))
    
    u_proj = focal_length * (cx / -cz)
    v_proj = focal_length * (cy / -cz)

    u_norm = u_proj / (persp_w / 2)
    v_norm = v_proj / (persp_h / 2)
    
    grid = torch.stack([u_norm, -v_norm], dim=-1)

    valid_mask = mask & (u_norm.abs() <= 1) & (v_norm.abs() <= 1)
    
    grid[~valid_mask] = 2.0 

    grid_b = grid.unsqueeze(0).expand(B, -1, -1, -1)
    
    # 4. 使用 grid_sample 从透视图像采样
    warped_rgb = F.grid_sample(persp_rgb, grid_b, mode='bilinear', padding_mode='zeros', align_corners=False)
    sampled_z_depth = F.grid_sample(persp_z_depth, grid_b, mode='nearest', padding_mode='zeros', align_corners=False)
    sampled_cam_normal = F.grid_sample(persp_cam_normal, grid_b, mode='bilinear', padding_mode='zeros', align_corners=False)
    warped_semantic = F.grid_sample(persp_semantic, grid_b, mode='nearest', padding_mode='zeros', align_corners=False)

    # 5. 修正深度和法线
    # 5.1 将 Z-Depth 转换回径向深度 (批量安全的方式)
    t = torch.zeros(B, pano_h, pano_w, device=device)
    cz_b = cz.unsqueeze(0).expand(B, -1, -1)
    
    valid_cz_mask_b = cz_b < -1e-8
    
    # 直接在批量维度上操作
    t[valid_cz_mask_b] = -sampled_z_depth.squeeze(1)[valid_cz_mask_b] / cz_b[valid_cz_mask_b]
    
    P_cam = rays_cam.unsqueeze(0) * t.unsqueeze(-1)
    
    warped_radial_depth = torch.linalg.norm(P_cam, dim=-1, keepdim=True).permute(0, 3, 1, 2)
    
    # 批量化掩码
    valid_mask_b = valid_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
    warped_radial_depth[~valid_mask_b] = 0

    # 5.2 将相机法线转换回世界法线 (批量安全的方式)
    sampled_cam_normal_flat = (-sampled_cam_normal).permute(0, 2, 3, 1).reshape(B, -1, 3)
    warped_normal_N_space_flat = sampled_cam_normal_flat @ cam_to_world_rot.T
    warped_normal_N_space = warped_normal_N_space_flat.reshape(B, pano_h, pano_w, 3).permute(0, 3, 1, 2)
    warped_normal_N_space = warped_normal_N_space * valid_mask_b

    return warped_rgb, warped_semantic, warped_radial_depth, warped_normal_N_space, valid_mask_b

# --- 主执行流程 ---
if __name__ == '__main__':
    INPUT_DIR = "input_warp_experiment"
    OUTPUT_DIR = "output_warp_experiment"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    # --- 相机参数 ---
    PERSP_H, PERSP_W = 480, 640
    H_FOV = 120.0
    YAW = 180.0
    PITCH = 0.0

    PITCH = PITCH + 180.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading data from '{INPUT_DIR}/'...")
    pano_rgb = load_rgb_image(os.path.join(INPUT_DIR, 'rgb_rawlight.png'), device)
    pano_depth = load_depth_map(os.path.join(INPUT_DIR, 'depth.png'), device)
    pano_normal = load_normal_map(os.path.join(INPUT_DIR, 'normal.png'), device)
    pano_semantic, NUM_CLASSES = load_semantic_map(os.path.join(INPUT_DIR, 'semantic.png'), device)

    # PANO_H, PANO_W = 512, 1024 
    # _, pano_depth, pano_normal = generate_box_panorama(
    #     PANO_H, PANO_W, side_length=2.0, device=device
    # )

    BATCH_SIZE, _, PANO_H, PANO_W = pano_rgb.shape
    global_max_depth = torch.max(pano_depth)
    print(f"Global maximum depth for normalization: {global_max_depth.item():.2f} meters")

    # --- 2. 正向变换: Panorama -> Perspective ---
    print("Step 1: Transforming Panorama -> Perspective (Correctly)...")
    persp_rgb, persp_semantic, persp_z_depth, persp_cam_normal = pano_to_perspective_correct(
        pano_rgb, pano_semantic, pano_depth, pano_normal, H_FOV, YAW, PITCH, PERSP_H, PERSP_W
    )

    # --- 3. 逆向变换: Perspective -> Panorama ---
    print("Step 2: Transforming Perspective -> Panorama (Correctly)...")
    warped_rgb, warped_semantic, warped_radial_depth, warped_world_normal, warped_valid_mask = perspective_to_pano_correct(
        persp_rgb, persp_semantic, persp_z_depth, persp_cam_normal, H_FOV, YAW, PITCH, PANO_H, PANO_W
    )

    # --- 4. 保存结果 ---
    print(f"Saving results to '{OUTPUT_DIR}/'...")
    # 保存原始全景图
    torchvision.utils.save_image(pano_rgb, os.path.join(OUTPUT_DIR, "original_pano_rgb.png"))
    save_semantic_map(pano_semantic, os.path.join(OUTPUT_DIR, "original_pano_semantic.png"))
    save_depth_map(pano_depth, os.path.join(OUTPUT_DIR, "original_pano_depth.png"), global_max_depth)
    save_normal_map(pano_normal, os.path.join(OUTPUT_DIR, "original_pano_normal.png"))

    # 保存正确的透视视图结果
    torchvision.utils.save_image(persp_rgb, os.path.join(OUTPUT_DIR, "persp_rgb.png"))
    save_semantic_map(persp_semantic, os.path.join(OUTPUT_DIR, "persp_semantic_correct.png"))
    save_depth_map(persp_z_depth, os.path.join(OUTPUT_DIR, "persp_depth_correct_zdepth.png"), global_max_depth)
    save_normal_map(persp_cam_normal, os.path.join(OUTPUT_DIR, "persp_normal_correct_camera.png"))
    
    # 保存warp回来的全景图结果
    torchvision.utils.save_image(warped_rgb, os.path.join(OUTPUT_DIR, "warped_back_rgb.png"))
    save_semantic_map(warped_semantic, os.path.join(OUTPUT_DIR, "warped_back_semantic.png"))
    save_depth_map(warped_radial_depth, os.path.join(OUTPUT_DIR, "warped_back_depth.png"), global_max_depth)
    save_normal_map(warped_world_normal, os.path.join(OUTPUT_DIR, "warped_back_normal.png"))
    torchvision.utils.save_image(warped_valid_mask.float(), os.path.join(OUTPUT_DIR, "warped_back_valid_mask.png"))

    print("All files saved successfully.")

    # --- 5. 计算 L1 重建损失 ---
    print("\n--- L1 Reconstruction Loss (lower is better) ---")
    valid_mask = warped_valid_mask
    
    if valid_mask.any():
        # 我们只在被重建的区域计算损失
        l1_loss_rgb = F.l1_loss(pano_rgb[valid_mask.expand_as(pano_rgb)], warped_rgb[valid_mask.expand_as(warped_rgb)])
        l1_loss_depth = F.l1_loss(pano_depth[valid_mask], warped_radial_depth[valid_mask])
        l1_loss_normal = F.l1_loss(pano_normal[valid_mask.expand_as(pano_normal)], warped_world_normal[valid_mask.expand_as(warped_world_normal)])
        l1_loss_semantic = F.l1_loss(
            pano_semantic[valid_mask.expand_as(pano_semantic)], 
            warped_semantic[valid_mask.expand_as(warped_semantic)]
        )
        
        print(f"RGB Loss:   {l1_loss_rgb.item():.6f}")
        print(f"Semantic Loss: {l1_loss_semantic.item():.6f} (in one-hot space)")
        print(f"Depth Loss: {l1_loss_depth.item():.6f} (in meters)")
        print(f"Normal Loss:{l1_loss_normal.item():.6f}")
    else:
        print("Could not calculate L1 Loss: The warped mask is empty.")