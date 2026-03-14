import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import math
import glob
import open3d as o3d
import random
import matplotlib.pyplot as plt
import matplotlib

# 新增引用
from easydict import EasyDict as edict
from huggingface_hub import hf_hub_download

from utils.common_config import get_model 
from utils.panorama_utils import pano_to_perspective_correct, pano_to_fisheye_stereographic

# ================= 新增：直接从 app.py 拿过来的硬编码 Config =================
def get_inference_config():
    cfg = edict()
    cfg.model = 'TransformerBFE-DINO-DPT'
    cfg.backbone = 'dinov3L'
    cfg.head = 'dpt_head'
    cfg.embed_dim = 512
    cfg.mtt_resolution_downsample_rate = 2
    cfg.PRED_OUT_NUM_CONSTANT = 64
    cfg.train_db_name = 'PanoMTDU'
    
    cfg.task_dictionary = edict({
        'include_semseg': True,
        'include_depth': True,
        'include_normals': True
    })
    
    cfg.TASKS = edict()
    cfg.TASKS.NAMES = ['semseg', 'depth', 'normals']
    cfg.TASKS.NUM_OUTPUT = {
        'semseg': 150,  
        'depth': 1,
        'normals': 3
    }
    
    cfg.TEST = edict()
    cfg.TEST.SCALE = (512, 1024)
    cfg.TEST.PANO_SCALE = (512, 1024)

    cfg.TRAIN = edict()
    cfg.TRAIN.SCALE = (512, 1024)
    cfg.TRAIN.PANO_SCALE = (512, 1024)
    return cfg

# --- 1. ADE20K Palette ---
ADE20K_PALETTE = [
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
    [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 174, 255], [0, 122, 255],
    [245, 0, 255], [255, 6, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
    [0, 204, 255], [255, 0, 112], [0, 8, 255], [255, 0, 31], [255, 61, 0],
    [204, 0, 255], [255, 0, 204], [255, 255, 0], [0, 153, 255], [0, 102, 255],
    [0, 255, 245], [0, 255, 102], [255, 163, 0], [255, 153, 0], [0, 255, 10],
    [255, 112, 0], [143, 255, 0], [0, 41, 255], [0, 255, 174], [255, 0, 10],
    [174, 255, 0], [255, 245, 0], [255, 0, 20], [255, 0, 143], [255, 0, 82],
    [0, 245, 255], [0, 61, 255], [0, 255, 71], [0, 255, 153], [255, 0, 163],
    [255, 0, 174], [255, 0, 51], [255, 0, 71], [0, 204, 255], [255, 10, 0],
    [0, 255, 41], [0, 255, 51], [255, 204, 0], [255, 0, 194], [255, 102, 0],
    [0, 153, 255], [0, 102, 255], [0, 255, 204], [255, 0, 224], [255, 0, 92],
    [255, 0, 112], [255, 0, 122], [0, 255, 31], [0, 102, 255], [255, 0, 153],
    [255, 0, 143], [255, 0, 163], [255, 0, 6], [255, 0, 184], [0, 255, 214],
    [0, 255, 194], [255, 0, 71], [0, 255, 224], [255, 0, 143], [255, 0, 133],
    [122, 255, 0], [255, 0, 10], [255, 153, 0], [0, 112, 255], [255, 163, 0],
    [255, 204, 0], [255, 0, 41], [255, 0, 10], [255, 0, 20], [255, 0, 204],
    [255, 0, 194], [255, 0, 153], [255, 10, 0], [255, 0, 122], [255, 0, 71],
    [255, 0, 51], [255, 0, 31], [102, 255, 0], [0, 255, 10], [172, 255, 0],
    [255, 29, 0], [255, 0, 28], [255, 122, 0], [0, 255, 143], [255, 255, 184]
]

def smooth_step(t):
    return 3 * t**2 - 2 * t**3

def fix_panorama_seam(img_np, margin=2, task_type='depth'):
    H, W = img_np.shape[:2]
    rolled = np.roll(img_np, W // 2, axis=1)
    center = W // 2
    
    left_idx = center - margin - 1
    right_idx = center + margin
    left_anchors = rolled[:, left_idx]
    right_anchors = rolled[:, right_idx]
    
    if task_type == 'depth':
        valid_rows = (left_anchors > 1e-4) & (right_anchors > 1e-4)
        invalid_val = 0.0
    elif task_type == 'normal':
        valid_left = np.any(left_anchors != 0.0, axis=-1)
        valid_right = np.any(right_anchors != 0.0, axis=-1)
        valid_rows = valid_left & valid_right
        invalid_val = 0.0
    else:
        return img_np
        
    steps = 2 * margin + 1
    for i, col in enumerate(range(center - margin, center + margin)):
        alpha = (i + 1) / steps 
        
        if task_type == 'depth':
            interp = (1 - alpha) * left_anchors[valid_rows] + alpha * right_anchors[valid_rows]
            rolled[valid_rows, col] = interp
            
        elif task_type == 'normal':
            vec_left = left_anchors[valid_rows]
            vec_right = right_anchors[valid_rows]
            vec_interp = (1 - alpha) * vec_left + alpha * vec_right
            
            norms = np.linalg.norm(vec_interp, axis=-1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            vec_norm = vec_interp / norms
            
            rolled[valid_rows, col] = vec_norm
            
        rolled[~valid_rows, col] = invalid_val

    return np.roll(rolled, -(W // 2), axis=1)

def colorize_depth_strict(depth_map_np, global_max, cmap_name="magma"):
    valid_mask = depth_map_np > 1e-4
    if not valid_mask.any():
        return np.zeros((depth_map_np.shape[0], depth_map_np.shape[1], 3), dtype=np.uint8)

    depth_norm = np.clip(depth_map_np / global_max, 0.0, 1.0)
    
    try:
        colormap_func = matplotlib.colormaps[cmap_name]
    except AttributeError:
        colormap_func = plt.get_cmap(cmap_name)
        
    colored = colormap_func(depth_norm)[..., :3] 
    depth_rgb = (colored * 255).astype(np.uint8)
    depth_rgb[~valid_mask] = [0, 0, 0]
    return cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)

# ================= 原有保留逻辑 =================

def ensure_save_point_cloud(points, colors, normals, filename):
    if o3d is None:
        print(f"Open3D not installed, skipping point cloud: {filename}")
        return
    if isinstance(points, torch.Tensor): points = points.cpu().numpy()
    if isinstance(colors, torch.Tensor): colors = colors.cpu().numpy()
    if isinstance(normals, torch.Tensor): normals = normals.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(filename, pcd)

def colorize_semantic(semseg_tensor, palette_name='ade20k'):
    semseg_np = semseg_tensor.squeeze().cpu().numpy().astype(np.uint8)
    h, w = semseg_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    palette = np.array(ADE20K_PALETTE, dtype=np.uint8)
        
    max_idx = len(palette) - 1
    safe_semseg = np.clip(semseg_np, 0, max_idx)
    color_img = palette[safe_semseg]
    
    if semseg_np.max() > max_idx:
         mask = semseg_np > max_idx
         color_img[mask] = [0, 0, 0]

    return torch.from_numpy(color_img / 255.0).permute(2, 0, 1).unsqueeze(0).float()

def save_colored_depth(depth_tensor, path, cmap_name="Spectral"):
    d_sq = depth_tensor.squeeze()  
    d_np = d_sq.cpu().numpy()      
    
    valid_mask = (d_np > 1e-4)     
    
    if not valid_mask.any():
        cv2.imwrite(path, np.zeros((d_np.shape[0], d_np.shape[1], 3), dtype=np.uint8))
        return

    current_max = d_np.max()
    if current_max == 0: current_max = 1.0

    depth_norm = np.clip(d_np / current_max, 0.0, 1.0)
    
    try:
        colormap_func = matplotlib.colormaps[cmap_name]
    except AttributeError:
        colormap_func = plt.get_cmap(cmap_name)
    
    colored = colormap_func(depth_norm)[..., :3] 
    depth_rgb = (colored * 255).astype(np.uint8)
    depth_rgb[~valid_mask] = [0, 0, 0]
    
    img_bgr = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def blend_mask_rgb(mask_bgr, rgb_bgr, alpha=0.7):
    if mask_bgr.shape != rgb_bgr.shape:
        mask_bgr = cv2.resize(mask_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    beta = 1.0 - alpha
    return cv2.addWeighted(mask_bgr, alpha, rgb_bgr, beta, 0.0)

# ================= 核心修改一：包含 Seam Fix 逻辑的推理 =================

def run_inference(model, img_tensor, device, palette_name='ade20k', delete_window=False):
    with torch.no_grad():
        inputs = img_tensor.to(device)
        outputs = model(inputs)
        
        results = {}
        if 'semseg' in outputs:
            sem_logits = outputs['semseg']
            sem_pred = torch.argmax(sem_logits, dim=1, keepdim=True)
            results['semseg'] = sem_pred.float()
            
        if 'depth' in outputs:
            results['depth'] = outputs['depth'] 
            results['raw_depth'] = outputs['depth'].clone()
            
        if 'normals' in outputs:
            norm_out = outputs['normals']
            norm_out = F.normalize(norm_out, p=2, dim=1)
            results['normals'] = norm_out 

        filter_class_ids = []
        if palette_name == 'ade20k':
            filter_class_ids = [2]
            if delete_window:
                filter_class_ids.append(8)
                filter_class_ids.append(68)

        if filter_class_ids and 'semseg' in results and ('depth' in results or 'normals' in results):
            sem_map = results['semseg']
            
            filter_mask = torch.zeros_like(sem_map, dtype=torch.bool)
            for cid in filter_class_ids:
                filter_mask = filter_mask | (sem_map == cid)
            
            if 'depth' in results:
                results['depth'][filter_mask] = 0.0
                results['raw_depth'][filter_mask] = 0.0  
            
            if 'normals' in results:
                filter_mask_3d = filter_mask.repeat(1, 3, 1, 1)
                results['normals'][filter_mask_3d] = 0.0

        if 'depth' in results:
            depth_np = results['depth'].squeeze().cpu().numpy()
            fixed_depth = fix_panorama_seam(depth_np, margin=2, task_type='depth')
            results['depth'] = torch.from_numpy(fixed_depth).unsqueeze(0).unsqueeze(0).to(device)
            
            raw_depth_np = results['raw_depth'].squeeze().cpu().numpy()
            fixed_raw_depth = fix_panorama_seam(raw_depth_np, margin=2, task_type='depth')
            results['raw_depth'] = torch.from_numpy(fixed_raw_depth).unsqueeze(0).unsqueeze(0).to(device)
            
        if 'normals' in results:
            norm_np = results['normals'].squeeze().permute(1, 2, 0).cpu().numpy()
            fixed_norm = fix_panorama_seam(norm_np, margin=2, task_type='normal')
            results['normals'] = torch.from_numpy(fixed_norm).permute(2, 0, 1).unsqueeze(0).to(device)

        return results


def generate_video(outputs_dict, raw_rgb, device, output_path, palette_name='ade20k', blend_alpha=0.7, cmap_name="Spectral", delete_window=False):
    pano_rgb = raw_rgb.to(device)
    pano_sem = outputs_dict.get('semseg').to(device)
    pano_depth_for_proj = outputs_dict.get('raw_depth', outputs_dict.get('depth')).to(device)
    pano_normal = outputs_dict.get('normals').to(device)
    
    PERSP_H, PERSP_W = 768, 768
    FPS = 30
    DURATION = 15  
    TOTAL_FRAMES = FPS * DURATION
    
    full_w = PERSP_W * 4
    full_h = PERSP_H
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (full_w, full_h))
    
    global_max = pano_depth_for_proj.max().item()
    if global_max == 0: global_max = 1.0

    print(f"  Rendering Professional Loop to {output_path}...")
    for i in range(TOTAL_FRAMES):
        progress = i / TOTAL_FRAMES
        
        if progress < 0.05:
            yaw, pitch, h_fov = -180.0, 180.0, 90.0
            current_proj = pano_to_fisheye_stereographic
            
        elif progress < 0.40:
            t = (progress - 0.05) / 0.35
            ease = smooth_step(t)
            yaw = -180.0 - ease * 360.0
            pitch, h_fov = 180.0, 90.0
            current_proj = pano_to_fisheye_stereographic
            
        elif progress < 0.45:
            yaw, pitch, h_fov = -540.0, 180.0, 90.0
            current_proj = pano_to_fisheye_stereographic
            
        elif progress < 0.70:
            t = (progress - 0.45) / 0.25
            ease = smooth_step(t)
            yaw = 180.0 + ease * 360.0
            pitch = 180.0 - ease * 90.0
            h_fov = 90.0 + ease * 130.0
            current_proj = pano_to_fisheye_stereographic
            
        elif progress < 0.75:
            yaw, pitch, h_fov = 540.0, 90.0, 220.0
            current_proj = pano_to_fisheye_stereographic
            
        else:
            t = (progress - 0.75) / 0.25
            ease = smooth_step(t)
            yaw = 540.0 - ease * 360.0
            pitch = 90.0 + ease * 90.0
            h_fov = 220.0 - ease * 130.0
            current_proj = pano_to_fisheye_stereographic

        persp_rgb, persp_sem_label, persp_depth_z, persp_normal = current_proj(
            pano_rgb, pano_sem, pano_depth_for_proj, pano_normal, h_fov, yaw, pitch, PERSP_H, PERSP_W
        )

        # 1. RGB
        img_rgb_vis = (persp_rgb.squeeze().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_rgb_vis = cv2.cvtColor(img_rgb_vis, cv2.COLOR_RGB2BGR)
        
        # 2. Semantic (Blended)
        persp_sem_color = colorize_semantic(persp_sem_label, palette_name=palette_name) 
        img_sem_vis = (persp_sem_color.squeeze().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img_sem_vis = cv2.cvtColor(img_sem_vis, cv2.COLOR_RGB2BGR)
        img_sem_blend = blend_mask_rgb(img_sem_vis, img_rgb_vis, alpha=blend_alpha)
        
        filter_class_ids = []
        if palette_name == 'ade20k':
            filter_class_ids = [2]
            if delete_window:
                filter_class_ids.append(8)
                filter_class_ids.append(68)

        persp_sem_np = persp_sem_label.squeeze().cpu().numpy()
        sky_mask = np.zeros_like(persp_sem_np, dtype=bool)
        if filter_class_ids:
            for cid in filter_class_ids:
                 sky_mask = sky_mask | (persp_sem_np == cid)

        # 3. Depth 
        d_val = persp_depth_z.squeeze().cpu().numpy()
        img_depth = colorize_depth_strict(d_val, global_max, cmap_name)
        img_depth[sky_mask] = [0, 0, 0]
        
        # 4. Normal
        n_np = persp_normal.squeeze().permute(1, 2, 0).cpu().numpy()
        img_norm = ((n_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        img_norm[sky_mask] = [127, 127, 127]
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
        
        grid = np.hstack((img_rgb_vis, img_sem_blend, img_depth, img_norm))
        
        cv2.putText(grid, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(grid, "Semantic", (PERSP_W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(grid, "Depth", (PERSP_W * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(grid, "Normal", (PERSP_W * 3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        video_writer.write(grid)
        
    video_writer.release()

def generate_point_clouds(outputs_dict, raw_rgb, device, output_base, palette_name='ade20k'):
    pano_rgb = raw_rgb.to(device)
    pano_depth = outputs_dict.get('depth').to(device)
    pano_normal = outputs_dict.get('normals').to(device)
    pano_sem = outputs_dict.get('semseg').to(device) 
    
    _, _, H, W = pano_rgb.shape
    
    v, u = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    theta = u * math.pi
    phi = v * math.pi / 2
    s_x = torch.cos(phi) * torch.sin(theta)
    s_y = torch.sin(phi)
    s_z = -torch.cos(phi) * torch.cos(theta)
    rays_world = torch.stack([s_z, s_y, s_x], dim=-1)
    
    points_world = rays_world * pano_depth.squeeze(0).permute(1, 2, 0)
    
    depth_sq = pano_depth.squeeze()
    
    MAX_DIST = 45.0  
    dist_mask = (depth_sq > 0.05) & (depth_sq < MAX_DIST)
    
    BORDER_CROP_RATIO = 0.001 
    border_threshold = 1.0 - (BORDER_CROP_RATIO * 2) 
    seam_mask = u.abs() < border_threshold
    
    valid_mask = dist_mask & seam_mask
    
    points_flat = points_world[valid_mask]
    normals_flat = (-pano_normal).squeeze(0).permute(1, 2, 0)[valid_mask]
    
    colors_rgb = pano_rgb.squeeze(0).permute(1, 2, 0)[valid_mask]
    ensure_save_point_cloud(points_flat, colors_rgb, normals_flat, f"{output_base}_world_rgb.ply")
    
    colors_normal = (normals_flat + 1.0) / 2.0
    colors_normal = torch.clamp(colors_normal, 0.0, 1.0)
    ensure_save_point_cloud(points_flat, colors_normal, normals_flat, f"{output_base}_world_normal.ply")

    if pano_sem is not None:
        sem_indices = pano_sem.long().squeeze()
        sem_flat = sem_indices[valid_mask].cpu().numpy()
        palette_np = np.array(ADE20K_PALETTE, dtype=np.float32) / 255.0
        max_idx = palette_np.shape[0] - 1
        safe_indices = np.clip(sem_flat, 0, max_idx)
        colors_semantic = torch.from_numpy(palette_np[safe_indices]).to(device)
        ensure_save_point_cloud(points_flat, colors_semantic, normals_flat, f"{output_base}_world_semantic.ply")

def vutils_save_image(tensor, path):
    from torchvision.utils import save_image
    save_image(tensor, path)

def main():
    parser = argparse.ArgumentParser(description='Inference, Video Generation and Point Cloud Creation')
    parser.add_argument('--weight', type=str, default='408k', choices=['140k', '408k'], help='Choose which HuggingFace weight to use')
    parser.add_argument('--input', required=True, help='Path to image or folder')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--save_video', action='store_true', help='If set, generate video visualization')
    parser.add_argument('--save_pcd', action='store_true', help='If set, generate point clouds')
    parser.add_argument('--sample_num', type=int, default=0, help='Randomly sample N images from dir. 0 means all.')
    parser.add_argument('--palette', type=str, default='ade20k', choices=['ade20k'], help='Color palette for semantic segmentation: ade20k')
    parser.add_argument('--blend_alpha', type=float, default=0.6, help='Alpha for semantic blending (0.0-1.0)')
    parser.add_argument('--cmap', type=str, default='Spectral', help='Matplotlib colormap for depth')
    parser.add_argument('--delete_window', action='store_true', help='If set, delete window')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    p = get_inference_config()

    print("Loading model...")
    model = get_model(p).to(device)
    
    weight_filename = f"mtpano_model_{args.weight}.pth.tar"
    print(f"Fetching {weight_filename} from Hugging Face (jdzhang0929/MTPano)...")
    
    try:
        checkpoint_path = hf_hub_download(repo_id="jdzhang0929/MTPano", filename=weight_filename)
        print(f"Successfully loaded checkpoint from: {checkpoint_path}")
    except Exception as e:
        print(f"Error downloading weights: {e}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Model loaded (strict).")
    except RuntimeError as e:
        print(f"Strict loading failed, trying non-strict. Error: {e}")
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded (non-strict).")
    
    model.eval()
    
    image_paths = []
    if os.path.isdir(args.input):
        exts = ['*.jpg', '*.png', '*.jpeg', '*.webp', '*.bmp', '*.tif']
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        print(f"Found {len(image_paths)} images total.")
        random.shuffle(image_paths)  
        
        if args.sample_num > 0 and args.sample_num < len(image_paths):
            image_paths = image_paths[:args.sample_num] 
            print(f"Randomly sampled {len(image_paths)} images for processing.")
    else:
        image_paths = [args.input]
        
    if not image_paths:
        print("No images found.")
        return
        
    os.makedirs(args.output, exist_ok=True)
    
    TARGET_H, TARGET_W = p.TRAIN.SCALE
    print(f"Target Input Size: {TARGET_H}x{TARGET_W}")

    for img_path in tqdm(image_paths, desc="Processing Images"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        curr_out_dir = os.path.join(args.output, base_name)
        os.makedirs(curr_out_dir, exist_ok=True)
        
        pil_img = Image.open(img_path).convert('RGB').resize((TARGET_W, TARGET_H), Image.BILINEAR)
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        
        raw_rgb_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() 
        
        vutils_save_image(raw_rgb_tensor, os.path.join(curr_out_dir, f"{base_name}_rgb_resized.png"))
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        input_tensor = (raw_rgb_tensor - mean) / std
        
        results = run_inference(model, input_tensor, device, palette_name=args.palette, delete_window=args.delete_window)
        
        if 'semseg' in results:
            sem_color_tensor = colorize_semantic(results['semseg'], palette_name=args.palette)
            vutils_save_image(sem_color_tensor, os.path.join(curr_out_dir, f"{base_name}_semseg.png"))
            
            rgb_vis_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
            rgb_vis_bgr = (rgb_vis_bgr * 255).astype(np.uint8)
            
            sem_vis_bgr = (sem_color_tensor.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            sem_vis_bgr = cv2.cvtColor(sem_vis_bgr, cv2.COLOR_RGB2BGR)
            
            overlay = blend_mask_rgb(sem_vis_bgr, rgb_vis_bgr, alpha=args.blend_alpha)
            cv2.imwrite(os.path.join(curr_out_dir, f"{base_name}_overlay.png"), overlay)
            
        if 'depth' in results:
            depth_np = results['depth'].squeeze().cpu().numpy()
            np.save(os.path.join(curr_out_dir, f"{base_name}_depth.npy"), depth_np)
            save_colored_depth(results['depth'], os.path.join(curr_out_dir, f"{base_name}_depth_vis.png"), cmap_name=args.cmap)
            
        if 'normals' in results:
            norm_vis = results['normals'].cpu() * 0.5 + 0.5
            vutils_save_image(norm_vis, os.path.join(curr_out_dir, f"{base_name}_normal.png"))
            
        if args.save_video:
            vid_path = os.path.join(curr_out_dir, f"{base_name}_demo.mp4")
            generate_video(results, raw_rgb_tensor, device, vid_path, 
                           palette_name=args.palette, blend_alpha=args.blend_alpha, cmap_name=args.cmap, delete_window=args.delete_window)

        if args.save_pcd:
            generate_point_clouds(results, raw_rgb_tensor, device, os.path.join(curr_out_dir, base_name), palette_name=args.palette)

if __name__ == "__main__":
    main()