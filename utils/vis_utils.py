import torch
import numpy as np
from PIL import Image
import wandb
import os
import math
from sklearn.decomposition import PCA
import torch.nn.functional as F

# --- Constants for RGB Denormalization ---
# NOTE: These values are derived from the problem description
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Semantic Segmentation Palette (40 classes, index 0-39)
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
num_colors_used = len(NYU_PALETTE_RGB)
num_colors_needed_to_fill = 255 - num_colors_used
NYUDV2_PALETTE_FLAT.extend([0, 0, 0] * num_colors_needed_to_fill) 
NYUDV2_PALETTE_FLAT.extend([0, 0, 0]) 
assert len(NYUDV2_PALETTE_FLAT) == 256 * 3, f"Palette size is incorrect: {len(NYUDV2_PALETTE_FLAT)}/768"


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

ADE20K_PALETTE_FLAT = [c for color in ADE20K_PALETTE for c in color]
ADE20K_PALETTE_FLAT.extend([0, 0, 0] * (256 - len(ADE20K_PALETTE)))


def tensor_to_pil(tensor, is_normalized_rgb=False):
    """(B, C, H, W) -> PIL Image (or (B, 1, H, W) -> numpy array for index maps)"""
    
    if tensor.dim() == 4:
        tensor = tensor[0] 
    
    tensor = tensor.cpu().detach()
    img_np = tensor.numpy() # Shape is now (C, H, W) or (1, H, W)
    
    if is_normalized_rgb:
        
        # --- FIX: RGB Denormalization Logic ---
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            # Transpose to (H, W, C) for Denorm operation
            img_np = img_np.transpose(1, 2, 0)
            
            # 1. Denormalize: I' = I_vis * std + mean
            img_np = img_np * IMG_STD + IMG_MEAN
            
            # 2. Rescale/Clip to [0, 255]
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_np, 'RGB')
        # --- END FIX ---
        
        # Handle 1-channel grayscale conversion (e.g. normalized depth maps)
        elif img_np.ndim == 3 and img_np.shape[0] == 1:
            # Transpose to (H, W, 1)
            img_np = img_np.transpose(1, 2, 0) 
            
            # Scale to 0-255
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            img_np = np.squeeze(img_np, axis=-1)
            return Image.fromarray(img_np, 'L').convert('RGB')
             
    # Default path for other scenarios (should be less relevant with fixed logic above)
    if img_np.ndim == 3 and img_np.shape[0] in [3, 1]:
        img_np = img_np.transpose(1, 2, 0) # (H, W, C)

    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = np.squeeze(img_np, axis=-1)
        return Image.fromarray(img_np, 'L').convert('RGB')
        
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    if img_np.ndim != 3 or img_np.shape[-1] != 3:
         raise ValueError(f"Final shape {img_np.shape} is not valid for Image.fromarray('RGB') in vis_utils.py. Expected (H, W, 3).")
         
    return Image.fromarray(img_np, 'RGB')


def save_semantic_map_for_vis(tensor, palette=NYUDV2_PALETTE_FLAT):
    """Converts (B, C, H, W) one-hot or (B, 1, H, W) index map to a colorized PIL image."""
    
    if tensor.dim() == 4:
        tensor = tensor[0] 
    
    if tensor.dim() == 3 and tensor.shape[0] > 1: 
        indices = torch.argmax(tensor, dim=0).cpu().numpy().astype(np.uint8)
    elif tensor.dim() in [2, 3] and (tensor.dim() == 2 or tensor.shape[-3] == 1):
        indices = tensor.squeeze().cpu().numpy().astype(np.uint8)
    else:
        raise ValueError("Semantic tensor shape is ambiguous for visualization.")

    indices = np.clip(indices, 0, 255).astype(np.uint8) 
        
    img = Image.fromarray(indices, 'P')
    img.putpalette(palette) # 使用了将 ignore 索引 255 设为黑色的新调色板
    return img.convert('RGB')


def create_vis_image(tag, pred, gt, input_img, max_depth=10.0, palette=NYUDV2_PALETTE_FLAT):
    """Create a single visualization image set for a task (Input/Pred/GT)."""
    log_dict = {}
    
    tag_parts = tag.split('_')
    if len(tag_parts) < 2:
        return {}
        
    task = tag_parts[-2].lower() 

    if pred.dim() == 4: pred = pred[0]
    if gt.dim() == 4: gt = gt[0]
    
    # ------------------ Input RGB ------------------
    if input_img is not None:
        if input_img.dim() == 4: input_img = input_img[0]
        
        input_pil = tensor_to_pil(input_img.unsqueeze(0), is_normalized_rgb=True)
        input_wandb = wandb.Image(input_pil, caption=f"{tag}_Input_RGB")
        log_dict[f"{tag}/Input_RGB"] = input_wandb
        
    # ------------------ Predictions & GT ------------------
    if task == 'semseg':
        pred_pil = save_semantic_map_for_vis(pred, palette=palette)
        gt_pil = save_semantic_map_for_vis(gt, palette=palette)
    elif task == 'depth':
        # Depth is in meters. Clamp and normalize by max_depth
        
        pred_vis = pred.clone().cpu()
        gt_vis = gt.clone().cpu()
        
        pred_vis_2d = pred_vis.squeeze(0) # (H, W)
        gt_vis_2d = gt_vis.squeeze(0)     # (H, W)
        
        # Mask calculation is still needed for debugging/analysis, but not for coloring
        pred_mask = (pred_vis_2d == 255).numpy() 
        gt_mask = (gt_vis_2d == 255).numpy()
        
        # --- FIX: Ensure Ignore value 255 is set to 0 (Black) for both Pred and GT ---
        pred_vis_2d[pred_vis_2d == 255] = 0
        gt_vis_2d[gt_vis_2d == 255] = 0
        # --- END FIX ---
        
        # Normalize: (H, W)
        pred_norm = torch.clamp(pred_vis_2d, 0, max_depth) / max_depth
        gt_norm = torch.clamp(gt_vis_2d, 0, max_depth) / max_depth
        
        # Add channel dimension back and expand to 3 channels: (3, H, W)
        pred_norm_3ch = pred_norm.unsqueeze(0).expand(3, -1, -1)
        gt_norm_3ch = gt_norm.unsqueeze(0).expand(3, -1, -1)
        
        pred_pil = tensor_to_pil(pred_norm_3ch.unsqueeze(0))
        gt_pil = tensor_to_pil(gt_norm_3ch.unsqueeze(0))
        
        # --- FIX: Removed the code that recolored the ignore mask to pink ---
        # Pred and GT ignore areas will naturally be black (0) after normalization/rescaling.
        
    elif task == 'normals':
        # Normals are in [-1, 1]. Map to [0, 1] for visualization
        
        pred_vis = pred.clone().cpu() # (3, H, W)
        gt_vis = gt.clone().cpu()     # (3, H, W)
        
        # Mask calculation
        pred_mask = (pred_vis == 255).all(dim=0).squeeze().numpy() # (H, W)
        gt_mask = (gt_vis == 255).all(dim=0).squeeze().numpy()     # (H, W)
        
        # Set ignore values to 0 for visualization
        pred_vis_masked = pred_vis.clone()
        gt_vis_masked = gt_vis.clone()

        # --- FIX: Ensure Ignore value 255 is set to 0 (Black) for both Pred and GT ---
        pred_vis_masked[pred_vis == 255] = 0
        gt_vis_masked[gt_vis == 255] = 0
        # --- END FIX ---

        norm_pred = torch.linalg.norm(pred_vis_masked, dim=0, keepdim=True)
        zero_mask_pred = (norm_pred == 0)
        norm_pred[zero_mask_pred] = 1
        pred_vis_masked = pred_vis_masked.div(norm_pred)
        pred_vis_masked[zero_mask_pred.expand_as(pred_vis_masked)] = 0
        
        # Normals: [-1, 1] -> [0, 1]. (3, H, W)
        pred_norm = (pred_vis_masked * 0.5) + 0.5
        gt_norm = (gt_vis_masked * 0.5) + 0.5
        
        pred_pil = tensor_to_pil(pred_norm.unsqueeze(0))
        gt_pil = tensor_to_pil(gt_norm.unsqueeze(0))
        
        # --- FIX: Removed the code that recolored the ignore mask to pink ---
        
    else:
        return {} # Unsupported task

    log_dict[f"{tag}/Prediction"] = wandb.Image(pred_pil, caption=f"{tag}_Prediction")
    log_dict[f"{tag}/Ground_Truth"] = wandb.Image(gt_pil, caption=f"{tag}_Ground_Truth")
    
    return log_dict


def visualize_results(p, batch, output, pred_pano, target_pano_masked, persp_rgb, targets_persp_synth, sample_idx, step_no):
    # 此函数与之前的逻辑保持一致，只是使用了包含 RGB Denormalization 和 
    # 统一黑色 Ignore 区域的新 create_vis_image 函数。
    log_dict = {}
    tasks = p.TASKS.NAMES 

    if 'PanoMTDU' in p.get('train_db_name', ''):
        current_palette = ADE20K_PALETTE_FLAT
    else:
        current_palette = NYUDV2_PALETTE_FLAT
    
    # Calculate the global max depth for normalization
    if 'depth' in tasks:
        pano_orig_depth_gt = batch['pano']['depth'][0].detach().clone() 
        valid_depth = pano_orig_depth_gt[pano_orig_depth_gt != 255]
        max_depth = valid_depth.max().item() if valid_depth.numel() > 0 else 10.0
    else:
        max_depth = 10.0
    
    # --- 1. PANO ORIGINAL ---
    pano_orig = batch['pano']
    
    # RGB
    log_dict[f"Pano_Original/RGB_{sample_idx}"] = wandb.Image(tensor_to_pil(pano_orig['image']))
    
    # Semantic
    if 'semseg' in tasks:
        log_dict[f"Pano_Original/Semantic_{sample_idx}"] = wandb.Image(save_semantic_map_for_vis(pano_orig['semseg'], palette=current_palette))
    
    # Original Depth Visualization (Pred=GT)
    if 'depth' in tasks:
        depth_vis_dict = create_vis_image(
            f"Pano_Original_Depth_{sample_idx}", 
            pano_orig['depth'].clone(), # Pred=GT for original
            pano_orig['depth'].clone(), 
            pano_orig['image'],
            max_depth
        )
        gt_key = f"Pano_Original_Depth_{sample_idx}/Ground_Truth"
        if gt_key in depth_vis_dict:
            log_dict[f"Pano_Original/Depth_{sample_idx}"] = depth_vis_dict[gt_key]
    
    # Original Normals Visualization (Pred=GT)
    if 'normals' in tasks:
        normals_vis_dict = create_vis_image(
            f"Pano_Original_Normals_{sample_idx}", 
            pano_orig['normals'].clone(), # Pred=GT for original
            pano_orig['normals'].clone(), 
            pano_orig['image']
        )
        gt_key = f"Pano_Original_Normals_{sample_idx}/Ground_Truth"
        if gt_key in normals_vis_dict:
            log_dict[f"Pano_Original/Normals_{sample_idx}"] = normals_vis_dict[gt_key]

    # --- 2. PERSPECTIVE (Pred vs GT) ---
    for task in tasks:
        if task in output and task in targets_persp_synth: 
            persp_pred = output[task]
            persp_gt = targets_persp_synth[task]
            
            vis_images = create_vis_image(
                f"Perspective_Warped_{task}_{sample_idx}", 
                persp_pred, 
                persp_gt, 
                persp_rgb,
                max_depth,
                palette=current_palette
            )
            if vis_images:
                log_dict.update(vis_images)

    # --- 3. WRAP BACK (Warped Pred vs Warped GT) ---
    for task in tasks:
        if task in pred_pano and task in target_pano_masked:
            wrap_pred = pred_pano[task]
            wrap_gt = target_pano_masked[task]

            vis_images = create_vis_image(
                f"Pano_Warped_Back_{task}_{sample_idx}", 
                wrap_pred, 
                wrap_gt, 
                pano_orig['image'], # Use original pano RGB as background reference
                max_depth,
                palette=current_palette
            )
            if vis_images:
                log_dict.update(vis_images)

    wandb.log(log_dict, step=step_no)


# --- 新增 helper: 计算 PCA RGB ---
def compute_pca_vis(feature_tensor, mask=None, target_shape=None):
    """
    对 Feature Map 进行 PCA 降维并生成 RGB 可视化图 (Min-Max Norm)。
    feature_tensor: (C, H, W) - 单张图的特征
    mask: (1, H, W) - 可选掩码
    target_shape: (H_out, W_out) - 可选输出尺寸
    Return: (3, H, W) Tensor, range [0, 1]
    """
    feat = feature_tensor.detach().cpu()
    C, h, w = feat.shape
    
    # 1. Flatten & Centering: (C, h*w) -> (h*w, C)
    feat_flat = feat.view(C, -1).permute(1, 0).numpy()
    feat_flat = feat_flat - feat_flat.mean(axis=0)
    
    # 2. PCA 降维: (h*w, C) -> (h*w, 3)
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat_flat)
    
    # 3. Min-Max Normalization (DINO 标准做法)
    feat_min = feat_pca.min(axis=0)
    feat_max = feat_pca.max(axis=0)
    feat_pca = (feat_pca - feat_min) / (feat_max - feat_min + 1e-6)
    
    # 4. Reshape 回图片: (h*w, 3) -> (3, h, w)
    feat_rgb = torch.from_numpy(feat_pca).float().permute(1, 0).view(3, h, w)
    
    # 5. 上采样 (如果 Feature 分辨率低，这里上采样方便观看)
    if target_shape is not None:
        feat_rgb = F.interpolate(feat_rgb.unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze(0)
        
    # 6. Apply Mask (如果有)
    if mask is not None:
        mask_cpu = mask.detach().cpu()
        if mask_cpu.dim() == 3: mask_cpu = mask_cpu[0] # (1, H, W) -> (H, W)
        if target_shape is not None:
            mask_cpu = F.interpolate(mask_cpu.view(1, 1, *mask_cpu.shape), size=target_shape, mode='nearest').squeeze()
        feat_rgb = feat_rgb * mask_cpu
        
    return torch.clamp(feat_rgb, 0, 1)

# --- 新增: 特征对齐可视化函数 ---
def visualize_feature_alignment(p, batch, pano_features, persp_features_warped, persp_features_fresh, 
                                persp_rgb, persp_masks, sample_idx, step_no):
    """
    可视化 Feature Alignment 效果。
    
    pano_features: List of (B, C, H_pano, W_pano)
    persp_features_warped: List of (B, C, H_p, W_p) (从 Pano 投影过来的)
    persp_features_fresh: List of (B, C, H_p, W_p) (直接在 Crop 上提取的)
    persp_rgb: (B, 3, H_p, W_p)
    persp_masks: List of (B, 1, H_p, W_p)
    """
    log_dict = {}
    
    # 取 Batch 中的第一个样本
    pano_img_tensor = batch['pano']['image'][0] # (3, H, W)
    persp_img_tensor = persp_rgb[0]             # (3, h, w)
    
    # 1. 记录 RGB 图像
    log_dict[f"FeatAlign/1_Pano_RGB_{sample_idx}"] = wandb.Image(tensor_to_pil(pano_img_tensor.unsqueeze(0)))
    log_dict[f"FeatAlign/2_Persp_Input_RGB_{sample_idx}"] = wandb.Image(tensor_to_pil(persp_img_tensor.unsqueeze(0)))
    
    # 2. 遍历每个 Feature Level 进行可视化
    # 假设 features 是 list，通常有 4 层
    num_layers = len(pano_features)
    
    for i in range(num_layers):
        # 2.1 Pano Feature PCA
        pano_feat_vis = compute_pca_vis(
            pano_features[i][0], 
            target_shape=pano_img_tensor.shape[-2:] # Resize 到原图大小
        )
        log_dict[f"FeatAlign/L{i}_Pano_Feat_{sample_idx}"] = wandb.Image(tensor_to_pil(pano_feat_vis.unsqueeze(0)))
        
        # 2.2 Compare: Warped vs Fresh
        # Resize 到 Perspective RGB 大小以便比较
        target_size = persp_img_tensor.shape[-2:]
        mask_vis = persp_masks[i][0] # (1, h, w)
        
        # (a) Warped Feature (来自全景图投影)
        warped_vis = compute_pca_vis(
            persp_features_warped[i][0], 
            mask=mask_vis,
            target_shape=target_size
        )
        
        # (b) Fresh Feature (来自透视图直接推理)
        fresh_vis = compute_pca_vis(
            persp_features_fresh[i][0], 
            mask=None, # Fresh feature 通常是全图有效的
            target_shape=target_size
        )
        
        # (c) 拼接: Left=Warped(Pseudo), Right=Fresh(Real)
        concat_vis = torch.cat([warped_vis, fresh_vis], dim=2) # 在宽度方向拼接
        
        log_dict[f"FeatAlign/L{i}_Compare_Warped_vs_Fresh_{sample_idx}"] = wandb.Image(
            tensor_to_pil(concat_vis.unsqueeze(0)), 
            caption="Left: Warped from Pano, Right: Fresh Inference"
        )
        
    wandb.log(log_dict, step=step_no)