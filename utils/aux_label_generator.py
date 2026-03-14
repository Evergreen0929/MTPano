import torch
import torch.nn.functional as F
import math

import numpy as np
import cv2
import os
from PIL import Image

class AuxLabelGenerator:
    def __init__(self, device='cuda', denorm_mean=[0.485, 0.456, 0.406], denorm_std=[0.229, 0.224, 0.225]):
        """
        辅助标签生成器 (Panorama Optimized Version)
        1. 左右边界：Circular Padding，保证梯度连续性。
        2. 上下边界：Replicate Padding + 主动清除梯度，防止极点伪影。
        """
        self.device = device
        
        # 反归一化参数
        self.mean = torch.tensor(denorm_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(denorm_std, device=device).view(1, 3, 1, 1)
        self.weights_gray = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)

        # Sobel 核
        self.sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)

    def _pad_panorama(self, img_tensor, pad=1):
        """
        全景图专用 Padding：
        - 宽度 (W): Circular (循环)，因为左右相连。
        - 高度 (H): Replicate (复制)，避免上下边缘产生虚假强梯度。
        """
        # 1. 左右循环填充 (Pad W)
        # F.pad format: (left, right, top, bottom)
        # 仅对最后两个维度操作，这里先 pad 左右
        img_padded = F.pad(img_tensor, (pad, pad, 0, 0), mode='circular')
        
        # 2. 上下复制填充 (Pad H)
        # 接着对高度 pad，使用 replicate 减少极点伪影
        img_padded = F.pad(img_padded, (0, 0, pad, pad), mode='replicate')
        
        return img_padded

    def _tensor_sobel(self, img_tensor):
        """
        计算梯度，使用 Valid Padding (因为输入已经手动 Pad 过了)
        Output: (B, 1, H, W)
        """
        # 输入 img_tensor 应该是 (B, 1, H+2, W+2)
        gx = F.conv2d(img_tensor, self.sobel_x, padding=0)
        gy = F.conv2d(img_tensor, self.sobel_y, padding=0)
        return gx, gy

    @torch.no_grad()
    def generate_gradient_map(self, images):
        """
        Task: Image Gradient (Panorama Corrected)
        """
        # 1. 反归一化
        img = images * self.std + self.mean
        
        # 2. 转灰度
        img_gray = F.conv2d(img, self.weights_gray)
        
        # 3. 全景 Padding (W循环, H复制)
        img_gray_pad = self._pad_panorama(img_gray, pad=1)
        
        # 4. 计算梯度 (结果已经是 HxW)
        gx, gy = self._tensor_sobel(img_gray_pad)
        
        # 5. 可选：清除上下极点处的梯度 (通常极点拉伸严重，梯度不可靠)
        # 这里只清除最上面和最下面 1 像素
        gx[..., 0, :] = 0; gx[..., -1, :] = 0
        gy[..., 0, :] = 0; gy[..., -1, :] = 0
        
        return torch.cat([gx, gy], dim=1)

    @torch.no_grad()
    def generate_sdf_map(self, images, edge_thresh=0.5, border_clear=8):
        """
        Task: SDF (Panorama Corrected)
        """
        # 1. 预处理 + Padding
        img = images * self.std + self.mean
        img_gray = F.conv2d(img, self.weights_gray)
        img_gray_pad = self._pad_panorama(img_gray, pad=1)
        
        # 2. 梯度计算
        gx, gy = self._tensor_sobel(img_gray_pad)
        grad_mag = torch.sqrt(gx**2 + gy**2)
        
        # 3. 生成 Mask
        edge_mask = (grad_mag > edge_thresh).float()
        
        # 4. 【关键】Border Clearing (仅清除 Top/Bottom)
        # 全景图左右是连通的，所以左右边界是有效区域，不能清除！
        # 只清除上下两极区域
        if border_clear > 0:
            edge_mask[..., :border_clear, :] = 0  # Top
            edge_mask[..., -border_clear:, :] = 0 # Bottom
            # Left/Right 不做处理！保持连通性
            
        # 5. JFA Distance Transform
        # 0=Target(Edge), 1=Background
        jfa_input = 1.0 - edge_mask
        sdf = self._distance_transform_jfa(jfa_input)
        sdf = torch.log(sdf + 1.0)
        
        return sdf

    def _distance_transform_jfa(self, mask):
        """
        Standard JFA (Tensor Implementation)
        """
        B, C, H, W = mask.shape
        
        rows = torch.arange(H, device=self.device, dtype=torch.float32)
        cols = torch.arange(W, device=self.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        init_val = torch.tensor(float('inf'), device=self.device)
        seeds = torch.where(mask.permute(0, 2, 3, 1) < 0.5, grid, init_val)
        
        step = 1
        while step < max(H, W): step <<= 1
        step >>= 1
        
        # Pre-compute shifts
        shifts = torch.tensor([[-1,-1], [0,-1], [1,-1], [-1,0], [0,0], [1,0], [-1,1], [0,1], [1,1]], device=self.device) * step

        while step >= 1:
            candidates = []
            for i in range(9):
                dx, dy = int(shifts[i, 0] // step), int(shifts[i, 1] // step)
                shift_x, shift_y = dx * step, dy * step
                
                # Slicing with INF padding
                shifted = torch.full_like(seeds, float('inf'))
                src_y_start, src_y_end = max(0, shift_y), min(H, H + shift_y)
                dst_y_start, dst_y_end = max(0, -shift_y), min(H, H - shift_y)
                src_x_start, src_x_end = max(0, shift_x), min(W, W + shift_x)
                dst_x_start, dst_x_end = max(0, -shift_x), min(W, W - shift_x)
                
                if src_y_end > src_y_start and src_x_end > src_x_start:
                    shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
                        seeds[:, src_y_start:src_y_end, src_x_start:src_x_end, :]
                candidates.append(shifted)

            candidates = torch.stack(candidates)
            diff = candidates - grid.unsqueeze(0)
            dist_sq = diff[..., 0]**2 + diff[..., 1]**2
            
            # Efficient Update without Gather
            best_seeds = seeds
            curr_min = dist_sq[4]
            for i in range(9):
                mask_better = dist_sq[i] < curr_min
                curr_min = torch.where(mask_better, dist_sq[i], curr_min)
                best_seeds = torch.where(mask_better.unsqueeze(-1), candidates[i], best_seeds)
            seeds = best_seeds
            
            step >>= 1
            shifts //= 2

        diff = seeds - grid
        dist_map = torch.sqrt(diff[..., 0]**2 + diff[..., 1]**2)
        return dist_map.unsqueeze(1)

    @torch.no_grad()
    def generate_point_map(self, depth):
        """Task: Metric Point Map"""
        B, _, H, W = depth.shape
        depth = depth.clone().detach() 
        invalid_mask = depth == 255
        depth[invalid_mask] = 0
        rows = torch.arange(H, device=self.device)
        cols = torch.arange(W, device=self.device)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
        
        u = 2.0 * (grid_x / (W - 1)) - 1.0
        v = 2.0 * (grid_y / (H - 1)) - 1.0
        
        theta = u * math.pi
        phi = v * (math.pi / 2)
        
        x = torch.cos(phi) * torch.sin(theta)
        y = torch.sin(phi)
        z = -torch.cos(phi) * torch.cos(theta)
        
        ray_dir = torch.stack([x, y, z], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        point_map = ray_dir * depth
        
        point_map[invalid_mask.expand_as(point_map)] = 255
        
        return point_map


class SimplePipeline:
    def __init__(self, device):
        self.device = device
        # ImageNet Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def load_and_preprocess(self, img_path, depth_path):
        # 1. Load RGB
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        pil_img = Image.open(img_path).convert('RGB')
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0 # [0, 1]
        
        # 2. Load Depth (16-bit mm)
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth not found: {depth_path}")
        pil_depth = Image.open(depth_path)
        depth_mm = np.array(pil_depth, dtype=np.uint16)
        depth_m = depth_mm.astype(np.float32) / 1000.0
        depth_arr = np.expand_dims(depth_m, axis=2) # H, W, 1
        
        # 3. To Tensor & Device
        img_tensor = torch.from_numpy(img_arr.transpose((2, 0, 1))).unsqueeze(0).to(self.device)
        depth_tensor = torch.from_numpy(depth_arr.transpose((2, 0, 1))).unsqueeze(0).to(self.device)
        
        # 4. Normalize RGB
        img_tensor = (img_tensor - self.mean) / self.std
        
        return img_tensor, depth_tensor


# ==========================================
# 3. 主函数 (测试入口)
# ==========================================
if __name__ == "__main__":
    # 配置
    IMG_PATH = "/mnt/localssd/code/PanoMTL/preprocess_dataset/rgb_rawlight.png"   # 请确保该文件存在
    DEPTH_PATH = "/mnt/localssd/code/PanoMTL/preprocess_dataset/depth.png"        # 请确保该文件存在
    OUTPUT_DIR = "/mnt/localssd/code/PanoMTL/preprocess_dataset/output_test_result"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running test on {DEVICE}...")

    try:
        # 1. 初始化
        pipeline = SimplePipeline(DEVICE)
        aux_gen = AuxLabelGenerator(device=DEVICE)
        
        # 2. 加载数据
        print(f"Loading {IMG_PATH} and {DEPTH_PATH}...")
        img_tensor, depth_tensor = pipeline.load_and_preprocess(IMG_PATH, DEPTH_PATH)
        print(f"Input Image: {img_tensor.shape}, Depth: {depth_tensor.shape}")

        # 3. 运行生成 (模拟训练 Loop)
        print("Generating Auxiliary Labels...")
        with torch.no_grad():
            # Task 1: Gradient
            gt_gradients = aux_gen.generate_gradient_map(img_tensor) # (B, 2, H, W)
            print(gt_gradients.max(), gt_gradients.min())
            
            # Task 2: SDF
            gt_sdf = aux_gen.generate_sdf_map(img_tensor, border_clear=8) # (B, 1, H, W)
            print(gt_sdf.max(), gt_sdf.min())
            
            # Task 3: Point Map
            gt_point_map = aux_gen.generate_point_map(depth_tensor) # (B, 3, H, W)
            print(gt_point_map.max(), gt_point_map.min())

        # 4. 保存结果用于检查
        print(f"Saving results to {OUTPUT_DIR}...")
        
        # --- Save Gradient (Visualize Magnitude) ---
        gx, gy = gt_gradients[:, 0], gt_gradients[:, 1]
        mag = torch.sqrt(gx**2 + gy**2)
        mag_norm = mag / (mag.max() + 1e-6)
        mag_vis = (mag_norm[0].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'vis_gradient_mag.png'), cv2.applyColorMap(mag_vis, cv2.COLORMAP_JET))
        
        # --- Save SDF (Visualize) ---
        sdf_norm = gt_sdf / (gt_sdf.max() + 1e-6)
        sdf_vis = (sdf_norm[0, 0].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'vis_sdf.png'), cv2.applyColorMap(sdf_vis, cv2.COLORMAP_JET))
        
        # --- Save Point Map (8-bit Visual) ---
        pm_min, pm_max = gt_point_map.min(), gt_point_map.max()
        pm_norm = (gt_point_map - pm_min) / (pm_max - pm_min + 1e-6)
        pm_vis = (pm_norm[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # RGB -> BGR for OpenCV
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'vis_point_map.png'), cv2.cvtColor(pm_vis, cv2.COLOR_RGB2BGR))
        
        print("\nTest Complete! Check the output directory.")
        print(f"Gradient shape: {gt_gradients.shape}")
        print(f"SDF shape: {gt_sdf.shape}")
        print(f"PointMap shape: {gt_point_map.shape}")

    except Exception as e:
        print(f"\nError Occurred: {e}")
        print("提示: 请检查当前目录下是否存在 rgb_rawlight.png 和 depth.png (16-bit) 文件。")