# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with ignore regions.
    """
    def __init__(self, ignore_index=255, class_weight=None, balanced=False):
        super().__init__()
        self.ignore_index = ignore_index
        if balanced:
            assert class_weight is None
        self.balanced = balanced
        if class_weight is not None:
            self.register_buffer('class_weight', class_weight)
        else:
            self.class_weight = None

    def forward(self, out, label, reduction='mean'):
        label = torch.squeeze(label, dim=1).long()
        if self.balanced:
            mask = (label != self.ignore_index)
            masked_label = torch.masked_select(label, mask)
            assert torch.max(masked_label) < 2  # binary
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w_pos = num_labels_neg / num_total
            class_weight = torch.stack((1. - w_pos, w_pos), dim=0)
            loss = nn.functional.cross_entropy(
                out, label, weight=class_weight, ignore_index=self.ignore_index, reduction='none')
        else:
            loss = nn.functional.cross_entropy(out,
                                               label,
                                               weight=self.class_weight,
                                               ignore_index=self.ignore_index,
                                               reduction='none')
        if reduction == 'mean':
            n_valid = (label != self.ignore_index).sum()
            return (loss.sum() / max(n_valid, 1)).float()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss

class DiceCELoss(nn.Module):
    """
    结合了 CrossEntropyLoss 和多类别 DiceLoss 的损失函数。
    
    这个类完全复制了您提供的 CrossEntropyLoss 的 __init__ 和 forward 接口，
    并在此基础上添加了 Dice Loss。

    - 'balanced' 标志按原始类的定义工作（仅用于二元CE）。
    - 'class_weight' 同时应用于 CE 和 Dice 损失。
    - 'reduction' 逻辑被保留，并以兼容的方式应用于两个损失分量。
    """
    def __init__(self, ignore_index=255, class_weight=None, balanced=False, 
                 ce_weight=1.0, dice_weight=1.0, dice_smooth=1e-6):
        """
        Args:
            ignore_index (int): 指定一个被忽略的目标值。
            class_weight (torch.Tensor, optional): 为每个类手动指定缩放权重。
            balanced (bool): 如果为True，则为二元CE损失计算动态权重。
            ce_weight (float): CrossEntropyLoss 分量的权重。默认为 1.0。
            dice_weight (float): DiceLoss 分量的权重。默认为 1.0（最常见的权重）。
            dice_smooth (float): Dice loss 的平滑因子，避免除以零。
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        
        # 存储 balanced 和 class_weight 以供两个损失计算使用
        if balanced:
            assert class_weight is None, "balanced=True 和 class_weight 不能同时设置"
        self.balanced = balanced
        
        if class_weight is not None:
            self.register_buffer('class_weight', class_weight)
        else:
            self.class_weight = None

    def _dice_loss(self, out, label_squeezed, num_classes):
        """
        计算多类别 Dice loss。
        返回原始的 per-batch, per-class dice loss: [B, C]
        """
        # 1. 获取概率
        pred_probs = F.softmax(out, dim=1)
        
        # 2. 创建有效像素掩码 (不被忽略的像素)
        # Shape: [B, H, W]
        valid_mask = (label_squeezed != self.ignore_index)
        
        # 3. 创建 one-hot 目标，同时处理 ignore_index
        # 将被忽略的像素安全地设置为 0（或其他任意有效类别）
        target_safe = label_squeezed.clone()
        target_safe[~valid_mask] = 0 
        target_one_hot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2)
        
        # 4. 将掩码应用到
        # 扩展掩码以便广播: [B, 1, H, W]
        valid_mask_expanded = valid_mask.unsqueeze(1)
        
        probs_masked = pred_probs * valid_mask_expanded
        target_one_hot_masked = target_one_hot * valid_mask_expanded
        
        # 5. 计算交集和基数 (per-class, per-batch)
        # 在空间维度 (H, W) 上求和
        intersection = (probs_masked * target_one_hot_masked).sum(dim=(2, 3)) # Shape: [B, C]
        cardinality = (probs_masked.sum(dim=(2, 3)) + target_one_hot_masked.sum(dim=(2, 3))) # Shape: [B, C]
        
        # 6. 计算 dice score 和 dice loss
        dice_score = (2. * intersection + self.dice_smooth) / (cardinality + self.dice_smooth) # Shape: [B, C]
        loss_dice_raw = 1.0 - dice_score # Shape: [B, C]
        
        # 7. 应用类别权重
        if self.class_weight is not None:
            weights = self.class_weight.to(loss_dice_raw.device)
            loss_dice_raw = loss_dice_raw * weights.view(1, -1) # 广播 [1, C] to [B, C]
            
        return loss_dice_raw # 返回 per-batch, per-class 损失

    def _ce_loss(self, out, label_squeezed):
        """
        计算 Cross Entropy loss。
        返回原始的、逐像素的损失: [B, H, W]
        """
        ce_class_weight = self.class_weight
        if self.balanced:
            # 动态计算二元CE的权重，逻辑与原类一致
            mask = (label_squeezed != self.ignore_index)
            masked_label = torch.masked_select(label_squeezed, mask)
            assert torch.max(masked_label) < 2, "balanced=True 仅适用于二元分割"
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w_pos = num_labels_neg / num_total
            # 确保权重在正确的设备上
            ce_class_weight = torch.stack((1. - w_pos, w_pos), dim=0).to(out.device)
        
        loss_ce_raw = F.cross_entropy(
            out,
            label_squeezed,
            weight=ce_class_weight,
            ignore_index=self.ignore_index,
            reduction='none' # 获取逐像素损失
        )
        return loss_ce_raw

    def forward(self, out, label, reduction='mean'):
        """
        Args:
            out (torch.Tensor): 模型的 Logits 输出。 Shape: [B, C, H, W]
            label (torch.Tensor): 真实标签。 Shape: [B, 1, H, W] or [B, H, W]
            reduction (str): 'mean', 'sum', or 'none'.
                             'mean': 遵循原始类的逻辑，返回 (weighted CE + weighted Dice) 
                                     在所有有效像素上的均值。
                             'sum': 返回 (weighted CE + weighted Dice) 的总和。
                             'none': 返回逐像素的 CE 损失 + 广播后的逐图像 Dice 损失。
                                     Shape: [B, H, W]。请谨慎使用。
        """
        num_classes = out.shape[1]
        label_squeezed = torch.squeeze(label, dim=1).long()
        
        # --- 1. 计算 CE Loss ---
        loss_ce_raw = self._ce_loss(out, label_squeezed) # Shape: [B, H, W]
        
        # --- 2. 计算 Dice Loss ---
        loss_dice_raw = self._dice_loss(out, label_squeezed, num_classes) # Shape: [B, C]

        # --- 3. 结合与规约 (Reduction) ---
        
        if reduction == 'mean':
            # CE 部分：遵循原始类的 'mean' 逻辑
            valid_mask = (label_squeezed != self.ignore_index)
            n_valid = valid_mask.sum()
            loss_ce = loss_ce_raw.sum() / max(n_valid, 1)
            
            # Dice 部分：使用标准的 'mean' (在 batch 和 class 上)
            loss_dice = loss_dice_raw.mean()
            
            # 组合
            total_loss = (self.ce_weight * loss_ce) + (self.dice_weight * loss_dice)
            return total_loss.float()

        elif reduction == 'sum':
            # CE 部分：求和
            loss_ce = loss_ce_raw.sum()
            
            # Dice 部分：也求和 (在 batch 和 class 上)
            loss_dice = loss_dice_raw.sum()
            
            # 组合
            total_loss = (self.ce_weight * loss_ce) + (self.dice_weight * loss_dice)
            return total_loss

        elif reduction == 'none':
            # 这是一个不寻常的组合，因为 CE 是逐像素的，Dice 是逐图像/逐类别的
            # 我们将 Dice 损失（在类别上取均值）广播到与 CE 相同的维度
            
            # 1. Dice loss 在类别上取均值，得到每个 batch 样本的损失
            loss_dice_per_image = loss_dice_raw.mean(dim=1) # Shape: [B]
            
            # 2. 广播到 [B, H, W]
            loss_dice_broadcasted = loss_dice_per_image.view(-1, 1, 1).expand_as(loss_ce_raw)
            
            # 3. 组合
            total_loss = (self.ce_weight * loss_ce_raw) + (self.dice_weight * loss_dice_broadcasted)
            return total_loss
        
        else:
            raise ValueError(f"不支持的 reduction 类型: {reduction}")

class BalancedBinaryCrossEntropyLoss(nn.Module):
    """
    Balanced binary cross entropy loss with ignore regions.
    """
    def __init__(self, pos_weight=None, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label, reduction='mean'):

        mask = (label != self.ignore_index)
        masked_label = torch.masked_select(label, mask)
        masked_output = torch.masked_select(output, mask)

        # weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w = num_labels_neg / num_total
            if w == 1.0:
                return 0
        else:
            w = torch.as_tensor(self.pos_weight, device=output.device)
        factor = 1. / (1 - w)

        loss = nn.functional.binary_cross_entropy_with_logits(
            masked_output,
            masked_label,
            pos_weight=w*factor,
            reduction=reduction)
        loss /= factor
        return loss


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """
    def __init__(self, size_average=True, normalize=False, norm=1):
        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = (label != ignore_label)
        n_valid = torch.sum(mask).item()

        if self.normalize is not None:
            out_norm = self.normalize(out)
            loss = self.loss_func(torch.masked_select(out_norm, mask), torch.masked_select(label, mask), reduction='sum')
        else:
            loss = self.loss_func(torch.masked_select(out, mask), torch.masked_select(label, mask), reduction='sum')

        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-6))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(label.size())))
                return ret_loss

        return loss


class L1Loss(nn.Module):
    """
    from ATRC
    L1 loss with ignore regions.
    normalize: normalization for surface normals
    """
    def __init__(self, normalize=False, ignore_index=255):
        super().__init__()
        self.normalize = normalize
        self.ignore_index = ignore_index

    def forward(self, out, label, reduction='mean'):

        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        if reduction == 'mean':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum') / max(n_valid, 1)
        elif reduction == 'sum':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum')
        elif reduction == 'none':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='none')


class L2Loss(nn.Module):
    """
    from ATRC
    L2 loss with ignore regions.
    normalize: normalization for surface normals
    """
    def __init__(self, normalize=False, ignore_index=255):
        super().__init__()
        self.normalize = normalize
        self.ignore_index = ignore_index

    def forward(self, out, label, reduction='mean'):

        if self.normalize and out.shape[1] != 1:
            out = nn.functional.normalize(out, p=2, dim=1)
            label = nn.functional.normalize(label, p=2, dim=1)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        if reduction == 'mean':
            return nn.functional.mse_loss(masked_out, masked_label, reduction='sum') / max(n_valid, 1)
        elif reduction == 'sum':
            return nn.functional.mse_loss(masked_out, masked_label, reduction='sum')
        elif reduction == 'none':
            return nn.functional.mse_loss(masked_out, masked_label, reduction='none')


class GatedL2Loss(nn.Module):
    """
    from ATRC
    L2 loss with ignore regions.
    normalize: normalization for surface normals
    """
    def __init__(self, normalize=False, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.softmax = nn.Softmax(dim=1)
        self.normalize = normalize

    def forward(self, out, label, reduction='mean'):
        C = out.shape[1]

        if self.normalize:
            out_norm = nn.functional.normalize(out, p=2, dim=1)
            label_norm = nn.functional.normalize(label, p=2, dim=1)

        out = (out_norm * self.softmax(out)).sum(1, keepdim=True)
        label = (label_norm * self.softmax(label)).sum(1, keepdim=True)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        
        if reduction == 'mean':
            return nn.functional.mse_loss(masked_out, masked_label, reduction='sum') * C / max(n_valid, 1)
        elif reduction == 'sum':
            return nn.functional.mse_loss(masked_out, masked_label, reduction='sum') * C
        elif reduction == 'none':
            return nn.functional.mse_loss(masked_out, masked_label, reduction='none') * C


# --- 从 loss_utils.py 复制的依赖项 ---
# 这些是 L1FreqAwareLoss 正常工作所必需的

class SobelOperator(nn.Module):
    """
    用于 EdgeLoss 的 Sobel 算子
    """
    def __init__(self):
        super(SobelOperator, self).__init__()
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight.data = self.sobel_kernel_x()
        self.conv_y.weight.data = self.sobel_kernel_y()

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
        return magnitude

    @staticmethod
    def sobel_kernel_x():
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        return kernel_x.view(1, 1, 3, 3)

    @staticmethod
    def sobel_kernel_y():
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        return kernel_y.view(1, 1, 3, 3)


class BinaryDilation(nn.Module):
    """
    用于 EdgeLoss 的二值膨胀
    """
    def __init__(self, kernel_size=7):
        super(BinaryDilation, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(1, 1, kernel_size, stride=1, padding=self.padding, bias=False)
        self.conv.weight.data = self.dilation_kernel(kernel_size)

    def forward(self, x):
        x = self.conv(x.float()) > 0
        return x

    @staticmethod
    def dilation_kernel(kernel_size):
        kernel = torch.ones(1, 1, kernel_size, kernel_size)
        return kernel

def get_magnitude_spectrum(img):
    """
    用于 FreqLoss：获取幅度谱
    """
    fft_img = torch.fft.fft2(img)
    shifted_magnitude_spectrum = torch.fft.fftshift(fft_img)
    return shifted_magnitude_spectrum

def obtain_mask(radius, shape, device):
    """
    用于 FreqLoss：获取高通滤波器掩码
    """
    B, C, H, W = shape
    mask = torch.ones((B, C, H, W), device=device)

    center_x, center_y = W // 2, H // 2
    Y, X = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    dist_from_center = dist_from_center[np.newaxis, np.newaxis, :, :]

    mask[torch.from_numpy(dist_from_center).to(device).repeat(B, C, 1, 1) <= radius] = 0
    return mask.bool()


# --- 新的集成损失类 ---

class L1FreqAwareLoss(nn.Module):
    """
    集成了 L1Loss（重建损失）、EdgeLoss（边缘损失）和 
    FrequencyDomainLoss（频域损失）的复合损失函数。

    它结合了：
    1. 原始 L1Loss 类中的带 ignore_index 的掩码 L1 损失 (rec_loss)。
    2. EdgeLoss 中的边缘/非边缘区域加权 L1 损失 (edge_loss)。
    3. frequency_domain_loss 中的高频幅度和相位 L1 损失 (freq_loss)。
    """
    def __init__(self, 
                 # L1Loss 参数
                 task,
                 normalize=False, 
                 ignore_index=255,
                 
                 # EdgeLoss 参数 (来自您的代码片段)
                 use_edge_loss=True,
                 edge_weight=0.5,
                 edge_threshold=0.55,
                 edge_ratio=0.95,
                 edge_kernel_size=7,
                 
                 # FreqLoss 参数 (来自您的代码片段)
                 use_freq_loss=True,
                 freq_weight=5e-3,
                 freq_radius=64
                ):
        super().__init__()
        
        # 1. L1 损失参数
        self.normalize = normalize
        self.ignore_index = ignore_index

        # 2. 边缘损失组件
        self.use_edge_loss = use_edge_loss
        self.edge_weight = edge_weight
        if self.use_edge_loss:
            # 实例化边缘损失所需的模块
            self.sobel_kernel = SobelOperator()
            self.dilation_kernel = BinaryDilation(kernel_size=edge_kernel_size)
            self.edge_threshold = edge_threshold
            self.edge_ratio = edge_ratio

        # 3. 频域损失组件
        self.use_freq_loss = use_freq_loss
        self.freq_weight = freq_weight
        self.freq_radius = freq_radius
        self.task = task

    def _calculate_l1_loss(self, out, label, reduction='mean'):
        """
        计算基础 L1 损失，并尊重 ignore_index。
        这是从本文件 (loss_functions.py) 中的 L1Loss 类复制的逻辑。
        """
            
        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        n_valid = torch.sum(mask).item()
        
        if n_valid == 0:
            # 如果没有有效像素，损失为0
            return torch.tensor(0.0, device=out.device)
            
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        
        if reduction == 'mean':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum') / max(n_valid, 1)
        elif reduction == 'sum':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum')
        elif reduction == 'none':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='none')
        
        # 默认回退
        return nn.functional.l1_loss(masked_out, masked_label, reduction='sum') / max(n_valid, 1)

    def _calculate_edge_loss(self, pred, label, valid_mask):
        """
        计算边缘损失组件。
        pred, label: 已经是 [0, 1] 归一化的张量
        valid_mask: [B, 1, H, W] 的掩码，来自 ignore_index
        """
        # 边缘损失在单通道（均值）上计算
        pred_mean, label_mean = pred.mean(1, keepdim=True), label.mean(1, keepdim=True)

        with torch.no_grad():
            label_edge = self.sobel_kernel(label_mean)
            # 1. Sobel 计算的边缘区域掩码
            edge_region_mask = self.dilation_kernel((label_edge >= self.edge_threshold).float())
            # 2. Sobel 计算的非边缘区域掩码
            reverse_edge_region_mask = ~edge_region_mask
        
        # --- 修改：结合 valid_mask ---
        # 最终的边缘掩码 = 边缘区域 AND 有效区域(valid_mask)
        final_edge_mask = edge_region_mask & valid_mask
        # 最终的非边缘掩码 = 非边缘区域 AND 有效区域(valid_mask)
        final_non_edge_mask = reverse_edge_region_mask & valid_mask
        # --- 结束修改 ---

        # 计算边缘区域的损失
        loss_edge = 0.0
        if final_edge_mask.int().sum() > 0: # <--- 使用 final_edge_mask
            masked_label = torch.masked_select(label_mean, final_edge_mask)
            masked_pred = torch.masked_select(pred_mean, final_edge_mask)
            loss_edge = nn.functional.l1_loss(masked_pred, masked_label)
        
        # 计算非边缘区域的损失
        loss_non_edge = 0.0
        if final_non_edge_mask.int().sum() > 0: # <--- 使用 final_non_edge_mask
            unmasked_label = torch.masked_select(label_mean, final_non_edge_mask)
            unmasked_pred = torch.masked_select(pred_mean, final_non_edge_mask)
            loss_non_edge = nn.functional.l1_loss(unmasked_pred, unmasked_label)
        
        if final_edge_mask.int().sum() == 0 and final_non_edge_mask.int().sum() == 0:
             return torch.tensor(0.0, device=pred.device)

        # 根据 ratio 组合
        return loss_edge * self.edge_ratio + loss_non_edge * (1 - self.edge_ratio)

    def _calculate_freq_loss(self, pred, label, valid_mask):
        """
        计算频域损失组件。
        pred, label: 已经是 [0, 1] 归一化的张量
        valid_mask: [B, C, H, W] 的掩码，来自 ignore_index
        """
        
        pred_masked = pred * valid_mask.float()
        label_masked = label * valid_mask.float()
        # --- 结束修改 ---

        # 在掩码后的图像上计算频谱
        pred_spectrum = get_magnitude_spectrum(pred_masked)
        label_spectrum = get_magnitude_spectrum(label_masked)

        pred_magnitude, pred_angle = torch.abs(pred_spectrum), torch.angle(pred_spectrum)
        label_magnitude, label_angle = torch.abs(label_spectrum), torch.angle(label_spectrum)

        loss = 0.0
        if self.freq_radius > 0:
            mask = obtain_mask(self.freq_radius, pred_spectrum.shape, pred_spectrum.device)
            
            if mask.int().sum() == 0:
                return torch.tensor(0.0, device=pred.device)

            masked_label_magnitude = torch.masked_select(label_magnitude, mask)
            masked_label_angle = torch.masked_select(label_angle, mask)
            masked_pred_magnitude = torch.masked_select(pred_magnitude, mask)
            masked_pred_angle = torch.masked_select(pred_angle, mask) 
            
            loss = nn.functional.l1_loss(masked_pred_magnitude, masked_label_magnitude) / 5.0 + \
                   nn.functional.l1_loss(masked_pred_angle, masked_label_angle)
        else:
            loss = nn.functional.l1_loss(pred_magnitude, label_magnitude) / 5.0 + \
                   nn.functional.l1_loss(pred_angle, label_angle)

        return loss

    def forward(self, out, label, reduction='mean'):

        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)
        
        rec_loss = self._calculate_l1_loss(out, label, reduction)
        total_loss = rec_loss
        
        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        n_valid = torch.sum(mask).item()
        if n_valid == 0:
            return total_loss

        if self.task == 'depth':
            label_clone = label.clone()
            label_clone[~mask.repeat(1, label.shape[1], 1, 1)] = -torch.inf
            max_vals = label_clone.amax(dim=(1, 2, 3), keepdim=True)
            max_vals = max_vals.clamp(min=1e-6) 

            out_norm = out / max_vals
            label_norm = label / max_vals
        else:
            out_norm = out
            label_norm = label

        
        # 2. 计算并添加边缘损失
        if self.use_edge_loss:
            edge_loss = self._calculate_edge_loss(out_norm, label_norm, mask)
            total_loss = total_loss + self.edge_weight * edge_loss
        
        # 3. 计算并添加频域损失
        if self.use_freq_loss:
            freq_mask = mask.repeat(1, out.shape[1], 1, 1)
            freq_loss = self._calculate_freq_loss(out_norm, label_norm, freq_mask)
            total_loss = total_loss + self.freq_weight * freq_loss
        
        # total_loss = total_loss.clamp(max=10.0)

        return total_loss