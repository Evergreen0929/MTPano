import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers.transformer_decoder import TransformerDecoder, SimpleTransformerNet
from transformers.models.dpt.modeling_dpt import DPTPreTrainedModel
from transformers import DPTConfig, DPTForSemanticSegmentation
from einops import rearrange as o_rearrange

# 沿用之前的定义
INTERPOLATE_MODE = 'bilinear'

class LinearAutoEncoder(nn.Module):
    def __init__(self, in_channels, compressed_channels=16):
        super(LinearAutoEncoder, self).__init__()
        self.encoder = nn.Conv2d(in_channels, compressed_channels, kernel_size=1, bias=True)
        self.decoder = nn.Conv2d(compressed_channels, in_channels, kernel_size=1, bias=True)
        
        self._init_weights()

    def _init_weights(self):
        # 使用 Kaiming 初始化
        nn.init.kaiming_normal_(self.encoder.weight, mode='fan_out', nonlinearity='linear')
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
            
        nn.init.kaiming_normal_(self.decoder.weight, mode='fan_out', nonlinearity='linear')
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        # x: [B, C, H, W]
        compressed = self.encoder(x)       # [B, 1, H, W]
        reconstructed = self.decoder(compressed) # [B, C, H, W]
        return compressed, reconstructed


class EquiRectangularDualConv(nn.Module):
    """
    全景图专用动态卷积层。
    输入/输出通道数保持一致 (类似于 ResNet 中的 3x3 卷积位置)。
    
    逻辑：
    - 赤道区域 (Lat ~ 0): 使用标准 3x3 卷积 (Dense features)。
    - 极点区域 (Lat ~ 90): 使用 3x9 Depthwise 卷积 (抗横向畸变)。
    - 混合: 基于 Ray 的 Y 分量 (cos(phi)) 进行硬编码加权。
    """
    def __init__(self, channels):
        super().__init__()
        self.equator_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pole_conv = nn.Conv2d(
            channels, channels, 
            kernel_size=(3, 9), 
            padding=(1, 4), 
            groups=channels  # Depthwise
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.equator_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pole_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.equator_conv.bias is not None: nn.init.zeros_(self.equator_conv.bias)
        if self.pole_conv.bias is not None: nn.init.zeros_(self.pole_conv.bias)

    def forward(self, x, ray_dir):
        """
        x: [B, C, H, W]
        ray_dir: [B, 3, H, W] (用于确定纬度权重)
        """
        pole_weight = ray_dir[:, 1:2, :, :].abs()
        
        feat_equator = self.equator_conv(x)
        feat_pole = self.pole_conv(x)
        
        out = (1 - pole_weight) * feat_equator + pole_weight * feat_pole
        
        return out


class GeometryModulatedBlock(nn.Module):
    """
    Variant 分支专用：同时接受 Ray (3D) 和 PE (2D) 进行特征调制。
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        # 1. 特征提取路径 (标准 Bottleneck)
        mid_channels = channels // reduction
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.act1 = nn.GELU()
        self.spatial_process = EquiRectangularDualConv(mid_channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, channels, kernel_size=1) # Zero-Init 目标

        # 2. 几何调制生成器 (输入: Ray(3) + PE(2) = 5通道)
        # 注意：这里的 PE 假设是 (Phi, Theta) 2通道图
        self.geo_mlp = nn.Sequential(
            nn.Conv2d(3 + 2, mid_channels, kernel_size=1), # 3(Ray) + 2(PE)
            nn.ReLU(),
            nn.Conv2d(mid_channels, channels * 2, kernel_size=1) # 生成 Scale 和 Shift
        )

        self._init_weights()

    def _init_weights(self):
        # Kaiming Init for normal convs
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # --- Zero Init: 保证初始状态不干扰 Backbone 特征 ---
        # 1. 卷积通路置零
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
            
        # 2. 调制 MLP 置零 (初始 Scale=0, Shift=0)
        nn.init.kaiming_normal_(self.geo_mlp[0].weight, mode='fan_out', nonlinearity='relu')
        if self.geo_mlp[0].bias is not None:
            nn.init.zeros_(self.geo_mlp[0].bias)
            
        nn.init.zeros_(self.geo_mlp[-1].weight)
        nn.init.zeros_(self.geo_mlp[-1].bias)

    def forward(self, x, ray_dir, pe_map):
        # x: [B, C, H, W]
        # ray_dir: [B, 3, H, W]
        # pe_map:  [B, 2, H, W]
        
        residual = x
        
        # Path A: 内容处理
        feat = self.act1(self.conv1(x))
        feat = self.spatial_process(feat, ray_dir)
        feat = self.conv2(self.act2(feat))

        # Path B: 几何调制
        # 拼接几何先验
        geo_cond = torch.cat([ray_dir, pe_map], dim=1) 
        style = self.geo_mlp(geo_cond)
        gamma, beta = style.chunk(2, dim=1)
        
        # AdaIN Modulation
        feat_modulated = feat * (1 + gamma) + beta
        
        return residual + feat_modulated

class StandardResidualBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = channels // reduction
        
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.act1 = nn.GELU()
        
        self.spatial_process = EquiRectangularDualConv(mid_channels)
        self.act2 = nn.GELU()
        
        self.conv2 = nn.Conv2d(mid_channels, channels, kernel_size=1) # Zero-Init

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None: nn.init.zeros_(self.conv2.bias)

    def forward(self, x, ray_dir):
        
        residual = x
        
        out = self.act1(self.conv1(x))
        out = self.act2(self.spatial_process(out, ray_dir)) # DualConv 需要 ray_dir
        out = self.conv2(out)
        
        return residual + out


class CascadedGroupRefiner(nn.Module):
    def __init__(self, channels, layers=2):
        super().__init__()
        # Invariant: 只有特征，没有几何输入
        self.invariant_layers = nn.ModuleList([
            StandardResidualBlock(channels) for _ in range(layers)
        ])
        
        # Variant: 特征 + Ray + PE
        self.variant_layers = nn.ModuleList([
            GeometryModulatedBlock(channels) for _ in range(layers)
        ])

    def forward(self, x, ray_dir, pe_map):
        if ray_dir.size(0) != x.size(0):
            ray_dir = ray_dir.expand(x.size(0), -1, -1, -1)
        if pe_map.size(0) != x.size(0):
            pe_map = pe_map.expand(x.size(0), -1, -1, -1)

        feat_inv = x
        for layer in self.invariant_layers:
            feat_inv = layer(feat_inv, ray_dir)
            
        feat_var = x
        for layer in self.variant_layers:
            feat_var = layer(feat_var, ray_dir, pe_map)
            
        return {'invariant': feat_inv, 'variant': feat_var}


class LayerNorm2d(nn.Module):
    """
    针对 [B, C, H, W] 输入的 LayerNorm。
    严格实现：permute(0, 2, 3, 1) -> nn.LayerNorm -> permute(0, 3, 1, 2)
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1) # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # [B, C, H, W]
        return x

class ScaleAggregator(nn.Module):
    """
    聚合多层 Backbone 特征。
    移除插值，直接 Concat (ViT 特征尺度一致)。
    """
    def __init__(self, backbone_channels, hidden_dim=256):
        super().__init__()
        # 计算总通道数
        total_in_channels = sum(backbone_channels)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_in_channels, hidden_dim, kernel_size=1, bias=False),
            LayerNorm2d(hidden_dim),
            nn.GELU()
        )

        self._init_weights()
    
    def _init_weights(self): # <--- 新增初始化方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):  
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, features):
        # features: list of [B, Ci, H, W]
        # 直接在通道维拼接
        concat_feat = torch.cat(features, dim=1)
        return self.fusion_conv(concat_feat)


class AuxTaskHead(nn.Module):
    """
    辅助任务头。
    1. 产生中间监督 (Aux Pred)
    2. 提取 Task Specific Feature 用于 Bridge (作为 Key/Value)
    注意：这里的 Conv3x3 使用标准卷积 (Standard Conv)，不使用 DualConv。
    """
    def __init__(self, in_channels, out_channels, task_feat_dim=64):
        super().__init__()
        
        # 提取用于 Bridge 的特征
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, task_feat_dim, kernel_size=1, bias=False),
            LayerNorm2d(task_feat_dim),
            nn.GELU()
        )
        
        self.classifier = nn.Conv2d(task_feat_dim, out_channels, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):  
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        task_feat = self.feature_extractor(x)
        pred = self.classifier(task_feat)
        return task_feat, pred

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv):
        # x_q: [B, Nq, C] (Generic)
        # x_kv: [B, Nkv, C] (Specific Concat)
        B, Nq, C = x_q.shape
        B, Nkv, _ = x_kv.shape

        q = self.to_q(x_q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.to_k(x_kv).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.to_v(x_kv).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BridgeBlock(nn.Module):
    """
    Bridge Feature Extractor (BFE) based on BridgeNet.
    Structure: Cross-Attention -> FFN
    - Query: Task-Generic Features (Aggregated Backbone Feats)
    - Key/Value: Task-Specific Features (Concatenated Aux Feats)
    """
    def __init__(self, all_tasks, dim, num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0.):
        super().__init__()
        
        self.all_tasks = all_tasks

        # 1. Cross Attention Block
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.kv_squeeze = nn.ModuleDict({
            t: nn.Conv2d(dim, dim, 3, 2, 1) for t in self.all_tasks
        })
        
        # 2. Feed Forward Block
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_generic, task_features_dict, task_names):
        """
        x_generic: [B, C, H, W] - The "Query"
        task_features_dict: Dict of [B, C, H, W] - The "Key/Value" sources
        """
        B, C, H, W = x_generic.shape
        
        # 1. Prepare Query (Flatten spatial)
        # [B, C, H, W] -> [B, H*W, C]
        q_tokens = x_generic.flatten(2).transpose(1, 2)
        
        # 2. Prepare Key/Value (Concat task tokens)
        # flatten -> [B, N_task * H * W, C]
        kv_list = []
        for t in self.all_tasks:
            # [B, C, H, W] -> [B, H*W, C]
            feat = self.kv_squeeze[t](task_features_dict[t]).flatten(2).transpose(1, 2)
            if t not in task_names:
                feat = feat.detach().clone()
            kv_list.append(feat)
            
        # [B, (Num_Tasks * H * W), C]
        kv_tokens = torch.cat(kv_list, dim=1)
        
        # 3. Cross Attention: Generic queries Specific
        # x + attn(norm(x))
        tmp_q = self.norm1_q(q_tokens)
        tmp_kv = self.norm1_kv(kv_tokens)
        
        attn_out = self.cross_attn(tmp_q, tmp_kv)
        x = q_tokens + attn_out
        
        # 4. FFN
        x = x + self.mlp(self.norm2(x))
        
        # 5. Reshape back to Spatial
        # [B, H*W, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x