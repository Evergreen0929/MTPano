# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers.transformer_decoder import TransformerDecoder, SimpleTransformerNet
from transformers.models.dpt.modeling_dpt import DPTPreTrainedModel
from transformers import DPTConfig, DPTForSemanticSegmentation, DPTForDepthEstimation
import pdb
from einops import rearrange as o_rearrange
from .pano_net_utils import ScaleAggregator, AuxTaskHead, BridgeBlock, CascadedGroupRefiner, LayerNorm2d

from transformers.models.dpt.modeling_dpt import DPTNeck

from easydict import EasyDict as edict
INTERPOLATE_MODE = 'bilinear'

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()



class TransformerBaselineBFE(nn.Module):
    def __init__(self, p, backbone, backbone_channels, heads):
        super(TransformerBaselineBFE, self).__init__()
        self.p = p
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        self.heads = heads
        
        # ==================== 1. 原 Baseline3 初始化逻辑 ====================
        print("Loading DPT configuration object...")
        config = DPTConfig.from_pretrained("Intel/dpt-large-ade")

        train_h, train_w = self.p.TRAIN.SCALE
        config.image_size = (train_h, train_w)
        print(f"Patched DPT config to use non-square image_size: {config.image_size}")
        print("Loading DPT model using the patched configuration...")

        self.necks = nn.ModuleDict()
        dpt_model = None # temp holder
        for task in self.tasks:
            dpt_model = DPTForSemanticSegmentation.from_pretrained(
                "Intel/dpt-large-ade",
                config=config, 
                use_safetensors=True,
                ignore_mismatched_sizes=True
            )
            self.necks[task] = dpt_model.neck
        self.backbone_out_indices = dpt_model.config.backbone_out_indices
        self.spatial_dim = p.spatial_dim
        print("Instantiated DPT Neck with correct non-square config.")

        # Group Refiner
        self.feature_grouping = nn.ModuleList()
        CASCADE_LAYERS = 2 
        for i in range(len(self.backbone_out_indices)):
            self.feature_grouping.append(
                CascadedGroupRefiner(backbone_channels[i], layers=CASCADE_LAYERS)
            )

        self.extended_tasks = ['grad', 'sdf']
        if 'depth' in self.tasks:
            self.extended_tasks.append('point')
        self.task_attributes = {'semseg': 'invariant', 'depth': 'invariant', 'normals': 'variant',
                                'grad': 'invariant', 'sdf': 'invariant', 'point': 'variant'}

        # Init Spherical PE
        patch_h, patch_w = self.spatial_dim[0]
        self._init_spherical_pe(patch_h, patch_w)

        # ==================== 2. 新增 BFE/Aux 模块初始化 ====================
        self.bridge_dim = 256
        self.task_feat_dim = 64
        
        # A. Aggregators
        self.aggregator_inv = ScaleAggregator(backbone_channels, self.bridge_dim)
        self.aggregator_var = ScaleAggregator(backbone_channels, self.bridge_dim)
        
        # B. Aux Heads (Input: bridge_dim -> Output: bridge_dim for attention)
        self.aux_out_channels = {t: p.TASKS.NUM_OUTPUT[t] for t in self.tasks}
        self.aux_heads = nn.ModuleDict({
            t: AuxTaskHead(self.bridge_dim, self.aux_out_channels[t], task_feat_dim=self.bridge_dim)
            for t in self.tasks
        })

        self.ext_out_channels = {'grad':2, 'sdf':1, 'point':3}
        self.extended_heads = nn.ModuleDict({
            t: AuxTaskHead(self.bridge_dim, self.ext_out_channels[t], task_feat_dim=self.bridge_dim)
            for t in self.extended_tasks
        })
        
        # C. Bridge Blocks (Cross Attention)
        self.inv_tasks = ['grad', 'sdf']  # 'semseg', 'depth'
        if 'semseg' in self.tasks:
            self.inv_tasks.append('semseg')
        if 'depth' in self.tasks:
            self.inv_tasks.append('depth')

        self.var_tasks = []  # 'normals', 'point'
        if 'normals' in self.tasks:
            self.var_tasks.append('normals')
        if 'depth' in self.tasks:
            self.var_tasks.append('point')
        
        all_tasks = self.inv_tasks + self.var_tasks
        self.bfe_invariant = BridgeBlock(all_tasks, dim=self.bridge_dim, num_heads=4, qkv_bias=True)
        self.bfe_variant = BridgeBlock(all_tasks, dim=self.bridge_dim, num_heads=4, qkv_bias=True)
        
        # D. Injectors (Zero Conv)
        # (1) Global Context Injectors: Bridge -> Backbone
        self.inv_injectors = nn.ModuleList([
            nn.Conv2d(self.bridge_dim, c, kernel_size=1, bias=True) for c in backbone_channels
        ])
        self.var_injectors = nn.ModuleList([
            nn.Conv2d(self.bridge_dim, c, kernel_size=1, bias=True) for c in backbone_channels
        ])
        
        # (2) Task Specific Injectors: Aux -> Task Features
        # 每个 Task 对应每个 Scale 都有一个独立的 Zero Conv
        self.aux_injectors = nn.ModuleDict()
        for t in self.tasks:
            self.aux_injectors[t] = nn.ModuleList([
                nn.Conv2d(self.bridge_dim, c, kernel_size=1, bias=True) for c in backbone_channels
            ])
        
        # (3) [NEW] Task Adapters: Base -> Task Features (Part 2 of Injection)
        # 结构: Conv3x3 -> LN -> GELU -> Conv1x1(Zero)
        self.task_adapters = nn.ModuleDict()
        for t in self.tasks:
            self.task_adapters[t] = nn.ModuleList()
            for c in backbone_channels:
                adapter = nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                    LayerNorm2d(c),
                    nn.GELU(),
                    nn.Conv2d(c, c, kernel_size=1, bias=True) # Last Layer: Zero Init
                )
                self.task_adapters[t].append(adapter)

        # 执行 Zero Init
        self._init_injectors()
        print("Initialized TransformerBaselineBFE with BridgeNet and Aux Injection.")

    def _init_spherical_pe(self, h, w):
        y_coords = torch.linspace(0, 1, h)
        x_coords = torch.linspace(0, 1, w)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        phi = y_grid * math.pi
        theta = x_grid * 2 * math.pi

        ray_x = torch.sin(phi) * torch.cos(theta)
        ray_y = torch.cos(phi)
        ray_z = torch.sin(phi) * torch.sin(theta)

        ray_dir = torch.stack([ray_x, ray_y, ray_z], dim=0).unsqueeze(0)
        self.register_buffer('RayDir', ray_dir)

        pe_sphere = torch.stack([phi, theta], dim=0).unsqueeze(0)
        self.register_buffer('PE', pe_sphere)

    def _init_injectors(self):
        # 批量初始化 Zero Conv
        all_injectors = list(self.inv_injectors) + list(self.var_injectors)
        for t in self.tasks:
            all_injectors.extend(list(self.aux_injectors[t]))
            
        for m in all_injectors:
            nn.init.zeros_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        
        for t in self.tasks:
            for adapter in self.task_adapters[t]:
                # adapter[0]: Conv3x3 -> Kaiming
                nn.init.kaiming_normal_(adapter[0].weight, mode='fan_out', nonlinearity='relu')
                # adapter[1]: LayerNorm -> 1, 0
                nn.init.constant_(adapter[1].norm.weight, 1.0)
                nn.init.constant_(adapter[1].norm.bias, 0.0)
                # adapter[3]: Conv1x1 -> Zero Init (Last Layer)
                last_conv = adapter[3]
                nn.init.zeros_(last_conv.weight)
                if last_conv.bias is not None: nn.init.zeros_(last_conv.bias)

    def forward_feature_grouping(self, backbone_feat, i): 
        out_grouped = self.feature_grouping[i](backbone_feat, self.RayDir, self.PE)
        return out_grouped
    
    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        # 1. Backbone
        backbone_outputs = self.backbone(x, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = backbone_outputs.hidden_states
        
        patch_h, patch_w = self.spatial_dim[0]
        
        # 临时存储
        cls_tokens = []
        inv_feats_list = []
        var_feats_list = []
        
        # === Phase 1: Feature Collection & Initial Refinement ===
        for i, hidden_state in enumerate([encoder_hidden_states[idx + 1] for idx in self.backbone_out_indices]):
            # 提取 CLS 和 Patch Tokens
            cls_token = hidden_state[:, 0:1, :]
            cls_tokens.append(cls_token)
            
            patch_tokens = hidden_state[:, 5:, :]
            B, _, C = patch_tokens.shape
            reshaped_feature = patch_tokens.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
            
            # Group Refiner
            grouped = self.forward_feature_grouping(reshaped_feature, i)
            inv_feats_list.append(grouped['invariant'])
            var_feats_list.append(grouped['variant'])

        # === Phase 2: BFE & Aux Process ===
        # 1. Aggregation (Generic Features)
        agg_inv = self.aggregator_inv(inv_feats_list)
        agg_var = self.aggregator_var(var_feats_list)
        
        # 2. Aux Heads (Task Specific Features)
        task_specific_feats = {} # {task: [B, bridge_dim, H, W]}
        aux_preds = {}
        extended_preds = {}
        for t in self.tasks:
            src = agg_inv if self.task_attributes[t] == 'invariant' else agg_var
            feat, pred = self.aux_heads[t](src)
            task_specific_feats[t] = feat
            aux_preds[t] = F.interpolate(pred, img_size, mode=INTERPOLATE_MODE)
        out['aux_preds'] = aux_preds

        for t in self.extended_tasks:
            src = agg_inv if self.task_attributes[t] == 'invariant' else agg_var
            feat, pred = self.extended_heads[t](src)
            task_specific_feats[t] = feat
            extended_preds[t] = F.interpolate(pred, img_size, mode=INTERPOLATE_MODE)
        out['ext_preds'] = extended_preds

        # 3. Bridge (Cross Attention)
        # Query: Generic, Key/Value: Specific
        bridge_inv = self.bfe_invariant(agg_inv, task_specific_feats, self.inv_tasks)
        bridge_var = self.bfe_variant(agg_var, task_specific_feats, self.var_tasks)

        # === Phase 3: Injection & Final Output ===
        selected_encoder_features = {task: [] for task in self.tasks}
        
        for i in range(len(inv_feats_list)):
            # A. Global Injection (Bridge -> Generic)
            feat_inv_refined = inv_feats_list[i] + self.inv_injectors[i](bridge_inv)
            feat_var_refined = var_feats_list[i] + self.var_injectors[i](bridge_var)
            
            level_group = {'invariant': feat_inv_refined, 'variant': feat_var_refined}
            
            # B. Task Specific Injection & Tokenization
            for t in self.tasks:
                attr = self.task_attributes[t]
                
                # Base Feature (from Invariant or Variant)
                base_feat = level_group[attr]
                
                # Aux Injection (Aux -> Task)
                aux_feat = task_specific_feats[t] # [B, BridgeDim, H, W]
                injection_1 = self.aux_injectors[t][i](aux_feat) 
                injection_2 = self.task_adapters[t][i](base_feat)
                
                # Combine
                final_feat = base_feat + injection_1 + injection_2
                
                # Reshape to Token: [B, C, H, W] -> [B, HW, C]
                final_tokens = final_feat.flatten(2).transpose(1, 2)
                
                # Concat CLS Token -> [B, HW+1, C]
                final_tokens = torch.cat([cls_tokens[i], final_tokens], dim=1)
                
                selected_encoder_features[t].append(final_tokens)

        # === Phase 4: Heads ===
        for t in self.tasks:
            neck_out = self.necks[t](selected_encoder_features[t], patch_height=patch_h, patch_width=patch_w)
            task_logits = self.heads[t](neck_out)
            if t == 'depth':
                task_logits = task_logits.unsqueeze(1)
            out[t] = task_logits

        return out


class DPTNormalsHead(DPTPreTrainedModel):
    def __init__(self, config: DPTConfig):
        super().__init__(config)
        self.config = config
        features = config.fusion_hidden_size # 通常是 256

        # 复制 DPTDepthEstimationHead 的 nn.Sequential 结构
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1), # 256 -> 128
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1), # 128 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0), # 32 -> 3 (R, G, B)
        )

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        if not isinstance(hidden_states, list):
            raise ValueError(f"DPT Head excepts a list input, but received {type(hidden_states)}")
        
        idx_to_use = self.config.head_in_index

        hidden_state = hidden_states[idx_to_use]
        
        # 运行头部
        logits = self.head(hidden_state)
        # 输出形状为 [B, 3, H, W]，这是正确的
        return logits