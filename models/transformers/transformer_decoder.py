# Modified from InvPT by Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BridgeNet import TRFL
import pdb
import numpy as np
from einops import rearrange as o_rearrange

from .attention_block import AttnBlock, FusionBlock

INTERPOLATE_MODE = 'bilinear'
BATCHNORM = nn.SyncBatchNorm # nn.BatchNorm2d

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


class SimpleTransformerNet(nn.Module):

    def __init__(self, p):
        super().__init__()

        self.mt_embed_dim = p.backbone_channels[0]
        out_dim = p.backbone_channels[-1]

        self.fuse_conv = nn.Conv2d(self.mt_embed_dim * 4, self.mt_embed_dim, 3, 1, 1)

        self.p = p

        self.all_tasks = p.TASKS.NAMES  # + ['back']

        self.mt_upsample = nn.ModuleDict()
        for task in self.all_tasks:
            self.mt_upsample[task] = nn.Sequential(
                        nn.ConvTranspose2d(self.mt_embed_dim, self.mt_embed_dim, kernel_size=4, stride=2, padding=1),
                        nn.GELU(),
                        nn.ConvTranspose2d(self.mt_embed_dim, self.mt_embed_dim, kernel_size=4, stride=2, padding=1)
                    )

        # sepOut
        self.mt_proj = nn.ModuleDict()
        for task in self.all_tasks:
            self.mt_proj[task] = nn.Sequential(nn.Conv2d(self.mt_embed_dim, self.mt_embed_dim, 3, padding=1),
                                               nn.BatchNorm2d(self.mt_embed_dim),
                                               nn.GELU(),
                                               nn.Conv2d(self.mt_embed_dim, self.mt_embed_dim, 3, padding=1),
                                               nn.BatchNorm2d(self.mt_embed_dim),
                                               nn.GELU()
                                               )

    def forward(self, x, selected_fea):

        inter_pred = None

        task_fea_dict = {}
        h, w = self.p.spatial_dim[0]
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        reshaped_fea = []
        for fea in selected_fea:
            reshaped_fea.append(rearrange(fea, 'b (h w) c -> b c h w', h=h, w=w))
        
        x = self.fuse_conv(torch.cat(reshaped_fea, dim=1))

        # print('backbone feature:', x.shape)

        for task in self.all_tasks:
            task_fea_dict[task] = x  # self.deconv[task](x)

        # only use last scale feature
        multi_scale_task_feature_out = {}
        for i, task in enumerate(self.all_tasks):
            _task_fea = task_fea_dict[task]
            _task_fea = F.interpolate(_task_fea, scale_factor=4, mode=INTERPOLATE_MODE, align_corners=False) + self.mt_upsample[task](_task_fea)

            # print(f'upsampled {task} feature:', _task_fea.shape)

            multi_scale_task_feature_out[task] = _task_fea

        x_dict = {}
        for i, task in enumerate(self.all_tasks):
            x_dict[task] = self.mt_proj[task](multi_scale_task_feature_out[task]) + multi_scale_task_feature_out[task]
        
        # for task, feat in x_dict.items():
            # print(f'decoded {task} feature:', feat.shape)

        return x_dict, inter_pred  # x, attns_list


class TransformerDecoder(nn.Module):

    def __init__(self, p):
        super().__init__()

        self.embed_dim = p.embed_dim
        embed_dim_with_pred = self.embed_dim + p.PRED_OUT_NUM_CONSTANT
        p.mtt_resolution = [_ // p.mtt_resolution_downsample_rate for _ in p.spatial_dim[-1]] # resolution at the input of transformer decoder
        self.p = p

        if p.backbone in ['vitL', 'vitB', 'intern_b', 'intern_l']:
            spec = {
                'ori_embed_dim': self.embed_dim,
                'NUM_STAGES': 3,
                'PATCH_SIZE': [0, 3, 3],
                'PATCH_STRIDE': [0, 1, 1],
                'PATCH_PADDING': [0, 2, 2],
                'DIM_EMBED': [embed_dim_with_pred, embed_dim_with_pred//2, embed_dim_with_pred//4],
                'NUM_HEADS': [2, 2, 2],
                'MLP_RATIO': [4., 4., 4.],
                'DROP_PATH_RATE': [0.15, 0.15, 0.15],
                'QKV_BIAS': [True, True, True],
                'KV_PROJ_METHOD': ['avg', 'avg', 'avg'],
                'KERNEL_KV': [2, 4, 8],
                'PADDING_KV': [0, 0, 0],
                'STRIDE_KV': [2, 4, 8],
                'Q_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
                'KERNEL_Q': [3, 3, 3],
                'PADDING_Q': [1, 1, 1],
                'STRIDE_Q': [2, 2, 2],
                'REFINE_DEPTH': p.REFINE_DEPTH,
                'DILATED_RATIO': p.DILATED_RATIO,
                'SQUEEZE_RATIO': p.SQUEEZE_RATIO
            }
        else:
            raise NotImplementedError(f'Unknown backbone type: {p.backbone}')

        # intermediate supervision
        input_channels = p.backbone_channels[-1]
        aux_channels = p.backbone_channels[-2]
        task_channels = self.embed_dim
        self.intermediate_head = nn.ModuleDict()
        if p.backbone in ['vitL', 'vitB', 'intern_b', 'intern_l']:
            self.trfl = TRFL(p, in_chans=embed_dim_with_pred, spec=spec,
                             back_channels=[spec['DIM_EMBED'][2], spec['DIM_EMBED'][1], spec['DIM_EMBED'][0], input_channels])
        else:
            self.trfl = TRFL(p, in_chans=embed_dim_with_pred, spec=spec,
                         back_channels=[spec['DIM_EMBED'][3], spec['DIM_EMBED'][2], spec['DIM_EMBED'][1], spec['DIM_EMBED'][0]])

        self.preliminary_decoder = nn.ModuleDict()
        self.atten_production = nn.ModuleDict()
        self.atten_fusion = nn.ModuleDict()

        task_no = len(p.TASKS.NAMES)

        dpr = [x.item() for x in torch.linspace(0, spec['DROP_PATH_RATE'][-1], 1)]

        for t in p.TASKS.NAMES:
            self.intermediate_head[t] = nn.Conv2d(task_channels, p.TASKS.NUM_OUTPUT[t], 1) 
            self.preliminary_decoder[t] = nn.Sequential(
                                            ConvBlock(input_channels, input_channels),
                                            ConvBlock(input_channels, task_channels),
                                        )
            self.atten_production[t] = AttnBlock(
                task_no=task_no,
                dim_in=task_channels,
                dim_out=task_channels,
                num_heads=2,
                drop=0,
                attn_drop=0,
                drop_path=dpr[0],
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                qkv_bias=True,
                stride_q=2
            )
            self.atten_fusion[t] = FusionBlock(
                task_no=task_no,
                dim_in=task_channels,
                dim_out=task_channels,
                num_heads=2,
                drop=0,
                attn_drop=0,
                drop_path=dpr[0],
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                qkv_bias=True,
                stride_q=2
            )

        self.scale_embed = nn.ModuleList()
        if p.backbone in ['vitL', 'vitB', 'intern_b', 'intern_l']:
            if p.backbone in ['intern_b', 'intern_l']:
                self.scale_embed.append(nn.Conv2d(p.backbone_channels[0], spec['DIM_EMBED'][2], 3, padding=0))
            else:
                self.scale_embed.append(nn.ConvTranspose2d(p.backbone_channels[0], spec['DIM_EMBED'][2], kernel_size=3, stride=2, padding=1,output_padding=1))
            self.scale_embed.append(nn.Conv2d(p.backbone_channels[1], spec['DIM_EMBED'][1], 3, padding=1))
            self.scale_embed.append(nn.Conv2d(p.backbone_channels[2], spec['DIM_EMBED'][0], 3, padding=1))
            if p.backbone in ['vitL', 'vitB', 'intern_b', 'intern_l']:
                self.scale_embed.append(None)
            else:
                self.scale_embed.append(nn.Conv2d(p.backbone_channels[3], p.backbone_channels[3] // 2, 1, padding=0))
        else:
            self.scale_embed.append(nn.Conv2d(p.backbone_channels[0], spec['DIM_EMBED'][3], 1, padding=0))
            self.scale_embed.append(nn.Conv2d(p.backbone_channels[1], spec['DIM_EMBED'][2], 1, padding=0))
            self.scale_embed.append(nn.Conv2d(p.backbone_channels[2], spec['DIM_EMBED'][1], 1, padding=0))
            self.scale_embed.append(nn.Conv2d(p.backbone_channels[3], spec['DIM_EMBED'][0], 1, padding=0))

    def forward(self, x_list):
        '''
        Input:
        Backbone multi-scale feature list: 4 * x: tensor [B, embed_dim, h, w]
        '''
        back_fea = []
        for sca in range(len(x_list)):
            oh, ow = self.p.spatial_dim[sca]
            _fea = x_list[sca]
            if len(_fea.shape) == 3:
                _fea = rearrange(_fea, 'b (h w) c -> b c h w', h=oh, w=ow)
            if sca == 3:
                x = _fea # use last scale feature as input of InvPT decoder
            # if sca == 2:
            #     aux_x = _fea # use 2 scale feature as aux decoder feature
            if self.scale_embed[sca] != None:
                _fea = self.scale_embed[sca](_fea)
            back_fea.append(_fea)

        h, w = self.p.mtt_resolution
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        # aux_x = F.interpolate(aux_x, size=(h, w), mode='bilinear', align_corners=False)

        # intermediate supervision
        ms_feat_dict = {}
        inter_pred = {}
        # aux_pred = {}
        atten_list = []
        param_dict = {}
        task_interfeat_dict = {}
        for task in self.p.TASKS.NAMES:
            _x = self.preliminary_decoder[task](x)
            # _x_aux = self.aux_decoder[task](aux_x)
            # _aux_p = self.aux_head[task](_x_aux)
            # aux_pred[task] = _aux_p

            ms_feat_dict[task] = _x
            task_interfeat_dict[task] = _x
            atten, params = self.atten_production[task](_x)
            atten_list.append(atten)
            param_dict[task] = params

        for task in self.p.TASKS.NAMES:
            _x = self.atten_fusion[task](atten_list, param_dict[task]) + task_interfeat_dict[task]
            _inter_p = self.intermediate_head[task](_x)
            inter_pred[task] = _inter_p

        x_dict = self.trfl(ms_feat_dict, inter_pred, back_fea) # multi-scale input
        return x_dict, inter_pred

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BATCHNORM
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class MLPHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MLPHead, self).__init__()

        self.linear_pred = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.Conv2d(256, num_classes, kernel_size=1)
            )

    def forward(self, x):
        return self.linear_pred(x) 