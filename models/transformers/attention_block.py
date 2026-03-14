from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange as o_rearrange
from einops.layers.torch import Rearrange
from utils.utils import to_2tuple
from timm.models.layers import DropPath, trunc_normal_
import pdb

BATCHNORM = nn.SyncBatchNorm # nn.BatchNorm2d
# BATCHNORM = nn.BatchNorm2d

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
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

class SelfAttention(nn.Module):
    def __init__(self,
                 fea_no,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=True,
                 q_method='dw_bn',
                 kv_method='avg',
                 kernel_size_q=3,
                 kernel_size_kv=3,
                 stride_kv=2,
                 stride_q=2,
                 padding_kv=1,
                 padding_q=1
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.fea_no = fea_no

        self.conv_proj_q = self._build_single_projection(
            dim_in, kernel_size_q, padding_q,
            stride_q, q_method
        )
        self.conv_proj_k = self._build_single_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, kv_method
        )
        self.conv_proj_v = self._build_single_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, kv_method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def _build_single_projection(self,
                                 dim_in,
                                 kernel_size,
                                 padding,
                                 stride,
                                 method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', BATCHNORM(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c'))]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c'))]))

        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))
        return proj

    def forward_conv(self, x, h ,w):

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        return q, k, v

    def forward(self, x, h, w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale

        return attn_score, v

class CommonAttention(nn.Module):
    def __init__(self,
                 fea_no,
                 dim_out,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.
                 ):
        super().__init__()
        self.dim = dim_out
        self.num_heads = num_heads
        self.fea_no = fea_no

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        self.fuse_attn = nn.Conv2d(num_heads * fea_no, num_heads, 1)

    def forward(self, attn_score, v):

        attn_score = self.fuse_attn(torch.cat(attn_score, dim=1))

        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class AttnBlock(nn.Module):

    def __init__(self,
                 task_no,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 stride_q=2):
        super().__init__()

        self.stride_q = stride_q
        self.embed_dim = dim_in
        self.task_no = task_no

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(self.embed_dim)
        self.attn_production = SelfAttention(
                fea_no=task_no,
                dim_in=self.embed_dim,
                dim_out=self.embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                q_method='dw_bn',
                kv_method='avg',
                kernel_size_q=3,
                kernel_size_kv=3,
                stride_kv=2,
                stride_q=stride_q,
                padding_kv=1,
                padding_q=1
        )

    def forward(self, x):

        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        res = x
        x = self.norm1(x)
        attn_score, v = self.attn_production(x, h, w)

        return attn_score, [v, res, h, w]

class FusionBlock(nn.Module):

    def __init__(self,
                 task_no,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 stride_q=2):
        super().__init__()

        self.stride_q = stride_q
        self.embed_dim = dim_in
        self.task_no = task_no

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(self.embed_dim)
        dim_mlp_hidden = int(self.embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )
        self.attn_fusion = CommonAttention(
            fea_no=task_no,
            dim_out=self.embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop
        )

    def forward(self, attn_scores, params):
        v, res, h, w = params

        attn = self.attn_fusion(attn_scores, v)

        # interpolate output of attention to previous resolution
        attn = rearrange(attn, 'b (h w) c -> b c h w', h=h // self.stride_q, w=w // self.stride_q)
        attn = F.interpolate(attn, size=(h, w), mode='bilinear', align_corners=False)
        attn = rearrange(attn, 'b c h w -> b (h w) c', h=h, w=w)

        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x

