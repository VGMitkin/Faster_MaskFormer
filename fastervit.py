import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg, _update_default_kwargs
from pathlib import Path
from utils import *
from dataclasses import dataclass
from typing import Optional, Tuple


class FasterViTLayer(nn.Module):
    """
    FasterViT layer with hierarchical attention.
    """

    def __init__(self,
                 dim,
                 depth,
                 input_resolution,
                 num_heads,
                 window_size,
                 ct_size=1,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 only_local=False,
                 hierarchy=True,
                 do_propagation=False
                 ):
        """
        Args:
            dim: feature size dimension.
            depth: layer depth.
            input_resolution: input resolution.
            num_heads: number of attention head.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window.
            conv: conv_based stage flag.
            downsample: downsample flag.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            layer_scale: layer scale coefficient.
            layer_scale_conv: conv layer scale coefficient.
            only_local: local attention flag.
            hierarchy: hierarchical attention flag.
            do_propagation: enable carrier token propagation.
        """
        super().__init__()
        self.conv = conv
        self.transformer_block = False
        sr_ratio=1
        H = input_resolution[0] + (window_size - input_resolution[0] % window_size) % window_size
        W = input_resolution[1] + (window_size - input_resolution[1] % window_size) % window_size
        input_resolution = [H,W]
        if conv:
            self.blocks = nn.ModuleList([
                ConvBlock(dim=dim,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          layer_scale=layer_scale_conv)
                for i in range(depth)])
            self.transformer_block = False
        else:
            sr_ratio = [ input_resolution[0] // window_size if not only_local else 1, input_resolution[1] // window_size if not only_local else 1]
            self.blocks = nn.ModuleList([
                HAT(dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    sr_ratio=sr_ratio,
                    window_size=window_size,
                    last=(i == depth-1),
                    layer_scale=layer_scale,
                    ct_size=ct_size,
                    do_propagation=do_propagation,
                    )
                for i in range(depth)])
            self.transformer_block = True
        self.downsample = None if not downsample else Downsample(dim=dim)
        if len(self.blocks) and not only_local and sr_ratio and hierarchy and not self.conv:
            self.global_tokenizer = TokenInitializer(dim,
                                                     input_resolution,
                                                     window_size,
                                                     ct_size=ct_size)
            self.do_gt = True
        else:
            self.do_gt = False

        self.window_size = window_size
        self.ln = nn.LayerNorm(dim*2)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
        ct = self.global_tokenizer(x) if self.do_gt else None
        if self.transformer_block:
            x = window_partition(x, self.window_size)
        for bn, blk in enumerate(self.blocks):
            x, ct = blk(x, ct)
        if self.transformer_block:
            x = window_reverse(x, self
                               .window_size, Hp, Wp, B)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is not None:
            x = self.downsample(x)
        x = x.permute(0, 2, 3, 1).view(B, -1, C*2)
        x = self.ln(x).permute(0, 2, 1).view(B, C*2, H//2, -1)
        return x
        


class FasterViT(nn.Module):
    """
    FasterViT module
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 ct_size,
                 mlp_ratio,
                 num_heads,
                 resolution=[224, 224],
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 layer_norm_last=False,
                 hat=[False, False, True, False],
                 do_propagation=False,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            in_dim: inner-plane feature size dimension.
            depths: layer depth.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window.
            mlp_ratio: MLP ratio.
            num_heads: number of attention head.
            resolution: image resolution.
            drop_path_rate: drop path rate.
            in_chans: input channel dimension.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            layer_scale: layer scale coefficient.
            layer_scale_conv: conv layer scale coefficient.
            layer_norm_last: last stage layer norm flag.
            hat: hierarchical attention flag.
            do_propagation: enable carrier token propagation.
        """
        super().__init__()
        if type(resolution)!=tuple and type(resolution)!=list:
            resolution = [resolution, resolution]

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        if hat is None: hat = [True, ]*len(depths)
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = FasterViTLayer(dim=int(dim * 2 ** i),
                                   depth=depths[i],
                                   num_heads=num_heads[i],
                                   window_size=window_size[i],
                                   ct_size=ct_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   conv=conv,
                                   drop=drop_rate,
                                   attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                   downsample=(i < 4),
                                   layer_scale=layer_scale,
                                   layer_scale_conv=layer_scale_conv,
                                   input_resolution=[int(2 ** (-1 - i) * resolution[0]), 
                                                     int(2 ** (-1 - i) * resolution[1])],
                                   only_local=not hat[i],
                                   do_propagation=do_propagation)
            self.levels.append(level)
        self.norm = LayerNorm2d(num_features) if layer_norm_last else nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self.channels=[128,256,512,1024]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        self.hidden_states=[]
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
            self.hidden_states.append(x.clone())
        return x

    def forward(self, x):
        x = self.forward_features(x)           
        return BackboneOutput(feature_maps=self.hidden_states, hidden_states=self.hidden_states, attentions=None)
