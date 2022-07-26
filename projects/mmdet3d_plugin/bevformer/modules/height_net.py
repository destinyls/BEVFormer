import math

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_conv_layer
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models.builder import NECKS

from .balancer import Balancer
from .focal_loss import FocalLoss

BN_MOMENTUM = 0.1

@NECKS.register_module()
class HeightNet(nn.Module):
    def __init__(self,
                 in_channels=256,
                 in_strides=[32,],
                 out_channels=256,
                 embed_dims=256,
                 min_height=-3.0,
                 max_height=4.0,
                 num_bins=14):
        super().__init__()
        self.in_strides = in_strides
        self.min_height = min_height
        self.max_height = max_height
        self.num_bins = num_bins
        self.downsample_factor = in_strides[0]
        self.inplanes = in_channels
        self.deconv_with_bias = False
        self.loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        self.balancer = Balancer(fg_weight=1.0, bg_weight=0.8)

        self.height_conv = nn.Sequential(
            BasicBlock(embed_dims, embed_dims),
            BasicBlock(embed_dims, embed_dims),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
        )
        self.init_conv(self.height_conv)
        
        self.deconv_layers = self._make_deconv_layer(embed_dims, 4)
        self.init_deconv(self.deconv_layers)
        self.height_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      embed_dims,
                      kernel_size=3,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims,
                      self.num_bins + 1,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )
        self.height_head[-1].bias.data.fill_(-2.19)
        
    def init_conv(self, layer):
        for _, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
    def init_deconv(self, layer):
        for _, m in layer.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters, num_kernels):
        layers = []
        kernel, padding, output_padding = self._get_deconv_cfg(num_kernels)
        planes = num_filters
        layers.append(
            nn.ConvTranspose2d(
                in_channels=self.inplanes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias))
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        self.inplanes = planes

        return nn.Sequential(*layers)
        
    def forward(self, mlvl_feats, img_metas, is_train=False):
        img_features = mlvl_feats[0]
        B, N, C, H, W = img_features.shape
        height_embed = self.height_conv(img_features.view(B * N, C, H, W))
        up_level = self.deconv_layers(height_embed)
        height_logits = self.height_head(up_level)  # [B*N, C, H, W]

        if is_train:
            # [B*N, H, W]
            gt_heights, bg_mask = self.get_downsampled_gt_height(img_metas, img_features.device)
            # Compute loss
            height_loss = self.loss_func(height_logits, gt_heights)
            # Compute foreground/background balancing
            height_loss = self.balancer(loss=height_loss, bg_mask=bg_mask)
            return [img_features], height_loss
        return [img_features]
    
    def get_downsampled_gt_height(self, img_metas, device, mode="UD", target=True):
        gt_heights = []
        for img_meta in img_metas:
            gt_heights_batch = []
            for idx in range(len(img_meta['height_map'])):
                gt_heights_batch.append(img_meta['height_map'][idx])
            gt_heights.append(gt_heights_batch)
        gt_heights = np.array(gt_heights)
        gt_heights = torch.tensor(gt_heights, dtype=torch.float32, device=device)
        
        B, N, H, W = gt_heights.shape
        gt_heights = gt_heights.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_heights = gt_heights.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_heights = gt_heights.view(-1, self.downsample_factor * self.downsample_factor)
        gt_heights_tmp = torch.where(gt_heights == 0.0,
                                    1e5 * torch.ones_like(gt_heights),
                                    gt_heights)
        gt_heights = torch.min(gt_heights_tmp, dim=-1).values
        gt_heights = gt_heights.view(B * N, H // self.downsample_factor, W // self.downsample_factor)
        bg_mask = (gt_heights == 1e5)

        gt_heights = torch.clamp(gt_heights, self.min_height, self.max_height)
        if mode == "UD":
            bin_size = (self.max_height - self.min_height) / self.num_bins
            indices = ((gt_heights - self.min_height) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (self.max_height - self.min_height) / (self.num_bins * (1 + self.num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (gt_heights - self.min_height) / bin_size)
        elif mode == "SID":
            indices = self.num_bins * (torch.log(1 + gt_heights) - math.log(1 + self.min_height)) / \
                      (math.log(1 + self.max_height) - math.log(1 + self.min_height))
        else:
            raise NotImplementedError
        indices = torch.ceil(indices)
        indices = torch.clamp(indices, 1, self.num_bins)
        indices[bg_mask] = 0
        indices = indices.type(torch.int64)
        return indices, bg_mask    # [B*N, H, W]
