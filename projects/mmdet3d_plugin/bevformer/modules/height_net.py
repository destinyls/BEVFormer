import math
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from mmdet3d.models.builder import NECKS

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
        self.bin_space = (max_height - min_height) / num_bins
        self.height_head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(),
            nn.Conv2d(embed_dims, embed_dims, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU())
        self.height_classifier = nn.Conv2d(embed_dims, self.num_bins + 1, kernel_size=(1, 1))
        self.emsemble_layer = nn.Sequential(
            nn.Conv2d(self.num_bins + 1 + embed_dims, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
    def forward(self, mlvl_feats, img_metas, is_train=False):
        img_features = mlvl_feats[0]
        B, N, C, H, W = img_features.shape
        img_features = img_features.view(B * N, C, H, W)
        height_features = self.height_head(img_features)
        height_logits = self.height_classifier(height_features)
        height_probs = F.softmax(height_logits, dim=1) 
        emsembled_features = torch.cat((img_features, height_probs), dim=1)
        emsembled_features = self.emsemble_layer(emsembled_features).view(B, N, -1, H, W)

        if is_train:
            height_probs = self.aligned_bilinear(height_probs, factor=self.in_strides[0], offset="none").squeeze(1)
            height_probs = height_probs.permute(0, 2, 3, 1).contiguous().view(-1, self.num_bins + 1)
            gt_heights, height_mask = self.bin_heights(img_metas, emsembled_features.device)
            depth_loss = (F.binary_cross_entropy(
                    height_probs[height_mask],
                    gt_heights[height_mask],
                    reduction='none',
                ).sum() / max(1.0, height_mask.sum()))
            return [emsembled_features], depth_loss
        return [emsembled_features]
     
    def aligned_bilinear(self, tensor, factor, offset="none"):
        """Adapted from AdelaiDet:
        https://github.com/aim-uofa/AdelaiDet/blob/master/adet/utils/comm.py
        """
        assert tensor.dim() == 4
        assert factor >= 1
        assert int(factor) == factor

        if factor == 1:
            return tensor
        h, w = tensor.size()[2:]
        tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
        oh = factor * h + 1
        ow = factor * w + 1
        tensor = F.interpolate(tensor, size=(oh, ow), mode='bilinear', align_corners=True)
        if offset == "half":
            tensor = F.pad(tensor, pad=(factor // 2, 0, factor // 2, 0), mode="replicate")
        return tensor[:, :, :oh - 1, :ow - 1]
    
    def bin_heights(self, img_metas, device, mode="UD", target=True):
        """
        Converts height map into bin indices
        Args:
            img_metas contains height_map and height_mask
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            gt_heights [torch.Tensor(N*N*H*W, num_bins)]: Depth bin indices
        """
        
        height_map, height_mask = [], []
        for img_meta in img_metas:
            height_map_batch, height_mask_batch = [], []
            for idx in range(len(img_meta['height_map'])):
                height_map_batch.append(img_meta['height_map'][idx])
                height_mask_batch.append(img_meta['height_mask'][idx])
            height_map.append(height_map_batch)
            height_mask.append(height_mask_batch)
        height_map = np.array(height_map)
        height_mask = np.array(height_mask)
        height_map = torch.tensor(height_map, dtype=torch.float32, device=device)
        height_mask = torch.tensor(height_mask, dtype=torch.bool, device=device)
        
        mask = (height_map == 0.0)
        height_map = torch.clamp(height_map, self.min_height, self.max_height)
        if mode == "UD":
            bin_size = (self.max_height - self.min_height) / self.num_bins
            indices = ((height_map - self.min_height) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (self.max_height - self.min_height) / (self.num_bins * (1 + self.num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (height_map - self.min_height) / bin_size)
        elif mode == "SID":
            indices = self.num_bins * (torch.log(1 + height_map) - math.log(1 + self.min_height)) / \
                      (math.log(1 + self.max_height) - math.log(1 + self.min_height))
        else:
            raise NotImplementedError
        if target:
            indices = torch.ceil(indices)
            indices = torch.clamp(indices, 1, self.num_bins)
            indices[mask] = 0 
            gt_heights = F.one_hot(indices.long(), num_classes=self.num_bins + 1).view(-1, self.num_bins + 1)
        return gt_heights.float(), height_mask.flatten()