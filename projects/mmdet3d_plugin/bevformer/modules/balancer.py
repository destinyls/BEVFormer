import torch
import torch.nn as nn

# based on
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/balancer.py


class Balancer(nn.Module):
    def __init__(self, fg_weight, bg_weight):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight

    def forward(self, loss, bg_mask):
        """
        Forward pass
        Args:
            loss [torch.Tensor(B, H, W)]: Pixel-wise loss
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
        Returns:
            loss [torch.Tensor(1)]: Total loss after foreground/background balancing
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        fg_mask = ~bg_mask
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()

        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels

        loss = fg_loss + bg_loss
        return loss