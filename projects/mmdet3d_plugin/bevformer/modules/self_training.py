import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms, roi_align, roi_pool

from mmdet3d.models.builder import NECKS

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

@NECKS.register_module()
class SelfTraining(nn.Module):
    def __init__(self,
                 in_dim=256,
                 proj_hidden_dim=2048,
                 pred_hidden_dim=512,
                 out_dim=2048,
                 pc_range=[],
                 bev_h=150,
                 bev_w=150
                 ):
        super().__init__()
        self.in_dim = in_dim
        
        self.projector = nn.Sequential(
                nn.Linear(in_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        
        self.predictor = self.layer1 = nn.Sequential(
            nn.Linear(out_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, out_dim)
        )
        
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.grid_length = [self.real_h / self.bev_h, self.real_w / self.bev_w]

    def forward(self, bev_embed, gt_bboxes_list): 
        bev_embed = bev_embed.permute(1, 0, 2).contiguous()
        bs = bev_embed.shape[0]
        bev_embed = bev_embed.view(bs, self.bev_h, self.bev_w, -1)
        bev_embed = bev_embed.permute(0, 3, 1, 2).contiguous()
        
        device = gt_bboxes_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list] 
        
        gt_bboxes_numpy_list = []
        for gt_bboxes in gt_bboxes_list:
            gt_bboxes_numpy = gt_bboxes.cpu().numpy()[np.newaxis,:,:]
            gt_bboxes_numpy_list.append(gt_bboxes_numpy)
        gt_bboxes_numpy = np.concatenate(gt_bboxes_numpy_list, axis=0) 
        
        gt_bboxes_numpy = gt_bboxes_numpy.reshape(-1, 9)
        print("gt_bboxes_numpy: ", gt_bboxes_numpy.shape)
        gt_locs = gt_bboxes_numpy[:, :3]
        proj_ps = self.point2bevpixel(gt_locs)
        proj_ps = torch.from_numpy(proj_ps).to(device=bev_embed.device)
        proj_ps = proj_ps.view(bs, -1, 2)
        
        batch_id = torch.arange(bs, dtype=torch.float, device=bev_embed.device).unsqueeze(1)
        roi_id = batch_id.repeat(1, proj_ps.shape[1]).view(-1, 1)
        proj_rois = torch.cat([proj_ps - 2, proj_ps + 2], dim=-1)
        proj_rois = proj_rois.reshape(-1, proj_rois.shape[-1])
        proj_rois = torch.cat([roi_id, proj_rois], dim=-1)   
        
        print("proj_rois: ", proj_rois.shape)     
        features_rois = roi_align(bev_embed, proj_rois, output_size=[1,1], spatial_scale=1, sampling_ratio=1)
        features_rois = features_rois.view(bs, -1, features_rois.shape[1])

        x1, x2 = features_rois[0], features_rois[1]
        z1, z2 = self.projector(x1), self.projector(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        
        loss = D(p1, z2) / 2 + D(p2, z1) / 2
        return loss
    
    def point2bevpixel(self, points):
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        pixels_w = (points[:, 0] - self.pc_range[0]) / self.grid_length[1]
        pixels_h = (points[:, 1] - self.pc_range[1]) / self.grid_length[0]
        pixels = np.concatenate((pixels_h[:, np.newaxis], pixels_w[:, np.newaxis]), axis=-1)
        pixels = pixels.astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, self.bev_h-1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, self.bev_w-1)
        return pixels
    
    def object_corners(self, locs, lwh, yaw):        
        l, w = lwh[:, 0][:, np.newaxis], lwh[:, 1][:, np.newaxis]
        x_corners = np.concatenate((l / 2, l / 2, -l / 2, -l / 2), axis=-1)[:,:,np.newaxis]
        y_corners = np.concatenate((w / 2, -w / 2, -w / 2, w / 2), axis=-1)[:,:,np.newaxis]
        corners = np.concatenate((x_corners, y_corners), axis=-1)
        locs = np.repeat(locs[:, np.newaxis, ], [4], axis=1)
        yaw = np.repeat(yaw[:, np.newaxis], [4], axis=1)
        sinyaw, cosyaw = np.sin(yaw), np.cos(yaw)
        corners_x = corners[:,:,0] * sinyaw - corners[:,:,1] * cosyaw
        corners_y = corners[:,:,0] * sinyaw + corners[:,:,1] * cosyaw
        corners = np.concatenate((corners_x[:,:,np.newaxis], corners_y[:,:,np.newaxis]), axis=-1)
        corners = corners + locs[:,:,:2]
        return corners   # [num, 4, 2]