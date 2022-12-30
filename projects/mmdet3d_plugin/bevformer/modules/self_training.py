import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms, roi_align, roi_pool

from mmdet3d.models.builder import NECKS

def D(p, z, version='original'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception
    
def MSE(p, z, reduction="mean"):
    return F.mse_loss(p, z.detach(), reduction=reduction) 

@NECKS.register_module()
class SelfTraining(nn.Module):
    def __init__(self,
                 in_dim=256,
                 proj_hidden_dim=2048,
                 pred_hidden_dim=512,
                 out_dim=256,
                 pc_range=[],
                 bev_h=150,
                 bev_w=150
                 ):
        super().__init__()
        self.in_dim = in_dim
        
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.grid_length = [self.real_h / self.bev_h, self.real_w / self.bev_w]
                
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, out_dim)
        )

    def forward(self, bev_embed, gt_bboxes_list, sample_idx): 
        
        bs = len(gt_bboxes_list)
        ids1 = np.arange(0, bs, 2)
        ids2 = np.arange(1, bs + 1, 2)  
        
        bev_embed = bev_embed.permute(1, 0, 2).contiguous()
        bev_embed = bev_embed.view(bs, self.bev_h, self.bev_w, -1)
        bev_embed = bev_embed.permute(0, 3, 1, 2).contiguous()

        pixel_points = self.bev_voxels(num_voxels=[50, 50])
        pixel_points = torch.from_numpy(pixel_points).to(device=bev_embed.device)
        pixel_points = pixel_points.view(1, -1, 2).repeat(bs, 1, 1)
        pixel_rois = torch.cat([pixel_points - 1, pixel_points + 1], dim=-1)
        batch_id = torch.arange(bs, dtype=torch.float, device=bev_embed.device).unsqueeze(1)
        batch_id = batch_id.repeat(1, pixel_rois.shape[1]).view(-1, 1)
        pixel_rois = torch.cat([batch_id, pixel_rois.view(-1, 4)], dim=-1)
        features_pixel_rois = roi_align(bev_embed, pixel_rois, output_size=[1,1], spatial_scale=1, sampling_ratio=1)
        features_pixel_rois = features_pixel_rois.view(bs, -1, features_pixel_rois.shape[1])
        
        x1, x2 = features_pixel_rois[ids1], features_pixel_rois[ids2]
        x1 = x1.view(-1, x1.shape[-1])
        x2 = x2.view(-1, x2.shape[-1])
        z1, z2 = self.predictor(x1), self.predictor(x2)
        loss = D(x1, z2) / 2 + D(z1, x2) / 2
        
        return loss
    
    def point2bevpixel(self, points):
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        pixels_w = (points[:, 0] - self.pc_range[0]) / self.grid_length[1]
        pixels_h = (points[:, 1] - self.pc_range[1]) / self.grid_length[0]
        pixels = np.concatenate((pixels_w[:, np.newaxis], pixels_h[:, np.newaxis]), axis=-1)
        pixels = pixels.astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, self.bev_w-1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, self.bev_h-1)
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
    
    def bev_voxels(self, num_voxels):
        u, v = np.ogrid[0:num_voxels[0], 0:num_voxels[1]]
        uu, vv = np.meshgrid(u, v, sparse=False)
        voxel_size = np.array([self.bev_h / num_voxels[0], self.bev_w / num_voxels[1]])
        uv = np.concatenate((uu[:,:,np.newaxis], vv[:,:,np.newaxis]), axis=-1)
        uv = uv * voxel_size + 0.5 * voxel_size
        return uv.astype(np.float32)
    
    def box_level_ssl(self, bev_embed, gt_bboxes_list):
        bs = bev_embed.shape[0]
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
        
        gt_locs = gt_bboxes_numpy[:, :3]
        proj_ps = self.point2bevpixel(gt_locs)
        '''
        bev_embed_debug = torch.sum(bev_embed[0], dim=0).detach().cpu().numpy() * 1000
        bev_embed_debug[proj_ps[:,1], proj_ps[:,0]] = 255
        cv2.imwrite(os.path.join("demo_bev", sample_idx + ".jpg"), bev_embed_debug)
        '''     
        proj_ps = torch.from_numpy(proj_ps).to(device=bev_embed.device)
        proj_ps = proj_ps.view(bs, -1, 2)
        
        batch_id = torch.arange(bs, dtype=torch.float, device=bev_embed.device).unsqueeze(1)
        roi_id = batch_id.repeat(1, proj_ps.shape[1]).view(-1, 1)
        proj_rois = torch.cat([proj_ps - 2, proj_ps + 2], dim=-1)
        proj_rois = proj_rois.reshape(-1, proj_rois.shape[-1])
        proj_rois = torch.cat([roi_id, proj_rois], dim=-1)   
        
        features_rois = roi_align(bev_embed, proj_rois, output_size=[1,1], spatial_scale=1, sampling_ratio=1)
        features_rois = features_rois.view(bs, -1, features_rois.shape[1])

        x1, x2 = features_rois[0], features_rois[1]
        if x1.shape[0] == 1:
            x1 = x1.repeat(2, 1)
            x2 = x2.repeat(2, 1)
        z1, z2 = self.projector(x1), self.projector(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        
        loss = D(p1, z2) / 2 + D(p2, z1) / 2
        return loss