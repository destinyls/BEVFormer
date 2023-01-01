import cv2
import math
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

    def forward(self, bev_embed, gt_bboxes_list, sample_idx): 
        bs = len(gt_bboxes_list)
        ids1 = np.arange(0, bs, 2)
        ids2 = np.arange(1, bs + 1, 2)  
        
        bev_embed = bev_embed.permute(1, 0, 2).contiguous()
        bev_embed = bev_embed.view(bs, self.bev_h, self.bev_w, -1)
        features1, features2 = bev_embed[ids1], bev_embed[ids2]
        
        bbox_mask = self.get_bbox_mask(gt_bboxes_list, resolution=0.2)
        
        bbox_mask_demo = bbox_mask.astype(np.int32)[0] * 255
        bbox_mask_demo = bbox_mask_demo[:,:,np.newaxis]
        bbox_mask_demo = np.repeat(bbox_mask_demo, 3, axis=2)
        cv2.imwrite(os.path.join("bbox_mask_demo", str(sample_idx) + ".jpg"), bbox_mask_demo)
        
        bbox_mask = torch.from_numpy(bbox_mask).to(device=bev_embed.device)
        bbox_mask = bbox_mask.float()  # [B, H, W]
        x1, x2 = features1 * bbox_mask.unsqueeze(-1), features2 * bbox_mask.unsqueeze(-1)
        x1, x2 = x1.view(-1, x1.shape[-1]), x2.view(-1, x2.shape[-1])
        loss = MSE(x1, x2) / 2 + MSE(x2, x1) / 2
        
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
    
    def bev_voxels(self, num_voxels):
        u, v = np.ogrid[0:num_voxels[0], 0:num_voxels[1]]
        uu, vv = np.meshgrid(u, v, sparse=False)
        voxel_size = np.array([self.bev_h / num_voxels[0], self.bev_w / num_voxels[1]])
        uv = np.concatenate((uu[:,:,np.newaxis], vv[:,:,np.newaxis]), axis=-1)
        uv = uv * voxel_size + 0.5 * voxel_size
        return uv.astype(np.float32)
    
    def local2global(self, points, center_lidar, yaw_lidar):
        points_3d_lidar = points.reshape(-1, 3)
        rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                            [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                            [0, 0, 1]])
        points_3d_lidar = np.matmul(rot_mat, points_3d_lidar.T).T + center_lidar
        return points_3d_lidar
    
    def get_bbox_mask(self, gt_bboxes_list, resolution=0.5):
        bbox_mask = np.ones((len(gt_bboxes_list), self.bev_h, self.bev_w), dtype=np.float) * 0.1
        
        device = gt_bboxes_list[0].device
        gt_boxes = [torch.cat(
            (gt_bbox.gravity_center, gt_bbox.tensor[:, 3:]),
            dim=1).to(device) for gt_bbox in gt_bboxes_list] 
        for batch_id in range(len(gt_boxes)):
            gt_bbox = gt_boxes[batch_id].cpu().numpy()
            if gt_bbox.shape[0] == 0:
                continue
            for obj_id in range(gt_bbox.shape[0]):
                loc, lwh, rot_y = gt_bbox[obj_id, :3], gt_bbox[obj_id, 3:6], gt_bbox[obj_id, 6]
                shape = np.array([lwh[1] / resolution, lwh[0] / resolution]).astype(np.int32)
                n, m = [(ss - 1.) / 2. for ss in shape]
                x, y = np.ogrid[-m:m + 1, -n:n + 1]
                xv, yv = np.meshgrid(x, y, sparse=False)
                xyz = np.concatenate((xv[:,:,np.newaxis], yv[:,:,np.newaxis], np.ones_like(xv)[:,:,np.newaxis]), axis=-1)
                obj_points = self.local2global(xyz * resolution, loc, rot_y)
                obj_pixels = self.point2bevpixel(obj_points)
                bbox_mask[batch_id, obj_pixels[:, 1], obj_pixels[:, 0]] = 1.0
        return bbox_mask
    
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