import math
from tkinter import Scale

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import NECKS

@NECKS.register_module()
class RayEmbeds(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 ):
        super().__init__()
        self.embed_dims = embed_dims
        self.ray_dims = embed_dims // 4
                
        self.ray_mlp = nn.Sequential(
            nn.Conv2d(6,
                      embed_dims,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True),
            nn.BatchNorm2d(embed_dims),
            nn.GELU(),
            
            nn.Conv2d(embed_dims,
                      self.ray_dims,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True),
            nn.BatchNorm2d(self.ray_dims),
            nn.GELU()
        )
        
    def calCoordinateFrom2PointsAndPlane(self, point, denorm):    
        pos = -1 * point * denorm[3] / np.inner(denorm[:3], np.array(point)).reshape(point.shape[0], 1)
        return pos 

    def decode_location(self, P, point2d, depth):
        P_inv = np.linalg.inv(P)
        point_extend = np.hstack([point2d[:, 0].reshape(point2d.shape[0], 1), point2d[:, 1].reshape(point2d.shape[0], 1), np.ones(point2d.shape[0]).reshape(point2d.shape[0], 1)])
        point_extend = point_extend * depth
        
        point_extend = np.hstack([point_extend, np.ones(point_extend.shape[0]).reshape(point_extend.shape[0], 1)])
        locations = np.matmul(P_inv, point_extend.T)
        locations = locations[:3]
        return locations
        
    def generate_ground_locs(self, P, denorm, point2d):
        locs = self.decode_location(P, point2d, 10).T
        ground_locs = self.calCoordinateFrom2PointsAndPlane(locs, denorm)
        ground_locs = ground_locs[:, :3]
        return ground_locs
    
    def generate_locs(self, P, denorm, point2d):
        locs = self.decode_location(P, point2d, 100).T
        return locs
    
    
    def locs2rk_matrix(self, ground_locs, lidar2cam, ft_shape):
        T = lidar2cam[:3, 3]
        cam2lidar = np.linalg.inv(lidar2cam)
        ground_locs_extend = np.concatenate((ground_locs, np.ones((ground_locs.shape[0], 1))), axis=1)
        ground_locs_lidar = np.matmul(cam2lidar, ground_locs_extend.T).T
        locs_map = ground_locs_lidar[:, :3].reshape(ft_shape[0], ft_shape[1], 3)
        locs_map = np.concatenate((locs_map, np.ones((locs_map.shape[0], locs_map.shape[1], 3))), axis=2)
        locs_map[:, :, 3:] = T
        return locs_map
    
    def reweight_intrinsic(self, cam_intrinsic, scale=8/320):
        cam_intrinsic[:,:,:2,:] *= scale
        return cam_intrinsic
    
    def equation_plane(self, points): 
        x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
        x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
        x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1
        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = (- a * x1 - b * y1 - c * z1)
        return [a, b, c, d]
    
    def get_denorm(self, lidar2cam):
        ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
        ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
        denorm = self.equation_plane(ground_points_cam)
        return denorm
    
    def get_point2d(self, ft_shape, B, N):
        H, W = ft_shape[0], ft_shape[1]
        xx1, yy1 = np.meshgrid(np.arange(0, W, 1), np.arange(0, H, 1))
        xx1 = xx1.reshape((H, W, 1))
        yy1 = yy1.reshape((H, W, 1))
        point2d = np.concatenate((xx1, yy1), axis=2).reshape(1, 1, -1, 2)
        point2d = np.repeat(point2d, B, axis=0)
        point2d = np.repeat(point2d, N, axis=1)   # [B, N, -1, 2]
        return point2d
        
    def forward(self, mlvl_feats, img_metas):        
        img_features = mlvl_feats[0]
        B, N, C, H, W = img_features.shape
        ft_shape = (H, W)
        img_features = img_features.view(B * N, C, H, W)
        
        lidar2cam = []
        for img_meta in img_metas:
            lidar2cam.append(img_meta['lidar2cam'])
        lidar2cam = np.asarray(lidar2cam)   # [B, N, 4, 4]
        cam_intrinsic = []
        for img_meta in img_metas:
            cam_intrinsic.append(img_meta['cam_intrinsic'])
        cam_intrinsic = np.asarray(cam_intrinsic)
        cam_intrinsic = self.reweight_intrinsic(cam_intrinsic)  # [B, N, 4, 4]
        point2d = self.get_point2d(ft_shape, B, N)
        
        loc_maps = []
        for b in range(B):
            temp = []
            for n in range(N):
                denorm = self.get_denorm(lidar2cam[b, n, ...])
                ground_locs = self.generate_ground_locs(cam_intrinsic[b, n, ...], denorm, point2d[b, n, ...])
                loc_map = self.locs2rk_matrix(ground_locs, lidar2cam[b, n, ...], ft_shape)
                temp.append(loc_map)
            loc_maps.append(temp)
        loc_maps = np.asarray(loc_maps)   # [B, N, 4, 4]
        
        ###  visual  ###
        '''
        filename =  img_metas[0]['filename'][0]
        image_demo = cv2.imread(filename)
        image_demo = cv2.resize(image_demo, (1536, 864))
        lidar2img = img_metas[0]['lidar2img']
        loc_maps_demo = loc_maps.reshape(-1, 6)[:, :3]
        points_extend = np.concatenate((loc_maps_demo, np.ones((loc_maps_demo.shape[0], 1))), axis=1)
        points_img = np.matmul(lidar2img, points_extend.T).T
        points_uv = points_img[:, :2] / points_img[:, 2][:, np.newaxis]
        points_uv = points_uv.astype(np.int32)
        points_uv[:,0] = np.clip(points_uv[:,0], 0, image_demo.shape[1]-1)
        points_uv[:,1] = np.clip(points_uv[:,1], 0, image_demo.shape[0]-1)
        image_demo[points_uv[:,1], points_uv[:,0]] = (255,0,0)
        for idx in range(points_uv.shape[0]):
            if idx % 10 != 0:
                continue
            org = (points_uv[idx,0][0], points_uv[idx,1][0]) 
            text = str(int(loc_maps_demo[idx,0]))
            image_demo = cv2.putText(image_demo, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite("image_demo.jpg", image_demo)
        '''
        ray_input = img_features.new_tensor(loc_maps)
        ray_input = ray_input.permute(0,1,4,2,3).contiguous()
        ray_input = ray_input.view(B*N, -1, H, W)
        
        ray_embeds = self.ray_mlp(ray_input)
        img_features = torch.cat((img_features, ray_embeds), dim=1).view(B, N, -1, H, W)
        
        return [img_features]