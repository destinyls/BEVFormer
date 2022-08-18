import os
import cv2
import pickle
import time

import numpy as np

from tools.data_converter.extrinsic_analysis import *

def transform_with_M(image, M):
    image_new = np.zeros_like(image)
    u = range(image.shape[1])
    v = range(image.shape[0])
    xu, yv = np.meshgrid(u, v)
    uv = np.concatenate((xu[:,:,np.newaxis], yv[:,:,np.newaxis]), axis=2)
    uvd = np.concatenate((uv, np.ones((uv.shape[0], uv.shape[1], 1))), axis=-1) * 10
    uvd = uvd.reshape(-1, 3)
    uvd_new = np.matmul(M, uvd.T).T
    uv_new = uvd_new[:,:2] / (uvd_new[:,2][:, np.newaxis])
    
    uv_new_ceil = np.ceil(uv_new).astype(np.int32)
    uv_new_ceil[:,0] = np.clip(uv_new_ceil[:,0], 0, image.shape[1]-1)
    uv_new_ceil[:,1] = np.clip(uv_new_ceil[:,1], 0, image.shape[0]-1)
    uv_new_ceil_flatten = uv_new_ceil.reshape(-1, 2)
    uv_new_ceil_flatten = uv_new_ceil_flatten[:, 1] * image.shape[1] + uv_new_ceil_flatten[:, 0]
    
    uv_new_floor = np.floor(uv_new).astype(np.int32)
    uv_new_floor[:,0] = np.clip(uv_new_floor[:,0], 0, image.shape[1]-1)
    uv_new_floor[:,1] = np.clip(uv_new_floor[:,1], 0, image.shape[0]-1)
    uv_new_floor_flatten = uv_new_floor.reshape(-1, 2)
    uv_new_floor_flatten = uv_new_floor_flatten[:, 1] * image.shape[1] + uv_new_floor_flatten[:, 0]
    
    image_flatten = image.reshape(-1, 3)
    image_new_flatten = image_new.reshape(-1, 3)
    image_new_flatten[uv_new_floor_flatten,:] = image_flatten
    image_new_flatten[uv_new_ceil_flatten,:] = image_flatten
    image_new = image_new_flatten.reshape(image.shape[0], image.shape[1], image.shape[2])
    
    mask = (image_new_flatten[:, :] == [0, 0, 0])
    mask = np.all(mask, axis=1)
    mask = mask.reshape(image.shape[0], image.shape[1])
    image_new[mask, :] = np.concatenate((image_new[:,1:,:], np.zeros((image_new.shape[0], 1, 3))), axis=1)[mask]
    
    return image_new

def transform_with_M_bilinear_v1(image, M):
    u = range(image.shape[1])
    v = range(image.shape[0])
    xu, yv = np.meshgrid(u, v)
    uv = np.concatenate((xu[:,:,np.newaxis], yv[:,:,np.newaxis]), axis=2)
    uvd = np.concatenate((uv, np.ones((uv.shape[0], uv.shape[1], 1))), axis=-1) * 10
    uvd = uvd.reshape(-1, 3)
    
    M = np.linalg.inv(M)
    uvd_new = np.matmul(M, uvd.T).T
    uv_new = uvd_new[:,:2] / (uvd_new[:,2][:, np.newaxis])
    
    uv_new[:,0] = np.clip(uv_new[:,0], 0, image.shape[1]-1)
    uv_new[:,1] = np.clip(uv_new[:,1], 0, image.shape[0]-1)
    
    uv_Q00 = np.floor(uv_new).astype(np.int32)
    uv_Q11 = np.ceil(uv_new).astype(np.int32)
    uv_Q01 = np.concatenate((uv_Q11[:,0][:,np.newaxis], uv_Q00[:,1][:,np.newaxis]), axis=1)
    uv_Q10 = np.concatenate((uv_Q00[:,0][:,np.newaxis], uv_Q11[:,1][:,np.newaxis]), axis=1)

    uv_Q00_flatten = uv_Q00[:, 1] * image.shape[1] + uv_Q00[:, 0]
    uv_Q01_flatten = uv_Q01[:, 1] * image.shape[1] + uv_Q01[:, 0]
    uv_Q10_flatten = uv_Q10[:, 1] * image.shape[1] + uv_Q10[:, 0]
    uv_Q11_flatten = uv_Q11[:, 1] * image.shape[1] + uv_Q11[:, 0]
    
    image_flatten = image.reshape(-1, 3)
    f_Q00 = image_flatten[uv_Q00_flatten]
    f_Q01 = image_flatten[uv_Q01_flatten]
    f_Q10 = image_flatten[uv_Q10_flatten]
    f_Q11 = image_flatten[uv_Q11_flatten]
    
    ratio_0 = (uv_new[:, 0] - uv_Q00[:, 0])[:, np.newaxis]
    ratio_1 = (uv_new[:, 1] - uv_Q00[:, 1])[:, np.newaxis]
    
    f_R0 = f_Q00 + (f_Q01 - f_Q00) * ratio_0
    f_R1 = f_Q10 + (f_Q11 - f_Q10) * ratio_0
    f_R = f_R0 + (f_R1 - f_R0) * ratio_1
    image_new = (f_Q00 + f_Q10 + f_Q11 + f_Q01).reshape(image.shape[0], image.shape[1], image.shape[2])
    return image_new

def transform_with_M_bilinear_v2(image, M):
    u = range(image.shape[1])
    v = range(image.shape[0])
    xu, yv = np.meshgrid(u, v)
    uv = np.concatenate((xu[:,:,np.newaxis], yv[:,:,np.newaxis]), axis=2)
    uvd = np.concatenate((uv, np.ones((uv.shape[0], uv.shape[1], 1))), axis=-1) * 10
    uvd = uvd.reshape(-1, 3)
    
    M = np.linalg.inv(M)
    uvd_new = np.matmul(M, uvd.T).T
    uv_new = uvd_new[:,:2] / (uvd_new[:,2][:, np.newaxis])
    uv_new[:,0] = np.clip(uv_new[:,0], 0, image.shape[1]-1)
    uv_new[:,1] = np.clip(uv_new[:,1], 0, image.shape[0]-1)
    uv_new = uv_new.reshape(image.shape[0], image.shape[1], 2)

    uv_Q00 = np.floor(uv_new).astype(np.int32)
    uv_Q11 = np.ceil(uv_new).astype(np.int32)
    uv_Q01 = np.concatenate((uv_Q11[:,:,0][:,:,np.newaxis], uv_Q00[:,:,1][:,:,np.newaxis]), axis=2)
    uv_Q10 = np.concatenate((uv_Q00[:,:,0][:,:,np.newaxis], uv_Q11[:,:,1][:,:,np.newaxis]), axis=2)

    f_Q00 = image[uv_Q00[:,:,1], uv_Q00[:,:,0]]
    f_Q01 = image[uv_Q01[:,:,1], uv_Q01[:,:,0]]
    f_Q10 = image[uv_Q10[:,:,1], uv_Q10[:,:,0]]
    f_Q11 = image[uv_Q11[:,:,1], uv_Q11[:,:,0]]
    
    ratio_0 = (uv_new[:,:,0] - uv_Q00[:,:,0])[:,:,np.newaxis]
    ratio_1 = (uv_new[:,:,1] - uv_Q00[:,:,1])[:,:,np.newaxis]
    
    f_R0 = f_Q00 + (f_Q01 - f_Q00) * ratio_0
    f_R1 = f_Q10 + (f_Q11 - f_Q10) * ratio_0
    f_R = f_R0 + (f_R1 - f_R0) * ratio_1

    image_new = f_Q00
    return image_new

def transform_with_M_bilinear(image, M):
    u = range(image.shape[1])
    v = range(image.shape[0])
    xu, yv = np.meshgrid(u, v)
    uv = np.concatenate((xu[:,:,np.newaxis], yv[:,:,np.newaxis]), axis=2)
    uvd = np.concatenate((uv, np.ones((uv.shape[0], uv.shape[1], 1))), axis=-1) * 10
    uvd = uvd.reshape(-1, 3)
    
    M = np.linalg.inv(M)
    uvd_new = np.matmul(M, uvd.T).T
    uv_new = uvd_new[:,:2] / (uvd_new[:,2][:, np.newaxis])
    uv_new_mask = uv_new.copy()
    uv_new_mask = uv_new_mask.reshape(image.shape[0], image.shape[1], 2)
    
    uv_new[:,0] = np.clip(uv_new[:,0], 0, image.shape[1]-2)
    uv_new[:,1] = np.clip(uv_new[:,1], 0, image.shape[0]-2)
    uv_new = uv_new.reshape(image.shape[0], image.shape[1], 2)
    
    corr_x, corr_y = uv_new[:,:,1], uv_new[:,:,0]
    point1 = np.concatenate((np.floor(corr_x)[:,:,np.newaxis].astype(np.int32), np.floor(corr_y)[:,:,np.newaxis].astype(np.int32)), axis=2)
    point2 = np.concatenate((point1[:,:,0][:,:,np.newaxis], (point1[:,:,1]+1)[:,:,np.newaxis]), axis=2)
    point3 = np.concatenate(((point1[:,:,0]+1)[:,:,np.newaxis], point1[:,:,1][:,:,np.newaxis]), axis=2)
    point4 = np.concatenate(((point1[:,:,0]+1)[:,:,np.newaxis], (point1[:,:,1]+1)[:,:,np.newaxis]), axis=2)

    fr1 = (point2[:,:,1]-corr_y)[:,:,np.newaxis] * image[point1[:,:,0], point1[:,:,1], :] + (corr_y-point1[:,:,1])[:,:,np.newaxis] * image[point2[:,:,0], point2[:,:,1], :]
    fr2 = (point2[:,:,1]-corr_y)[:,:,np.newaxis] * image[point3[:,:,0], point3[:,:,1], :] + (corr_y-point1[:,:,1])[:,:,np.newaxis] * image[point4[:,:,0], point4[:,:,1], :]
    image_new = (point3[:,:,0] - corr_x)[:,:,np.newaxis] * fr1 + (corr_x - point1[:,:,0])[:,:,np.newaxis] * fr2
    
    mask_1 = np.logical_or(uv_new_mask[:,:,0] < 0, uv_new_mask[:,:,0] > image.shape[1] -2)
    mask_2 = np.logical_or(uv_new_mask[:,:,1] < 0, uv_new_mask[:,:,1] > image.shape[0] -2)
    mask = np.logical_or(mask_1, mask_2)
    image_new[mask] = [0,0,0]
    
    return image_new

def get_lidar2cam(info):
    cam_info = info["cams"]["CAM_FRONT"]    
    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    lidar2cam_t = cam_info[
        'sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = cam_info['cam_intrinsic']
    viewpad = np.eye(4)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    lidar2cam, lidar2img = lidar2cam_rt, lidar2img_rt
    return lidar2cam, lidar2img

def get_cam_intrinsic(src_calib_file):
    with open(src_calib_file) as f:
        lines = f.readlines()
    obj = lines[0].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32).reshape(3, 4)
    return P2

def get_M(R, K, R_r, K_r):
    R_inv = np.linalg.inv(R)
    K_inv = np.linalg.inv(K)
    M = np.matmul(K_r, R_r)
    M = np.matmul(M, R_inv)
    M = np.matmul(M, K_inv)
    return M

def ground_ref_points(image, gt_bboxes_3d, lidar2img, x_size, y_size, idx):
    resolution = 0.04
    center_lidar = gt_bboxes_3d[0][:3]
    w, l = 76.8, 50
    shape_top = np.array([w / resolution, l / resolution]).astype(np.int32)
    bev_img = np.zeros((shape_top[1], shape_top[0], 3))
    bev_img = bev_img.reshape(-1, 3)
    
    n, m = [(ss - 1.) / 2. for ss in shape_top]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    xv, yv = np.meshgrid(x, y, sparse=False)
    xyz = np.concatenate((xv[:,:,np.newaxis], yv[:,:,np.newaxis], (center_lidar[2] * np.ones_like(xv)[:,:,np.newaxis] / resolution).astype(np.int32)), axis=-1)
    xyz = xyz + np.array([(0.5 * w + 12.0)/ resolution, 0.0, 0.0]).astype(np.int32).reshape(1,3)        
    xyz_points = xyz * resolution
    xyz_points = xyz_points.reshape(-1, 3)
    
    xyz_img = np.matmul(lidar2img, np.concatenate((xyz_points, np.ones((xyz_points.shape[0],1))), axis=1).T).T
    xyz_img = xyz_img[:,:2] / (xyz_img[:,2] + 10e-6)[:, np.newaxis]
    xyz_img = xyz_img.astype(np.int32)
            
    xyz_img_src = xyz_img.copy()
    xyz_img_src[:,0] = np.clip(xyz_img_src[:,0], 0, x_size[idx]-1)
    xyz_img_src[:,1] = np.clip(xyz_img_src[:,1], 0, y_size[idx]-1)
    bev_img = image[xyz_img_src[:,1], xyz_img_src[:,0],:]
    
    bev_img[xyz_img[:,0] < 0, :] = 0
    bev_img[xyz_img[:,0] > x_size[idx]-1, :] = 0
    bev_img[xyz_img[:,1] < 0, :] = 0
    bev_img[xyz_img[:,1] > y_size[idx]-1, :] = 0
    bev_img = bev_img.reshape(shape_top[1], shape_top[0], 3)
    return bev_img

def roll_transform(info, roll_rect):
    image = cv2.imread(info["cams"]["CAM_FRONT"]["data_path"])
    roll, pitch = parse_roll_pitch_v2(info, is_train=True)
    lidar2cam, lidar2img = get_lidar2cam(info)
    
    target_roll_status = roll + roll_rect
    roll = target_roll_status - roll        
    roll_rad = degree2rad(roll)
    rectify_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0, 0], 
                             [math.sin(roll_rad), math.cos(roll_rad), 0, 0], 
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    lidar2cam_rectify = np.matmul(rectify_roll, lidar2cam)
    
    intrinsic = info["cams"]["CAM_FRONT"]["cam_intrinsic"]
    viewpad = np.eye(4)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img_rectify = (viewpad @ lidar2cam_rectify.T)
                
    print(lidar2cam_rectify.shape)
    M = get_M(lidar2cam[:3,:3], intrinsic[:3,:3], lidar2cam_rectify[:3,:3], intrinsic[:3,:3])
    image = transform_with_M(image, M)
    
    gt_bboxes_3d = info["gt_boxes"]
    y_size = [int(image.shape[0])]
    x_size = [int(image.shape[1])]
    idx = 0
    bev_img = ground_ref_points(image, gt_bboxes_3d, lidar2img_rectify, x_size, y_size, idx)
    
    return image, bev_img
    

if __name__ == "__main__":
    train_data_pkl = "data/rope3d/rope3d_infos_temporal_train_mini.pkl"
    with open(train_data_pkl, 'rb') as f:
        train_infos = pickle.load(f)
    infos = train_infos["infos"]
    info = infos[1]
    
    
    roll_rect = 2.0
    image, bev_img = roll_transform(info, roll_rect)
    total_img = np.vstack([image, bev_img])
    cv2.imwrite(os.path.join("debug_roll", "total_img_" + str(roll_rect) + ".jpg"), total_img)
    
    