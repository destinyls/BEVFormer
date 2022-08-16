# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Lei Yang
# ---------------------------------------------
import os
import csv
import argparse
import yaml
import pickle
import random
import mmcv

import numpy as np

from tqdm import tqdm
from tools.v2x.dataset import SUPPROTED_DATASETS
from tools.v2x.v2x_utils import range2box, id_to_str, Evaluator
from tools.v2x.v2x_utils.vis_utils import *
from tools.v2x.v2x_utils.kitti_utils import *

from scipy.spatial.transform import Rotation as R

name2nuscenceclass = {
    "car": "car",
    "van": "car",
    "truck": "truck",
    "bus": "bus",
    "cyclist": "bicycle",
    "tricyclist": "trailer",
    "motorcyclist": "motorcycle",
    "barrow": "bicycle",
    "pedestrian": "pedestrian",
    "trafficcone": "traffic_cone",
}

class DemoVisual(object):
    def __init__(self, resolution=0.1):
        self.resolution = resolution
        
    def __call__(self, info):
        cam_info = info["cams"]["CAM_FRONT"]
        data_path = cam_info["data_path"]
        
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
 
        gt_boxes = info["gt_boxes"]
        image = cv2.imread(data_path)
        surface_points_list = []
        for obj_id in range(gt_boxes.shape[0]):
            gt_bbox = gt_boxes[obj_id]
            
            lwh = gt_bbox[3:6]
            # center_lidar = gt_bbox[:3] + [0.0, 0.0, 0.5 * lwh[2]]
            center_lidar = gt_bbox[:3]
            yaw_lidar = gt_bbox[6]
            surface_points = self.box3d_surface(lwh, center_lidar, -yaw_lidar, lidar2cam)   
            surface_points_list.append(surface_points)
        surface_points = np.vstack(surface_points_list)   
        
        surface_points_img = np.matmul(lidar2img, np.concatenate((surface_points, np.ones((surface_points.shape[0],1))), axis=1).T).T
        surface_points_img = surface_points_img[:,:2] / (surface_points_img[:,2] + 10e-6)[:, np.newaxis]
        surface_points_img = surface_points_img.astype(np.int32)
        surface_points_img[:,0] = np.clip(surface_points_img[:,0], 0, image.shape[1]-1)
        surface_points_img[:,1] = np.clip(surface_points_img[:,1], 0, image.shape[0]-1)
                
        image[surface_points_img[:,1], surface_points_img[:,0]] = (255,0,0)
        
        pc_path = None if info["lidar_path"] == "" else info["lidar_path"]
        bev_image = self.visual_bev(gt_boxes, pc_path)
        return image, bev_image

    def visual_bev(self, gt_boxes, pc_path=None):
        range_list = [(-60, 60), (0, 100), (-2., -2.), 0.1]
        points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
        if pc_path is not None:
            bev_image = points_filter.get_bev_image(pc_path)
        else:
            bev_image = points_filter.get_meshgrid()
            bev_image = cv2.merge([bev_image, bev_image, bev_image])
        boxes = []
        for obj_id in range(gt_boxes.shape[0]):
            gt_bbox = gt_boxes[obj_id]
            obj_size = gt_bbox[3:6]
            center_lidar = gt_bbox[:3]
            yaw_lidar = gt_bbox[6]
            yaw_lidar = -yaw_lidar - np.pi / 2
            
            box = get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar)
            boxes.append(box)   
        for n in range(len(boxes)):
            corner_points = boxes[n]
            x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
            x_img = x_img[:, 0]
            y_img = y_img[:, 0]
            for i in np.arange(4):
                cv2.line(bev_image, (int(x_img[0]), int(y_img[0])), (int(x_img[1]), int(y_img[1])), (255,255,0), 2)
                cv2.line(bev_image, (int(x_img[0]), int(y_img[0])), (int(x_img[3]), int(y_img[3])), (255,255,0), 2)
                cv2.line(bev_image, (int(x_img[1]), int(y_img[1])), (int(x_img[2]), int(y_img[2])), (255,255,0), 2)
                cv2.line(bev_image, (int(x_img[2]), int(y_img[2])), (int(x_img[3]), int(y_img[3])), (255,255,0), 2)
        return bev_image

    def local2global(self, points, center_lidar, yaw_lidar):
        points_3d_lidar = points.reshape(-1, 3)
        rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                            [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                            [0, 0, 1]])
        points_3d_lidar = np.matmul(rot_mat, points_3d_lidar.T).T + center_lidar
        return points_3d_lidar
    
    def global2cam(self, points, lidar2cam):
        points = np.concatenate((points[:, :3], np.ones((points.shape[0], 1))), axis=1)
        points = np.matmul(lidar2cam, points.T).T
        return points[:, :3]
    
    def distance(self, point):
        return np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    
    def box3d_surface(self, lwh, center_lidar, yaw_lidar, lidar2cam):
        l, w, h = lwh[0], lwh[1], lwh[2]
        surface_points = []
        # top
        shape_top = np.array([w / self.resolution, l / self.resolution]).astype(np.int32)
        n, m = [(ss - 1.) / 2. for ss in shape_top]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        xv, yv = np.meshgrid(x, y, sparse=False)
        xyz = np.concatenate((xv[:,:,np.newaxis], yv[:,:,np.newaxis],  0.5 * np.ones_like(xv)[:,:,np.newaxis] * h / self.resolution), axis=-1)
        points_top = self.local2global(xyz * self.resolution, center_lidar, yaw_lidar)
        # left
        shape_left = np.array([h / self.resolution, l / self.resolution]).astype(np.int32)
        n, m = [(ss - 1.) / 2. for ss in shape_left]
        x, z = np.ogrid[-m:m + 1, -n:n + 1]
        xv, zv = np.meshgrid(x, z, sparse=False)    
        xyz = np.concatenate((0.5 * np.ones_like(xv)[:,:,np.newaxis] * w / self.resolution, xv[:,:,np.newaxis], zv[:,:,np.newaxis]), axis=-1)
        points_left = self.local2global(xyz * self.resolution, center_lidar, yaw_lidar)
        points_left_mean = self.local2global(np.array([0.5 * w, 0.0, 0.0]), center_lidar, yaw_lidar)
        points_left_mean = self.global2cam(points_left_mean, lidar2cam)[0]
        # right
        xyz = np.concatenate((-0.5 * np.ones_like(xv)[:,:,np.newaxis] * w / self.resolution, xv[:,:,np.newaxis], zv[:,:,np.newaxis]), axis=-1)
        points_right = self.local2global(xyz * self.resolution, center_lidar, yaw_lidar)
        points_right_mean = self.local2global(np.array([-0.5 * w, 0.0, 0.0]), center_lidar, yaw_lidar)
        points_right_mean = self.global2cam(points_right_mean, lidar2cam)[0]
        # front
        shape_front = np.array([h / self.resolution, w / self.resolution]).astype(np.int32)
        n, m = [(ss - 1.) / 2. for ss in shape_front]
        y, z = np.ogrid[-m:m + 1, -n:n + 1]
        yv, zv = np.meshgrid(y, z, sparse=False)
        xyz = np.concatenate((yv[:,:,np.newaxis], -0.5 * np.ones_like(yv)[:,:,np.newaxis] * l / self.resolution, zv[:,:,np.newaxis]), axis=-1)
        points_front = self.local2global(xyz * self.resolution, center_lidar, yaw_lidar)
        points_front_mean = self.local2global(np.array([0.0, -0.5 * l, 0.0]), center_lidar, yaw_lidar)
        points_front_mean = self.global2cam(points_front_mean, lidar2cam)[0]
        # rear
        xyz = np.concatenate((yv[:,:,np.newaxis], 0.5 * np.ones_like(yv)[:,:,np.newaxis] * l / self.resolution, zv[:,:,np.newaxis]), axis=-1)
        points_rear = self.local2global(xyz * self.resolution, center_lidar, yaw_lidar)
        points_rear_mean = self.local2global(np.array([0.0, 0.5 * l, 0.0]), center_lidar, yaw_lidar)
        points_rear_mean = self.global2cam(points_rear_mean, lidar2cam)[0]
        surface_points.append(points_top)
        if self.distance(points_left_mean) < self.distance(points_right_mean):
            surface_points.append(points_left)
        else:
            surface_points.append(points_right)
            
        if self.distance(points_front_mean) < self.distance(points_rear_mean):
            surface_points.append(points_front)
        else:
            surface_points.append(points_rear)                
        surface_points = np.vstack(surface_points)
        return surface_points

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data/dair-v2x',
        help='specify the root path of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/dair-v2x',
        required=False,
        help='path to save the exported json')
    parser.add_argument(
        '--split-data',
        type=str,
        default='./tools/data/cooperative-split-data.json',
        required=False,
        help='path to split data')
    args = parser.parse_args()
    return args

def get_cam2world(src_extrinsics_path):
    '''
    with open(src_extrinsics_path, encoding='utf8')as fp:
        calib_lidar2world = yaml.load(fp.read(), Loader=yaml.FullLoader)
    serial_number = calib_lidar2world["serial_number"]
    transform = calib_lidar2world["transform"]["translation"]
    rotation = calib_lidar2world["transform"]["rotation"]
    
    t_velo2world = np.array([transform['x'], transform['y'], transform['z']])
    r_velo2world = np.array([rotation['w'], rotation['x'], rotation['y'], rotation['z']])
    '''
    t_velo2world = np.array([0.0, 0.0, 0.0])
    r_velo2world = np.array([0.0, 0.0, 0.0, 0.0])
    serial_number = "rope3d scenes"
    return r_velo2world, t_velo2world, serial_number

def get_cam_intrinsic(src_calib_file):
    with open(src_calib_file) as f:
        lines = f.readlines()
    obj = lines[0].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32).reshape(3, 4)[:3,:3]
    return P2

def get_denorm(src_denorm_file):
    with open(src_denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

def get_cam2lidar(src_denorm_file):
    denorm = get_denorm(src_denorm_file)
    
    Rx = np.array([[1, 0, 0], 
                   [0, 0, 1], 
                   [0, -1, 0]])
    
    Rz = np.array([[0, 1, 0], 
                   [-1, 0, 0],  
                   [0, 0, 1]])
    
    origin_vector = np.array([0, 1, 0])
    target_vector = -1 * np.array([denorm[0], denorm[1], denorm[2]])
    target_vector_norm = target_vector / np.sqrt(target_vector[0]**2 + target_vector[1]**2 + target_vector[2]**2)       
    sita = math.acos(np.inner(target_vector_norm, origin_vector))
    n_vector = np.cross(target_vector_norm, origin_vector) 
    n_vector = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    n_vector = n_vector.astype(np.float32)
    cam2lidar, _ = cv2.Rodrigues(n_vector * sita)
    cam2lidar = cam2lidar.astype(np.float32)
    cam2lidar = np.matmul(Rx, cam2lidar)
    cam2lidar = np.matmul(Rz, cam2lidar)
    
    Ax, By, Cz, D = denorm[0], denorm[1], denorm[2], denorm[3]
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(D) / mod_area
    
    Tr_cam2lidar = np.eye(4)
    Tr_cam2lidar[:3, :3] = cam2lidar
    Tr_cam2lidar[:3, 3] = [0, 0, d]
    r_cam2velo, t_cam2velo = Tr_cam2lidar[:3, :3], Tr_cam2lidar[:3, 3]
    
    return r_cam2velo, t_cam2velo, Tr_cam2lidar

def alpha2roty(alpha, pos):
    ry = alpha + np.arctan2(pos[0], pos[2])
    if ry > np.pi:
        ry -= 2 * np.pi
    if ry < -np.pi:
        ry += 2 * np.pi
    return ry

def clip2pi(ry):
    if ry > 2 * np.pi:
        ry -= 2 * np.pi
    if ry < - 2* np.pi:
        ry += 2 * np.pi
    return ry

def get_annos(src_label_path, Tr_cam2lidar):
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']
    annos = []
    with open(src_label_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            if row["type"] in name2nuscenceclass.keys():
                alpha = float(row["alpha"])
                pos = np.array((float(row['lx']), float(row['ly']), float(row['lz'])), dtype=np.float32)
                ry = float(row["ry"])
                if alpha > np.pi:
                    alpha -= 2 * np.pi
                    ry = alpha2roty(alpha, pos)
                alpha = clip2pi(alpha)
                ry = clip2pi(ry)
                rotation =  0.5 * np.pi - ry
                
                name = name2nuscenceclass[row["type"]]
                dim = [float(row['dl']), float(row['dw']), float(row['dh'])]
                box2d = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                truncated_state = int(row["truncated"])
                occluded_state = int(row["occluded"])
                if sum(dim) == 0:
                    continue
                
                loc_cam = np.array([float(row['lx']), float(row['ly']), float(row['lz']), 1.0]).reshape(4, 1)
                loc_lidar = np.matmul(Tr_cam2lidar, loc_cam).squeeze(-1)[:3]
                loc_lidar[2] += 0.5 * float(row['dh'])
                anno = {"dim": dim, "loc": loc_lidar, "rotation": rotation, "name": name, "box2d": box2d, "truncated_state": truncated_state, "occluded_state": occluded_state}
                annos.append(anno)

    return annos

def create_data(src_root, out_dir, split='train', version="v1.0", demo=False):
    if split == 'train':
        src_dir = os.path.join(src_root, "training")
        img_paths = ["training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d"]
        depth_img_path = "training-depth_2"
    else:
        src_dir = os.path.join(src_root, "validation")
        img_paths = ["validation-image_2"]
        depth_img_path = "validation-depth_2"

    src_depth_img_path = os.path.join(src_dir, "../", depth_img_path)
    src_label_path = os.path.join(src_dir, "label_2")
    src_calib_path = os.path.join(src_dir, "calib")
    src_denorm_path = os.path.join(src_dir, "denorm")
    src_extrinsics_path = os.path.join(src_dir, "extrinsics")
    
    split_txt = os.path.join(src_dir, "train.txt" if split=='train' else 'val.txt')
    idx_list = [x.strip() for x in open(split_txt).readlines()]
    index_list = []
    for index in idx_list:
        for img_path in img_paths:
            src_img_path = os.path.join(src_dir, "../", img_path)
            img_file = os.path.join(src_img_path, index + ".jpg")
            if os.path.exists(img_file):
                index_list.append((img_path, index))
                break
    
    # index_list=random.sample(index_list, int(len(index_list) * 0.2))
    rope3d_infos = []
    for idx in tqdm(range(len(index_list))):
        img_path, index = index_list[idx]
        src_img_path = os.path.join(src_dir, "../", img_path)
        
        src_img_file = os.path.join(src_img_path, index + ".jpg")
        src_label_file = os.path.join(src_label_path, index + ".txt")
        src_calib_file = os.path.join(src_calib_path, index + ".txt")
        src_denorm_file = os.path.join(src_denorm_path, index + ".txt")
        src_extrinsics_file = os.path.join(src_extrinsics_path, index + ".yaml")
    
        r_velo2world, t_velo2world, serial_number = get_cam2world(src_extrinsics_file)
        r_velo2ego = R.from_matrix(np.eye(3)).as_quat()
        can_bus = np.zeros((18,))
        info = {
            'lidar_path': "",
            'token': index,
            'prev': "",
            'next': "",
            'can_bus': can_bus,
            'frame_idx': -1,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': serial_number,
            'lidar2ego_translation': np.zeros((3,)),
            'lidar2ego_rotation': r_velo2ego,
            'ego2global_translation': t_velo2world,
            'ego2global_rotation': r_velo2world,
            'timestamp': int(6666666666666),
        }

        camera_types = [
            'CAM_FRONT',
        ]
        for cam in camera_types:
            r_cam2velo, t_cam2velo, Tr_cam2lidar = get_cam2lidar(src_denorm_file)
            r_cam2velo_qua = R.from_matrix(r_cam2velo).as_quat()
            cam_intrinsic = get_cam_intrinsic(src_calib_file)
            cam_info = {
                "data_path": src_img_file,
                "type": cam,
                "sample_data_token": index,
                "sensor2ego_translation": t_cam2velo, 
                "sensor2ego_rotation": r_cam2velo_qua,
                "ego2global_translation": t_velo2world,
                "ego2global_rotation": r_velo2world,
                "timestamp": int(6666666666666),
                "sensor2lidar_translation": t_cam2velo,
                "sensor2lidar_rotation" : r_cam2velo,
                "cam_intrinsic": cam_intrinsic,
            }
            info['cams'].update({cam: cam_info})
            
        # obtain sweeps for a single key-frame        
        info['sweeps'] = []
        
        # obtain annotation
        annos = get_annos(src_label_file, Tr_cam2lidar)
        
        names = np.array([anno["name"] for anno in annos])
        locs = np.array([anno["loc"] for anno in annos]).reshape(-1, 3)
        dims = np.array([anno["dim"] for anno in annos]).reshape(-1, 3)
        rots = np.array([anno["rotation"] for anno in annos]).reshape(-1, 1)
        velocity = np.zeros((dims.shape[0], 2))
        valid_flag = np.array([True for anno in annos], dtype=bool).reshape(-1)        
        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2)
        info['valid_flag'] = valid_flag
        rope3d_infos.append(info)
        
        if demo:
            image, bev_image = demo_tool(info)
            cv2.imwrite("demo_image.jpg", image)
            cv2.imwrite("demo_bev_image.jpg", bev_image)      
        
    metadata = dict(version=version)
    print(split, ' sample: {}'.format(len(rope3d_infos)))
    data = dict(infos=rope3d_infos, metadata=metadata)
    info_path = os.path.join(out_dir, 'rope3d_infos_temporal_{}.pkl'.format(split))
    mmcv.dump(data, info_path)

if __name__ == "__main__":   
    args = parse_args()
    data_root, out_dir, split_data = args.data_root, args.out_dir, args.split_data
    demo_tool = DemoVisual(0.03)

    create_data(data_root, out_dir, split_data)
    
    '''
    with open(os.path.join("data/dair-v2x", "dair_v2x_i_infos_temporal_trainval.pkl"), 'rb') as f:
        train_infos = pickle.load(f)
    for info in train_infos["infos"]:
        image, bev_image = demo_tool(info)
        cv2.imwrite("demo_image.jpg", image)
        cv2.imwrite("demo_bev_image.jpg", bev_image)
    '''
    