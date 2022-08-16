import argparse
import pickle
import cv2
import os
import math

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-pkl',
        type=str,
        default='',
        help='specify the path of dataset pickle file')
    
    args = parser.parse_args()
    return args

def parse_roll_pitch(self, lidar2cam):
        ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
        ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
        denorm = self.equation_plane(ground_points_cam)
        origin_vector = np.array([0, 1.0, 0])
        target_vector_xy = np.array([denorm[0], denorm[1], 0.0])
        target_vector_yz = np.array([0.0, denorm[1], denorm[2]])
        target_vector_xy = target_vector_xy / np.sqrt(target_vector_xy[0]**2 + target_vector_xy[1]**2 + target_vector_xy[2]**2)       
        target_vector_yz = target_vector_yz / np.sqrt(target_vector_yz[0]**2 + target_vector_yz[1]**2 + target_vector_yz[2]**2)       
        roll = math.acos(np.inner(origin_vector, target_vector_xy))
        pitch = math.acos(np.inner(origin_vector, target_vector_yz))
        roll = -1 * self.rad2degree(roll) if target_vector_xy[0] > 0 else self.rad2degree(roll)
        pitch = -1 * self.rad2degree(pitch) if target_vector_yz[1] > 0 else self.rad2degree(pitch)
        return roll, pitch
    
if __name__ == "__main__":
    args = parse_args()
    
    with open(args.data_pkl, 'rb') as f:
        train_infos = pickle.load(f)
    for info in train_infos["infos"]:
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
