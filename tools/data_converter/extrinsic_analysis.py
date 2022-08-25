import argparse
import pickle
from turtle import distance
import cv2
import os
import math
import random
import json
import mmcv
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-pkl',
        type=str,
        default='',
        help='specify the path of dataset pickle file')
    args = parser.parse_args()
    return args

def equation_plane(points): 
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

def rad2degree(radian):
    return radian * 180 / np.pi

def degree2rad(degree):
    return degree * np.pi / 180

def parse_roll_pitch(sample_token, is_train=False):
    denorm_file = os.path.join("data/rope3d/training/denorm", sample_token + ".txt")
    if not os.path.exists(denorm_file):
        denorm_file = os.path.join("data/rope3d/validation/denorm", sample_token + ".txt")
        
    denorm = get_denorm(denorm_file)
    denorm = -1 * denorm
    origin_vector = np.array([0, 1.0, 0])
    target_vector_xy = np.array([denorm[0], denorm[1], 0.0])
    target_vector_yz = np.array([0.0, denorm[1], denorm[2]])
    target_vector_xy = target_vector_xy / np.sqrt(target_vector_xy[0]**2 + target_vector_xy[1]**2 + target_vector_xy[2]**2)       
    target_vector_yz = target_vector_yz / np.sqrt(target_vector_yz[0]**2 + target_vector_yz[1]**2 + target_vector_yz[2]**2)       
    roll = math.acos(np.inner(origin_vector, target_vector_xy))
    pitch = math.acos(np.inner(origin_vector, target_vector_yz))
    roll = -1 * rad2degree(roll) if target_vector_xy[0] > 0 else rad2degree(roll)
    pitch = -1 * rad2degree(pitch) if target_vector_yz[1] > 0 else rad2degree(pitch)
    
    mod_area = np.sqrt(np.sum(np.square([denorm[0], denorm[1], denorm[2]])))
    distance = abs(denorm[3]) / mod_area
    return roll, pitch, distance

def get_denorm(src_denorm_file):
    with open(src_denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

def scatter_plot(roll_list, pitch_list, s=3, c='g', marker='*', alpha=0.65):
    roll_array = np.array(roll_list)
    pitch_array = np.array(pitch_list)
    return plt.scatter(roll_array, pitch_array, s=s, c=c, marker=marker, alpha=alpha)
    
def parse_data_infos(data_pkl, is_train=False):
    with open(data_pkl, 'rb') as f:
        train_infos = pickle.load(f)
    roll_list, pitch_list, distance_list, f_list = [], [], [], []
    file_list = []
    scenes_json = dict()
    for info in tqdm(train_infos["infos"]):
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
        roll, pitch, distance = parse_roll_pitch(info["token"], is_train)
        
        fx, fy = intrinsic[0,0], intrinsic[1,1]
        ext_key = str(round(abs(roll), 4)) + "_" + str(round(abs(pitch), 4))
        if round(pitch, 4) not in pitch_list:
            scenes_json[ext_key] = [info["token"]]
            roll_list.append(round(roll, 4))
            pitch_list.append(round(pitch, 4))
            distance_list.append(round(distance, 4))
            file_list.append(info["token"])
            f_list.append(round(0.5 * (fx + fy), 4))
        else:
            scenes_json[ext_key].append(info["token"])
    return roll_list, pitch_list, distance_list, f_list, file_list, scenes_json

def create_val_mini_infos(data_pkl, is_train=False):
    with open(data_pkl, 'rb') as f:
        val_infos = pickle.load(f)
        
    rope3d_infos=random.sample(val_infos["infos"], 1000)
    '''
    rope3d_infos = []
    for info in val_infos["infos"]:
        roll, pitch = parse_roll_pitch_v2(info, is_train)
        if round(roll, 3) == 0.495 and round(pitch, 3) == -12.135:
            rope3d_infos.append(info)
    print(len(rope3d_infos))
    '''
    metadata = dict(version="v1.0")
    data = dict(infos=rope3d_infos, metadata=metadata)
    info_path = os.path.join("data/rope3d", 'rope3d_infos_temporal_val_hom_mini.pkl')
    mmcv.dump(data, info_path)

def create_train_mini_infos(data_pkl, is_train=False):
    with open(data_pkl, 'rb') as f:
        train_infos = pickle.load(f)
    
    rope3d_infos=random.sample(train_infos["infos"], 2333)
    '''
    pitch_candidates = [-13.9, 12.19, -10.11]
    rope3d_infos = []
    for info in train_infos["infos"]:
        roll, pitch = parse_roll_pitch_v2(info, is_train)
        if round(pitch, 2) in pitch_candidates:
            rope3d_infos.append(info)
    print(len(rope3d_infos))
    '''
    metadata = dict(version="v1.0")
    data = dict(infos=rope3d_infos, metadata=metadata)
    info_path = os.path.join("data/rope3d", 'rope3d_infos_temporal_train_hom_mini.pkl')
    mmcv.dump(data, info_path)

if __name__ == "__main__":
    args = parse_args()
    train_data_pkl = "data/rope3d/rope3d_infos_temporal_hom_train.pkl"
    val_data_pkl = "data/rope3d/rope3d_infos_temporal_hom_val.pkl"
    create_train_mini_infos(train_data_pkl, True)
    create_val_mini_infos(val_data_pkl, False)

    print("finished ...")
    img_paths = ["training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d"]
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    roll_list, pitch_list, distance_list, f_list, file_list, scenes_json = parse_data_infos(train_data_pkl, True)
    with open("data/rope3d/scenes_json_train.json",'w') as file:
        json.dump(scenes_json, file)
    # s1 = scatter_plot(roll_list, pitch_list, s=250, c='b', marker="*", alpha=1.0)
    for idx in range(len(file_list)):   
        image_name = file_list[idx] + ".jpg"
        for img_path in img_paths:
            img_f = os.path.join("data/rope3d", img_path, image_name)
            if os.path.exists(img_f):
                img = cv2.imread(img_f)
                cv2.imwrite(os.path.join("scence_image", str(roll_list[idx]) + "_" + str(pitch_list[idx]) + "_" + str(distance_list[idx]) + "_" + str(f_list[idx]) + ".jpg"), img)

    roll_list, pitch_list, distance_list_2, f_list_2, file_list, scenes_json = parse_data_infos(val_data_pkl, False)
    with open("data/rope3d/scenes_json_val.json",'w') as file:
            json.dump(scenes_json, file)        
    # s2 = scatter_plot(roll_list, pitch_list, s=50, c='r', marker="o", alpha=1.0) 
    
    for idx in range(len(file_list)):        
        image_name = file_list[idx] + ".jpg"
        img_f = os.path.join("data/rope3d", "validation-image_2", image_name)
        if os.path.exists(img_f):
            img = cv2.imread(img_f)
            cv2.imwrite(os.path.join("scence_image", str(roll_list[idx]) + "_" + str(pitch_list[idx]) + "_" + str(distance_list[idx])+ "_" + str(f_list[idx]) + ".jpg"), img)
    '''
    plt.xlabel("roll", fontdict={'weight': 'normal', 'size': 15})
    plt.ylabel("pitch", fontdict={'weight': 'normal', 'size': 15})   
    plt.legend((s1,s2),('train','val') ,loc = 'best')
    plt.savefig('roll_pitch_distribution.jpg')
    '''
    print(f_list, distance_list)
    print(f_list_2, distance_list_2)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    s1 = scatter_plot(f_list, distance_list, s=250, c='b', marker="*", alpha=1.0)
    s2 = scatter_plot(f_list_2, distance_list_2, s=50, c='r', marker="o", alpha=1.0)
    plt.xlabel("f", fontdict={'weight': 'normal', 'size': 15})
    plt.ylabel("height", fontdict={'weight': 'normal', 'size': 15})   
    plt.legend((s1,s2),('train','val') ,loc = 'best')
    plt.savefig('f_height_distribution.jpg')