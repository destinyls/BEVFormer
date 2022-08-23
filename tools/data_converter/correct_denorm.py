import os
import json
import csv
import yaml
import random

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

import numpy as np

from tqdm import tqdm
from mpl_toolkits import mplot3d

from scipy.spatial.transform import Rotation as R

from tools.data_converter.extrinsic_analysis import *

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1
        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic

def get_denorm(src_denorm_file):
    with open(src_denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

def plot_plane(a, b, c, d, scale=(-75, -50.0)):
    xx, yy = np.mgrid[:100, scale[0]:scale[1]]
    return xx, yy, (-d - a * xx - b * yy) / c

def plane_distance(locs, denorm):
    distance_list = []
    for idx in range(locs.shape[0]):
        loc = locs[idx]        
        dis = (np.sum(loc * np.array(denorm[:3])) + denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
        distance_list.append(dis)
    return distance_list

def estimate_denorm(loc_array):
    max_iterations = 100
    n = loc_array.shape[0]
    goal_inliers = n * 0.8
    m, b = run_ransac(loc_array, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations) 
    dis_list = np.array(plane_distance(loc_array, m))
    mask = np.logical_and(dis_list > -0.8, dis_list < 0.8)
    loc_array = loc_array[mask]
    m, b = run_ransac(loc_array, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations) 
    dis_list = np.array(plane_distance(loc_array, m))
    dis_list = plane_distance(loc_array, m)
    print("m: ", sum(dis_list)/len(dis_list))
    return m
    
def create_denorm(scenes_json, is_train=True):
    split_path = "training" if is_train else "validation"
    with open(os.path.join(scenes_json), "r") as f:
        scenes_train = json.load(f)
    for scene_key in list(scenes_train.keys()):
        scene_key = "0.6398_12.2654"
        scene_list = scenes_train[scene_key]
        loc_list = []
        for frame_name in tqdm(scene_list):
            label_path = os.path.join("data/rope3d", split_path, "label_2", frame_name + ".txt")
            fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw', 'dl', 'lx', 'ly', 'lz', 'ry']
            with open(label_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
                for line, row in enumerate(reader):
                    loc = [float(row['lx']), float(row['ly']), float(row['lz'])]
                    if sum(loc) != 0.0 and loc[2] < 60 and loc[2] > 20:
                        loc_list.append(loc)
                    if len(loc_list) > 3000:
                        break
        loc_array = np.array(loc_list)
        m = estimate_denorm(loc_array)
        denorm = -1 * m
        origin_vector = np.array([0, 1.0, 0])
        target_vector_xy = np.array([denorm[0], denorm[1], 0.0])
        target_vector_yz = np.array([0.0, denorm[1], denorm[2]])
        target_vector_xy = target_vector_xy / np.sqrt(target_vector_xy[0]**2 + target_vector_xy[1]**2 + target_vector_xy[2]**2)       
        target_vector_yz = target_vector_yz / np.sqrt(target_vector_yz[0]**2 + target_vector_yz[1]**2 + target_vector_yz[2]**2)       
        roll = math.acos(np.inner(origin_vector, target_vector_xy))
        pitch = math.acos(np.inner(origin_vector, target_vector_yz))
        roll = -1 * rad2degree(roll) if target_vector_xy[0] > 0 else rad2degree(roll)
        pitch = -1 * rad2degree(pitch) if target_vector_yz[1] > 0 else rad2degree(pitch)
        print("new roll pitch: ", roll, pitch)
        scene_list = scenes_train[scene_key]
        for frame_name in tqdm(scene_list):
            os.makedirs(os.path.join("data/rope3d", split_path, "denorm"), exist_ok=True)
            denorm_path = os.path.join("data/rope3d", split_path, "denorm", frame_name + ".txt")
            with open(denorm_path, "w") as f:
                line_string = str(round(m[0], 7)) + " " + str(round(m[1], 7)) + " " + str(round(m[2], 7)) + " " + str(round(m[3], 7))
                f.write(line_string)

def create_denorm_v2(sample_token, is_train=True):
    split_flag = "training" if is_train else "validation"
    extrinsics_file = os.path.join("data/rope3d", split_flag, "extrinsics", sample_token + ".yaml")
    with open(extrinsics_file, 'r', encoding='utf-8') as f:
        extrinsics = yaml.load(f.read(), Loader=yaml.FullLoader)
    translation = extrinsics["transform"]["translation"]
    translation_matrix = np.array([translation['x'], translation['y'], translation['z']]).reshape(3, 1)
    rotation = extrinsics["transform"]["rotation"]
    Rq=[rotation['x'], rotation['y'], rotation['z'], rotation['w']]
    Rm = R.from_quat(Rq)
    rotation_matrix = Rm.as_matrix()
    Tr_world_camera = np.hstack([rotation_matrix, translation_matrix])
    Tr_world_camera = np.vstack([Tr_world_camera, np.array([0, 0, 0, 1.0]).reshape(1, 4)])
    Tr_world_camera = np.linalg.inv(Tr_world_camera)
    if "points_lane_detection" not in extrinsics.keys():
        return None, None
    points_lane = np.array(extrinsics['points_lane_detection']).reshape(-1, 3)
    points_lane_extend = np.hstack([points_lane, np.ones((points_lane.shape[0], 1))])
    points_lane_camera = np.matmul(points_lane_extend, Tr_world_camera.T)
    
    points_lane_camera = points_lane_camera[:,:3]
    mask = np.logical_and(points_lane_camera[:,2] > 0, points_lane_camera[:,2] < 50)
    
    print(points_lane_camera.shape, mask.shape)
    # points_lane_camera = points_lane_camera[mask]
    print(points_lane_camera.shape)

    m = estimate_denorm(points_lane_camera)
    denorm = -1 * m
    origin_vector = np.array([0, 1.0, 0])
    target_vector_xy = np.array([denorm[0], denorm[1], 0.0])
    target_vector_yz = np.array([0.0, denorm[1], denorm[2]])
    target_vector_xy = target_vector_xy / np.sqrt(target_vector_xy[0]**2 + target_vector_xy[1]**2 + target_vector_xy[2]**2)       
    target_vector_yz = target_vector_yz / np.sqrt(target_vector_yz[0]**2 + target_vector_yz[1]**2 + target_vector_yz[2]**2)       
    roll = math.acos(np.inner(origin_vector, target_vector_xy))
    pitch = math.acos(np.inner(origin_vector, target_vector_yz))
    roll = -1 * rad2degree(roll) if target_vector_xy[0] > 0 else rad2degree(roll)
    pitch = -1 * rad2degree(pitch) if target_vector_yz[1] > 0 else rad2degree(pitch)

    return roll, pitch
    
if __name__ == "__main__":
    scenes_json_train = os.path.join("data/rope3d", "scenes_json_train.json")
    with open(os.path.join(scenes_json_train), "r") as f:
        scenes_train = json.load(f)
    roll, pitch = create_denorm_v2(scenes_train["0.6398_12.2654"][0])
    print(roll, pitch)
    create_denorm(scenes_json_train)
    # create_denorm(scenes_json_val, False)
