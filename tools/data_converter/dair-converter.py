# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Lei Yang
# ---------------------------------------------
import os
import argparse
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
    "barrowlist": "barrier",
    "pedestrian": "pedestrian",
    "trafficcone": "traffic_cone",
}

superclass = {
    -1: "ignore",
    0: "pedestrian",
    1: "cyclist",
    2: "car",
    3: "ignore",
}

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

def get_cam2lidar(root_path, calib_virtuallidar_to_camera_path):
    calib_lidar2cam_path = calib_virtuallidar_to_camera_path
    calib_lidar2cam_path = os.path.join(root_path, "infrastructure-side", calib_lidar2cam_path)
    calib_lidar2cam = read_json(calib_lidar2cam_path)
    r_velo2cam, t_velo2cam = get_lidar2cam(calib_lidar2cam)

    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = r_velo2cam
    Tr_velo2cam[:3, 3] = t_velo2cam[:, 0]
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam) 
    r_cam2velo, t_cam2velo = Tr_cam2velo[:3, :3], Tr_cam2velo[:3, 3]
    return r_cam2velo, t_cam2velo

def get_lidar2world(root_path, calib_virtuallidar_to_world_path):
    calib_virtuallidar_to_world_path = os.path.join(root_path, "infrastructure-side", calib_virtuallidar_to_world_path)
    calib_lidar2world = read_json(calib_virtuallidar_to_world_path)
    t_velo2world = np.array(calib_lidar2world['translation']).squeeze(-1)
    r_velo2world = np.array(calib_lidar2world['rotation'])
    return r_velo2world, t_velo2world

def get_annos(root_path, label_camera_std_path):
    label_path = os.path.join(root_path, "infrastructure-side", label_camera_std_path)
    oris = read_json(label_path)
    annos = []
    for ori in oris:
        if "rotation" not in ori.keys():
            ori["rotation"] = 0.0
        dim = [float(ori["3d_dimensions"]["l"]), float(ori["3d_dimensions"]["w"]), float(ori["3d_dimensions"]["h"])]
        loc = [float(ori["3d_location"]["x"]), float(ori["3d_location"]["y"]), float(ori["3d_location"]["z"])]
        box2d = [float(ori["2d_box"]["xmin"]), float(ori["2d_box"]["ymin"]), float(ori["2d_box"]["xmax"]), float(ori["2d_box"]["ymax"])]
        rotation = float(ori["rotation"])
        if rotation > np.pi:
            rotation -= np.pi
        elif rotation < -np.pi:
            rotation += np.pi
        name = ori["type"]
        truncated_state = int(ori["truncated_state"])
        occluded_state = int(ori["occluded_state"])
        anno = {"dim": dim, "loc": loc, "rotation": rotation, "name": name, "box2d": box2d, "truncated_state": truncated_state, "occluded_state": occluded_state}
        annos.append(anno)
    return annos

def fill_infos(root_path, dataset, max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    dair_infos = []
    frame_idx = 0
    for idx in tqdm(range(len(dataset))):        
        sample, gt_label, filt = dataset[idx]
        lidar_path = sample["pointcloud_path"]
        token = sample["token"]
        r_velo2world, t_velo2world = \
            get_lidar2world(root_path, sample["calib_virtuallidar_to_world_path"])
        r_velo2world_qua = R.from_matrix(r_velo2world).as_quat()
        r_velo2ego_qua = R.from_matrix(np.eye(3)).as_quat()
        can_bus = np.zeros((18,))
        cam_instrinsic_path = os.path.join(root_path, "infrastructure-side", sample["calib_camera_intrinsic_path"]) 
        calib_intrinsic = get_cam_calib_intrinsic(cam_instrinsic_path)
        info = {
            'lidar_path': os.path.join("data/dair-v2x", lidar_path),
            'token': token,
            'prev': sample["prev_token"],
            'next': sample["next_token"],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['batch_id'],  # temporal related info
            'lidar2ego_translation': np.zeros((3,)),
            'lidar2ego_rotation': r_velo2ego_qua,
            'ego2global_translation': t_velo2world,
            'ego2global_rotation': r_velo2world_qua,
            'timestamp': int(sample['image_timestamp']),
        }
        
        if sample["next_token"] == "":
            frame_idx = 0
        else:
            frame_idx += 1

        camera_types = [
            'CAM_FRONT',
        ]
        for cam in camera_types:
            r_cam2velo, t_cam2velo = get_cam2lidar(root_path, sample["calib_virtuallidar_to_camera_path"])
            r_cam2velo_qua = R.from_matrix(r_cam2velo).as_quat()
            cam_info = {
                "data_path": os.path.join("data/dair-v2x", sample["image_path"]),
                "type": cam,
                "sample_data_token": token,
                "sensor2ego_translation": t_cam2velo, 
                "sensor2ego_rotation": r_cam2velo_qua,
                "ego2global_translation": t_velo2world,
                "ego2global_rotation": r_velo2world_qua,
                "timestamp": int(sample['image_timestamp']),
                "sensor2lidar_translation": t_cam2velo,
                "sensor2lidar_rotation" : r_cam2velo,
                "cam_intrinsic": calib_intrinsic
            }
            info['cams'].update({cam: cam_info})
            
        # obtain sweeps for a single key-frame
        sweeps = []
        while len(sweeps) < max_sweeps:
            if sample["prev_token"] != "":
                prev_sample, _ = dataset.cur_inf_frame(sample["prev_token"])
                r_cam2velo, t_cam2velo = get_cam2lidar(root_path, prev_sample["calib_virtuallidar_to_camera_path"])
                r_velo2world, t_velo2world = get_lidar2world(root_path, prev_sample["calib_virtuallidar_to_world_path"])
                r_cam2velo_qua = R.from_matrix(r_cam2velo).as_quat()
                r_velo2world_qua = R.from_matrix(r_velo2world).as_quat()
                
                cam_instrinsic_path = os.path.join(root_path, "infrastructure-side", prev_sample["calib_camera_intrinsic_path"]) 
                calib_intrinsic = get_cam_calib_intrinsic(cam_instrinsic_path)
                sweep = {
                    "data_path": os.path.join("data/dair-v2x", prev_sample["image_path"]),
                    "type": "CAM_FRONT",
                    "sample_data_token": prev_sample["token"],
                    "sensor2ego_translation": t_cam2velo,
                    "sensor2ego_rotation": r_cam2velo_qua,
                    "ego2global_translation": t_velo2world,
                    "ego2global_rotation": r_velo2world_qua,
                    "timestamp": int(prev_sample["image_timestamp"]),
                    "sensor2lidar_rotation": r_cam2velo,
                    "sensor2lidar_translation": t_cam2velo,
                    "cam_intrinsic": calib_intrinsic
                }
                sweeps.append(sweep)
            else:
                break
        info['sweeps'] = sweeps
        
        # obtain annotation
        label_camera_std_path = sample["label_camera_std_path"]
        annos = get_annos(root_path, label_camera_std_path)
        
        locs = np.array([anno["loc"] for anno in annos]).reshape(-1, 3)
        dims = np.array([anno["dim"] for anno in annos]).reshape(-1, 3)
        rots = np.array([anno["rotation"] for anno in annos]).reshape(-1, 1)
                
        velocity = np.zeros((dims.shape[0], 2))
        valid_flag = np.array([True for anno in annos], dtype=bool).reshape(-1)
            
        names = [anno["name"] for anno in annos]
        for i in range(len(names)):
            names[i] = name2nuscenceclass[names[i].lower()]
        names = np.array(names)
        
        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2)
        info['valid_flag'] = valid_flag

        dair_infos.append(info)
    return dair_infos
    
def create_dair_infos(root_path,
                      out_path,
                      split_data_path,
                      info_prefix="dair_v2x_i",
                      version="v1.0",
                      sensortype="camera",
                      max_sweeps=10,
                      extended_range = [-10, -49.68, -3, 79.12, 49.68, 1]):
    """Create info file of dair-v2x dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    dair_train = SUPPROTED_DATASETS["dair-v2x-i"](
        root_path,
        split_data_path,
        split="train",
        sensortype=sensortype,
        extended_range=extended_range,
    )
    dair_val = SUPPROTED_DATASETS["dair-v2x-i"](
        root_path,
        split_data_path,
        split="val",
        sensortype=sensortype,
        extended_range=extended_range,
    )
    train_dair_infos, val_dair_infos = _fill_trainval_infos(
        root_path, dair_train, dair_val, max_sweeps)
    
    metadata = dict(version=version)
    print('train sample: {}, val sample: {}'.format(len(train_dair_infos), len(val_dair_infos)))
    data = dict(infos=train_dair_infos, metadata=metadata)
    info_path = os.path.join(out_path, '{}_infos_temporal_train.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    
    data['infos'] = val_dair_infos
    info_val_path = os.path.join(out_path,'{}_infos_temporal_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_val_path)

def _fill_trainval_infos(root_path,
                         dair_train,
                         dair_val,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_dair_infos = fill_infos(root_path, dair_train, max_sweeps)
    val_dair_infos = fill_infos(root_path, dair_val, max_sweeps)
    return train_dair_infos, val_dair_infos
        
if __name__ == "__main__":   
    args = parse_args()
    data_root, out_dir, split_data = args.data_root, args.out_dir, args.split_data
    
    box_range = np.array([-10, -49.68, -3, 79.12, 49.68, 1])
    indexs = [
        [0, 1, 2],
        [3, 1, 2],
        [3, 4, 2],
        [0, 4, 2],
        [0, 1, 5],
        [3, 1, 5],
        [3, 4, 5],
        [0, 4, 5],
    ]
    extended_range = np.array([[box_range[index] for index in indexs]])
    create_dair_infos(data_root, out_dir, split_data, extended_range=None)