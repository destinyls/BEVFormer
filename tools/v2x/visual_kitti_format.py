import argparse
import os
import csv

import numpy as np
from v2x_utils.vis_utils import *

def _read_imageset_file(imageset_txt):
    with open(imageset_txt, 'r') as f:
        lines = f.readlines()
    img_ids = [int(line) for line in lines]
    return img_ids

def get_cam_8_points_v2(label_file):
    camera_8_points_list, box2d_list = [], []
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                    'dl', 'lx', 'ly', 'lz', 'ry']
    with open(label_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            h, w, l = float(row['dh']), float(row['dw']), float(row['dl'])
            x, y, z = float(row['lx']), float(row['ly']), float(row['lz'])
            ry = float(row["ry"])
            xmin, ymin, xmax, ymax = float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])

            x_corners = [0, l, l, l, l, 0, 0, 0]
            y_corners = [0, 0, h, h, 0, 0, h, h]
            z_corners = [0, 0, 0, w, w, w, w, 0]
            x_corners += - np.float32(l) / 2
            y_corners += - np.float32(h)
            z_corners += - np.float32(w) / 2
            corners_3d = np.array([x_corners, y_corners, z_corners])
            rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)],
                                [0, 1, 0],
                                [-np.sin(ry), 0, np.cos(ry)]])
            corners_3d = np.matmul(rot_mat, corners_3d)
            corners_3d += np.array([x, y, z]).reshape([3, 1])
            camera_8_points_list.append(corners_3d.T)
            box2d_list.append([xmin, ymin, xmax, ymax])
            
    return camera_8_points_list, box2d_list

def get_calib_intrinsic(calib_file):
    # get camera intrinsic matrix K
    with open(calib_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P2 = row[1:]
                P2 = [float(i) for i in P2]
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                break
    return P2

def plot_box3d_om_img(camera_8_points_list, calib_intrinsic, box2d_list, img, index, save_path="demo"):
    cam8points = np.array(camera_8_points_list)
    num_bbox = cam8points.shape[0]
    
    uv_origin = points_cam2img(cam8points, calib_intrinsic)
    uv_origin = (uv_origin - 1).round()
    plot_rect3d_on_img(img, num_bbox, uv_origin)
    for i in range(len(box2d_list)):
        box2d = box2d_list[i]
        img = cv2.rectangle(img, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), (255, 255, 0), 2)
    
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, '{:06d}'.format(index) + ".png"), img)

if __name__ == "__main__":
    root_path = "/home/yanglei/DataSets/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side-kitti"
    image_path = os.path.join(root_path, "training", "image_2")
    calib_path = os.path.join(root_path, "training", "calib")
    # label_path = os.path.join(root_path, "training", "label_2")
    label_path = "/home/yanglei/BEVFormer/test/bevformer_small_dair_v2x/data"

    imageset_txt = os.path.join(root_path, "ImageSets", "val.txt")
    img_ids = _read_imageset_file(imageset_txt)
    
    for idx in img_ids:
        image_file = os.path.join(image_path, '{:06d}'.format(idx) + ".jpg")
        calib_file = os.path.join(calib_path, '{:06d}'.format(idx) + ".txt")
        label_file = os.path.join(label_path, '{:06d}'.format(idx) + ".txt")
        
        image = cv2.imread(image_file)
        camera_8_points_list, box2d_list = get_cam_8_points_v2(label_file)
        
        calib_intrinsic = get_calib_intrinsic(calib_file)
        if len(camera_8_points_list) > 0:
            plot_box3d_om_img(camera_8_points_list, calib_intrinsic, box2d_list, image, idx, "demo")
        