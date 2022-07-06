import argparse
import os

import numpy as np
from dataset import SUPPROTED_DATASETS
from v2x_utils import range2box, id_to_str, Evaluator
from v2x_utils.vis_utils import *
from v2x_utils.kitti_utils import *

def add_arguments(parser):
    parser.add_argument("--input", type=str, default="/home/yanglei/DataSets/DAIR-V2X/cooperative-vehicle-infrastructure")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--split-data-path", type=str, default="/home/yanglei/BEVFormer/tools/v2x/cooperative-split-data.json"
    )
    parser.add_argument("--dataset", type=str, default="dair-v2x-i")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--pred-classes", nargs="+", default=["car"])
    parser.add_argument("--model", type=str, default="single_veh")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-point-cloud", action="store_true")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--extended-range", type=float, nargs="+", default=[-10, -49.68, -3, 79.12, 49.68, 1])
    parser.add_argument("--sensortype", type=str, default="lidar")
    parser.add_argument("--eval-single", action="store_true")

def image_vis(args, inf_frame, output_path='./'):
    image_path = os.path.join(args.input, "infrastructure-side", inf_frame["image_path"])
    label_path = os.path.join(args.input, "infrastructure-side", inf_frame["label_camera_std_path"])
    lidar2cam_path = os.path.join(args.input, "infrastructure-side", inf_frame["calib_virtuallidar_to_camera_path"]) 
    cam_instrinsic_path = os.path.join(args.input, "infrastructure-side", inf_frame["calib_camera_intrinsic_path"]) 
    labels = []
    oris = read_json(label_path)
    for ori in oris:
        if "rotation" not in ori.keys():
            ori["rotation"] = 0.0
        labels.append([ori["3d_dimensions"], ori["3d_location"], ori["rotation"]])
    camera_8_points_list = get_cam_8_points(labels, lidar2cam_path) 
    vis_label_in_img(camera_8_points_list, image_path, cam_instrinsic_path, output_path)

def pcd_vis(args, inf_frame):
    pcd_path =  os.path.join(args.input, "infrastructure-side", inf_frame["pointcloud_path"])
    label_path = os.path.join(args.input, "infrastructure-side", inf_frame["label_camera_std_path"])
    x, y, z = read_pcd(pcd_path)
    boxes = read_label_bboxes(label_path)
    fig = plot_box_pcd(x, y, z, boxes)
    
def pcd_vis_2(args, inf_frame):
    pcd_path =  os.path.join(args.input, "infrastructure-side", inf_frame["pointcloud_path"])
    label_path = os.path.join(args.input, "infrastructure-side", inf_frame["label_camera_std_path"])
    range_list = [(-60, 60), (0, 100), (-2., -2.), 0.1]
    points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
    bev_image = points_filter.get_bev_image(pcd_path)
    boxes = read_label_bboxes(label_path)
    
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
    cv2.imwrite("demo.jpg", bev_image)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_arguments(parser)
    args, _ = parser.parse_known_args()
    extended_range = range2box(np.array(args.extended_range))
    dataset = SUPPROTED_DATASETS[args.dataset](
        args.input,
        args.split_data_path,
        split=args.split,
        sensortype=args.sensortype,
        extended_range=extended_range,
    )
    
    inf_frame, gt_label, filt = dataset[10]
    print(inf_frame.keys())
    pointcloud_id = inf_frame["pointcloud_path"].split('/')[1].split('.')[0]
    # pre_inf_frame, _, _ = dataset.prev_inf_frame(pointcloud_id)
    image_vis(args, inf_frame)
    pcd_vis_2(args, inf_frame)
    
    # print(inf_frame["pointcloud_path"], pre_inf_frame["pointcloud_path"], inf_frame["valid_batch_splits"][0]["batch_start_id"], inf_frame["valid_batch_splits"][0]["batch_end_id"])    
