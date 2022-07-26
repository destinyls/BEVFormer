import math
import cv2
import numpy as np
from numpy import random

import torch
import mmcv

from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC

from tools.v2x.v2x_utils.vis_utils import *
from tools.v2x.v2x_utils.kitti_utils import *

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str



@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus', 'height_map', 'height_mask',
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
       
        data = {}
        img_metas = {}
      
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'

@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str
    
@PIPELINES.register_module()
class ProduceHeightMap(object):
    def __init__(self, resolution=[], back_ratio=[]):
        self.resolution = resolution
        self.back_ratio = back_ratio
        assert len(self.resolution)==1
        assert len(self.back_ratio)==1

    def __call__(self, results):
        height_map = [np.zeros((img.shape[0], img.shape[1])) for idx, img in enumerate(results['img'])]

        y_size = [int(img.shape[0]) for img in results['img']]
        x_size = [int(img.shape[1]) for img in results['img']]
        gt_bboxes_3d  = torch.clone(results["gt_bboxes_3d"].tensor).numpy()
        
        for idx in range(len(height_map)):
            lidar2img = results["lidar2img"][idx]
            lidar2cam = results["lidar2cam"][idx]
            surface_points_list = []
            for obj_id in range(gt_bboxes_3d.shape[0]):
                gt_bbox = gt_bboxes_3d[obj_id]
                lwh = gt_bbox[3:6]
                center_lidar = gt_bbox[:3] + [0.0, 0.0, 0.5 * lwh[2]]
                yaw_lidar = gt_bbox[6]
                surface_points = self.box3d_surface(lwh, center_lidar, -yaw_lidar, lidar2cam)   
                surface_points_list.append(surface_points)
            surface_points = np.vstack(surface_points_list)   
            
            surface_points_img = np.matmul(lidar2img, np.concatenate((surface_points, np.ones((surface_points.shape[0],1))), axis=1).T).T
            surface_points_img = surface_points_img[:,:2] / (surface_points_img[:,2] + 10e-6)[:, np.newaxis]
            surface_points_img = surface_points_img.astype(np.int32)
            surface_points_img[:,0] = np.clip(surface_points_img[:,0], 0, x_size[idx]-1)
            surface_points_img[:,1] = np.clip(surface_points_img[:,1], 0, y_size[idx]-1)
            
            height = surface_points[:, 2]
            height_map[idx][surface_points_img[:,1], surface_points_img[:,0]] = height
            
        # cv2.imwrite("height_map.jpg", height_map[idx] * 255)        
        results['height_map'] = height_map
        return results
    
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