from email.mime import image
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
                            'can_bus', 'height_map', 'height_mask', 'cam_intrinsic', 'lidar2cam',
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
        self.image_reatify = ImageReactify(target_roll=[0.0,], pitch_abs=None)
        
    def __call__(self, results):
        height_map = [np.zeros((img.shape[0], img.shape[1])) for idx, img in enumerate(results['img'])]
        images = [img for idx, img in enumerate(results['img'])]

        y_size = [int(img.shape[0]) for img in results['img']]
        x_size = [int(img.shape[1]) for img in results['img']]
        gt_bboxes_3d  = torch.clone(results["gt_bboxes_3d"].tensor).numpy()
        
        for idx in range(len(height_map)):
            lidar2img = results["lidar2img"][idx]
            lidar2cam = results["lidar2cam"][idx]
            bev_img = self.ground_ref_points(images[0], gt_bboxes_3d, lidar2img, x_size, y_size, idx)

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
            
            images[idx][surface_points_img[:,1], surface_points_img[:,0]] = (255,0,0)
            roll, pitch = self.image_reatify.parse_roll_pitch(lidar2cam)

        frame_idx = results["sample_idx"].split('/')[1].split('.')[0]        
        cv2.imwrite(os.path.join("debug", frame_idx + "_K_" + str(round(roll, 2)) + ".jpg"), images[0])
        cv2.imwrite(os.path.join("debug", frame_idx + "_K_bev_img_" + str(round(roll, 2)) + ".jpg"), bev_img)
        
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
    
    def ground_ref_points(self, image, gt_bboxes_3d, lidar2img, x_size, y_size, idx):
        center_lidar = gt_bboxes_3d[0][:3]
        w, l = 100, 60
        shape_top = np.array([w / self.resolution, l / self.resolution]).astype(np.int32)
        bev_img = np.zeros((shape_top[1], shape_top[0], 3))
        bev_img = bev_img.reshape(-1, 3)
        
        n, m = [(ss - 1.) / 2. for ss in shape_top]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        xv, yv = np.meshgrid(x, y, sparse=False)
        xyz = np.concatenate((xv[:,:,np.newaxis], yv[:,:,np.newaxis], (center_lidar[2] * np.ones_like(xv)[:,:,np.newaxis] / self.resolution).astype(np.int32)), axis=-1)
        xyz = xyz + np.array([(0.5 * w + 0.5)/ self.resolution, 0.0, 0.0]).astype(np.int32).reshape(1,3)        
        xyz_points = xyz * self.resolution
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
    
@PIPELINES.register_module()
class ImageReactify(object):
    def __init__(self, target_roll, pitch_abs):
        self.target_roll = target_roll
        self.pitch_abs = pitch_abs

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
    
    def reactify_roll_params(self, lidar2cam, cam_intrinsic, image, roll_status, target_roll=[-0.48,]):
        if len(target_roll) > 1:
            target_roll_status = np.random.uniform(target_roll[0], target_roll[1])
        else:
            target_roll_status = target_roll[0]
        
        roll = target_roll_status - roll_status        
        roll_rad = self.degree2rad(roll)
        rectify_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0, 0], 
                                 [math.sin(roll_rad), math.cos(roll_rad), 0, 0], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        lidar2cam_rectify = np.matmul(rectify_roll, lidar2cam)
        lidar2img_rectify = (cam_intrinsic @ lidar2cam_rectify)
        
        M = self.get_M(lidar2cam[:3,:3], cam_intrinsic[:3,:3], lidar2cam_rectify[:3,:3], cam_intrinsic[:3,:3])
        image = self.transform_with_M(image, M)
        '''
        h, w, _ = image.shape
        center = (w // 2, h // 2)
        center = (int(cam_intrinsic[0, 2]), int(cam_intrinsic[1, 2]))
        M = cv2.getRotationMatrix2D(center, -1 * self.rad2degree(roll_rad), 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        '''
        return lidar2cam_rectify, lidar2img_rectify, image
    
    def reactify_pitch_params(self, lidar2cam, cam_intrinsic, pitch, pitch_abs=2.0):
        target_pitch_status = np.random.uniform(pitch - pitch_abs, pitch + pitch_abs)
        pitch = target_pitch_status - pitch
        pitch = self.degree2rad(pitch)
        rectify_pitch = np.array([[1, 0, 0, 0],
                                  [0,math.cos(pitch), -math.sin(pitch), 0], 
                                  [0,math.sin(pitch), math.cos(pitch), 0],
                                  [0, 0, 0, 1]])
        lidar2cam_rectify = np.matmul(rectify_pitch, lidar2cam)
        lidar2img_rectify = (cam_intrinsic @ lidar2cam_rectify)
        return lidar2cam_rectify, lidar2img_rectify

    def rad2degree(self, radian):
        return radian * 180 / np.pi
    
    def degree2rad(self, degree):
        return degree * np.pi / 180
    
    def __call__(self, results):
        for idx, image in enumerate(results['img']):
            lidar2cam = results["lidar2cam"][idx].copy()            
            cam_intrinsic = results["cam_intrinsic"][idx].copy()
            image = results["img"][idx].copy()
            
            roll_init, pitch_init = self.parse_roll_pitch(lidar2cam)
            lidar2cam_roll_rectify, lidar2img_rectify, image = self.reactify_roll_params(lidar2cam, cam_intrinsic, image, roll_init, target_roll=self.target_roll)
            lidar2cam_rectify = lidar2cam_roll_rectify
            
            if self.pitch_abs is not None:
                lidar2cam_pitch_rectify, lidar2img_rectify = self.reactify_pitch_params(lidar2cam_roll_rectify, cam_intrinsic, pitch_init, pitch_abs=self.pitch_abs)            
                lidar2cam_rectify = lidar2cam_pitch_rectify
                M = self.get_M(lidar2cam_roll_rectify[:3,:3], cam_intrinsic[:3,:3], lidar2cam_pitch_rectify[:3,:3], cam_intrinsic[:3,:3])
                image = self.transform_with_M(image, M)
                
            '''
            roll_check, pitch_check = self.parse_roll_pitch(lidar2cam_rectify)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (300, 100)
            org_1 = (300, 400)
            fontScale = 3   
            color = (0, 0, 255)
            thickness = 2
            image = cv2.putText(image, str(round(roll_check, 2)) +" " + str(round(pitch_check, 2)), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
            image = cv2.putText(image, str(round(roll_init, 2)) +" " + str(round(pitch_init ,2)), org_1, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
            '''
            results["lidar2cam"][idx] = lidar2cam_rectify
            results["cam_intrinsic"][idx] = cam_intrinsic
            results["lidar2img"][idx] = lidar2img_rectify
            results["img"][idx] = image
              
        return results
    
    def get_M(self, R, K, R_r, K_r):
        R_inv = np.linalg.inv(R)
        K_inv = np.linalg.inv(K)
        M = np.matmul(K_r, R_r)
        M = np.matmul(M, R_inv)
        M = np.matmul(M, K_inv)
        return M
    
    def transform_with_M_shift(self, image, M):
        ref_center = np.array([image.shape[1]//2, image.shape[0]//2, 1.0]).reshape(3,1)
        ref_center_new = np.matmul(M, ref_center)
        ref_center_new = ref_center_new[:2, 0] / ref_center_new[2, 0]
        ref_center_new = ref_center_new.astype(np.int32)
        
        shift = ref_center_new[1] - ref_center[1]
        height, width, _ = image.shape
        mat = np.float32([[1, 0, 0], [0, 1, shift]])
        image = cv2.warpAffine(image, mat, (width, height))
        return image
                    
    def transform_with_M(self, image, M):
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
    
@PIPELINES.register_module()
class ImageRangeFilter(object):
    """Filter objects outside image view.
    Args:
        bool object center or eight corners.
    """
    def __init__(self, use_center=False):
        self.use_center = use_center

    def __call__(self, results):
        """Call function to filter objects by the range.
        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = results['gt_bboxes_3d']
        gt_labels_3d = results['gt_labels_3d']
        
        image = results["img"][0]
        lidar2img = results["lidar2img"][0]
        gt_bboxes_3d_np = gt_bboxes_3d.tensor.clone().numpy()        
        if gt_bboxes_3d_np.shape[0] == 0:
            return results
        indices = []
        for obj_id in range(gt_bboxes_3d_np.shape[0]):
            gt_bbox = gt_bboxes_3d_np[obj_id]
            yaw_lidar = gt_bbox[6]
            lwh = gt_bbox[3:6]
            center_lidar = gt_bbox[:3] + [0.0, 0.0, 0.5 * lwh[2]]
            if self.use_center:
                # center-based filter
                center_lidar = np.array(center_lidar)[:, np.newaxis]
                center_lidar_extend = np.concatenate((center_lidar, np.ones((1, 1))), axis=0)
                center_img = np.matmul(lidar2img, center_lidar_extend).T
                center_img_uv = center_img[0, :2] / center_img[0, 2]
                if center_img_uv[0] > 0 and center_img_uv[0] < image.shape[1] - 1 and center_img_uv[1] > 0 and center_img_uv[1] < image.shape[0] - 1:
                    indices.append(True)
                else:
                    indices.append(False)
            else:
                # corner-based filter
                bottom_center = gt_bbox[:3]
                lidar_8_points = compute_box_3d(lwh, bottom_center, yaw_lidar)
                lidar_8_points_extend = np.concatenate((lidar_8_points, np.ones((lidar_8_points.shape[0],1))), axis=1)
                img_8_points = np.matmul(lidar2img, lidar_8_points_extend.T).T
                uv_8_points = img_8_points[:,:2] /img_8_points[:,2][:,np.newaxis]
                if np.any(uv_8_points[:,0] > 0) and np.any((uv_8_points[:,0]) < image.shape[1] - 1) and np.any(uv_8_points[:,1] > 0) and np.any(uv_8_points[:,1] < image.shape[0] -1):
                    indices.append(True)
                else:
                    indices.append(False)
        mask = torch.tensor(np.array(indices), device=gt_bboxes_3d.tensor.device)
        gt_bboxes_3d = gt_bboxes_3d[mask]        
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]
        results['gt_bboxes_3d'] = gt_bboxes_3d
        results['gt_labels_3d'] = gt_labels_3d
        return results