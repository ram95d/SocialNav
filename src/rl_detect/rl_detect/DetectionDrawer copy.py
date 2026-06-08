import cv2
import numpy as np
import open3d as o3d
import pdb
import time
import os
from scipy.stats import zscore
class DetectionDrawer():

    def __init__(self, class_names, pinhole_camera_intrinsic, extrinsic, intr):
        self.class_names = class_names
        num_classes = len(class_names)
        self.pinhole_camera_intrinsic = pinhole_camera_intrinsic
        self.extrinsic = extrinsic
        self.intr = intr
        self.colors = self.generate_colors(num_classes)
        self.flip_matrix = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
    def generate_colors(self, num_classes):
        colors = []
        for _ in range(num_classes):
            color = np.random.rand(3) * 255 
            if np.mean(color) < 128: 
                color = color + (255 - np.mean(color)) * 0.5
            colors.append(color)
        return np.array(colors)
    def __call__(self, depth_image, masks, scores, class_ids):
        return self.draw_detections(depth_image, masks, scores, class_ids)
    def bool_mask_to_int(self, mask, true_value=(255, 0, 0)):
        true_value = np.array(true_value)
        return mask.astype(np.uint8)[:, :, None] * true_value
    def draw_detections(self, depth_image, masks, scores, class_ids):
        if class_ids.shape[0] == 0:
            return [], o3d.geometry.PointCloud()

        bbox_3d = []
        pcl = o3d.geometry.PointCloud()
        colors = self.colors[class_ids]

        for i, color in enumerate(colors):
            mask = masks[i]

            if mask.ndim == 3:
                mask = mask.squeeze(0)

            mask = (mask > 0.5).astype(np.uint8)

            if mask.shape != depth_image.shape:
                mask = cv2.resize(
                    mask,
                    (depth_image.shape[1], depth_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # print("mask shape:", mask.shape, "depth shape:", depth_image.shape)

            bx3d, depth, pcd = self.draw_3d_bounding_box(depth_image, mask)

            if len(pcd.points) == 0:
                continue

            normalized_color = [c / 255.0 for c in color]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(np.array(normalized_color), (len(pcd.points), 1))
            )

            pcl.points.extend(pcd.points)
            pcl.colors.extend(pcd.colors)
            bbox_3d.append(bx3d)

        return bbox_3d, pcl

    # def draw_3d_bounding_box(self, depth_image, color_mask):
    #     pcd = o3d.geometry.PointCloud()
    #     mask = color_mask.astype(np.uint8)

    #     eroded_ann_mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
    #     isolated_depth = np.where((eroded_ann_mask > 0) & (depth_image > 0), depth_image, np.nan)

    #     valid = ~np.isnan(isolated_depth)
    #     if np.count_nonzero(valid) < 20:
    #         return o3d.geometry.AxisAlignedBoundingBox(), np.nan, pcd

    #     depth = np.median(isolated_depth[valid])
    #     non_nan_points = np.argwhere(valid)
    #     non_nan_depth_values = isolated_depth[non_nan_points[:, 0], non_nan_points[:, 1]]        
    #     if len(non_nan_depth_values) < 5:
    #         return o3d.geometry.AxisAlignedBoundingBox(), depth, pcd

    #     z_scores = zscore(non_nan_depth_values)
    #     filtered_indices = np.abs(z_scores) <= 2.5

    #     non_nan_depth_values = non_nan_depth_values[filtered_indices]
    #     non_nan_points = non_nan_points[filtered_indices]

    #     if len(non_nan_depth_values) < 5:
    #         return o3d.geometry.AxisAlignedBoundingBox(), depth, pcd

    #     # Intrinsics
    #     fx = self.intr.fx
    #     fy = self.intr.fy
    #     cx = self.intr.ppx
    #     cy = self.intr.ppy

    #     # Convert depth image to 3D point cloud
    #     u = non_nan_points[:, 1]  # x coordinates in image
    #     v = non_nan_points[:, 0]  # y coordinates in image
    #     z = non_nan_depth_values * 0.001 

    #     x = (u - cx) * z / fx
    #     y = (v - cy) * z / fy

    #     points = np.stack((x, y, z), axis=-1)
    #     pcd.points = o3d.utility.Vector3dVector(points)        
    #     if len(points) < 5:
    #         return o3d.geometry.AxisAlignedBoundingBox(), depth, pcd

    #     pcd.points = o3d.utility.Vector3dVector(points)

    #     final_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=2.0)
    #     if len(final_pcd.points) == 0:
    #         return o3d.geometry.AxisAlignedBoundingBox(), depth, pcd

    #     bbox_3d = final_pcd.get_axis_aligned_bounding_box()
    #     bbox_3d.color = (0, 0, 1)
    #     return bbox_3d, depth, final_pcd
    def draw_3d_bounding_box(self, depth_image, color_mask):
        pcd = o3d.geometry.PointCloud()
        mask = color_mask.astype(np.uint8)

        eroded_ann_mask = cv2.erode(mask, kernel=np.ones((5, 5), np.uint8), iterations=2)  # erode more aggressively
        isolated_depth = np.where((eroded_ann_mask > 0) & (depth_image > 0), depth_image, np.nan)

        valid = ~np.isnan(isolated_depth)
        if np.count_nonzero(valid) < 20:
            return o3d.geometry.AxisAlignedBoundingBox(), np.nan, pcd

        depth_values = isolated_depth[valid]

        # ── Percentile clipping instead of z-score ──────────────────────────────
        # Clip to [5th, 95th] percentile — removes background AND foreground noise
        p_low  = np.percentile(depth_values, 5)
        p_high = np.percentile(depth_values, 95)
        depth_values_clipped = depth_values[
            (depth_values >= p_low) & (depth_values <= p_high)
        ]

        if len(depth_values_clipped) < 5:
            return o3d.geometry.AxisAlignedBoundingBox(), np.nan, pcd

        depth = np.median(depth_values_clipped)

        # Reconstruct valid point indices using clipped range
        clipped_mask = (
            (eroded_ann_mask > 0) &
            (depth_image > 0) &
            (depth_image >= p_low) &
            (depth_image <= p_high)
        )
        non_nan_points = np.argwhere(clipped_mask)
        non_nan_depth_values = depth_image[clipped_mask].astype(np.float32)

        # Intrinsics
        fx = self.intr.fx;  fy = self.intr.fy
        cx = self.intr.ppx; cy = self.intr.ppy

        u = non_nan_points[:, 1]
        v = non_nan_points[:, 0]
        z = non_nan_depth_values * 0.001

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack((x, y, z), axis=-1)
        pcd.points = o3d.utility.Vector3dVector(points)

        final_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
        if len(final_pcd.points) == 0:
            return o3d.geometry.AxisAlignedBoundingBox(), depth, pcd

        bbox_3d = final_pcd.get_axis_aligned_bounding_box()
        bbox_3d.color = (0, 0, 1)
        return bbox_3d, depth, final_pcd