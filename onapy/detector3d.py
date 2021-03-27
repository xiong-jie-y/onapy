from collections import deque
from onapy.tracker2d import TrackDetectionFusedTracker
from onapy.tracker3d import create_tracker, get_tracker_names
import time
import click

import cv2

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)

import cupoch as cph
from mmcv.runner import checkpoint
import numpy as np
import open3d as o3d
from remimi.datasets.open3d import Open3DReconstructionDataset
from remimi.visualizers.sixdof import OnahoPointCloudVisualizer

from mmdet.apis import inference_detector, init_detector


def compute_projection(points_3D,internal_calibration):
    points_3D = points_3D.T
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

class OnahoBoundingBox3DDetector:
    def __init__(self, intrinsic, K):
        self.intrinsic = intrinsic
        self.K = K

    def filter_by_instance_mask(self, pcd, result):
        start_pf_filter = time.time()
        points_in_seg = []
        colors = []
        start_proj = time.time()
        proj_points = compute_projection(np.array(pcd.points), self.K)
        end_proj = time.time()
        print(f"proj: {end_proj - start_proj}")
        segmentations = result.mask
        print(len(pcd.points))
        for proj_point, point in zip(proj_points.T, pcd.points):
            # if point[2] > 0.8:
            #     # colors.append([0 , 0, 1.0])
            #     continue
            found = False
            y = int(proj_point[1])
            x = int(proj_point[0])
            for seg_list in segmentations:
                for seg in seg_list:
                    # mport IPython; IPython.embed()
                    if seg[y, x]:
                        found = True
            if found:
                points_in_seg.append(point)
                # colors.append([1.0, 0, 0])
                # cv2.circle(color_image,(x,y), 5, (0, 0, 255), -1)
            else:
                pass
                # colors.append([0 , 0, 1.0])
                # cv2.circle(color_image,(x,y), 5, (255, 0, 0), -1)

        end_pr_filter = time.time()
        print(f"Point Cloud Filtering: {end_pr_filter - start_pf_filter}")

        return points_in_seg

    def filter_by_bounding_box(self, pcd, result):
        start_pf_filter = time.time()
        points_in_seg = []
        colors = []
        start_proj = time.time()
        proj_points = compute_projection(np.array(pcd.points), self.K)
        end_proj = time.time()
        print(f"proj: {end_proj - start_proj}")

        top_confidence_one = result.bounding_box
            
        print(len(pcd.points))
        for proj_point, point in zip(proj_points.T, pcd.points):
            # if point[2] > 0.8:
            #     # colors.append([0 , 0, 1.0])
            #     continue
            found = False
            y = int(proj_point[1])
            x = int(proj_point[0])
            if top_confidence_one is not None and top_confidence_one[0] < x < top_confidence_one[2] and top_confidence_one[1] < y < top_confidence_one[3]:
                found = True
            if found:
                points_in_seg.append(point)
                # colors.append([1.0, 0, 0])
                # cv2.circle(color_image,(x,y), 5, (0, 0, 255), -1)
            else:
                pass
                # colors.append([0 , 0, 1.0])
                # cv2.circle(color_image,(x,y), 5, (255, 0, 0), -1)

        end_pr_filter = time.time()
        print(f"Point Cloud Filtering: {end_pr_filter - start_pf_filter}")

        return points_in_seg

    def get_onaho_3d_bounding_box(self, color_image, depth_image, result):
    #     color_image = cv2.imread(color_file)
        # depth = o3d.io.read_image(depth_image)
        depth = o3d.geometry.Image(depth_image)

        color = o3d.geometry.Image(color_image)
        # if project_semantic_to_point_cloud:
        #     color = o3d.geometry.Image(seg_image)
        # else:
        #     color = o3d.io.read_image(color_file)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000,
            depth_trunc=0.5,
            convert_rgb_to_intensity=False)
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.intrinsic)

        pcd = pcd.voxel_down_sample(voxel_size = 0.01)

        points_in_seg = []
        if result.bounding_box is not None:
            points_in_seg = self.filter_by_instance_mask(pcd, result)
    #         points_in_seg = self.filter_by_bounding_box(pcd, result)

        start_clustering = time.time()
        # pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        # print(len(points_in_seg))
        made_from_2d_detection = False
        if len(points_in_seg) > 0:
            target_points = np.array(points_in_seg)
            made_from_2d_detection = True
        else:
            target_points = np.array(pcd.points)

        gpu_cloud = cph.geometry.PointCloud(target_points)
        labels = np.array(
            gpu_cloud.cluster_dbscan(eps=0.04, min_points=30, print_progress=True).cpu())

        from collections import defaultdict
        groups = defaultdict(list)
        for i, label in enumerate(labels):
            groups[label].append(target_points[i])

        # print(len(groups))

        min_distance = 1000000000
        closest_bounding_box = None
        for id, points in groups.items():
            if len(points) < 4:
                continue
            try:
                np_points = np.array(points)
                bounding_box = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np_points)).get_oriented_bounding_box()
            except RuntimeError:
                continue

            camera_to_obj_distance = np.linalg.norm(bounding_box.center)
            if min_distance > camera_to_obj_distance:
                min_distance = camera_to_obj_distance
                closest_bounding_box = bounding_box

        end_clustering = time.time()
        print(f"Clustering: {end_clustering - start_clustering}")

        # seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)

        return closest_bounding_box, made_from_2d_detection, pcd