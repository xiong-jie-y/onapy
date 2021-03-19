import warnings
warnings.resetwarnings()
warnings.filterwarnings('ignore')


from onaho_controller.motion_recognizer import MotionLineDetector, OnahoStateEstimator, VelocityBasedInsertionEstimator
from onaho_controller.detector3d import OnahoBoundingBox3DDetector
from onaho_controller.tracker2d import TrackDetectionFusedTracker
from onaho_controller.tracker3d import create_tracker, get_tracker_names
import time
import click

import sys

import cv2

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)

from remimi.datasets.open3d import Open3DReconstructionDataset
from remimi.visualizers.sixdof import OnahoPointCloudVisualizer

from mmdet.apis import inference_detector, init_detector
from remimi.sensors.realsense import RealsenseD435i

@click.command()
@click.option("--tracker-name", required=True, default=get_tracker_names()[0])
@click.option("--debug-mode", is_flag=True)
# @click.option("--insertion-estimator", required=True, default=get_tracker_names()[0])
def main(tracker_name, debug_mode):
    # Just for reading intrinsics.
    src_dataset = Open3DReconstructionDataset("configs")
    intrinsic = src_dataset.get_intrinsic("open3d")

    K = src_dataset.get_intrinsic("matrix")

    visualizer = OnahoPointCloudVisualizer()

    # segmenter = SemanticSegmenter()

    project_semantic_to_point_cloud = False

    current_posiitons = []

    camera = RealsenseD435i()
    tracker = TrackDetectionFusedTracker()

    bounding_box_3d_tracker = create_tracker(tracker_name)
    bounding_box_3D_detector = OnahoBoundingBox3DDetector(intrinsic, K)

    motion_line_detector = MotionLineDetector()
    # state_estimator = OnahoStateEstimator()
    state_estimator = VelocityBasedInsertionEstimator()

    import warnings
    warnings.resetwarnings()
    warnings.filterwarnings('ignore')


    while True:
        # camera.capture()
        # frame, depth = camera.get_color_and_depth()
        color_image, depth_image = camera.get_color_and_depth()

    # for color_file, depth_file in list(zip(
    #     src_dataset.get_rgb_paths(), src_dataset.get_depth_paths())):

        result = tracker.get_onaho_bounding_box(color_image)
        closest_bounding_box, made_from_2d_detection, pcd = \
            bounding_box_3D_detector.get_onaho_3d_bounding_box(color_image, depth_image, result)

        tracked_onahole = bounding_box_3d_tracker.get_tracked_onahole(closest_bounding_box, made_from_2d_detection)
        if tracked_onahole is not None:
            line = motion_line_detector.get_next_line(tracked_onahole.center)

            if line is not None:
                visualizer.update_axis([tracked_onahole.center, tracked_onahole.center + line * 4])
                state_estimator.add_current_state(line, tracked_onahole.center)
                # current_posiitons.append(state_estimator.current_position)
                if state_estimator.state == state_estimator.OnahoState.INSERTING:
                    tracked_onahole.color = [1, 0, 0]
                elif state_estimator.state == state_estimator.OnahoState.OUTGOING:
                    tracked_onahole.color = [0, 0, 1]
                elif state_estimator.state == state_estimator.OnahoState.NO_MOTION:
                    tracked_onahole.color = [0, 0, 0]
            visualizer.update_bounding_box(tracked_onahole)

        visualizer.update_pcd(pcd)

        if debug_mode:
            cv2.imshow("ColorImage", color_image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            import matplotlib.pyplot as plt
            # plt.plot(list(range(0, len(current_posiitons))), current_posiitons)
            # plt.show()
            plt.plot(state_estimator.timestamps, state_estimator.velocities)
            plt.show()

if __name__ == "__main__":
    main()