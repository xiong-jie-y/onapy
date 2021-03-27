
from collections import deque
import os
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
# from motpy import# Detection, MultiObjectTracker, NpImage, Box

class DetectionResult:
    def __init__(self, result):
        self.result = result

    # def get_bounding_boxes(self):
    #     return [det for det in result[0] if det.shape[0]  > 0]

    def get_most_confident_detection(self):
        segmentations = self.result[0]

        top_confidence_one = None
        max_confidence = 0.2
        for bounding_box_wrap in segmentations:
            if bounding_box_wrap.shape[0] == 0:
                continue
            bounding_box = bounding_box_wrap[0]
            if bounding_box[4] > max_confidence:
                top_confidence_one = bounding_box
                max_confidence = bounding_box[4]

        return top_confidence_one
        # if top_confidence_one[4] > 0.40:
        #     return top_confidence_one
        # else:
        #     return None

    def get_mask(self):
        return self.result[1]

class Detection2D:
    def __init__(self, bounding_box, mask):
        self.bounding_box = bounding_box
        self.mask = mask

MODEL_ROOT_PATH = os.path.expanduser("~/.cache/onapy/models")

import gdown

class TrackDetectionFusedTracker:
    def __init__(self):
        device = 'cuda:0'

        config_file = os.path.join(os.path.dirname(__file__), "configs/yolact_onaho.py")
        checkpoint_file = os.path.join(MODEL_ROOT_PATH, "models/onaho_model.pth")

        if not os.path.exists(checkpoint_file):
            os.makedirs(os.path.dirname(checkpoint_file))
            gdown.download(
                "https://drive.google.com/uc?id=1BhFSaFhk_w0BTHrstSTMYCTrMh6FE-Gw",
                checkpoint_file
            )

        self.model = init_detector(config_file, checkpoint_file, device=device)

        # self.tracker = cv2.legacy.TrackerMedianFlow_create()
        self.tracker_initialized = False
        # previous_result = None

        # model_spec = {'order_pos': 1, 'dim_pos': 2,
        #                 'order_size': 0, 'dim_size': 2,
        #                 'q_var_pos': 5000., 'r_var_pos': 0.1}

        # dt = 1 / 30.0  # assume 15 fps
        # tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)

    def get_onaho_bounding_box(self, color_image):
        start_inference = time.time()
        result = inference_detector(self.model, color_image)

            # This should be ignored.
            # import IPython; IPython.embed()
            # return Detection2D(bounding_box=None, mask=None)
        # detection_image = self.model.show_result(
        #         color_image, result, score_thr=0.40, wait_time=0, show=False)
        result = DetectionResult(result)

        bounding_box = result.get_most_confident_detection()

        result_detection = Detection2D(bounding_box=bounding_box, mask=result.get_mask())
        # if bounding_box is not None:
        #     x1, y1, x2, y2, confidence = bounding_box
        #     self.last_ok = self.tracker.init(color_image, tuple([x1, y1, x2 - x1, y2 - y1]))
        #     result_detection = Detection2D(bounding_box=bounding_box)
        # else:
        #     ok, bbox = self.tracker.update(color_image)
        #     x, y, w, h = bbox
        #     bd = np.array([x, y, x + w, y + h])
        #     if ok:
        #         result_detection = Detection2D(bounding_box=bd)

        # if not self.tracker_initialized:
        #     if bounding_box is not None:
        #         # Only use first
        #         x1, y1, x2, y2, confidence = bounding_box
        #         ok = self.tracker.init(color_image, tuple([x1, y1, x2 - x1, y2 - y1]))
        #         if ok:
        #             self.tracker_initialized = True

        #         # Only usable result is detection.
        #         result_detection = Detection2D(bounding_box=bounding_box[:4])
        # else:
        #     # if bounding_box is not None:
        #     ok, bbox = self.tracker.update(color_image)

        #     if ok:
        #         x, y, w, h = bbox
        #         bd = np.array([x, y, x + w, y + h])
        #         if bounding_box is not None:
        #             result_detection = Detection2D(bounding_box=(bd + bounding_box[:4]) / 2)
        #             if (bd[2] - bd[0]) - (bounding_box[2] - bounding_box[0]) > 40 or (bd[3] - bd[1]) - (bounding_box[3] - bounding_box[1]) > 40:
        #                 x1, y1, x2, y2, confidence = bounding_box
        #                 ok = self.tracker.init(color_image, tuple([x1, y1, x2 - x1, y2 - y1]))
        #         else:
        #             result_detection = Detection2D(bounding_box=bd)
        #         # result_detection = Detection2D(bounding_box=bounding_box[:4])
        #     else:
        #         if bounding_box is not None:
        #             result_detection = Detection2D(bounding_box=bounding_box[:4])

        end_inference = time.time()
        print(f"Inference: {end_inference - start_inference}")

        if result_detection.bounding_box is not None:
            # import IPython; IPython.embed()
            try:
                cv2.rectangle(color_image,tuple(result_detection.bounding_box[:2].astype(int)), tuple(result_detection.bounding_box[[2, 3]].astype(int)) ,(0,255,0),3)
            except TypeError:
                import IPython; IPython.embed()
        cv2.imshow("dtec", color_image)

        # cv2.imshow("det", detection_image)

        return result_detection
        # try:
        #     tracker.step([Detection(box=det[0][:4], score=det[0][4]) for det in result[0] if det.shape[0]  > 0])
        # except:
        #     import IPython; IPython.embed()
        # tracks = tracker.active_tracks(min_steps_alive=1)
        # if len(tracks) > 0:
        #     # import IPython; IPython.embed()
        #     result = ([np.array([tracks[0].box.tolist() + [tracks[0].score]])], result[1])

        # print(tracks)
    # import IPython; IPython.embed()

        # import IPython; IPython.embed()
        # seg_image = segmenter.convert_to_semantic_image(color_image)
