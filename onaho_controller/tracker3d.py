import re

import open3d as o3d
import numpy as np

# from motpy import Detection, MultiObjectTracker, NpImage, Box

class SimpleTracker:
    """Handcrafted 3d tracker to track a object th at is held by hand."""

    RED = np.array([1, 0, 0])
    GREEN = np.array([0, 1, 0])

    def __init__(self):
        self.previous_previous_bounding_box = None
        self.previous_bounding_box = None

    def get_tracked_onahole(self, closest_bounding_box: o3d.geometry.OrientedBoundingBox, made_from_2d_detection: bool):
        """Track and returns most probable bounding box for single object held by hand.

        It is optimized to heuristically track hand holded object.
        For example, it uses heuristic of physical hand speed to eliminate wrong bounding box.
        """
        # aaa = [pcd]

        if closest_bounding_box is not None:
            new_box = o3d.geometry.OrientedBoundingBox(closest_bounding_box)
            if made_from_2d_detection:
                new_box.color = self.RED
            else:
                if self.previous_bounding_box is not None:
                    if self.previous_previous_bounding_box is not None:
                        new_box.extent = self.previous_bounding_box.extent + (self.previous_bounding_box.extent - self.previous_previous_bounding_box.extent)
                        new_box.center = self.previous_bounding_box.center + (self.previous_bounding_box.center - self.previous_previous_bounding_box.center)
                        new_box.color = self.GREEN
                    else:
                        new_box.extent = self.previous_bounding_box.extent
                        new_box.center = self.previous_bounding_box.center
                        new_box.color = self.GREEN
                else:
                    pass
                    # import IPython; IPython.embed()

            # For trajectory visualization.
            # aaa.append(closest_bounding_box)
            # box = o3d.geometry.OrientedBoundingBox()
            # box.center = closest_bounding_box.center
            # box.extent = [0.05, 0.05, 0.05]

            # visualizer.vis.add_geometry(box)

            self.previous_previous_bounding_box = self.previous_bounding_box
            self.previous_bounding_box = o3d.geometry.OrientedBoundingBox(new_box)

            return new_box
        else:
            return None

class KalmanMultiTracker:
    def __init__(self):
        # model_spec = {'order_pos': 2, 'dim_pos': 3,
        #         'order_size': 0, 'dim_size': 3,
        #         'q_var_pos': 0.05, 'r_var_pos': 0.05}
        model_spec = {'order_pos': 0, 'dim_pos': 3,
                'order_size': 0, 'dim_size': 3,
                'q_var_pos': 0.01, 'r_var_pos': 0.01}
        dt = 1 / 30.0
        self._multi_tracker = MultiObjectTracker(
            dt=dt, model_spec=model_spec,
            active_tracks_kwargs={'min_steps_alive': 0, 'max_staleness': 12},
            tracker_kwargs={'max_staleness': 12}
        )
        
    def _get_longest_life_track(self):
        tracks = self._multi_tracker.active_tracks(min_steps_alive=3)

        longest_life_track = None
        longest_life_length = 0

        for track in tracks:
            if track.tracker.steps_positive > longest_life_length:
                longest_life_length = track.tracker.steps_positive
                longest_life_track = track

        if longest_life_track is not None:
            min_p = longest_life_track.box[:3]
            max_p = longest_life_track.box[3:]
            bounding_box = o3d.geometry.OrientedBoundingBox()
            extent = max_p - min_p
            bounding_box.extent = extent
            bounding_box.center = min_p + extent / 2.0

            return bounding_box
        else:
            return None

    def get_tracked_onahole(self, closest_bounding_box: o3d.geometry.OrientedBoundingBox, made_from_2d_detection: bool):
        if closest_bounding_box is not None:
            if made_from_2d_detection:
                half_size = closest_bounding_box.extent / 2.0
                min_p = closest_bounding_box.center - half_size
                max_p = closest_bounding_box.center + half_size

                self._multi_tracker.step(detections=[Detection(
                    box = min_p.tolist() + max_p.tolist(),
                    score = 1.0
                )])
                return self._get_longest_life_track()
            else:
                self._multi_tracker.step(detections=[])
                return self._get_longest_life_track()
        else:
            self._multi_tracker.step(detections=[])
            return self._get_longest_life_track()

trackers = {}
for tracker in [SimpleTracker, KalmanMultiTracker]:
    tracker_name = name = re.sub(r'(?<!^)(?=[A-Z])', '_', tracker.__name__).lower()
    trackers[tracker_name] = tracker

print(trackers)

def get_tracker_names():
    return list(trackers.keys())

def create_tracker(tracker_name):
    return trackers.get(tracker_name, None)()