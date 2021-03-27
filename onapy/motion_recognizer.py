from collections import deque
import enum
import time
import os

import numpy as np

from skspatial.objects import Line
from skspatial.objects import Points

class MotionLineDetector:
    MAX_POINT = 60

    def __init__(self):
        self.past_points = deque([])
        self.last_time = None

    def get_next_line(self, point):
        self.past_points.append(point)

        if len(self.past_points) > self.MAX_POINT:
            self.past_points.popleft()

        self.last_time = time.time()

        if len(self.past_points) == self.MAX_POINT:
            max_movement = 0
            for pt2, pt1 in zip(list(self.past_points)[1:], list(self.past_points)[:-1]):
                movement = np.linalg.norm(pt2 - pt1)
                if movement > max_movement:
                    max_movement = movement

            if (max_movement / (time.time() - self.last_time)) < 0.1:
                return None

            points = Points(list(self.past_points))

            line_fit = Line.best_fit(points)
            direction = np.array(line_fit.direction)

            # I defined this side will be the positive direction.
            if direction[0] < 0:
                direction *= -1

            direction = direction / np.linalg.norm(direction)
            
            return direction
        else:
            return None

from pydub import AudioSegment
from pydub.playback import play
import threading
import glob

# Load mp3s.
songs = [AudioSegment.from_mp3(sound_path) for sound_path in glob.glob("sounds/*.mp3")]

def play_ex():
    song_index = np.random.randint(0, len(songs))
    play(songs[song_index])

class OnahoStateEstimator:
    MAX_FRAME = 30 * 2
    WAIT_TIME = 30
    def __init__(self):
        self.previous_center = None
        self.current_position = 0
        self.recent_positions = deque([])
        self.remaining_wait = 0

    def add_current_state(self, line, center):
        if self.previous_center is not None:
            move_distance = np.dot(line, center - self.previous_center)
            self.current_position += move_distance

        self.previous_center = center

        if len(self.recent_positions) == self.MAX_FRAME:
            self.recent_positions.popleft()

        self.recent_positions.append(self.current_position)

        min_pos = min(self.recent_positions)
        max_pos = max(self.recent_positions)

        rate = (max_pos - min_pos) * 0.5

        print(max_pos - min_pos)
        
        if (max_pos - min_pos) > 0.05:
            if min_pos > self.current_position - rate and self.remaining_wait <= 0:
                t = threading.Thread(target=play_ex)
                t.start()
                self.remaining_wait = 30
            self.remaining_wait -= 1

from dataclasses import dataclass

@dataclass
class VelocityBasedInsertionEstimatorOption:
    forward_backward_velocity_threashold: float = 0.35
    no_motion_velocity_threashold: float = 0.20
    sound_wait_time: int = 5

class VelocityBasedInsertionEstimator:
    
    class OnahoState(enum.Enum):
        INSERTING = "inserting"
        OUTGOING = "outgoing"
        NO_MOTION = "no_motion"

    def __init__(self, sound_dir, option=VelocityBasedInsertionEstimatorOption()):
        self.previous_center = None
        self.previous_time = None
        self.remaining_wait = 0
        self.state = self.OnahoState.NO_MOTION
        self.option = option
        self.songs = [
            AudioSegment.from_mp3(sound_path) 
            for sound_path in glob.glob(os.path.join(sound_dir, "*.mp3"))]

        # For Debug.
        self.velocities = []
        self.timestamps = []

    def play_ex(self):
        song_index = np.random.randint(0, len(self.songs))
        play(self.songs[song_index])

    def add_current_state(self, line, center):
        # Just skip at the first frame.
        if self.previous_center is not None and self.previous_time is not None:
            delta_t = time.time() - self.previous_time

            move_distance = np.dot(line, center - self.previous_center)
            velocity = move_distance / delta_t
            if velocity > self.option.forward_backward_velocity_threashold:
                self.state = self.OnahoState.INSERTING
            elif velocity < -self.option.forward_backward_velocity_threashold:
                self.state = self.OnahoState.OUTGOING
            else:
                if abs(velocity) < self.option.no_motion_velocity_threashold:
                    if self.state == self.OnahoState.INSERTING:
                        if self.remaining_wait <= 0:
                            t = threading.Thread(target=self.play_ex)
                            t.start()
                            self.remaining_wait = self.option.sound_wait_time
                    print(self.state)
                    self.state = self.OnahoState.NO_MOTION

            self.velocities.append(velocity)
            self.timestamps.append(time.time())

        self.previous_center = center
        self.previous_time = time.time()
        
        self.remaining_wait -= 1