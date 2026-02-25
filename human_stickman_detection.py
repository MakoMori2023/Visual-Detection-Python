import cv2
from config import CONFIG
from utils import (
    load_mediapipe_model,
    convert_bgr_to_mp_image,
    flip_frame,
    draw_landmark_points,
    draw_landmark_connections
)

class HumanStickmanDetection:
    def __init__(self):
        self.enabled = False
        self.detector = None

        self.model_path = CONFIG["human_stickman_detection"]["model_path"]
        self.num_poses = CONFIG["human_stickman_detection"]["num_poses"]
        self.min_pose_detection_confidence = CONFIG["human_stickman_detection"]["min_pose_detection_confidence"]
        self.min_pose_presence_confidence = CONFIG["human_stickman_detection"]["min_pose_presence_confidence"]
        self.min_tracking_confidence = CONFIG["human_stickman_detection"]["min_tracking_confidence"]
        self.POSE_CONNECTIONS = CONFIG["human_stickman_detection"]["pose_connections"]
        self.draw_config = CONFIG["human_stickman_detection"]["draw"]

        self._init_detector()

    def _init_detector(self):
        self.detector = load_mediapipe_model(
            model_path=self.model_path,
            task_type="pose_landmarker",
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def enable(self):
        if self.detector is None:
            print("Human Stick Figure Detection cannot be enabled (model missing/failed to load)")
            return
        self.enabled = True
        print("Human Stick Figure Detection enabled")

    def disable(self):
        self.enabled = False
        print("Human Stick Figure Detection disabled")

    def detect(self, frame):
        if not self.enabled or frame is None or self.detector is None:
            return None

        mp_image = convert_bgr_to_mp_image(frame)
        if mp_image is None:
            return None
        return self.detector.detect(mp_image)

    def draw(self, frame):
        if not self.enabled or frame is None or self.detector is None:
            return frame
        
        # frame = flip_frame(frame)
        detection_result = self.detect(frame)
        
        if not detection_result or not detection_result.pose_landmarks:
            return frame
        
        frame_size = frame.shape[:2]  # (height, width)
        
        for pose_landmarks in detection_result.pose_landmarks:

            frame = draw_landmark_points(
                frame=frame,
                landmarks=pose_landmarks,
                frame_size=frame_size,
                circle_radius=self.draw_config["circle_radius"],
                circle_color=tuple(self.draw_config["circle_color"])
            )

            frame = draw_landmark_connections(
                frame=frame,
                landmarks=pose_landmarks,
                connections=self.POSE_CONNECTIONS,
                frame_size=frame_size,
                line_thickness=self.draw_config["line_thickness"],
                line_color=tuple(self.draw_config["line_color"])
            )
        
        return frame